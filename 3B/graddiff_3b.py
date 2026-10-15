#!/usr/bin/env python3
import argparse, json, math, random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


def parse_args():
    p = argparse.ArgumentParser(description="GradDiff baseline for xLAM/BFCL (3B)")
    p.add_argument("--model-name", default="Salesforce/xLAM-2-3b-fc-r")
    p.add_argument("--root", default="~/Desktop/BFCL")
    p.add_argument("--forget-file", default=None)
    p.add_argument("--retain-file", default=None)
    p.add_argument("--output-dir", required=True)

    # Same hyperparameters as the 1B BFCL GradDiff experiment.
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--forget-coef", type=float, default=1.0)
    p.add_argument("--retain-coef", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--retain-per-forget", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.root).expanduser()
FORGET_FILE = (
    Path(ARGS.forget_file).expanduser()
    if ARGS.forget_file
    else ROOT / "final_forget_train.json"
)
RETAIN_FILE = (
    Path(ARGS.retain_file).expanduser()
    if ARGS.retain_file
    else ROOT / "final_retain_train_4040.json"
)
OUTPUT_DIR = Path(ARGS.output_dir).expanduser()

MICRO_BATCH = 1


# ============================================================
# SEED
# ============================================================

random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)
torch.backends.cuda.matmul.allow_tf32 = True


if ARGS.epochs < 1:
    raise ValueError("--epochs must be >= 1")

if ARGS.forget_coef < 0 or ARGS.retain_coef < 0:
    raise ValueError("coefficients must be non-negative")

if ARGS.grad_accum < 1:
    raise ValueError("--grad-accum must be >= 1")

if ARGS.retain_per_forget < 1:
    raise ValueError("--retain-per-forget must be >= 1")


# ============================================================
# DATA
# ============================================================

with open(FORGET_FILE) as f:
    forget_rows = json.load(f)

with open(RETAIN_FILE) as f:
    retain_rows = json.load(f)

assert len(forget_rows) == 2020, len(forget_rows)
assert len(retain_rows) == 4040, len(retain_rows)


# ============================================================
# TOKENIZER / XLAM PROMPT
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    ARGS.model_name,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token




def maybe_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def get_query(row):
    q = maybe_json(row.get("query", ""))
    return q if isinstance(q, str) else json.dumps(q, ensure_ascii=False)


def get_tools(row):
    tools = maybe_json(row.get("tools", []))
    return [tools] if isinstance(tools, dict) else tools


def get_answers(row):
    ans = maybe_json(row.get("answers", []))
    return [ans] if isinstance(ans, dict) else ans


def target_names(row):
    return [
        x["name"]
        for x in get_answers(row)
        if isinstance(x, dict) and x.get("name")
    ]


NATIVE_FORMAT_VERSION = "xlam2_native_tools_json_list_v1"


def build_target(row):
    """
    xLAM-2 native function-call target:
      [{"name": "...", "arguments": {...}}, ...]
    """
    return json.dumps(
        get_answers(row),
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ============================================================
# DATASET
# ============================================================

class ToolDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # xLAM-2 native interface:
        #   - query is the user message
        #   - tool schemas are passed through tools=...
        messages = [
            {"role": "user", "content": get_query(row)}
        ]

        chat = tokenizer.apply_chat_template(
            messages,
            tools=get_tools(row),
            add_generation_prompt=True,
            tokenize=False,
        )

        # The rendered template already contains its special-token structure.
        prompt_ids = tokenizer(
            chat,
            add_special_tokens=False,
        )["input_ids"]

        target_ids = tokenizer(
            build_target(row),
            add_special_tokens=False,
        )["input_ids"] + [tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > ARGS.max_length:
            raise RuntimeError(
                f"Example {idx} length {len(input_ids)} "
                f"> MAX_LENGTH={ARGS.max_length}"
            )

        return {
            "idx": idx,
            "input_ids": input_ids,
            "labels": labels,
        }


def collate(batch):
    m = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    masks = []
    idxs = []

    for x in batch:
        n = len(x["input_ids"])
        pad = m - n

        input_ids.append(
            x["input_ids"] + [tokenizer.pad_token_id] * pad
        )
        labels.append(
            x["labels"] + [-100] * pad
        )
        masks.append(
            [1] * n + [0] * pad
        )
        idxs.append(
            x["idx"]
        )

    return {
        "idx": torch.tensor(idxs, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ============================================================
# MEMORY-EFFICIENT TARGET-SUFFIX FORWARD
#
# Same objective as 1B GradDiff; this only avoids materializing
# vocabulary logits for prompt positions when supported.
# ============================================================

def target_suffix_forward(model, input_ids, attention_mask, labels):
    supervised = torch.nonzero(
        labels[0] != -100,
        as_tuple=False,
    ).flatten()

    if supervised.numel() == 0:
        raise RuntimeError("Batch has no supervised target tokens")

    first_target = int(supervised[0].item())
    seq_len = labels.shape[1]

    # Need one position before first target token for causal shift.
    keep = seq_len - first_target + 1

    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            num_logits_to_keep=keep,
        )
        score_labels = labels[:, -keep:]
    except TypeError:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        score_labels = labels

    return out, score_labels


def token_ce_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


forget_dataset = ToolDataset(forget_rows)
retain_dataset = ToolDataset(retain_rows)


# ============================================================
# API-BALANCED FORGET SAMPLER
# ============================================================

api_counts = Counter()

for row in forget_rows:
    for name in set(target_names(row)):
        api_counts[name] += 1


weights = []

for row in forget_rows:
    targets = set(target_names(row))

    if not targets:
        raise RuntimeError("Forget row has no target API")

    weights.append(
        float(
            np.mean([
                1.0 / api_counts[name]
                for name in targets
            ])
        )
    )


forget_sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(forget_dataset),
    replacement=True,
    generator=torch.Generator().manual_seed(ARGS.seed),
)


forget_loader = DataLoader(
    forget_dataset,
    batch_size=MICRO_BATCH,
    sampler=forget_sampler,
    collate_fn=collate,
)


retain_loader = DataLoader(
    retain_dataset,
    batch_size=1,
    shuffle=True,
    generator=torch.Generator().manual_seed(ARGS.seed + 1),
    collate_fn=collate,
)


assert (
    len(retain_loader)
    ==
    ARGS.retain_per_forget * len(forget_loader)
)


# ============================================================
# CONFIG
# ============================================================

print("=" * 90)
print("GRADDIFF 3B CONFIG")
print("=" * 90)

print("Model:", ARGS.model_name)
print("Forget / retain:", len(forget_rows), "/", len(retain_rows))
print("Forget coef:", ARGS.forget_coef)
print("Retain coef:", ARGS.retain_coef)
print("Epochs:", ARGS.epochs)
print("LR:", ARGS.learning_rate)
print("Weight decay:", ARGS.weight_decay)
print("Grad accum:", ARGS.grad_accum)
print("Retain per forget:", ARGS.retain_per_forget)
print("Max length:", ARGS.max_length)
print("Warmup ratio:", ARGS.warmup_ratio)
print("Output:", OUTPUT_DIR)


# ============================================================
# MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    ARGS.model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(ARGS.device)

model.config.use_cache = False

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

model.train()


# ============================================================
# OPTIMIZER / SCHEDULER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=ARGS.learning_rate,
    weight_decay=ARGS.weight_decay,
)


steps_per_epoch = math.ceil(
    len(forget_loader) / ARGS.grad_accum
)

total_steps = (
    ARGS.epochs * steps_per_epoch
)

warmup_steps = (
    max(
        1,
        int(
            ARGS.warmup_ratio
            * total_steps
        ),
    )
    if total_steps > 1
    else 0
)


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


print("Steps/epoch:", steps_per_epoch)
print("Total optimizer steps:", total_steps)
print("Warmup steps:", warmup_steps)


# ============================================================
# TRAIN
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

history = []

optimizer.zero_grad(
    set_to_none=True
)


for epoch in range(
    ARGS.epochs
):

    retain_iter = iter(
        retain_loader
    )

    f_hist = []
    r_hist = []
    obj_hist = []
    gn_hist = []

    print("\n" + "=" * 90)
    print(
        f"GRADDIFF EPOCH "
        f"{epoch+1}/{ARGS.epochs}"
    )
    print("=" * 90)

    for step, fbatch in enumerate(
        forget_loader
    ):

        # ====================================================
        # FORGET: maximize CE by minimizing -CE
        # ====================================================

        f_input = (
            fbatch["input_ids"]
            .to(ARGS.device)
        )

        f_mask = (
            fbatch["attention_mask"]
            .to(ARGS.device)
        )

        f_labels = (
            fbatch["labels"]
            .to(ARGS.device)
        )


        fout, f_score_labels = target_suffix_forward(
            model,
            f_input,
            f_mask,
            f_labels,
        )


        forget_ce = token_ce_loss(
            fout.logits,
            f_score_labels,
        )


        (
            -ARGS.forget_coef
            * forget_ce
            / ARGS.grad_accum
        ).backward()


        f_hist.append(
            float(
                forget_ce.detach()
            )
        )

        del fout


        # ====================================================
        # RETAIN: minimize CE; same 2:1 exposure as 1B
        # ====================================================

        step_retain = []


        for _ in range(
            ARGS.retain_per_forget
        ):

            rbatch = next(
                retain_iter
            )

            r_input = (
                rbatch["input_ids"]
                .to(ARGS.device)
            )

            r_mask = (
                rbatch["attention_mask"]
                .to(ARGS.device)
            )

            r_labels = (
                rbatch["labels"]
                .to(ARGS.device)
            )


            rout, r_score_labels = target_suffix_forward(
                model,
                r_input,
                r_mask,
                r_labels,
            )


            retain_ce = token_ce_loss(
                rout.logits,
                r_score_labels,
            )


            (
                ARGS.retain_coef
                * retain_ce
                / ARGS.retain_per_forget
                / ARGS.grad_accum
            ).backward()


            step_retain.append(
                float(
                    retain_ce.detach()
                )
            )

            del rout


        mean_r = float(
            np.mean(
                step_retain
            )
        )

        r_hist.append(
            mean_r
        )

        obj_hist.append(
            -ARGS.forget_coef
            * f_hist[-1]
            + ARGS.retain_coef
            * mean_r
        )


        # ====================================================
        # OPTIMIZER
        # ====================================================

        boundary = (
            (step + 1)
            % ARGS.grad_accum
            == 0
        )

        final_step = (
            step + 1
            == len(forget_loader)
        )


        if boundary or final_step:

            gn = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    ARGS.max_grad_norm,
                )
            )

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            gn_hist.append(
                float(
                    gn.detach().cpu()
                )
                if torch.is_tensor(gn)
                else float(gn)
            )


        if (
            step % 50 == 0
            or final_step
        ):

            print(
                f"epoch={epoch+1} "
                f"step={step+1:4d}/"
                f"{len(forget_loader)} "
                f"forgetCE="
                f"{np.mean(f_hist[-50:]):.4f} "
                f"retainCE="
                f"{np.mean(r_hist[-50:]):.4f} "
                f"objective="
                f"{np.mean(obj_hist[-50:]):+.4f} "
                f"lr="
                f"{scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )


    rec = {
        "epoch": epoch + 1,
        "forget_ce": float(
            np.mean(f_hist)
        ),
        "retain_ce": float(
            np.mean(r_hist)
        ),
        "objective": float(
            np.mean(obj_hist)
        ),
        "mean_grad_norm": (
            float(np.mean(gn_hist))
            if gn_hist
            else None
        ),
    }

    history.append(
        rec
    )

    print(
        json.dumps(
            rec,
            indent=2,
        )
    )

    # Save EVERY epoch so each checkpoint can be evaluated separately.
    epoch_dir = OUTPUT_DIR / f"epoch_{epoch + 1}"
    epoch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.config.use_cache = True

    model.save_pretrained(
        epoch_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        epoch_dir
    )

    model.config.use_cache = False

    config = {
        "base_model": ARGS.model_name,
        "method": "GradDiff",
        "objective": (
            "-forget_coef * CE_forget "
            "+ retain_coef * CE_retain"
        ),
        "format_version": NATIVE_FORMAT_VERSION,
        "prompt_format": (
            "xLAM-2 native chat template with tools=..."
        ),
        "target_format": (
            "native JSON list of {name, arguments} calls"
        ),
        "forget_coef": ARGS.forget_coef,
        "retain_coef": ARGS.retain_coef,
        "learning_rate": ARGS.learning_rate,
        "weight_decay": ARGS.weight_decay,
        "epochs_requested": ARGS.epochs,
        "checkpoint_epoch": epoch + 1,
        "max_length": ARGS.max_length,
        "grad_accum": ARGS.grad_accum,
        "retain_per_forget": ARGS.retain_per_forget,
        "max_grad_norm": ARGS.max_grad_norm,
        "warmup_ratio": ARGS.warmup_ratio,
        "forget_n": len(forget_rows),
        "retain_n": len(retain_rows),
        "seed": ARGS.seed,
        "cli_args": vars(ARGS),
        "history": history,
    }

    with open(
        epoch_dir / "unlearning_config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print("\nSaved model:", epoch_dir)


# ============================================================
# FINAL STATUS
# ============================================================

print("\nTRAINING COMPLETE")
print("Final checkpoint:", OUTPUT_DIR / f"epoch_{ARGS.epochs}")
