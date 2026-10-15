import os
import gc
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Salesforce/xLAM-1b-fc-r"

ROOT = Path("~/Desktop/BFCL").expanduser()

FORGET_FILE = ROOT / "final_forget_train.json"
RETAIN_FILE = ROOT / "final_retain_train_4040.json"
REF_CACHE = ROOT / "xlam_npo_reference_logps.pt"


OUTPUT_DIR = Path(
    "/projects/standard/mhong/shared/hadir/out_dir/bfcl/1B/npo/epoch_1"
)


SEED = 42
MAX_LENGTH = 3072
NUM_EPOCHS = 1
BETA = 0.1
RETAIN_COEF = 1.0
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MICRO_BATCH = 1

# 32 forget examples / optimizer update.
GRAD_ACCUM = 32

# Two retain trajectories per forget trajectory.
# Over one epoch:
#   2020 forget draws
#   4040 retain examples
RETAIN_PER_FORGET = 2

MAX_GRAD_NORM = 1.0

# Reference cache forward pass
REF_BATCH_SIZE = 2

DEVICE = "cuda"

torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD DATA
# ============================================================

with open(FORGET_FILE) as f:
    forget_rows = json.load(f)

with open(RETAIN_FILE) as f:
    retain_rows = json.load(f)


print("=" * 90)
print("DATA")
print("=" * 90)

print("Forget:", len(forget_rows))
print("Retain:", len(retain_rows))

assert len(forget_rows) == 2020
assert len(retain_rows) == 4040


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# XLAM PROMPT
# ============================================================

TASK_INSTRUCTION = """
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out and refuse to answer.
If the given question lacks the parameters required by the function, also point it out.
""".strip()


FORMAT_INSTRUCTION = """
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct.
If no function call is needed, please make tool_calls an empty list '[]'.

{
  "tool_calls": [
    {
      "name": "func_name1",
      "arguments": {
        "argument1": "value1",
        "argument2": "value2"
      }
    }
  ]
}
""".strip()


def maybe_json(x):

    if isinstance(x, str):

        try:
            return json.loads(x)

        except Exception:
            return x

    return x


def get_query(row):

    q = maybe_json(
        row.get("query", "")
    )

    if isinstance(q, str):
        return q

    return json.dumps(
        q,
        ensure_ascii=False,
    )


def get_tools(row):

    tools = maybe_json(
        row.get("tools", [])
    )

    if isinstance(tools, dict):
        tools = [tools]

    return tools


def get_answers(row):

    answers = maybe_json(
        row.get("answers", [])
    )

    if isinstance(answers, dict):
        answers = [answers]

    return answers


def target_names(row):

    return [
        x["name"]
        for x in get_answers(row)
        if (
            isinstance(x, dict)
            and x.get("name")
        )
    ]


def build_user_prompt(row):

    return (
        "[BEGIN OF TASK INSTRUCTION]\n"
        + TASK_INSTRUCTION
        + "\n[END OF TASK INSTRUCTION]\n\n"

        + "[BEGIN OF AVAILABLE TOOLS]\n"
        + json.dumps(
            get_tools(row),
            ensure_ascii=False,
        )
        + "\n[END OF AVAILABLE TOOLS]\n\n"

        + "[BEGIN OF FORMAT INSTRUCTION]\n"
        + FORMAT_INSTRUCTION
        + "\n[END OF FORMAT INSTRUCTION]\n\n"

        + "[BEGIN OF QUERY]\n"
        + get_query(row)
        + "\n[END OF QUERY]"
    )


def build_target(row):

    return json.dumps(
        {
            "tool_calls":
                get_answers(row)
        },
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

        user_prompt = build_user_prompt(
            row
        )

        chat_text = (
            tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": user_prompt,
                }],
                add_generation_prompt=True,
                tokenize=False,
            )
        )

        prompt_ids = tokenizer(
            chat_text,
            add_special_tokens=True,
        )["input_ids"]

        target_ids = tokenizer(
            build_target(row),
            add_special_tokens=False,
        )["input_ids"]

        target_ids = (
            target_ids
            + [tokenizer.eos_token_id]
        )

        input_ids = (
            prompt_ids
            + target_ids
        )

        labels = (
            [-100] * len(prompt_ids)
            + target_ids
        )

        if len(input_ids) > MAX_LENGTH:

            raise RuntimeError(
                f"Example {idx} length "
                f"{len(input_ids)} > "
                f"MAX_LENGTH={MAX_LENGTH}"
            )

        return {
            "idx": idx,
            "input_ids": input_ids,
            "labels": labels,
        }


forget_dataset = ToolDataset(
    forget_rows
)

retain_dataset = ToolDataset(
    retain_rows
)


# ============================================================
# COLLATOR
# ============================================================

def collate(batch):

    max_len = max(
        len(x["input_ids"])
        for x in batch
    )

    input_ids = []
    labels = []
    attention_masks = []
    indices = []

    for x in batch:

        n = len(x["input_ids"])
        pad = max_len - n

        input_ids.append(
            x["input_ids"]
            + [tokenizer.pad_token_id] * pad
        )

        labels.append(
            x["labels"]
            + [-100] * pad
        )

        attention_masks.append(
            [1] * n
            + [0] * pad
        )

        indices.append(
            x["idx"]
        )

    return {
        "idx": torch.tensor(
            indices,
            dtype=torch.long,
        ),

        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),

        "attention_mask": torch.tensor(
            attention_masks,
            dtype=torch.long,
        ),

        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
    }


# ============================================================
# SEQUENCE LOG PROBABILITY
#
# log pi(y | x)
#
# ONLY target/tool-call tokens count.
# Prompt tokens are masked with -100.
# ============================================================

def sequence_logp(
    logits,
    labels,
):

    # Causal shift
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = (
        shift_labels != -100
    )

    safe_labels = (
        shift_labels.clone()
    )

    safe_labels[
        ~mask
    ] = 0

    token_logps = (
        F.log_softmax(
            shift_logits.float(),
            dim=-1,
        )
        .gather(
            -1,
            safe_labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    token_logps = (
        token_logps
        * mask
    )

    # Original NPO uses full sequence probability:
    #
    # log pi(y|x) =
    # sum token log probabilities
    #
    return token_logps.sum(
        dim=-1
    )


# ============================================================
# RETAIN CE
# ============================================================

def retain_ce_loss(
    logits,
    labels,
):

    shift_logits = (
        logits[:, :-1, :]
        .contiguous()
    )

    shift_labels = (
        labels[:, 1:]
        .contiguous()
    )

    return F.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.size(-1),
        ),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ============================================================
# PRECOMPUTE REFERENCE LOG P(y|x)
#
# This lets us unload the frozen reference model before
# optimization, saving substantial GPU memory.
# ============================================================

if REF_CACHE.exists():

    print("\nLoading cached reference log-probs:")
    print(REF_CACHE)

    reference_logps = torch.load(
        REF_CACHE,
        map_location="cpu",
    )

    assert (
        len(reference_logps)
        == len(forget_dataset)
    )

else:

    print("\n" + "=" * 90)
    print("PRECOMPUTING REFERENCE LOG-PROBABILITIES")
    print("=" * 90)

    reference_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(DEVICE)
    )

    reference_model.eval()

    for p in reference_model.parameters():
        p.requires_grad = False

    reference_loader = DataLoader(
        forget_dataset,
        batch_size=REF_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    reference_logps = torch.empty(
        len(forget_dataset),
        dtype=torch.float32,
    )

    with torch.inference_mode():

        for step, batch in enumerate(
            reference_loader
        ):

            input_ids = (
                batch["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(DEVICE)
            )

            labels = (
                batch["labels"]
                .to(DEVICE)
            )

            outputs = reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            logps = sequence_logp(
                outputs.logits,
                labels,
            )

            reference_logps[
                batch["idx"]
            ] = (
                logps
                .detach()
                .cpu()
            )

            if (
                step % 100 == 0
                or step + 1
                == len(reference_loader)
            ):
                print(
                    f"Reference "
                    f"{step+1}/"
                    f"{len(reference_loader)}"
                )

    torch.save(
        reference_logps,
        REF_CACHE,
    )

    print(
        "\nSaved reference cache:",
        REF_CACHE,
    )

    del reference_model
    del reference_loader

    gc.collect()
    torch.cuda.empty_cache()


print(
    "\nReference logP mean:",
    float(reference_logps.mean())
)

print(
    "Reference logP min/max:",
    float(reference_logps.min()),
    float(reference_logps.max()),
)


# ============================================================
# API-BALANCED FORGET SAMPLER
# ============================================================

api_counts = Counter()

for row in forget_rows:

    for name in set(
        target_names(row)
    ):
        api_counts[name] += 1


weights = []

for row in forget_rows:

    targets = set(
        target_names(row)
    )

    w = np.mean([
        1.0 / api_counts[name]
        for name in targets
    ])

    weights.append(
        float(w)
    )


forget_sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(forget_dataset),
    replacement=True,
    generator=torch.Generator().manual_seed(
        SEED
    ),
)


forget_loader = DataLoader(
    forget_dataset,
    batch_size=MICRO_BATCH,
    sampler=forget_sampler,
    collate_fn=collate,
)


# Retain dataset has exactly 2x the number of forget draws.
retain_loader = DataLoader(
    retain_dataset,
    batch_size=1,
    shuffle=True,
    generator=torch.Generator().manual_seed(
        SEED + 1
    ),
    collate_fn=collate,
)


assert (
    len(retain_loader)
    ==
    RETAIN_PER_FORGET
    * len(forget_loader)
)


# ============================================================
# LOAD TRAINABLE MODEL
# ============================================================

print("\n" + "=" * 90)
print("LOADING TRAINABLE XLAM")
print("=" * 90)

model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    .to(DEVICE)
)

model.config.use_cache = False

model.gradient_checkpointing_enable()

model.train()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


optimizer_steps_per_epoch = math.ceil(
    len(forget_loader)
    / GRAD_ACCUM
)

total_optimizer_steps = (
    NUM_EPOCHS
    * optimizer_steps_per_epoch
)

# Literature-style first-epoch warmup.
# warmup_steps = min(
#     optimizer_steps_per_epoch,
#     total_optimizer_steps - 1
# ) if total_optimizer_steps > 1 else 0

warmup_steps = (
    max(
        1,
        int(0.10 * total_optimizer_steps),
    )
    if total_optimizer_steps > 1
    else 0
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optimizer_steps,
)


print(
    "Forget steps/epoch:",
    len(forget_loader)
)

print(
    "Optimizer steps/epoch:",
    optimizer_steps_per_epoch
)

print(
    "Total optimizer steps:",
    total_optimizer_steps
)

print(
    "Gradient accumulation:",
    GRAD_ACCUM
)

print(
    "Beta:",
    BETA
)

print(
    "Retain coefficient:",
    RETAIN_COEF
)

print(
    "LR:",
    LEARNING_RATE
)


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
    NUM_EPOCHS
):

    print("\n" + "=" * 90)
    print(
        f"NPO EPOCH {epoch+1}/{NUM_EPOCHS}"
    )
    print("=" * 90)

    retain_iter = iter(
        retain_loader
    )

    epoch_npo = []
    epoch_retain = []
    epoch_ratio = []

    optimizer_step = 0


    for step, fbatch in enumerate(
        forget_loader
    ):

        # ====================================================
        # FORGET / NPO
        # ====================================================

        f_idx = fbatch["idx"]

        f_input = (
            fbatch["input_ids"]
            .to(DEVICE)
        )

        f_mask = (
            fbatch["attention_mask"]
            .to(DEVICE)
        )

        f_labels = (
            fbatch["labels"]
            .to(DEVICE)
        )


        outputs = model(
            input_ids=f_input,
            attention_mask=f_mask,
            use_cache=False,
        )


        current_logp = sequence_logp(
            outputs.logits,
            f_labels,
        )


        ref_logp = (
            reference_logps[
                f_idx
            ]
            .to(
                DEVICE,
                dtype=torch.float32,
            )
        )


        log_ratio = (
            current_logp
            - ref_logp
        )


        # Eq. 3 of NPO:
        #
        # -(2/beta) log sigmoid(
        #     -beta * log(pi_theta/pi_ref)
        # )
        #
        npo_loss = (
            -(2.0 / BETA)
            * F.logsigmoid(
                -BETA
                * log_ratio
            )
        ).mean()


        (
            npo_loss
            / GRAD_ACCUM
        ).backward()


        epoch_npo.append(
            float(
                npo_loss.detach()
            )
        )

        epoch_ratio.append(
            float(
                log_ratio.mean().detach()
            )
        )


        del outputs


        # ====================================================
        # RETAIN LOSS
        #
        # Two retain examples per forget example.
        # Backprop separately to avoid keeping both
        # computation graphs in memory.
        # ====================================================

        step_retain = []


        for _ in range(
            RETAIN_PER_FORGET
        ):

            rbatch = next(
                retain_iter
            )

            r_input = (
                rbatch["input_ids"]
                .to(DEVICE)
            )

            r_mask = (
                rbatch["attention_mask"]
                .to(DEVICE)
            )

            r_labels = (
                rbatch["labels"]
                .to(DEVICE)
            )


            rout = model(
                input_ids=r_input,
                attention_mask=r_mask,
                use_cache=False,
            )


            rloss = retain_ce_loss(
                rout.logits,
                r_labels,
            )


            # Average the two retain samples.
            scaled_retain = (
                RETAIN_COEF
                * rloss
                / RETAIN_PER_FORGET
                / GRAD_ACCUM
            )


            scaled_retain.backward()


            step_retain.append(
                float(
                    rloss.detach()
                )
            )


            del rout


        epoch_retain.append(
            float(
                np.mean(
                    step_retain
                )
            )
        )


        # ====================================================
        # OPTIMIZER
        # ====================================================

        boundary = (
            (step + 1) % GRAD_ACCUM
            == 0
        )

        final_step = (
            step + 1
            == len(forget_loader)
        )


        if boundary or final_step:

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )
            )


            optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_step += 1


        # ====================================================
        # LOG
        # ====================================================

        if (
            step % 50 == 0
            or final_step
        ):

            print(
                f"epoch={epoch+1} "
                f"step={step+1:4d}/"
                f"{len(forget_loader)} "
                f"NPO={np.mean(epoch_npo[-50:]):.4f} "
                f"retainCE="
                f"{np.mean(epoch_retain[-50:]):.4f} "
                f"logratio="
                f"{np.mean(epoch_ratio[-50:]):+.4f} "
                f"lr="
                f"{scheduler.get_last_lr()[0]:.2e}"
            )


    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    record = {
        "epoch": epoch + 1,
        "npo_loss": float(
            np.mean(epoch_npo)
        ),
        "retain_ce": float(
            np.mean(epoch_retain)
        ),
        "forget_log_ratio": float(
            np.mean(epoch_ratio)
        ),
    }

    history.append(
        record
    )


    print("\nEpoch summary:")
    print(
        json.dumps(
            record,
            indent=2,
        )
    )


# ============================================================
# SAVE MODEL
# ============================================================

FINAL_DIR = (
    OUTPUT_DIR
    / f"epoch_{NUM_EPOCHS}"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("\n" + "=" * 90)
print("SAVING")
print("=" * 90)


model.save_pretrained(
    FINAL_DIR,
    safe_serialization=True,
)

tokenizer.save_pretrained(
    FINAL_DIR
)


config = {
    "base_model": MODEL_NAME,
    "method": "NPO+retain",
    "beta": BETA,
    "retain_coef": RETAIN_COEF,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "epochs": NUM_EPOCHS,
    "max_length": MAX_LENGTH,
    "grad_accum": GRAD_ACCUM,
    "retain_per_forget": RETAIN_PER_FORGET,
    "forget_n": len(forget_rows),
    "retain_n": len(retain_rows),
    "seed": SEED,
    "history": history,
}


with open(
    FINAL_DIR / "unlearning_config.json",
    "w",
) as f:

    json.dump(
        config,
        f,
        indent=2,
    )


print("Saved model:")
print(FINAL_DIR)

print("\nHistory:")
print(
    json.dumps(
        history,
        indent=2,
    )
)

