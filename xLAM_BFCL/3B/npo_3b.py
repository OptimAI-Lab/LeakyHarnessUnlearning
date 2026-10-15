#!/usr/bin/env python3
import argparse, gc, json, math, random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


def parse_args():
    p = argparse.ArgumentParser(description="NPO baseline for xLAM/BFCL (3B-ready)")
    p.add_argument("--model-name", default="Salesforce/xLAM-2-3b-fc-r")
    p.add_argument("--root", default="~/Desktop/BFCL")
    p.add_argument("--forget-file", default=None)
    p.add_argument("--retain-file", default=None)
    p.add_argument("--reference-cache", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--retain-coef", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--retain-per-forget", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--ref-batch-size", type=int, default=1)
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


ARGS = parse_args()
ROOT = Path(ARGS.root).expanduser()
FORGET_FILE = Path(ARGS.forget_file).expanduser() if ARGS.forget_file else ROOT / "final_forget_train.json"
RETAIN_FILE = Path(ARGS.retain_file).expanduser() if ARGS.retain_file else ROOT / "final_retain_train_4040.json"
REF_CACHE = Path(ARGS.reference_cache).expanduser() if ARGS.reference_cache else ROOT / "xlam_3b_npo_reference_logps.pt"
OUTPUT_DIR = Path(ARGS.output_dir).expanduser()
MICRO_BATCH = 1

random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)
torch.backends.cuda.matmul.allow_tf32 = True

if ARGS.epochs < 1:
    raise ValueError("--epochs must be >= 1")
if ARGS.beta <= 0:
    raise ValueError("--beta must be > 0")
if ARGS.retain_coef < 0:
    raise ValueError("--retain-coef must be non-negative")

with open(FORGET_FILE) as f:
    forget_rows = json.load(f)
with open(RETAIN_FILE) as f:
    retain_rows = json.load(f)

assert len(forget_rows) == 2020, len(forget_rows)
assert len(retain_rows) == 4040, len(retain_rows)

tokenizer = AutoTokenizer.from_pretrained(ARGS.model_name, trust_remote_code=True)
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
    return [x["name"] for x in get_answers(row) if isinstance(x, dict) and x.get("name")]


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


class ToolDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # xLAM-2 native interface:
        #   - the user message contains only the query
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

        # The rendered chat template already contains its special-token
        # structure, so do not add special tokens again here.
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
                f"Example {idx} length {len(input_ids)} > MAX_LENGTH={ARGS.max_length}"
            )

        return {
            "idx": idx,
            "input_ids": input_ids,
            "labels": labels,
        }


def collate(batch):
    m = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, masks, idxs = [], [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = m - n
        input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        masks.append([1] * n + [0] * pad)
        idxs.append(x["idx"])
    return {
        "idx": torch.tensor(idxs, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def target_suffix_forward(model, input_ids, attention_mask, labels):
    supervised = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    if supervised.numel() == 0:
        raise RuntimeError("Batch has no supervised target tokens")
    first_target = int(supervised[0].item())
    seq_len = labels.shape[1]
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


def sequence_logp(logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0
    token_logps = (
        F.log_softmax(shift_logits.float(), dim=-1)
        .gather(-1, safe_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    return (token_logps * mask).sum(dim=-1)


def retain_ce_loss(logits, labels):
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
# REFERENCE LOG P(y|x)
# ============================================================

if REF_CACHE.exists():
    print("\nLoading cached reference log-probs:")
    print(REF_CACHE)
    payload = torch.load(REF_CACHE, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Reference cache is from an older format. "
            "Use a new --reference-cache path so it can be recomputed."
        )

    if payload.get("model_name") != ARGS.model_name:
        raise RuntimeError(
            f"Reference-cache model mismatch: "
            f"{payload.get('model_name')} != {ARGS.model_name}"
        )

    if payload.get("format_version") != NATIVE_FORMAT_VERSION:
        raise RuntimeError(
            "Reference cache does not match the xLAM-2 native prompt/target format. "
            "Use a new --reference-cache path, for example "
            "xlam_3b_npo_reference_logps_native.pt"
        )

    if payload.get("max_length") != ARGS.max_length:
        raise RuntimeError(
            f"Reference-cache max_length mismatch: "
            f"{payload.get('max_length')} != {ARGS.max_length}"
        )

    reference_logps = payload["reference_logps"]
    assert len(reference_logps) == len(forget_dataset)

else:
    print("\n" + "=" * 90)
    print("PRECOMPUTING REFERENCE LOG-PROBABILITIES")
    print("=" * 90)

    reference_model = AutoModelForCausalLM.from_pretrained(
        ARGS.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(ARGS.device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    reference_loader = DataLoader(
        forget_dataset,
        batch_size=ARGS.ref_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    reference_logps = torch.empty(len(forget_dataset), dtype=torch.float32)

    with torch.inference_mode():
        for step, batch in enumerate(reference_loader):
            input_ids = batch["input_ids"].to(ARGS.device)
            attention_mask = batch["attention_mask"].to(ARGS.device)
            labels = batch["labels"].to(ARGS.device)

            if input_ids.shape[0] == 1:
                outputs, score_labels = target_suffix_forward(
                    reference_model, input_ids, attention_mask, labels
                )
            else:
                outputs = reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                score_labels = labels

            logps = sequence_logp(outputs.logits, score_labels)
            reference_logps[batch["idx"]] = logps.detach().cpu()

            if step % 100 == 0 or step + 1 == len(reference_loader):
                print(f"Reference {step+1}/{len(reference_loader)}", flush=True)

    REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": ARGS.model_name,
            "max_length": ARGS.max_length,
            "format_version": NATIVE_FORMAT_VERSION,
            "prompt_format": "tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)",
            "target_format": "native JSON list of {name, arguments} calls",
            "logp_definition": "sum target-token log probabilities",
            "reference_logps": reference_logps,
        },
        REF_CACHE,
    )
    print("\nSaved reference cache:", REF_CACHE)

    del reference_model
    del reference_loader
    gc.collect()
    torch.cuda.empty_cache()

print("\nReference logP mean:", float(reference_logps.mean()))
print("Reference logP min/max:", float(reference_logps.min()), float(reference_logps.max()))

if ARGS.cache_only:
    print("\nREFERENCE CACHE COMPLETE")
    print("Examples:", len(reference_logps))
    print("Saved:", REF_CACHE)
    raise SystemExit(0)

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
    weights.append(float(np.mean([1.0 / api_counts[name] for name in targets])))

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

assert len(retain_loader) == ARGS.retain_per_forget * len(forget_loader)

print("=" * 90)
print("NPO CONFIG")
print("=" * 90)
print("Model:", ARGS.model_name)
print("Forget / retain:", len(forget_rows), "/", len(retain_rows))
print("Beta:", ARGS.beta)
print("Retain coef:", ARGS.retain_coef)
print("Epochs:", ARGS.epochs)
print("LR:", ARGS.learning_rate)
print("Weight decay:", ARGS.weight_decay)
print("Grad accum:", ARGS.grad_accum)
print("Retain per forget:", ARGS.retain_per_forget)
print("Max length:", ARGS.max_length)
print("Warmup ratio:", ARGS.warmup_ratio)
print("Reference cache:", REF_CACHE)
print("Output:", OUTPUT_DIR)

model = AutoModelForCausalLM.from_pretrained(
    ARGS.model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(ARGS.device)

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=ARGS.learning_rate,
    weight_decay=ARGS.weight_decay,
)

steps_per_epoch = math.ceil(len(forget_loader) / ARGS.grad_accum)
total_steps = ARGS.epochs * steps_per_epoch
warmup_steps = max(1, int(ARGS.warmup_ratio * total_steps)) if total_steps > 1 else 0
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
history = []
optimizer.zero_grad(set_to_none=True)

for epoch in range(ARGS.epochs):
    retain_iter = iter(retain_loader)
    npo_hist, retain_hist, ratio_hist, gn_hist = [], [], [], []

    print("\n" + "=" * 90)
    print(f"NPO EPOCH {epoch+1}/{ARGS.epochs}")
    print("=" * 90)

    for step, fbatch in enumerate(forget_loader):
        # Forget: NPO.
        f_idx = fbatch["idx"]
        f_input = fbatch["input_ids"].to(ARGS.device)
        f_mask = fbatch["attention_mask"].to(ARGS.device)
        f_labels = fbatch["labels"].to(ARGS.device)

        fout, f_score_labels = target_suffix_forward(model, f_input, f_mask, f_labels)
        current_logp = sequence_logp(fout.logits, f_score_labels)
        ref_logp = reference_logps[f_idx].to(ARGS.device, dtype=torch.float32)
        log_ratio = current_logp - ref_logp

        npo_loss = (
            -(2.0 / ARGS.beta)
            * F.logsigmoid(-ARGS.beta * log_ratio)
        ).mean()

        # Exact same accumulation normalization as the 1B implementation.
        (npo_loss / ARGS.grad_accum).backward()
        npo_hist.append(float(npo_loss.detach()))
        ratio_hist.append(float(log_ratio.mean().detach()))
        del fout

        # Retain: CE. Same 2:1 exposure as 1B / GradDiff.
        step_retain = []
        for _ in range(ARGS.retain_per_forget):
            rbatch = next(retain_iter)
            r_input = rbatch["input_ids"].to(ARGS.device)
            r_mask = rbatch["attention_mask"].to(ARGS.device)
            r_labels = rbatch["labels"].to(ARGS.device)

            rout, r_score_labels = target_suffix_forward(model, r_input, r_mask, r_labels)
            retain_ce = retain_ce_loss(rout.logits, r_score_labels)

            (
                ARGS.retain_coef
                * retain_ce
                / ARGS.retain_per_forget
                / ARGS.grad_accum
            ).backward()

            step_retain.append(float(retain_ce.detach()))
            del rout

        retain_hist.append(float(np.mean(step_retain)))

        boundary = (step + 1) % ARGS.grad_accum == 0
        final_step = step + 1 == len(forget_loader)
        if boundary or final_step:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), ARGS.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            gn_hist.append(float(gn.detach().cpu()) if torch.is_tensor(gn) else float(gn))

        if step % 50 == 0 or final_step:
            print(
                f"epoch={epoch+1} step={step+1:4d}/{len(forget_loader)} "
                f"NPO={np.mean(npo_hist[-50:]):.4f} "
                f"retainCE={np.mean(retain_hist[-50:]):.4f} "
                f"logratio={np.mean(ratio_hist[-50:]):+.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    rec = {
        "epoch": epoch + 1,
        "npo_loss": float(np.mean(npo_hist)),
        "retain_ce": float(np.mean(retain_hist)),
        "forget_log_ratio": float(np.mean(ratio_hist)),
        "mean_grad_norm": float(np.mean(gn_hist)) if gn_hist else None,
    }
    history.append(rec)
    print(json.dumps(rec, indent=2))

    # Save every epoch so epoch_1, epoch_2, epoch_3, ... can be
    # evaluated independently for the best forget-retain tradeoff.
    epoch_dir = OUTPUT_DIR / f"epoch_{epoch + 1}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    model.config.use_cache = True
    model.save_pretrained(epoch_dir, safe_serialization=True)
    tokenizer.save_pretrained(epoch_dir)
    model.config.use_cache = False

    config = {
        "base_model": ARGS.model_name,
        "method": "NPO+retain",
        "objective": "NPO_forget + retain_coef * CE_retain",
        "format_version": NATIVE_FORMAT_VERSION,
        "prompt_format": "xLAM-2 native chat template with tools=...",
        "target_format": "native JSON list of {name, arguments} calls",
        "beta": ARGS.beta,
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
        "reference_cache": str(REF_CACHE),
        "reference_logp_definition": "sum target-token log probabilities",
        "seed": ARGS.seed,
        "cli_args": vars(ARGS),
        "history": history,
    }

    with open(epoch_dir / "unlearning_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nSaved model:", epoch_dir)

print("\nTRAINING COMPLETE")
print("Final checkpoint:", OUTPUT_DIR / f"epoch_{ARGS.epochs}")
