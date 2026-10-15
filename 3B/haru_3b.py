#!/usr/bin/env python3
import argparse, gc, json, math, random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

FORMAT_VERSION = "xlam2_native_tools_json_list_v1"
MICRO_BATCH = 1


def parse_args():
    p = argparse.ArgumentParser(
        description="HARU (NPO + safe target + retain) for xLAM-2 3B/BFCL"
    )
    p.add_argument("--model-name", default="Salesforce/xLAM-2-3b-fc-r")
    p.add_argument("--root", default="~/Desktop/BFCL")
    p.add_argument("--forget-file", default=None)
    p.add_argument("--retain-file", default=None)
    p.add_argument("--ref-cache", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--safe-coef", type=float, default=0.5)
    p.add_argument("--retain-coef", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--retain-per-forget", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--ref-batch-size", type=int, default=1)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


ARGS = parse_args()
ROOT = Path(ARGS.root).expanduser()
FORGET_FILE = Path(ARGS.forget_file).expanduser() if ARGS.forget_file else ROOT / "final_forget_train.json"
RETAIN_FILE = Path(ARGS.retain_file).expanduser() if ARGS.retain_file else ROOT / "final_retain_train_4040.json"
REF_CACHE = Path(ARGS.ref_cache).expanduser() if ARGS.ref_cache else ROOT / "xlam2_3b_haru_reference_logps_native.pt"
OUTPUT_DIR = Path(ARGS.output_dir).expanduser()
SAFE_TARGET_TEXT = "[]"  # native xLAM-2 no-tool response

if ARGS.epochs < 1:
    raise ValueError("--epochs must be >= 1")
if ARGS.beta <= 0:
    raise ValueError("--beta must be > 0")
if ARGS.safe_coef < 0 or ARGS.retain_coef < 0:
    raise ValueError("--safe-coef and --retain-coef must be non-negative")
if ARGS.retain_per_forget < 1 or ARGS.grad_accum < 1 or ARGS.ref_batch_size < 1:
    raise ValueError("retain-per-forget, grad-accum and ref-batch-size must be >= 1")

random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)
torch.backends.cuda.matmul.allow_tf32 = True

with open(FORGET_FILE) as f:
    forget_rows = json.load(f)
with open(RETAIN_FILE) as f:
    retain_rows = json.load(f)

assert len(forget_rows) == 2020, len(forget_rows)
assert len(retain_rows) == 4040, len(retain_rows)

tokenizer = AutoTokenizer.from_pretrained(ARGS.model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.eos_token_id is None:
    raise RuntimeError("Tokenizer has no eos_token_id")


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


def build_target(row):
    # Native xLAM-2 target is the direct JSON list, no {"tool_calls": ...} wrapper.
    return json.dumps(get_answers(row), ensure_ascii=False, separators=(",", ":"))


def prompt_ids_for_row(row):
    messages = [{"role": "user", "content": get_query(row)}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=get_tools(row),
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def encode_row_with_target(row, target_text, idx):
    prompt_ids = prompt_ids_for_row(row)
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > ARGS.max_length:
        raise RuntimeError(f"Example {idx} length {len(input_ids)} > max_length={ARGS.max_length}")
    return {"idx": idx, "input_ids": input_ids, "labels": labels}


class ToolDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return encode_row_with_target(self.rows[idx], build_target(self.rows[idx]), idx)


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


def target_suffix_forward(model, batch):
    input_ids = batch["input_ids"].to(ARGS.device)
    attention_mask = batch["attention_mask"].to(ARGS.device)
    labels = batch["labels"].to(ARGS.device)

    # For the training microbatch=1, avoid materializing full-sequence logits when supported.
    if input_ids.shape[0] == 1:
        supervised = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
        if supervised.numel() == 0:
            raise RuntimeError("No supervised target tokens")
        first_target = int(supervised[0].item())
        keep = labels.shape[1] - first_target + 1
        try:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                num_logits_to_keep=keep,
            )
            return out, labels[:, -keep:]
        except TypeError:
            pass

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out, labels


def sequence_logp(logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0
    token_logps = F.log_softmax(shift_logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def ce_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def forward_logp(model, batch):
    out, labels = target_suffix_forward(model, batch)
    return sequence_logp(out.logits, labels), out


forget_dataset = ToolDataset(forget_rows)
retain_dataset = ToolDataset(retain_rows)

# -------------------- NPO frozen reference cache --------------------
if REF_CACHE.exists():
    print("Loading reference cache:", REF_CACHE)
    payload = torch.load(REF_CACHE, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Old/incompatible reference cache; use a fresh 3B-native cache path")
    if payload.get("model") != ARGS.model_name:
        raise RuntimeError("Reference-cache model mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError("Reference cache is not native xLAM-2 format")
    if payload.get("max_length") != ARGS.max_length:
        raise RuntimeError("Reference-cache max_length mismatch")
    reference_logps = payload.get("forget_logps")
    if reference_logps is None or len(reference_logps) != len(forget_dataset):
        raise RuntimeError("Invalid forget_logps in reference cache")
else:
    print("=" * 90)
    print("PRECOMPUTING xLAM-2 3B NATIVE NPO REFERENCE LOG-PROBABILITIES")
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
            ids = batch["idx"]
            lp, out = forward_logp(reference_model, batch)
            reference_logps[ids] = lp.detach().cpu()
            del out
            if step % 100 == 0 or step + 1 == len(reference_loader):
                print(f"Reference {step+1}/{len(reference_loader)}", flush=True)

    REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": ARGS.model_name,
            "format_version": FORMAT_VERSION,
            "max_length": ARGS.max_length,
            "forget_logps": reference_logps,
        },
        REF_CACHE,
    )
    print("Saved reference cache:", REF_CACHE)
    del reference_model, reference_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------- API-balanced data loaders --------------------
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
print("HARU xLAM-2 3B NATIVE CONFIG")
print("=" * 90)
print("Model:", ARGS.model_name)
print("Objective: NPO + safe_coef*CE_safe + retain_coef*CE_retain")
print("Epochs:", ARGS.epochs)
print("Beta:", ARGS.beta)
print("Gamma / safe coefficient:", ARGS.safe_coef)
print("Lambda / retain coefficient:", ARGS.retain_coef)
print("LR:", ARGS.learning_rate)
print("Warmup ratio:", ARGS.warmup_ratio)
print("Safe target:", SAFE_TARGET_TEXT)
print("Reference cache:", REF_CACHE)
print("Output dir:", OUTPUT_DIR)

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
warmup_steps = (
    max(1, int(ARGS.warmup_ratio * total_steps))
    if ARGS.warmup_ratio > 0 and total_steps > 1
    else 0
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
history = []
optimizer.zero_grad(set_to_none=True)

for epoch in range(ARGS.epochs):
    ep = epoch + 1
    print("\n" + "=" * 90)
    print(f"HARU xLAM-2 3B EPOCH {ep}/{ARGS.epochs}")
    print("=" * 90)

    retain_iter = iter(retain_loader)
    npo_hist, safe_hist, retain_hist, ratio_hist, grad_hist = [], [], [], [], []

    for step, fbatch in enumerate(forget_loader):
        f_idx = fbatch["idx"]

        # 1) NPO forget loss on original target completion.
        outputs, f_labels = target_suffix_forward(model, fbatch)
        current_logp = sequence_logp(outputs.logits, f_labels)
        ref_logp = reference_logps[f_idx].to(ARGS.device, dtype=torch.float32)
        log_ratio = current_logp - ref_logp
        npo_loss = (-(2.0 / ARGS.beta) * F.logsigmoid(-ARGS.beta * log_ratio)).mean()
        (npo_loss / ARGS.grad_accum).backward()
        npo_hist.append(float(npo_loss.detach()))
        ratio_hist.append(float(log_ratio.mean().detach()))
        del outputs

        # 2) Safe loss on same forget prompt -> []
        row_idx = int(f_idx.item())
        safe_ex = encode_row_with_target(forget_rows[row_idx], SAFE_TARGET_TEXT, row_idx)
        safe_batch = collate([safe_ex])
        sout, s_labels = target_suffix_forward(model, safe_batch)
        safe_ce = ce_loss(sout.logits, s_labels)
        (ARGS.safe_coef * safe_ce / ARGS.grad_accum).backward()
        safe_hist.append(float(safe_ce.detach()))
        del sout

        # 3) Retain CE.
        step_retain = []
        for _ in range(ARGS.retain_per_forget):
            rbatch = next(retain_iter)
            rout, r_labels = target_suffix_forward(model, rbatch)
            retain_ce = ce_loss(rout.logits, r_labels)
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
            grad_hist.append(float(gn.detach().cpu()) if torch.is_tensor(gn) else float(gn))

        if step % 50 == 0 or final_step:
            print(
                f"epoch={ep} step={step+1:4d}/{len(forget_loader)} "
                f"NPO={np.mean(npo_hist[-50:]):.4f} "
                f"safeCE={np.mean(safe_hist[-50:]):.4f} "
                f"retainCE={np.mean(retain_hist[-50:]):.4f} "
                f"logratio={np.mean(ratio_hist[-50:]):+.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    record = {
        "epoch": ep,
        "npo_loss": float(np.mean(npo_hist)),
        "safe_ce": float(np.mean(safe_hist)),
        "retain_ce": float(np.mean(retain_hist)),
        "forget_log_ratio": float(np.mean(ratio_hist)),
        "mean_grad_norm": float(np.mean(grad_hist)) if grad_hist else None,
    }
    history.append(record)
    print(json.dumps(record, indent=2))

    # Save every epoch.
    epoch_dir = OUTPUT_DIR / f"epoch_{ep}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(epoch_dir, safe_serialization=True)
    tokenizer.save_pretrained(epoch_dir)
    model.config.use_cache = False

    config = {
        "base_model": ARGS.model_name,
        "method": "HARU-safe-target",
        "architecture": "xLAM-2-3B",
        "format_version": FORMAT_VERSION,
        "objective": "NPO_forget + safe_coef * CE_safe + retain_coef * CE_retain",
        "beta": ARGS.beta,
        "safe_coef": ARGS.safe_coef,
        "safe_target_text": SAFE_TARGET_TEXT,
        "retain_coef": ARGS.retain_coef,
        "learning_rate": ARGS.learning_rate,
        "weight_decay": ARGS.weight_decay,
        "epochs_requested": ARGS.epochs,
        "checkpoint_epoch": ep,
        "max_length": ARGS.max_length,
        "grad_accum": ARGS.grad_accum,
        "retain_per_forget": ARGS.retain_per_forget,
        "max_grad_norm": ARGS.max_grad_norm,
        "ref_batch_size": ARGS.ref_batch_size,
        "warmup_ratio": ARGS.warmup_ratio,
        "reference_cache": str(REF_CACHE),
        "forget_n": len(forget_rows),
        "retain_n": len(retain_rows),
        "seed": ARGS.seed,
        "cli_args": vars(ARGS),
        "history": history,
    }
    with open(epoch_dir / "unlearning_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved checkpoint:", epoch_dir)

print("\n" + "=" * 90)
print("HARU xLAM-2 3B TRAINING COMPLETE")
print("Final checkpoint:", OUTPUT_DIR / f"epoch_{ARGS.epochs}")
