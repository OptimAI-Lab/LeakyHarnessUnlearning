#!/usr/bin/env python3
"""
8-GPU retain-only retraining baseline for xLAM-2 3B / BFCL.

- Base model: Qwen/Qwen2.5-3B-Instruct
- Tokenizer/interface: Salesforce/xLAM-2-3b-fc-r
- Train only on the full retain corpus.
- Native xLAM-2 targets are direct JSON lists.
- DDP over multiple GPUs.
- Response-only CE loss.
- max_length=3072; preserve complete target and truncate only prompt.
- Left-pad minibatches so target suffixes align.
- Project vocabulary logits only over the target suffix for memory efficiency.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

NATIVE_FORMAT_VERSION = "xlam2_native_tools_json_list_v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--tokenizer-source", default="Salesforce/xLAM-2-3b-fc-r")
    p.add_argument("--retain-file", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)

    # Global effective batch = batch_size * world_size * grad_accum.
    # Recommended on 8 GPUs: 2 * 8 * 4 = 64.
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)

    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


ARGS = parse_args()

if ARGS.epochs < 1:
    raise ValueError("--epochs must be >= 1")
if ARGS.batch_size < 1:
    raise ValueError("--batch-size must be >= 1")
if ARGS.grad_accum < 1:
    raise ValueError("--grad-accum must be >= 1")

if "LOCAL_RANK" not in os.environ:
    raise RuntimeError(
        "Launch with the active conda Python, e.g.\n"
        '  "$CONDA_PREFIX/bin/python" -m torch.distributed.run '
        "--standalone --nproc_per_node=8 train_retrain_xlam_3b_ddp_v2.py ..."
    )

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group(backend="nccl")

seed = ARGS.seed + rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cuda.matmul.allow_tf32 = True

RETAIN_FILE = Path(ARGS.retain_file).expanduser()
OUTPUT_DIR = Path(ARGS.output_dir).expanduser()

with open(RETAIN_FILE, "r", encoding="utf-8") as f:
    retain_rows = json.load(f)

if not retain_rows:
    raise RuntimeError("Retain file is empty.")

tokenizer = AutoTokenizer.from_pretrained(
    ARGS.tokenizer_source,
    trust_remote_code=True,
)

if not hasattr(tokenizer, "apply_chat_template"):
    raise RuntimeError(
        "Tokenizer does not provide apply_chat_template. "
        f"Python={sys.executable}; transformers={transformers.__version__}. "
        "Launch from the xlam3b conda environment."
    )

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.eos_token_id is None:
    raise RuntimeError("Tokenizer has no eos_token_id.")


def maybe_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def get_query(row):
    q = maybe_json(row.get("query", ""))
    if isinstance(q, str):
        return q
    return json.dumps(q, ensure_ascii=False)


def get_tools(row):
    tools = maybe_json(row.get("tools", []))
    if isinstance(tools, dict):
        tools = [tools]
    return tools


def get_answers(row):
    ans = maybe_json(row.get("answers", []))
    if isinstance(ans, dict):
        ans = [ans]
    return ans


def build_target(row):
    return json.dumps(
        get_answers(row),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def head_tail_truncate(ids, budget):
    if len(ids) <= budget:
        return ids
    if budget <= 0:
        return []

    head = (budget + 1) // 2
    tail = budget - head

    if tail == 0:
        return ids[:head]

    return ids[:head] + ids[-tail:]


def encode_row(row, idx):
    messages = [
        {
            "role": "user",
            "content": get_query(row),
        }
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tools=get_tools(row),
        add_generation_prompt=True,
        tokenize=False,
    )

    prompt_ids = tokenizer(
        rendered,
        add_special_tokens=False,
    )["input_ids"]

    target_ids = tokenizer(
        build_target(row),
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    if len(target_ids) >= ARGS.max_length:
        raise RuntimeError(
            f"Retain example {idx} target alone has {len(target_ids)} tokens, "
            f"which is >= max_length={ARGS.max_length}"
        )

    prompt_budget = ARGS.max_length - len(target_ids)

    if len(prompt_ids) > prompt_budget:
        original_len = len(prompt_ids)
        prompt_ids = head_tail_truncate(prompt_ids, prompt_budget)

        if rank == 0 and idx % 5000 == 0:
            print(
                f"[truncate] example={idx} "
                f"prompt={original_len}->{len(prompt_ids)} "
                f"target={len(target_ids)} "
                f"total={len(prompt_ids)+len(target_ids)}",
                flush=True,
            )

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    assert len(input_ids) <= ARGS.max_length
    assert len(input_ids) == len(labels)

    return {
        "idx": idx,
        "input_ids": input_ids,
        "labels": labels,
        "target_len": len(target_ids),
    }


class RetainDataset(Dataset):
    def __len__(self):
        return len(retain_rows)

    def __getitem__(self, idx):
        return encode_row(retain_rows[idx], idx)


def collate(batch):
    # LEFT padding aligns all targets to the right edge.
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    attention_mask = []
    target_lens = []
    idxs = []

    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n

        input_ids.append(
            [tokenizer.pad_token_id] * pad + x["input_ids"]
        )
        labels.append(
            [-100] * pad + x["labels"]
        )
        attention_mask.append(
            [0] * pad + [1] * n
        )
        target_lens.append(x["target_len"])
        idxs.append(x["idx"])

    return {
        "idx": torch.tensor(idxs, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "target_lens": torch.tensor(target_lens, dtype=torch.long),
    }


dataset = RetainDataset()

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=ARGS.seed,
    drop_last=False,
)

loader = DataLoader(
    dataset,
    batch_size=ARGS.batch_size,
    sampler=sampler,
    collate_fn=collate,
    pin_memory=True,
    num_workers=ARGS.num_workers,
)


class QwenTargetSuffixModel(nn.Module):
    """
    Run the full Qwen backbone over the prompt, but apply lm_head only
    to the final target suffix. This avoids materializing vocabulary
    logits for all prompt positions.
    """

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

        if not hasattr(actor, "model"):
            raise RuntimeError("Expected Qwen-style `.model` backbone.")
        if not hasattr(actor, "lm_head"):
            raise RuntimeError("Expected Qwen-style `.lm_head` layer.")

    def forward(self, input_ids, attention_mask, keep_logits):
        outputs = self.actor.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state[:, -keep_logits:, :]
        logits = self.actor.lm_head(hidden)
        return logits


actor = AutoModelForCausalLM.from_pretrained(
    ARGS.base_model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

embedding_count = actor.get_input_embeddings().weight.shape[0]

if len(tokenizer) > embedding_count:
    if rank == 0:
        print(
            f"Expanding token embeddings: "
            f"{embedding_count} -> {len(tokenizer)}"
        )
    actor.resize_token_embeddings(len(tokenizer))

elif len(tokenizer) < embedding_count:
    if rank == 0:
        print(
            f"Keeping base-model embedding table at {embedding_count}; "
            f"tokenizer length is {len(tokenizer)} (no shrink)."
        )

actor.config.use_cache = False

try:
    actor.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
except TypeError:
    actor.gradient_checkpointing_enable()

wrapped_model = QwenTargetSuffixModel(actor).to(device)

model = DDP(
    wrapped_model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,
    broadcast_buffers=False,
)

model.train()


def target_ce_loss(suffix_logits, suffix_labels):
    shift_logits = suffix_logits[:, :-1, :].contiguous()
    shift_labels = suffix_labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=ARGS.learning_rate,
    weight_decay=ARGS.weight_decay,
)

batches_per_epoch = len(loader)
optimizer_steps_per_epoch = math.ceil(
    batches_per_epoch / ARGS.grad_accum
)
total_optimizer_steps = ARGS.epochs * optimizer_steps_per_epoch

warmup_steps = (
    max(1, int(ARGS.warmup_ratio * total_optimizer_steps))
    if ARGS.warmup_ratio > 0 and total_optimizer_steps > 1
    else 0
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optimizer_steps,
)

if rank == 0:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dist.barrier()
optimizer.zero_grad(set_to_none=True)

history = []

if rank == 0:
    print("=" * 96)
    print("xLAM-2 3B RETRAIN BASELINE — DDP")
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("=" * 96)
    print("Base model:", ARGS.base_model)
    print("Tokenizer/interface:", ARGS.tokenizer_source)
    print("Native format:", NATIVE_FORMAT_VERSION)
    print("World size / GPUs:", world_size)
    print("Retain examples:", len(dataset))
    print("Per-GPU batch size:", ARGS.batch_size)
    print("Gradient accumulation:", ARGS.grad_accum)
    print(
        "Global effective batch:",
        ARGS.batch_size * world_size * ARGS.grad_accum,
    )
    print("Local batches / epoch:", batches_per_epoch)
    print("Optimizer steps / epoch:", optimizer_steps_per_epoch)
    print("Epochs:", ARGS.epochs)
    print("Output:", OUTPUT_DIR)
    print("=" * 96)


for epoch in range(ARGS.epochs):
    ep = epoch + 1
    sampler.set_epoch(epoch)

    losses = []
    grad_norms = []

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        max_target_len = int(
            batch["target_lens"].max().item()
        )

        # One extra causal position is needed to predict the first target token.
        keep_logits = max_target_len + 1

        # Correctly scale a final partial accumulation group.
        group_start = (
            ((step - 1) // ARGS.grad_accum) * ARGS.grad_accum + 1
        )
        group_end = min(
            group_start + ARGS.grad_accum - 1,
            batches_per_epoch,
        )
        accum_this_group = group_end - group_start + 1
        should_sync = step == group_end

        sync_context = (
            torch.enable_grad()
            if should_sync
            else model.no_sync()
        )

        with sync_context:
            suffix_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                keep_logits=keep_logits,
            )

            suffix_labels = labels[:, -keep_logits:]

            loss = target_ce_loss(
                suffix_logits,
                suffix_labels,
            )

            (loss / accum_this_group).backward()

        losses.append(float(loss.detach()))
        del suffix_logits

        if should_sync:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                ARGS.max_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            grad_norms.append(
                float(grad_norm.detach().cpu())
                if torch.is_tensor(grad_norm)
                else float(grad_norm)
            )

        if rank == 0 and (
            step % 50 == 0
            or step == batches_per_epoch
        ):
            print(
                f"epoch={ep} "
                f"local_batch={step}/{batches_per_epoch} "
                f"retainCE={np.mean(losses[-50:]):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    local_mean = torch.tensor(
        [float(np.mean(losses))],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        local_mean,
        op=dist.ReduceOp.SUM,
    )

    global_mean = float(
        local_mean.item() / world_size
    )

    if rank == 0:
        record = {
            "epoch": ep,
            "retain_ce": global_mean,
            "mean_grad_norm_rank0": (
                float(np.mean(grad_norms))
                if grad_norms
                else None
            ),
        }

        history.append(record)
        print(json.dumps(record, indent=2))

        epoch_dir = OUTPUT_DIR / f"epoch_{ep}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        raw_actor = model.module.actor
        raw_actor.config.use_cache = True

        raw_actor.save_pretrained(
            epoch_dir,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(epoch_dir)

        raw_actor.config.use_cache = False

        with open(
            epoch_dir / "retrain_config.json",
            "w",
        ) as f:
            json.dump(
                {
                    "method": "retrain-retain-only-ddp",
                    "base_model": ARGS.base_model,
                    "tokenizer_source": ARGS.tokenizer_source,
                    "interface": NATIVE_FORMAT_VERSION,
                    "retain_file": str(RETAIN_FILE),
                    "retain_n": len(dataset),
                    "world_size": world_size,
                    "batch_size_per_gpu": ARGS.batch_size,
                    "grad_accum": ARGS.grad_accum,
                    "global_effective_batch": (
                        ARGS.batch_size
                        * world_size
                        * ARGS.grad_accum
                    ),
                    "learning_rate": ARGS.learning_rate,
                    "weight_decay": ARGS.weight_decay,
                    "max_grad_norm": ARGS.max_grad_norm,
                    "max_length": ARGS.max_length,
                    "warmup_ratio": ARGS.warmup_ratio,
                    "seed": ARGS.seed,
                    "checkpoint_epoch": ep,
                    "history": history,
                    "cli_args": vars(ARGS),
                },
                f,
                indent=2,
            )

        print("Saved checkpoint:", epoch_dir)

    dist.barrier()


if rank == 0:
    print(
        "Final checkpoint:",
        OUTPUT_DIR / f"epoch_{ARGS.epochs}",
    )

dist.destroy_process_group()
