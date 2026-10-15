#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import transformers
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="8-GPU DDP retain-only retraining baseline for xLAM-1B/BFCL."
    )
    p.add_argument("--base-model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--tokenizer-source", default="Salesforce/xLAM-1b-fc-r")
    p.add_argument("--retain-file", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)

    # Per-GPU batch. Global effective batch =
    # batch_size * world_size * grad_accum.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)

    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


ARGS = parse_args()

if "LOCAL_RANK" not in os.environ:
    raise RuntimeError(
        "This script must be launched with torchrun, e.g. "
        "`torchrun --standalone --nproc_per_node=8 train_retrain_xlam_1b_ddp.py ...`"
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

tokenizer = AutoTokenizer.from_pretrained(
    ARGS.tokenizer_source,
    trust_remote_code=True,
)

if not hasattr(tokenizer, "apply_chat_template"):
    raise RuntimeError(
        "Loaded transformers/tokenizer does not provide apply_chat_template. "
        f"Python={sys.executable}; transformers={transformers.__version__}. "
        "Launch with the active conda environment via "
        "`$CONDA_PREFIX/bin/python -m torch.distributed.run ...`, "
        "not a ~/.local/bin/torchrun from another Python installation."
    )

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    q = maybe_json(row.get("query", ""))
    return q if isinstance(q, str) else json.dumps(q, ensure_ascii=False)


def get_tools(row):
    tools = maybe_json(row.get("tools", []))
    return [tools] if isinstance(tools, dict) else tools


def get_answers(row):
    ans = maybe_json(row.get("answers", []))
    return [ans] if isinstance(ans, dict) else ans


def build_user_prompt(row):
    return (
        "[BEGIN OF TASK INSTRUCTION]\n"
        + TASK_INSTRUCTION
        + "\n[END OF TASK INSTRUCTION]\n\n"
        + "[BEGIN OF AVAILABLE TOOLS]\n"
        + json.dumps(get_tools(row), ensure_ascii=False)
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
        {"tool_calls": get_answers(row)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def head_tail_truncate(ids, budget):
    """
    Preserve both the beginning and the end of a long prompt.

    For BFCL this keeps the task/tool-format prefix at the beginning and
    the query / final tool-schema context at the end, while always
    preserving the complete supervised target.
    """
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
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_user_prompt(row)}],
        add_generation_prompt=True,
        tokenize=False,
    )

    prompt_ids = tokenizer(
        rendered,
        add_special_tokens=True,
    )["input_ids"]

    target_ids = tokenizer(
        build_target(row),
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    # Always preserve the complete tool-call target.  If the rendered
    # prompt is too long for the 3,072-token training limit, truncate
    # only the prompt using a head+tail strategy.
    if len(target_ids) >= ARGS.max_length:
        raise RuntimeError(
            f"Retain example {idx} target alone has {len(target_ids)} tokens, "
            f"which is >= max_length={ARGS.max_length}"
        )

    prompt_budget = ARGS.max_length - len(target_ids)

    if len(prompt_ids) > prompt_budget:
        original_prompt_len = len(prompt_ids)
        prompt_ids = head_tail_truncate(
            prompt_ids,
            prompt_budget,
        )

        # Keep this concise; only rank 0 reports occasional examples.
        if rank == 0 and idx % 5000 == 0:
            print(
                f"[truncate] example={idx} "
                f"prompt={original_prompt_len}->{len(prompt_ids)} "
                f"target={len(target_ids)} total={len(prompt_ids) + len(target_ids)}",
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
    }


class RetainDataset(Dataset):
    def __len__(self):
        return len(retain_rows)

    def __getitem__(self, idx):
        return encode_row(retain_rows[idx], idx)


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n

        input_ids.append(
            x["input_ids"] + [tokenizer.pad_token_id] * pad
        )
        labels.append(
            x["labels"] + [-100] * pad
        )
        attention_mask.append(
            [1] * n + [0] * pad
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def ce_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


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

model = AutoModelForCausalLM.from_pretrained(
    ARGS.base_model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)

embedding_count = model.get_input_embeddings().weight.shape[0]

if len(tokenizer) > embedding_count:
    if rank == 0:
        print(
            f"Expanding token embeddings: "
            f"{embedding_count} -> {len(tokenizer)}"
        )
    model.resize_token_embeddings(len(tokenizer))
elif len(tokenizer) < embedding_count:
    if rank == 0:
        print(
            f"Keeping base-model embedding table at {embedding_count}; "
            f"tokenizer length is {len(tokenizer)} (no shrink)."
        )

model.config.use_cache = False

try:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
except TypeError:
    model.gradient_checkpointing_enable()

model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,
)

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=ARGS.learning_rate,
    weight_decay=ARGS.weight_decay,
)

batches_per_epoch = len(loader)
optimizer_steps_per_epoch = math.ceil(
    batches_per_epoch / ARGS.grad_accum
)
total_optimizer_steps = (
    ARGS.epochs * optimizer_steps_per_epoch
)

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
    print("=" * 88)
    print("xLAM-1B RETRAIN BASELINE — DDP")
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("=" * 88)
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
    print("=" * 88)


for epoch in range(ARGS.epochs):
    ep = epoch + 1
    sampler.set_epoch(epoch)

    losses = []
    grad_norms = []

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Avoid unnecessary DDP gradient synchronization during accumulation.
        should_sync = (
            step % ARGS.grad_accum == 0
            or step == batches_per_epoch
        )

        sync_context = (
            torch.enable_grad()
            if should_sync
            else model.no_sync()
        )

        with sync_context:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            loss = ce_loss(outputs.logits, labels)
            (loss / ARGS.grad_accum).backward()

        losses.append(float(loss.detach()))
        del outputs

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
            step % 50 == 0 or step == batches_per_epoch
        ):
            print(
                f"epoch={ep} "
                f"local_batch={step}/{batches_per_epoch} "
                f"retainCE={np.mean(losses[-50:]):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    # Average epoch loss across ranks for reporting.
    local_mean = torch.tensor(
        [float(np.mean(losses))],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local_mean, op=dist.ReduceOp.SUM)
    global_mean = float(local_mean.item() / world_size)

    if rank == 0:
        record = {
            "epoch": ep,
            "retain_ce": global_mean,
            "mean_grad_norm_rank0": (
                float(np.mean(grad_norms))
                if grad_norms else None
            ),
        }
        history.append(record)

        epoch_dir = OUTPUT_DIR / f"epoch_{ep}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        raw_model = model.module
        raw_model.config.use_cache = True
        raw_model.save_pretrained(
            epoch_dir,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(epoch_dir)
        raw_model.config.use_cache = False

        with open(
            epoch_dir / "retrain_config.json",
            "w",
        ) as f:
            json.dump(
                {
                    "method": "retrain-retain-only-ddp",
                    "base_model": ARGS.base_model,
                    "tokenizer_source": ARGS.tokenizer_source,
                    "interface": "xlam1_legacy_tool_calls_wrapper",
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
