#!/usr/bin/env python3
import argparse
import gc
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


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


def parse_args():
    p = argparse.ArgumentParser(
        description="TOOLDELETE-SFT / TOOLDELETE-DPO adapted to the fixed xLAM/BFCL setup."
    )
    p.add_argument("--method", choices=["sft", "dpo"], required=True)
    p.add_argument("--model-name", default="Salesforce/xLAM-1b-fc-r")
    p.add_argument("--root", default="~/Desktop/BFCL")
    p.add_argument("--forget-file", default=None)
    p.add_argument("--retain-file", default=None)
    p.add_argument("--toolfree-cache", default=None)
    p.add_argument("--dpo-ref-cache", default=None)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--forget-coef", type=float, default=1.0)
    p.add_argument("--retain-coef", type=float, default=1.0)
    p.add_argument("--dpo-beta", type=float, default=0.1)

    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--retain-per-forget", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--ref-batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


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
    if isinstance(tools, dict):
        tools = [tools]
    return tools


def get_answers(row):
    answers = maybe_json(row.get("answers", []))
    if isinstance(answers, dict):
        answers = [answers]
    return answers


def target_names(row):
    return [x["name"] for x in get_answers(row) if isinstance(x, dict) and x.get("name")]


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


def build_original_target(row):
    return json.dumps(
        {"tool_calls": get_answers(row)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main():
    args = parse_args()
    root = Path(args.root).expanduser()
    forget_file = Path(args.forget_file).expanduser() if args.forget_file else root / "final_forget_train.json"
    retain_file = Path(args.retain_file).expanduser() if args.retain_file else root / "final_retain_train_4040.json"
    toolfree_cache_path = Path(args.toolfree_cache).expanduser() if args.toolfree_cache else root / "tooldelete_toolfree_deepseek13b.json"
    dpo_ref_cache_path = Path(args.dpo_ref_cache).expanduser() if args.dpo_ref_cache else root / "tooldelete_dpo_reference_logps.pt"
    output_dir = Path(args.output_dir).expanduser()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.forget_coef < 0 or args.retain_coef < 0:
        raise ValueError("Loss coefficients must be non-negative")
    if args.dpo_beta <= 0:
        raise ValueError("--dpo-beta must be > 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    with open(forget_file) as f:
        forget_rows = json.load(f)
    with open(retain_file) as f:
        retain_rows = json.load(f)
    with open(toolfree_cache_path) as f:
        toolfree_cache = json.load(f)

    assert len(forget_rows) == 2020, len(forget_rows)
    assert len(retain_rows) == 4040, len(retain_rows)
    if len(toolfree_cache.get("forget", [])) != len(forget_rows):
        raise RuntimeError("Tool-free forget cache length mismatch")
    if len(toolfree_cache.get("retain", [])) != len(retain_rows):
        raise RuntimeError("Tool-free retain cache length mismatch")
    if any(not isinstance(x, str) or not x.strip() for x in toolfree_cache["forget"]):
        raise RuntimeError("Tool-free forget cache contains missing/empty responses")
    if any(not isinstance(x, str) or not x.strip() for x in toolfree_cache["retain"]):
        raise RuntimeError("Tool-free retain cache contains missing/empty responses")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prompt_ids_for_row(row):
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_user_prompt(row)}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer(chat_text, add_special_tokens=True)["input_ids"]

    def encode_target(row, target_text, idx):
        prompt_ids = prompt_ids_for_row(row)
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) > args.max_length:
            raise RuntimeError(
                f"Example {idx} length {len(input_ids)} > max_length={args.max_length}. "
                "Do not silently truncate TOOLDELETE targets."
            )
        return {"idx": idx, "input_ids": input_ids, "labels": labels}

    class SFTDataset(Dataset):
        def __init__(self, rows, targets):
            self.rows = rows
            self.targets = targets
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            return encode_target(self.rows[idx], self.targets[idx], idx)

    class PairDataset(Dataset):
        def __init__(self, rows, chosen_targets, rejected_targets):
            self.rows = rows
            self.chosen_targets = chosen_targets
            self.rejected_targets = rejected_targets
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            return {
                "idx": idx,
                "chosen": encode_target(self.rows[idx], self.chosen_targets[idx], idx),
                "rejected": encode_target(self.rows[idx], self.rejected_targets[idx], idx),
            }

    def pad_examples(examples):
        max_len = max(len(x["input_ids"]) for x in examples)
        input_ids, labels, masks, indices = [], [], [], []
        for x in examples:
            n = len(x["input_ids"])
            pad = max_len - n
            input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            masks.append([1] * n + [0] * pad)
            indices.append(x["idx"])
        return {
            "idx": torch.tensor(indices, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def sft_collate(batch):
        return pad_examples(batch)

    def pair_collate(batch):
        return {
            "idx": torch.tensor([x["idx"] for x in batch], dtype=torch.long),
            "chosen": pad_examples([x["chosen"] for x in batch]),
            "rejected": pad_examples([x["rejected"] for x in batch]),
        }

    def token_ce_loss(logits, labels):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def sequence_logp(logits, labels):
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        safe = shift_labels.clone()
        safe[~mask] = 0
        token_logps = F.log_softmax(shift_logits.float(), dim=-1).gather(
            -1, safe.unsqueeze(-1)
        ).squeeze(-1)
        return (token_logps * mask).sum(dim=-1)

    def forward_logp(model, batch):
        out = model(
            input_ids=batch["input_ids"].to(args.device),
            attention_mask=batch["attention_mask"].to(args.device),
            use_cache=False,
        )
        lp = sequence_logp(out.logits, batch["labels"].to(args.device))
        return lp, out

    original_forget = [build_original_target(x) for x in forget_rows]
    original_retain = [build_original_target(x) for x in retain_rows]
    toolfree_forget = [x.strip() for x in toolfree_cache["forget"]]
    toolfree_retain = [x.strip() for x in toolfree_cache["retain"]]

    if args.method == "sft":
        # Paper: Tf uses Y'_f=f0(Q_f); Tr remains unmodified.
        forget_dataset = SFTDataset(forget_rows, toolfree_forget)
        retain_dataset = SFTDataset(retain_rows, original_retain)
        collate = sft_collate
    else:
        # Paper: Tf prefers tool-free over original; Tr prefers original over tool-free.
        forget_dataset = PairDataset(forget_rows, toolfree_forget, original_forget)
        retain_dataset = PairDataset(retain_rows, original_retain, toolfree_retain)
        collate = pair_collate

    # Same API-balanced forget exposure as the user's NPO baseline.
    api_counts = Counter()
    for row in forget_rows:
        for name in set(target_names(row)):
            api_counts[name] += 1

    weights = []
    for row in forget_rows:
        names = set(target_names(row))
        if not names:
            raise RuntimeError("Forget row has no target API")
        weights.append(float(np.mean([1.0 / api_counts[n] for n in names])))

    forget_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(forget_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=1,
        sampler=forget_sampler,
        collate_fn=collate,
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        collate_fn=collate,
    )
    assert len(retain_loader) == args.retain_per_forget * len(forget_loader)

    # DPO reference cache from the original xLAM model.
    reference_cache = None
    if args.method == "dpo":
        if dpo_ref_cache_path.exists():
            print("Loading DPO reference cache:", dpo_ref_cache_path)
            reference_cache = torch.load(dpo_ref_cache_path, map_location="cpu")
        else:
            print("=" * 90)
            print("PRECOMPUTING TOOLDELETE-DPO REFERENCE LOG-PROBABILITIES")
            print("=" * 90)
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(args.device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

            def cache_pairs(dataset, label):
                loader = DataLoader(
                    dataset,
                    batch_size=args.ref_batch_size,
                    shuffle=False,
                    collate_fn=pair_collate,
                )
                ch = torch.empty(len(dataset), dtype=torch.float32)
                rj = torch.empty(len(dataset), dtype=torch.float32)
                with torch.inference_mode():
                    for step, batch in enumerate(loader):
                        ids = batch["idx"]
                        c_lp, c_out = forward_logp(ref_model, batch["chosen"])
                        r_lp, r_out = forward_logp(ref_model, batch["rejected"])
                        ch[ids] = c_lp.detach().cpu()
                        rj[ids] = r_lp.detach().cpu()
                        del c_out, r_out
                        if step % 100 == 0 or step + 1 == len(loader):
                            print(f"{label} reference {step+1}/{len(loader)}")
                return ch, rj

            f_ch, f_rj = cache_pairs(forget_dataset, "forget")
            r_ch, r_rj = cache_pairs(retain_dataset, "retain")
            reference_cache = {
                "model": args.model_name,
                "forget_chosen": f_ch,
                "forget_rejected": f_rj,
                "retain_chosen": r_ch,
                "retain_rejected": r_rj,
            }
            torch.save(reference_cache, dpo_ref_cache_path)
            print("Saved DPO reference cache:", dpo_ref_cache_path)
            del ref_model
            gc.collect()
            torch.cuda.empty_cache()

        for key, n in [("forget_chosen", len(forget_rows)), ("forget_rejected", len(forget_rows)),
                       ("retain_chosen", len(retain_rows)), ("retain_rejected", len(retain_rows))]:
            if key not in reference_cache or len(reference_cache[key]) != n:
                raise RuntimeError(f"Invalid DPO reference cache entry: {key}")

    print("=" * 90)
    print(f"TOOLDELETE-{args.method.upper()} CONFIG")
    print("=" * 90)
    print("Model:", args.model_name)
    print("Tool-free cache:", toolfree_cache_path)
    print("Output dir:", output_dir)
    print("Epochs:", args.epochs)
    print("Forget coefficient:", args.forget_coef)
    print("Retain coefficient:", args.retain_coef)
    if args.method == "dpo":
        print("DPO beta:", args.dpo_beta)
    print("LR:", args.learning_rate)
    print("Seed:", args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    opt_steps_per_epoch = math.ceil(len(forget_loader) / args.grad_accum)
    total_opt_steps = args.epochs * opt_steps_per_epoch
    warmup_steps = max(1, int(args.warmup_ratio * total_opt_steps)) if total_opt_steps > 1 else 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    def dpo_loss_for_batch(pair_batch, ref_chosen, ref_rejected):
        c_lp, c_out = forward_logp(model, pair_batch["chosen"])
        r_lp, r_out = forward_logp(model, pair_batch["rejected"])
        ref_c = ref_chosen.to(args.device, dtype=torch.float32)
        ref_r = ref_rejected.to(args.device, dtype=torch.float32)
        pi_logratio = c_lp - r_lp
        ref_logratio = ref_c - ref_r
        logits = args.dpo_beta * (pi_logratio - ref_logratio)
        loss = -F.logsigmoid(logits).mean()
        reward_margin = (pi_logratio - ref_logratio).mean()
        return loss, reward_margin, c_out, r_out

    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        ep = epoch + 1
        print("\n" + "=" * 90)
        print(f"TOOLDELETE-{args.method.upper()} EPOCH {ep}/{args.epochs}")
        print("=" * 90)
        retain_iter = iter(retain_loader)

        forget_losses, retain_losses, margins = [], [], []
        grad_norms = []

        for step, fbatch in enumerate(forget_loader):
            f_idx = fbatch["idx"]

            if args.method == "sft":
                f_out = model(
                    input_ids=fbatch["input_ids"].to(args.device),
                    attention_mask=fbatch["attention_mask"].to(args.device),
                    use_cache=False,
                )
                f_loss = token_ce_loss(f_out.logits, fbatch["labels"].to(args.device))
                (args.forget_coef * f_loss / args.grad_accum).backward()
                forget_losses.append(float(f_loss.detach()))
                del f_out
            else:
                f_loss, margin, c_out, r_out = dpo_loss_for_batch(
                    fbatch,
                    reference_cache["forget_chosen"][f_idx],
                    reference_cache["forget_rejected"][f_idx],
                )
                (args.forget_coef * f_loss / args.grad_accum).backward()
                forget_losses.append(float(f_loss.detach()))
                margins.append(float(margin.detach()))
                del c_out, r_out

            step_retain = []
            for _ in range(args.retain_per_forget):
                rbatch = next(retain_iter)
                r_idx = rbatch["idx"]

                if args.method == "sft":
                    r_out = model(
                        input_ids=rbatch["input_ids"].to(args.device),
                        attention_mask=rbatch["attention_mask"].to(args.device),
                        use_cache=False,
                    )
                    r_loss = token_ce_loss(r_out.logits, rbatch["labels"].to(args.device))
                    (
                        args.retain_coef
                        * r_loss
                        / args.retain_per_forget
                        / args.grad_accum
                    ).backward()
                    del r_out
                else:
                    r_loss, margin, c_out, r_out = dpo_loss_for_batch(
                        rbatch,
                        reference_cache["retain_chosen"][r_idx],
                        reference_cache["retain_rejected"][r_idx],
                    )
                    (
                        args.retain_coef
                        * r_loss
                        / args.retain_per_forget
                        / args.grad_accum
                    ).backward()
                    margins.append(float(margin.detach()))
                    del c_out, r_out

                step_retain.append(float(r_loss.detach()))

            retain_losses.append(float(np.mean(step_retain)))

            boundary = (step + 1) % args.grad_accum == 0
            final_step = step + 1 == len(forget_loader)
            if boundary or final_step:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                grad_norms.append(float(gn.detach().cpu()) if torch.is_tensor(gn) else float(gn))

            if step % 50 == 0 or final_step:
                msg = (
                    f"epoch={ep} step={step+1:4d}/{len(forget_loader)} "
                    f"forget={np.mean(forget_losses[-50:]):.4f} "
                    f"retain={np.mean(retain_losses[-50:]):.4f} "
                )
                if args.method == "dpo" and margins:
                    msg += f"margin={np.mean(margins[-100:]):+.4f} "
                msg += f"lr={scheduler.get_last_lr()[0]:.2e}"
                print(msg)

        record = {
            "epoch": ep,
            "forget_loss": float(np.mean(forget_losses)),
            "retain_loss": float(np.mean(retain_losses)),
            "dpo_reward_margin": float(np.mean(margins)) if margins else None,
            "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
        }
        history.append(record)
        print("Epoch summary:")
        print(json.dumps(record, indent=2))

    # Save post-optimization weights BEFORE paper's GCR task arithmetic.
    pre_dir = output_dir / f"epoch_{args.epochs}_pre_task_arithmetic"
    pre_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(pre_dir, safe_serialization=True)
    tokenizer.save_pretrained(pre_dir)

    config = {
        "base_model": args.model_name,
        "method": f"TOOLDELETE-{args.method.upper()}",
        "tool_free_model": toolfree_cache.get("metadata", {}).get("tool_free_model"),
        "toolfree_cache": str(toolfree_cache_path),
        "dpo_ref_cache": str(dpo_ref_cache_path) if args.method == "dpo" else None,
        "forget_coef": args.forget_coef,
        "retain_coef": args.retain_coef,
        "dpo_beta": args.dpo_beta if args.method == "dpo" else None,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "grad_accum": args.grad_accum,
        "retain_per_forget": args.retain_per_forget,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "forget_n": len(forget_rows),
        "retain_n": len(retain_rows),
        "seed": args.seed,
        "history": history,
        "cli_args": vars(args),
    }
    with open(pre_dir / "unlearning_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 90)
    print("POST-OPTIMIZATION TOOLDELETE CHECKPOINT SAVED")
    print("=" * 90)
    print(pre_dir)
    print("Apply task arithmetic next to obtain the full TOOLDELETE checkpoint.")


if __name__ == "__main__":
    main()
