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


NATIVE_FORMAT_VERSION = "xlam2_native_tools_json_list_v1"
MICRO_BATCH = 1


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Unified ToolDelete-SFT / ToolDelete-DPO trainer for "
            "Salesforce/xLAM-2-3b-fc-r using the native xLAM-2 tool interface."
        )
    )
    p.add_argument("--method", choices=["sft", "dpo"], required=True)
    p.add_argument("--model-name", default="Salesforce/xLAM-2-3b-fc-r")
    p.add_argument("--root", default="~/Desktop/BFCL")

    p.add_argument("--forget-file", default=None)
    p.add_argument("--retain-file", default=None)
    p.add_argument("--toolfree-cache", default=None)
    p.add_argument("--dpo-ref-cache", default=None)
    p.add_argument("--output-dir", required=True)

    # If omitted: SFT=1 epoch, DPO=3 epochs.
    p.add_argument("--epochs", type=int, default=None)

    # Paper adaptation uses alpha=0, i.e. no task arithmetic.
    p.add_argument("--alpha", type=float, default=0.0)

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
    p.add_argument("--ref-batch-size", type=int, default=1)
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
    return [
        x["name"]
        for x in get_answers(row)
        if isinstance(x, dict) and x.get("name")
    ]


def build_original_target(row):
    """
    Native xLAM-2 tool-call target:
      [{"name":"...", "arguments":{...}}, ...]
    """
    return json.dumps(
        get_answers(row),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main():
    args = parse_args()

    if args.epochs is None:
        args.epochs = 1 if args.method == "sft" else 3

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.forget_coef < 0 or args.retain_coef < 0:
        raise ValueError("Loss coefficients must be non-negative")
    if args.dpo_beta <= 0:
        raise ValueError("--dpo-beta must be > 0")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.retain_per_forget < 1:
        raise ValueError("--retain-per-forget must be >= 1")
    if args.ref_batch_size < 1:
        raise ValueError("--ref-batch-size must be >= 1")

    # This unified script implements the alpha=0 adaptation only.
    if abs(args.alpha) > 1e-12:
        raise ValueError(
            "This xLAM-2 3B ToolDelete script is for alpha=0 only "
            "(no post-hoc task arithmetic). Set --alpha 0."
        )

    root = Path(args.root).expanduser()

    forget_file = (
        Path(args.forget_file).expanduser()
        if args.forget_file
        else root / "final_forget_train.json"
    )
    retain_file = (
        Path(args.retain_file).expanduser()
        if args.retain_file
        else root / "final_retain_train_4040.json"
    )
    toolfree_cache_path = (
        Path(args.toolfree_cache).expanduser()
        if args.toolfree_cache
        else root / "tooldelete_toolfree_deepseek13b.json"
    )
    dpo_ref_cache_path = (
        Path(args.dpo_ref_cache).expanduser()
        if args.dpo_ref_cache
        else root / "xlam2_3b_tooldelete_dpo_reference_logps.pt"
    )
    output_dir = Path(args.output_dir).expanduser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
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

    if any(
        not isinstance(x, str) or not x.strip()
        for x in toolfree_cache["forget"]
    ):
        raise RuntimeError(
            "Tool-free forget cache contains missing/empty responses"
        )

    if any(
        not isinstance(x, str) or not x.strip()
        for x in toolfree_cache["retain"]
    ):
        raise RuntimeError(
            "Tool-free retain cache contains missing/empty responses"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id")

    # ------------------------------------------------------------------
    # Native xLAM-2 prompt/target encoding
    # ------------------------------------------------------------------
    def prompt_ids_for_row(row):
        messages = [
            {"role": "user", "content": get_query(row)}
        ]

        chat_text = tokenizer.apply_chat_template(
            messages,
            tools=get_tools(row),
            add_generation_prompt=True,
            tokenize=False,
        )

        return tokenizer(
            chat_text,
            add_special_tokens=False,
        )["input_ids"]

    def encode_target(row, target_text, idx):
        prompt_ids = prompt_ids_for_row(row)

        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
        )["input_ids"] + [tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > args.max_length:
            raise RuntimeError(
                f"Example {idx} length {len(input_ids)} "
                f"> max_length={args.max_length}. "
                "Do not silently truncate ToolDelete targets."
            )

        return {
            "idx": idx,
            "input_ids": input_ids,
            "labels": labels,
        }

    class SFTDataset(Dataset):
        def __init__(self, rows, targets):
            self.rows = rows
            self.targets = targets

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return encode_target(
                self.rows[idx],
                self.targets[idx],
                idx,
            )

    class PairDataset(Dataset):
        def __init__(
            self,
            rows,
            chosen_targets,
            rejected_targets,
        ):
            self.rows = rows
            self.chosen_targets = chosen_targets
            self.rejected_targets = rejected_targets

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return {
                "idx": idx,
                "chosen": encode_target(
                    self.rows[idx],
                    self.chosen_targets[idx],
                    idx,
                ),
                "rejected": encode_target(
                    self.rows[idx],
                    self.rejected_targets[idx],
                    idx,
                ),
            }

    def pad_examples(examples):
        max_len = max(
            len(x["input_ids"])
            for x in examples
        )

        input_ids = []
        labels = []
        masks = []
        indices = []

        for x in examples:
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
            masks.append(
                [1] * n + [0] * pad
            )
            indices.append(x["idx"])

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
                masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }

    def sft_collate(batch):
        return pad_examples(batch)

    def pair_collate(batch):
        return {
            "idx": torch.tensor(
                [x["idx"] for x in batch],
                dtype=torch.long,
            ),
            "chosen": pad_examples(
                [x["chosen"] for x in batch]
            ),
            "rejected": pad_examples(
                [x["rejected"] for x in batch]
            ),
        }

    # ------------------------------------------------------------------
    # Memory-efficient forward helpers
    # ------------------------------------------------------------------
    def target_suffix_forward(model, batch):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        # ref-batch-size may be >1; fall back to normal forward there.
        if input_ids.shape[0] != 1:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return out, labels

        supervised = torch.nonzero(
            labels[0] != -100,
            as_tuple=False,
        ).flatten()

        if supervised.numel() == 0:
            raise RuntimeError(
                "Batch has no supervised target tokens"
            )

        first_target = int(
            supervised[0].item()
        )
        seq_len = labels.shape[1]

        # One extra logit position because of causal shifting.
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
            shift_logits
            .view(-1, shift_logits.size(-1))
            .float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def sequence_logp(logits, labels):
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        mask = shift_labels != -100

        safe = shift_labels.clone()
        safe[~mask] = 0

        token_logps = (
            F.log_softmax(
                shift_logits.float(),
                dim=-1,
            )
            .gather(
                -1,
                safe.unsqueeze(-1),
            )
            .squeeze(-1)
        )

        return (
            token_logps * mask
        ).sum(dim=-1)

    def forward_logp(model, batch):
        out, score_labels = target_suffix_forward(
            model,
            batch,
        )
        lp = sequence_logp(
            out.logits,
            score_labels,
        )
        return lp, out

    # ------------------------------------------------------------------
    # Targets / datasets
    # ------------------------------------------------------------------
    original_forget = [
        build_original_target(x)
        for x in forget_rows
    ]
    original_retain = [
        build_original_target(x)
        for x in retain_rows
    ]

    toolfree_forget = [
        x.strip()
        for x in toolfree_cache["forget"]
    ]
    toolfree_retain = [
        x.strip()
        for x in toolfree_cache["retain"]
    ]

    if args.method == "sft":
        # Forget: train toward teacher tool-free response.
        # Retain: preserve original native tool call.
        forget_dataset = SFTDataset(
            forget_rows,
            toolfree_forget,
        )
        retain_dataset = SFTDataset(
            retain_rows,
            original_retain,
        )
        collate = sft_collate

    else:
        # Forget: prefer tool-free response over original tool call.
        forget_dataset = PairDataset(
            forget_rows,
            toolfree_forget,
            original_forget,
        )

        # Retain: prefer original tool call over tool-free response.
        retain_dataset = PairDataset(
            retain_rows,
            original_retain,
            toolfree_retain,
        )
        collate = pair_collate

    # ------------------------------------------------------------------
    # Same API-balanced forget sampling as the 1B setup.
    # ------------------------------------------------------------------
    api_counts = Counter()

    for row in forget_rows:
        for name in set(target_names(row)):
            api_counts[name] += 1

    weights = []

    for row in forget_rows:
        names = set(target_names(row))

        if not names:
            raise RuntimeError(
                "Forget row has no target API"
            )

        weights.append(
            float(
                np.mean(
                    [
                        1.0 / api_counts[n]
                        for n in names
                    ]
                )
            )
        )

    forget_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(forget_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(
            args.seed
        ),
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
        generator=torch.Generator().manual_seed(
            args.seed + 1
        ),
        collate_fn=collate,
    )

    assert (
        len(retain_loader)
        == args.retain_per_forget
        * len(forget_loader)
    )

    # ------------------------------------------------------------------
    # DPO reference cache
    # ------------------------------------------------------------------
    reference_cache = None

    if args.method == "dpo":
        if dpo_ref_cache_path.exists():
            print(
                "Loading DPO reference cache:",
                dpo_ref_cache_path,
            )

            payload = torch.load(
                dpo_ref_cache_path,
                map_location="cpu",
                weights_only=False,
            )

            if not isinstance(payload, dict):
                raise RuntimeError(
                    "DPO reference cache is from an old format. "
                    "Use a new 3B native cache path."
                )

            if payload.get("model") != args.model_name:
                raise RuntimeError(
                    "DPO reference-cache model mismatch: "
                    f"{payload.get('model')} != {args.model_name}"
                )

            if payload.get("format_version") != NATIVE_FORMAT_VERSION:
                raise RuntimeError(
                    "DPO reference cache is not from the native "
                    "xLAM-2 prompt/target format."
                )

            if payload.get("max_length") != args.max_length:
                raise RuntimeError(
                    "DPO reference-cache max_length mismatch: "
                    f"{payload.get('max_length')} != {args.max_length}"
                )

            reference_cache = payload

        else:
            print("=" * 90)
            print(
                "PRECOMPUTING TOOLDELETE-DPO "
                "3B NATIVE REFERENCE LOG-PROBABILITIES"
            )
            print("=" * 90)

            ref_model = (
                AutoModelForCausalLM
                .from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                .to(args.device)
            )

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

                chosen = torch.empty(
                    len(dataset),
                    dtype=torch.float32,
                )
                rejected = torch.empty(
                    len(dataset),
                    dtype=torch.float32,
                )

                with torch.inference_mode():
                    for step, batch in enumerate(loader):
                        ids = batch["idx"]

                        c_lp, c_out = forward_logp(
                            ref_model,
                            batch["chosen"],
                        )
                        r_lp, r_out = forward_logp(
                            ref_model,
                            batch["rejected"],
                        )

                        chosen[ids] = (
                            c_lp.detach().cpu()
                        )
                        rejected[ids] = (
                            r_lp.detach().cpu()
                        )

                        del c_out, r_out

                        if (
                            step % 100 == 0
                            or step + 1 == len(loader)
                        ):
                            print(
                                f"{label} reference "
                                f"{step+1}/{len(loader)}",
                                flush=True,
                            )

                return chosen, rejected

            f_ch, f_rj = cache_pairs(
                forget_dataset,
                "forget",
            )
            r_ch, r_rj = cache_pairs(
                retain_dataset,
                "retain",
            )

            reference_cache = {
                "model": args.model_name,
                "format_version": (
                    NATIVE_FORMAT_VERSION
                ),
                "max_length": args.max_length,
                "prompt_format": (
                    "xLAM-2 native chat template "
                    "with tools=..."
                ),
                "original_target_format": (
                    "native JSON list of "
                    "{name, arguments} calls"
                ),
                "forget_chosen": f_ch,
                "forget_rejected": f_rj,
                "retain_chosen": r_ch,
                "retain_rejected": r_rj,
            }

            dpo_ref_cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                reference_cache,
                dpo_ref_cache_path,
            )

            print(
                "Saved DPO reference cache:",
                dpo_ref_cache_path,
            )

            del ref_model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for key, n in [
            ("forget_chosen", len(forget_rows)),
            ("forget_rejected", len(forget_rows)),
            ("retain_chosen", len(retain_rows)),
            ("retain_rejected", len(retain_rows)),
        ]:
            if (
                key not in reference_cache
                or len(reference_cache[key]) != n
            ):
                raise RuntimeError(
                    f"Invalid DPO reference cache entry: {key}"
                )

    # ------------------------------------------------------------------
    # Model / optimizer
    # ------------------------------------------------------------------
    print("=" * 90)
    print(
        f"TOOLDELETE-{args.method.upper()} "
        "xLAM-2 3B NATIVE CONFIG"
    )
    print("=" * 90)
    print("Model:", args.model_name)
    print("Format:", NATIVE_FORMAT_VERSION)
    print("Tool-free cache:", toolfree_cache_path)
    print("Output dir:", output_dir)
    print("Epochs:", args.epochs)
    print("Alpha:", args.alpha, "(no task arithmetic)")
    print("Forget coefficient:", args.forget_coef)
    print("Retain coefficient:", args.retain_coef)

    if args.method == "dpo":
        print("DPO beta:", args.dpo_beta)
        print(
            "DPO reference cache:",
            dpo_ref_cache_path,
        )

    print("LR:", args.learning_rate)
    print("Weight decay:", args.weight_decay)
    print("Grad accum:", args.grad_accum)
    print(
        "Retain per forget:",
        args.retain_per_forget,
    )
    print("Max length:", args.max_length)
    print(
        "Warmup ratio:",
        args.warmup_ratio,
    )
    print("Seed:", args.seed)

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        .to(args.device)
    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    opt_steps_per_epoch = math.ceil(
        len(forget_loader)
        / args.grad_accum
    )

    total_opt_steps = (
        args.epochs
        * opt_steps_per_epoch
    )

    warmup_steps = (
        max(
            1,
            int(
                args.warmup_ratio
                * total_opt_steps
            ),
        )
        if total_opt_steps > 1
        else 0
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    def dpo_loss_for_batch(
        pair_batch,
        ref_chosen,
        ref_rejected,
    ):
        c_lp, c_out = forward_logp(
            model,
            pair_batch["chosen"],
        )
        r_lp, r_out = forward_logp(
            model,
            pair_batch["rejected"],
        )

        ref_c = ref_chosen.to(
            args.device,
            dtype=torch.float32,
        )
        ref_r = ref_rejected.to(
            args.device,
            dtype=torch.float32,
        )

        pi_logratio = c_lp - r_lp
        ref_logratio = ref_c - ref_r

        logits = (
            args.dpo_beta
            * (
                pi_logratio
                - ref_logratio
            )
        )

        loss = (
            -F.logsigmoid(logits)
            .mean()
        )

        reward_margin = (
            pi_logratio
            - ref_logratio
        ).mean()

        return (
            loss,
            reward_margin,
            c_out,
            r_out,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    optimizer.zero_grad(
        set_to_none=True
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        ep = epoch + 1

        print("\n" + "=" * 90)
        print(
            f"TOOLDELETE-{args.method.upper()} "
            f"EPOCH {ep}/{args.epochs}"
        )
        print("=" * 90)

        retain_iter = iter(retain_loader)

        forget_losses = []
        retain_losses = []
        margins = []
        grad_norms = []

        for step, fbatch in enumerate(forget_loader):
            f_idx = fbatch["idx"]

            # ----------------------------------------------------------
            # Forget objective
            # ----------------------------------------------------------
            if args.method == "sft":
                f_out, f_labels = (
                    target_suffix_forward(
                        model,
                        fbatch,
                    )
                )

                f_loss = token_ce_loss(
                    f_out.logits,
                    f_labels,
                )

                (
                    args.forget_coef
                    * f_loss
                    / args.grad_accum
                ).backward()

                forget_losses.append(
                    float(f_loss.detach())
                )

                del f_out

            else:
                (
                    f_loss,
                    margin,
                    c_out,
                    r_out,
                ) = dpo_loss_for_batch(
                    fbatch,
                    reference_cache[
                        "forget_chosen"
                    ][f_idx],
                    reference_cache[
                        "forget_rejected"
                    ][f_idx],
                )

                (
                    args.forget_coef
                    * f_loss
                    / args.grad_accum
                ).backward()

                forget_losses.append(
                    float(f_loss.detach())
                )
                margins.append(
                    float(margin.detach())
                )

                del c_out, r_out

            # ----------------------------------------------------------
            # Retain objective
            # ----------------------------------------------------------
            step_retain = []

            for _ in range(
                args.retain_per_forget
            ):
                rbatch = next(retain_iter)
                r_idx = rbatch["idx"]

                if args.method == "sft":
                    r_out, r_labels = (
                        target_suffix_forward(
                            model,
                            rbatch,
                        )
                    )

                    r_loss = token_ce_loss(
                        r_out.logits,
                        r_labels,
                    )

                    (
                        args.retain_coef
                        * r_loss
                        / args.retain_per_forget
                        / args.grad_accum
                    ).backward()

                    del r_out

                else:
                    (
                        r_loss,
                        margin,
                        c_out,
                        r_out,
                    ) = dpo_loss_for_batch(
                        rbatch,
                        reference_cache[
                            "retain_chosen"
                        ][r_idx],
                        reference_cache[
                            "retain_rejected"
                        ][r_idx],
                    )

                    (
                        args.retain_coef
                        * r_loss
                        / args.retain_per_forget
                        / args.grad_accum
                    ).backward()

                    margins.append(
                        float(
                            margin.detach()
                        )
                    )

                    del c_out, r_out

                step_retain.append(
                    float(
                        r_loss.detach()
                    )
                )

            retain_losses.append(
                float(
                    np.mean(
                        step_retain
                    )
                )
            )

            boundary = (
                (step + 1)
                % args.grad_accum
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
                        args.max_grad_norm,
                    )
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                grad_norms.append(
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
                msg = (
                    f"epoch={ep} "
                    f"step={step+1:4d}/"
                    f"{len(forget_loader)} "
                    f"forget="
                    f"{np.mean(forget_losses[-50:]):.4f} "
                    f"retain="
                    f"{np.mean(retain_losses[-50:]):.4f} "
                )

                if (
                    args.method == "dpo"
                    and margins
                ):
                    msg += (
                        "margin="
                        f"{np.mean(margins[-100:]):+.4f} "
                    )

                msg += (
                    "lr="
                    f"{scheduler.get_last_lr()[0]:.2e}"
                )

                print(
                    msg,
                    flush=True,
                )

        record = {
            "epoch": ep,
            "forget_loss": float(
                np.mean(
                    forget_losses
                )
            ),
            "retain_loss": float(
                np.mean(
                    retain_losses
                )
            ),
            "dpo_reward_margin": (
                float(
                    np.mean(margins)
                )
                if margins
                else None
            ),
            "mean_grad_norm": (
                float(
                    np.mean(
                        grad_norms
                    )
                )
                if grad_norms
                else None
            ),
        }

        history.append(record)

        print("Epoch summary:")
        print(
            json.dumps(
                record,
                indent=2,
            )
        )

        # Save every epoch so DPO epoch 1/2/3 can all be evaluated.
        epoch_dir = (
            output_dir
            / f"epoch_{ep}"
        )

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
            "base_model": args.model_name,
            "method": (
                "TOOLDELETE-"
                + args.method.upper()
            ),
            "architecture": (
                "xLAM-2-3B"
            ),
            "format_version": (
                NATIVE_FORMAT_VERSION
            ),
            "prompt_format": (
                "xLAM-2 native chat "
                "template with tools=..."
            ),
            "original_target_format": (
                "native JSON list of "
                "{name, arguments} calls"
            ),
            "tool_free_model": (
                toolfree_cache
                .get("metadata", {})
                .get("tool_free_model")
            ),
            "toolfree_cache": str(
                toolfree_cache_path
            ),
            "dpo_ref_cache": (
                str(
                    dpo_ref_cache_path
                )
                if args.method == "dpo"
                else None
            ),
            "alpha": args.alpha,
            "task_arithmetic_applied": False,
            "forget_coef": (
                args.forget_coef
            ),
            "retain_coef": (
                args.retain_coef
            ),
            "dpo_beta": (
                args.dpo_beta
                if args.method == "dpo"
                else None
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "epochs_requested": (
                args.epochs
            ),
            "checkpoint_epoch": ep,
            "max_length": (
                args.max_length
            ),
            "grad_accum": (
                args.grad_accum
            ),
            "retain_per_forget": (
                args.retain_per_forget
            ),
            "max_grad_norm": (
                args.max_grad_norm
            ),
            "warmup_ratio": (
                args.warmup_ratio
            ),
            "forget_n": len(
                forget_rows
            ),
            "retain_n": len(
                retain_rows
            ),
            "seed": args.seed,
            "history": history,
            "cli_args": vars(args),
        }

        with open(
            epoch_dir
            / "unlearning_config.json",
            "w",
        ) as f:
            json.dump(
                config,
                f,
                indent=2,
            )

        print(
            "\nSaved checkpoint:",
            epoch_dir,
        )

    print("\n" + "=" * 90)
    print(
        f"TOOLDELETE-{args.method.upper()} "
        "3B TRAINING COMPLETE"
    )
    print("=" * 90)
    print(
        "Final checkpoint:",
        output_dir
        / f"epoch_{args.epochs}",
    )
    print(
        "alpha=0 -> no task arithmetic applied."
    )


if __name__ == "__main__":
    main()
