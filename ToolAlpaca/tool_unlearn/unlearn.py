"""Entry point for tool unlearning on ToolAlpaca.

    python tool_unlearn/unlearn.py --algo npo_gdr \
        --model_dir $ACTOR_MODEL \
        --forget_file tool_unlearn/data/forget.json \
        --retain_file tool_unlearn/data/retain.json \
        --out_dir tool_unlearn/ckpt/npo_gdr

Structure follows MUSE's `baselines/unlearn.py`; the task-vector and
who's-harry-potter branches are dropped because this study only needs the
gradient-based methods (GA / NPO and their retain-regularised variants).
"""
import argparse
import os
import sys

import torch
import transformers
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_unlearn.dataset import ForgetRetainDataset, CTX_LEN
from tool_unlearn.trainer import ToolUnlearner, VALID_LOSSES, _seq_logp


def load_causal_lm(path, device, local_rank, world_size, tag):
    """Load a bf16 causal LM, moving it straight to `device` so it never lingers
    on the host.

    A 7B copy is about 14 GB on CPU while loading.
    `from_pretrained(..., low_cpu_mem_usage=True)`
    skips the fp32 materialisation, and `.to(device)` frees the CPU copy at once.
    To stop N ranks from all holding their copy at the same time (8 x 14 GB > 64 GB),
    ranks take turns loading one model each, gated by barriers.  The already-loaded
    ranks hold nothing on CPU, so the host peak is a single model's ~14 GB.
    """
    model = None
    for r in range(world_size):
        if local_rank == r:
            print("[unlearn] rank %d loading %s from %s" % (r, tag, path), flush=True)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            ).to(device)
        if dist.is_initialized():
            dist.barrier()
    return model


def load_tokenizer(path):
    tok = transformers.AutoTokenizer.from_pretrained(
        path, use_fast=False, model_max_length=CTX_LEN,
    )
    # train.py line 203: ToolAlpaca pads with <unk>, not </s>.  Using </s> here
    # would make every pad position collide with the real end-of-sequence token
    # that train.preprocess appends, and the label mask is derived from
    # `input_ids.ne(pad_token_id)` -- so the wrong pad token silently truncates
    # the supervised span of every example.
    tok.pad_token = tok.unk_token
    return tok


def compute_ref_seq_logp(ref_model, forget_dataset, device, batch_size=8):
    """Frozen-reference sequence log-probs over every forget example.

    Exactly what `_seq_logp` computes in the NPO loss -- the sum of log p_ref
    over the supervised positions -- evaluated once and cached, so the 7B
    reference never has to share a 40GB card with the trainable policy.  All
    forget examples are pre-padded to CTX_LEN by train.preprocess, so a plain
    stack collates without extra padding.
    """
    ref_model.eval()
    n = len(forget_dataset)
    logps = torch.zeros(n, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            items = [forget_dataset[i] for i in range(start, min(start + batch_size, n))]
            input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
            labels = torch.stack([it["labels"] for it in items]).to(device)
            attention_mask = torch.stack([it["attention_mask"] for it in items]).to(device)
            out = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            logps[start:start + len(items)] = _seq_logp(out.logits, labels).cpu()
    return logps


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # The deepspeed launcher sets RANK/WORLD_SIZE/MASTER_*; the Trainer's own
    # deepspeed init reuses this group.  We need it now so the model loads can
    # be serialised across ranks (host RAM is capped at 64 GB in this job).
    if not dist.is_initialized() and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
    local_rank = args.local_rank if args.local_rank is not None else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = load_tokenizer(args.tokenizer_dir or args.model_dir)

    needs_retain = "gdr" in args.algo or "klr" in args.algo
    if needs_retain and not args.retain_file:
        raise SystemExit("--retain_file is required for algo %r" % args.algo)

    dataset = ForgetRetainDataset(
        args.forget_file, tokenizer,
        retain_file=args.retain_file if needs_retain else None,
        seed=args.seed,
    )

    ref_model = None
    if "npo" in args.algo:
        # NPO needs log pi_ref(y|x_f), which does not depend on theta.  Compute
        # it once and cache it so the 7B reference never has to sit on the same
        # 40GB card as the trainable policy (ZeRO-2/3 shard the optimizer and
        # gradients, but both full model copies would still exceed the card).
        cache_path = os.path.join(args.out_dir, "ref_seq_logp.pt")
        if local_rank == 0 and not os.path.exists(cache_path):
            print("[unlearn] rank 0 computing frozen-reference forget log-probs "
                  "from %s" % (args.ref_model_dir or args.model_dir), flush=True)
            ref = transformers.AutoModelForCausalLM.from_pretrained(
                args.ref_model_dir or args.model_dir, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(device)
            ref.config.use_cache = False
            logps = compute_ref_seq_logp(ref, dataset.forget, device)
            torch.save(logps, cache_path)
            print("[unlearn] cached ref forget log-probs -> %s" % cache_path, flush=True)
            del ref, logps
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        dataset.ref_seq_logp = torch.load(cache_path, map_location="cpu")
        print("[unlearn] ref forget log-probs: %d values" % len(dataset.ref_seq_logp),
              flush=True)
    elif "kl" in args.algo:
        # The KL retain term needs the reference forward every step, so keep the
        # frozen model around (14GB on top of training).
        ref_model = load_causal_lm(
            args.ref_model_dir or args.model_dir, device, local_rank, world_size, "reference",
        )
        ref_model.config.use_cache = False
        ref_model.eval()

    print("[unlearn] loading policy model from %s" % args.model_dir)
    model = load_causal_lm(args.model_dir, device, local_rank, world_size, "policy")
    model.config.use_cache = False          # required with gradient checkpointing

    training_args = transformers.TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        optim=args.optim,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        # The collate_fn returns a (forget, retain) tuple rather than a dict, so
        # nothing in Trainer may try to reshape it.
        remove_unused_columns=False,
        label_names=[],
        # One process owns the retain sampler's RNG; see dataset.py.
        dataloader_num_workers=0,
        deepspeed=args.deepspeed,
        seed=args.seed,
        report_to=[],
    )

    trainer = ToolUnlearner(
        model=model,
        ref_model=ref_model,
        ref_seq_logp=getattr(dataset, "ref_seq_logp", None),
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataset.get_collate_fn(),
        loss_type=args.algo,
        beta=args.beta,
    )

    print("[unlearn] algo=%s  forget=%d  retain=%s  epochs=%s  lr=%s"
          % (args.algo, len(dataset),
             len(dataset.retain) if dataset.retain is not None else "unused",
             args.epochs, args.lr))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("[unlearn] wrote unlearned model to %s" % args.out_dir)


def get_args():
    p = argparse.ArgumentParser(description="Tool unlearning (GA / NPO family)")
    p.add_argument("--algo", required=True, choices=["ga", "npo"] + list(VALID_LOSSES),
                   help="the FORGET objective. ga = gradient ascent, "
                        "npo = negative preference optimisation")
    p.add_argument("--retain_loss", default=None, choices=["none", "ce", "kl"],
                   help="how the RETAIN set is regularised.  "
                        "ce  = ordinary cross-entropy descent on retain, i.e. the "
                        "same likelihood objective as forget but pushed the "
                        "opposite way (this is MUSE's 'gdr' / gradient difference); "
                        "kl  = KL to the frozen reference on retain (MUSE's 'klr'); "
                        "none = no retain term.  "
                        "Defaults to whatever a compound --algo like ga_gdr implies, "
                        "or 'none' for a bare ga/npo.")
    p.add_argument("--model_dir", default="TangQiaoYu/ToolAlpaca-7B",
                   help="the tool-augmented model to unlearn from")
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--ref_model_dir", default=None,
                   help="frozen reference for NPO/KL; defaults to --model_dir")
    p.add_argument("--forget_file", default="tool_unlearn/data/forget_train.json")
    p.add_argument("--retain_file", default="tool_unlearn/data/retain_train.json")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1, help="NPO temperature")
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--optim", default="adamw_torch",
                   help="adamw_torch (needs ZeRO-3 offload for a 7B on 24GB "
                        "cards) or adafactor (far smaller optimiser state)")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--deepspeed", default=None,
                   help="path to a DeepSpeed config, e.g. "
                        "tool_unlearn/configs/ds_zero3_offload.json")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_strategy", default="epoch", choices=["no", "epoch", "steps"])
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--resume_from_checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # The `deepspeed` launcher injects --local_rank=N; without this the
    # Trainer's own distributed init can never see it.
    p.add_argument("--local_rank", type=int, default=0)
    args = p.parse_args()
    args.algo = resolve_algo(args.algo, args.retain_loss)
    return args


RETAIN_SUFFIX = {"none": "", "ce": "_gdr", "kl": "_klr"}


def resolve_algo(algo, retain_loss):
    """Combine the forget objective and the retain term into MUSE's loss name.

    Two spellings are accepted so neither audience is surprised: MUSE's compound
    strings (`ga_gdr`, `npo_klr`) work as-is, and the explicit pair
    `--algo npo --retain_loss ce` means the same thing but says out loud which
    retain term is in play.
    """
    base, _, suffix = algo.partition("_")
    if suffix:                       # compound form, e.g. ga_gdr
        implied = {"gdr": "ce", "klr": "kl"}[suffix]
        if retain_loss is not None and retain_loss != implied:
            raise SystemExit(
                "--algo %s already means --retain_loss %s; you passed %s"
                % (algo, implied, retain_loss))
        return algo
    if retain_loss in (None, "none"):
        return base
    return base + RETAIN_SUFFIX[retain_loss]


if __name__ == "__main__":
    main()
