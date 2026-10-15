"""Independent full-parameter baseline entrypoint (python -m ...baselines.train)."""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, set_seed

from .artifacts import checkpoint_signature, new_output_dir, write_json
from .data import REFERENCE_METHODS, add_data_args, load_tokenizer, make_dataset
from .prepare_reference import attach_reference
from .trainer import BaselineTrainer


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--reference_cache")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--forget_weight", type=float, default=1.0)
    p.add_argument("--retain_weight", type=float, default=1.0)
    p.add_argument("--optim", choices=("adafactor", "adamw_torch"), default="adafactor")
    p.add_argument("--deepspeed")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_strategy", choices=("steps", "epoch", "no"), default="steps")
    p.add_argument("--save_total_limit", type=int, default=2,
                   help="Checkpoint retention limit; 0 keeps every checkpoint")
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--dry_run", action="store_true", help="Validate data/cache on CPU; no model load")
    p.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    args = p.parse_args()
    if (args.beta <= 0 or args.lr <= 0 or args.epochs <= 0 or args.grad_accum < 1
            or args.per_device_batch_size < 1 or args.save_steps < 1 or args.logging_steps < 1
            or args.forget_weight <= 0 or args.retain_weight <= 0
            or (args.max_steps != -1 and args.max_steps < 1)):
        p.error("Rates, weights, beta and counts must be positive (max_steps may be -1)")
    if args.save_total_limit < 0:
        p.error("save_total_limit must be >= 0")
    if args.cpu and (args.deepspeed or args.dtype != "float32"):
        p.error("CPU smoke tests require --dtype float32 and no DeepSpeed")
    if args.method in REFERENCE_METHODS and not args.reference_cache:
        p.error("Run prepare_reference first and supply --reference_cache")
    return args


def main():
    args = get_args()
    if args.cpu:
        # Accelerate 1.14 otherwise calls torch.cpu.set_device, unavailable in
        # this environment's torch 2.1 when starting multi-process CPU tests.
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.tokenizer_dir or args.model_dir)
    dataset = make_dataset(args, tokenizer)
    if args.method in REFERENCE_METHODS:
        attach_reference(dataset, args.reference_cache, args.model_dir)
    manifest = {"schema": "baseline-run-v1", "args": vars(args),
                "method": args.method, "dataset_sha256": dataset.fingerprint(),
                "source_hashes": dataset.source_hashes, "dropped_rows": dataset.dropped,
                "stream_sizes": {k: len(v) for k, v in dataset.streams.items()},
                "model_signature": checkpoint_signature(args.model_dir),
                "task_arithmetic_applied": False}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    output = Path(args.out_dir).resolve()
    if args.resume_from_checkpoint:
        with (output / "baseline_manifest.json").open() as handle:
            previous = json.load(handle)
        comparable = ("method", "dataset_sha256", "model_signature", "stream_sizes")
        if any(previous[k] != manifest[k] for k in comparable):
            raise ValueError("Resume inputs do not match original run")
        mutable = {"resume_from_checkpoint", "local_rank", "dry_run", "max_steps", "epochs"}
        legacy_defaults = {"save_strategy": "steps", "save_total_limit": 2}
        if any(previous["args"].get(k, legacy_defaults.get(k)) != value
               for k, value in vars(args).items() if k not in mutable):
            raise ValueError("Resume hyperparameters do not match original run")
        checkpoint = Path(args.resume_from_checkpoint).resolve()
        if checkpoint.parent != output or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Resume must name a Trainer checkpoint inside out_dir")
    elif rank == 0:
        new_output_dir(output, (args.model_dir, args.tokenizer_dir))
    # TrainingArguments must exist BEFORE loading the model: this enables
    # ZeRO-3 partitioned initialization without materializing 13B on each GPU.
    training_args = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, max_steps=args.max_steps, optim=args.optim,
        lr_scheduler_type="constant", weight_decay=0.0,
        bf16=args.dtype == "bfloat16", no_cuda=args.cpu,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        logging_steps=args.logging_steps, save_strategy=args.save_strategy, save_steps=args.save_steps,
        save_total_limit=args.save_total_limit or None, remove_unused_columns=False, label_names=[],
        dataloader_num_workers=0, deepspeed=args.deepspeed, seed=args.seed,
        report_to=[], local_rank=local_rank, ddp_find_unused_parameters=False)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        write_json(output / "baseline_manifest.json", manifest)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                 torch_dtype=getattr(torch, args.dtype))
    model.config.use_cache = False
    # Reference scores were computed in eval mode; disable dropout for DPO/NPO
    # so a frozen-reference ratio of zero is reproducible at initialization.
    if args.method in REFERENCE_METHODS:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    dataset.generator.manual_seed(args.seed + rank)
    trainer = BaselineTrainer(
        model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset,
        data_collator=dataset.collate, method=args.method, beta=args.beta,
        forget_weight=args.forget_weight, retain_weight=args.retain_weight)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output)
        manifest.update(status="complete", metrics=result.metrics)
        write_json(output / "baseline_manifest.json", manifest)
        print("[baseline] saved full checkpoint to %s" % output)


if __name__ == "__main__":
    main()
