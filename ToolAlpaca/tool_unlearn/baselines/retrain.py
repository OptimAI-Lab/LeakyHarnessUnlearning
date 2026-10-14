"""Full-parameter retain-only SFT, with complete checkpoints every two epochs."""
import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

from tool_unlearn.baselines.artifacts import checkpoint_signature, write_json
from tool_unlearn.baselines.data import BaselineDataset, load_tokenizer, stack_examples


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EpochCheckpoints(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            write_json(Path(args.output_dir) / "progress.json", {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "epoch": state.epoch, "global_step": state.global_step,
                "max_steps": state.max_steps, "metrics": logs,
            })
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if abs(state.epoch - epoch) > 1e-6:
            raise RuntimeError(f"Epoch ended at fractional epoch {state.epoch}")
        control.should_save = epoch > 0 and epoch % 2 == 0
        return control

    def on_save(self, args, state, control, **kwargs):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if state.is_world_process_zero:
            root = Path(args.output_dir)
            checkpoint = root / f"checkpoint-{state.global_step}"
            epoch = round(state.epoch)
            write_json(checkpoint / "complete.json", {
                "epoch": epoch, "global_step": state.global_step,
                "full_model": True, "optimizer_and_rng_saved": True,
            })
            alias = root / f"epoch-{epoch:02d}"
            if not alias.exists():
                alias.symlink_to(checkpoint.name, target_is_directory=True)
            print(f"[checkpoint] epoch={epoch} path={checkpoint}", flush=True)
        return control


class FiniteLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite retain training loss")
        return (loss, outputs) if return_outputs else loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--retain_file", required=True)
    p.add_argument("--split_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--deepspeed", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint")
    args = p.parse_args()
    if args.epochs < 2 or args.epochs % 2:
        p.error("epochs must be a positive multiple of two")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 8:
        raise ValueError("This retraining configuration requires eight GPUs")
    output = Path(args.out_dir).resolve()
    from tool_unlearn.baselines.checkpoint_io import install
    install()
    set_seed(args.seed)
    # Creating TrainingArguments first enables ZeRO-3 partitioned model loading.
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, optim="adamw_torch", weight_decay=0.0,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, tf32=True, gradient_checkpointing=True,
        save_strategy="epoch", save_total_limit=None,
        logging_strategy="steps", logging_steps=1, logging_nan_inf_filter=False,
        evaluation_strategy="no", report_to=[], disable_tqdm=True,
        dataloader_num_workers=0, seed=args.seed, data_seed=args.seed,
        deepspeed=args.deepspeed, ddp_timeout=3600,
    )
    tokenizer = load_tokenizer(args.model_dir)
    rows = json.loads(Path(args.retain_file).read_text())
    split = json.loads(Path(args.split_file).read_text())
    tools = {r["tool"] for r in rows}
    if tools & set(split["forget_tools"]):
        raise ValueError("Retain input contains frozen forget tools")
    if any(split["per_tool"][t]["group"] != "retain" for t in tools):
        raise ValueError("Retain input disagrees with the frozen partition")
    data = BaselineDataset("retain_ft", tokenizer, None, args.retain_file)
    # Transformers 4.29 drops the final partial accumulation window. Reject it
    # so every saved epoch is a complete pass, including the tenth epoch.
    batches_per_epoch = math.ceil(math.ceil(len(data) / world) / args.batch_size)
    if batches_per_epoch % args.grad_accum:
        raise ValueError(f"{batches_per_epoch} batches/epoch is not divisible by "
                         f"grad_accum={args.grad_accum}; choose a divisor")
    manifest = {
        "schema": "retain-retrain-v1", "args": vars(args),
        "base_model_id": "lmsys/vicuna-7b-v1.1",
        "model_signature": checkpoint_signature(args.model_dir),
        "retain_sha256": sha256(args.retain_file), "split_sha256": sha256(args.split_file),
        "dataset_sha256": data.fingerprint(), "raw_examples": len(rows),
        "training_examples": len(data), "tools": len(tools), "dropped": data.dropped,
        "forget_tool_overlap": 0, "world_size": world,
        "effective_batch_size": world * args.batch_size * args.grad_accum,
        "batches_per_epoch": batches_per_epoch,
        "steps_per_epoch": batches_per_epoch // args.grad_accum,
        "checkpoint_epochs": list(range(2, args.epochs + 1, 2)),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "optimizer": "AdamW", "scheduler": "cosine", "warmup_ratio": 0.03,
        "status": "running",
        "checkpoint_io": "serial optimizer writes, fsync, cache release, serial restore",
        "optimizer_storage": os.environ.get("RETRAIN_OPTIMIZER_ROOT"),
    }
    if args.resume_from_checkpoint:
        previous = json.loads((output / "manifest.json").read_text())
        for key in ("model_signature", "retain_sha256", "split_sha256", "dataset_sha256",
                    "world_size", "effective_batch_size", "steps_per_epoch"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume input mismatch: {key}")
        for key, value in vars(args).items():
            if key != "resume_from_checkpoint" and previous["args"][key] != value:
                raise ValueError(f"Resume argument mismatch: {key}")
        cp = Path(args.resume_from_checkpoint).resolve()
        if cp.parent != output or not (cp / "complete.json").is_file():
            raise ValueError("Resume must use a complete checkpoint in this run")
    elif rank == 0 and output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite {output}")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "manifest.json", manifest)
        print("[data] " + json.dumps(manifest), flush=True)
    torch.distributed.barrier()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    # Vicuna v1.1 config contains legacy token IDs; use its actual tokenizer IDs.
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        setattr(model.config, key, getattr(tokenizer, key))
        setattr(model.generation_config, key, getattr(tokenizer, key))
    total = sum(getattr(p, "ds_numel", p.numel()) for p in model.parameters())
    trainable = sum(getattr(p, "ds_numel", p.numel()) for p in model.parameters() if p.requires_grad)
    if trainable != total:
        raise ValueError("Full fine-tuning requires every parameter to be trainable")
    if rank == 0:
        manifest.update(total_parameters=total, trainable_parameters=trainable)
        write_json(output / "manifest.json", manifest)
        print(f"[model] full fine-tuning: {trainable:,}/{total:,} parameters", flush=True)
    trainer = FiniteLossTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=data.streams["retain"], data_collator=stack_examples,
        callbacks=[EpochCheckpoints()],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if abs(trainer.state.epoch - args.epochs) > 1e-6:
        raise RuntimeError(f"Training stopped at epoch {trainer.state.epoch}")
    torch.distributed.barrier()
    if rank == 0:
        for epoch in manifest["checkpoint_epochs"]:
            if not (output / f"epoch-{epoch:02d}" / "complete.json").is_file():
                raise RuntimeError(f"Missing checkpoint for epoch {epoch}")
        final = output / "final"
        if not final.exists():
            final.symlink_to(f"epoch-{args.epochs:02d}", target_is_directory=True)
        trainer.save_state()
        manifest.update(status="complete", metrics=result.metrics, final_model=str(final))
        write_json(output / "manifest.json", manifest)
        print(f"[complete] {args.epochs} epochs; final model: {final}", flush=True)


if __name__ == "__main__":
    main()
