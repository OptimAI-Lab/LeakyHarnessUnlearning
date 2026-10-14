"""HARU training worker; reuses the original datasets and GA/NPO losses."""
import gc
import os
from datetime import timedelta

from tool_unlearn.haru import file_hash, stage_paths, write_json


def score_stream(model, stream, device, batch_size=1):
    """Compute frozen-reference sequence log probabilities in small batches."""
    import torch
    from tool_unlearn.trainer import _seq_logp

    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(stream), batch_size):
            items = stream[start:start + batch_size]
            width = max(int(item["attention_mask"].sum()) for item in items)
            batch = {
                key: torch.stack([item[key][:width] for item in items]).to(device)
                for key in ("input_ids", "labels", "attention_mask")
            }
            output = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
            scores.append(_seq_logp(output.logits, batch["labels"]).float().cpu())
    return torch.cat(scores)


def train_stage(args, stage):
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, TrainingArguments, set_seed
    from tool_unlearn.haru_dataset import HaruDataset
    from tool_unlearn.haru_trainer import HaruTrainer
    from tool_unlearn.unlearn import load_tokenizer

    set_seed(args.seed)
    if args.cpu:
        os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; --cpu is for small smoke tests only")
    device = torch.device("cpu" if args.cpu else "cuda:%d" % local_rank)
    if not args.cpu:
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
    rank = dist.get_rank() if dist.is_initialized() else 0
    model_path, forget_path, output, epochs = stage_paths(args, stage)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing to overwrite training stage %s" % output)
        output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    tokenizer = load_tokenizer(args.tokenizer_dir or args.model_dir)
    dataset = HaruDataset(
        forget_path, tokenizer,
        retain_file=args.retain_file if args.retain_loss == "ce" else None,
        seed=args.seed + rank)
    if not len(dataset) or (dataset.retain is not None and not len(dataset.retain)):
        raise ValueError("No supervised tokens remain after truncation")
    dtype = torch.float32 if args.cpu else torch.bfloat16
    if args.algo == "npo":
        # Always reference the ORIGINAL model, including newly mined rows.
        # Each stage gets a fresh cache after tokenization/filtering, so scores
        # cannot accidentally be reused for different prompts or row indices.
        cache = output / "ref_seq_logp.pt"
        if rank == 0:
            reference = AutoModelForCausalLM.from_pretrained(
                args.model_dir, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
            reference.config.use_cache = False
            values = score_stream(reference, dataset.forget, device, args.ref_batch_size)
            torch.save(values, cache)
            del reference, values
            gc.collect()
            if not args.cpu:
                torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        dataset.ref_seq_logp = torch.load(cache, map_location="cpu")
        if len(dataset.ref_seq_logp) != len(dataset):
            raise ValueError("Reference score count differs from the training dataset")
    # Create TrainingArguments before policy loading to enable ZeRO-3 init.
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        optim=args.optim, lr_scheduler_type="constant", weight_decay=0.0,
        bf16=not args.cpu, no_cuda=args.cpu,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        save_strategy=args.save_strategy, save_total_limit=None, logging_steps=1, remove_unused_columns=False,
        label_names=[], dataloader_num_workers=0, deepspeed=args.deepspeed,
        seed=args.seed, report_to=[], ddp_find_unused_parameters=False,
        local_rank=local_rank if dist.is_initialized() else -1)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.config.use_cache = False
    loss_type = args.algo + ("_gdr" if args.retain_loss == "ce" else "")
    trainer = HaruTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset, data_collator=dataset.get_collate_fn(),
        method=args.algo, beta=args.beta, forget_weight=args.forget_weight,
        retain_weight=args.retain_weight)
    result = trainer.train()
    trainer.save_model(str(output))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output)
        write_json(output / "haru_stage.json", {
            "stage": stage, "policy_start": model_path, "reference_model": args.model_dir,
            "loss": loss_type, "epochs": epochs, "forget_rows": len(dataset),
            "forget_weight": args.forget_weight, "retain_weight": args.retain_weight,
            "save_strategy": args.save_strategy,
            "forget_sha256": file_hash(forget_path), "metrics": result.metrics})
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
