"""HARU: unlearn -> mine planner-assisted failures -> continue unlearning.

The coordinator uses separate processes so training and generation models never
coexist in GPU memory. Existing training/generation entry points are unchanged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", default="TangQiaoYu/ToolAlpaca-7B", help="Original actor checkpoint")
    p.add_argument("--planner_model_dir", default="TangQiaoYu/ToolAlpaca-13B", help="ToolAlpaca-13B checkpoint")
    p.add_argument("--out_dir", required=True, help="New/empty run directory")
    p.add_argument("--algo", choices=("npo", "ga"), default="npo")
    p.add_argument("--retain_loss", choices=("none", "ce"), default="none")
    p.add_argument("--forget_weight", type=float, default=1.0)
    p.add_argument("--retain_weight", type=float, default=1.0)
    p.add_argument("--save_strategy", choices=("no", "epoch"), default="epoch")
    p.add_argument("--initial_epochs", type=float, default=1)
    p.add_argument("--final_epochs", type=float, default=2)
    p.add_argument("--num_generations", type=int, default=10)
    p.add_argument("--forget_file", default=str(ROOT / "tool_unlearn/data/forget_train.json"))
    p.add_argument("--retain_file", default=str(ROOT / "tool_unlearn/data/retain_train.json"))
    p.add_argument("--api_file", default=str(ROOT / "data/train_data.json"))
    p.add_argument("--split_file", default=str(ROOT / "tool_unlearn/data/split.json"))
    p.add_argument("--tokenizer_dir")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--optim", choices=("adafactor", "adamw_torch"), default="adafactor")
    p.add_argument("--ref_batch_size", type=int, default=1)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--train_gpus", help="CUDA_VISIBLE_DEVICES for training; default inherits environment")
    p.add_argument("--num_train_processes", type=int,
                   help="Training processes/GPUs: default 8 (1 with --cpu)")
    p.add_argument("--deepspeed",
                   help="Defaults to tool_unlearn/ds_zero3_adafactor.json for GPU training")
    p.add_argument("--generation_gpus", help="CUDA_VISIBLE_DEVICES for generation")
    p.add_argument("--num_generation_workers", type=int,
                   help="Independent planner/actor workers: default 4 (1 with --cpu)")
    p.add_argument("--planner_device", default="cuda:0")
    p.add_argument("--actor_device", default="cuda:1")
    p.add_argument("--temperature", type=float, default=0.8, help="Actor sampling temperature")
    p.add_argument("--planner_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--planner_max_new_tokens", type=int, default=256)
    p.add_argument("--generation_mode", choices=("rollout", "first_action"), default="first_action")
    p.add_argument("--max_tool_steps", type=int, default=5)
    p.add_argument("--simulator_model", default=os.getenv("SIMULATOR_MODEL", "gpt-3.5-turbo"))
    p.add_argument("--simulator_port", type=int, default=5697)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Float32 CPU smoke tests")
    p.add_argument("--dry_run", action="store_true", help="Check inputs and print the stages without loading models")
    p.add_argument("--_stage", choices=("initial", "augment", "augment_worker", "final"), help=argparse.SUPPRESS)
    p.add_argument("--_generation_worker_id", type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.num_train_processes is None:
        args.num_train_processes = 1 if args.cpu else 8
    if not args.cpu and args.deepspeed is None:
        args.deepspeed = str(ROOT / "tool_unlearn/ds_zero3_adafactor.json")
    if args.num_generation_workers is None:
        args.num_generation_workers = 1 if args.cpu else 4
    positive = ("initial_epochs", "final_epochs", "num_generations", "lr", "beta",
                "per_device_batch_size", "grad_accum", "ref_batch_size", "num_train_processes",
                "temperature", "planner_temperature", "max_new_tokens",
                "planner_max_new_tokens", "max_tool_steps", "num_generation_workers",
                "forget_weight", "retain_weight")
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in positive):
        p.error("Epochs, rates, temperatures and counts must be positive and finite")
    if not 0 < args.top_p <= 1 or not 1 <= args.simulator_port <= 65535:
        p.error("top_p must be in (0, 1]; simulator_port must be in [1, 65535]")
    if args.cpu and (args.deepspeed or args.num_train_processes != 1):
        p.error("--cpu requires one training process and no DeepSpeed")
    if args.cpu:
        args.actor_device = args.planner_device = "cpu"
    if args.generation_mode == "rollout" and args.num_generation_workers != 1:
        p.error("Parallel HARU generation uses first_action; rollout requires --num_generation_workers 1")
    return args


def stage_paths(args, stage):
    output = Path(args.out_dir).resolve()
    if stage == "initial":
        return args.model_dir, args.forget_file, output / "initial", args.initial_epochs
    if stage == "final":
        return str(output / "initial"), str(output / "forget_augmented.json"), output / "final", args.final_epochs
    raise ValueError(stage)


def command_for_stage(args, stage):
    command = [sys.executable]
    if stage not in ("augment", "augment_worker") and args.num_train_processes > 1:
        command += ["-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node", str(args.num_train_processes)]
    command.append(str(Path(__file__).resolve()))
    for key, value in vars(args).items():
        if key in ("_stage", "dry_run") or value is None or value is False:
            continue
        command.append("--" + key)
        if value is not True:
            command.append(str(value))
    return command + ["--_stage", stage]


def main(argv=None):
    args = parse_args(argv)
    if args._stage:
        if args._stage == "augment_worker":
            from tool_unlearn.haru_augment import augmentation_worker
            augmentation_worker(args)
        elif args._stage == "augment":
            from tool_unlearn.haru_augment import augment
            augment(args)
        else:
            from tool_unlearn.haru_train import train_stage
            train_stage(args, args._stage)
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch haru.py with python; use --num_train_processes for distributed training")
    # Reconstruct ONLY the frozen training split, before spending GPU time.
    from tool_unlearn.source_data import reconstruct_sources
    from tool_unlearn.haru_augment import generation_gpu_groups
    # Validate the parallel placement before either training stage starts.
    gpu_groups = generation_gpu_groups(args)
    rows, _ = reconstruct_sources(args.forget_file, args.api_file, args.split_file)
    if not rows:
        raise ValueError("The forget training set is empty")
    if args.retain_loss == "ce":
        retain = read_json(args.retain_file)
        if not retain or {r["tool"] for r in rows} & {r["tool"] for r in retain}:
            raise ValueError("Retain data must be nonempty and its tools disjoint from forget")
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new/empty --out_dir; existing runs are never overwritten")
    commands = [command_for_stage(args, stage) for stage in ("initial", "augment", "final")]
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        print("[HARU] %d queries x %d generations; epochs %g + %g" % (
            len(rows), args.num_generations, args.initial_epochs, args.final_epochs))
        print("[HARU] generation workers=%d, GPU groups=%s" % (len(gpu_groups), gpu_groups))
        return
    if args.generation_mode == "rollout":
        from tool_unlearn.source_data import preflight_remote_model
        preflight_remote_model(args.simulator_model)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": vars(args), "reference_model": args.model_dir,
                "input_hashes": {k: file_hash(getattr(args, k)) for k in
                                 ("forget_file", "api_file", "split_file")}}
    if args.retain_loss == "ce":
        manifest["input_hashes"]["retain_file"] = file_hash(args.retain_file)
    try:
        for stage, command in zip(("initial", "augment", "final"), commands):
            manifest["status"] = stage
            write_json(output / "manifest.json", manifest)
            env = dict(os.environ)
            gpus = args.generation_gpus if stage == "augment" else args.train_gpus
            if gpus is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpus
            env["TOKENIZERS_PARALLELISM"] = "false"
            subprocess.run(command, env=env, check=True)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="failed", failed_stage=stage, error=str(exc))
        write_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="complete", final_model=str(output / "final"))
    write_json(output / "manifest.json", manifest)
    print("[HARU] final model: %s" % (output / "final"), flush=True)


if __name__ == "__main__":
    main()
