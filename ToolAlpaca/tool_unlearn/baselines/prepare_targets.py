"""Generate resumable, shardable responses with a pre-tool-training model f0."""
import argparse
import fcntl
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, set_seed

from .artifacts import checkpoint_signature, write_json
from .data import digest, load_tokenizer, prompt_of, read_rows, row_id


def generate(args):
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Need 0 <= shard_id < num_shards")
    if not 1 <= args.max_new_tokens < 2048:
        raise ValueError("max_new_tokens must be in [1, 2047]")
    rows = read_rows(args.input_file)
    # Repeated augmented trajectories may share a source ID. Score each unique
    # source once; training still retains every occurrence in the source array.
    unique = list({row_id(row): row for row in rows}.values())
    metadata = {"source_sha256": digest([row_id(row) for row in rows]),
                "teacher_signature": checkpoint_signature(args.teacher_model_dir),
                "teacher_model_dir": str(Path(args.teacher_model_dir).resolve()),
                "tokenizer_dir": str(Path(args.tokenizer_dir or args.teacher_model_dir).resolve()),
                "num_shards": args.num_shards, "shard_id": args.shard_id,
                "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                "dtype": args.dtype, "prompt_mode": "original-prefix-no-execution",
                "context_policy": "full-prompt-adaptive-completion-reserve-eos-v1"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_name(output.name + ".progress.jsonl")
    meta_path = output.with_name(output.name + ".progress.meta.json")
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise ValueError("Target resume metadata mismatch; choose a new output")
    elif journal.exists() or output.exists():
        raise ValueError("Existing target output lacks matching provenance")
    else:
        write_json(meta_path, metadata)
    selected = unique[args.shard_id::args.num_shards]
    expected = {row_id(row): prompt_of(row) for row in selected}
    completed = {}
    if journal.exists():
        with journal.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = record["source_id"]
                if (key in completed or key not in expected or record["prompt"] != expected[key]
                        or not isinstance(record.get("response"), str)
                        or not record["response"].strip()):
                    raise ValueError("Invalid target journal record: %s" % key)
                completed[key] = record
    pending = [row for row in selected if row_id(row) not in completed]
    if pending:
        set_seed(args.seed)
        tokenizer = load_tokenizer(args.tokenizer_dir or args.teacher_model_dir)
        # Never silently truncate the question or tool specification in a pair.
        for row in pending:
            size = len(tokenizer(prompt_of(row)).input_ids)
            if size > 2046:
                raise ValueError("Prompt %s has %d tokens; no room for a response (no silent truncation)"
                                 % (row_id(row), size))
        # Eight workers must not each materialize an unsharded 7B CPU copy at
        # once. Only loading is serialized; generation proceeds in parallel.
        with (output.parent / ".teacher_load.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_dir, torch_dtype=getattr(torch, args.dtype),
                low_cpu_mem_usage=True).to(args.device).eval()
        with journal.open("a", encoding="utf-8") as handle, torch.no_grad():
            for row in pending:
                prompt = prompt_of(row)
                encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
                budget = min(args.max_new_tokens, 2047 - encoded.input_ids.shape[1])
                generated = model.generate(
                    **encoded, do_sample=False, max_new_tokens=budget,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                suffix = generated[0, encoded.input_ids.shape[1]:]
                response = tokenizer.decode(suffix, skip_special_tokens=True)
                if not response.strip():
                    raise ValueError("Empty teacher response for %s" % row_id(row))
                record = {"source_id": row_id(row), "prompt": prompt, "response": response,
                          "generated_tokens": len(suffix), "token_budget": budget,
                          "hit_token_limit": len(suffix) == budget
                          and int(suffix[-1]) != tokenizer.eos_token_id}
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                completed[row_id(row)] = record
                print("[targets %d] %d/%d" % (args.shard_id, len(completed), len(selected)), flush=True)
    write_json(output, {"schema": "tooldelete-targets-v1", "metadata": metadata,
                        "records": [completed[row_id(row)] for row in selected]})
    print("[targets] wrote %s" % output)


def merge(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite %s" % output)
    rows = read_rows(args.input_file)
    expected = {row_id(row): prompt_of(row) for row in rows}
    records, provenance, shard_ids = {}, None, set()
    for path in args.shards:
        with open(path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact["schema"] != "tooldelete-targets-v1":
            raise ValueError("Invalid target shard: %s" % path)
        meta = dict(artifact["metadata"])
        shard_id = meta.pop("shard_id")
        if shard_id in shard_ids:
            raise ValueError("Duplicate shard ID")
        shard_ids.add(shard_id)
        if provenance is not None and provenance != meta:
            raise ValueError("Inconsistent shard provenance")
        provenance = meta
        for record in artifact["records"]:
            key = record["source_id"]
            if (key in records or key not in expected or record["prompt"] != expected[key]
                    or not isinstance(record.get("response"), str) or not record["response"].strip()):
                raise ValueError("Duplicate, stale, or invalid target record")
            records[key] = record
    if (provenance["source_sha256"] != digest([row_id(row) for row in rows])
            or shard_ids != set(range(provenance["num_shards"])) or set(records) != set(expected)):
        raise ValueError("Missing shards or mismatched source data")
    write_json(output, {"schema": "tooldelete-targets-v1", "metadata": provenance,
                        "records": [records[key] for key in expected]})
    print("[targets] merged %d targets -> %s" % (len(records), output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--input_file", required=True)
    gen.add_argument("--teacher_model_dir", required=True)
    gen.add_argument("--tokenizer_dir")
    gen.add_argument("--output", required=True)
    gen.add_argument("--device", default="cuda:0")
    gen.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    gen.add_argument("--max_new_tokens", type=int, default=256)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--num_shards", type=int, default=1)
    gen.add_argument("--shard_id", type=int, default=0)
    combine = sub.add_parser("merge")
    combine.add_argument("--input_file", required=True)
    combine.add_argument("--shards", nargs="+", required=True)
    combine.add_argument("--output", required=True)
    args = parser.parse_args()
    (generate if args.command == "generate" else merge)(args)


if __name__ == "__main__":
    main()
