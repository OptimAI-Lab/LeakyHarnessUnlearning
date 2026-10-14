"""Cache frozen original-model scores before launching distributed training."""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from tool_unlearn.trainer import _seq_logp
from .artifacts import checkpoint_signature
from .data import (REFERENCE_METHODS, add_data_args, load_tokenizer, make_dataset,
                   stack_examples)


def reference_names(dataset):
    if dataset.method == "tooldelete_dpo":
        return sorted(dataset.streams)
    return ["forget"]


def metadata(dataset, model_dir):
    return {"schema": "baseline-reference-v1", "method": dataset.method,
            "dataset_sha256": dataset.fingerprint(),
            "checkpoint_signature": checkpoint_signature(model_dir)}


@torch.no_grad()
def score_stream(model, stream, device, batch_size=1):
    scores = []
    model.eval()
    for start in range(0, len(stream), batch_size):
        batch = {k: v.to(device) for k, v in
                 stack_examples(stream[start:start + batch_size]).items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        scores.append(_seq_logp(out.logits, batch["labels"]).float().cpu())
        if start % 50 == 0:
            print("[reference] %d/%d" % (start, len(stream)), flush=True)
    return torch.cat(scores)


def attach_reference(dataset, path, model_dir):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["metadata"] != metadata(dataset, model_dir):
        raise ValueError("Reference cache is stale or belongs to another model/dataset/method")
    if set(artifact["scores"]) != set(reference_names(dataset)):
        raise ValueError("Incomplete reference cache")
    for name, score in artifact["scores"].items():
        if score.shape != (len(dataset.streams[name]),) or not bool(torch.isfinite(score).all()):
            raise ValueError("Invalid reference scores for %s" % name)
    dataset.ref = artifact["scores"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    if args.method not in REFERENCE_METHODS:
        parser.error("Only NPO and DPO need a reference cache")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("Refusing to overwrite %s" % output)
    tokenizer = load_tokenizer(args.tokenizer_dir or args.model_dir)
    dataset = make_dataset(args, tokenizer)
    provenance = metadata(dataset, args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True
    ).to(args.device)
    model.config.use_cache = False
    scores = {name: score_stream(model, dataset.streams[name], args.device, args.batch_size)
              for name in reference_names(dataset)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    torch.save({"metadata": provenance, "scores": scores}, temp)
    temp.replace(output)
    print("[reference] saved %s" % output)


if __name__ == "__main__":
    main()
