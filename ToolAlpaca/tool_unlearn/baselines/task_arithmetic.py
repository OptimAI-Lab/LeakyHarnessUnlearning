"""CPU, shardwise ToolDelete GCR: theta_out = theta_unlearned + alpha*(theta0-thetaR)."""
import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import checkpoint_signature, new_output_dir, write_json


class CheckpointReader:
    """Keep at most one .bin shard in memory; safetensors are opened per tensor."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.cached_name, self.cached = None, None
        self.config = json.loads((self.directory / "config.json").read_text())
        self.weight_map = None
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (self.directory / name).exists():
                self.weight_map = json.loads((self.directory / name).read_text())["weight_map"]
                break
        if self.weight_map is None:
            if (self.directory / "model.safetensors").exists():
                with safe_open(str(self.directory / "model.safetensors"), framework="pt") as handle:
                    self.weight_map = {key: "model.safetensors" for key in handle.keys()}
            elif (self.directory / "pytorch_model.bin").exists():
                self._load_bin("pytorch_model.bin")
                self.weight_map = {key: "pytorch_model.bin" for key in self.cached}
            else:
                raise ValueError("No HF weights in %s" % directory)

    def _load_bin(self, name):
        if name != self.cached_name:
            self.cached = None
            self.cached = torch.load(self.directory / name, map_location="cpu", weights_only=True)
            self.cached_name = name

    def get(self, key):
        name = self.weight_map[key]
        if name.endswith(".safetensors"):
            with safe_open(str(self.directory / name), framework="pt") as handle:
                return handle.get_tensor(key)
        self._load_bin(name)
        return self.cached[key]


def write_checkpoint(tensors, output, config_from, shard_bytes):
    if shard_bytes < 1:
        raise ValueError("Shard size must be positive")
    weight_map, shard, size, total = {}, {}, 0, 0
    shard_id = 0

    def flush():
        nonlocal shard, size, shard_id
        if not shard:
            return
        shard_id += 1
        name = "model-%05d.safetensors" % shard_id
        save_file(shard, str(output / name), metadata={"format": "pt"})
        weight_map.update({key: name for key in shard})
        shard, size = {}, 0

    for key, value in tensors:
        nbytes = value.numel() * value.element_size()
        if size + nbytes > shard_bytes:
            flush()
        shard[key] = value.contiguous().clone()
        size += nbytes
        total += nbytes
    flush()
    write_json(output / "model.safetensors.index.json",
               {"metadata": {"total_size": total}, "weight_map": weight_map})
    for name in ("config.json", "generation_config.json", "tokenizer.model", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        path = Path(config_from) / name
        if path.is_file():
            shutil.copy2(path, output / name)
    return shard_id


def combine(unlearned_dir, toolfree_dir, random_dir, output_dir, alpha, shard_bytes=2_000_000_000):
    if not math.isfinite(alpha) or alpha < 0 or shard_bytes < 1:
        raise ValueError("alpha must be finite and nonnegative, shard size must be positive")
    inputs = (unlearned_dir, toolfree_dir, random_dir)
    readers = [CheckpointReader(path) for path in inputs]
    key_set = set(readers[0].weight_map)
    if any(set(r.weight_map) != key_set for r in readers[1:]):
        raise ValueError("Checkpoint parameter names differ (including tied-weight serialization)")
    architecture = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "vocab_size",
                    "tie_word_embeddings", "rope_scaling", "rope_theta")
    if any(any(r.config.get(k) != readers[0].config.get(k) for k in architecture)
           for r in readers[1:]):
        raise ValueError("Task arithmetic requires identical architectures and vocabulary")
    tokenizers = [Path(path) / "tokenizer.model" for path in inputs[:2]]
    if all(p.exists() for p in tokenizers) and tokenizers[0].read_bytes() != tokenizers[1].read_bytes():
        raise ValueError("Unlearned and tool-free tokenizers have different vocabularies")
    output = new_output_dir(output_dir, inputs)

    def parameters():
        for key in sorted(key_set, key=lambda k: (readers[0].weight_map[k], k)):
            tensors = [r.get(key) for r in readers]
            if any(t.shape != tensors[0].shape for t in tensors):
                raise ValueError("Parameter shape mismatch: %s" % key)
            source = tensors[0]
            if source.is_floating_point():
                value = (source.float() + alpha * (tensors[1].float() - tensors[2].float())).to(source.dtype)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite task-arithmetic parameter: %s" % key)
            else:
                if any(not torch.equal(source, t) for t in tensors[1:]):
                    raise ValueError("Nonfloating buffers differ: %s" % key)
                value = source.clone()
            yield key, value

    shard_id = write_checkpoint(parameters(), output, unlearned_dir, shard_bytes)
    write_json(output / "task_arithmetic_manifest.json", {
        "schema": "tooldelete-gcr-v1", "alpha": alpha, "task_arithmetic_applied": True,
        "equation": "unlearned + alpha * (toolfree - random)",
        "inputs": [{"path": str(Path(p).resolve()), "signature": checkpoint_signature(p)}
                   for p in inputs]})
    print("[GCR] wrote %d shards to %s" % (shard_id, output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlearned_model_dir", required=True)
    parser.add_argument("--toolfree_model_dir", required=True)
    parser.add_argument("--random_model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--shard_mb", type=int, default=2000)
    args = parser.parse_args()
    combine(args.unlearned_model_dir, args.toolfree_model_dir, args.random_model_dir,
            args.out_dir, args.alpha, args.shard_mb * 1_000_000)


if __name__ == "__main__":
    main()
