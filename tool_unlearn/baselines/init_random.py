"""Create a seeded, architecture-matched random fR on CPU for task arithmetic."""
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, set_seed

from .artifacts import new_output_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_from", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = new_output_dir(args.out_dir, (args.config_from,))
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.config_from)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).cpu()
    model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    write_json(output / "random_manifest.json", {"seed": args.seed, "config_from": args.config_from,
                                                "dtype": "bfloat16", "historical_initialization": False})


if __name__ == "__main__":
    main()
