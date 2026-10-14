"""Preserve original row masks; mask HARU's conditioning prefix exactly."""
import copy

import torch
from torch.utils.data import Dataset

from train import IGNORE_TOKEN_ID, preprocess
from tool_unlearn.dataset import CTX_LEN, ForgetRetainDataset, ToolUnlearnDataset
from tool_unlearn.haru import read_json


def encode_action_row(row, tokenizer):
    prefix, action = row["process"]
    prefix_ids = tokenizer(prefix).input_ids
    encoded = tokenizer(prefix + action + " " + tokenizer.eos_token,
                        return_tensors="pt", padding="max_length",
                        max_length=CTX_LEN, truncation=True)
    ids = encoded.input_ids[0]
    # All generated action boundaries occur after whitespace/newline. Assert
    # prefix stability rather than silently supervising a reasoning-tail token
    # through the upstream segment-length heuristic (len(segment) - 2).
    n = min(len(prefix_ids), CTX_LEN)
    if ids[:n].tolist() != prefix_ids[:n]:
        raise ValueError("HARU tokenizer changed tokens across the reasoning/action boundary")
    labels = ids.clone()
    labels[:n] = IGNORE_TOKEN_ID
    labels[~encoded.attention_mask[0].bool()] = IGNORE_TOKEN_ID
    return {"input_ids": ids, "labels": labels, "attention_mask": encoded.attention_mask[0]}


class HaruForgetDataset(Dataset):
    def __init__(self, path, tokenizer):
        rows = read_json(path)
        self.items = []
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            encoded = preprocess(
                [copy.deepcopy([r["process"], r["trainable"]]) for r in chunk], tokenizer)
            for i, row in enumerate(chunk):
                item = (encode_action_row(row, tokenizer) if row.get("haru_action_only")
                        else {key: value[i] for key, value in encoded.items()})
                if item["labels"][1:].ne(IGNORE_TOKEN_ID).any():
                    self.items.append(item)
        print("[HARU data] %d rows, %d dropped after truncation" %
              (len(self.items), len(rows) - len(self.items)), flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class HaruDataset(ForgetRetainDataset):
    def __init__(self, forget_file, tokenizer, retain_file=None, seed=42):
        self.forget = HaruForgetDataset(forget_file, tokenizer)
        self.retain = ToolUnlearnDataset(retain_file, tokenizer) if retain_file else None
        self.ref_seq_logp = None
        self._gen = torch.Generator().manual_seed(seed)
