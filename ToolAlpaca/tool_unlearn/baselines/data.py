"""Baseline datasets using the original ToolAlpaca assistant/observation masks."""
import copy
import hashlib
import json

import torch
from torch.utils.data import Dataset

from train import preprocess, IGNORE_TOKEN_ID
from tool_unlearn.dataset import _trim


METHODS = ("ga", "graddiff", "npo", "tooldelete_sft",
           "tooldelete_dpo", "retain_ft")
RETAIN_METHODS = set(METHODS) - {"ga", "npo"}
REFERENCE_METHODS = {"npo", "tooldelete_dpo"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def row_id(row):
    return digest({key: row[key] for key in ("tool", "process", "trainable")})


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON row array: %s" % path)
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Invalid row %d in %s" % (i, path))
        proc, flags = row.get("process"), row.get("trainable")
        if (not isinstance(row.get("tool"), str) or not isinstance(proc, list)
                or not isinstance(flags, list) or len(proc) != len(flags)
                or not proc or not all(isinstance(x, str) for x in proc)
                or not all(type(x) is bool for x in flags) or not any(flags)
                or flags[0]):
            raise ValueError("Invalid segmented ToolAlpaca row %d in %s" % (i, path))
    return rows


def prompt_of(row):
    return "".join(row["process"][:row["trainable"].index(True)])


def alternate_rows(rows, path):
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("schema") != "tooldelete-targets-v1":
        raise ValueError("Use prepare_targets to create %s" % path)
    records = artifact["records"]
    by_id = {record["source_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate target IDs in %s" % path)
    expected = {row_id(row) for row in rows}
    if set(by_id) != expected:
        raise ValueError("Target IDs do not match source data: %s" % path)
    result = []
    for row in rows:
        record = by_id[row_id(row)]
        response = record["response"]
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Empty tool-free response for %s" % row_id(row))
        if record["prompt"] != prompt_of(row):
            raise ValueError("Preference pair does not share its prompt")
        result.append({"tool": row["tool"], "process": [record["prompt"], response],
                       "trainable": [False, True]})
    return result


def encode(rows, tokenizer):
    pieces = []
    for start in range(0, len(rows), 64):
        sources = [copy.deepcopy([r["process"], r["trainable"]])
                   for r in rows[start:start + 64]]
        out = preprocess(sources, tokenizer)
        pieces.extend({k: v[i] for k, v in out.items()} for i in range(len(sources)))
    return pieces


def stack_examples(items):
    return _trim({k: torch.stack([item[k] for item in items])
                  for k in ("input_ids", "labels", "attention_mask")})


class BaselineDataset(Dataset):
    def __init__(self, method, tokenizer, forget_file, retain_file=None,
                 forget_targets=None, retain_targets=None, seed=42):
        if method not in METHODS:
            raise ValueError("Unknown baseline: %s" % method)
        self.method = method
        self.streams = {}
        self.ref = {}
        self.source_hashes = {}
        self.dropped = {}
        self.generator = torch.Generator().manual_seed(seed)
        groups = ["retain"] if method == "retain_ft" else ["forget"]
        if method in RETAIN_METHODS and method != "retain_ft":
            groups.append("retain")
        tool_sets = {}
        for group in groups:
            path = forget_file if group == "forget" else retain_file
            if not path:
                raise ValueError("%s needs --%s_file" % (method, group))
            rows = read_rows(path)
            tool_sets[group] = {r["tool"] for r in rows}
            self.source_hashes[group] = digest([row_id(r) for r in rows])
            encoded = encode(rows, tokenizer)
            alt = None
            needs_alt = (method == "tooldelete_dpo" or
                         (method == "tooldelete_sft" and group == "forget"))
            if needs_alt:
                target_path = forget_targets if group == "forget" else retain_targets
                if not target_path:
                    raise ValueError("%s needs --%s_targets" % (method, group))
                alt = encode(alternate_rows(rows, target_path), tokenizer)
            keep = [i for i, x in enumerate(encoded)
                    if bool(x["labels"][1:].ne(IGNORE_TOKEN_ID).any())
                    and (alt is None or bool(alt[i]["labels"][1:].ne(IGNORE_TOKEN_ID).any()))]
            self.dropped[group] = len(rows) - len(keep)
            if not keep:
                raise ValueError("All %s rows have no supervised tokens after truncation" % group)
            encoded = [encoded[i] for i in keep]
            if alt is not None:
                alt = [alt[i] for i in keep]
            if method == "tooldelete_dpo":
                # Forget: f0 wins. Retain: original tool trajectory wins.
                chosen, rejected = (alt, encoded) if group == "forget" else (encoded, alt)
                if any(torch.equal(a["input_ids"], b["input_ids"])
                       and torch.equal(a["labels"], b["labels"])
                       for a, b in zip(chosen, rejected)):
                    raise ValueError("Identical chosen/rejected pair after tokenization")
                self.streams[group + "_chosen"] = chosen
                self.streams[group + "_rejected"] = rejected
            else:
                if method == "tooldelete_sft" and group == "forget":
                    encoded = alt
                self.streams[group] = encoded
        if tool_sets.get("forget", set()) & tool_sets.get("retain", set()):
            raise ValueError("Forget and retain tools overlap")
        self.primary = "retain" if method == "retain_ft" else "forget"

    def group_size(self, group):
        name = group + "_chosen" if self.method == "tooldelete_dpo" else group
        return len(self.streams[name])

    def __len__(self):
        return self.group_size(self.primary)

    def __getitem__(self, i):
        indices = {self.primary: i}
        if self.method in RETAIN_METHODS and self.method != "retain_ft":
            indices["retain"] = int(torch.randint(self.group_size("retain"), (1,),
                                                  generator=self.generator))
        result = {}
        for name, stream in self.streams.items():
            group = name.split("_")[0]
            index = indices[group]
            result[name] = stream[index]
            if name in self.ref:
                result["ref_" + name] = self.ref[name][index]
        return result

    @staticmethod
    def collate(items):
        return {name: (torch.stack([x[name] for x in items]) if name.startswith("ref_")
                       else stack_examples([x[name] for x in items]))
                for name in items[0]}

    def fingerprint(self):
        h = hashlib.sha256()
        for name, stream in sorted(self.streams.items()):
            h.update(name.encode())
            for item in stream:
                for key in ("input_ids", "labels", "attention_mask"):
                    h.update(item[key].numpy().tobytes())
        return h.hexdigest()


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, model_max_length=2048,
                                               padding_side="right")
    if tokenizer.unk_token_id is None:
        raise ValueError("ToolAlpaca preprocessing requires an UNK padding token")
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def add_data_args(parser):
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir")
    parser.add_argument("--forget_file", default="tool_unlearn/data/forget_train.json")
    parser.add_argument("--retain_file", default="tool_unlearn/data/retain_train.json")
    parser.add_argument("--forget_targets")
    parser.add_argument("--retain_targets")
    parser.add_argument("--seed", type=int, default=42)


def make_dataset(args, tokenizer):
    return BaselineDataset(args.method, tokenizer, args.forget_file, args.retain_file,
                           args.forget_targets, args.retain_targets, args.seed)
