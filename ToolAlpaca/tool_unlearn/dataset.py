"""Datasets for tool unlearning.

Adapted from MUSE's `baselines/baselines/dataset.py`, with two changes that
matter for correctness on ToolAlpaca:

1.  MUSE unlearns a raw text *corpus*: it sets `labels = input_ids`, so the loss
    covers every token.  ToolAlpaca is an instruction-tuned agent -- its training
    loss only ever covered the segments the model itself produced (Thought /
    Action / Action Input / final Response), never the prompt and never the tool
    Observations.  Carrying MUSE's behaviour over would make gradient ascent push
    up the likelihood of the *prompt text* and the *simulated API responses* as
    well, which is not the knowledge being removed and is a fast way to destroy
    general ability.  So labels are masked exactly as in training.

    To guarantee "exactly", tokenization is delegated to `train.preprocess` --
    the upstream ToolAlpaca function that produced the model in the first place --
    rather than reimplemented here.  That includes its quirks: `" </s>"` appended
    to the final segment, the `-2` correction for BOS plus SentencePiece's
    leading-space token, `pad_token = unk_token`, right truncation at 2048.

2.  MUSE pairs forget sample i with retain sample `i % len(retain)`.  With 802
    forget and 3135 retain examples that means the same 802 retain examples are
    reused every epoch and the other 74% are never seen, silently weakening the
    retain term of gd / KL.  Here the retain partner is drawn uniformly at
    random instead, so the whole retain set contributes.
"""
import copy
import json
import os
import sys
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import preprocess, IGNORE_TOKEN_ID   # noqa: E402  (upstream tokenizer)

CTX_LEN = 2048          # ToolAlpaca is a LLaMA-1 fine-tune; train.preprocess pads here


class ToolUnlearnDataset(Dataset):
    """One split (forget or retain) of the tool-level partition."""

    def __init__(self, file_path, tokenizer, chunk_size=256, verbose=True):
        rows = json.load(open(file_path, encoding="utf-8"))
        self.tools = [r["tool"] for r in rows]

        input_ids, labels, attn = [], [], []
        for i in range(0, len(rows), chunk_size):
            # deep copy: preprocess mutates sources in place (it appends the EOS
            # token to the last segment), so reusing a row would double-append.
            sources = [copy.deepcopy([r["process"], r["trainable"]])
                       for r in rows[i:i + chunk_size]]
            out = preprocess(sources, tokenizer)
            input_ids.append(out["input_ids"])
            labels.append(out["labels"])
            attn.append(out["attention_mask"])

        self.input_ids = torch.cat(input_ids)
        self.labels = torch.cat(labels)
        self.attention_mask = torch.cat(attn)

        # An example longer than the 2048-token context loses its tail to
        # truncation.  If the tail was the only supervised part, the example
        # carries no signal at all -- keeping it would contribute a NaN-prone
        # all-masked loss row, so drop it and say how many.
        keep = (self.labels != IGNORE_TOKEN_ID).any(dim=1)
        n_drop = int((~keep).sum())
        if n_drop:
            self.input_ids = self.input_ids[keep]
            self.labels = self.labels[keep]
            self.attention_mask = self.attention_mask[keep]
            self.tools = [t for t, k in zip(self.tools, keep.tolist()) if k]
        if verbose:
            print("[data] %s: %d examples (%d dropped: truncation left no "
                  "supervised tokens), %d tools"
                  % (os.path.basename(file_path), len(self.input_ids), n_drop,
                     len(set(self.tools))))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i],
                "labels": self.labels[i],
                "attention_mask": self.attention_mask[i]}


def _trim(batch):
    """Drop trailing all-padding columns.

    train.preprocess pads every example to 2048.  Attention masks and -100
    labels already make those positions inert, so cutting the columns no example
    uses is numerically identical and saves most of the compute on a corpus
    whose median length is ~1400.
    """
    keep = int(batch["attention_mask"].sum(dim=0).nonzero().max()) + 1
    return {k: v[:, :keep] for k, v in batch.items()}


class ForgetRetainDataset(Dataset):
    """Yields (forget_example, retain_example); retain is None if unused."""

    def __init__(self, forget_file, tokenizer, retain_file=None, seed=42,
                 ref_seq_logp=None):
        self.forget = ToolUnlearnDataset(forget_file, tokenizer)
        self.retain = (ToolUnlearnDataset(retain_file, tokenizer)
                       if retain_file is not None else None)
        # Frozen-reference sequence log-probs for each forget example, summed
        # over supervised positions.  Computed once by unlearn.py and attached
        # there (set after construction); defaults to None => no NPO shortcut.
        self.ref_seq_logp = ref_seq_logp
        # Sampling the retain partner rather than pairing by index; see the
        # module docstring.  Keep dataloader_num_workers=0 so this single
        # generator drives the whole epoch.
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, i):
        f = dict(self.forget[i])
        if self.ref_seq_logp is not None:
            f["ref_seq_logp"] = self.ref_seq_logp[i]
        if self.retain is None:
            return f, None
        j = int(torch.randint(len(self.retain), (1,), generator=self._gen))
        return f, self.retain[j]

    def get_collate_fn(self):
        def collate_fn(batch: List[Tuple[dict, Optional[dict]]]):
            def stack(items):
                return _trim({k: torch.stack([it[k] for it in items])
                              for k in ("input_ids", "labels", "attention_mask")})

            x_f = stack([p[0] for p in batch])
            if batch[0][0].get("ref_seq_logp") is not None:
                x_f["ref_seq_logp"] = torch.stack([p[0]["ref_seq_logp"] for p in batch])
            x_r = stack([p[1] for p in batch]) if batch[0][1] is not None else None
            return x_f, x_r
        return collate_fn
