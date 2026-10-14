"""Unlearning losses: GA, GD (gdr), NPO, and their KL-retain variants.

Adapted from MUSE's `baselines/baselines/iterative.py`, with three corrections.

1.  **NPO was computed on raw logits.**  MUSE has

        neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
        loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

    `logits` is `[B, T, V]` of unnormalised scores, so this averages a
    difference over the whole vocabulary.  NPO is defined on the *sequence
    log-likelihood of the labels*:

        L = -(2/beta) * E[ log sigmoid( beta * (log pi_ref(y|x) - log pi_theta(y|x)) ) ]

    which needs a log_softmax and a gather at the label ids, summed over the
    supervised positions of each sequence.  Implemented in `_seq_logp` below.
    Sanity check: at initialisation theta == ref, so the ratio is 0 and the loss
    starts at (2/beta)*log 2 == 13.86 for beta=0.1.

2.  **KL was computed on raw logits too.**  `F.kl_div` expects log-probabilities;
    MUSE passes logits, so the "KL" it minimises is not a KL.  Here both sides go
    through log_softmax, and the divergence is restricted to supervised positions
    (padding and prompt tokens would otherwise dominate the average).

3.  **Python 3.8.**  MUSE uses `str | None` annotations (3.10+).  This repo's
    interpreter is 3.8.10, so `Optional[...]` is used throughout.

Loss names are matched by substring, as in MUSE: `ga`, `ga_gdr`, `ga_klr`,
`npo`, `npo_gdr`, `npo_klr`.
"""
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import IGNORE_TOKEN_ID   # noqa: E402

VALID_LOSSES = ("ga", "ga_gdr", "ga_klr", "npo", "npo_gdr", "npo_klr")


def _seq_logp(logits, labels):
    """Sum of log p(y_t) over supervised positions, one value per sequence."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != IGNORE_TOKEN_ID
    # cross_entropy needs a valid class index everywhere; masked positions are
    # zeroed out afterwards.  Going through cross_entropy rather than an explicit
    # log_softmax + gather keeps one fewer [B, T, V] tensor alive -- at V=32000
    # and T=2048 that is ~260MB per copy, on top of a training step that is
    # already the memory bottleneck.
    safe = labels.masked_fill(~mask, 0)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        safe.reshape(-1),
        reduction="none",
    ).view_as(safe)
    return -(nll * mask).sum(dim=-1)


def _masked_kl(logits_p, logits_q, labels):
    """KL(q || p) averaged over supervised positions. q is the reference."""
    logits_p = logits_p[:, :-1, :]
    logits_q = logits_q[:, :-1, :]
    mask = (labels[:, 1:] != IGNORE_TOKEN_ID)
    if not bool(mask.any()):
        return logits_p.sum() * 0.0
    log_p = torch.log_softmax(logits_p.float(), dim=-1)
    log_q = torch.log_softmax(logits_q.float(), dim=-1)
    # F.kl_div(input, target) == KL(target || input); both already log-space.
    kl = F.kl_div(log_p, log_q, reduction="none", log_target=True).sum(-1)
    return (kl * mask).sum() / mask.sum()


class ToolUnlearner(Trainer):

    def __init__(self, *args, loss_type="ga", ref_model=None, ref_seq_logp=None,
                 beta=0.1, **kwargs):
        if loss_type not in VALID_LOSSES:
            raise ValueError("loss_type must be one of %s, got %r"
                             % (list(VALID_LOSSES), loss_type))
        self.loss_type = loss_type
        self.ref_model = ref_model
        # Frozen reference sequence-log-probs for every forget example, summed
        # over the supervised positions (see _seq_logp).  Precomputed once by
        # unlearn.py and cached, so the 14GB reference model never needs to sit
        # on the same GPU as the trainable policy.  Numerically identical to
        # running the reference forward every step: log pi_ref(y|x) does not
        # depend on theta, so it is a constant of the NPO objective.
        self.ref_seq_logp = ref_seq_logp
        self.beta = beta
        needs_ref = "npo" in loss_type or "kl" in loss_type
        if needs_ref and ref_model is None and ref_seq_logp is None:
            raise ValueError("loss_type %r needs a reference model or "
                             "precomputed reference log-probs" % loss_type)
        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        x_f, x_r = inputs
        # The collate_fn packs the precomputed frozen-reference log-probs into
        # the forget batch, but they are a loss input, not a model kwarg -- pull
        # them out before the forward so LlamaForCausalLM never sees the key.
        ref_logp = x_f.pop("ref_seq_logp", None)
        needs_retain = "gdr" in self.loss_type or "klr" in self.loss_type
        if needs_retain and x_r is None:
            raise ValueError("loss_type %r needs a retain batch but the dataset "
                             "was built without a retain file" % self.loss_type)

        out_f = model(**x_f)

        if self.loss_type.startswith("ga"):
            # Gradient ascent: maximise the forget NLL.  Unbounded below -- it
            # will happily diverge if trained too long, which is why the paper
            # reports GradAscent wrecking T_T and T_r.  Control it with epochs.
            loss = -out_f.loss
        else:
            if ref_logp is None:
                # No precomputed cache: evaluate the frozen reference on the fly.
                with torch.no_grad():
                    out_f_ref = self.ref_model(**x_f)
                ref_logp = _seq_logp(out_f_ref.logits, x_f["labels"])
            neg_log_ratio = ref_logp - _seq_logp(out_f.logits, x_f["labels"])
            loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        if "gdr" in self.loss_type:
            loss = loss + model(**x_r).loss

        if "klr" in self.loss_type:
            out_r = model(**x_r)
            with torch.no_grad():
                out_r_ref = self.ref_model(**x_r)
            loss = loss + _masked_kl(out_r.logits, out_r_ref.logits, x_r["labels"])

        return (loss, out_f) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        x_f, _ = inputs
        x_f = {k: v for k, v in x_f.items() if k != "ref_seq_logp"}
        with torch.no_grad():
            out = model(**x_f)
        return (out.loss, out.logits, x_f["labels"])
