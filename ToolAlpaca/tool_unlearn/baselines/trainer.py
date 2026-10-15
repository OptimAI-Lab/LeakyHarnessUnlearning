"""Losses for the opt-in baseline runner. All log probabilities are masked."""
import torch
import torch.nn.functional as F
from transformers import Trainer

from tool_unlearn.trainer import _seq_logp


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = (chosen - rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()


def npo_loss(logp, ref_logp, beta):
    if beta <= 0:
        raise ValueError("beta must be positive")
    return -2.0 / beta * F.logsigmoid(beta * (ref_logp - logp)).mean()


class BaselineTrainer(Trainer):
    def __init__(self, *args, method, beta=0.1, forget_weight=1.0,
                 retain_weight=1.0, **kwargs):
        self.method = method
        self.beta = beta
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # One forward avoids reusing checkpointed layers in multiple reentrant
        # backward passes under DDP. Each group still has its own normalization.
        names = [name for name in inputs if not name.startswith("ref_")]
        width = max(inputs[name]["input_ids"].shape[1] for name in names)
        joined = {key: torch.cat([
            F.pad(inputs[name][key], (0, width - inputs[name][key].shape[1]),
                  value=-100 if key == "labels" else 0) for name in names])
                  for key in ("input_ids", "labels", "attention_mask")}
        output = model(input_ids=joined["input_ids"], attention_mask=joined["attention_mask"])
        scores, counts, offset = {}, {}, 0
        for name in names:
            end = offset + inputs[name]["input_ids"].shape[0]
            labels = joined["labels"][offset:end]
            scores[name] = _seq_logp(output.logits[offset:end], labels)
            counts[name] = labels[:, 1:].ne(-100).sum()
            offset = end

        def ce(name):
            return -scores[name].sum() / counts[name]

        def logp(name):
            return scores[name]

        if self.method == "tooldelete_dpo":
            terms = {}
            for group in ("forget", "retain"):
                a, b = group + "_chosen", group + "_rejected"
                terms[group] = dpo_loss(logp(a), logp(b), inputs["ref_" + a],
                                        inputs["ref_" + b], self.beta)
            loss = self.forget_weight * terms["forget"] + self.retain_weight * terms["retain"]
        elif self.method == "retain_ft":
            loss = self.retain_weight * ce("retain")
        else:
            if self.method == "npo":
                loss = npo_loss(logp("forget"), inputs["ref_forget"], self.beta)
            elif self.method in ("ga", "graddiff"):
                loss = -ce("forget")
            else:
                loss = ce("forget")
            loss = self.forget_weight * loss
            if "retain" in inputs:
                loss = loss + self.retain_weight * ce("retain")
        return (loss, output) if return_outputs else loss
