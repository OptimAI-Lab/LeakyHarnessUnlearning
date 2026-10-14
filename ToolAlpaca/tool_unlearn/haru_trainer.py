"""Weighted GA/NPO objectives for HARU's failure-augmented training."""
import torch
import torch.nn.functional as F
from transformers import Trainer

from tool_unlearn.trainer import _seq_logp


def npo_loss(logp, reference_logp, beta):
    if beta <= 0:
        raise ValueError("beta must be positive")
    return -2.0 / beta * F.logsigmoid(beta * (reference_logp - logp)).mean()


class BaselineTrainer(Trainer):
    def __init__(self, *args, method, beta=0.1, forget_weight=1.0,
                 retain_weight=1.0, **kwargs):
        self.method = method
        self.beta = beta
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        names = [name for name in inputs if not name.startswith("ref_")]
        width = max(inputs[name]["input_ids"].shape[1] for name in names)
        joined = {key: torch.cat([
            F.pad(inputs[name][key], (0, width - inputs[name][key].shape[1]),
                  value=-100 if key == "labels" else 0) for name in names])
                  for key in ("input_ids", "labels", "attention_mask")}
        output = model(input_ids=joined["input_ids"],
                       attention_mask=joined["attention_mask"])
        scores, counts, offset = {}, {}, 0
        for name in names:
            end = offset + inputs[name]["input_ids"].shape[0]
            labels = joined["labels"][offset:end]
            scores[name] = _seq_logp(output.logits[offset:end], labels)
            counts[name] = labels[:, 1:].ne(-100).sum()
            offset = end

        def ce(name):
            return -scores[name].sum() / counts[name]

        if self.method == "npo":
            loss = npo_loss(scores["forget"], inputs["ref_forget"], self.beta)
        elif self.method in ("ga", "graddiff"):
            loss = -ce("forget")
        else:
            raise ValueError("Unknown HARU objective: %s" % self.method)
        loss = self.forget_weight * loss
        if "retain" in inputs:
            loss = loss + self.retain_weight * ce("retain")
        return (loss, output) if return_outputs else loss


class HaruTrainer(BaselineTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget, retain = inputs
        forget = dict(forget)
        reference = forget.pop("ref_seq_logp", None)
        streams = {"forget": forget}
        if reference is not None:
            streams["ref_forget"] = reference
        if retain is not None:
            streams["retain"] = retain
        # BaselineTrainer uses one combined forward for forget and retain, so
        # gradient checkpointing does not re-enter the same DDP layers twice.
        return super().compute_loss(model, streams, return_outputs, **kwargs)
