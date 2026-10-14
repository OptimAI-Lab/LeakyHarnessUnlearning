"""Model loading for the local planner/actor evaluation harness."""
import os
import json
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

ACTOR_PATH = os.getenv("ACTOR_MODEL", "TangQiaoYu/ToolAlpaca-7B")
PLANNER_PATH = os.getenv("PLANNER_MODEL", "TangQiaoYu/ToolAlpaca-13B")

# Two layouts.  COLLAB_LAYOUT=split (default) is the original 2-card plan:
# GPU1 also hosts the fp16 actor, so the planner is only allowed a small slice
# of it; the bulk of the 13B lands on GPU0.  COLLAB_LAYOUT=single puts the whole
# 13B fp16 (26GB) and the 7B actor (13.5GB) on one 48GB A40 card (~40GB used),
# so two collab runs can sit on the two cards in parallel.
COLLAB_LAYOUT = os.getenv("COLLAB_LAYOUT", "split")
if COLLAB_LAYOUT == "single":
    PLANNER_MAX_MEMORY = {0: os.getenv("PLANNER_MEM0", "40GiB")}
else:
    PLANNER_MAX_MEMORY = {0: os.getenv("PLANNER_MEM0", "20GiB"),
                          1: os.getenv("PLANNER_MEM1", "6GiB")}
ACTOR_DEVICE = int(os.getenv("ACTOR_DEVICE", "1"))

# Actor inference backend.  "vllm" (default when ACTOR_SERVER_URL is set) sends
# generation to an external vLLM OpenAI-compatible server; "transformers" loads
# the 7B in-process.  The planner always stays on transformers, so on a single
# card the memory split is: vLLM actor (~16GB) + transformers planner (~26GB).
ACTOR_BACKEND = os.getenv("ACTOR_BACKEND",
                          "vllm" if os.getenv("ACTOR_SERVER_URL") else "transformers")
ACTOR_SERVER_URL = os.getenv("ACTOR_SERVER_URL", "http://127.0.0.1:8001/v1")
ACTOR_SERVED_NAME = os.getenv("ACTOR_SERVED_NAME", "actor")


class _TextStopCriteria(StoppingCriteria):
    """Stop after a complete textual stop marker in a one-item greedy decode.

    This is opt-in because the original harness deliberately generated the full
    256-token continuation and truncated it afterwards.  For complete-ReAct
    experiments, the agent discards everything following Observation anyway;
    ending generation as soon as that marker appears preserves the parsed model
    decision while avoiding hundreds of unused tokens per tool call.
    """

    def __init__(self, tokenizer, prompt_tokens, stops):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.stops = tuple(stops or ())

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0][self.prompt_tokens:], skip_special_tokens=True)
        if any(marker in text for marker in self.stops):
            return True
        # A ReAct executor needs only a syntactically complete Action Input to
        # execute the call.  Do not spend another ~200 tokens letting the model
        # invent an Observation that the real tool immediately replaces.  This
        # is opt-in alongside TOOLALPACA_EARLY_STOP and leaves Response-only
        # completions untouched so final answers are still generated normally.
        if os.getenv("TOOLALPACA_STOP_AFTER_ACTION_INPUT", "0") == "1":
            marker = "ASSISTANT Action Input:"
            if marker in text:
                payload = text.split(marker, 1)[1].lstrip()
                try:
                    json.JSONDecoder().raw_decode(payload)
                    return True
                except json.JSONDecodeError:
                    pass
        return False


def _truncate_stop(text, stop):
    # mirrors greedy_generate: generate to the cap, then cut at the first stop
    # string, byte-identical to langchain's HuggingFacePipeline behaviour.
    if stop:
        cuts = [text.find(s) for s in stop if text.find(s) != -1]
        if cuts:
            text = text[:min(cuts)]
    return text


class VLLMActorClient:
    """Actor served by an external vLLM OpenAI-compatible /v1 server."""

    def __init__(self, server_url=ACTOR_SERVER_URL, served_name=ACTOR_SERVED_NAME):
        self.server_url = server_url
        self.served_name = served_name

    def generate(self, prompt, max_new_tokens=256, stop=None):
        r = requests.post(
            self.server_url + "/completions",
            json={"model": self.served_name, "prompt": prompt,
                  "max_tokens": max_new_tokens, "temperature": 0.0, "top_p": 1.0},
            timeout=600,
        )
        r.raise_for_status()
        return _truncate_stop(r.json()["choices"][0]["text"], stop)


class TransformersActor:
    def __init__(self, tok, model):
        self.tok = tok
        self.model = model

    def generate(self, prompt, max_new_tokens=256, stop=None):
        return greedy_generate(self.tok, self.model, prompt,
                               max_new_tokens=max_new_tokens, stop=stop)


def _load_tokenizer(path):
    # ToolAlpaca ships a SentencePiece tokenizer; fast conversion is broken
    # against this pinned transformers/protobuf pair.
    if not path:
        raise ValueError("Set ACTOR_MODEL and PLANNER_MODEL to local checkpoint paths")
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


def load_actor():
    tok = _load_tokenizer(ACTOR_PATH)
    if ACTOR_BACKEND == "vllm":
        return tok, VLLMActorClient()
    model = AutoModelForCausalLM.from_pretrained(
        ACTOR_PATH, trust_remote_code=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(f"cuda:{ACTOR_DEVICE}")
    model.eval()
    return tok, TransformersActor(tok, model)


def load_planner():
    tok = _load_tokenizer(PLANNER_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        PLANNER_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        # NOT "auto": that routes through accelerate's get_balanced_memory, which
        # rewrites max_memory to even out the cards.  It cannot even them out here
        # (GPU1 is deliberately capped low because the actor sits on it), but it
        # still cuts GPU0 from 20GiB down to ~12.9GiB, leaving a 19.9GiB budget for
        # a 24.2GiB model -- the remainder silently lands on "disk" and
        # from_pretrained dies asking for an offload_folder.  "sequential" honours
        # max_memory literally and fills GPU0 first, which is what we want.
        device_map="sequential",
        max_memory=PLANNER_MAX_MEMORY,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


@torch.inference_mode()
def greedy_generate(tok, model, prompt, max_new_tokens=512, stop=None):
    """Deterministic completion, with manual stop-sequence truncation.

    `stop` mirrors what langchain's HuggingFacePipeline does: generation is not
    interrupted, the first occurrence is cut afterwards.  Keeping it identical
    across conditions matters more than saving the tokens.
    """
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    generation_kwargs = {}
    if os.getenv("TOOLALPACA_EARLY_STOP", "0") == "1" and stop:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_TextStopCriteria(tok, inputs["input_ids"].shape[1], stop)]
        )
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.eos_token_id,
        **generation_kwargs,
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if stop:
        cuts = [text.find(s) for s in stop if text.find(s) != -1]
        if cuts:
            text = text[:min(cuts)]
    return text
