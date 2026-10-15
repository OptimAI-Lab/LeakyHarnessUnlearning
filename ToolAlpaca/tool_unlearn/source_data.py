"""Reconstruct frozen HARU sources without experiment-launcher dependencies."""
import copy
import json
import random
import time

import requests

from build_dataset import build_dataset
from tool_unlearn.prepare_data import surviving_instances, subset_api


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def training_indices(api, eval_ratio, split_seed):
    indices = surviving_instances(api)
    random.Random("%s/%d" % (api["Name"], split_seed)).shuffle(indices)
    n_eval = (min(len(indices) - 1,
                  max(1, int(round(len(indices) * eval_ratio))))
              if len(indices) > 1 else 0)
    return sorted(indices[n_eval:])


def reconstruct_sources(forget_file, api_file, split_file):
    """Recover the API and original instance behind each frozen forget row."""
    forget_rows = read_json(forget_file)
    api_data = read_json(api_file)
    split = read_json(split_file)
    forget_tools = set(split["forget_tools"])
    expected_rows = []
    sources = []

    for api_index, api in enumerate(api_data):
        if api.get("Function_Description") is None or api["Name"] not in forget_tools:
            continue
        indices = training_indices(api, split["eval_ratio"], split["seed"])
        if not indices:
            continue
        rebuilt = build_dataset(subset_api(api, indices))
        if len(rebuilt) != len(indices):
            raise RuntimeError("Training split reconstruction drifted for %s" % api["Name"])
        route_name = "forget_aug_api_%05d" % api_index
        routed_api = copy.deepcopy(api)
        routed_api["Name"] = route_name
        for instance_index, (process, trainable) in zip(indices, rebuilt):
            expected_rows.append({
                "tool": api["Name"], "process": process, "trainable": trainable,
            })
            instance = copy.deepcopy(api["Instances"][instance_index])
            question = instance["input"].rsplit("\nHint: ", 1)[0]
            sources.append({
                "api_index": api_index,
                "route_name": route_name,
                "tool": api["Name"],
                "api": routed_api,
                "original_api": api,
                "instance": instance,
                "question": question,
            })

    if expected_rows != forget_rows:
        mismatch = next((i for i, pair in enumerate(zip(expected_rows, forget_rows))
                         if pair[0] != pair[1]), None)
        raise RuntimeError(
            "forget_train.json differs from the frozen split reconstruction "
            "(rebuilt=%d, file=%d, first mismatch=%s)"
            % (len(expected_rows), len(forget_rows), mismatch)
        )
    if len(sources) != len(forget_rows):
        raise RuntimeError("Source reconstruction count mismatch")
    return forget_rows, sources


def unique_simulator_apis(sources):
    by_route = {}
    for source in sources:
        by_route.setdefault(source["route_name"], source["api"])
    return list(by_route.values())


def wait_for_server(url, process, timeout=180):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Simulator exited with code %s" % process.returncode)
        try:
            response = requests.get(url + "/__simulator_status__", timeout=2)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError("Simulator did not become ready: %s" % last_error)


def preflight_remote_model(model):
    """Verify simulator access before loading generation checkpoints."""
    from utils import openai_chat_completions
    response = openai_chat_completions(
        [{"role": "user", "content": "Reply OK."}], model=model,
        temperature=0, max_tokens=2, max_retries=1, num_workers=1,
    )
    if not response.get("choices"):
        raise RuntimeError("Remote model %s returned no choices" % model)
