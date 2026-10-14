import os
import re
import json
import argparse
import hashlib
from pathlib import Path
from string import Template

from utils import openai_chat_completions


def parse_verdict(completion):
    """Accept the original verdict vocabulary; API/format failures are not No."""
    try:
        choice = completion["choices"][0]
        text = choice["message"]["content"]
        if not isinstance(text, str) or not text or choice.get("finish_reason") in ("length", "content_filter"):
            return None
    except (TypeError, KeyError, IndexError):
        return None
    results = text.split("## Results", 1)[-1]
    verdicts = []
    for label in ("Process Correctness", "Final Response Correctness"):
        matches = re.findall(rf"{label}: (\w+)", results)
        if len(matches) != 1 or matches[0] not in ("Yes", "No", "Uncertain"):
            return None
        verdicts.append(matches[0])
    return text, *verdicts


def update_metrics(statistics):
    """Paper process accuracy uses all evaluated tasks, including empty traces."""
    n = statistics["num"]
    statistics["complete"] = (
        not statistics.get("judge_errors")
        and not statistics.get("rollout_errors")
        and n == statistics.get("expected_num", n)
    )
    denominator = n if statistics["complete"] else 0
    statistics["primary_metric"] = "process_accuracy"
    statistics["process_accuracy"] = statistics["process"]["Yes"] / denominator if denominator else None
    statistics["response_accuracy"] = statistics["response"]["Yes"] / denominator if denominator else None
    statistics["overall_accuracy"] = statistics["both"] / denominator if denominator else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp", "--template_path", type=str, default="./prompts/Evaluation.txt")
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-gold", "--golden_answer_path", type=str, default="")
    parser.add_argument("-out", "--output_path", type=str, default="")
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4-0613",
        help="original GPT-4 judge; independent of the simulator model setting",
    )
    parser.add_argument("--instruction_offset", type=int, default=0)
    parser.add_argument("--continue_run", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int,
                        default=int(os.getenv("JUDGE_WORKERS", "16")),
                        help="concurrent judge requests")
    args = parser.parse_args()

    with open(args.template_path, "r") as stream:
        template = Template(stream.read())

    with open(args.api_data_path, "r") as stream:
        api_data = json.load(stream)
    if os.path.exists(args.golden_answer_path):
        with open(args.golden_answer_path, "r") as stream:
            golden_answer = json.load(stream)
        for k, v in zip(api_data, golden_answer):
            k["Golden_Answers"] = v["Golden_Answers"]

    original_data = {}
    original_data["statistics"] = {
        "num": 0,
        "error_num": 0,
        "process": {
            "Yes": 0,
            "No": 0,
            "Uncertain": 0
        },
        "response": {
            "Yes": 0,
            "No": 0,
            "Uncertain": 0
        },
        "both": 0
    }

    exist_ids = None
    if args.continue_run:
        with open(args.output_path, "r") as stream:
            original_data = json.load(stream)
        previous_model = original_data["statistics"].get("judge_model")
        if previous_model != args.judge_model:
            raise ValueError("Cannot resume judgments without matching judge_model provenance")
        exist_ids = {i: [j["id"] for j in original_data[i]] for i in original_data if i != "statistics"}

    original_data["statistics"]["judge_model"] = args.judge_model
    original_data["statistics"]["evaluator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_data["statistics"]["template_sha256"] = hashlib.sha256(Path(args.template_path).read_bytes()).hexdigest()
    original_data["statistics"]["input_sha256"] = hashlib.sha256(Path(args.api_data_path).read_bytes()).hexdigest()
    if os.path.exists(args.golden_answer_path):
        original_data["statistics"]["golden_answer_sha256"] = hashlib.sha256(
            Path(args.golden_answer_path).read_bytes()).hexdigest()
    original_data["statistics"]["expected_num"] = sum(len(api["Instances"]) for api in api_data)
    original_data["statistics"]["judge_errors"] = []
    original_data["statistics"]["rollout_errors"] = []

    def save_results():
        update_metrics(original_data["statistics"])
        with open(args.output_path, "w") as stream:
            json.dump(original_data, stream, indent=4, ensure_ascii=False)

    retry_cases = None

    # ------------------------------------------------------------------ pass 1
    # Build the whole work list before touching the API.  Judging used to be one
    # blocking call per item, which left the endpoint idle between requests; the
    # prompts do not depend on each other, so they can all be prepared up front
    # and then issued concurrently.
    pending = []                      # (api_name, ques_id, messages)
    for api_info in api_data:
        api_name = api_info.get("Name", api_info.get("API"))
        if retry_cases is not None and api_name not in retry_cases:
            continue
        if exist_ids is None or api_name not in exist_ids:
            original_data[api_name] = []
        for local_ques_id, instance in enumerate(api_info["Instances"]):
            ques_id = args.instruction_offset + local_ques_id
            ques = api_info["Instructions"][ques_id]
            if exist_ids is not None and ques_id in exist_ids.get(api_name, []):
                continue
            if retry_cases is not None and ques_id not in retry_cases.get(api_name, []):
                continue

            rollout_error = str(instance.get("error", ""))
            if re.search(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
                         rollout_error, re.IGNORECASE):
                original_data["statistics"]["rollout_errors"].append({
                    "api": api_name, "id": ques_id, "error": rollout_error,
                })
                save_results()
                raise RuntimeError("GPU memory failure in rollout; rerun the affected rollout before judging.")

            if "intermediate_steps" not in instance or len(instance["intermediate_steps"]) == 0:
                # Generation failed outright: no trajectory to judge.  Counted into
                # `num` (so it lands in the denominator) and into `error_num`.
                original_data["statistics"]["num"] += 1
                original_data["statistics"]["error_num"] += 1
                original_data[api_name].append({"id": ques_id, "input": "", "output": ""})
                continue

            golden_answer = api_info["Golden_Answers"][ques_id]
            standard_answer = ""
            for ans_id, ans in enumerate(golden_answer):
                standard_answer += f"{ans_id + 1}. Function: {ans['Action']}\nParameters: {ans['Action_Input']}\n"

            solution = ""
            for sol_id, sol in enumerate(instance["intermediate_steps"]):
                solution += f"{sol_id + 1}. Function: {sol[0][0]}\nParameters: {sol[0][1]}\nRetruns: {sol[1]}\n"
            # Parsing may fail after valid tool steps, leaving no final response.
            # Judge the saved partial trajectory without inventing an answer.
            solution += f"{sol_id + 2}. Final Response: {instance.get('output', '')}"

            prompt = template.substitute(
                documentation=api_info["NLDocumentation"],
                instruction=ques,
                standard=standard_answer,
                solution=solution
            )
            pending.append((api_name, ques_id, [{"role": "user", "content": prompt}]))

    # ------------------------------------------------------------------ pass 2
    W = max(1, args.num_workers)
    print(f"[judge] {len(pending)} items to judge, {W} at a time", flush=True)

    for start in range(0, len(pending), W):
        chunk = pending[start:start + W]
        batch = [m for _, _, m in chunk]

        outs = [None] * len(batch)
        verdicts = [None] * len(batch)
        for _attempt in range(3):
            bad = [i for i, verdict in enumerate(verdicts) if verdict is None]
            if not bad:
                break
            try:
                responses = openai_chat_completions(
                    [batch[i] for i in bad], model=args.judge_model,
                    temperature=0.2, num_workers=min(W, len(bad)))
                if len(responses) != len(bad):
                    raise RuntimeError("Judge returned the wrong number of completions")
            except Exception as exc:
                original_data["statistics"]["judge_errors"].append({
                    "items": [[chunk[i][0], chunk[i][1]] for i in bad],
                    "error": str(exc),
                })
                save_results()
                raise
            for i, response in zip(bad, responses):
                outs[i] = response
                verdicts[i] = parse_verdict(response)

        for (api_name, ques_id, messages), verdict, raw in zip(chunk, verdicts, outs):
            if verdict is None:
                original_data["statistics"]["judge_errors"].append({
                    "api": api_name, "id": ques_id,
                    "error": "Invalid judge response after 3 attempts",
                    "completion": raw,
                })
                continue
            text, pc, rc = verdict

            original_data["statistics"]["num"] += 1
            original_data["statistics"]["process"][pc] += 1
            original_data["statistics"]["response"][rc] += 1
            if pc == rc == "Yes":
                original_data["statistics"]["both"] += 1

            original_data[api_name].append({
                "id": ques_id,
                "input": messages,
                "output": text,
                "process_correctness": pc,
                "final_response_correctness": rc,
            })

        # Flush after every chunk so --continue_run can pick up from here.
        save_results()
        if original_data["statistics"]["judge_errors"]:
            raise RuntimeError("Judge failures saved; run is incomplete. Resume with --continue_run.")
        print(f"[judge] {min(start + W, len(pending))}/{len(pending)}", flush=True)

    # Items that failed generation are appended during pass 1 and judged items
    # during pass 2, so the per-API lists come out interleaved.  Nothing reads
    # these by position -- every record carries its own "id" -- but keep them in
    # id order so the files stay diffable against the sequential-era outputs.
    for k in original_data:
        if k != "statistics":
            original_data[k].sort(key=lambda it: it["id"])

    save_results()
    stats = original_data["statistics"]
    print(f"[judge] done: process_accuracy={stats['process_accuracy']} "
          f"({stats['process']['Yes']}/{stats['num']}); "
          f"overall_accuracy={stats['overall_accuracy']}", flush=True)
