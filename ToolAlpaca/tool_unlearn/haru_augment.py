"""Mine actual actor calls under planner reasoning; no judge or SFT targets."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any

from tool_unlearn.haru import ROOT, command_for_stage, read_json, write_json

ACTION = re.compile(
    r"^[ \t]*ASSISTANT[ \t]+Action[ \t]*\d*[ \t]*:([^\r\n]*)\r?\n"
    r"[ \t]*ASSISTANT[ \t]+Action[ \t]*\d*[ \t]*Input[ \t]*\d*[ \t]*:[ \t]*(.*)",
    re.DOTALL | re.MULTILINE)
DECISION = re.compile(r"^[ \t]*ASSISTANT[ \t]+(?:Action[ \t]*\d*|Response)[ \t]*:", re.MULTILINE)
NEXT_TURN = re.compile(r"^[ \t]*ASSISTANT[ \t]+(?:Action[ \t]*\d*|Response|Thought|Observation)[ \t]*:", re.MULTILINE)


def failure_row(tool, forgotten_actions, actor_prompt, actor_output, first_action_only=True):
    """Match the executor's Action/Action Input protocol, not names in prose.

    API names are the forget units; function names are resolved within that API
    to avoid confusing identically named operations in retained APIs. getDetails
    is a shared helper, not an operation belonging to the forgotten API.
    """
    if not first_action_only and "ASSISTANT Response:" in actor_output:
        return None  # The existing executor gives a final response precedence.
    match = ACTION.search(actor_output)
    if not match or match.group(1).strip() == "getDetails":
        return None
    decision = DECISION.search(actor_output)
    if first_action_only and (decision is None or decision.start() != match.start()):
        return None
    if match.group(1).strip() not in forgotten_actions:
        return None
    # Exclude hallucinated observations, subsequent actions and final answers.
    next_turn = NEXT_TURN.search(actor_output, match.start(2))
    end = next_turn.start() if next_turn else len(actor_output)
    return {"tool": tool,
            "process": [actor_prompt + actor_output[:match.start()], actor_output[match.start():end].rstrip()],
            "trainable": [False, True], "haru_action_only": True}


def merge_failures(original, attempts):
    # Keep every sampled failure, including repeats, as requested. The original
    # rows retain their exact segmentation and masks.
    return list(original) + [row for attempt in attempts for row in attempt["failure_rows"]]


def generation_gpu_groups(args):
    """Preserve scheduler visibility, assigning disjoint GPU groups to workers."""
    if args.cpu:
        return [None] * args.num_generation_workers
    indices = []
    for device in (args.planner_device, args.actor_device):
        match = re.fullmatch(r"cuda:(\d+)", device)
        if not match:
            raise ValueError("GPU generation devices must use cuda:N within each worker group")
        indices.append(int(match.group(1)))
    width = max(indices) + 1
    visible = args.generation_gpus
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [part.strip() for part in visible.split(",") if part.strip()]
    elif args.dry_run:
        devices = [str(i) for i in range(args.num_train_processes)]
    else:
        import torch
        devices = [str(i) for i in range(torch.cuda.device_count())]
    if len(devices) != len(set(devices)) or "-1" in devices:
        raise ValueError("generation_gpus must contain distinct, available GPU IDs")
    needed = args.num_generation_workers * width
    if len(devices) < needed:
        raise ValueError("%d generation workers need %d visible GPUs, got %d; adjust "
                         "--generation_gpus or --num_generation_workers" %
                         (args.num_generation_workers, needed, len(devices)))
    return [",".join(devices[i:i + width]) for i in range(0, needed, width)]


def sample_jobs(num_sources, num_generations, worker_id=0, num_workers=1):
    if num_workers < 1 or not 0 <= worker_id < num_workers:
        raise ValueError("Invalid generation worker ID/count")
    for flat_index in range(worker_id, num_sources * num_generations, num_workers):
        yield divmod(flat_index, num_generations)


def collect_attempts(output, num_sources, num_generations, num_workers):
    """Refuse partial/duplicate results before publishing the new training set."""
    indexed = {}
    for worker_id in range(num_workers):
        path = output / ("worker-%02d" % worker_id) / "attempts.jsonl"
        expected = set(sample_jobs(num_sources, num_generations, worker_id, num_workers))
        found = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                attempt = json.loads(line)
                key = (attempt["source_id"], attempt["sample_id"])
                if key not in expected or key in indexed:
                    raise ValueError("Unexpected/duplicate generation result %s in %s" % (key, path))
                found.add(key)
                indexed[key] = attempt
        if found != expected:
            raise ValueError("Incomplete generation worker %d: %d/%d samples" %
                             (worker_id, len(found), len(expected)))
    return [indexed[key] for key in sorted(indexed)]


def run_generation_workers(args, output, groups):
    """Load replicas sequentially to bound host RAM, then generate in parallel."""
    processes, logs = [], []

    def check_workers():
        for worker_id, process in enumerate(processes):
            code = process.poll()
            if code is not None and code != 0:
                raise RuntimeError("Generation worker %d failed (%d); see %s" %
                                   (worker_id, code, output / ("worker-%02d.log" % worker_id)))

    try:
        for worker_id, gpus in enumerate(groups):
            worker_args = copy.copy(args)
            worker_args._generation_worker_id = worker_id
            env = dict(os.environ)
            if gpus is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpus
            env["TOKENIZERS_PARALLELISM"] = "false"
            log = open(output / ("worker-%02d.log" % worker_id), "w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen(command_for_stage(worker_args, "augment_worker"),
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            print("[HARU] loading generation worker %d on %s" % (worker_id, gpus or "CPU"), flush=True)
            ready = output / ("worker-%02d" % worker_id) / "ready.json"
            deadline = time.monotonic() + 900
            while not ready.exists():
                check_workers()
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Worker %d did not load its models; see its log" % worker_id)
                time.sleep(1)
            check_workers()
            print("[HARU] generation worker %d ready" % worker_id, flush=True)
        last_report = 0
        while any(process.poll() is None for process in processes):
            check_workers()
            if time.monotonic() - last_report >= 20:
                completed = 0
                for worker_id in range(len(groups)):
                    progress = output / ("worker-%02d" % worker_id) / "progress.json"
                    if progress.exists():
                        completed += read_json(progress)["completed"]
                print("[HARU] parallel generation: %d samples completed" % completed, flush=True)
                last_report = time.monotonic()
            time.sleep(1)
        check_workers()
    finally:
        # Never leave other GPU workers running after one worker fails.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


class ContextExhausted(ValueError):
    pass


class SamplingModel:
    def __init__(self, path, device, temperature, top_p, tokenizer_path=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            low_cpu_mem_usage=True).to(device).eval()
        self.model.config.use_cache = True
        self.device, self.temperature, self.top_p = device, temperature, top_p
        self.calls = []

    def generate(self, prompt, max_new_tokens=256, stop=None, stop_after_action=True):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        length = inputs["input_ids"].shape[1]
        # Training in this repository is capped at 2048 as well.
        context = min(2048, int(getattr(self.model.config, "max_position_embeddings", 2048)))
        if context - length < 2:
            raise ContextExhausted("Prompt uses %d of %d tokens" % (length, context))
        tokenizer = self.tokenizer

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0][length:], skip_special_tokens=True)
                if any(marker in text for marker in (stop or [])):
                    return True
                match = ACTION.search(text) if stop_after_action else None
                if match:
                    try:
                        json.JSONDecoder().raw_decode(match.group(2).lstrip())
                        return True
                    except json.JSONDecodeError:
                        pass
                return False

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=min(max_new_tokens, context - length),
                do_sample=True, temperature=self.temperature, top_p=self.top_p,
                num_beams=1, pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Stop()]))
        text = self.tokenizer.decode(generated[0][length:], skip_special_tokens=True)
        cuts = [text.index(marker) for marker in (stop or []) if marker in text]
        if cuts:
            text = text[:min(cuts)]
        self.calls.append({"prompt": prompt, "output": text})
        return text


def assisted_llm(actor, planner, args):
    from harness.planner_llm import PlannerAssistedLLM, split_planner_output

    class SamplingPlannerLLM(PlannerAssistedLLM):
        planner_runner: Any = None

        def _plan(self, prompt, stop, instructed=False):
            raw = self.planner_runner.generate(
                prompt, max_new_tokens=self.planner_max_new_tokens,
                stop=list(stop or []) + ["ASSISTANT Action:", "ASSISTANT Response:"],
                stop_after_action=False)
            return (raw,) + split_planner_output(raw)

    return SamplingPlannerLLM(
        actor_tok=actor.tokenizer, actor_model=actor, planner_runner=planner,
        mode="planner", max_new_tokens=args.max_new_tokens,
        planner_max_new_tokens=args.planner_max_new_tokens)


@contextmanager
def simulator(args, sources, output):
    if args.generation_mode == "first_action":
        yield None
        return
    import requests
    from tool_unlearn.source_data import unique_simulator_apis, wait_for_server

    # Refuse an occupied port before polling; otherwise an unrelated simulator
    # could answer the health check and receive calls for the wrong API spec.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.simulator_port))
    api_path = output / "simulator_apis.json"
    write_json(api_path, unique_simulator_apis(sources))
    url = "http://127.0.0.1:%d" % args.simulator_port
    command = [sys.executable, str(ROOT / "instance_generation/simulator.py"),
               "-api", str(api_path), "--port", str(args.simulator_port),
               "--model", args.simulator_model]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with open(output / "simulator.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_server(url, process)
            requests.get(url + "/__simulator_cache__/close", timeout=10).raise_for_status()
            yield url
            status = requests.get(url + "/__simulator_status__", timeout=10)
            status.raise_for_status()
            if status.json().get("failures", 0):
                raise RuntimeError("Simulator failures occurred; inspect augmentation/simulator.log")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def mine(args, sources, actor, planner, server_url, output, worker_id=0, num_workers=1):
    from transformers import set_seed
    from agent.get_agent import get_agent
    from agent.agent_prompts import prompt_proj

    llm = assisted_llm(actor, planner, args)
    attempts = []
    assigned = set(sample_jobs(len(sources), args.num_generations, worker_id, num_workers))
    with open(output / "attempts.jsonl", "w", encoding="utf-8", buffering=1) as log:
        for source_id, source in enumerate(sources):
            forgotten = set(source["original_api"]["Function_Projection"])
            for sample_id in range(args.num_generations):
                if (source_id, sample_id) not in assigned:
                    continue
                # getDetails carries chat history; construct a fresh executor
                # for every independent sample rather than leaking that state.
                agent = get_agent(llm, source["api"], server_url,
                                  agent_prompt=prompt_proj["test_v1"], enable_getDetails=True)
                agent.verbose = False
                agent.max_iterations = args.max_tool_steps
                seed = args.seed + source_id * args.num_generations + sample_id
                set_seed(seed)
                llm.trace = []
                actor.calls = []
                planner.calls = []
                agent.reset_execution_record()
                error = None
                try:
                    if args.generation_mode == "first_action":
                        agent.agent.plan([], input=source["question"])
                    else:
                        agent(source["question"])
                except ContextExhausted as exc:
                    error = "context_exhausted: %s" % exc
                except ValueError as exc:
                    # A malformed generation is auditable but is not a
                    # successful forgetting result. Keep any earlier failures.
                    if "Could not parse LLM output" not in str(exc):
                        raise
                    error = "parse_error: %s" % exc
                rows = []
                for call, trace in zip(actor.calls, llm.trace):
                    row = failure_row(source["tool"], forgotten, call["prompt"], call["output"],
                                      first_action_only=args.generation_mode == "first_action")
                    if row is not None:
                        rows.append(row)
                    trace["actor_prompt"] = call["prompt"]
                attempt = {"source_id": source_id, "sample_id": sample_id, "seed": seed,
                           "tool": source["tool"], "query": source["question"],
                           "forget_called": bool(rows), "error": error,
                           "steps": llm.trace, "failure_rows": rows, "judge": "regex_exact_match"}
                log.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                attempts.append(attempt)
                write_json(output / "progress.json", {"completed": len(attempts), "total": len(assigned)})
                print("[HARU] query=%d/%d sample=%d/%d forget_calls=%d%s" % (
                    source_id + 1, len(sources), sample_id + 1, args.num_generations,
                    len(rows), " incomplete" if error else ""), flush=True)
    return attempts


def augmentation_worker(args):
    # The shared getDetails helper also uses the simulator endpoint/model.
    os.environ.setdefault("OPENAI_DEFAULT_MODEL", args.simulator_model)
    from tool_unlearn.source_data import reconstruct_sources

    _, sources = reconstruct_sources(args.forget_file, args.api_file, args.split_file)
    worker_id = args._generation_worker_id
    if worker_id is None or not 0 <= worker_id < args.num_generation_workers:
        raise ValueError("Invalid generation worker ID")
    run_dir = Path(args.out_dir).resolve()
    output = run_dir / "augmentation" / ("worker-%02d" % worker_id)
    output.mkdir(parents=True, exist_ok=False)
    actor = SamplingModel(str(run_dir / "initial"), args.actor_device,
                          args.temperature, args.top_p, args.tokenizer_dir)
    planner = SamplingModel(args.planner_model_dir, args.planner_device,
                            args.planner_temperature, args.top_p)
    write_json(output / "ready.json", {"worker_id": worker_id})
    with simulator(args, sources, output) as server_url:
        mine(args, sources, actor, planner, server_url, output,
             worker_id=worker_id, num_workers=args.num_generation_workers)


def augment(args):
    from tool_unlearn.source_data import reconstruct_sources

    original, sources = reconstruct_sources(args.forget_file, args.api_file, args.split_file)
    groups = generation_gpu_groups(args)
    output = Path(args.out_dir).resolve() / "augmentation"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "workers.json", {"gpu_groups": groups, "judge": "regex_exact_match"})
    run_generation_workers(args, output, groups)
    attempts = collect_attempts(output, len(sources), args.num_generations, len(groups))
    with (output / "attempts.jsonl").open("w", encoding="utf-8") as log:
        for attempt in attempts:
            log.write(json.dumps(attempt, ensure_ascii=False) + "\n")
    augmented = merge_failures(original, attempts)
    write_json(output.parent / "forget_augmented.json", augmented)
    write_json(output / "summary.json", {
        "queries": len(sources), "generations_per_query": args.num_generations,
        "workers": len(groups), "judge": "regex_exact_match",
        "attempts": len(attempts), "failed_unlearning_attempts": sum(a["forget_called"] for a in attempts),
        "incomplete_attempts": sum(a["error"] is not None for a in attempts),
        "original_rows": len(original), "added_rows": len(augmented) - len(original),
        "total_rows": len(augmented)})
    print("[HARU] augmented dataset: %d original + %d failures = %d total rows -> %s" % (
        len(original), len(augmented) - len(original), len(augmented),
        output.parent / "forget_augmented.json"), flush=True)
