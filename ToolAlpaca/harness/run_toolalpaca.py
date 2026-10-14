"""Run the ToolAlpaca simulated benchmark under one harness condition.

Conditions:

  B0   actor=7B,  mode=vanilla   official single-model ReAct baseline
  B1   actor=7B,  mode=neutral   wrapper control: Thought slot filled with a
                                 constant, task-independent placeholder
  H1   actor=7B,  mode=planner   Thought slot written by the 13B planner, whose
                                 own tool call is removed by truncation
  H2   actor=7B,  mode=planner_instructed
                                 same, but the planner is told it is advising and
                                 that the assistant makes the final call.  It may
                                 name the tool it thinks fits -- the traces show
                                 tool selection is where the 13B adds value.  The
                                 7B is still the only thing that can emit an
                                 Action (enforced by truncation, not by wording).
  S13  actor=13B, mode=vanilla   planner's own ceiling, run alone

Everything else -- simulator, cache policy, agent prompt, decoding, judge -- is
held fixed so B0/B1/H1 are strictly paired on the same task ids.
"""
import argparse
import json
import logging
import os
import sys

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.get_agent import get_agent
from agent.agent_prompts import prompt_proj
from utils import load_openapi_spec, analyze_openapi_spec
from harness import models
from harness.planner_llm import PlannerAssistedLLM

logger = logging.getLogger(__name__)

CONDITIONS = {
    "B0":  {"actor": "7b",  "mode": "vanilla"},
    "B1":  {"actor": "7b",  "mode": "neutral"},
    "H1":  {"actor": "7b",  "mode": "planner"},
    "H2":  {"actor": "7b",  "mode": "planner_instructed"},
    "S13": {"actor": "13b", "mode": "vanilla"},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    p.add_argument("-api", "--api_data_path", default="./data/eval_simulated.json")
    p.add_argument("-out", "--output_dir", default="./outputs/harness")
    p.add_argument("--server_url", default="http://127.0.0.1:5678")
    p.add_argument("--agent_prompt", default="test_v1")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--length", type=int, default=-1)
    p.add_argument("--instruction_offset", type=int, default=0)
    p.add_argument("--instruction_length", type=int, default=-1)
    p.add_argument("--use_cache", action="store_true", default=True)
    p.add_argument("--without_getDetails", action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=False,
                   help="skip API groups already fully processed in the output file")
    args = p.parse_args()

    cfg = CONDITIONS[args.condition]
    os.makedirs(args.output_dir, exist_ok=True)

    print("[harness] condition=%s actor=%s mode=%s" % (args.condition, cfg["actor"], cfg["mode"]))
    if cfg["actor"] == "7b":
        actor_tok, actor_model = models.load_actor()
    elif cfg["actor"] == "13b":
        actor_tok, raw_actor_model = models.load_planner()  # 13B, sharded
        # PlannerAssistedLLM calls a common ``generate(prompt, stop=...)``
        # interface.  The raw Transformers model instead expects token IDs;
        # adapt it exactly as the local 7B actor is adapted in models.load_actor.
        actor_model = models.TransformersActor(actor_tok, raw_actor_model)
    else:
        actor_tok = actor_model = None
    planner_tok = planner_model = None
    if cfg["mode"] in ("planner", "planner_instructed") and not cfg.get("remote_planner"):
        planner_tok, planner_model = models.load_planner()

    llm = PlannerAssistedLLM(
        actor_tok=actor_tok, actor_model=actor_model,
        planner_tok=planner_tok, planner_model=planner_model,
        mode=cfg["mode"],
    )

    api_data = json.load(open(args.api_data_path))
    if args.length == -1:
        args.length = len(api_data) - args.offset
    api_data = api_data[args.offset:args.offset + args.length]

    out_path = os.path.join(args.output_dir, "%s.json" % args.condition)
    trace_path = os.path.join(args.output_dir, "%s_trace.json" % args.condition)

    # Resume: keep API groups already fully processed in a previous run.
    done = {}
    if args.resume and os.path.exists(out_path):
        try:
            for a in json.load(open(out_path)):
                if a.get("Instances"):
                    done[a["Name"]] = a
        except Exception as e:          # corrupt/partial output -> start fresh
            logger.error("resume: could not load %s (%s)", out_path, e)
            done = {}
        if done:
            print("[harness] resume: %d/%d API groups already done, skipping them"
                  % (len(done), len(api_data)))

    all_traces = []
    if args.resume and os.path.exists(trace_path):
        try:
            all_traces = json.load(open(trace_path))
            keep = set(done)
            all_traces = [t for t in all_traces if t.get("api") in keep]
        except Exception:
            all_traces = []

    if args.use_cache:
        requests.get("%s/__simulator_cache__/open" % args.server_url)

    for api in tqdm(api_data):
        if api["Name"] in done:
            api["Instances"] = done[api["Name"]]["Instances"]
            continue
        api["Instances"] = []
        if not api.get("Instructions"):
            continue
        spec = load_openapi_spec(api["Documentation"])
        in_ok, out_ok = analyze_openapi_spec(spec)
        if not (in_ok and out_ok):
            continue

        agent = get_agent(
            llm=llm, api_data=api, server_url=args.server_url,
            agent_prompt=prompt_proj[args.agent_prompt],
            enable_getDetails=not args.without_getDetails,
        )

        end = None
        if args.instruction_length != -1:
            end = args.instruction_offset + args.instruction_length
        selected = api["Instructions"][args.instruction_offset:end]

        answers = []
        for idx, inst in enumerate(selected, start=args.instruction_offset):
            if api.get("Authentication"):
                inst += "\nAuthentication information: " + " ".join(
                    "%s=%s" % (k, v) for k, v in api["Authentication"].items())
            llm.trace = []
            agent.reset_execution_record()
            try:
                output = agent(inst)
                json.dumps(output, ensure_ascii=False)
            except json.JSONDecodeError:
                output = str(output)
            except Exception as e:                     # noqa: BLE001 - mirrors upstream
                logger.error(e)
                # Do not turn an exception after one or more tool calls into a
                # false “no tool call”.  The captured calls are serializable in
                # the same format as AgentExecutor's normal return value.
                output = {"error": str(e), **agent.execution_record()}

            golden = api.get("Golden_Answers") or []
            ga = golden[idx] if idx < len(golden) else None
            all_traces.append({
                "condition": args.condition, "api": api["Name"],
                "instruction_id": idx, "instruction": inst,
                "golden_answers": ga,
                "steps": list(llm.trace),
                "output": output if isinstance(output, dict) else {"raw": output},
                "failed": isinstance(output, dict) and "error" in output,
                "execution_record": agent.execution_record(),
            })
            if args.use_cache:
                requests.get("%s/__simulator_cache__/clear/%s" % (args.server_url, api["Name"]))
            answers.append(output)

        api["Instances"] = answers
        json.dump(api_data, open(out_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        json.dump(all_traces, open(trace_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # During a resumed run, the last newly processed API can precede groups
    # restored from the existing output. Flush once more after the loop so
    # those later groups are retained on disk too.
    if args.resume:
        json.dump(api_data, open(out_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        json.dump(all_traces, open(trace_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    n_leak = sum(1 for t in all_traces for s in t["steps"] if s.get("planner_leaked"))
    print("[harness] %s done: %d tasks, %d failed, %d planner steps leaked a call"
          % (args.condition, len(all_traces),
             sum(t["failed"] for t in all_traces), n_leak))
    print("[harness] wrote %s and %s" % (out_path, trace_path))


if __name__ == "__main__":
    main()
