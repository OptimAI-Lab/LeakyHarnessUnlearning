"""Build the frozen forget / retain split, each with its own train and eval half.

ToolDelete (arXiv 2502.01083) reports three numbers, and they come from three
DIFFERENT sets of data:

    T_T (up)    the held-out ToolAlpaca test set -- APIs the model never saw at
                all.  This is `data/eval_simulated.json`, and the paper's
                "Original 60.0" is exactly this repo's baseline Overall score.
    T_r (up)    demonstrations of the RETAINED training tools.
    T_f (down)  demonstrations of the FORGOTTEN training tools.  The paper's
                Original T_f is 75.7, higher than T_T=60.0, precisely because
                these are training tools the model has memorised.

So the unlearning data and the evaluation data must not be the same rows: if
T_f were measured on the exact rows gradient ascent was run on, it would report
optimisation success, not forgetting.  Every tool's demonstrations are therefore
split into a train half (used to unlearn) and an eval half (never trained on).

The tool-level partition itself follows the paper's section 2:

    D_f = {T_f, Q_f, Y_f}   k < N tools + ALL their demonstrations
    D_r = D \\ D_f           every remaining tool + its demonstrations

and its description of the baselines: "we treat all data related to T_f as
unlearning examples and all data related to T_r as remaining examples."  The
split is at *tool* granularity -- picking individual samples would be
sample-level unlearning, the thing the paper is explicitly not doing.  A "tool"
is one API, matching the paper's count for ToolAlpaca (495 tools).

The partition is FROZEN.  It is computed once and written to `split.json`; every
later run reads the tool lists back out of that file instead of re-drawing them,
so the forget set cannot silently change under a finished experiment.  Use
--refreeze to deliberately draw a new one.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_dataset import build_dataset


def surviving_instances(api):
    """Indices of Instances that build_dataset keeps, plus one extra filter.

    Mirrors build_dataset's own drops (no intermediate_steps, more than 5 steps,
    or nothing but getDetails calls) so that the instance indices here line up
    exactly with the rows build_dataset will emit -- asserted below.

    The extra filter is the empty question: 157 instances in train_data.json have
    an empty `input`, and `Instructions` is not reliably parallel to `Instances`
    (42 APIs disagree on length), so the query is always taken from the instance
    itself, the way build_dataset does it.  An empty query cannot be evaluated
    and is not worth unlearning.
    """
    keep = []
    for idx, ans in enumerate(api.get("Instances", [])):
        if not ans.get("intermediate_steps"):
            continue
        if len(ans["intermediate_steps"]) > 5:
            continue
        used = {step[0][0] for step in ans["intermediate_steps"]}
        if used == {"getDetails"}:
            continue
        if not ans.get("input", "").rsplit("\nHint: ", 1)[0].strip():
            continue
        keep.append(idx)
    return keep


def subset_api(api, indices):
    """A shallow copy of `api` carrying only the chosen Instances."""
    out = dict(api)
    out["Instances"] = [api["Instances"][i] for i in indices]
    return out


def to_eval_api(api, indices):
    """Render held-out instances in the schema the harness evaluates.

    Same keys as data/eval_simulated.json, so run_toolalpaca.py, the simulator
    and evaluation.py all consume it unchanged.  Golden_Answers is reconstructed
    from each instance's intermediate_steps, whose first element is exactly
    [function name, action input] -- the same pair the eval files store.
    """
    out = {k: v for k, v in api.items() if k != "Instances"}
    out["Instructions"] = [
        api["Instances"][i]["input"].rsplit("\nHint: ", 1)[0] for i in indices
    ]
    out["Golden_Answers"] = [
        [{"Action": step[0][0], "Action_Input": step[0][1]}
         for step in api["Instances"][i]["intermediate_steps"]]
        for i in indices
    ]
    return out


def choose_forget_tools(names, ratio, seed):
    ordered = sorted(names)                 # sort first: never depend on file order
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return set(ordered[:int(round(len(ordered) * ratio))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="./data/train_data.json")
    p.add_argument("--out_dir", default="./tool_unlearn/data")
    p.add_argument("--forget_ratio", type=float, default=0.20,
                   help="fraction of TOOLS to unlearn; the paper sweeps 2-20%%")
    p.add_argument("--eval_ratio", type=float, default=0.25,
                   help="fraction of each tool's demonstrations held out for "
                        "measuring T_f / T_r.  Never trained on.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refreeze", action="store_true",
                   help="draw a NEW tool partition, overwriting the frozen one")
    args = p.parse_args()

    api_data = json.load(open(args.train_data, encoding="utf-8"))
    # Same filter as build_dataset.py's __main__: an API with no function
    # descriptions cannot be rendered into a prompt.
    apis = [a for a in api_data if a.get("Function_Description") is not None]
    names = sorted({a["Name"] for a in apis})

    split_path = os.path.join(args.out_dir, "split.json")
    if os.path.exists(split_path) and not args.refreeze:
        frozen = json.load(open(split_path, encoding="utf-8"))
        forget_tools = set(frozen["forget_tools"])
        print("[split] reusing the frozen partition in %s (%d forget tools). "
              "Pass --refreeze to draw a new one." % (split_path, len(forget_tools)))
    else:
        forget_tools = choose_forget_tools(names, args.forget_ratio, args.seed)
        print("[split] drawing a new partition: %d/%d tools marked for unlearning"
              % (len(forget_tools), len(names)))

    buckets = {"forget_train": [], "retain_train": []}
    per_tool = {}

    eval_apis = {"forget": [], "retain": []}
    name_counts = {"forget": {}, "retain": {}}
    for api in apis:
        name = api["Name"]
        idx = surviving_instances(api)
        if not idx:
            continue
        # Deterministic per-tool shuffle so the eval half is a random sample of
        # the tool's demonstrations, not whichever happen to come last.
        random.Random("%s/%d" % (name, args.seed)).shuffle(idx)

        # At least one eval row for any tool with two or more demonstrations; a
        # tool with a single one keeps it for training, since a forget tool with
        # nothing to unlearn is useless.
        n_eval = (min(len(idx) - 1, max(1, int(round(len(idx) * args.eval_ratio))))
                  if len(idx) > 1 else 0)
        ev_idx, tr_idx = sorted(idx[:n_eval]), sorted(idx[n_eval:])

        group = "forget" if name in forget_tools else "retain"

        rows = build_dataset(subset_api(api, tr_idx))
        assert len(rows) == len(tr_idx), (
            "build_dataset kept %d of %d instances for %s -- surviving_instances "
            "has drifted from the upstream filters" % (len(rows), len(tr_idx), name))
        for proc, trainable in rows:
            buckets[group + "_train"].append(
                {"tool": name, "process": proc, "trainable": trainable})

        if ev_idx:
            entry = to_eval_api(api, ev_idx)
            # 19 API names occur more than once in train_data.json, and 15 of
            # those pairs carry DIFFERENT Documentation and Function_Projection
            # -- they are distinct APIs that collide on name.  The simulator
            # keys its registry as {api["Name"]: api}, so a duplicate would
            # overwrite the other and answer calls against the wrong OpenAPI
            # spec.  Disambiguate here; the suffix only ever appears in the
            # routing key and the cache key, never in a tool name or a prompt.
            seen = name_counts[group]
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                entry["Name"] = "%s__%d" % (name, seen[name])
            eval_apis[group].append(entry)

        per_tool.setdefault(name, {"group": group, "n_train": 0, "n_eval": 0})
        per_tool[name]["n_train"] += len(tr_idx)
        per_tool[name]["n_eval"] += len(ev_idx)

    os.makedirs(args.out_dir, exist_ok=True)
    for key, rows in buckets.items():
        path = os.path.join(args.out_dir, "%s.json" % key)
        json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        print("  %-16s %5d examples  %4d tools  -> %s   (unlearning input)"
              % (key, len(rows), len({r["tool"] for r in rows}), path))
    for group, entries in eval_apis.items():
        path = os.path.join(args.out_dir, "%s_eval_api.json" % group)
        json.dump(entries, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        n = sum(len(e["Instructions"]) for e in entries)
        assert len({e["Name"] for e in entries}) == len(entries), \
            "duplicate API names survived into %s -- the simulator would " \
            "serve one of them with the wrong spec" % path
        print("  %-16s %5d instructions %4d APIs  -> %s   (T_%s, harness format)"
              % (group + "_eval", n, len(entries), path, group[0]))

    meta = {
        "train_data": args.train_data,
        "seed": args.seed,
        "forget_ratio": args.forget_ratio,
        "eval_ratio": args.eval_ratio,
        "n_tools_total": len(names),
        "n_tools_forget_marked": len(forget_tools),
        # A marked tool can still contribute nothing: build_dataset drops
        # demonstrations with no intermediate_steps, with more than 5 steps, or
        # that only ever call getDetails.  Record what actually landed on disk so
        # the split is not over-reported.
        "n_tools_forget_realised": len({r["tool"] for r in buckets["forget_train"]}),
        "n_tools_retain_realised": len({r["tool"] for r in buckets["retain_train"]}),
        "n_examples": dict(
            [(k, len(v)) for k, v in buckets.items()]
            + [(g + "_eval", sum(len(e["Instructions"]) for e in entries))
               for g, entries in eval_apis.items()]),
        "forget_tools": sorted(forget_tools),
        "per_tool": per_tool,
    }
    json.dump(meta, open(split_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n[split] frozen partition written to %s" % split_path)
    print("[split] T_T is measured on data/eval_simulated.json (10 unseen APIs), "
          "not on any file above.")


if __name__ == "__main__":
    main()
