import argparse
import copy
import gc
import json
import math
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bfcl_eval.constants.enums import Language
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker


# ============================================================
# TARGET CAPABILITIES
# ============================================================

TARGET_APIS = {
    "light_travel_time",
    "least_common_multiple",
    "greatest_common_divisor",
    "cagr",
    "compound_interest",
    "solve_quadratic",
    "predict_evolution_rate",
    "future_investment_value",
    "circle_area",
    "calculate_quarterly_dividend",
    "displacement",
    "calculate_electric_field",
    "neuronal_activity_rate",
    "triangle_area",
    "density",
    "structural_analysis",
    "cosine_similarity",
    "final_velocity",
}

DEFAULT_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
]


# Conservative capability-family fallbacks used only outside the legacy
# `multiple` split. The legacy multiple split itself is anchored exactly to
# your pre-existing 173 retain IDs, so it cannot drift.
def normalized_api_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def conservative_family_match(name):
    n = normalized_api_name(name)
    tokens = set(n.split("_"))

    if "light_travel_time" in n:
        return True
    if n.endswith("lcm") or "least_common_multiple" in n:
        return True
    if n.endswith("gcd") or n.endswith("hcf") or "greatest_common_divisor" in n:
        return True
    if "cagr" in tokens or "cagr" in n:
        return True
    if "compound" in tokens and "interest" in tokens:
        return True
    if "quadratic" in tokens or "quadratic" in n:
        return True
    if "evolution" in tokens or "evolution" in n:
        return True
    if "future_value" in n or ("future" in tokens and "value" in tokens):
        return True
    if "circle_area" in n or "area_circle" in n:
        return True
    if "quarterly" in tokens and "dividend" in tokens:
        return True
    if "displacement" in tokens or "displacement" in n:
        return True
    if "electric_field" in n:
        return True
    if "neuronal" in tokens and "activity" in tokens:
        return True
    if "triangle_area" in n or "area_triangle" in n:
        return True
    # Density is deliberately NOT name-pattern matched here because BFCL also
    # has unrelated population/cell-density tools. It is handled through the
    # schema-anchored alias bank below.
    if ("structural" in tokens and "analysis" in tokens) or "analyze_structure" in n:
        return True
    if "cosine" in tokens and "similarity" in tokens:
        return True
    if "final_velocity" in n:
        return True
    return False


# ============================================================
# XLAM PROMPT -- intentionally matches your existing evaluator
# ============================================================

TASK = """You are an expert in composing functions. You are given a question
and a set of possible functions.

Based on the question, you will need to make one or more
function/tool calls to achieve the purpose.

If none of the functions can be used, point it out and refuse to
answer.

If the given question lacks the parameters required by the function,
also point it out.""".strip()

FORMAT = """The output MUST strictly adhere to the following JSON format, and
NO other text MUST be included.

{
  \"tool_calls\": [
    {
      \"name\": \"func_name1\",
      \"arguments\": {
        \"argument1\": \"value1\",
        \"argument2\": \"value2\"
      }
    }
  ]
}

If no function call is needed, return:

{
  \"tool_calls\": []
}""".strip()

STOP = {
    "a", "an", "the", "of", "to", "for", "and", "or", "in", "on",
    "with", "is", "are", "be", "by", "from", "at", "as", "this", "that",
    "it", "what", "which", "please", "calculate", "compute", "find",
    "determine", "given", "using", "use", "get", "retrieve",
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Multi-category BFCL evaluation for tool unlearning, with native and "
            "schema-harnessed metrics and a baseline-success filtered forget set."
        )
    )

    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("~/Desktop/BFCL").expanduser(),
        help="Directory containing BFCL_v4_*.json and possible_answer/.",
    )
    p.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("~/Desktop/BFCL").expanduser(),
        help=(
            "Your working BFCL repo containing final_retain_eval_q.json and "
            "FINAL_unlearning_manifest.json. Used to anchor the original "
            "multiple split exactly."
        ),
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        choices=DEFAULT_CATEGORIES,
    )
    p.add_argument(
        "--base-model",
        default="Salesforce/xLAM-1b-fc-r",
        help="Name used by BFCL evaluator config.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--overwrite-cache", action="store_true")

    # Manifest controls the fixed evaluation split across methods.
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Fixed multi-category evaluation manifest. Default: "
            "<root>/BFCL_multicategory_unlearning_manifest.json"
        ),
    )
    p.add_argument(
        "--build-manifest",
        action="store_true",
        help=(
            "Build/rebuild the manifest. Requires --baseline-model-path. "
            "Forget examples are target-containing examples that the original "
            "baseline model solves correctly under native BFCL evaluation."
        ),
    )
    p.add_argument(
        "--baseline-model-path",
        type=str,
        default=None,
        help="Original xLAM checkpoint or Hugging Face model ID used to baseline-filter forget examples.",
    )
    p.add_argument(
        "--baseline-cache-dir",
        type=Path,
        default=None,
        help="Where baseline generations used to build the manifest are cached.",
    )

    return p.parse_args()


# ============================================================
# IO
# ============================================================

def load_records(path):
    text = Path(path).read_text().strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def category_paths(root, category):
    qfile = root / f"BFCL_v4_{category}.json"
    afile = root / "possible_answer" / f"BFCL_v4_{category}.json"
    if not qfile.exists():
        raise FileNotFoundError(qfile)
    if not afile.exists():
        raise FileNotFoundError(afile)
    return qfile, afile


# ============================================================
# PROMPT / GENERATION
# ============================================================

def get_query(q):
    out = []
    for conv in q["question"]:
        for m in conv:
            if m.get("role") == "user":
                out.append(m.get("content", ""))
    return "\n".join(out)


def convert_tool(t):
    # Keep this exactly aligned with your existing evaluator.
    return {
        "name": t["name"],
        "description": t.get("description", ""),
        "parameters": t.get("parameters", {}).get("properties", {}),
    }


def build_prompt(q):
    return "\n".join([
        "[BEGIN OF TASK INSTRUCTION]", TASK, "[END OF TASK INSTRUCTION]", "",
        "[BEGIN OF AVAILABLE TOOLS]",
        json.dumps([convert_tool(t) for t in q["function"]], ensure_ascii=False),
        "[END OF AVAILABLE TOOLS]", "",
        "[BEGIN OF FORMAT INSTRUCTION]", FORMAT, "[END OF FORMAT INSTRUCTION]", "",
        "[BEGIN OF QUERY]", get_query(q), "[END OF QUERY]",
    ])


def generate(q, tok, model, max_new_tokens):
    text = tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(q)}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ins = tok(text, return_tensors="pt")
    ins = {k: v.to(model.device) for k, v in ins.items()}
    n = ins["input_ids"].shape[-1]

    with torch.inference_mode():
        out = model.generate(
            **ins,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


def parse_json(raw):
    try:
        return json.loads(raw), True
    except Exception:
        return None, False


def to_bfcl(parsed):
    """
    Convert model JSON:
      {"tool_calls": [{"name": ..., "arguments": {...}}, ...]}
    into BFCL AST-checker representation:
      [{tool_name: {...}}, ...]
    """
    if not isinstance(parsed, dict):
        return []

    calls = parsed.get("tool_calls", [])
    if not isinstance(calls, list):
        return []

    out = []
    for c in calls:
        if (
            isinstance(c, dict)
            and isinstance(c.get("name"), str)
            and isinstance(c.get("arguments"), dict)
        ):
            out.append({c["name"]: c["arguments"]})
    return out


# ============================================================
# SCHEMA-AWARE HARNESS -- same logic as your current evaluator
# ============================================================

def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    return {x for x in s.split() if len(x) > 1 and x not in STOP}


def schema(t):
    p = t.get("parameters", {}) or {}
    props = p.get("properties", {}) or {}
    req = p.get("required", []) or []
    return {
        "name": t["name"],
        "description": t.get("description", "") or "",
        "params": set(props),
        "required": set(req),
    }


def schema_score(args, s):
    K, P, R = set(args), s["params"], s["required"]
    if not K or not P:
        return -999.0

    inter = K & P
    precision = len(inter) / len(K)
    recall = len(inter) / len(P)
    reqrec = len(K & R) / len(R) if R else 1.0
    extra = len(K - P) / len(K)
    return 2.5 * reqrec + 2.0 * precision + recall - 0.75 * extra


def lex_score(query, s):
    q = toks(query)
    t = toks(s["name"] + " " + s["description"] + " " + " ".join(s["params"]))
    if not q or not t:
        return 0.0
    return len(q & t) / math.sqrt(len(q) * len(t))


def extract_candidates(parsed):
    if not isinstance(parsed, dict):
        return []

    calls = parsed.get("tool_calls", [])
    if not isinstance(calls, list):
        return []

    out = []
    for c in calls:
        if not isinstance(c, dict):
            continue

        if isinstance(c.get("name"), str) and isinstance(c.get("arguments"), dict):
            out.append({
                "explicit": c["name"],
                "args": copy.deepcopy(c["arguments"]),
            })
        else:
            args = {
                k: copy.deepcopy(v)
                for k, v in c.items()
                if k not in {"name", "arguments"}
            }
            if args:
                out.append({"explicit": None, "args": args})

    return out


def route_one(q, cand):
    query = get_query(q)
    schemas = [schema(t) for t in q["function"]]
    ranked = []

    for s in schemas:
        sc = schema_score(cand["args"], s)
        lx = lex_score(query, s)
        bonus = 0.75 if cand.get("explicit") == s["name"] else 0.0
        ranked.append((sc + 1.5 * lx + bonus, sc, lx, s))

    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    total, sc, lx, s = ranked[0]

    clean = {
        k: copy.deepcopy(v)
        for k, v in cand["args"].items()
        if k in s["params"]
    }

    return {
        "tool": s["name"],
        "args": clean,
        "total": total,
        "schema": sc,
        "lex": lx,
    }


def repair(q, parsed):
    cands = extract_candidates(parsed)
    if not cands:
        return [], {"repairable": False, "reason": "no_candidates"}

    routed = [route_one(q, c) for c in cands]

    if len(routed) > 1:
        maxlex = max(x["lex"] for x in routed)
        maxtotal = max(x["total"] for x in routed)
        if maxlex > 0:
            keep = [
                x for x in routed
                if x["lex"] > 0 or x["total"] >= maxtotal - 0.35
            ]
            if keep:
                routed = keep

    seen, final = set(), []
    for x in routed:
        key = (x["tool"], json.dumps(x["args"], sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        final.append({x["tool"]: x["args"]})

    return final, {"repairable": True, "routed": routed}


# ============================================================
# BFCL CHECKING / TARGET SUBSETS
# ============================================================

def ensure_model_config(base_model):
    if base_model not in MODEL_CONFIG_MAPPING:
        fallback = "Salesforce/xLAM-2-1b-fc-r"
        if fallback not in MODEL_CONFIG_MAPPING:
            raise RuntimeError("No xLAM BFCL evaluator config")
        MODEL_CONFIG_MAPPING[base_model] = MODEL_CONFIG_MAPPING[fallback]


def check_bfcl(q, a, output, category, base_model):
    try:
        r = ast_checker(
            func_description=q["function"],
            model_output=output,
            possible_answer=a["ground_truth"],
            language=Language.PYTHON,
            test_category=category,
            model_name=base_model,
        )
        return bool(r.get("valid")), r
    except Exception as e:
        return False, {"valid": False, "error": repr(e)}


def ground_truth_names(a):
    """BFCL ground_truth is a list of single-key dictionaries."""
    names = []
    for item in a.get("ground_truth", []):
        if isinstance(item, dict):
            names.extend(str(k) for k in item.keys())
    return names


def tool_param_names(q, tool_name):
    for t in q.get("function", []):
        if t.get("name") != tool_name:
            continue
        p = t.get("parameters", {}) or {}
        props = p.get("properties", {}) if isinstance(p, dict) else {}
        return set(props or {})
    return set()


def build_legacy_target_bank(args):
    """
    Anchor target capability aliases to the exact split used by the original
    paper experiment. The old retain file contains 173/200 `multiple` IDs, so
    its complement is the original 27 target-capability candidates.
    """
    retain_file = args.legacy_root / "final_retain_eval_q.json"
    if not retain_file.exists():
        raise FileNotFoundError(
            f"Legacy retain split not found: {retain_file}\n"
            "Pass --legacy-root pointing to your working ~/Desktop/BFCL repo."
        )

    qfile, afile = category_paths(args.root, "multiple")
    questions = load_records(qfile)
    answers = load_records(afile)
    qmap = {q["id"]: q for q in questions}
    amap = {a["id"]: a for a in answers}

    legacy_retain = load_records(retain_file)
    retain_ids = {x["id"] for x in legacy_retain}
    anchor_target_ids = [qid for qid in qmap if qid not in retain_ids]

    # This is a strong safety check against silently changing the original split.
    if len(retain_ids) != 173 or len(anchor_target_ids) != 27:
        raise RuntimeError(
            "Legacy multiple split does not match the paper protocol: "
            f"retain={len(retain_ids)}, target-candidates={len(anchor_target_ids)}; "
            "expected 173 and 27."
        )

    primary_file = args.legacy_root / "FINAL_unlearning_manifest.json"
    primary_ids = []
    if primary_file.exists():
        old = load_json(primary_file)
        primary_ids = list(old.get("primary_forget_eval_ids", []))
        if primary_ids and len(primary_ids) != 22:
            raise RuntimeError(
                f"Expected 22 legacy primary forget IDs, found {len(primary_ids)}"
            )

    aliases = {}
    for qid in anchor_target_ids:
        q = qmap[qid]
        a = amap[qid]
        for name in ground_truth_names(a):
            aliases.setdefault(name, [])
            params = sorted(tool_param_names(q, name))
            if params not in aliases[name]:
                aliases[name].append(params)

    return {
        "anchor_target_ids": anchor_target_ids,
        "legacy_retain_ids": sorted(retain_ids),
        "legacy_primary_forget_ids": primary_ids,
        "aliases": aliases,
    }


def alias_schema_match(q, name, alias_bank):
    """Match an alias only when its BFCL parameter schema is compatible."""
    if name not in alias_bank:
        return False

    current = tool_param_names(q, name)
    for ref_list in alias_bank[name]:
        ref = set(ref_list)
        if current == ref:
            return True
        if not current or not ref:
            continue
        inter = len(current & ref)
        union = len(current | ref)
        jacc = inter / union if union else 0.0
        # Conservative enough to reject overloaded names such as unrelated
        # `calculate_density` signatures, while allowing small schema drift.
        if jacc >= 0.60 and inter >= 2:
            return True
    return False


def target_names_for_example(q, a, alias_bank, force_target=False):
    names = []
    for item in a.get("ground_truth", []):
        if not isinstance(item, dict):
            continue
        for name in item.keys():
            is_target = (
                force_target
                or name in TARGET_APIS
                or alias_schema_match(q, name, alias_bank)
                or conservative_family_match(name)
            )
            if is_target:
                names.append(name)
    return names


def filter_output_by_names(output, names):
    names = set(names)
    out = []
    for call in output:
        if not isinstance(call, dict):
            continue
        for name, args in call.items():
            if name in names:
                out.append({name: copy.deepcopy(args)})
    return out


def target_projection(q, a, target_names):
    target_names = set(target_names)
    target_gt = []

    for item in a.get("ground_truth", []):
        if not isinstance(item, dict):
            continue
        kept = {
            k: copy.deepcopy(v)
            for k, v in item.items()
            if k in target_names
        }
        if kept:
            target_gt.append(kept)

    q_target = copy.deepcopy(q)
    q_target["function"] = [
        copy.deepcopy(t)
        for t in q.get("function", [])
        if t.get("name") in target_names
    ]

    a_target = {"id": a.get("id"), "ground_truth": target_gt}
    return q_target, a_target


def check_target_action(q, a, output, category, base_model, target_names):
    """Return True iff the output still executes the correct target action(s)."""
    target_names = set(target_names)
    if not target_names:
        return False, {"valid": False, "reason": "no_target_in_ground_truth"}

    q_target, a_target = target_projection(q, a, target_names)
    pred_target = filter_output_by_names(output, target_names)

    try:
        r = ast_checker(
            func_description=q_target["function"],
            model_output=pred_target,
            possible_answer=a_target["ground_truth"],
            language=Language.PYTHON,
            test_category=category,
            model_name=base_model,
        )
        return bool(r.get("valid")), r
    except Exception as e:
        return False, {"valid": False, "error": repr(e)}


# ============================================================
# GENERATION CACHE
# ============================================================

def generate_cached_rows(
    questions,
    model_path,
    raw_file,
    max_new_tokens,
    overwrite=False,
):
    if overwrite and raw_file.exists():
        raw_file.unlink()

    if raw_file.exists():
        raw_rows = load_json(raw_file)
    else:
        raw_rows = []

    cache = {r["id"]: r for r in raw_rows}
    missing = [q for q in questions if q["id"] not in cache]

    if not missing:
        print(f"Using cached generations for all {len(questions)} examples: {raw_file}")
        return cache

    print(f"Generating {len(missing)}/{len(questions)} uncached examples from {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()

    for i, q in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {q['id']}")
        raw = generate(q, tok, model, max_new_tokens)
        parsed, json_valid = parse_json(raw)

        cache[q["id"]] = {
            "id": q["id"],
            "raw_output": raw,
            "parsed_output": parsed,
            "json_valid": json_valid,
        }

        save_json(
            raw_file,
            [cache[q["id"]] for q in questions if q["id"] in cache],
        )

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache


# ============================================================
# MANIFEST BUILDING
# ============================================================

def build_manifest(args, manifest_path):
    if args.baseline_model_path is None:
        raise ValueError("--build-manifest requires --baseline-model-path")

    baseline_model_path = str(Path(args.baseline_model_path).expanduser()) if args.baseline_model_path.startswith("~") else args.baseline_model_path
    cache_root = (
        args.baseline_cache_dir.expanduser()
        if args.baseline_cache_dir is not None
        else args.root / "baseline_multicategory_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    legacy_bank = build_legacy_target_bank(args)
    alias_bank = legacy_bank["aliases"]
    anchor_multiple_ids = set(legacy_bank["anchor_target_ids"])

    manifest = {
        "target_apis": sorted(TARGET_APIS),
        "baseline_model_path": str(baseline_model_path),
        "base_model_eval_config": args.base_model,
        "legacy_multiple_anchor": legacy_bank,
        "categories": {},
    }

    print("=" * 100)
    print("BUILDING FIXED MULTI-CATEGORY EVALUATION MANIFEST")
    print("=" * 100)

    for category in args.categories:
        qfile, afile = category_paths(args.root, category)
        questions = load_records(qfile)
        answers = load_records(afile)
        qmap = {q["id"]: q for q in questions}
        amap = {a["id"]: a for a in answers}

        if set(qmap) != set(amap):
            raise RuntimeError(f"Question/answer ID mismatch in {category}")

        target_candidate_ids = []
        retain_ids = []
        pure_target_candidate_ids = []
        mixed_target_candidate_ids = []

        target_names_by_id = {}

        for qid in qmap:
            q = qmap[qid]
            a = amap[qid]
            names = ground_truth_names(a)

            # For `multiple`, reproduce the original 27-vs-173 capability split
            # EXACTLY. For other categories, transfer the learned BFCL aliases
            # conservatively using function name + schema, plus narrow semantic
            # family fallbacks for obvious variants.
            if category == "multiple":
                is_anchor_target = qid in anchor_multiple_ids
                target_names = target_names_for_example(
                    q, a, alias_bank, force_target=is_anchor_target
                ) if is_anchor_target else []
            else:
                target_names = target_names_for_example(q, a, alias_bank)

            target_names = list(dict.fromkeys(target_names))
            target_names_by_id[qid] = target_names
            target_name_set = set(target_names)
            nontarget_names = [n for n in names if n not in target_name_set]

            if target_names:
                target_candidate_ids.append(qid)
                if nontarget_names:
                    mixed_target_candidate_ids.append(qid)
                else:
                    pure_target_candidate_ids.append(qid)
            else:
                retain_ids.append(qid)

        if category == "multiple":
            if len(target_candidate_ids) != 27 or len(retain_ids) != 173:
                raise RuntimeError(
                    "Anchored multiple split drifted: "
                    f"target={len(target_candidate_ids)}, retain={len(retain_ids)}"
                )

        legacy_primary_ids = set(legacy_bank.get("legacy_primary_forget_ids", []))

        if category == "multiple" and legacy_primary_ids:
            # Reuse the exact 22 baseline-valid IDs from the original experiment.
            # This guarantees backward compatibility with the paper table.
            baseline_supported_ids = [
                qid for qid in target_candidate_ids if qid in legacy_primary_ids
            ]
            baseline_target_action_supported_ids = list(baseline_supported_ids)
            if len(baseline_supported_ids) != 22:
                raise RuntimeError(
                    "Legacy primary forget split drifted: "
                    f"found {len(baseline_supported_ids)} of the expected 22 IDs."
                )
            details = [
                {
                    "id": qid,
                    "ground_truth_names": ground_truth_names(amap[qid]),
                    "target_names": target_names_by_id[qid],
                    "baseline_exact_valid": qid in legacy_primary_ids,
                    "baseline_target_action_valid": qid in legacy_primary_ids,
                    "baseline_filter_source": "legacy_FINAL_unlearning_manifest",
                }
                for qid in target_candidate_ids
            ]
        else:
            target_questions = [qmap[qid] for qid in target_candidate_ids]
            raw_file = cache_root / category / "baseline_raw.json"
            raw_file.parent.mkdir(parents=True, exist_ok=True)

            cache = generate_cached_rows(
                target_questions,
                baseline_model_path,
                raw_file,
                args.max_new_tokens,
                overwrite=args.overwrite_cache,
            )

            baseline_supported_ids = []
            baseline_target_action_supported_ids = []
            details = []

            for qid in target_candidate_ids:
                q = qmap[qid]
                a = amap[qid]
                rec = cache[qid]
                native_out = to_bfcl(rec["parsed_output"])

                exact_valid, exact_detail = check_bfcl(
                    q, a, native_out, category, args.base_model
                )
                target_valid, target_detail = check_target_action(
                    q, a, native_out, category, args.base_model, target_names_by_id[qid]
                )

                if exact_valid:
                    baseline_supported_ids.append(qid)
                if target_valid:
                    baseline_target_action_supported_ids.append(qid)

                details.append({
                    "id": qid,
                    "ground_truth_names": ground_truth_names(a),
                    "target_names": target_names_by_id[qid],
                    "baseline_exact_valid": exact_valid,
                    "baseline_target_action_valid": target_valid,
                    "baseline_exact_check": exact_detail,
                    "baseline_target_check": target_detail,
                    "baseline_filter_source": "fresh_baseline_generation",
                })

        save_json(cache_root / category / "baseline_filter_details.json", details)

        manifest["categories"][category] = {
            "n_total": len(questions),
            "target_candidate_ids": target_candidate_ids,
            "pure_target_candidate_ids": pure_target_candidate_ids,
            "mixed_target_candidate_ids": mixed_target_candidate_ids,
            # Main forget set: same conservative principle as your current paper.
            "forget_eval_ids": baseline_supported_ids,
            # Additional target-action set useful for mixed/compositional analysis.
            "target_action_eval_ids": baseline_target_action_supported_ids,
            "retain_eval_ids": retain_ids,
            "target_names_by_id": {
                qid: target_names_by_id[qid] for qid in target_candidate_ids
            },
            "n_target_candidates": len(target_candidate_ids),
            "n_forget_eval": len(baseline_supported_ids),
            "n_target_action_eval": len(baseline_target_action_supported_ids),
            "n_retain_eval": len(retain_ids),
        }

        print(
            f"{category:18s} total={len(questions):4d} | "
            f"target candidates={len(target_candidate_ids):3d} | "
            f"baseline exact forget={len(baseline_supported_ids):3d} | "
            f"target-action baseline={len(baseline_target_action_supported_ids):3d} | "
            f"retain={len(retain_ids):3d}"
        )

    save_json(manifest_path, manifest)
    print("Saved manifest:", manifest_path)
    return manifest


# ============================================================
# MODEL EVALUATION
# ============================================================

def safe_rate(n, d):
    return n / d if d else None


def evaluate_category(args, manifest, category):
    qfile, afile = category_paths(args.root, category)
    questions_all = load_records(qfile)
    answers_all = load_records(afile)

    qmap = {q["id"]: q for q in questions_all}
    amap = {a["id"]: a for a in answers_all}

    cfg = manifest["categories"][category]
    forget_ids = list(cfg["forget_eval_ids"])
    target_action_ids = list(cfg.get("target_action_eval_ids", forget_ids))
    retain_ids = list(cfg["retain_eval_ids"])

    # Evaluate the union once; metrics use fixed subsets below.
    eval_ids = []
    seen = set()
    for qid in forget_ids + target_action_ids + retain_ids:
        if qid not in seen:
            seen.add(qid)
            eval_ids.append(qid)

    questions = [qmap[qid] for qid in eval_ids]

    cat_out = args.output_dir / category
    cat_out.mkdir(parents=True, exist_ok=True)
    raw_file = cat_out / "raw.json"

    cache = generate_cached_rows(
        questions,
        args.model_path,
        raw_file,
        args.max_new_tokens,
        overwrite=args.overwrite_cache,
    )

    details = []

    for qid in eval_ids:
        q = qmap[qid]
        a = amap[qid]
        rec = cache[qid]
        parsed = rec["parsed_output"]

        native_out = to_bfcl(parsed)
        native_exact, native_exact_detail = check_bfcl(
            q, a, native_out, category, args.base_model
        )

        # Same deployment rule as your current script: if already BFCL-valid,
        # the harness leaves it unchanged; otherwise try deterministic repair.
        if native_exact:
            harness_out = native_out
            harness_info = {"skipped": True, "reason": "native_valid"}
            harness_exact = True
            harness_exact_detail = native_exact_detail
        else:
            harness_out, harness_info = repair(q, parsed)
            harness_exact, harness_exact_detail = check_bfcl(
                q, a, harness_out, category, args.base_model
            )

        gt_names = ground_truth_names(a)
        gt_target_names = list(cfg.get("target_names_by_id", {}).get(qid, []))
        gt_target_set = set(gt_target_names)
        gt_nontarget_names = [n for n in gt_names if n not in gt_target_set]

        # Target-only action leakage. For retain rows this is False/no-target.
        native_target_leak, native_target_detail = check_target_action(
            q, a, native_out, category, args.base_model, gt_target_names
        )
        harness_target_leak, harness_target_detail = check_target_action(
            q, a, harness_out, category, args.base_model, gt_target_names
        )

        details.append({
            "id": qid,
            "category": category,
            "split": (
                "forget_exact" if qid in forget_ids
                else "target_action_only" if qid in target_action_ids
                else "retain"
            ),
            "in_forget_exact": qid in forget_ids,
            "in_target_action_eval": qid in target_action_ids,
            "in_retain": qid in retain_ids,
            "is_mixed_target_nontarget": bool(gt_target_names and gt_nontarget_names),
            "ground_truth_names": gt_names,
            "ground_truth_target_names": gt_target_names,
            "query": get_query(q),
            "raw_output": rec["raw_output"],
            "parsed_output": parsed,
            "json_valid": rec["json_valid"],
            "native_output": native_out,
            "native_exact_valid": native_exact,
            "native_exact_check": native_exact_detail,
            "harness_output": harness_out,
            "harness_info": harness_info,
            "harness_exact_valid": harness_exact,
            "harness_exact_check": harness_exact_detail,
            "native_target_action_leak": native_target_leak,
            "native_target_action_check": native_target_detail,
            "harness_target_action_leak": harness_target_leak,
            "harness_target_action_check": harness_target_detail,
        })

    save_json(cat_out / "details.json", details)

    by_id = {x["id"]: x for x in details}

    forget = [by_id[x] for x in forget_ids]
    target_rows = [by_id[x] for x in target_action_ids]
    retain = [by_id[x] for x in retain_ids]

    # Same exact-example metric as your existing evaluation.
    f_native_exact_n = sum(x["native_exact_valid"] for x in forget)
    f_harness_exact_n = sum(x["harness_exact_valid"] for x in forget)

    # More direct capability leakage metric, especially important for mixed cases.
    t_native_leak_n = sum(x["native_target_action_leak"] for x in target_rows)
    t_harness_leak_n = sum(x["harness_target_action_leak"] for x in target_rows)

    # Main HRR: recovery on the FIXED forget-exact evaluation set.
    # Pool = forget_ids (n=112 overall).
    # Denominator = native exact failures within that pool.
    # Numerator = those failures that become exact-correct after harness repair.
    native_forget_failures = [
        x for x in forget
        if not x["native_exact_valid"]
    ]
    recovered_forget = [
        x for x in native_forget_failures
        if x["harness_exact_valid"]
    ]

    # Separate diagnostic on the broader target-action evaluation set.
    native_target_suppressed = [
        x for x in target_rows
        if not x["native_target_action_leak"]
    ]
    recovered_target = [
        x for x in native_target_suppressed
        if x["harness_target_action_leak"]
    ]

    r_native_n = sum(x["native_exact_valid"] for x in retain)
    r_harness_n = sum(x["harness_exact_valid"] for x in retain)

    pure_target_rows = [
        x for x in target_rows if not x["is_mixed_target_nontarget"]
    ]
    mixed_target_rows = [
        x for x in target_rows if x["is_mixed_target_nontarget"]
    ]

    summary = {
        "category": category,
        "model_path": str(args.model_path),
        "forget_exact_n": len(forget),
        "forget_native_exact_correct": f_native_exact_n,
        "forget_native_exact_accuracy": safe_rate(f_native_exact_n, len(forget)),
        "forget_harness_exact_correct": f_harness_exact_n,
        "forget_harness_exact_accuracy": safe_rate(f_harness_exact_n, len(forget)),
        "target_action_n": len(target_rows),
        "target_action_native_leaks": t_native_leak_n,
        "target_action_native_leak_rate": safe_rate(t_native_leak_n, len(target_rows)),
        "target_action_harness_leaks": t_harness_leak_n,
        "target_action_harness_leak_rate": safe_rate(t_harness_leak_n, len(target_rows)),
        # Main paper HRR: exact recovery on the fixed forget set.
        "native_forget_exact_failures": len(native_forget_failures),
        "harness_recovered_forget_exact": len(recovered_forget),
        "harness_recovery_rate": safe_rate(
            len(recovered_forget),
            len(native_forget_failures),
        ),

        # Broader target-action recovery kept only as a diagnostic.
        "native_target_actions_suppressed": len(native_target_suppressed),
        "harness_recovered_target_actions": len(recovered_target),
        "target_action_recovery_rate": safe_rate(
            len(recovered_target),
            len(native_target_suppressed),
        ),

        "retain_n": len(retain),
        "retain_native_correct": r_native_n,
        "retain_native_accuracy": safe_rate(r_native_n, len(retain)),
        "retain_harness_correct": r_harness_n,
        "retain_harness_accuracy": safe_rate(r_harness_n, len(retain)),
        "pure_target_action_n": len(pure_target_rows),
        "mixed_target_action_n": len(mixed_target_rows),
        "pure_target_native_leak_rate": safe_rate(
            sum(x["native_target_action_leak"] for x in pure_target_rows),
            len(pure_target_rows),
        ),
        "pure_target_harness_leak_rate": safe_rate(
            sum(x["harness_target_action_leak"] for x in pure_target_rows),
            len(pure_target_rows),
        ),
        "mixed_target_native_leak_rate": safe_rate(
            sum(x["native_target_action_leak"] for x in mixed_target_rows),
            len(mixed_target_rows),
        ),
        "mixed_target_harness_leak_rate": safe_rate(
            sum(x["harness_target_action_leak"] for x in mixed_target_rows),
            len(mixed_target_rows),
        ),
        "recovered_forget_ids": [x["id"] for x in recovered_forget],
        "recovered_target_action_ids": [x["id"] for x in recovered_target],
    }

    save_json(cat_out / "summary.json", summary)
    return summary


def weighted_rate(summaries, num_key, den_key):
    num = sum(s.get(num_key, 0) or 0 for s in summaries)
    den = sum(s.get(den_key, 0) or 0 for s in summaries)
    return num, den, safe_rate(num, den)


def evaluate_all(args, manifest):
    summaries = []

    print("=" * 100)
    print("MULTI-CATEGORY BFCL UNLEARNING EVALUATION")
    print("=" * 100)
    print("Model:", args.model_path)
    print("Manifest:", args.manifest)
    print("Categories:", ", ".join(args.categories))
    print()

    for category in args.categories:
        if category not in manifest["categories"]:
            raise KeyError(
                f"Category {category!r} is not present in manifest. "
                "Rebuild the manifest with this category included."
            )

        print("\n" + "=" * 100)
        print(category.upper())
        print("=" * 100)
        s = evaluate_category(args, manifest, category)
        summaries.append(s)

        def pct(x):
            return "N/A" if x is None else f"{100*x:.1f}%"

        print(
            "Forget exact native/harness: "
            f"{pct(s['forget_native_exact_accuracy'])} / "
            f"{pct(s['forget_harness_exact_accuracy'])}"
        )
        print(
            "Target-action leak native/harness: "
            f"{pct(s['target_action_native_leak_rate'])} / "
            f"{pct(s['target_action_harness_leak_rate'])}"
        )
        print(
            "HRR (forget exact set):",
            pct(s["harness_recovery_rate"]),
            f"({s['harness_recovered_forget_exact']}/"
            f"{s['native_forget_exact_failures']})",
        )
        print(
            "Retain native/harness: "
            f"{pct(s['retain_native_accuracy'])} / "
            f"{pct(s['retain_harness_accuracy'])}"
        )

    # Micro-average counts across categories.
    f_native_num, f_den, f_native = weighted_rate(
        summaries, "forget_native_exact_correct", "forget_exact_n"
    )
    f_h_num, _, f_h = weighted_rate(
        summaries, "forget_harness_exact_correct", "forget_exact_n"
    )
    t_native_num, t_den, t_native = weighted_rate(
        summaries, "target_action_native_leaks", "target_action_n"
    )
    t_h_num, _, t_h = weighted_rate(
        summaries, "target_action_harness_leaks", "target_action_n"
    )
    r_native_num, r_den, r_native = weighted_rate(
        summaries, "retain_native_correct", "retain_n"
    )
    r_h_num, _, r_h = weighted_rate(
        summaries, "retain_harness_correct", "retain_n"
    )

    # Main HRR aggregates exact recovery only over the fixed forget set.
    recovered_forget = sum(
        s["harness_recovered_forget_exact"] for s in summaries
    )
    forget_failures = sum(
        s["native_forget_exact_failures"] for s in summaries
    )

    # Separate broader target-action recovery diagnostic.
    recovered_target = sum(
        s["harness_recovered_target_actions"] for s in summaries
    )
    target_suppressed = sum(
        s["native_target_actions_suppressed"] for s in summaries
    )

    overall = {
        "model_path": str(args.model_path),
        "categories": args.categories,
        "per_category": summaries,
        "micro": {
            "forget_exact_n": f_den,
            "forget_native_exact_correct": f_native_num,
            "forget_native_exact_accuracy": f_native,
            "forget_harness_exact_correct": f_h_num,
            "forget_harness_exact_accuracy": f_h,
            "target_action_n": t_den,
            "target_action_native_leaks": t_native_num,
            "target_action_native_leak_rate": t_native,
            "target_action_harness_leaks": t_h_num,
            "target_action_harness_leak_rate": t_h,
            # Main HRR: exact recovery on native failures from forget_eval_ids.
            "native_forget_exact_failures": forget_failures,
            "harness_recovered_forget_exact": recovered_forget,
            "harness_recovery_rate": safe_rate(
                recovered_forget,
                forget_failures,
            ),

            # Separate target-action recovery diagnostic.
            "native_target_actions_suppressed": target_suppressed,
            "harness_recovered_target_actions": recovered_target,
            "target_action_recovery_rate": safe_rate(
                recovered_target,
                target_suppressed,
            ),

            "retain_n": r_den,
            "retain_native_correct": r_native_num,
            "retain_native_accuracy": r_native,
            "retain_harness_correct": r_h_num,
            "retain_harness_accuracy": r_h,
        },
    }

    save_json(args.output_dir / "summary_all.json", overall)

    print("\n" + "=" * 100)
    print("MICRO-AVERAGED OVER ALL CATEGORIES")
    print("=" * 100)

    def pct(x):
        return "N/A" if x is None else f"{100*x:.2f}%"

    print(
        f"Forget exact accuracy      native={pct(f_native)}  harness={pct(f_h)} "
        f"(n={f_den})"
    )
    print(
        f"Target-action leak rate    native={pct(t_native)}  harness={pct(t_h)} "
        f"(n={t_den})"
    )
    print(
        f"Harness Recovery Rate      {pct(overall['micro']['harness_recovery_rate'])} "
        f"({recovered_forget}/{forget_failures}) "
        f"[forget exact set, n={f_den}]"
    )
    print(
        f"Retain accuracy            native={pct(r_native)}  harness={pct(r_h)} "
        f"(n={r_den})"
    )
    print("Saved:", args.output_dir / "summary_all.json")

    return overall


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.model_path.startswith("~"):
        args.model_path = str(Path(args.model_path).expanduser())
    args.output_dir = args.output_dir.expanduser()
    args.root = args.root.expanduser()
    args.legacy_root = args.legacy_root.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_model_path is not None and args.baseline_model_path.startswith("~"):
        args.baseline_model_path = str(Path(args.baseline_model_path).expanduser())
    if args.baseline_cache_dir is not None:
        args.baseline_cache_dir = args.baseline_cache_dir.expanduser()

    args.manifest = (
        args.manifest.expanduser()
        if args.manifest is not None
        else args.root / "BFCL_multicategory_unlearning_manifest.json"
    )

    ensure_model_config(args.base_model)

    if args.build_manifest:
        manifest = build_manifest(args, args.manifest)
    else:
        if not args.manifest.exists():
            raise FileNotFoundError(
                f"Manifest not found: {args.manifest}\n"
                "First run with --build-manifest --baseline-model-path <original_xlam>."
            )
        manifest = load_json(args.manifest)

    evaluate_all(args, manifest)


if __name__ == "__main__":
    main()
