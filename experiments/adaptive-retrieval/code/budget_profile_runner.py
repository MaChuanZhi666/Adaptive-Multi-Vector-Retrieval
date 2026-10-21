"""Assemble multi-budget profiles from existing runtime interfaces or CSV.

Retrieval remains owned by ``server_two_stage_runtime.py``. This module only
normalizes rows returned by ``run_full``/``run_two_stage`` and writes JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

from profile_io import csv_to_profiles, profile_from_runtime_row, read_jsonl, write_jsonl
from profiling_schema import BUDGETS, QueryBudgetProfile


def profiles_from_runtime_rows(rows: Iterable[Dict[str, Any]], query_texts: Dict[str, str] | None = None) -> List[QueryBudgetProfile]:
    grouped: Dict[str, QueryBudgetProfile] = {}
    for row in rows:
        qid = str(row.get("query_id", ""))
        profile = grouped.setdefault(qid, QueryBudgetProfile(qid, (query_texts or {}).get(qid, row.get("query_text", ""))))
        budget = str(row.get("budget", ""))
        if budget:
            profile.budgets[budget] = profile_from_runtime_row(row)
    for profile in grouped.values():
        profile.ensure_budgets(BUDGETS)
        from feature_extractor import candidate_overlap, topk_churn
        for old_budget, new_budget in zip(BUDGETS, BUDGETS[1:]):
            old = profile.budgets[str(old_budget)].candidate_info
            new = profile.budgets[str(new_budget)].candidate_info
            profile.transition_features[f"{old_budget}_to_{new_budget}"] = {
                "topk_churn": topk_churn(old.topk_ids, new.topk_ids),
                "candidate_overlap": candidate_overlap(old.candidate_ids, new.candidate_ids),
            }
    return list(grouped.values())


def load_query_texts(path: str | Path) -> Dict[str, str]:
    """Load query_id -> text from JSON or JSONL without assuming one schema."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    rows = json.loads(raw) if path.suffix.lower() != ".jsonl" else [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = rows.get("queries", rows.get("data", []))
    result = {}
    for row in rows:
        qid = row.get("query_id", row.get("qid", row.get("id")))
        text = row.get("query", row.get("text", row.get("query_text", "")))
        if qid is not None:
            result[str(qid)] = str(text)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load the deliberately small YAML config, with a dependency-free fallback."""
    path = Path(path)
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        data: Dict[str, Any] = {"dataset": {}, "budgets": [], "device": None, "output": None}
        section = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":") and not line.startswith("-"):
                section = line[:-1]
                if section == "dataset": data[section] = {}
                continue
            if line.startswith("-"):
                data.setdefault(section, []).append(int(line[1:].strip()))
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            value = value.strip("'\"")
            if section == "dataset": data[section][key] = value
            else: data[key] = value
    dataset = data.get("dataset", {})
    budgets = [int(x) for x in data.get("budgets", list(BUDGETS))]
    if budgets != list(BUDGETS):
        raise ValueError(f"profile config budgets must be {list(BUDGETS)}, got {budgets}")
    if str(data.get("device", "cpu")).lower() != "cpu":
        raise ValueError("profiling must run with device: cpu")
    required = ["query_path", "collection_path", "index_path"]
    missing = [key for key in required if not dataset.get(key)]
    if missing:
        raise ValueError(f"profile config missing dataset fields: {missing}")
    return {"dataset": dataset, "budgets": budgets, "device": "cpu",
            "output": data.get("output", "budget_profile_full.jsonl")}


def _is_complete(profile: QueryBudgetProfile, budgets=BUDGETS) -> bool:
    return all(profile.budgets[str(b)].latency_ms is not None for b in budgets)


def _print_summary(profiles: List[QueryBudgetProfile], output: str | Path, budgets=BUDGETS) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"total_queries": len(profiles), "budget_coverage": {}, "average_latency_ms": {}}
    for budget in budgets:
        values = [p.budgets[str(budget)].latency_ms for p in profiles]
        valid = [float(value) for value in values if value is not None]
        summary["budget_coverage"][str(budget)] = f"{len(valid)}/{len(profiles)}"
        summary["average_latency_ms"][str(budget)] = statistics.mean(valid) if valid else None
    summary_path = Path(output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_live_dataset(config_path: str | Path = "configs/profile_config.yaml",
                     output: str | Path | None = None, resume: bool = False) -> List[QueryBudgetProfile]:
    """Execute real fixed and staged runs through existing runtime functions.

    This function is intended to run on the configured CPU server, where the
    paths and ColBERT dependencies used by ``server_two_stage_runtime`` exist.
    It deliberately does not duplicate candidate generation or scoring.
    """
    config = load_config(config_path)
    os.environ["PLAID_DEVICE"] = "cpu"
    import server_two_stage_runtime as runtime
    dataset = config["dataset"]
    qas = runtime.queries(Path(dataset["query_path"]))
    selected = sorted(qas)
    output = output or config["output"]
    existing = read_jsonl(output) if resume and Path(output).exists() else []
    completed = {p.query_id for p in existing if _is_complete(p)}
    profiles: List[QueryBudgetProfile] = list(existing)
    searcher = runtime.create_searcher(Path(dataset["index_path"]), Path(dataset["collection_path"]))
    if selected:
        runtime.warmup(searcher, qas[selected[0]]["text"])
    for qid in selected:
        if str(qid) in completed:
            print(json.dumps({"query_id": str(qid), "status": "resume_skip"}, ensure_ascii=False), flush=True)
            continue
        info = qas[qid]
        group = str(info.get("group", "unknown"))
        rows = []
        for budget in BUDGETS:
            fixed = runtime.run_full(searcher, info["text"], info["positives"], dataset.get("name", "profile"), qid, group, 0, budget)
            staged = runtime.run_two_stage(searcher, info["text"], info["positives"], dataset.get("name", "profile"), qid, group, 0, budget)
            merged = dict(fixed)
            merged["approx_features_json"] = staged.get("approx_features_json", "{}")
            merged["candidate_pids_json"] = staged.get("candidate_pids_json")
            merged["approx_top100_pids_json"] = staged.get("approx_top100_pids_json")
            merged["candidate_generation_failure"] = staged.get("candidate_generation_failure")
            merged["candidate_count"] = staged.get("candidate_count", budget)
            rows.append(merged)
            print(json.dumps({"query_id": str(qid), "current_budget": budget,
                              "elapsed_ms": merged.get("total_retrieval_latency_ms"),
                              "recall@100": merged.get("recall_at_100"),
                              "candidate_count": merged.get("candidate_count")}, ensure_ascii=False), flush=True)
        profile = profiles_from_runtime_rows(rows, {str(qid): info["text"]})[0]
        profiles = [p for p in profiles if p.query_id != str(qid)] + [profile]
        write_jsonl(profiles, output)
    write_jsonl(profiles, output)
    _print_summary(profiles, output)
    return profiles


def collect_query_profiles(searcher, qtext: str, positives, dataset: str, qid: str,
                           group: str = "unknown", repeat: int = 0) -> QueryBudgetProfile:
    """Call the existing runtime functions for one query and normalize results.

    ``run_full`` remains the fixed-budget measurement. ``run_two_stage`` is
    used only to supply checkpoint features; neither function is reimplemented
    here. This function is intended for the remote execution environment where
    the existing ColBERT dependencies and searcher are available.
    """
    from server_two_stage_runtime import run_full, run_two_stage

    rows: List[Dict[str, Any]] = []
    for budget in BUDGETS:
        fixed = run_full(searcher, qtext, positives, dataset, qid, group, repeat, budget)
        staged = run_two_stage(searcher, qtext, positives, dataset, qid, group, repeat, budget)
        merged = dict(fixed)
        # Preserve fixed latency/metrics while taking only measured checkpoint
        # signals from the existing staged execution.
        merged["approx_features_json"] = staged.get("approx_features_json", "{}")
        merged["candidate_generation_failure"] = staged.get("candidate_generation_failure")
        merged["candidate_count"] = staged.get("candidate_count", budget)
        rows.append(merged)
    return profiles_from_runtime_rows(rows, {str(qid): qtext})[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", help="existing runtime CSV")
    parser.add_argument("--query-file", help="raw JSON/JSONL query file")
    parser.add_argument("--execute", action="store_true", help="run the existing runtime on the CPU server")
    parser.add_argument("--config", default="configs/profile_config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.execute:
        config = load_config(args.config)
        run_live_dataset(args.config, args.output or config["output"], args.resume)
        return
    if not args.from_csv:
        parser.error("--from-csv is required unless --execute is used")
    profiles = csv_to_profiles(args.from_csv)
    if args.query_file:
        query_texts = load_query_texts(args.query_file)
        for profile in profiles:
            profile.query_text = query_texts.get(profile.query_id, "")
    output = args.output or "budget_profile_full.jsonl"
    write_jsonl(profiles, output)
    print(json.dumps({"query_count": len(profiles), "budgets": list(BUDGETS), "output": output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
