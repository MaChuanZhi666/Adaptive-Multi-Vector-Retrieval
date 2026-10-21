"""Per-query oracle budget analysis from budget_profile.jsonl."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable

from profile_io import read_jsonl
from profiling_schema import BUDGETS


def _num(value: Any):
    return None if value is None else float(value)


def analyze(profiles, alphas=(0.98, 0.95), latency_by_query=None) -> list[dict]:
    rows = []
    for profile in profiles:
        values = profile.to_dict()["budgets"]
        ref = values["1024"]
        quality_ref = _num(ref["metrics"].get("recall@100"))
        per_query_latency = (latency_by_query or {}).get(profile.query_id, {})
        latency_ref = _num(per_query_latency.get(1024)) if latency_by_query is not None else _num(ref.get("latency_ms"))
        if quality_ref is None or latency_ref is None:
            continue
        result = {"query_id": profile.query_id, "quality_1024": quality_ref, "latency_1024_ms": latency_ref}
        for alpha in alphas:
            feasible = []
            for budget in BUDGETS:
                item = values[str(budget)]
                quality = _num(item["metrics"].get("recall@100"))
                latency = (_num(per_query_latency.get(budget)) if latency_by_query is not None
                           else _num(item.get("latency_ms")))
                if quality is not None and latency is not None and quality >= alpha * quality_ref:
                    feasible.append((latency, budget, quality))
            key = f"{alpha:.2f}"
            if not feasible:
                result[f"oracle_budget_{key}"] = None
                result[f"latency_saved_vs_1024_{key}_ms"] = None
                result[f"quality_loss_vs_1024_{key}"] = None
            else:
                latency, budget, quality = min(feasible, key=lambda item: (item[0], item[1]))
                result[f"oracle_budget_{key}"] = budget
                result[f"latency_saved_vs_1024_{key}_ms"] = latency_ref - latency
                result[f"quality_loss_vs_1024_{key}"] = quality_ref - quality
        rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_jsonl")
    parser.add_argument("--output", default="oracle_distribution.csv")
    parser.add_argument("--repeat-results", help="use per-query repeated median latency for oracle selection")
    args = parser.parse_args()
    latency_by_query = None
    if args.repeat_results:
        latency_by_query = {}
        with Path(args.repeat_results).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                runs = [float(value) for value in item.get("runs", [])]
                if len(runs) >= 5:
                    latency_by_query.setdefault(str(item["query_id"]), {})[int(item["budget"])] = float(
                        item.get("median_latency", statistics.median(runs)))
        latency_by_query = {qid: values for qid, values in latency_by_query.items()
                            if set(values) == set(BUDGETS)}
    rows = analyze(read_jsonl(args.profile_jsonl), latency_by_query=latency_by_query)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    for alpha in (0.98, 0.95):
        key = f"oracle_budget_{alpha:.2f}"
        counts = {}
        for row in rows:
            budget = row.get(key)
            counts[str(budget)] = counts.get(str(budget), 0) + 1
        print(json.dumps({"alpha": alpha, "query_count": len(rows), "distribution": counts}, ensure_ascii=False))
    if args.repeat_results:
        print(json.dumps({"label_source": "per-query median of 5 repeated runs",
                          "repeated_query_count": len(latency_by_query or {}),
                          "labeled_query_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
