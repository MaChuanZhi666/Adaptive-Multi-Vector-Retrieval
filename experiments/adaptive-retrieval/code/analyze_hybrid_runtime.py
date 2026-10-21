"""Summarize the online HybridAdaptive runtime rows against fixed baselines."""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("PLAID_RUNTIME_DIR", str(ROOT / "results_cpu_signal")))
RUNTIME = RESULTS / "plaid_two_stage_runtime.csv"
OUT = RESULTS / "plaid_hybrid_adaptive_policy_results.csv"
REPORT = RESULTS / "plaid_hybrid_adaptive_runtime_report.md"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(rows, key):
    xs = [num(r.get(key)) for r in rows]
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x):
    return "NA" if x is None else f"{x:.4f}"


def main():
    rows = read(RUNTIME)
    keys = sorted({(r["dataset"], r["query_id"], r["repeat"]) for r in rows})
    rmap = {(r["dataset"], r["query_id"], r["repeat"], r["scenario"]): r for r in rows}
    summaries = []
    trace_rows = []
    for dataset in sorted({r["dataset"] for r in rows}):
        dskeys = [k for k in keys if k[0] == dataset]
        for policy in ("Full400", "Full512", "Full1024", "HybridAdaptive"):
            chosen = [rmap[k + (policy,)] for k in dskeys]
            budgets = [int(r["budget"]) for r in chosen]
            summaries.append({
                "dataset": dataset, "policy": policy, "n": len(chosen),
                "recall_at_100": mean(chosen, "recall_at_100"),
                "latency_mean_ms": mean(chosen, "total_retrieval_latency_ms"),
                "latency_p50_ms": statistics.median([num(r["total_retrieval_latency_ms"]) for r in chosen]),
                "latency_p95_ms": sorted(num(r["total_retrieval_latency_ms"]) for r in chosen)[int(.95 * (len(chosen)-1))],
                "stop400_fraction": sum(b == 400 for b in budgets) / len(budgets),
                "stop512_fraction": sum(b == 512 for b in budgets) / len(budgets),
                "continue1024_fraction": sum(b == 1024 for b in budgets) / len(budgets),
            })
        for k in dskeys:
            r = rmap[k + ("HybridAdaptive",)]
            trace = json.loads(r["approx_features_json"])
            trace_rows.append({"dataset": k[0], "query_id": k[1], "repeat": k[2],
                               "selected_budget": r["budget"],
                               "decision_trace": json.dumps(trace.get("decision_trace", []), ensure_ascii=False),
                               "recall_at_100": r["recall_at_100"],
                               "latency_ms": r["total_retrieval_latency_ms"],
                               "candidate_generation_upper_bound": r["candidate_generation_recall_upper_bound"]})
    with OUT.open("w", newline="", encoding="utf-8") as f:
        fields = list(summaries[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summaries)
    trace_out = RESULTS / "plaid_hybrid_adaptive_decision_trace.csv"
    with trace_out.open("w", newline="", encoding="utf-8") as f:
        fields = list(trace_rows[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(trace_rows)
    lines = ["# PLAID online HybridAdaptive runtime report", "",
             "The HybridAdaptive rows were executed online on CPU. Decisions used only approximate checkpoint signals; qrels were used only after timing to compute Recall.",
             "No index rebuild or parameter sweep was performed.", "", "| Dataset | Policy | Recall@100 | Mean ms | P50 ms | P95 ms | stop400 | stop512 | continue1024 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['dataset']} | {s['policy']} | {fmt(s['recall_at_100'])} | {fmt(s['latency_mean_ms'])} | {fmt(s['latency_p50_ms'])} | {fmt(s['latency_p95_ms'])} | {fmt(s['stop400_fraction'])} | {fmt(s['stop512_fraction'])} | {fmt(s['continue1024_fraction'])} |")
    lines += ["", "## Interpretation", "", "The online policy is now runnable end-to-end. Its output must be compared with fixed baselines on the same CPU host. Candidate-generation upper bound is an offline diagnostic and is not used in the decision.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
