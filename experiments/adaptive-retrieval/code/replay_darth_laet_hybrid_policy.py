"""Offline replay of the simple DARTH/LAET-inspired PLAID policy.

This does not run a benchmark.  It reuses existing runtime/features CSVs and
reports deployable policy decisions separately from qrels-based diagnostics.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from darth_laet_hybrid_policy import HybridPolicyConfig, choose_budget


ROOT = Path(__file__).resolve().parent
RESULTS = Path(__import__("os").environ.get("PLAID_RUNTIME_DIR", str(ROOT / "results_cpu_signal")))
FEATURES = RESULTS / "plaid_approx_dynamic_features_raw.csv"
RUNTIME = RESULTS / "plaid_two_stage_runtime.csv"
OUT = RESULTS / "plaid_darth_laet_hybrid_policy_replay.csv"
REPORT = RESULTS / "plaid_darth_laet_hybrid_policy_replay.md"


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


def main():
    raw = read(FEATURES)
    runtime = read(RUNTIME)
    fmap = {(r["dataset"], int(r["query_id"]), int(r["repeat"]), int(r["budget"])): r for r in raw}
    rmap = {(r["dataset"], int(r["query_id"]), int(r["repeat"]), r["scenario"]): r for r in runtime}
    keys = sorted({(r["dataset"], int(r["query_id"]), int(r["repeat"])) for r in runtime})
    cfg = HybridPolicyConfig()
    out = []
    for dataset, qid, rep in keys:
        r400, r512 = fmap[(dataset, qid, rep, 400)], fmap[(dataset, qid, rep, 512)]
        target = choose_budget(r400, r512, cfg)
        chosen = rmap[(dataset, qid, rep, f"TwoStage{target}")]
        full1024 = rmap[(dataset, qid, rep, "Full1024")]
        # qrels-based fields are explicitly diagnostic, never used by policy.
        out.append({
            "dataset": dataset, "query_id": qid, "repeat": rep,
            "chosen_budget": target,
            "darth_confidence_400": r400.get("approx_topk_overlap_stage1_vs_refine"),
            "darth_confidence_512": r512.get("approx_topk_overlap_stage1_vs_refine"),
            "laet_stable_400": str(target == 400).lower(),
            "laet_stable_512": str(target == 512).lower(),
            "recall_at_100": chosen.get("recall_at_100"),
            "latency_ms": chosen.get("total_retrieval_latency_ms"),
            "full1024_recall_at_100": full1024.get("recall_at_100"),
            "candidate_coverage_diagnostic": r400.get("candidate_generation_recall_upper_bound"),
            "candidate_generation_failure_diagnostic": r400.get("candidate_generation_failure"),
        })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        fields = list(out[0]) if out else []
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    lines = ["# DARTH + LAET inspired PLAID policy replay", "",
             "This is an offline replay over existing CSVs; no benchmark was rerun.",
             "The policy uses only checkpoint-local approximate signals. Candidate coverage fields are qrels-based diagnostics and are not used for decisions.", ""]
    for dataset in sorted({r["dataset"] for r in out}):
        rows = [r for r in out if r["dataset"] == dataset]
        lines += [f"## {dataset}", "",
                   f"- queries: {len(rows)}",
                   f"- stop@400: {sum(r['chosen_budget'] == 400 for r in rows)}/{len(rows)}",
                   f"- stop@512: {sum(r['chosen_budget'] == 512 for r in rows)}/{len(rows)}",
                   f"- continue@1024: {sum(r['chosen_budget'] == 1024 for r in rows)}/{len(rows)}",
                   f"- mean Recall@100: {mean(rows, 'recall_at_100'):.4f}",
                   f"- mean latency: {mean(rows, 'latency_ms'):.3f} ms",
                   f"- mean Always1024 Recall@100 reference: {mean(rows, 'full1024_recall_at_100'):.4f}",
                   f"- candidate-generation failure diagnostic: {sum(str(r['candidate_generation_failure_diagnostic']).lower() == 'true' for r in rows)}/{len(rows)}", ""]
    lines += ["## Interpretation", "",
              "The overlap threshold is the DARTH-style confidence proxy; low tail variance and a small newly-admitted fraction are LAET-style stability guards.",
              "This replay is a mechanism check, not a claim of final adaptive success. A production candidate-coverage proxy still needs to be instrumented because true qrels coverage is unavailable online.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(out), "csv": str(OUT), "report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
