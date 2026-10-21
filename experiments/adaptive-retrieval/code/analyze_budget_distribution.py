"""Generate budget distribution and quality-latency analysis artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from analyze_budget_oracle import analyze
from profile_io import read_jsonl
from profiling_schema import BUDGETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_jsonl")
    parser.add_argument("--output-dir", default="budget_analysis")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    profiles = read_jsonl(args.profile_jsonl)
    oracle_rows = analyze(profiles)

    # Per-query feasible-minimal budgets at the requested quality levels.
    summary = {"query_count": len(profiles), "oracle_query_count": len(oracle_rows), "budgets": list(BUDGETS)}
    for alpha in (0.98, 0.95):
        key = f"oracle_budget_{alpha:.2f}"
        counter = Counter(str(row[key]) for row in oracle_rows if row.get(key) is not None)
        summary[f"oracle_distribution_{alpha:.2f}"] = dict(counter)
        summary[f"not_need_1024_fraction_{alpha:.2f}"] = (
            sum(row.get(key) not in (None, 1024) for row in oracle_rows) / len(oracle_rows)
            if oracle_rows else None)

    # Explicit checks for the paper hypothesis. These remain null when the
    # profile is incomplete rather than silently treating missing budgets as
    # zero quality.
    ref_qualities = []
    q400_95 = []
    for profile in profiles:
        values = profile.to_dict()["budgets"]
        ref = values["1024"]["metrics"].get("recall@100")
        q400 = values["400"]["metrics"].get("recall@100")
        if ref is not None:
            ref_qualities.append((float(ref), q400))
            q400_95.append(q400 is not None and float(q400) >= 0.95 * float(ref))
    summary["fraction_400_reaches_95pct_of_1024"] = (sum(q400_95) / len(q400_95)) if q400_95 else None
    dist95 = summary.get("oracle_distribution_0.95", {})
    oracle_count = sum(int(v) for v in dist95.values())
    summary["fraction_oracle_budget_over_512_alpha_0.95"] = (
        sum(int(v) for k, v in dist95.items() if k not in {"None", "400", "512"}) / oracle_count
        if oracle_count else None)
    gains = []
    for profile in profiles:
        values = profile.to_dict()["budgets"]
        q400 = values["400"]["metrics"].get("recall@100")
        q1024 = values["1024"]["metrics"].get("recall@100")
        if q400 is not None and q1024 is not None:
            gains.append(float(q1024) - float(q400))
    summary["quality_gain_400_to_1024"] = {
        "mean": statistics.mean(gains) if gains else None,
        "max": max(gains) if gains else None,
        "p90": (statistics.quantiles(gains, n=10)[8] if len(gains) >= 2 else (gains[0] if gains else None)),
        "count": len(gains),
    }
    summary["query_complexity_correlation"] = None
    summary["query_complexity_correlation_note"] = "未提供 query complexity 特征，当前不做相关性推断。"

    # Mean quality/latency curve for fixed budgets.
    curve = []
    for budget in BUDGETS:
        values = [p.to_dict()["budgets"][str(budget)] for p in profiles]
        quality = [v["metrics"].get("recall@100") for v in values if v["metrics"].get("recall@100") is not None]
        latency = [v.get("latency_ms") for v in values if v.get("latency_ms") is not None]
        curve.append({"method": f"fixed{budget}", "budget": budget,
                      "mean_recall@100": statistics.mean(quality) if quality else None,
                      "mean_latency_ms": statistics.mean(latency) if latency else None,
                      "coverage": len(quality)})
    for alpha in (0.98, 0.95):
        key = f"oracle_budget_{alpha:.2f}"
        latencies, qualities = [], []
        by_id = {p.query_id: p for p in profiles}
        for row in oracle_rows:
            budget = row.get(key)
            if budget is None: continue
            item = by_id[row["query_id"]].to_dict()["budgets"][str(budget)]
            if item.get("latency_ms") is not None and item["metrics"].get("recall@100") is not None:
                latencies.append(item["latency_ms"]); qualities.append(item["metrics"]["recall@100"])
        curve.append({"method": f"oracle{alpha:.2f}", "budget": None,
                      "mean_recall@100": statistics.mean(qualities) if qualities else None,
                      "mean_latency_ms": statistics.mean(latencies) if latencies else None,
                      "coverage": len(qualities)})
    with (out / "quality_latency_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys()); writer.writeheader(); writer.writerows(curve)

    # Per-query spread of available latency/quality-optimal budgets.
    variance_rows = []
    for profile in profiles:
        available = []
        for budget in BUDGETS:
            item = profile.to_dict()["budgets"][str(budget)]
            if item.get("latency_ms") is not None and item["metrics"].get("recall@100") is not None:
                available.append((budget, item["metrics"]["recall@100"], item["latency_ms"]))
        best = min(available, key=lambda x: (-x[1], x[2]))[0] if available else None
        variance_rows.append({"query_id": profile.query_id, "available_budget_count": len(available),
                              "best_quality_budget": best,
                              "budget_range": (max(x[0] for x in available) - min(x[0] for x in available)) if available else None})
    with (out / "per_query_budget_variance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=variance_rows[0].keys() if variance_rows else ["query_id"])
        writer.writeheader(); writer.writerows(variance_rows)

    # Plots are optional; CSV/JSON outputs remain the source of truth.
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4)); plt.bar(list(summary["oracle_distribution_0.95"]), list(summary["oracle_distribution_0.95"].values())); plt.xlabel("Oracle budget (alpha=0.95)"); plt.ylabel("Query count"); plt.tight_layout(); plt.savefig(out / "budget_histogram.png", dpi=160); plt.close()
        plt.figure(figsize=(6, 4));
        for row in curve:
            if row["mean_latency_ms"] is not None and row["mean_recall@100"] is not None:
                plt.scatter(row["mean_latency_ms"], row["mean_recall@100"], label=row["method"])
        plt.xlabel("Mean latency (ms/query)"); plt.ylabel("Mean Recall@100"); plt.legend(); plt.tight_layout(); plt.savefig(out / "quality_latency_curve.png", dpi=160); plt.close()
    except Exception as exc:
        summary["plot_warning"] = str(exc)
    (out / "budget_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
