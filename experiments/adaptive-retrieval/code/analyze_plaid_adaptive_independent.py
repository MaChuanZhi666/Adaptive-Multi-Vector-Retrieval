#!/usr/bin/env python3
"""Analyze the frozen-rule PLAID independent-query online validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "plaid_adaptive_independent_queries.csv"
META = ROOT / "plaid_adaptive_independent_metadata.json"
SUMMARY = ROOT / "plaid_adaptive_independent_summary.csv"
CONFUSION = ROOT / "plaid_adaptive_independent_confusion.csv"
REPORT = ROOT / "plaid_adaptive_independent_analysis.md"

STRATEGIES = ["original_baseline", "fixed_ndocs_400", "adaptive_400_1024"]
LABELS = {
    "original_baseline": "Original baseline",
    "fixed_ndocs_400": "Fixed ndocs=400",
    "adaptive_400_1024": "Adaptive 400/1024",
}
QUALITY = ["success_at_10", "mrr_at_10", "recall_at_10", "recall_at_100"]
EPS = 1e-12


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def safe_div(a, b):
    return float(a / b) if b else np.nan


def validate(df, meta):
    required = {
        "query_id", "repeat", "strategy", "query_embedding_norm_std",
        "centroid_top1_top2_gap_mean", "predicted_hard", "actual_ndocs",
        "total_latency_ms", "feature_computation_latency_ms",
        "retrieval_latency_ms", *QUALITY,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"raw CSV lacks required columns: {missing}")
    if set(df["strategy"].unique()) != set(STRATEGIES):
        raise ValueError(f"unexpected strategies: {sorted(df['strategy'].unique())}")
    nq = int(df["query_id"].nunique())
    if not 50 <= nq <= 100:
        raise ValueError(f"expected 50..100 independent queries, got {nq}")
    expected = nq * 3 * len(STRATEGIES)
    if len(df) != expected:
        raise ValueError(f"expected {expected} rows, got {len(df)}")
    id_overlap = meta.get("old_query_id_overlap_count", meta.get("old_query_overlap_count", -1))
    if int(id_overlap) != 0:
        raise ValueError("metadata does not prove zero ID overlap")
    if int(meta.get("old_query_text_overlap_count", -1)) != 0:
        raise ValueError("metadata does not prove zero text overlap")


def query_means(df):
    cols = [
        "total_latency_ms", "feature_computation_latency_ms", "retrieval_latency_ms",
        "query_embedding_norm_std", "centroid_top1_top2_gap_mean", *QUALITY,
    ]
    return df.groupby(["query_id", "strategy"], as_index=False)[cols].mean()


def confusion_row(perq, metric):
    wide = perq.pivot(index="query_id", columns="strategy", values=metric)
    actual = wide["fixed_ndocs_400"] < wide["original_baseline"] - EPS
    pred = (
        perq[perq.strategy == "adaptive_400_1024"]
        .set_index("query_id")["predicted_hard"]
        .astype(bool)
        .reindex(wide.index)
    )
    tp = int((pred & actual).sum()); fn = int((~pred & actual).sum())
    fp = int((pred & ~actual).sum()); tn = int((~pred & ~actual).sum())
    return {
        "hard_definition": f"fixed400 {metric} < original baseline {metric}",
        "quality_metric": metric, "num_queries": len(wide),
        "actual_hard": int(actual.sum()), "predicted_hard": int(pred.sum()),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "hard_recall": safe_div(tp, tp + fn),
        "hard_precision": safe_div(tp, tp + fp),
        "specificity": safe_div(tn, tn + fp),
        "accuracy": safe_div(tp + tn, len(wide)),
        "false_negative_query_ids": ",".join(map(str, wide.index[actual & ~pred].tolist())),
        "false_positive_query_ids": ",".join(map(str, wide.index[~actual & pred].tolist())),
    }


def make_summary(df, perq):
    rows = []
    for strategy in STRATEGIES:
        raw = df[df.strategy == strategy]
        pq = perq[perq.strategy == strategy]
        mean_ms = float(raw.total_latency_ms.mean())
        row = {
            "strategy": strategy, "label": LABELS[strategy],
            "num_queries": int(raw.query_id.nunique()), "repetitions": int(raw.repeat.nunique()),
            "latency_mean_ms": mean_ms,
            "latency_p50_ms": percentile(raw.total_latency_ms, 50),
            "latency_p95_ms": percentile(raw.total_latency_ms, 95),
            "qps": 1000.0 / mean_ms,
            "feature_computation_mean_ms": float(raw.feature_computation_latency_ms.mean()),
            "feature_cost_percent_of_total": safe_div(
                float(raw.feature_computation_latency_ms.mean()) * 100.0, mean_ms
            ),
        }
        for metric in QUALITY:
            row[metric] = float(pq[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(df, perq, summary, rule):
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    colors = palette[:len(STRATEGIES)]
    x = np.arange(len(STRATEGIES))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    means = summary.set_index("strategy").loc[STRATEGIES, "latency_mean_ms"]
    p95s = summary.set_index("strategy").loc[STRATEGIES, "latency_p95_ms"]
    ax.bar(x, means, color=colors, label="Mean")
    ax.scatter(x, p95s, color="black", marker="D", s=42, label="P95")
    ax.set_xticks(x, [LABELS[s] for s in STRATEGIES]); ax.set_ylabel("Latency (ms/query)")
    ax.set_title("Independent-query online latency (feature cost included)")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(ROOT / "plaid_independent_latency_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    vals = summary.set_index("strategy").loc[STRATEGIES, "recall_at_100"]
    bars = ax.bar(x, vals, color=colors)
    ax.set_xticks(x, [LABELS[s] for s in STRATEGIES]); ax.set_ylabel("Recall@100")
    ax.set_ylim(max(0, float(vals.min()) - .08), min(1.0, float(vals.max()) + .04))
    ax.set_title("Independent-query Recall@100")
    for bar, value in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.4f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(ROOT / "plaid_independent_recall100_comparison.png", dpi=180); plt.close(fig)

    wide = perq.pivot(index="query_id", columns="strategy", values="recall_at_100")
    delta = pd.DataFrame({
        "fixed400": wide.fixed_ndocs_400 - wide.original_baseline,
        "adaptive": wide.adaptive_400_1024 - wide.original_baseline,
    }).sort_values("fixed400")
    fig, ax = plt.subplots(figsize=(12, 5.4))
    xx = np.arange(len(delta)); ax.bar(xx-.2, delta.fixed400, .4, label="Fixed400 - baseline", color=colors[1])
    ax.bar(xx+.2, delta.adaptive, .4, label="Adaptive - baseline", color=colors[2])
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(xx, delta.index.astype(str), rotation=90, fontsize=7)
    ax.set_ylabel("Per-query Recall@100 delta"); ax.set_xlabel("Independent query ID")
    ax.set_title("Quality change relative to original baseline"); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / "plaid_independent_per_query_quality_delta.png", dpi=180); plt.close(fig)

    adapt = perq[perq.strategy == "adaptive_400_1024"].set_index("query_id")
    actual = wide.fixed_ndocs_400 < wide.original_baseline - EPS
    pred = adapt.predicted_hard.astype(bool)
    fig, ax = plt.subplots(figsize=(8, 6))
    for is_hard, marker, label, color in [(False, "o", "Actual easy", "#4C78A8"), (True, "^", "Actual hard", "#E45756")]:
        ids = actual.index[actual == is_hard]
        ax.scatter(adapt.loc[ids, "query_embedding_norm_std"], adapt.loc[ids, "centroid_top1_top2_gap_mean"],
                   marker=marker, c=color, s=55, alpha=.8, label=label,
                   linewidths=np.where(pred.loc[ids], 1.8, .3), edgecolors=np.where(pred.loc[ids], "black", "white"))
    r = rule.get("two_variable_or", rule)
    ax.axvline(r["threshold_1"], ls="--", color="#666", label="Frozen norm threshold")
    ax.axhline(r["threshold_2"], ls=":", color="#222", label="Frozen gap threshold")
    ax.set_xlabel("query_embedding_norm_std"); ax.set_ylabel("centroid_top1_top2_gap_mean")
    ax.set_title("Frozen two-feature rule on independent queries\nblack edge = predicted hard")
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(ROOT / "plaid_independent_two_feature_scatter.png", dpi=180); plt.close(fig)


def main():
    if not RAW.exists() or not META.exists():
        raise FileNotFoundError(f"missing raw result: {RAW} or {META}")
    df = pd.read_csv(RAW)
    meta = json.loads(META.read_text(encoding="utf-8"))
    rule = meta["frozen_rule"]
    validate(df, meta)
    df["predicted_hard"] = df["predicted_hard"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if df.predicted_hard.isna().any(): raise ValueError("invalid predicted_hard values")
    perq = query_means(df)
    pred_map = df[df.strategy == "adaptive_400_1024"].groupby("query_id").predicted_hard.first()
    perq["predicted_hard"] = perq.query_id.map(pred_map)

    summary = make_summary(df, perq)
    summary.to_csv(SUMMARY, index=False)
    confusion = pd.DataFrame([confusion_row(perq, m) for m in ["recall_at_100", "recall_at_10", "mrr_at_10"]])
    confusion.to_csv(CONFUSION, index=False)
    make_plots = plot_outputs(df, perq, summary, rule)

    s = summary.set_index("strategy")
    c = confusion.set_index("quality_metric").loc["recall_at_100"]
    base, fixed, adapt = s.loc["original_baseline"], s.loc["fixed_ndocs_400"], s.loc["adaptive_400_1024"]
    lost = base.recall_at_100 - fixed.recall_at_100
    restored = adapt.recall_at_100 - fixed.recall_at_100
    recovery = safe_div(restored, lost) * 100 if abs(lost) > EPS else np.nan
    speed = (1 - adapt.latency_mean_ms / base.latency_mean_ms) * 100

    wide_lat = perq.pivot(index="query_id", columns="strategy", values="total_latency_ms")
    actual_hard = perq.pivot(index="query_id", columns="strategy", values="recall_at_100").fixed_ndocs_400 < perq.pivot(index="query_id", columns="strategy", values="recall_at_100").original_baseline - EPS
    pred = perq[perq.strategy == "adaptive_400_1024"].set_index("query_id").predicted_hard.astype(bool)
    fp_ids = actual_hard.index[(~actual_hard) & pred]
    fp_extra = float((wide_lat.loc[fp_ids, "adaptive_400_1024"] - wide_lat.loc[fp_ids, "fixed_ndocs_400"]).mean()) if len(fp_ids) else 0.0
    all_extra = float((wide_lat.adaptive_400_1024 - wide_lat.fixed_ndocs_400).mean())

    def pct(v): return "NA" if pd.isna(v) else f"{100*v:.2f}%"
    lines = [
        "# PLAID adaptive independent-query validation",
        "",
        "## Validation boundary",
        "",
        f"- Independent queries: **{int(meta['num_queries'])}**; old-ID overlap: **0**; old-text overlap: **0**.",
        "- The two features, thresholds, and OR decision rule were frozen from the old 48-query analysis before this run.",
        "- New qrels, final quality, final latency, and TopK were used only after retrieval for evaluation; they did not tune the rule.",
        "- The existing PLAID 50k index, checkpoint, commit, CPU thread count, TopK=100, warm-up=5, and repetitions=3 were reused.",
        "",
        "## New independent-set online results",
        "",
        "| Strategy | Mean ms | P50 ms | P95 ms | QPS | Success@10 | MRR@10 | Recall@10 | Recall@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for st in STRATEGIES:
        r = s.loc[st]
        lines.append(f"| {LABELS[st]} | {r.latency_mean_ms:.3f} | {r.latency_p50_ms:.3f} | {r.latency_p95_ms:.3f} | {r.qps:.3f} | {r.success_at_10:.4f} | {r.mrr_at_10:.4f} | {r.recall_at_10:.4f} | {r.recall_at_100:.4f} |")
    lines += [
        "", "## Frozen-rule classification on the new set", "",
        f"- Recall@100-defined actual hard queries: **{int(c.actual_hard)}**.",
        f"- TP/FN/FP/TN: **{int(c.tp)}/{int(c.fn)}/{int(c.fp)}/{int(c.tn)}**.",
        f"- Hard-query recall: **{pct(c.hard_recall)}**; precision: **{pct(c.hard_precision)}**.",
        f"- False-negative IDs: `{c.false_negative_query_ids or 'none'}`; false-positive IDs: `{c.false_positive_query_ids or 'none'}`.",
        "", "## Answers to the validation questions", "",
        f"- **A — Quality recovery:** fixed400 lost {lost:.6f} mean Recall@100; adaptive restored {restored:.6f} ({'NA' if pd.isna(recovery) else f'{recovery:.2f}%'} of that loss).",
        f"- **B — Speed:** adaptive is {speed:.2f}% {'faster' if speed >= 0 else 'slower'} than original baseline by mean end-to-end latency.",
        f"- **C — Feature cost:** {adapt.feature_computation_mean_ms:.6f} ms/query, {adapt.feature_cost_percent_of_total:.4f}% of adaptive end-to-end latency.",
        f"- **D — Hard recall:** {pct(c.hard_recall)} under the primary Recall@100 hard definition, with {int(c.fn)} false negatives.",
        f"- **E — False-positive cost:** {int(c.fp)} easy queries were promoted. Their mean adaptive-minus-fixed400 latency was {fp_extra:.3f} ms; averaged across all queries, adaptive added {all_extra:.3f} ms over fixed400.",
        f"- **F — Policy support:** {'The independent result supports the simple default-400/promote-to-1024 policy.' if (c.fn == 0 and adapt.recall_at_100 + EPS >= base.recall_at_100 and adapt.latency_mean_ms < base.latency_mean_ms) else 'The independent result does not fully validate all three desired conditions (no missed hard query, baseline-level Recall@100, and lower latency); treat the policy as preliminary.'}",
        "", "## Old 48-query result versus this validation", "",
        "- Old 48-query offline replay: baseline 160.12 ms and Recall@100 0.8704; fixed400 133.06 ms and 0.8501; adaptive 138.88 ms and 0.8704; Oracle 130.47 ms and 0.8704.",
        "- Those old numbers are historical/offline and are not pooled with the independent-set statistics above.",
        "- Only behavior reproduced on this zero-overlap set is evidence of limited generalization. With at most 100 queries and potentially few hard cases, uncertainty remains high; this is a small validation, not a population-level claim.",
        "- Oracle was not an online strategy and did not influence thresholds or decisions.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(confusion.to_string(index=False))
    print(f"saved: {SUMMARY}\n{CONFUSION}\n{REPORT}")


if __name__ == "__main__":
    main()
