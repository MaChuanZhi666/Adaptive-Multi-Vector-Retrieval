#!/usr/bin/env python3
"""Local final analysis for the separately indexed independent validation."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "analyze_plaid_adaptive_independent.py"
spec = importlib.util.spec_from_file_location("base_independent_analysis", BASE_PATH)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

base.ROOT = ROOT
base.RAW = ROOT / "plaid_independent_validation_per_query.csv"
base.META = ROOT / "plaid_independent_validation_run_metadata.json"
base.SUMMARY = ROOT / "plaid_independent_validation_summary.csv"
base.CONFUSION = ROOT / "plaid_independent_validation_confusion.csv"
base.REPORT = ROOT / "plaid_independent_validation_analysis.md"
base.STRATEGIES = ["original_baseline", "fixed_ndocs_400", "adaptive_400_1024", "fixed_ndocs_1024"]
base.LABELS = {
    "original_baseline": "Original baseline",
    "fixed_ndocs_400": "Fixed ndocs=400",
    "adaptive_400_1024": "Adaptive 400/1024",
    "fixed_ndocs_1024": "Fixed ndocs=1024",
}


def extras():
    df = pd.read_csv(base.RAW)
    confusion = pd.read_csv(base.CONFUSION)
    primary = confusion[confusion.quality_metric == "recall_at_100"].iloc[0]
    matrix = np.array([[primary.tn, primary.fp], [primary.fn, primary.tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted easy", "Predicted hard"])
    ax.set_yticks([0, 1], ["Actual easy", "Actual hard"])
    for i in range(2):
        for j in range(2): ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=16)
    ax.set_title("Frozen-rule confusion (Recall@100 hard label)")
    fig.colorbar(im, ax=ax, fraction=.046); fig.tight_layout()
    fig.savefig(ROOT / "plaid_independent_confusion_visualization.png", dpi=180); plt.close(fig)

    means = df.groupby(["query_id", "strategy"], as_index=False).total_latency_ms.mean()
    wide = means.pivot(index="query_id", columns="strategy", values="total_latency_ms")
    saving = (wide.original_baseline - wide.adaptive_400_1024).sort_values()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    colors = np.where(saving >= 0, "#54A24B", "#E45756")
    ax.bar(np.arange(len(saving)), saving.values, color=colors)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(np.arange(len(saving)), saving.index.astype(str), rotation=90, fontsize=7)
    ax.set_ylabel("Baseline − adaptive latency (ms)"); ax.set_xlabel("Independent query ID")
    ax.set_title("Per-query adaptive latency saving"); fig.tight_layout()
    fig.savefig(ROOT / "plaid_independent_adaptive_latency_saving.png", dpi=180); plt.close(fig)

    renames = {
        "plaid_independent_latency_comparison.png": "plaid_independent_four_strategy_latency.png",
        "plaid_independent_recall100_comparison.png": "plaid_independent_four_strategy_recall100.png",
        "plaid_independent_two_feature_scatter.png": "plaid_independent_actual_predicted_feature_scatter.png",
    }
    for src, dst in renames.items(): shutil.copy2(ROOT / src, ROOT / dst)

    summary = pd.read_csv(base.SUMMARY).set_index("strategy")
    perq = df.groupby(["query_id", "strategy"], as_index=False).agg(
        total_latency_ms=("total_latency_ms", "mean"),
        recall_at_100=("recall_at_100", "mean"),
        predicted_hard=("predicted_hard", "first"),
        feature_ms=("feature_computation_latency_ms", "mean"),
        norm=("query_embedding_norm_std", "mean"),
        gap=("centroid_top1_top2_gap_mean", "mean"),
    )
    adaptive = perq[perq.strategy == "adaptive_400_1024"].set_index("query_id")
    promoted = adaptive.predicted_hard.astype(str).str.lower().isin(["true", "1"])
    fixed1024 = summary.loc["fixed_ndocs_1024"]
    adapts = summary.loc["adaptive_400_1024"]
    baseline = summary.loc["original_baseline"]
    fixed400 = summary.loc["fixed_ndocs_400"]
    reduction_baseline = (1 - adapts.latency_mean_ms / baseline.latency_mean_ms) * 100
    reduction_1024 = (1 - adapts.latency_mean_ms / fixed1024.latency_mean_ms) * 100
    recall_loss = baseline.recall_at_100 - fixed400.recall_at_100
    recovery = adapts.recall_at_100 - fixed400.recall_at_100
    recovery_pct = np.nan if abs(recall_loss) < 1e-12 else recovery / recall_loss * 100
    primary = confusion[confusion.quality_metric == "recall_at_100"].iloc[0]
    recall_wide = perq.pivot(index="query_id", columns="strategy", values="recall_at_100")
    actual_hard = recall_wide.fixed_ndocs_400 < recall_wide.original_baseline - 1e-12
    norm_hit = adaptive["norm"] <= 3.5723282110211585e-08
    gap_hit = adaptive["gap"] <= 0.0400056634098291
    fn_mask = actual_hard & ~promoted
    fn_recall_delta = float((recall_wide.adaptive_400_1024 - recall_wide.original_baseline)[fn_mask].mean()) if fn_mask.any() else 0.0
    collection_meta = json.loads((ROOT / "plaid_independent_validation_collection_metadata.json").read_text(encoding="utf-8"))
    run_meta = json.loads((ROOT / "plaid_independent_validation_run_metadata.json").read_text(encoding="utf-8"))
    excluded = sorted(set(collection_meta["selected_query_ids"]) - set(run_meta["selected_query_ids"]))
    verdict = (
        "supports limited generalization on this independent validation set"
        if primary.fn == 0 and adapts.recall_at_100 + 1e-12 >= baseline.recall_at_100 and adapts.latency_mean_ms < baseline.latency_mean_ms
        else "does not fully validate the policy; at least one required condition failed"
    )
    report = base.REPORT.read_text(encoding="utf-8")
    report = report.replace(
        "The existing PLAID 50k index, checkpoint, commit, CPU thread count, TopK=100, warm-up=5, and repetitions=3 were reused.",
        "A separate deterministic 50k validation collection and separate PLAID index were built; the checkpoint, commit, CPU thread count, TopK=100, warm-up=5, and repetitions=3 match the prior protocol. The original formal index was not modified.",
    )
    report += (
        "\n## Independent-index policy details\n\n"
        f"- Promoted queries: **{int(promoted.sum())}/{len(promoted)} ({promoted.mean()*100:.2f}%)**.\n"
        f"- Adaptive mean-latency reduction vs original baseline: **{reduction_baseline:.2f}%**.\n"
        f"- Adaptive mean-latency reduction vs fixed1024: **{reduction_1024:.2f}%**.\n"
        f"- Recall@100 recovered over fixed400: **{recovery:.6f}** ({'NA' if np.isnan(recovery_pct) else f'{recovery_pct:.2f}%'} of fixed400 loss).\n"
        f"- Frozen detector specificity: **{primary.specificity:.4f}**.\n"
        f"- Frozen norm threshold hit **{int((actual_hard & norm_hit).sum())}/{int(actual_hard.sum())}** actual-hard and **{int((~actual_hard & norm_hit).sum())}** actual-easy queries; frozen centroid-gap threshold hit **{int((actual_hard & gap_hit).sum())}/{int(actual_hard.sum())}** actual-hard and **{int((~actual_hard & gap_hit).sum())}** actual-easy queries.\n"
        f"- Hard/easy feature means: embedding-norm std **{adaptive.loc[actual_hard, 'norm'].mean():.3e}/{adaptive.loc[~actual_hard, 'norm'].mean():.3e}**; centroid gap **{adaptive.loc[actual_hard, 'gap'].mean():.6f}/{adaptive.loc[~actual_hard, 'gap'].mean():.6f}**. The old norm threshold captured no new hard query, while the gap threshold captured only 3/9.\n"
        f"- The false negatives averaged **{fn_recall_delta:.6f}** adaptive-minus-baseline Recall@100. This is the direct quality cost of missed promotions.\n"
        f"- Collection construction selected 100 zero-overlap queries. Formal timing used 94 after a safety check required at least 1,024 early candidates; excluded IDs were `{','.join(map(str, excluded))}`. This did not use qrels quality or latency, but it may still introduce candidate-rich selection bias and must be disclosed.\n"
        f"- Overall judgment: the result **{verdict}**. A distribution shift from constructing a new positive-complete collection remains a limitation; the frozen thresholds were not changed.\n"
        "- If performance differs from the old 48-query replay, the experiment can establish detector/threshold transfer failure and collection-distribution change, but cannot uniquely attribute causality among them without another independently pre-registered validation set.\n"
    )
    base.REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    base.main()
    extras()
