"""Offline analysis of the frozen two-stage 45-query runtime outputs."""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("PLAID_RUNTIME_DIR", str(ROOT)))
RUNTIME_DEVICE = os.environ.get("PLAID_RUNTIME_DEVICE", "unknown")
RUNTIME = RESULTS_DIR / "plaid_two_stage_runtime.csv"
FEATURE_RAW = RESULTS_DIR / "plaid_approx_dynamic_features_raw.csv"
FEATURE_OUT = RESULTS_DIR / "plaid_approx_dynamic_features.csv"
STABILITY_OUT = RESULTS_DIR / "plaid_approx_signal_stability.csv"
POLICY_OUT = RESULTS_DIR / "plaid_two_stage_policy_results.csv"
REPORT_OUT = RESULTS_DIR / "plaid_two_stage_adaptive_final_report.md"
FEATURES = [
    "approx_cutoff_margin", "tail_slope", "tail_variance",
    "candidate_overlap_previous", "newly_admitted_fraction",
    "approx_score_entropy", "refine_saturation",
    "stage2_new_candidate_fraction", "approx_topk_overlap_stage1_vs_refine",
    "approx_score_gain_vs_previous_budget",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def flag(x):
    return str(x).lower() in {"true", "1", "yes"}


def auc(values, labels):
    pairs = [(v, bool(y)) for v, y in zip(values, labels) if v is not None]
    pos = [v for v, y in pairs if y]; neg = [v for v, y in pairs if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def effect_size(hard, easy):
    if len(hard) < 2 or len(easy) < 2:
        return None
    vh, ve = statistics.variance(hard), statistics.variance(easy)
    denom = math.sqrt(((len(hard)-1)*vh + (len(easy)-1)*ve) / (len(hard)+len(easy)-2))
    return (statistics.mean(hard)-statistics.mean(easy))/denom if denom else 0.0


def percentile(values, p):
    xs = sorted(values)
    if not xs: return None
    pos = (len(xs)-1)*p; lo = int(pos); hi = min(lo+1, len(xs)-1)
    return xs[lo] + (xs[hi]-xs[lo])*(pos-lo)


def fmt(x, digits=4):
    return "NA" if x is None or not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def query_features(raw):
    groups = {}
    for r in raw:
        key = (r["dataset"], int(r["query_id"]), int(r["budget"]))
        groups.setdefault(key, []).append(r)
    out = []
    for (dataset, qid, budget), rows in sorted(groups.items()):
        row = {"dataset": dataset, "query_id": qid, "budget": budget,
               "query_group": rows[0]["query_group"],
               "truncation_risk_hard": flag(rows[0].get("truncation_risk_at_budget", rows[0].get("truncation_risk_hard"))),
               "candidate_generation_failure": flag(rows[0].get("candidate_generation_failure")),
               "any_quality_drop_at_400": flag(rows[0].get("any_quality_drop_at_400")),
               "candidate_generation_recall_upper_bound": statistics.mean([num(x["candidate_generation_recall_upper_bound"]) for x in rows if num(x.get("candidate_generation_recall_upper_bound")) is not None])}
        for feature in FEATURES:
            vals = [num(r.get(feature)) for r in rows]
            vals = [x for x in vals if x is not None]
            row[feature] = statistics.mean(vals) if vals else None
        for metric in ("recall_at_100_400", "recall_at_100_512", "recall_at_100_1024"):
            row[metric] = num(rows[0].get(metric))
        out.append(row)
    return out


def threshold_fit(train, feature, budget):
    label_key = "truncation_risk_hard"
    data = [r for r in train if int(r["budget"]) == budget and r.get(feature) is not None]
    labels = [bool(r[label_key]) for r in data]
    positives = sum(labels)
    if not data or not positives:
        return {"feature": feature, "budget": budget, "direction": "conservative_all_risk",
                "threshold": None, "all_risk": True, "train_recall": 1.0 if positives == 0 else None,
                "train_precision": positives / len(data) if data else None}
    x = [float(r[feature]) for r in data]
    raw_auc = auc(x, labels)
    direction = "high" if raw_auc is None or raw_auc >= 0.5 else "low"
    candidates = sorted(set(x))
    best = None
    for threshold in candidates:
        pred = [(v >= threshold) if direction == "high" else (v <= threshold) for v in x]
        tp = sum(p and y for p, y in zip(pred, labels)); fp = sum(p and not y for p, y in zip(pred, labels))
        fn = positives - tp
        key = (tp / positives, -fp, tp / max(1, tp+fp), threshold if direction == "high" else -threshold)
        if best is None or key > best[0]: best = (key, threshold, tp, fp, fn)
    _, threshold, tp, fp, fn = best
    return {"feature": feature, "budget": budget, "direction": direction,
            "threshold": threshold, "all_risk": False,
            "train_recall": tp / positives if positives else None,
            "train_precision": tp / (tp+fp) if tp+fp else 0.0,
            "train_fn": fn, "train_fp": fp, "train_n": len(data)}


def is_risk(row, rule):
    if rule["all_risk"]: return True
    x = num(row.get(rule["feature"]))
    if x is None: return True
    return x >= rule["threshold"] if rule["direction"] == "high" else x <= rule["threshold"]


def summarize_policy(name, dataset, chosen, rule_text=""):
    lat = [num(r["total_retrieval_latency_ms"]) for r in chosen]
    lat = [x for x in lat if x is not None]
    row = {"dataset": dataset, "policy": name, "n_query_runs": len(chosen),
        "latency_mean_ms": statistics.mean(lat) if lat else None,
        "latency_p50_ms": percentile(lat, .5), "latency_p95_ms": percentile(lat, .95),
        "qps_from_mean_latency": 1000/statistics.mean(lat) if lat and statistics.mean(lat)>0 else None,
        "promoted_beyond_400_fraction": statistics.mean([int(r["budget"]) > 400 for r in chosen]) if chosen else None,
        "stop_at_400_fraction": statistics.mean([int(r["budget"]) == 400 for r in chosen]) if chosen else None,
        "stop_at_512_fraction": statistics.mean([int(r["budget"]) == 512 for r in chosen]) if chosen else None,
        "continue_to_1024_fraction": statistics.mean([int(r["budget"]) == 1024 for r in chosen]) if chosen else None,
        "candidate_generation_failure_fraction": statistics.mean([flag(r.get("candidate_generation_failure")) or (num(r.get("candidate_generation_recall_upper_bound")) is not None and num(r.get("candidate_generation_recall_upper_bound")) < 1.0) for r in chosen]) if chosen else None,
        "rule": rule_text}
    for metric in ("recall_at_100", "recall_at_10", "mrr_at_10", "success_at_10"):
        vals = [num(r.get(metric)) for r in chosen]; vals = [v for v in vals if v is not None]
        row[metric + "_mean"] = statistics.mean(vals) if vals else None
    row["candidate_generation_recall_upper_bound_mean"] = statistics.mean([num(r["candidate_generation_recall_upper_bound"]) for r in chosen if num(r.get("candidate_generation_recall_upper_bound")) is not None]) if chosen else None
    return row


def main():
    raw_features = read_csv(FEATURE_RAW)
    runtime = read_csv(RUNTIME)
    feats = query_features(raw_features)
    write_csv(FEATURE_OUT, feats)

    metric_rows = []
    for budget in (400, 512):
        label = "truncation_risk_hard"
        for dataset in ("old", "independent"):
            ds = [r for r in feats if r["dataset"] == dataset and int(r["budget"]) == budget]
            y = [bool(r[label]) for r in ds]
            for feature in FEATURES:
                x = [num(r.get(feature)) for r in ds]
                pairs = [(a, b) for a, b in zip(x, y) if a is not None]
                xx, yy = [a for a, _ in pairs], [b for _, b in pairs]
                raw_auc = auc(xx, yy)
                oriented = max(raw_auc, 1-raw_auc) if raw_auc is not None else None
                hard = [num(r[feature]) for r in ds if bool(r[label]) and num(r.get(feature)) is not None]
                easy = [num(r[feature]) for r in ds if not bool(r[label]) and num(r.get(feature)) is not None]
                metric_rows.append({"dataset": dataset, "budget": budget, "feature": feature,
                    "n_queries": len(pairs), "n_truncation_risk": sum(yy),
                    "risk_median": statistics.median(hard) if hard else None,
                    "easy_median": statistics.median(easy) if easy else None,
                    "cohens_d_risk_minus_easy": effect_size(hard, easy),
                    "raw_auc": raw_auc, "oriented_auc": oriented,
                    "direction": ("higher_risk" if raw_auc >= .5 else "lower_risk") if raw_auc is not None else "undefined"})
    stability_rows = []
    for budget in (400, 512):
        for feature in FEATURES:
            by_ds = {r["dataset"]: r for r in metric_rows if r["budget"] == budget and r["feature"] == feature}
            old, independent = by_ds.get("old", {}), by_ds.get("independent", {})
            od, nd = old.get("direction"), independent.get("direction")
            stable = od in {"higher_risk", "lower_risk"} and nd == od
            oa, na = old.get("oriented_auc"), independent.get("oriented_auc")
            stability_rows.append({"budget": budget, "feature": feature,
                "old_n": old.get("n_queries"), "independent_n": independent.get("n_queries"),
                "old_risk_n": old.get("n_truncation_risk"), "independent_risk_n": independent.get("n_truncation_risk"),
                "old_raw_auc": old.get("raw_auc"), "independent_raw_auc": independent.get("raw_auc"),
                "old_oriented_auc": oa, "independent_oriented_auc": na,
                "old_direction": od, "independent_direction": nd,
                "direction_consistent": stable,
                "minimum_oriented_auc": min(oa, na) if oa is not None and na is not None else None,
                "mean_oriented_auc": statistics.mean([oa, na]) if oa is not None and na is not None else None,
                "old_risk_median": old.get("risk_median"), "old_easy_median": old.get("easy_median"),
                "old_effect_size": old.get("cohens_d_risk_minus_easy"),
                "independent_risk_median": independent.get("risk_median"),
                "independent_easy_median": independent.get("easy_median"),
                "independent_effect_size": independent.get("cohens_d_risk_minus_easy")})
    write_csv(STABILITY_OUT, stability_rows)

    old400 = [r for r in metric_rows if r["dataset"] == "old" and r["budget"] == 400 and r["oriented_auc"] is not None]
    forced_feature = os.environ.get("PLAID_FIXED_RULE_FEATURE")
    chosen_feature = forced_feature or (max(old400, key=lambda r: r["oriented_auc"])["feature"] if old400 else FEATURES[0])
    old_feats = [r for r in feats if r["dataset"] == "old"]
    if forced_feature and os.environ.get("PLAID_FIXED_RULE_T400") and os.environ.get("PLAID_FIXED_RULE_T512"):
        rule400 = {"feature": chosen_feature, "budget": 400, "direction": os.environ.get("PLAID_FIXED_RULE_D400", "high"), "threshold": float(os.environ["PLAID_FIXED_RULE_T400"]), "all_risk": False}
        rule512 = {"feature": chosen_feature, "budget": 512, "direction": os.environ.get("PLAID_FIXED_RULE_D512", "high"), "threshold": float(os.environ["PLAID_FIXED_RULE_T512"]), "all_risk": False}
    else:
        rule400 = threshold_fit(old_feats, chosen_feature, 400)
        rule512 = threshold_fit(old_feats, chosen_feature, 512)
    rule_text = (f"{chosen_feature}: 400 {rule400['direction']} {fmt(rule400['threshold'])}; "
                 f"512 {rule512['direction']} {fmt(rule512['threshold'])}; thresholds selected on old only")

    rmap = {(r["dataset"], int(r["query_id"]), int(r["repeat"]), r["scenario"]): r for r in runtime}
    fmap = {(r["dataset"], int(r["query_id"]), int(r["repeat"]), int(r["budget"])): r for r in raw_features}
    datasets = ("old", "independent")
    policies = []
    for dataset in datasets:
        queries = sorted({(r["dataset"], int(r["query_id"]), int(r["repeat"])) for r in runtime if r["dataset"] == dataset})
        for budget in (400, 512, 1024):
            chosen = [rmap[(ds, qid, rep, f"Full{budget}")] for ds, qid, rep in queries]
            policies.append(summarize_policy(f"Always{budget}", dataset, chosen))
        adaptive, oracle = [], []
        for ds, qid, rep in queries:
            f400 = fmap[(ds, qid, rep, 400)]
            f512 = fmap[(ds, qid, rep, 512)]
            risk400 = is_risk(f400, rule400)
            risk512 = is_risk(f512, rule512)
            target = 512 if risk400 and not risk512 else 1024 if risk400 else 400
            adaptive.append(rmap[(ds, qid, rep, f"TwoStage{target}")])
            true400 = flag(f400.get("truncation_risk_at_budget", f400.get("truncation_risk_hard")))
            true512 = flag(f512.get("truncation_risk_at_budget", f512.get("truncation_risk_hard")))
            oracle_target = 512 if true400 and not true512 else 1024 if true400 else 400
            orow = dict(rmap[(ds, qid, rep, f"TwoStage{oracle_target}")])
            orow["candidate_generation_failure"] = f400.get("candidate_generation_failure")
            oracle.append(orow)
        policies.append(summarize_policy("TwoStageAdaptive_frozen_old_thresholds", dataset, adaptive, rule_text))
        policies.append(summarize_policy("Oracle_upper_bound", dataset, oracle, "uses qrels/1024 outcomes; not deployable"))
    write_csv(POLICY_OUT, policies)

    old_fails = [r for r in feats if r["dataset"] == "old" and int(r["budget"]) == 400 and r["candidate_generation_failure"]]
    ind_fails = [r for r in feats if r["dataset"] == "independent" and int(r["budget"]) == 400 and r["candidate_generation_failure"]]
    old_risks = [r for r in feats if r["dataset"] == "old" and int(r["budget"]) == 400 and r["truncation_risk_hard"]]
    ind_risks = [r for r in feats if r["dataset"] == "independent" and int(r["budget"]) == 400 and r["truncation_risk_hard"]]
    old400f = [r for r in feats if r["dataset"] == "old" and int(r["budget"]) == 400]
    ind400f = [r for r in feats if r["dataset"] == "independent" and int(r["budget"]) == 400]
    old_ceiling = statistics.mean([r["candidate_generation_recall_upper_bound"] for r in old400f if r["candidate_generation_recall_upper_bound"] is not None])
    ind_ceiling = statistics.mean([r["candidate_generation_recall_upper_bound"] for r in ind400f if r["candidate_generation_recall_upper_bound"] is not None])
    policy_map = {(r["dataset"], r["policy"]): r for r in policies}
    lines = ["# PLAID two-stage adaptive runtime and signal report", "",
        "## Scope", "",
        "45 frozen queries (old=9; independent=36), 3 repetitions, existing indexes only. The preceding six-query parity gate passed 18/18. This report contains a one-exact-finalize two-stage API: 400/512 checkpoints are approximate-only. No index build or parameter sweep was run.", "",
        "## Approximate signal stability", "",
        f"The rule feature was selected using old-set budget-400 AUC only: **{chosen_feature}**. Its cutoff direction and thresholds were then frozen from old queries for both 400 and 512 decisions. Independent labels were not used to choose the feature or thresholds.", "",
        f"- Old: truncation-risk queries at 400 = {len(old_risks)}/{len(old400f)}; candidate-generation-failure queries = {len(old_fails)}/{len(old400f)}; mean candidate-generation Recall ceiling = {fmt(old_ceiling)}.",
        f"- Independent: truncation-risk queries at 400 = {len(ind_risks)}/{len(ind400f)}; candidate-generation-failure queries = {len(ind_fails)}/{len(ind400f)}; mean candidate-generation Recall ceiling = {fmt(ind_ceiling)}.", "",
        "Feature AUC/effect summaries are in `plaid_approx_signal_stability.csv`. Direction agreement must be judged by comparing old and independent direction for each feature/budget; oriented AUC alone can conceal a reversal.", "",
        "## Policy results", "",
        "| Dataset | Policy | Mean ms | P50 ms | P95 ms | QPS* | Recall@100 | Recall@10 | MRR@10 | Success@10 | >400 fraction | Candidate-gen failure | Candidate-gen ceiling |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in policies:
        lines.append("| {dataset} | {policy} | {latency_mean_ms} | {latency_p50_ms} | {latency_p95_ms} | {qps_from_mean_latency} | {recall_at_100_mean} | {recall_at_10_mean} | {mrr_at_10_mean} | {success_at_10_mean} | {promoted_beyond_400_fraction} | {candidate_generation_failure_fraction} | {candidate_generation_recall_upper_bound_mean} |".format(
            dataset=row["dataset"], policy=row["policy"], **{k: fmt(row.get(k), 3) for k in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "qps_from_mean_latency", "recall_at_100_mean", "recall_at_10_mean", "mrr_at_10_mean", "success_at_10_mean", "promoted_beyond_400_fraction", "candidate_generation_failure_fraction", "candidate_generation_recall_upper_bound_mean")}))
    lines += ["", "*QPS is 1000 / mean single-query end-to-end latency, not batched throughput.", "",
        "## Interpretation", "",
        f"Two-stage runtime device: **{RUNTIME_DEVICE}**. It includes query encoding, candidate generation, all approximate checkpoints used, decision/state work, one final exact decode/MaxSim and sorting. Synchronization is only included for GPU runs; CPU runs use wall-clock timing. GPU-event sub-timings are reported separately when available; full legacy score_pids is reported as a unified filter+refine+sort interval because it has no internal event instrumentation.", "",
        "Candidate-generation failures are separate from truncation risk. The per-query candidate-generation recall ceiling is recorded in the feature CSV; ndocs adaptation cannot recover relevant PIDs absent from the generated candidate set. The oracle policy is only an upper bound because it uses qrels and full-budget outcomes.", "",
        "## Generated files", "",
        "- `plaid_two_stage_runtime.csv` — per-query/repetition latency and stage timings.",
        "- `plaid_approx_dynamic_features.csv` — query-level approximate signals and truncation/candidate-generation labels.",
        "- `plaid_approx_signal_stability.csv` — old/independent AUC, medians, effect sizes and direction.",
        "- `plaid_two_stage_policy_results.csv` — Always400/512/1024, frozen-old adaptive and qrels oracle summaries.", ""]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
