#!/usr/bin/env python3
"""Offline mechanism and cross-dataset analysis for PLAID ndocs=400 failures."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OLD_PATH = ROOT / "plaid_query_parameter_sweep.csv"
NEW_PATH = ROOT / "plaid_independent_validation_per_query.csv"
QUERY_PATH = ROOT / "plaid_independent_validation_queries.csv"
FAILURE = ROOT / "plaid_hard_query_failure_analysis.csv"
FAILURE_TYPES = ROOT / "plaid_hard_query_failure_types.csv"
STABILITY = ROOT / "plaid_cross_dataset_feature_stability.csv"
RULES = ROOT / "plaid_cross_dataset_rule_evaluation.csv"
REPORT = ROOT / "plaid_hard_query_mechanism_analysis.md"

COMMON_FEATURES = [
    "query_token_length",
    "query_embedding_norm_std",
    "centroid_top1_top2_gap_mean",
    "candidate_count",
    "candidate_density_per_token",
]
REQUESTED_MISSING = [
    "centroid_top1_score_per_token",
    "centroid_top1_top2_gap_min",
    "centroid_top1_top2_gap_max",
    "centroid_top1_top2_gap_std",
    "centroid_top1_top2_gap_p10",
    "centroid_top1_top2_gap_p25",
    "centroid_rank_distribution",
    "token_centroid_concentration",
    "candidate_count_per_query_token",
    "candidate_overlap_across_tokens",
    "candidate_score_cutoff_margin",
    "candidate_score_tail_slope",
    "relevant_document_baseline_rank",
    "relevant_document_prune_trace",
]


def auc_higher_is_hard(values, labels):
    x = np.asarray(values, float); y = np.asarray(labels, bool)
    hard, easy = x[y], x[~y]
    if not len(hard) or not len(easy): return np.nan
    return float(np.mean((hard[:, None] > easy[None, :]) + 0.5 * (hard[:, None] == easy[None, :])))


def spearman(x, y):
    a = pd.Series(x).rank(method="average"); b = pd.Series(y).rank(method="average")
    return float(a.corr(b)) if a.nunique() > 1 and b.nunique() > 1 else np.nan


def summarize_source(df, group_col, baseline_name, fixed_name, feature_source_name):
    numeric = [
        "query_token_length", "query_embedding_norm_std", "centroid_top1_top2_gap_mean",
        "candidate_count", "recall_at_100", "recall_at_10", "mrr_at_10", "success_at_10",
        "total_latency_ms",
    ]
    optional = [
        "centroid_top1_mean", "centroid_top2_mean", "centroid_top32_mean",
        "centroid_top32_variance", "centroid_active_count_at_045", "centroid_score_entropy",
        "final_scored_count", "prune_ms", "decompress_ms", "maxsim_or_gemm_ms",
        "decompressed_embedding_count",
    ]
    cols = [c for c in numeric + optional if c in df.columns]
    q = df.groupby(["query_id", group_col], as_index=False)[cols].mean()
    base = q[q[group_col] == baseline_name].set_index("query_id")
    fixed = q[q[group_col] == fixed_name].set_index("query_id")
    common_ids = base.index.intersection(fixed.index)
    base, fixed = base.loc[common_ids].copy(), fixed.loc[common_ids].copy()
    out = base.copy()
    out["baseline_recall_at_100"] = base.recall_at_100
    out["fixed400_recall_at_100"] = fixed.recall_at_100
    out["quality_drop_recall_at_100"] = base.recall_at_100 - fixed.recall_at_100
    out["is_hard"] = out.quality_drop_recall_at_100 > 1e-12
    out["candidate_density_per_token"] = out.candidate_count / out.query_token_length.clip(lower=1)
    if "final_scored_count" in out:
        out["prune_reduction_ratio"] = 1 - out.final_scored_count / out.candidate_count
    out["source"] = feature_source_name
    return out, q


def feature_stats(old, new):
    rows = []
    for feature in COMMON_FEATURES:
        row = {"feature": feature, "availability": "available_both"}
        for prefix, data in (("old", old), ("new", new)):
            hard = data[data.is_hard][feature].dropna(); easy = data[~data.is_hard][feature].dropna()
            auc = auc_higher_is_hard(data[feature], data.is_hard)
            row.update({
                f"{prefix}_hard_n": len(hard), f"{prefix}_easy_n": len(easy),
                f"{prefix}_hard_median": float(hard.median()),
                f"{prefix}_easy_median": float(easy.median()),
                f"{prefix}_median_difference_hard_minus_easy": float(hard.median() - easy.median()),
                f"{prefix}_auc_higher_is_hard": auc,
                f"{prefix}_auc_separability": max(auc, 1 - auc),
                f"{prefix}_cliffs_delta": 2 * auc - 1,
                f"{prefix}_spearman_with_recall100_drop": spearman(data[feature], data.quality_drop_recall_at_100),
            })
        old_sign = np.sign(row["old_median_difference_hard_minus_easy"])
        new_sign = np.sign(row["new_median_difference_hard_minus_easy"])
        row["sign_consistent"] = bool(old_sign == new_sign and old_sign != 0)
        rows.append(row)
    result = pd.DataFrame(rows)
    result["old_separability_rank"] = result.old_auc_separability.rank(ascending=False, method="min").astype(int)
    result["new_separability_rank"] = result.new_auc_separability.rank(ascending=False, method="min").astype(int)
    result["rank_difference"] = (result.old_separability_rank - result.new_separability_rank).abs()
    for feature in REQUESTED_MISSING:
        result.loc[len(result)] = {"feature": feature, "availability": "not_recorded_in_existing_per_query_outputs"}
    return result


def predict_threshold(train, test, feature):
    x = train[feature].to_numpy(float); y = train.is_hard.to_numpy(bool)
    candidates = np.unique(x)
    best = None
    for direction in ("<=", ">="):
        for threshold in candidates:
            pred = x <= threshold if direction == "<=" else x >= threshold
            tp = int((pred & y).sum()); fn = int((~pred & y).sum()); fp = int((pred & ~y).sum())
            recall = tp / max(1, int(y.sum())); precision = tp / max(1, tp + fp)
            key = (recall, precision, -fp)
            if best is None or key > best[0]: best = (key, direction, float(threshold))
    _, direction, threshold = best
    pred = test[feature].to_numpy(float) <= threshold if direction == "<=" else test[feature].to_numpy(float) >= threshold
    return pred, direction, threshold


def metrics(pred, truth):
    pred=np.asarray(pred,bool); truth=np.asarray(truth,bool)
    tp=int((pred&truth).sum()); fn=int((~pred&truth).sum()); fp=int((pred&~truth).sum()); tn=int((~pred&~truth).sum())
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"hard_recall":tp/max(1,tp+fn),
            "hard_precision":tp/max(1,tp+fp),"specificity":tn/max(1,tn+fp),"accuracy":(tp+tn)/len(truth)}


def cross_rules(old, new):
    rows=[]
    for train_name, train, test_name, test in (("old48",old,"independent94",new),("independent94",new,"old48",old)):
        for feature in COMMON_FEATURES:
            pred,direction,threshold=predict_threshold(train,test,feature)
            rows.append({"rule":f"single:{feature}","train_dataset":train_name,"test_dataset":test_name,
                         "logic":"single threshold","direction":direction,"threshold_1":threshold,**metrics(pred,test.is_hard)})
    # Evaluate the already-frozen old rule without changing it.
    for test_name,test in (("old48",old),("independent94",new)):
        pred=(test.query_embedding_norm_std<=3.5723282110211585e-08)|(test.centroid_top1_top2_gap_mean<=0.0400056634098291)
        rows.append({"rule":"frozen_old_two_feature_OR","train_dataset":"old48_preexisting","test_dataset":test_name,
                     "logic":"norm_std <= 3.572328e-08 OR gap_mean <= 0.04000566",
                     "direction":"fixed","threshold_1":3.5723282110211585e-08,"threshold_2":0.0400056634098291,
                     **metrics(pred,test.is_hard)})
    return pd.DataFrame(rows)


def list_ids(value):
    if isinstance(value,list): return [int(x) for x in value]
    return [int(x) for x in ast.literal_eval(str(value))]


def hard_failure_rows(new_base, new_q, raw_new):
    query_counts = pd.read_csv(QUERY_PATH).set_index("query_id").positive_count
    averaged = raw_new.groupby(["query_id","strategy"],as_index=False).agg(
        recall_at_100=("recall_at_100","mean"), candidate_count=("candidate_count","mean"),
        score_ms=("score_ms","mean"), total_latency_ms=("total_latency_ms","mean"),
        predicted_hard=("predicted_hard","first"), query_token_length=("query_token_length","mean"),
        query_embedding_norm_std=("query_embedding_norm_std","mean"),
        centroid_top1_top2_gap_mean=("centroid_top1_top2_gap_mean","mean"),
    )
    rows=[]
    for qid in new_base.index[new_base.is_hard]:
        per=averaged[averaged.query_id==qid].set_index("strategy")
        raw_b=raw_new[(raw_new.query_id==qid)&(raw_new.strategy=="original_baseline")].iloc[0]
        raw_f=raw_new[(raw_new.query_id==qid)&(raw_new.strategy=="fixed_ndocs_400")].iloc[0]
        bset=set(list_ids(raw_b.top100_result_ids)); fset=set(list_ids(raw_f.top100_result_ids))
        row={
            "query_id":int(qid),"positive_count":int(query_counts.get(qid,np.nan)),
            "baseline_recall_at_100":per.loc["original_baseline","recall_at_100"],
            "fixed400_recall_at_100":per.loc["fixed_ndocs_400","recall_at_100"],
            "adaptive_recall_at_100":per.loc["adaptive_400_1024","recall_at_100"],
            "fixed1024_recall_at_100":per.loc["fixed_ndocs_1024","recall_at_100"],
            "recall100_drop_fixed400":per.loc["original_baseline","recall_at_100"]-per.loc["fixed_ndocs_400","recall_at_100"],
            "predicted_hard":bool(str(per.loc["adaptive_400_1024","predicted_hard"]).lower() in ("true","1")),
            "candidate_count_baseline":per.loc["original_baseline","candidate_count"],
            "candidate_count_fixed400":per.loc["fixed_ndocs_400","candidate_count"],
            "candidate_count_equal":bool(per.loc["original_baseline","candidate_count"]==per.loc["fixed_ndocs_400","candidate_count"]),
            "query_token_length":per.loc["original_baseline","query_token_length"],
            "query_embedding_norm_std":per.loc["original_baseline","query_embedding_norm_std"],
            "centroid_top1_top2_gap_mean":per.loc["original_baseline","centroid_top1_top2_gap_mean"],
            "baseline_fixed400_top100_overlap":len(bset&fset)/100,
            "baseline_only_top100_count":len(bset-fset),"fixed400_only_top100_count":len(fset-bset),
            "candidate_generation_failure":"ruled_out_for_baseline-relevant_docs_by_identical_candidate_generation",
            "truncation_pruning_failure":"mechanistically_consistent_and_primary",
            "ranking_refine_failure":"not_supported_under_same_exact_scorer_and_subset_monotonicity",
            "failure_type":"B_truncation_or_pruning_inferred",
            "evidence_level":"mechanism_inference; no per-PID prune trace recorded",
            "relevant_baseline_rank":"NA_not_recorded_with_remapped_qrels",
            "relevant_fixed400_stage":"NA_no_per-PID_candidate_or_prune_trace",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def placeholder(path, title, detail):
    fig,ax=plt.subplots(figsize=(8,4.5));ax.axis("off");ax.text(.5,.6,title,ha="center",va="center",fontsize=15,weight="bold")
    ax.text(.5,.4,detail,ha="center",va="center",fontsize=10,wrap=True);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def plots(old,new,stability,failures):
    placeholder(ROOT/"plaid_worst_token_centroid_gap_distribution.png","Worst-token centroid gap unavailable",
                "Only query-level mean gap was recorded. Token-level gaps, p10/p25, min, max, and std cannot be reconstructed offline.")
    placeholder(ROOT/"plaid_candidate_cutoff_margin_distribution.png","Candidate cutoff margin unavailable",
                "Candidate score arrays and the score at the ndocs=400 boundary were not saved in either run.")
    fig,ax=plt.subplots(figsize=(8,5));
    vals=[old.loc[~old.is_hard,"prune_reduction_ratio"],old.loc[old.is_hard,"prune_reduction_ratio"]]
    ax.boxplot(vals,tick_labels=["Old easy","Old hard"],showmeans=True);ax.set_ylabel("1 − final_scored / candidates")
    ax.set_title("Prune reduction ratio (old48 only)\nIndependent run did not record final_scored_count");ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(ROOT/"plaid_prune_reduction_ratio_distribution.png",dpi=180);plt.close(fig)
    avail=stability[stability.availability=="available_both"]
    x=np.arange(len(avail));fig,ax=plt.subplots(figsize=(10,5.5));
    ax.bar(x-.2,avail.old_auc_separability,.4,label="Old48",color="#4C78A8");ax.bar(x+.2,avail.new_auc_separability,.4,label="Independent94",color="#F58518")
    ax.axhline(.5,color="black",lw=.8,ls="--");ax.set_xticks(x,avail.feature,rotation=25,ha="right");ax.set_ylim(.45,1.02);ax.set_ylabel("ROC AUC separability")
    ax.set_title("Early-feature hard/easy separability across datasets");ax.legend();fig.tight_layout();fig.savefig(ROOT/"plaid_old_vs_new_feature_auc.png",dpi=180);plt.close(fig)
    matrix=np.zeros((len(failures),3));matrix[:,1]=1
    fig,ax=plt.subplots(figsize=(7.5,5.8));im=ax.imshow(matrix,cmap="YlOrRd",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks([0,1,2],["Candidate generation","Truncation/pruning","Ranking/refine"],rotation=15,ha="right")
    ax.set_yticks(np.arange(len(failures)),failures.query_id.astype(str));ax.set_ylabel("Independent hard query ID")
    ax.set_title("Failure-stage classification\npruning is mechanism-inferred; per-PID traces were not recorded")
    for i in range(len(failures)):
        for j,t in enumerate(["ruled out","likely","unsupported"]):ax.text(j,i,t,ha="center",va="center",fontsize=8,color="black")
    fig.tight_layout();fig.savefig(ROOT/"plaid_nine_hard_query_stage_failure.png",dpi=180);plt.close(fig)


def main():
    old_raw=pd.read_csv(OLD_PATH);new_raw=pd.read_csv(NEW_PATH)
    old,old_q=summarize_source(old_raw,"config_id","baseline","ndocs_400","old48")
    new,new_q=summarize_source(new_raw,"strategy","original_baseline","fixed_ndocs_400","independent94")
    if old.is_hard.sum()!=3 or new.is_hard.sum()!=9:raise RuntimeError(f"Unexpected hard counts old={old.is_hard.sum()} new={new.is_hard.sum()}")
    # Verify candidate generation is parameter-independent for every hard query.
    candidate_wide=new_q.pivot(index="query_id",columns="strategy",values="candidate_count")
    if not (candidate_wide.original_baseline==candidate_wide.fixed_ndocs_400).all():raise RuntimeError("Candidate counts differ between baseline and fixed400")
    stability=feature_stats(old,new);stability.to_csv(STABILITY,index=False)
    rules=cross_rules(old,new);rules.to_csv(RULES,index=False)
    failures=hard_failure_rows(new,new_q,new_raw);failures.to_csv(FAILURE,index=False)
    failures[["query_id","failure_type","evidence_level","candidate_generation_failure","truncation_pruning_failure","ranking_refine_failure"]].to_csv(FAILURE_TYPES,index=False)
    plots(old,new,stability,failures)

    available=stability[stability.availability=="available_both"].sort_values(["sign_consistent","new_auc_separability"],ascending=[False,False])
    rank_corr=spearman(available.old_separability_rank,available.new_separability_rank)
    best_consistent=available[available.sign_consistent == True].head(2).feature.tolist()
    frozen_new=rules[(rules.rule=="frozen_old_two_feature_OR")&(rules.test_dataset=="independent94")].iloc[0]
    lines=[
        "# PLAID hard-query mechanism analysis", "", "## Direct observations", "",
        "- Old48 has 3 fixed400-sensitive queries (31, 40, 42). Independent94 has 9 (65, 83, 88, 100, 106, 112, 117, 128, 142).",
        "- For every independent query, baseline and fixed400 have exactly the same pre-ndocs candidate count. The two strategies differ only at the ndocs-controlled filtering/refinement budget.",
        "- Fixed1024 restores baseline Recall@100 for all nine hard queries. Therefore the 1024 path is capable of recovering the lost relevant documents.",
        "- The frozen rule found 3/9 hard queries and missed 6; its independent precision was 14.29% with 18 false positives.",
        "", "## Failure stage", "",
        "All nine independent hard queries are classified as **B: truncation/pruning failure (mechanism-inferred)**. Candidate-generation failure is ruled out for the relevant documents visible in baseline because candidate generation is identical and occurs before ndocs is applied. A retained relevant document scored by the same exact scorer against a subset of competitors cannot acquire a worse rank solely from reducing the competitor set, so ranking/refine failure is not supported. However, per-PID candidate/prune traces were not saved, so this is a structural inference rather than direct PID-level observation.",
        "", "## Feature stability", "",
        f"- Cross-feature separability-rank correlation between datasets: **{rank_corr:.3f}**.",
        f"- Direction-consistent common features ranked highest: **{', '.join(best_consistent) if best_consistent else 'none'}**.",
        "- The old mean centroid-gap and embedding-norm thresholds do not transfer: the norm threshold catches 0/9 new hard queries; the gap threshold catches 3/9.",
        "- Token-level worst-gap/quantile features, candidate cutoff margin, token candidate overlap, and independent prune-reduction ratio were not recorded. Their stability cannot be evaluated offline.",
        "", "## Cross-dataset rules", "",
        f"- The frozen old two-feature OR rule on independent94: TP={int(frozen_new.tp)}, FN={int(frozen_new.fn)}, FP={int(frozen_new.fp)}, hard recall={frozen_new.hard_recall:.3f}, precision={frozen_new.hard_precision:.3f}.",
        "- Single-feature thresholds were fit in each source and applied only to the other source. These are diagnostic transfer tests, not proposed final rules. Results are in `plaid_cross_dataset_rule_evaluation.csv`.",
        "- No new 2–3 feature rule is promoted from this offline analysis because the most mechanism-relevant requested inputs (worst-token gap and cutoff margin) are absent. Fitting a replacement only from aggregate means on independent94 would repeat the same overfitting mistake.",
        "", "## Missing evidence", "",
        "The current files do not contain per-token centroid scores/gaps, per-token centroid ranks, per-token candidate lists, candidate-score arrays around the cutoff, or per-PID membership before/after pruning. Remapped positive PIDs were also not included in the downloaded result package, so exact relevant-document ranks cannot be reconstructed. Placeholder figures explicitly mark these gaps rather than displaying invented values.",
        "", "## Answers", "",
        "1. **Mechanism:** ndocs=400 loses Recall@100 at the budget-controlled filtering/truncation step after common candidate generation.",
        "2. **Stage for the nine queries:** all nine are most consistent with truncation/pruning; this is inferred from control flow and result monotonicity, not directly PID-traced.",
        "3. **More stable feature:** none of the currently shared aggregate features is strong enough to replace gap mean confidently. Candidate density and candidate count can be compared cross-dataset, but their transfer metrics must be treated as exploratory.",
        "4. **Feature versus threshold:** both are implicated. The direction of some aggregate signals is weakly consistent, but separability changes and the frozen thresholds fail badly. This is more than a simple threshold shift for embedding norm.",
        "5. **Continue adaptive PLAID?** Yes as a mechanism study, not yet as a validated policy. The achievable latency/quality trade-off remains real, while detection is the unresolved component.",
        "6. **Next features:** record **worst-token/lower-quantile centroid gap** and **candidate score margin at the ndocs=400 cutoff**. These directly represent ambiguous query tokens and truncation risk. Candidate overlap/density is the next supporting feature.",
    ]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(stability[stability.availability=="available_both"].to_string(index=False))
    print(failures[["query_id","failure_type","predicted_hard","recall100_drop_fixed400"]].to_string(index=False))
    print(f"saved: {FAILURE}\n{FAILURE_TYPES}\n{STABILITY}\n{RULES}\n{REPORT}")


if __name__=="__main__":main()
