from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"dynamic_signal_v1"
RAW=OUT/"plaid_dynamic_signal_budget_per_query.csv"
V2=ROOT/"v2_features"/"plaid_v2_feature_shift_analysis.csv"
BUDGETS=[200,300,400,512,768,1024]

def auc_high(values,y):
    v=np.asarray(values,float); y=np.asarray(y,bool); p=v[y]; n=v[~y]
    if not len(p) or not len(n):return np.nan
    return float(np.mean([(a>b)+.5*(a==b) for a in p for b in n]))

def effect_d(h,e):
    h=np.asarray(h,float);e=np.asarray(e,float)
    if len(h)<2 or len(e)<2:return np.nan
    pooled=np.sqrt(((len(h)-1)*h.var(ddof=1)+(len(e)-1)*e.var(ddof=1))/(len(h)+len(e)-2))
    return float((h.mean()-e.mean())/pooled) if pooled else np.nan

def confusion(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool)
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"hard_recall":tp/(tp+fn) if tp+fn else np.nan,
            "precision":tp/(tp+fp) if tp+fp else np.nan}

def choose_threshold(x,y):
    x=np.asarray(x,float); y=np.asarray(y,bool); vals=np.unique(x[np.isfinite(x)])
    if not len(vals):return np.inf,confusion(y,np.zeros(len(y),bool))
    thresholds=np.r_[vals[0]-1e-12,(vals[:-1]+vals[1:])/2,vals[-1]+1e-12]
    cand=[]
    for t in thresholds:
        c=confusion(y,x>=t);cand.append((c["FN"],c["FP"],-np.nan_to_num(c["precision"],nan=0),t,c))
    _,_,_,t,c=min(cand,key=lambda z:(z[0],z[1],z[2],z[3]))
    return float(t),c

def savefig(fig,path):
    fig.tight_layout();fig.savefig(OUT/path,dpi=180,bbox_inches="tight");plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(RAW).sort_values(["dataset","query_id","budget"]).reset_index(drop=True)
    raw["query_id"]=raw.query_id.astype(int);raw["budget"]=raw.budget.astype(int)
    if raw.groupby("dataset").query_id.nunique().to_dict()!={"independent":36,"old":9}:
        raise ValueError(f"Unexpected query sample sizes: {raw.groupby('dataset').query_id.nunique().to_dict()}")
    pivot=raw.pivot(index=["dataset","query_id","query_group"],columns="budget",values="recall_at_100")
    base=pivot[1024]
    hard400=pivot[400] < base-1e-12
    raw["actual_hard_label_posthoc"]=raw.apply(lambda r:int(hard400.loc[(r["dataset"],r["query_id"],r["query_group"])]),axis=1)
    raw["quality_below_1024_at_budget"]=raw.apply(lambda r:int(r["recall_at_100"] < base.loc[(r["dataset"],r["query_id"],r["query_group"])]-1e-12),axis=1)
    raw["hard_label_source"]="posthoc fixed400 Recall@100 < ndocs1024 baseline; constant query label, not used by online signal/rule"
    if not raw.groupby(["dataset","query_id"]).candidate_input_sha256.nunique().eq(1).all():
        raise ValueError("Raw retrieve candidate PID set changed across budget stages")
    if not raw.input_candidate_ids_unchanged_check.astype(bool).all():
        raise ValueError("score_pids mutated the candidate tensor intended for reuse")
    # Retrospective per-step signal derivation using only current and previous budget.
    prev_count=raw.groupby(["dataset","query_id"]).retained_candidate_count.shift(1).replace(0,np.nan)
    raw["candidate_growth_fraction_prev"]=raw.candidate_new_prev_count/prev_count
    raw["candidate_churn_prev"]=1-raw.candidate_overlap_prev_count/prev_count
    raw["top100_new_fraction"]=raw.top100_new_prev_count/100.0
    raw["candidate_saturation_ratio_prev_current"]=prev_count/raw.retained_candidate_count.replace(0,np.nan)
    # At budget 400, this only uses 200->300 and 300->400 transitions.
    rows=[]
    for (ds,qid,group),g in raw.groupby(["dataset","query_id","query_group"]):
        g=g.set_index("budget").reindex(BUDGETS)
        r400=g.loc[400]; r300=g.loc[300]; r200=g.loc[200]
        # ndocs=300 returns only 75 exact PIDs in this ColBERT fork (final budget = ndocs//4).
        # Use the comparable Top75 prefix as the pre-400 signal; Top100 cannot be compared
        # fairly until both adjacent stages return >=100 PIDs.
        churn300=float(r400.common_prefix_churn_prev); ov300=float(r400.common_prefix_overlap_prev_count)
        cgrowth=float(r400.candidate_growth_fraction_prev); cover=float(r400.candidate_overlap_prev_count)
        cchurn=float(r400.candidate_churn_prev); saturation=float(r400.candidate_saturation_ratio_prev_current)
        s200=float(r300.common_prefix_churn_prev); c200=float(r300.candidate_churn_prev)
        rows.append({"dataset":ds,"query_id":qid,"query_group":group,"hard_label_posthoc":int(r400.actual_hard_label_posthoc),
            "top75_churn_300_400":churn300,"top75_overlap_300_400":ov300,"top100_target_churn_300_400_capacity_limited":float(r400.top100_churn_prev),
            "top100_full_churn_400_512":float(g.loc[512].top100_churn_prev),"top100_full_overlap_400_512":float(g.loc[512].top100_overlap_prev_count),
            "top100_new_fraction_300_400":float(r400.top100_new_fraction),
            "candidate_growth_300_400":cgrowth,"candidate_overlap_300_400":cover,"candidate_churn_300_400":cchurn,
            "candidate_saturation_ratio_300_400":saturation,"top100_churn_200_300":s200,"candidate_churn_200_300":c200,
            "consecutive_stability_through400":int(churn300<=.02 and s200<=.02 and cchurn<=.02 and c200<=.02),
            "retained_count_300":int(r300.retained_candidate_count),"retained_count_400":int(r400.retained_candidate_count),
            "recall100_400_posthoc":float(r400.recall_at_100),"recall100_1024_posthoc":float(g.loc[1024].recall_at_100),
            "stage_score_ms_200":float(g.loc[200].budget_score_refine_ms),"stage_score_ms_300":float(r300.budget_score_refine_ms),
            "stage_score_ms_400":float(r400.budget_score_refine_ms),"stage_score_ms_1024":float(g.loc[1024].budget_score_refine_ms),
            "naive_progressive_to400_ms":float(r400.cumulative_naive_progressive_ms),
            "standalone400_ms":float(r400.standalone_budget_e2e_ms),"naive_extra_over_standalone400_ms":float(r400.cumulative_naive_progressive_ms-r400.standalone_budget_e2e_ms),
            "naive_progressive_to1024_ms":float(g.loc[1024].cumulative_naive_progressive_ms),
            "standalone1024_ms":float(g.loc[1024].standalone_budget_e2e_ms)})
    decision=pd.DataFrame(rows)
    raw.to_csv(OUT/"plaid_dynamic_signal_per_query.csv",index=False,float_format="%.10g")
    raw[raw.budget>200].to_csv(OUT/"plaid_dynamic_signal_stability.csv",index=False,float_format="%.10g")
    decision.to_csv(OUT/"plaid_dynamic_signal_rule_features_per_query.csv",index=False,float_format="%.10g")

    signals={"top75_churn_300_400":True,"top75_overlap_300_400":False,"top100_full_churn_400_512":True,
      "top100_full_overlap_400_512":False,"candidate_growth_300_400":True,"candidate_overlap_300_400":False,"candidate_churn_300_400":True,
      "candidate_saturation_ratio_300_400":False,"consecutive_stability_through400":False}
    summaries=[]
    for ds,g in decision.groupby("dataset"):
        y=g.hard_label_posthoc.astype(bool).to_numpy()
        for sig,risk_high in signals.items():
            h=g.loc[g.hard_label_posthoc==1,sig].astype(float).to_numpy();e=g.loc[g.hard_label_posthoc==0,sig].astype(float).to_numpy()
            vals=pd.to_numeric(g[sig],errors="coerce").to_numpy(float);a=auc_high(vals,y); oriented=a if risk_high else 1-a
            summaries.append({"dataset":ds,"signal":sig,"risk_direction":"higher" if risk_high else "lower",
              "hard_n":len(h),"easy_n":len(e),"hard_mean":np.nanmean(h) if len(h) else np.nan,"hard_median":np.nanmedian(h) if len(h) else np.nan,
              "hard_iqr":np.nanpercentile(h,75)-np.nanpercentile(h,25) if len(h) else np.nan,
              "easy_mean":np.nanmean(e) if len(e) else np.nan,"easy_median":np.nanmedian(e) if len(e) else np.nan,
              "easy_iqr":np.nanpercentile(e,75)-np.nanpercentile(e,25) if len(e) else np.nan,
              "cohens_d_hard_minus_easy":effect_d(h,e),"auc_raw_higher_is_hard":a,"oriented_auc":oriented})
    hs=pd.DataFrame(summaries);hs.to_csv(OUT/"plaid_dynamic_signal_hard_easy_analysis.csv",index=False,float_format="%.10g")

    # A single frozen cutoff: choose on old controls only; no independent labels
    # enter threshold selection. Report reverse direction diagnostically only.
    old=decision[decision.dataset=="old"];new=decision[decision.dataset=="independent"]
    t,_=choose_threshold(old.top75_churn_300_400,old.hard_label_posthoc)
    rule_rows=[]
    for ds,g in [("old_train_apparent",old),("independent_frozen_test",new)]:
        pred=g.top75_churn_300_400.to_numpy(float)>=t
        rule_rows.append({"rule":"top75_churn_300_400 >= old-trained threshold","threshold":t,"dataset_role":ds,
            "signals_used":"top75_churn_300_400 only","n":len(g),**confusion(g.hard_label_posthoc,pred),
            "threshold_fit_on":"old 3 hard + 6 random easy only","independent_used_for_fit":False})
    # Secondary two-signal feasibility rule, similarly source-fitted on old only.
    # OR thresholds are selected lexicographically to recall all old hard cases,
    # then minimize old false promotions; it remains exploratory.
    ox=old.top75_churn_300_400.to_numpy(float);oz=old.candidate_growth_300_400.to_numpy(float);oy=old.hard_label_posthoc.to_numpy(bool)
    thresholds=lambda x: np.r_[np.min(x)-1e-12,(np.unique(x)[:-1]+np.unique(x)[1:])/2,np.max(x)+1e-12]
    best=None
    for a in thresholds(ox):
      for b in thresholds(oz):
        pp=(ox>=a)|(oz>=b);c=confusion(oy,pp);key=(c["FN"],c["FP"],a,b)
        if best is None or key<best[0]:best=(key,a,b,c)
    _,a,b,cm=best
    for ds,g in [("old_train_apparent",old),("independent_frozen_test",new)]:
        pp=(g.top75_churn_300_400.to_numpy(float)>=a)|(g.candidate_growth_300_400.to_numpy(float)>=b)
        rule_rows.append({"rule":"top75_churn OR candidate_growth at 300->400","threshold":json.dumps({"churn_gte":a,"growth_gte":b}),
            "dataset_role":ds,"signals_used":"top75_churn_300_400 + candidate_growth_300_400","n":len(g),**confusion(g.hard_label_posthoc,pp),
            "threshold_fit_on":"old 3 hard + 6 random easy only","independent_used_for_fit":False})
    pd.DataFrame(rule_rows).to_csv(OUT/"plaid_dynamic_signal_rule_feasibility.csv",index=False,float_format="%.10g")

    # Dynamic vs previous static feature AUC (orient both by old hard direction).
    compare=[]
    for _,r in hs.iterrows():compare.append({"feature_family":"dynamic","feature":r["signal"],"dataset":r["dataset"],"oriented_auc":r["oriented_auc"]})
    if V2.exists():
        s=pd.read_csv(V2)
        gap=s[s["feature"]=="gap_p10"]
        if len(gap)!=1:raise ValueError(f"Expected one summary row for static gap_p10, got {len(gap)}")
        r=gap.iloc[0]
        compare.append({"feature_family":"static","feature":"gap_p10","dataset":"old","oriented_auc":r["old_oriented_auc"]})
        compare.append({"feature_family":"static","feature":"gap_p10","dataset":"independent","oriented_auc":r["independent_oriented_auc_old_direction"]})
    comp=pd.DataFrame(compare);comp.to_csv(OUT/"plaid_dynamic_vs_static_auc.csv",index=False,float_format="%.10g")

    # Requested plots.
    g=decision[decision.dataset=="independent"]
    fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([g.loc[g.hard_label_posthoc==0,"top75_churn_300_400"],g.loc[g.hard_label_posthoc==1,"top75_churn_300_400"],g.loc[g.hard_label_posthoc==0,"top100_full_churn_400_512"],g.loc[g.hard_label_posthoc==1,"top100_full_churn_400_512"]],tick_labels=["easy\nTop75 300→400","hard\nTop75 300→400","easy\nTop100 400→512","hard\nTop100 400→512"]);ax.set_ylabel("Result-set churn");ax.grid(axis="y",alpha=.25);savefig(fig,"plaid_dynamic_hard_easy_top100_churn.png")
    fig,ax=plt.subplots(figsize=(6,4));ax.boxplot([g.loc[g.hard_label_posthoc==0,"candidate_growth_300_400"],g.loc[g.hard_label_posthoc==1,"candidate_growth_300_400"]],tick_labels=["easy","hard"]);ax.set_ylabel("retained candidate growth, 300→400");ax.grid(axis="y",alpha=.25);savefig(fig,"plaid_dynamic_hard_easy_candidate_growth.png")
    fig,ax=plt.subplots(figsize=(7,4))
    for hard,label,color in [(0,"easy","tab:blue"),(1,"hard","tab:red")]:
        d=raw[(raw.dataset=="independent")&(raw.actual_hard_label_posthoc==hard)]
        means=[]
        for b in BUDGETS[1:]:means.append(d[d.budget==b].top100_overlap_prev_count.mean())
        ax.plot(BUDGETS[1:],means,marker="o",label=label,color=color)
    ax.set_xlabel("current budget (adjacent previous→current)");ax.set_ylabel("mean Top100 intersection count");ax.set_xticks(BUDGETS[1:]);ax.legend();ax.grid(alpha=.25);savefig(fig,"plaid_dynamic_budget_top100_overlap.png")
    fig,ax=plt.subplots(figsize=(7,4))
    for hard,label,color in [(0,"easy","tab:blue"),(1,"hard","tab:red")]:
        d=raw[(raw.dataset=="independent")&(raw.actual_hard_label_posthoc==hard)]
        ax.plot(BUDGETS,[d[d.budget==b].recall_at_100.mean() for b in BUDGETS],marker="o",label=label,color=color)
    ax.set_xlabel("ndocs budget");ax.set_ylabel("Recall@100 (post-hoc qrels)");ax.set_xticks(BUDGETS);ax.legend();ax.grid(alpha=.25);savefig(fig,"plaid_dynamic_budget_recall100_posthoc.png")
    fig,ax=plt.subplots(figsize=(7,4))
    for qid,label,color in [(65,"hard q65","tab:red"),(50,"easy q50","tab:blue")]:
        d=raw[(raw.dataset=="independent")&(raw.query_id==qid)].sort_values("budget")
        ax.plot(d.budget,d.top100_churn_prev,marker="o",label=label,color=color)
    ax.set_xlabel("current budget");ax.set_ylabel("Top100 churn from previous budget");ax.set_xticks(BUDGETS);ax.legend();ax.grid(alpha=.25);savefig(fig,"plaid_dynamic_representative_churn.png")
    fig,ax=plt.subplots(figsize=(9,4));d=hs[hs.signal.isin(["top75_churn_300_400","candidate_growth_300_400","candidate_churn_300_400","top100_full_churn_400_512"])]
    for ds,marker in [("old","o"),("independent","s")]:
        x=d[d.dataset==ds];ax.scatter(x.signal,x.oriented_auc,label=ds,marker=marker,s=55)
    s=pd.read_csv(V2);s=s[s["feature"]=="gap_p10"].iloc[0]
    ax.scatter(["gap_p10","gap_p10"],[float(s["old_oriented_auc"]),float(s["independent_oriented_auc_old_direction"])],marker="D",label="static gap_p10")
    ax.axhline(.5,color="gray",linestyle="--");ax.set_ylim(0,1);ax.set_ylabel("oriented AUC");ax.tick_params(axis="x",rotation=25);ax.legend();ax.grid(axis="y",alpha=.25);savefig(fig,"plaid_dynamic_vs_static_auc.png")

    meta=json.loads((OUT/"metadata.json").read_text())
    summary=[]
    for ds,g0 in raw.groupby("dataset"):
        hard=g0[g0.actual_hard_label_posthoc==1];easy=g0[g0.actual_hard_label_posthoc==0]
        for b in BUDGETS:
            z=g0[g0.budget==b]
            summary.append({"dataset":ds,"budget":b,"n_queries":z.query_id.nunique(),"fixed400_hard_queries":int(z.actual_hard_label_posthoc.sum()),
              "fixed400_easy_queries":int(z.query_id.nunique()-z.actual_hard_label_posthoc.sum()),"mean_score_stage_ms":z.budget_score_refine_ms.mean(),
              "mean_standalone_e2e_ms":z.standalone_budget_e2e_ms.mean(),"mean_recall100_posthoc":z.recall_at_100.mean(),
              "hard_mean_recall100":z[z.actual_hard_label_posthoc==1].recall_at_100.mean(),"easy_mean_recall100":z[z.actual_hard_label_posthoc==0].recall_at_100.mean(),
              "mean_retained_count":z.retained_candidate_count.mean(),"mean_top100_overlap_prev":z.top100_overlap_prev_count.mean()})
    pd.DataFrame(summary).to_csv(OUT/"plaid_dynamic_signal_budget_summary.csv",index=False,float_format="%.10g")
    independent=decision[decision.dataset=="independent"]
    aucpick=hs[(hs.dataset=="independent")&hs.signal.isin(["top75_churn_300_400","candidate_growth_300_400","candidate_churn_300_400","top100_full_churn_400_512"])].sort_values("oriented_auc",ascending=False)
    rules=pd.DataFrame(rule_rows)
    trainfp=rules[(rules.rule.str.startswith("top75_churn"))&(rules.dataset_role=="old_train_apparent")].iloc[0]
    transfer=rules[(rules.rule.str.startswith("top75_churn"))&(rules.dataset_role=="independent_frozen_test")].iloc[0]
    dyn_auc=lambda ds,sig: float(hs[(hs.dataset==ds)&(hs.signal==sig)].oriented_auc.iloc[0])
    old_gap_auc=float(pd.read_csv(V2).loc[lambda x:x.feature=="gap_p10","old_oriented_auc"].iloc[0])
    new_gap_auc=float(pd.read_csv(V2).loc[lambda x:x.feature=="gap_p10","independent_oriented_auc_old_direction"].iloc[0])
    hard_ind=raw[(raw.dataset=="independent")&(raw.query_group=="hard")]
    recovery={b:int((hard_ind[hard_ind.budget==b].recall_at_100 >= hard_ind[hard_ind.budget==1024].set_index("query_id").loc[hard_ind[hard_ind.budget==b].query_id,"recall_at_100"].to_numpy()-1e-12).sum()) for b in BUDGETS}
    # Through-budget latency is measured with one shared encode/candidate-generation,
    # but every score_pids stage is rerun from cloned candidates (not incremental).
    cost=decision[decision.dataset=="independent"]
    lines=["# PLAID dynamic-signal feasibility: diagnostic results","",
      "## Protocol","",
      f"- Instrumentation cohort: independent {9} known fixed400-hard + 27 seed-random easy; old {3} known hard + 6 seed-random easy (45 total). Budgets: {BUDGETS}.",
      "- Existing 50k indexes/checkpoint were reused; CPU-only, 2 threads; no index build, full benchmark, or 13-config sweep.",
      "- Query encoding and `ranker.retrieve` candidate generation were done once per query. For each budget, `score_pids` was invoked on a cloned copy of that same candidate list; this supports clean budget comparison but recomputes pruning/refinement at each budget.",
      "- Qrels/Recall labels are attached after all searches for hindsight analysis only. Dynamic features/rules use only outputs available through ndocs=400 when deciding whether to continue.",
      "","## Main independent-set findings","",
      f"- Fixed400<1024 Recall@100 label: {int(independent.hard_label_posthoc.sum())} hard / {len(independent)-int(independent.hard_label_posthoc.sum())} easy. At 300→400, matched Top75 churn averages {independent[independent.hard_label_posthoc==1].top75_churn_300_400.mean():.3f} for hard vs {independent[independent.hard_label_posthoc==0].top75_churn_300_400.mean():.3f} for easy (oriented AUC {dyn_auc('independent','top75_churn_300_400'):.3f}); old controls go in the opposite direction (AUC {dyn_auc('old','top75_churn_300_400'):.3f}).",
      f"- Candidate growth is exactly 0.333 for both classes (AUC 0.500): retained counts are mechanically ndocs//4, so this is not a discriminating saturation signal. Full Top100 churn at 400→512 is also weak (independent AUC {dyn_auc('independent','top100_full_churn_400_512'):.3f}).",
      f"- Frozen old-trained Top75-churn cutoff: independent hard recall {transfer.hard_recall:.0%}, precision {transfer.precision:.1%}, FN={int(transfer.FN)}, FP={int(transfer.FP)} of 27 easy queries. It catches all 9 hard queries, but promotes 24 easy ones; the two-signal OR promotes all 36.",
      f"- Static gap_p10 oriented AUC is old {old_gap_auc:.3f}, independent {new_gap_auc:.3f}; the dynamic Top75-churn signal does not beat it consistently across datasets.",
      f"- Independent hard queries matching their own 1024 Recall@100 by budget: {', '.join(f'{b}:{recovery[b]}/9' for b in BUDGETS)}. The mean Recall@100 of the 36-query cohort is 0.850 at 400 and 0.908 at 1024; mean standalone end-to-end latency is {cost.standalone400_ms.mean():.2f} and {cost.standalone1024_ms.mean():.2f} ms/query, respectively.",
      f"- At budget 400, naive sequential scoring through 200→300→400 costs mean {cost.naive_progressive_to400_ms.mean():.2f} ms/query, versus {cost.standalone400_ms.mean():.2f} ms for standalone 400 (extra {cost.naive_extra_over_standalone400_ms.mean():.2f} ms). Through 1024, six full score_pids calls cost mean {cost.naive_progressive_to1024_ms.mean():.2f} ms versus standalone 1024 {cost.standalone1024_ms.mean():.2f} ms. This is not optimized incremental latency.",
      "- Approximate score values at the native cutoff remain unavailable. No score margin/slope was fabricated.",
      "","## Rule feasibility (exploratory only)","",
      f"- One-signal Top75 churn threshold selected on old 3-hard/6-easy only: {t if 't' in locals() else trainfp.threshold}. Old apparent hard recall/FP: {trainfp.hard_recall:.3f}/{int(trainfp.FP)}. Frozen independent result: hard recall={transfer.hard_recall:.3f}, FP={int(transfer.FP)}, FN={int(transfer.FN)}.",
      "- This is a tiny source set and is only feasibility evidence; do not present it as a validated policy. The two-signal OR row is also included in rule-feasibility CSV.",
      "","## Interpretation and answers","",
      "1. Current evidence for an early dynamic signal is weak and dataset-unstable: Top75 churn AUC is 0.652 independent but 0.167 old; candidate growth is quota-driven and AUC 0.5.",
      "2. The existing fork returns ndocs//4 final scored PIDs: ndocs=200 yields only 50 and ndocs=300 only 75, so a true Top100 comparison at 300→400 is impossible without changing search semantics. The pre-400 signal uses matched Top75 prefixes; full Top100 churn begins at 400→512, which is too late to decide at 400.",
      "3. At the frozen cutoff, hard recall is 9/9 but it falsely escalates 24/27 easy queries (precision 27.3%); this is not an efficient policy.",
      "4. gap_p10 is numerically more stable than Top75 churn in direction (old-oriented AUC 0.807/0.642), though its independent separation is moderate and the samples are tiny.",
      "5. No practical continue/stop rule is validated: avoiding misses currently costs escalating two-thirds of the independent easy sample. Treat as a negative/inconclusive feasibility result, not a deployment predictor.",
      "6. Measured latency is a naive repeated `score_pids` progression with one shared encoding/candidate-generation pass. It is not the cost of an incremental refinement implementation.",
      "7. PLAID code can reuse query encoding and raw candidate generation across budgets, since `retrieve` is independent of `ndocs`; `score_pids` currently applies native filtering, decompression and MaxSim in one call.",
      "8. Minimal engineering: factor `score_pids` into (a) cached approximate per-PID scores/order and nested retained-PID selection, then (b) incremental residual decompression/MaxSim refinement with a resumable accumulator. The CPU native `filter_pids` returns only selected PIDs, not scores or continuation state; expose a read-only score/order buffer or keep it in Python without changing ordering. Add equivalence tests that selected PID sets and final scores match current code at every fixed budget.",
      "9. Next step: **incremental refinement/early termination inside fixed candidate generation** for the truncation-dominant failure mode; handle candidate-generation failure separately with adaptive `ncells/nprobe`. Do not claim a final adaptive algorithm yet.",
      "","## Limits","",
        "The budget-stage calls reuse one candidate-generation result but rerun the existing `score_pids` function for every ndocs value. Thus the instrumentation measures budget outcomes and a naive sequential cost, not true streaming/incremental search. Candidate-input PID set hashes are invariant across all budgets; retained sets after pruning are what can change. The retained count is mechanically ndocs//4 in this fork, so its count growth itself is quota-driven rather than an independent saturation signal.",
      ""]
    (OUT/"plaid_dynamic_signal_final_analysis.md").write_text("\n".join(lines),encoding="utf-8")
    feasibility=["# PLAID incremental-search engineering feasibility","",
      "## Call-chain findings","",
      "`Searcher.encode(query)` → `IndexScorer.retrieve(config,Q)` → `IndexScorer.score_pids(config,Q,pids,centroid_scores)`.",
      "- Query encoding is independent of ndocs and can be reused.",
      "- `retrieve` uses query embeddings, ncells and IVF lookup; ndocs is not consumed there, so the raw candidate PID list and centroid score matrix can be reused for every ndocs checkpoint.",
      "- CPU `score_pids` computes centroid threshold mask, calls native `filter_pids` with ndocs (which internally performs staged approximate selection), then decompresses residual embeddings and runs exact MaxSim. The call returns only final scores/PIDs plus, in this fork, timing/counter fields; native `filter_pids` exposes no boundary score array or resumable state.",
      "- The instrumentation confirms the input candidate PID tensor is unchanged when cloned into score_pids. It deliberately invokes score_pids from scratch per budget to preserve reference semantics.",
      "","## What is reusable vs repeated","",
      "Reusable now: encoded Q, raw candidate PIDs, full centroid score matrix, index-resident doclens/codes/residuals. Repeated now: native approximate PID scoring/filtering, final retained PID selection, residual decompression and MaxSim for every budget call.",
      "A true 400→512→1024 path should retain the approximate order/score buffer and extend the selected prefix, then refine only newly admitted PIDs and update top-k incrementally. This requires refactoring `IndexScorer.score_pids` and the C++ `filter_pids` boundary; the current opaque function cannot continue from a previous budget.",
      "","## Minimal safe change","",
      "1. Add an internal helper returning (filtered_pid_order, approximate_scores, stage metadata) while keeping the current public score_pids behavior as a wrapper.",
      "2. Ensure deterministic nested prefixes for fixed ncells/threshold and exact parity with current native filter at ndocs 200/300/400/512/768/1024.",
      "3. Add an incremental refiner state containing selected PIDs already decompressed/scored, current top100 and enough score/rank state to insert only newly admitted PIDs.",
      "4. Add diagnostics-only mode first; assert identical final candidate sets and top100 to unmodified search at each budget before timing or policy evaluation.",
      "5. Only then add a continue condition based on pre-qrels signals (boundary margin, current top100 stability after new-PID refinement). Do not tune on the held-out independent labels.",
      "","## Boundary-score feasibility","",
      "The CPU native function reads candidate IDs/codes/doc lengths/centroid scores and returns selected PIDs. It does not return approximate scores, score ranks or heap state. Current saved cutoff feature rows mark those values unavailable. A small API refactor can return the already-computed heap/prefix scores; do not recompute them separately with a potentially different tie/order implementation.",
      "","## Decision","",
      "The existing pipeline supports **reuse of encoding and candidate generation**, but not true incremental refinement as a public operation. The most appropriate next implementation is incremental refinement with checkpoints on a fixed candidate pool; adaptive ncells is a distinct later branch for candidate-generation misses.",
      ""]
    (OUT/"plaid_incremental_search_feasibility.md").write_text("\n".join(feasibility),encoding="utf-8")
    print(f"Dynamic analysis outputs created in {OUT}")

if __name__=="__main__":main()
