"""Freeze a simple old48-only rule, test it unchanged on independent94, and replay offline."""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "v2_features"
FEATURES = OUT / "plaid_pruning_risk_full_features.csv"
SWEEP = ROOT / "plaid_query_parameter_sweep.csv"
NEW = ROOT / "plaid_independent_validation_per_query.csv"
OLD_RULE = ROOT / "plaid_frozen_adaptive_rule.json"
TRACES = ROOT / "plaid_relevant_pid_trace.csv"

GAPS = ["gap_p10", "worst_token_gap", "gap_p25"]
PAIR_RULES = [("gap_p10", "worst_token_gap"), ("gap_p10", "candidate_density_per_token")]

def qsummary(g):
    cols = ["total_latency_ms", "success_at_10", "mrr_at_10", "recall_at_10", "recall_at_100"]
    x = g.copy()
    # CSV inputs are parsed as strings by default; normalize IDs before they
    # become DataFrame indexes so later integer lookups cannot miss.
    x["query_id"] = pd.to_numeric(x["query_id"], errors="raise").astype(int)
    return x.groupby("query_id", as_index=False)[cols].mean()

def confusion(y, p):
    y=np.asarray(y,dtype=bool); p=np.asarray(p,dtype=bool)
    tp=int(np.sum(y&p)); fp=int(np.sum(~y&p)); fn=int(np.sum(y&~p)); tn=int(np.sum(~y&~p))
    prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
    spec=tn/(tn+fp) if tn+fp else 0.0; acc=(tp+tn)/max(len(y),1)
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    return dict(TP=tp,FP=fp,TN=tn,FN=fn,hard_recall=rec,hard_precision=prec,specificity=spec,accuracy=acc,F1=f1)

def threshold_values(s):
    a=np.sort(pd.to_numeric(s,errors="coerce").dropna().unique())
    if len(a)==0:return [float("inf")]
    mids=((a[:-1]+a[1:])/2).tolist()
    return [float(a[0])-1e-12,*[float(x) for x in mids],float(a[-1])+1e-12]

def predict(df, rule):
    if rule["logic"]=="single":
        f=rule["feature"]
        return pd.to_numeric(df[f],errors="coerce").le(rule["threshold"]).fillna(False).to_numpy()
    a=pd.to_numeric(df[rule["feature_1"]],errors="coerce"); b=pd.to_numeric(df[rule["feature_2"]],errors="coerce")
    pa=a.le(rule["threshold_1"]).fillna(False).to_numpy()
    pb=(b.ge(rule["threshold_2"]) if rule.get("direction_2")==">=" else b.le(rule["threshold_2"])).fillna(False).to_numpy()
    return (pa|pb) if rule["logic"]=="OR" else (pa&pb)

def fit_old_rule(old):
    y=old.ndocs400_sensitive.astype(bool).to_numpy()
    candidates=[]
    # Risk directions are fixed by feature mechanism: lower centroid gaps, higher candidate density.
    for f in ["gap_p10","worst_token_gap","gap_p25"]:
        for t in threshold_values(old[f]):
            candidates.append(({"logic":"single","feature":f,"direction":"<=","threshold":t},))
    for f1,f2 in PAIR_RULES:
        s1=old[f1]; s2=old[f2]
        dir2=">=" if f2=="candidate_density_per_token" else "<="
        for t1,t2 in itertools.product(threshold_values(s1),threshold_values(s2)):
            # Keep the paired lower-tail rule interpretable. Density is high-risk in its prespecified direction.
            for logic in ("OR","AND"):
                candidates.append(({"logic":logic,"feature_1":f1,"direction_1":"<=","threshold_1":t1,
                                    "feature_2":f2,"direction_2":dir2,"threshold_2":t2},))
    best=None
    for (r,) in candidates:
        if r["logic"]=="single": p=pd.to_numeric(old[r["feature"]],errors="coerce").le(r["threshold"]).fillna(False).to_numpy()
        else:
            x=pd.to_numeric(old[r["feature_1"]],errors="coerce"); z=pd.to_numeric(old[r["feature_2"]],errors="coerce")
            p1=x.le(r["threshold_1"]).fillna(False).to_numpy()
            p2=(z.ge(r["threshold_2"]) if r["direction_2"]==">=" else z.le(r["threshold_2"])).fillna(False).to_numpy()
            p=(p1|p2) if r["logic"]=="OR" else (p1&p2)
        m=confusion(y,p)
        # Prioritize zero missed risk queries; among ties, minimize false promotions, then prefer simpler rules.
        complexity=1 if r["logic"]=="single" else 2
        key=(m["FN"],m["FP"],complexity,-m["hard_precision"])
        if best is None or key<best[0]: best=(key,r,m,p)
    return best[1],best[2],best[3]

def metrics(rowdf, pred=None, feat_cost=None):
    d=rowdf.copy()
    if feat_cost is not None:
        d["total_latency_ms"] = d.total_latency_ms + d.query_id.map(feat_cost).fillna(0)
    lat=d.total_latency_ms.to_numpy(float); mean=float(np.mean(lat))
    out={"mean_latency_ms":mean,"p50_latency_ms":float(np.percentile(lat,50)),
         "p95_latency_ms":float(np.percentile(lat,95)),"qps":1000.0/mean if mean else np.nan,
         "success_at_10":float(d.success_at_10.mean()),"mrr_at_10":float(d.mrr_at_10.mean()),
         "recall_at_10":float(d.recall_at_10.mean()),"recall_at_100":float(d.recall_at_100.mean())}
    return out

def add_label(features, qmetrics, ds):
    base=qmetrics["baseline"].set_index("query_id"); fast=qmetrics["fixed400"].set_index("query_id")
    labels=(fast.recall_at_100.reindex(base.index)<base.recall_at_100-1e-12)
    f=features[features.dataset==ds].copy()
    f["ndocs400_sensitive"]=f.query_id.map(labels.to_dict()).fillna(False).astype(bool)
    f["recall100_drop_vs_baseline"]=f.query_id.map((base.recall_at_100-fast.recall_at_100).to_dict())
    return f

def build_old_metrics(sweep):
    result={}
    # The old 48-query sweep did not contain a separately named ndocs_1024
    # arm: its baseline config is exactly ncells=2, threshold=.45, ndocs=1024.
    for key,cfg in [("baseline","baseline"),("fixed400","ndocs_400"),("fixed1024","baseline")]:
        g=sweep[sweep.config_id==cfg].copy()
        g.total_latency_ms=pd.to_numeric(g.total_latency_ms)
        for c in ["success_at_10","mrr_at_10","recall_at_10","recall_at_100"]:g[c]=pd.to_numeric(g[c])
        result[key]=qsummary(g).rename(columns={"total_latency_ms":"total_latency_ms"})
    return result

def build_new_metrics(new):
    result={}
    for key,strategy in [("baseline","original_baseline"),("fixed400","fixed_ndocs_400"),("fixed1024","fixed_ndocs_1024")]:
        g=new[new.strategy==strategy].copy()
        for c in ["total_latency_ms","success_at_10","mrr_at_10","recall_at_10","recall_at_100"]:g[c]=pd.to_numeric(g[c])
        result[key]=qsummary(g)
    return result

def make_policy(ds, fm, qm, rule, feature_cost, old_rule=None):
    feats=fm.set_index("query_id")
    ids=sorted(set(qm["baseline"].query_id.astype(int)))
    pred=predict(feats.loc[ids].reset_index(),rule)
    oldpred=(old_rule(feats.loc[ids].reset_index()) if callable(old_rule) else predict(feats.loc[ids].reset_index(),old_rule)) if old_rule else None
    base=qm["baseline"].assign(query_id=lambda x: pd.to_numeric(x.query_id).astype(int)).set_index("query_id")
    a=qm["fixed400"].assign(query_id=lambda x: pd.to_numeric(x.query_id).astype(int)).set_index("query_id")
    b=qm["fixed1024"].assign(query_id=lambda x: pd.to_numeric(x.query_id).astype(int)).set_index("query_id")
    y=fm.set_index("query_id").ndocs400_sensitive.reindex(ids).astype(bool).to_numpy()
    policies={"Original baseline":(base,None),"Fixed 400":(a,None),"Fixed 1024":(b,None)}
    for name,p in [("New v2 adaptive 400/1024",pred),("Old adaptive 400/1024",oldpred)]:
        if p is None:continue
        chosen=[]
        for q,escalate in zip(ids,p):
            row=(b if escalate else a).loc[q].copy(); row["query_id"]=q
            row["total_latency_ms"]+=float(feature_cost.get(q,0.0))
            chosen.append(row)
        policies[name]=(pd.DataFrame(chosen),p)
    # Per-query quality-preserving oracle: fastest among baseline/400/1024 with baseline-level Recall@100.
    oracle=[]
    for q in ids:
        options=[base.loc[q],a.loc[q],b.loc[q]]
        target=float(base.loc[q].recall_at_100)
        valid=[r for r in options if float(r.recall_at_100)>=target-1e-12]
        row=min(valid,key=lambda r:float(r.total_latency_ms)).copy(); row["query_id"]=q; oracle.append(row)
    policies["Oracle upper bound"]=(pd.DataFrame(oracle),None)
    rows=[]; new_pred=pred.astype(bool); actual=y
    conf=confusion(actual,new_pred)
    for name,(frame,pp) in policies.items():
        m=metrics(frame)
        m.update({"dataset":ds,"policy":name,"promoted_queries":int(np.sum(pp)) if pp is not None else (len(ids) if name=="Fixed 1024" else 0),
                  "promoted_fraction":float(np.mean(pp)) if pp is not None else (1.0 if name=="Fixed 1024" else 0.0),
                  "hard_recall":conf["hard_recall"] if name.startswith("New") else (np.nan if name not in ("Fixed 400","Fixed 1024") else np.nan),
                  "false_negatives":conf["FN"] if name.startswith("New") else np.nan,
                  "false_positives":conf["FP"] if name.startswith("New") else np.nan})
        rows.append(m)
    return rows,conf,pred,oldpred

def main():
    required=[FEATURES,SWEEP,NEW,OLD_RULE]
    miss=[str(p) for p in required if not p.exists()]
    if miss:raise FileNotFoundError("Missing prerequisite(s): "+", ".join(miss))
    feat=pd.read_csv(FEATURES); sweep=pd.read_csv(SWEEP); new=pd.read_csv(NEW)
    # Keep the remotely collected predictor inputs immutable/idempotent. A prior
    # interrupted local analysis may have appended post-hoc columns to this CSV.
    raw_feature_cols=["dataset","query_id","feature_extraction_latency_ms","query_encode_latency_ms",
        "candidate_generation_latency_ms","query_token_length","gap_p10","gap_p25","worst_token_gap",
        "gap_mean","gap_std","bottom2_gap_mean","bottom4_gap_mean","centroid_top1_concentration",
        "centroid_entropy","candidate_count_at_ndocs1024","candidate_density_per_token",
        "query_embedding_norm_mean","query_embedding_norm_std","token_gap_json"]
    miss_raw=[c for c in raw_feature_cols if c not in feat.columns]
    if miss_raw: raise ValueError(f"Feature extraction CSV missing raw columns: {miss_raw}")
    feat=feat[raw_feature_cols].copy()
    if len(feat)!=142 or feat.groupby("dataset").size().to_dict()!={"independent":94,"old":48}:
        raise ValueError(f"Expected 48 old + 94 independent feature rows; got {feat.groupby('dataset').size().to_dict()}")
    if set(feat[feat.dataset=="old"].query_id)&set(feat[feat.dataset=="independent"].query_id):raise ValueError("query ID overlap")
    oldqm=build_old_metrics(sweep); newqm=build_new_metrics(new)
    old=add_label(feat,oldqm,"old"); ind=add_label(feat,newqm,"independent")
    old=old.merge(oldqm["baseline"][["query_id","total_latency_ms"]].rename(columns={"total_latency_ms":"baseline_latency_ms"}),on="query_id")
    # Candidate density has to be extracted before selecting the final rule.
    fitted,fitconf,fitpred=fit_old_rule(old)
    frozen={"version":2,"status":"frozen; trained only on old 48 queries",
        "label_definition":"ndocs400_sensitive: Recall@100 for fixed ndocs=400 is lower than original baseline for the same query",
        "features_used":([fitted["feature"]] if fitted["logic"]=="single" else [fitted["feature_1"],fitted["feature_2"]]),
        "decision_rule":fitted,"training_dataset":"old","training_query_ids":sorted(old.query_id.astype(int).tolist()),
        "easy_ndocs":400,"risk_ndocs":1024,"selection_objective":"minimize FN, then FP, then feature count on old48 only",
        "independent_data_used_for_selection":False}
    (OUT/"plaid_pruning_risk_rule_v2_frozen.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    oldrule=json.loads(OLD_RULE.read_text(encoding="utf-8"))["two_variable_or"]
    oldrule={"logic":"OR","feature_1":"query_embedding_norm_std","direction_1":"<=","threshold_1":oldrule["threshold_1"],
             "feature_2":"gap_mean","direction_2":"<=","threshold_2":oldrule["threshold_2"]}
    # Generalize prediction helper to the existing frozen rule's gap-mean alias.
    def old_predict(d):
        return ((pd.to_numeric(d.query_embedding_norm_std)<=oldrule["threshold_1"]) |
                (pd.to_numeric(d.gap_mean)<=oldrule["threshold_2"])).fillna(False).to_numpy()
    ind_pred=predict(ind,fitted); old_pred=old_predict(ind)
    ynew=ind.ndocs400_sensitive.to_numpy(bool)
    conf_rows=[]
    for name,p in [("old frozen rule",old_pred),("new v2 frozen rule",ind_pred)]:conf_rows.append({"rule":name,**confusion(ynew,p)})
    test=ind.copy();test["old_rule_predicted_risk"]=old_pred;test["v2_rule_predicted_risk"]=ind_pred
    test["v2_confusion_category"]=["TP" if a and p else "FN" if a else "FP" if p else "TN" for a,p in zip(ynew,ind_pred)]
    test.to_csv(OUT/"plaid_pruning_risk_v2_independent_test.csv",index=False,float_format="%.10g")
    # Full feature table includes labels strictly for post-hoc analysis; predictors remain early-only.
    full=pd.concat([old,ind],ignore_index=True,sort=False)
    full.to_csv(OUT/"plaid_pruning_risk_full_analysis_table.csv",index=False,float_format="%.10g")

    allpol=[]
    cost_old=old.set_index("query_id").feature_extraction_latency_ms.to_dict()
    cost_new=ind.set_index("query_id").feature_extraction_latency_ms.to_dict()
    for ds,fm,qm,cost in [("old",old,oldqm,cost_old),("independent",ind,newqm,cost_new)]:
        rows,conf,pred,oldp=make_policy(ds,fm,qm,fitted,cost,old_predict)
        allpol.extend(rows)
        if ds=="independent":
            # Add false-positive compute cost and false-negative quality loss separately.
            ids=sorted(ind.query_id.astype(int)); a=newqm["fixed400"].set_index("query_id"); b=newqm["fixed1024"].set_index("query_id")
            actual=ind.set_index("query_id").ndocs400_sensitive.astype(bool)
            pp=pd.Series(pred,index=ids); fp=[q for q in ids if pp[q] and not actual[q]]; fn=[q for q in ids if not pp[q] and actual[q]]
            fp_extra=float(np.mean([b.loc[q].total_latency_ms-a.loc[q].total_latency_ms for q in fp])) if fp else 0.0
            fn_drop=float(np.mean([float(b.loc[q].recall_at_100-a.loc[q].recall_at_100) for q in fn])) if fn else 0.0
            for row in allpol:
                if row.get("dataset")=="independent" and row.get("policy")=="New v2 adaptive 400/1024":
                    row["false_positive_extra_retrieval_ms_per_fp"]=fp_extra
                    row["false_negative_mean_recall100_loss_vs_1024"]=fn_drop
                    row["feature_overhead_mean_ms"]=float(ind.feature_extraction_latency_ms.mean())
    policies=pd.DataFrame(allpol)
    policies.to_csv(OUT/"plaid_pruning_risk_v2_policy_comparison.csv",index=False,float_format="%.8f")

    # Direct candidate-generation failure labels exist only for the traced subset.
    trace_note="No complete per-PID trace for all queries; only the existing traced subset is counted."
    cg_count=np.nan; trace_ids=[]
    if TRACES.exists():
        tr=pd.read_csv(TRACES); tr=tr[tr.ndocs==400]
        if "candidate_generation_present" in tr:
            present=tr.candidate_generation_present.astype(str).str.lower().isin(["true","1"])
            bad=tr[~present]
            trace_ids=sorted(bad.query_id.unique().astype(int).tolist()); cg_count=len(trace_ids)
    # ROC AUCs and feature direction are descriptive only; threshold selection remains old-only.
    auc_rows=[]
    feature_list=["gap_p10","worst_token_gap","gap_p25","gap_mean","query_embedding_norm_std","candidate_density_per_token","centroid_top1_concentration"]
    directions={}
    for f in feature_list:
        hm=old.loc[old.ndocs400_sensitive,f].median(); em=old.loc[~old.ndocs400_sensitive,f].median()
        directions[f]="<=" if hm<em else ">="
    for ds,g in [("old",old),("independent",ind)]:
        y=g.ndocs400_sensitive.to_numpy(bool)
        for f in feature_list:
            vals=pd.to_numeric(g[f],errors="coerce").to_numpy(float); good=np.isfinite(vals); yy=y[good]; vv=vals[good]
            pos=vv[yy]; neg=vv[~yy]
            raw=float(np.mean([(x>z)+0.5*(x==z) for x in pos for z in neg])) if len(pos) and len(neg) else np.nan
            oriented=raw if directions[f]==">=" else 1-raw
            auc_rows.append({"dataset":ds,"feature":f,"risk_direction_frozen_from_old":directions[f],"n":len(vv),"hard_n":int(yy.sum()),"auc_raw":raw,"auc_oriented":oriented,
                             "hard_median":float(np.median(pos)) if len(pos) else np.nan,
                             "easy_median":float(np.median(neg)) if len(neg) else np.nan})
    pd.DataFrame(auc_rows).to_csv(OUT/"plaid_pruning_risk_v2_feature_auc.csv",index=False,float_format="%.8f")

    # Figures: old/new distributions and policy outcomes.
    for feature,title,file in [("gap_p10","gap_p10: old48 vs independent94","plaid_v2_gap_p10_distribution.png"),
                               ("worst_token_gap","Worst-token centroid gap","plaid_v2_worst_token_gap_distribution.png"),
                               ("candidate_density_per_token","Candidate density per query token","plaid_v2_candidate_density_distribution.png")]:
        fig,axs=plt.subplots(1,2,figsize=(9,4))
        for ax,(ds,g) in zip(axs,[("old",old),("independent",ind)]):
            ax.boxplot([g.loc[g.ndocs400_sensitive,feature].dropna(),g.loc[~g.ndocs400_sensitive,feature].dropna()],labels=["hard","easy"])
            ax.set_title(f"{ds} (n={len(g)})");ax.set_ylabel(feature);ax.grid(axis="y",alpha=.25)
        fig.suptitle(title);fig.tight_layout();fig.savefig(OUT/file,dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,4));
    for ax,name,pred0 in zip(axs,["Old frozen rule","V2 frozen rule"],[old_pred,ind_pred]):
        cm0=confusion(ynew,pred0); mat=np.array([[cm0["TN"],cm0["FP"]],[cm0["FN"],cm0["TP"]]])
        im=ax.imshow(mat,cmap="Blues");ax.set_xticks([0,1],["easy","risk"]);ax.set_yticks([0,1],["easy","risk"]);ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.set_title(name)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(mat[i,j]),ha="center",va="center",fontsize=13)
    fig.suptitle("Independent94 confusion matrices");fig.tight_layout();fig.savefig(OUT/"plaid_v2_confusion.png",dpi=180);plt.close(fig)
    p=policies[(policies.dataset=="independent") & policies.policy.isin(["Original baseline","Fixed 400","Old adaptive 400/1024","New v2 adaptive 400/1024","Fixed 1024","Oracle upper bound"])]
    order=["Original baseline","Fixed 400","Old adaptive 400/1024","New v2 adaptive 400/1024","Fixed 1024","Oracle upper bound"]
    p=p.set_index("policy").reindex(order).reset_index()
    for col,ylabel,file in [("recall_at_100","Recall@100","plaid_v2_policy_recall100.png"),("mean_latency_ms","Mean latency including feature overhead (ms)","plaid_v2_policy_latency.png")]:
        fig,ax=plt.subplots(figsize=(10,4.5)); bars=ax.bar(p.policy,p[col],color=["#4472C4","#ED7D31","#A5A5A5","#70AD47","#5B9BD5","#8064A2"])
        ax.set_ylabel(ylabel);ax.tick_params(axis="x",rotation=25);ax.grid(axis="y",alpha=.25);ax.set_axisbelow(True)
        for b,v in zip(bars,p[col]):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:.3f}",ha="center",va="bottom",fontsize=8)
        fig.tight_layout();fig.savefig(OUT/file,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));
    for ds,g in [("old",old),("independent",ind)]:
        pred=predict(g,fitted);hard=g.ndocs400_sensitive.to_numpy(bool)
        frac=float(np.mean(pred));rec=confusion(hard,pred)["hard_recall"]
        ax.scatter(frac,rec,label=ds,s=65)
    ax.set_xlabel("Promoted query fraction");ax.set_ylabel("Hard-query recall");ax.set_ylim(-.03,1.03);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/"plaid_v2_hard_recall_vs_promotion.png",dpi=180);plt.close(fig)

    # Report exact cross-dataset answers, no threshold changes after freeze.
    oldm=confusion(old.ndocs400_sensitive,fitpred); newm=confusion(ynew,ind_pred); oldrulem=confusion(ynew,old_pred)
    baseline=policies[(policies.dataset=="independent")&(policies.policy=="Original baseline")].iloc[0]
    fixed=policies[(policies.dataset=="independent")&(policies.policy=="Fixed 400")].iloc[0]
    adaptive=policies[(policies.dataset=="independent")&(policies.policy=="New v2 adaptive 400/1024")].iloc[0]
    recovered=(adaptive.recall_at_100-fixed.recall_at_100)/(baseline.recall_at_100-fixed.recall_at_100) if baseline.recall_at_100!=fixed.recall_at_100 else np.nan
    residual=(baseline.recall_at_100-adaptive.recall_at_100)
    fixed1024=policies[(policies.dataset=="independent")&(policies.policy=="Fixed 1024")].iloc[0]
    residual1024=float(baseline.recall_at_100-fixed1024.recall_at_100)
    lines=["# PLAID v2 pruning-risk predictor: final validation", "",
      "## Protocol and leakage control", "",
      "- Feature-only collection over old48 and independent94; existing indexes reused; no index build, parameter sweep, qrels/rank/TopK access, or final document scoring during feature extraction.",
      "- Rule/threshold fitted only on old48 and immediately frozen in `plaid_pruning_risk_rule_v2_frozen.json`; independent94 used only for the held-out test and offline replay.",
      "- ndocs400_sensitive label is post-hoc: fixed400 Recall@100 < baseline Recall@100 for the same query.",
      "- Feature extraction latency (encode + early retrieval) is added to new-v2 adaptive replay. This is a conservative standalone-computation accounting; a cached implementation could avoid duplicated work.", "",
      "## Frozen rule", "",f"- Rule: `{json.dumps(fitted,ensure_ascii=False)}`",f"- Old48 fit: hard recall={oldm['hard_recall']:.3f}, precision={oldm['hard_precision']:.3f}, FN={oldm['FN']}, FP={oldm['FP']}.","",
      "## Independent94 classification", "",f"- Old frozen rule: recall={oldrulem['hard_recall']:.3f}, precision={oldrulem['hard_precision']:.3f}, FN={oldrulem['FN']}, FP={oldrulem['FP']}.",
      f"- New v2 frozen rule: recall={newm['hard_recall']:.3f}, precision={newm['hard_precision']:.3f}, FN={newm['FN']}, FP={newm['FP']}, specificity={newm['specificity']:.3f}, accuracy={newm['accuracy']:.3f}, F1={newm['F1']:.3f}.","",
      "## Independent94 offline policy replay", "",f"- Baseline: mean={baseline.mean_latency_ms:.3f} ms, Recall@100={baseline.recall_at_100:.4f}.",
      f"- Fixed400: mean={fixed.mean_latency_ms:.3f} ms, Recall@100={fixed.recall_at_100:.4f}.",
      f"- New v2 adaptive (feature overhead included): mean={adaptive.mean_latency_ms:.3f} ms, Recall@100={adaptive.recall_at_100:.4f}; promoted {int(adaptive.promoted_queries)}/{len(ind)} ({adaptive.promoted_fraction:.1%}); feature overhead mean={adaptive.feature_overhead_mean_ms:.3f} ms.",
      f"- Fixed400 Recall@100 loss recovered by v2: {recovered:.1%}; residual gap to baseline={residual:.4f}.",
      f"- Maximum observed ndocs-only recovery using fixed1024: {1-residual1024/(baseline.recall_at_100-fixed.recall_at_100) if baseline.recall_at_100!=fixed.recall_at_100 else float('nan'):.1%}; remaining fixed1024-to-baseline Recall@100 gap={residual1024:.4f}.",
      f"- Candidate-generation failure traces are incomplete; directly traced IDs not covered by ndocs promotion: {trace_ids if trace_ids else 'not available'} ({trace_note})", "",
      "## Answers", "",
      f"1. gap_p10 oriented AUC old/new: {next((x['auc_oriented'] for x in auc_rows if x['dataset']=='old' and x['feature']=='gap_p10'),float('nan')):.3f}/{next((x['auc_oriented'] for x in auc_rows if x['dataset']=='independent' and x['feature']=='gap_p10'),float('nan')):.3f}; direction is frozen from old48 (see feature AUC CSV).",
      "2. gap_p10 is compared with gap_mean and embedding_norm_std in `plaid_pruning_risk_v2_feature_auc.csv`; these are small-sample descriptive AUCs, not proof of generalization.",
      "3. The old48 threshold is applied unchanged on independent94.",
      f"4-5. v2 hard recall on independent94 is {newm['hard_recall']:.1%}; compared with old rule {oldrulem['hard_recall']:.1%}.",
      f"6-7. Replay recovers {recovered:.1%} of fixed400 Recall@100 loss; mean latency change vs baseline is {(1-adaptive.mean_latency_ms/baseline.mean_latency_ms):.1%} after feature overhead.",
      "8. Candidate-generation failures are separately trace-limited; do not count untraced queries as confirmed truncation cases.",
      "9-10. If hard recall remains low, FP promotion is excessive, or quality remains below baseline, stop tuning on independent94. The next step should be a dynamic search-time signal (candidate cutoff/pruning margin); a learned controller is not justified until broader data exist.","",
      "## Caveat and stopping decision", "",
      "This is a single held-out collection and a small set of truly hard queries. If v2 misses hard queries or needs many false promotions, conclude that simple query-side early features are insufficiently stable and do not retune on independent94.",
      ""]
    (OUT/"plaid_pruning_risk_v2_final_analysis.md").write_text("\n".join(lines),encoding="utf-8")
    print("Generated v2 frozen rule, independent test, policy replay, analysis and figures in",OUT)

if __name__=="__main__":main()
