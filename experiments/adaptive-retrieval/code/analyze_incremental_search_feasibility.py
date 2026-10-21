"""Offline feasibility analysis for a true incremental PLAID refiner.

Important: the source run called score_pids independently at each budget. This
script reports restart-reference traces and explicitly marks native continuation
and its latency as unmeasured; it does not relabel them as incremental results.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dynamic_signal_v1"
RAW=DATA/"plaid_dynamic_signal_budget_per_query.csv"
OUT=DATA
POC=ROOT.parent.parent/"plaid_incremental_poc"
STAGES=[400,512,1024]

def setjson(v):
    return set(map(int,json.loads(v)))

def overlap(a,b):
    i=len(a&b); u=len(a|b)
    return i,u,(i/u if u else 1.0)

def main():
    d=pd.read_csv(RAW)
    d["query_id"]=d.query_id.astype(int); d["budget"]=d.budget.astype(int)
    by={(r.dataset,int(r.query_id),int(r.budget)):r for r in d.itertuples(index=False)}
    qs=sorted(set((r.dataset,int(r.query_id)) for r in d.itertuples(index=False)))
    trace=[]; comp=[]; policies=[]
    # The only frozen policy considered here is the previously selected old-set
    # Top75 churn threshold. Candidate growth is recorded, not re-tuned.
    CHURN_THRESHOLD=0.12
    GROWTH_THRESHOLD=0.34
    for ds,qid in qs:
        a,b,c=(by[(ds,qid,k)] for k in STAGES)
        sets={k:setjson(by[(ds,qid,k)].retained_candidate_pids_json) for k in STAGES}
        tops={k:setjson(by[(ds,qid,k)].top100_ids_json) for k in STAGES}
        ranks={k:{pid:i+1 for i,pid in enumerate(json.loads(by[(ds,qid,k)].top100_ids_json))} for k in STAGES}
        scores={k:{pid:s for pid,s in zip(json.loads(by[(ds,qid,k)].top100_ids_json),json.loads(by[(ds,qid,k)].top100_scores_json))} for k in STAGES}
        for prev,cur in [(400,512),(512,1024),(400,1024)]:
            x,y=sets[prev],sets[cur]; i,u,j=overlap(x,y)
            ti,tu,tj=overlap(tops[prev],tops[cur])
            common=sorted(tops[prev]&tops[cur])
            rankrho=float(np.corrcoef([ranks[prev][p] for p in common],[ranks[cur][p] for p in common])[0,1]) if len(common)>1 else np.nan
            both_score=sorted(set(scores[prev])&set(scores[cur]))
            score_delta=float(np.mean([scores[cur][p]-scores[prev][p] for p in both_score])) if both_score else np.nan
            trace.append({"dataset":ds,"query_id":qid,"query_group":a.query_group,"from_budget":prev,"to_budget":cur,
                "restart_reference_only":True,"incremental_execution_tested":False,
                "retained_before":len(x),"retained_after":len(y),"candidate_added":len(y-x),"candidate_removed":len(x-y),
                "candidate_overlap":i,"candidate_jaccard":j,"top100_overlap":ti,"top100_jaccard":tj,
                "top100_churn_fraction":1-ti/100.0,"top100_new":len(tops[cur]-tops[prev]),"top100_removed":len(tops[prev]-tops[cur]),
                "common_pid_count":len(common),"common_pid_rank_pearson":rankrho,
                "common_top100_mean_score_delta_to":score_delta,
                "recall100_before":by[(ds,qid,prev)].recall_at_100,"recall100_after":by[(ds,qid,cur)].recall_at_100,
                "recall10_before":by[(ds,qid,prev)].recall_at_10,"recall10_after":by[(ds,qid,cur)].recall_at_10,
                "mrr10_before":by[(ds,qid,prev)].mrr_at_10,"mrr10_after":by[(ds,qid,cur)].mrr_at_10,
                "success10_before":by[(ds,qid,prev)].success_at_10,"success10_after":by[(ds,qid,cur)].success_at_10})
        # Direct 400->512->1024 quality/identity parity reference.
        identical_candidate_inputs=(a.candidate_input_sha256==b.candidate_input_sha256==c.candidate_input_sha256)
        # Stage-400 signals were computed from the 300->400 transition, before
        # any qrels-based label is consulted. Threshold is frozen from old data.
        sig=by[(ds,qid,400)]
        churn=float(sig.common_prefix_churn_prev)
        previous_retained=float(sig.candidate_overlap_prev_count + sig.candidate_removed_prev_count)
        growth=float(sig.candidate_new_prev_count/previous_retained) if previous_retained else np.nan
        stop=(churn<CHURN_THRESHOLD and growth<GROWTH_THRESHOLD)
        hard=int(a.recall_at_100<c.recall_at_100-1e-12) # post-hoc eval only
        seq_cost=float(a.standalone_budget_e2e_ms + b.budget_score_refine_ms + c.budget_score_refine_ms +
                      sig.signal_compute_ms_from_previous + by[(ds,qid,512)].signal_compute_ms_from_previous +
                      by[(ds,qid,1024)].signal_compute_ms_from_previous)
        policies.append({"dataset":ds,"query_id":qid,"query_group":a.query_group,
            "frozen_churn_threshold":CHURN_THRESHOLD,"frozen_candidate_growth_threshold":GROWTH_THRESHOLD,
            "observed_churn_300_400":churn,"observed_candidate_growth_300_400":growth,
            "stop_at_400_by_frozen_rule":int(stop),"selected_final_budget":400 if stop else 1024,
            "posthoc_fixed400_hard_label":hard,"posthoc_only_label_not_used_for_decision":True,
            "policy_recall100":a.recall_at_100 if stop else c.recall_at_100,
            "policy_recall10":a.recall_at_10 if stop else c.recall_at_10,
            "policy_mrr10":a.mrr_at_10 if stop else c.mrr_at_10,
            "policy_success10":a.success_at_10 if stop else c.success_at_10,
            "always400_recall100":a.recall_at_100,"always1024_recall100":c.recall_at_100,
            "always400_e2e_ms":a.standalone_budget_e2e_ms,"always1024_e2e_ms":c.standalone_budget_e2e_ms,
            "oracle_posthoc_selected_budget":1024 if hard else 400,
            "oracle_posthoc_e2e_ms":c.standalone_budget_e2e_ms if hard else a.standalone_budget_e2e_ms,
            "restart_checkpoint_sequence_400_512_1024_ms_proxy":seq_cost,
            "true_incremental_additional_time_measured":False,
            "incremental_top100_exact_match_tested":False,"incremental_recall_match_tested":False,
            "saved_restart_reference_512_available":True,"saved_restart_reference_1024_available":True,
            "raw_candidate_set_same_all_stages":identical_candidate_inputs})
        # Stage-oriented trace rows retain each recorded full-refinement result.
        for k in STAGES:
            r=by[(ds,qid,k)]
            trace.append({"dataset":ds,"query_id":qid,"query_group":r.query_group,"from_budget":np.nan,"to_budget":k,
                "restart_reference_only":True,"incremental_execution_tested":False,"retained_before":np.nan,
                "retained_after":r.retained_candidate_count,"candidate_added":np.nan,"candidate_removed":np.nan,
                "candidate_overlap":np.nan,"candidate_jaccard":np.nan,"top100_overlap":np.nan,"top100_jaccard":np.nan,
                "top100_churn_fraction":np.nan,"top100_new":np.nan,"top100_removed":np.nan,"common_pid_count":np.nan,
                "common_pid_rank_pearson":np.nan,"common_top100_mean_score_delta_to":np.nan,
                "recall100_before":np.nan,"recall100_after":r.recall_at_100,"recall10_before":np.nan,"recall10_after":r.recall_at_10,
                "mrr10_before":np.nan,"mrr10_after":r.mrr_at_10,"success10_before":np.nan,"success10_after":r.success_at_10,
                "full_score_pids_ms_restart":r.budget_score_refine_ms,"e2e_ms_restart":r.standalone_budget_e2e_ms})
    pd.DataFrame(trace).to_csv(OUT/"plaid_incremental_query_trace.csv",index=False,float_format="%.10g")
    comp_df=pd.DataFrame(policies)
    comp_df.to_csv(OUT/"plaid_incremental_vs_restart.csv",index=False,float_format="%.10g")
    # Aggregate standard policies and the frozen historical rule by dataset.
    ag=[]
    for ds,g in comp_df.groupby("dataset"):
        source=d[d.dataset==ds]
        metric=lambda budget,col: source[source.budget==budget].set_index("query_id").loc[g.query_id,col].to_numpy()
        for name,mask,budget_col in [
            ("always400",np.ones(len(g),bool),None),
            ("always1024",np.ones(len(g),bool),None),
            ("frozen_early_stop_rule",np.ones(len(g),bool),None),
            ("oracle_posthoc_upper_bound",np.ones(len(g),bool),None)]:
            if name=="always400":
                lat=metric(400,"standalone_budget_e2e_ms"); q=g.always400_recall100.to_numpy(); q10=metric(400,"recall_at_10"); mrr=metric(400,"mrr_at_10"); suc=metric(400,"success_at_10"); promoted=np.zeros(len(g),int)
            elif name=="always1024":
                lat=metric(1024,"standalone_budget_e2e_ms"); q=g.always1024_recall100.to_numpy(); q10=metric(1024,"recall_at_10"); mrr=metric(1024,"mrr_at_10"); suc=metric(1024,"success_at_10"); promoted=np.ones(len(g),int)
            elif name=="frozen_early_stop_rule":
                use400=g.stop_at_400_by_frozen_rule.to_numpy()==1; lat=np.where(use400,g.always400_e2e_ms.to_numpy(),g.always1024_e2e_ms.to_numpy()); q=g.policy_recall100.to_numpy(); q10=g.policy_recall10.to_numpy(); mrr=g.policy_mrr10.to_numpy(); suc=g.policy_success10.to_numpy(); promoted=(~use400).astype(int)
            else:
                hard=g.posthoc_fixed400_hard_label.to_numpy()==1; lat=g.oracle_posthoc_e2e_ms.to_numpy(); q=np.where(hard,g.always1024_recall100.to_numpy(),g.always400_recall100.to_numpy()); q10=np.where(hard,metric(1024,"recall_at_10"),metric(400,"recall_at_10")); mrr=np.where(hard,metric(1024,"mrr_at_10"),metric(400,"mrr_at_10")); suc=np.where(hard,metric(1024,"success_at_10"),metric(400,"success_at_10")); promoted=hard.astype(int)
            ag.append({"dataset":ds,"policy":name,"n_queries":len(g),"mean_e2e_ms_restart_reference":float(np.mean(lat)),
                "p50_e2e_ms_restart_reference":float(np.median(lat)),"p95_e2e_ms_restart_reference":float(np.percentile(lat,95)),
                "mean_recall100":float(np.nanmean(q)),"mean_recall10":float(np.nanmean(q10)),"mean_mrr10":float(np.nanmean(mrr)),"success_at_10_rate":float(np.nanmean(suc)),
                "promoted_fraction":float(np.mean(promoted)),"mean_qps_inverse_latency":float(1000/np.mean(lat)),
                "true_incremental_latency_measured":False})
    pd.DataFrame(ag).to_csv(OUT/"plaid_early_stop_policy_analysis.csv",index=False,float_format="%.10g")

    # Requested POC runtime tables. These are restart-reference measurements;
    # all unavailable incremental components remain blank, never imputed.
    runtime=[]; early=[]
    for ds,qid in qs:
        r400=by[(ds,qid,400)]; r512=by[(ds,qid,512)]; r1024=by[(ds,qid,1024)]
        enc=float(r400.query_encode_ms); gen=float(r400.candidate_generation_ms)
        s400=float(r400.budget_score_refine_ms); s512=float(r512.budget_score_refine_ms); s1024=float(r1024.budget_score_refine_ms)
        restart_e2e=(enc+gen+s400)+(enc+gen+s1024)
        runtime.append({"dataset":ds,"query_id":qid,"n_docs":50000,"cpu_threads":2,
            "restart_400_encode_ms":enc,"restart_400_candidate_generation_ms":gen,
            "restart_400_score_pids_total_ms":s400,"restart_1024_encode_ms":enc,
            "restart_1024_candidate_generation_ms":gen,"restart_1024_score_pids_total_ms":s1024,
            "restart_400_plus_1024_e2e_replay_ms":restart_e2e,
            "restart_prune_ms_separate":np.nan,"restart_decompress_ms_separate":np.nan,"restart_maxsim_ms_separate":np.nan,
            "incremental_encode_ms_expected_once":enc,"incremental_candidate_generation_ms_expected_once":gen,
            "incremental_stage400_score_pids_reference_ms":s400,"incremental_400_to_1024_delta_ms":np.nan,
            "incremental_total_e2e_ms":np.nan,"incremental_prune_ms":np.nan,"incremental_decompress_ms":np.nan,
            "incremental_maxsim_ms":np.nan,"latency_saved_vs_restart_ms":np.nan,"incremental_parity_status":"NOT_RUN",
            "note":"saved restart references only; score_pids timer includes filter+decompress+MaxSim; incremental delta not measured"})
        churn=float(r400.common_prefix_churn_prev)
        old_n=float(r400.candidate_overlap_prev_count+r400.candidate_removed_prev_count)
        growth=float(r400.candidate_new_prev_count/old_n) if old_n else np.nan
        stop=bool(churn<CHURN_THRESHOLD and growth<GROWTH_THRESHOLD)
        early.append({"dataset":ds,"query_id":qid,"stage_budget":400,
            "top75_churn_300_400":churn,"candidate_growth_300_400":growth,
            "common_pid_rank_pearson_300_400":float(r400.common_top100_spearman_prev),
            "score_margin_signal":"unavailable in saved instrumentation",
            "frozen_rule":"STOP iff churn < 0.12 AND growth < 0.34",
            "predicted_decision":"STOP" if stop else "CONTINUE","predicted_next_budget":400 if stop else 1024,
            "decision_compute_ms_restart_reference":float(r400.signal_compute_ms_from_previous),
            "true_incremental_state_used":False,"qrels_or_future_quality_used":False,
            "signal_source":"restart-reference 300/400 outputs, not live incremental state"})
    pd.DataFrame(runtime).to_csv(POC/"incremental_vs_restart_runtime.csv",index=False,float_format="%.10g")
    pd.DataFrame(early).to_csv(POC/"early_stop_runtime_simulation.csv",index=False,float_format="%.10g")

    # Evidence-driven engineering analysis.
    hard_counts=comp_df.groupby("dataset").posthoc_fixed400_hard_label.agg(["sum","count"])
    independent=comp_df[comp_df.dataset=="independent"]
    frozen=independent[independent.stop_at_400_by_frozen_rule==0]
    stop=independent[independent.stop_at_400_by_frozen_rule==1]
    stage= d[d.dataset=="independent"].groupby("budget").agg(mean_e2e=("standalone_budget_e2e_ms","mean"),mean_score=("budget_score_refine_ms","mean"))
    nesting=[]
    for ds in sorted(comp_df.dataset.unique()):
        g=d[(d.dataset==ds)&d.budget.isin(STAGES)]
        maps={int(b):{int(q):setjson(ids) for q,ids in zip(x.query_id,x.retained_candidate_pids_json)} for b,x in g.groupby("budget")}
        n=len(maps[400]); ok=sum(maps[400][q].issubset(maps[512][q]) and maps[512][q].issubset(maps[1024][q]) for q in maps[400])
        nesting.append(f"{ds} {ok}/{n}")
    seq_mean=float(independent.restart_checkpoint_sequence_400_512_1024_ms_proxy.mean())
    fresh1024=float(independent.always1024_e2e_ms.mean())
    lines=["# PLAID incremental-search feasibility and small-sample evidence","",
      "## Scope / honesty of the prototype","",
      "The downloaded experiment is restart-reference instrumentation, not a true continuation implementation. For each query, query encoding and `retrieve()` ran once, but `score_pids()` ran independently from cloned raw candidate IDs at each of six budgets. Therefore this report does **not** claim that incremental Top100/Recall parity or incremental latency has been tested. A read-only source excerpt was inspected from the isolated remote worktree; no remote code has yet been changed or executed for this PoC.",
      "","## 1. Pipeline / state flow","",
      "`Searcher.search()` / `IndexScorer.rank()` → `IndexScorer.retrieve(config,Q)` → `CandidateGeneration.generate_candidates()` → `IndexScorer.score_pids(config,Q,pids,centroid_scores)` → `colbert_score_packed()` → caller sorts TopK. `generate_candidates()` chooses `ncells`, looks up IVF PIDs, sorts/deduplicates them, and returns raw PIDs plus the all-centroid score matrix. CPU `score_pids()` applies the centroid threshold, calls native `filter_pids`, decompresses selected residuals, then computes `D_packed @ Q.T` and runs `segmented_maxsim`.",
      "","Reusable without algorithm change: encoded query tensor Q; raw candidate PID list; centroid-score matrix from retrieval; loaded index doclens/codes/residual files remain resident in IndexScorer. Not a continuation state today: native filter's internal order/score/heap, decompressed embeddings per PID, partial MaxSim accumulators, and current Top100 accumulator (these are not returned by `score_pids`).",
      "","`ndocs` is not consumed by query encoding or raw candidate generation. In `filter_pids.cpp`, the first approximate pass ranks all candidates using centroid scores under the threshold mask and selects top `ndocs`; a second pass over that prefix with all centroids enabled selects top `ndocs//4`. Only then do residual decompression and exact GEMM/segmented MaxSim run. Thus exact-refinement budgets are 100/128/256 for 400/512/1024. `ncells` is a separate candidate-generation control.",
      "","## 2. Saved-data feasibility probe","",
      f"- Cohort: 36 independent (9 known fixed400-hard + 27 easy) and 9 old (3 hard + 6 easy); each has separate recorded restart outputs at 400, 512, 1024. The raw candidate PID hash is identical across these budgets, consistent with one shared retrieval pass.",
      f"- Independent mean standalone restart E2E: 400={stage.loc[400,'mean_e2e']:.2f}ms, 512={stage.loc[512,'mean_e2e']:.2f}ms, 1024={stage.loc[1024,'mean_e2e']:.2f}ms. These are full-call reference timings, not incremental extension costs.",
      f"- The selected repeated-call 400→512→1024 checkpoint proxy averages {seq_mean:.2f}ms vs a fresh 1024 call at {fresh1024:.2f}ms (extra {seq_mean-fresh1024:.2f}ms, +{(seq_mean/fresh1024-1)*100:.1f}%). This quantifies restart overhead only; true extension-only cost is still unmeasured.",
      f"- Saved retained-PID lists happen to be nested at every checkpoint ({'; '.join(nesting)}); this is necessary evidence for prefix extension, but does not prove an incremental implementation will preserve identical ranking/scores.",
      f"- Frozen historical rule (Top75 churn < {CHURN_THRESHOLD:.2f} AND candidate growth < {GROWTH_THRESHOLD:.2f} ⇒ stop at 400) stops {len(stop)}/{len(independent)} independent queries and escalates {len(frozen)}/{len(independent)}. The thresholds were not retuned here. The per-query policy file carries post-hoc hard labels only for evaluation.",
      "- The 400/512/1024 exact Top100 and quality comparisons in `plaid_incremental_query_trace.csv` are restart-reference transitions. The fields `incremental_execution_tested`, `incremental_top100_exact_match_tested`, and `true_incremental_additional_time_measured` deliberately remain false; parity cannot be inferred from separate invocations.",
      "","## 3. Policy latency / quality interpretation","",
      "`plaid_early_stop_policy_analysis.csv` compares Always400, Always1024, the frozen signal rule and the post-hoc Oracle upper bound. Their latency columns are explicitly labeled restart-reference E2E. The current runner reports no true adaptive additional cost, so those numbers must not be presented as a measured incremental policy speedup. Oracle uses fixed400-vs-1024 qrels outcomes only as a hindsight ceiling and is never an online decision rule.",
      "","## 4. Minimal continuation PoC design (next code change)","",
      "No algorithm change has been applied yet. Minimal API design: expose from the native filter the full first-stage sorted `(approx_score,pid)` order using its existing accumulation and pair tie-break; cache second-stage full-centroid approximate scores as the coarse prefix expands; at each budget select exactly `ndocs//4`; decompress and MaxSim-score only selected PIDs lacking a cached scalar score; cache final per-PID scores and merge Top100. Keep legacy `score_pids()` unchanged as the restart oracle.",
      "- Cache first: Q, raw candidate IDs and centroid matrix; then native first-stage approximate order/scores and per-PID second-stage approximate scores; then final per-PID MaxSim scalars. Decompressed vectors need not persist once the scalar score is cached.",
      "- Required parity test: for each query and each stage 400/512/1024, compare incremental Top100 IDs and scores with a clean restart call (exact IDs, scores within explicit FP tolerance, then Recall@100/@10, MRR@10, Success@10). Test tied scores deterministically. Only after parity passes, time `encode + retrieve + stage400 + decision + delta512 + delta1024` with synchronized timing appropriate to device.",
      "- True signal at 400 must use stage-400 state only; qrels, relevant PIDs and any future budget outputs remain evaluation-only. For a valid staged rule, report decision overhead separately and include all previous stage costs in end-to-end latency.",
      "","## Final answers","",
      "1. **Incremental refinement is plausible but not exposed by the current API.** Encoding/raw retrieval state are reusable; native two-stage PID score/order state is opaque and the current call returns only final PIDs and scores.",
      "2. **The offline/restart gap cannot be reduced to one honest measured number yet.** Existing per-budget standalone values contain full score_pids cost; cumulative sums describe repeated full calls, while extension-only cost is unmeasured. The report preserves this distinction rather than inventing savings.",
      "3. Cache Q/raw candidates/centroid matrix; expose native first-stage order and second-stage scores; then cache final per-PID MaxSim scalars. Keeping decompressed embeddings or per-token partials is unnecessary for the minimal design.",
      "4. Potential exists because 400→1024 adds refinement and quality on hard queries, but the current signal rule's useful stopping efficiency and true incremental latency are unproven.",
      "5. **Next priority: incremental refinement on a fixed candidate pool.** This directly tests truncation recovery without repeating search. Adaptive nprobe/ncells is a separate follow-up for candidate-generation misses; do not conflate it with ndocs continuation.",
      "","## Outputs and limitation","",
      "This is an offline feasibility report plus an evidence trace. No index was built and no full benchmark was run. The isolated remote worktree exists at the verified commit, but no source was modified. `incremental_vs_restart_runtime.csv` contains saved restart measurements and leaves incremental delta/component timings blank; `early_stop_runtime_simulation.csv` is a restart-derived, no-qrels signal replay, not an online runtime decision.",""]
    (OUT/"plaid_incremental_pipeline_analysis.md").write_text("\n".join(lines),encoding="utf-8")
    (OUT/"plaid_incremental_search_final_report.md").write_text("\n".join(lines),encoding="utf-8")
    (POC/"plaid_incremental_poc_report.md").write_text("\n".join(lines),encoding="utf-8")
    print("Created incremental feasibility outputs under",OUT)

if __name__=="__main__": main()
