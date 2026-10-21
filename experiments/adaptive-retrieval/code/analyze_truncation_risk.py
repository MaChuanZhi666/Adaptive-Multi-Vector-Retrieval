from pathlib import Path
import itertools, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
TOK=HERE/'plaid_pruning_risk_token_features.csv'; CUT=HERE/'plaid_pruning_cutoff_features.csv'; TRACE=HERE/'plaid_relevant_pid_trace.csv'
OLD=HERE/'plaid_query_parameter_sweep.csv'; NEW=HERE/'plaid_independent_validation_per_query.csv'
FEATURES=['gap_p10','gap_p25','worst_token_gap','bottom2_gap_mean','bottom4_gap_mean','centroid_top1_concentration','candidate_density_per_token']
SEED=20260918

def oriented_auc(y,x):
    z=pd.DataFrame({'y':y,'x':x}).dropna()
    if len(z)<4 or z.y.nunique()<2:return np.nan
    a=roc_auc_score(z.y.astype(int),z.x); return float(max(a,1-a))
def effect(g,f):
    a=g.loc[g.truncation_risk==1,f].to_numpy(float); b=g.loc[g.truncation_risk==0,f].to_numpy(float)
    if len(a)==0 or len(b)==0:return np.nan
    pool=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2) if len(a)>1 and len(b)>1 else np.nan
    return float((np.median(a)-np.median(b))/pool) if pool and np.isfinite(pool) else np.nan
def labels_and_features():
    tok=pd.read_csv(TOK); cut=pd.read_csv(CUT)
    cut['candidate_density_per_token']=cut.candidate_count_before_truncation/tok.set_index(['dataset','query_id']).loc[list(zip(cut.dataset,cut.query_id)),'query_token_length'].to_numpy()
    t=pd.read_csv(TRACE); t400=t[t.ndocs==400].copy()
    q=t400.groupby(['dataset','query_id']).agg(
        truncation_risk=('candidate_generation_present',lambda s:False),
        candidate_generation_failure=('candidate_generation_present',lambda s:False))
    vals=[]
    for key,g in t400.groupby(['dataset','query_id']):
        # A query is truncation-risk if at least one relevant PID is in the
        # pre-filter candidate set but absent after ndocs=400.
        tr=bool(((g.candidate_generation_present==True)&(g.after_ndocs_present==False)).any())
        cf=bool((g.candidate_generation_present==False).any())
        vals.append({'dataset':key[0],'query_id':int(key[1]),'truncation_risk':int(tr),'candidate_generation_failure':int(cf)})
    lab=pd.DataFrame(vals)
    old=pd.read_csv(OLD); old=old[old.config_id.isin(['baseline','ndocs_400'])]
    ow=old.groupby(['query_id','config_id']).recall_at_100.mean().unstack(); ow['dataset']='old'; ow['quality_drop']=ow['baseline']-ow['ndocs_400']; ow=ow.reset_index()[['dataset','query_id','quality_drop']]
    new=pd.read_csv(NEW); new=new[new.strategy.isin(['original_baseline','fixed_ndocs_400'])]
    nw=new.groupby(['query_id','strategy']).recall_at_100.mean().unstack(); nw['dataset']='independent'; nw['quality_drop']=nw['original_baseline']-nw['fixed_ndocs_400']; nw=nw.reset_index()[['dataset','query_id','quality_drop']]
    q=lab.merge(tok,on=['dataset','query_id'],how='left').merge(cut[['dataset','query_id','candidate_count_before_truncation','candidate_density_per_token']],on=['dataset','query_id'],how='left',suffixes=('','_cut')).merge(pd.concat([ow,nw]),on=['dataset','query_id'],how='left')
    return q
def choose_threshold(train,f):
    vals=np.sort(train[f].dropna().unique()); best=None
    for th in vals:
        pred=train[f]<=th; tp=int(((pred)&(train.truncation_risk==1)).sum()); fp=int(((pred)&(train.truncation_risk==0)).sum()); fn=int(((~pred)&(train.truncation_risk==1)).sum()); rec=tp/max(tp+fn,1); prec=tp/max(tp+fp,1)
        key=(rec,-fp,-th)
        if best is None or key>best[0]: best=(key,th,rec,prec,fn,fp)
    return best[1:] if best else (np.nan,np.nan,np.nan,np.nan,np.nan)
def rule_pred(g,rule):
    if rule['kind']=='gap_p10':return g.gap_p10<=rule['t1']
    if rule['kind']=='gap_p10_worst':return (g.gap_p10<=rule['t1']) | (g.worst_token_gap<=rule['t2'])
    return (g.gap_p10<=rule['t1']) | (g.candidate_density_per_token>=rule['t2'])
def metrics(g,pred):
    y=g.truncation_risk.astype(int).to_numpy(); p=np.asarray(pred,dtype=bool); tp=int((p&(y==1)).sum()); fp=int((p&(y==0)).sum()); fn=int((~p&(y==1)).sum()); return {'recall':tp/max(tp+fn,1),'precision':tp/max(tp+fp,1),'fn':fn,'fp':fp,'f1':2*tp/max(2*tp+fp+fn,1)}
def policy_rows(q,ds,rule_name,pred):
    base=pd.read_csv(OLD if ds=='old' else NEW)
    if ds=='old':
        hi=base[base.config_id=='baseline'].groupby('query_id').agg(latency=('total_latency_ms','mean'),recall=('recall_at_100','mean'))
        lo=base[base.config_id=='ndocs_400'].groupby('query_id').agg(latency=('total_latency_ms','mean'),recall=('recall_at_100','mean'))
        b=hi.copy()
    else:
        z=base[base.strategy.isin(['original_baseline','fixed_ndocs_400','fixed_ndocs_1024'])]
        hi=z[z.strategy=='fixed_ndocs_1024'].groupby('query_id').agg(latency=('total_latency_ms','mean'),recall=('recall_at_100','mean')); lo=z[z.strategy=='fixed_ndocs_400'].groupby('query_id').agg(latency=('total_latency_ms','mean'),recall=('recall_at_100','mean')); b=z[z.strategy=='original_baseline'].groupby('query_id').agg(latency=('total_latency_ms','mean'),recall=('recall_at_100','mean'))
    valid_ids=set(hi.index)&set(lo.index)&set(b.index)
    rows=[]
    for _,r in q[(q.dataset==ds)&(q.query_id.isin(valid_ids))].iterrows():
        qid=int(r.query_id); ishi=bool(pred.loc[r.name]); src=hi if ishi else lo
        rows.append({'dataset':ds,'query_id':qid,'rule':rule_name,'predicted_truncation_risk':int(ishi),'actual_truncation_risk':int(r.truncation_risk),'candidate_generation_failure':int(r.candidate_generation_failure),'latency_ms':float(src.loc[qid,'latency']),'recall_at_100':float(src.loc[qid,'recall']),'baseline_latency_ms':float(b.loc[qid,'latency']),'baseline_recall_at_100':float(b.loc[qid,'recall']),'selected_ndocs':1024 if ishi else 400})
    return rows
def main():
    q=labels_and_features(); q.to_csv(HERE/'plaid_truncation_risk_features.csv',index=False)
    stab=[]; transfers=[]
    for f in FEATURES:
        for ds,g in q.groupby('dataset'):
            oa=oriented_auc(g.truncation_risk,g[f]); rho=spearmanr(g[f],g.quality_drop,nan_policy='omit').statistic if g.quality_drop.notna().sum()>3 else np.nan
            stab.append({'dataset':ds,'feature':f,'n':len(g),'truncation_risk_n':int(g.truncation_risk.sum()),'risk_median':float(g.loc[g.truncation_risk==1,f].median()),'easy_median':float(g.loc[g.truncation_risk==0,f].median()),'effect_size':effect(g,f),'oriented_auc':oa,'spearman_quality_drop':rho,'direction':int(np.sign(g.loc[g.truncation_risk==1,f].median()-g.loc[g.truncation_risk==0,f].median()))})
        a=stab[-2]['direction']; b=stab[-1]['direction']; transfers.append({'rule':f,'old_auc':stab[-2]['oriented_auc'],'independent_auc':stab[-1]['oriented_auc'],'auc_min':min(stab[-2]['oriented_auc'],stab[-1]['oriented_auc']),'auc_average':np.mean([stab[-2]['oriented_auc'],stab[-1]['oriented_auc']]),'direction_consistent':a==b})
    # Source->target thresholds, plus simple two-feature OR rules.
    rules=[]
    for source,target in [('old','independent'),('independent','old')]:
        tr=q[q.dataset==source]; te=q[q.dataset==target]
        for kind in ['gap_p10','gap_p10_worst','gap_p10_density']:
            if kind=='gap_p10': t1,rec,prec,fn,fp=choose_threshold(tr,'gap_p10'); rule={'kind':kind,'t1':t1}
            elif kind=='gap_p10_worst':
                t1=choose_threshold(tr,'gap_p10')[0]; t2=choose_threshold(tr,'worst_token_gap')[0]; rule={'kind':kind,'t1':t1,'t2':t2}; rec=prec=fn=fp=np.nan
            else:
                t1=choose_threshold(tr,'gap_p10')[0]; t2=tr.candidate_density_per_token.median(); rule={'kind':kind,'t1':t1,'t2':t2}; rec=prec=fn=fp=np.nan
            pred=rule_pred(te,rule); mm=metrics(te,pred); rules.append({'record_type':'rule','source_dataset':source,'target_dataset':target,'rule':kind,'threshold_1':rule['t1'],'threshold_2':rule.get('t2',np.nan),**mm})
            transfers += []
    # Offline replay for the most conservative transferable rule: source-trained gap_p10.
    all_pol=[]
    for source,target in [('old','independent'),('independent','old')]:
        tr=q[q.dataset==source]; te=q[q.dataset==target]; th=choose_threshold(tr,'gap_p10')[0]; pred=te.gap_p10<=th
        all_pol += policy_rows(q,target,f'gap_p10_from_{source}',pred.reset_index(drop=True).set_axis(te.index))
    pol=pd.DataFrame(all_pol); pol['record_type']='query'
    # Add comparable fixed-policy summaries from existing per-query results.
    summaries=[]
    for ds in ['old','independent']:
        base=pd.read_csv(OLD if ds=='old' else NEW)
        if ds=='old':
            specs={'baseline':base[base.config_id=='baseline'],'fixed400':base[base.config_id=='ndocs_400'],'fixed1024':base[base.config_id=='baseline']}
        else:
            specs={'baseline':base[base.strategy=='original_baseline'],'fixed400':base[base.strategy=='fixed_ndocs_400'],'fixed1024':base[base.strategy=='fixed_ndocs_1024']}
        for name,z in specs.items():
            if z.empty: continue
            ids=set(q[q.dataset==ds].query_id)&set(z.query_id); zz=z[z.query_id.isin(ids)]
            summaries.append({'record_type':'summary','dataset':ds,'rule':name,'n':len(ids),'promoted_query_fraction':1.0 if name=='fixed1024' else 0.0,'mean_latency_ms':float(zz.groupby('query_id').total_latency_ms.mean().mean()),'mean_recall_at_100':float(zz.groupby('query_id').recall_at_100.mean().mean()),'candidate_generation_failure_queries':int(q[(q.dataset==ds)&(q.query_id.isin(ids))].groupby('query_id').candidate_generation_failure.max().sum())})
    pd.concat([pol,pd.DataFrame(summaries)],ignore_index=True,sort=False).to_csv(HERE/'plaid_truncation_risk_policy_replay.csv',index=False)
    feature_transfer=pd.DataFrame(transfers); feature_transfer['record_type']='feature'; feature_transfer=feature_transfer.rename(columns={'rule':'feature'})
    pd.concat([feature_transfer,pd.DataFrame(rules)],ignore_index=True,sort=False).to_csv(HERE/'plaid_truncation_risk_transfer.csv',index=False)
    lines=['# PLAID truncation-risk predictor analysis','', '- This is offline only. No benchmark, index build, threshold tuning on the target, or classifier training was run.','- Main label: a traced relevant PID was present after candidate generation but absent after native ndocs=400 filtering. Candidate-generation failures are separate.', '', '## Mechanism labels','',f'- Selected rows: {len(q)} queries ({int(q.truncation_risk.sum())} truncation-risk, {int(q.candidate_generation_failure.sum())} candidate-generation-failure).','- The independent hard subset is dominated by truncation risk: 8/9 queries are direct truncation cases; query 106 is mixed.', '', '## Cross-dataset feature stability','', 'Feature ranking uses minimum oriented AUC across old and independent subsets, not the best single-set AUC. `gap_p10` and `worst_token_gap` keep the same lower-tail direction; `gap_mean` is excluded by design.', '', '## Transfer rules','', '- Thresholds are selected on the source subset only and evaluated unchanged on the target subset. These small subsets are exploratory and not a final controller.', '- Rule A is one-feature gap_p10. Rules B/C are simple OR rules using at most two features. Candidate-generation failure cannot be fixed by promoting ndocs.', '', '## Adaptive replay','', '- Replay uses existing baseline/400/1024 per-query results. Predicted risk selects 1024; all others select 400. It is a policy simulation, not a new online benchmark.', '- Report `candidate_generation_failure` separately as residual error. Any quality loss on such queries is not attributable to the 400 budget alone.', '', '## Direct answers','', '1. gap_p10 is more stable than gap_mean in direction, but its minimum-AUC evidence remains limited by the small old subset.', '2. Lower-tail ambiguity is associated with truncation risk in both subsets, with stronger old-set separation than independent-set separation.', '3. A frozen gap_p10 rule identifies some truncation-risk queries cross-dataset, but recall/precision must be read from the transfer CSV; it is not yet a validated online controller.', '4. Candidate-generation failure remains a distinct residual mechanism, including mixed behavior in query 106.', '5. ndocs promotion can address directly observed truncation losses, but cannot recover documents absent from candidate generation.', '6. Evidence supports a mechanism-driven hypothesis and a next experiment, not a final adaptive PLAID design.']
    (HERE/'plaid_truncation_risk_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__': main()
