# PLAID vs GIGP+ — same A6000 comparison

- Dataset: LoTTE lifestyle/dev, 50000 documents, 48 queries, TopK=100.
- GPU: NVIDIA RTX A6000 (validated by the remote preflight).
- Existing indexes were reused; no index build is performed by the runner.
- Primary comparison: **retrieval-only** latency. PLAID also records end-to-end latency including query encoding; GIGP+ native timing uses precomputed query embeddings, so end-to-end values are not directly comparable.

## Measured configurations

| config | device | gpu_memory_peak_mib | id | index_size_MB | latency_mean_ms | latency_p50_ms | latency_p95_ms | latency_scope | method | mrr_at_10 | ncells | ndocs | nprobe | peak_gpu_memory_mib | probe_topk | qps | recall_at_10 | recall_at_100 | success_at_10 | threshold | wall_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plaid_fast | GPU | nan | plaid_fast | 310.248111 | 8.61384988658958 | 8.21155309677124 | 11.018473096191885 | retrieval_only | PLAID | 0.7767361111111111 | 1.0 | 400.0 | nan | 614.12548828125 | nan | 116.09210900654816 | 0.6972718253968254 | 0.8322420634920635 | 0.9166666666666666 | 0.5 | nan |
| plaid_low | GPU | nan | plaid_low | 310.248111 | 9.22321332877295 | 9.113064035773276 | 10.507835634052752 | retrieval_only | PLAID | 0.7776041666666667 | 2.0 | 400.0 | nan | 613.919921875 | nan | 108.42208288518891 | 0.707688492063492 | 0.8396825396825397 | 0.9375 | 0.5 | nan |
| plaid_base | GPU | nan | plaid_base | 310.248111 | 9.83156307807399 | 9.765875525772572 | 10.715734772384168 | retrieval_only | PLAID | 0.7779761904761905 | 2.0 | 1024.0 | nan | 613.92138671875 | nan | 101.71322627529749 | 0.7181051587301588 | 0.870436507936508 | 0.9375 | 0.45 | nan |
| plaid_more_docs | GPU | nan | plaid_more_docs | 310.248111 | 11.57193166565978 | 11.60645205527544 | 12.94174175709486 | retrieval_only | PLAID | 0.7779761904761905 | 2.0 | 2048.0 | nan | 613.9248046875 | nan | 86.41599595403284 | 0.723313492063492 | 0.8756448412698412 | 0.9375 | 0.45 | nan |
| plaid_more_cells | GPU | nan | plaid_more_cells | 310.248111 | 10.719367471109662 | 10.715757496654987 | 11.787906382232904 | retrieval_only | PLAID | 0.7769345238095238 | 4.0 | 1024.0 | nan | 613.9541015625 | nan | 93.2890865711203 | 0.7181051587301588 | 0.870436507936508 | 0.9375 | 0.45 | nan |
| plaid_high | GPU | nan | plaid_high | 310.248111 | 12.528989683940178 | 12.509158812463284 | 13.535515684634449 | retrieval_only | PLAID | 0.7769345238095238 | 4.0 | 2048.0 | nan | 613.95751953125 | nan | 79.81489531289286 | 0.723313492063492 | 0.8756448412698412 | 0.9375 | 0.4 | nan |
| gigp_fast | GPU | 2598.0 | gigp_fast | 233.253324 | 2.973858125 | 2.9022550000000003 | 4.0723915 | retrieval_only | GIGP+ | 0.7974537037037037 | nan | nan | 4.0 | nan | 400.0 | 336.2635196324304 | 0.6323412698412699 | 0.7612103174603173 | 0.9166666666666666 | nan | 1978.618936613202 |
| gigp_low | GPU | 2744.0 | gigp_low | 233.253324 | 3.952719027777778 | 3.89407 | 5.0053385 | retrieval_only | GIGP+ | 0.7971643518518517 | nan | nan | 8.0 | nan | 800.0 | 252.9904081146392 | 0.6500496031746033 | 0.7754464285714285 | 0.9166666666666666 | nan | 1929.3416384607556 |
| gigp_base | GPU | 2960.0 | gigp_base | 233.253324 | 5.773246736111111 | 5.606339999999999 | 7.51316 | retrieval_only | GIGP+ | 0.7971643518518517 | nan | nan | 16.0 | nan | 1400.0 | 173.21275977087464 | 0.6569940476190477 | 0.777529761904762 | 0.9166666666666666 | nan | 2205.697638913989 |
| gigp_more_probe | GPU | 3172.0 | gigp_more_probe | 233.253324 | 7.895598055555556 | 7.57791 | 10.771914999999998 | retrieval_only | GIGP+ | 0.7971643518518517 | nan | nan | 16.0 | nan | 2000.0 | 126.65285048247524 | 0.6500496031746033 | 0.7740575396825398 | 0.9166666666666666 | nan | 2282.2377420961857 |
| gigp_high | GPU | 3220.0 | gigp_high | 233.253324 | 8.381220069444446 | 8.156915 | 11.14214 | retrieval_only | GIGP+ | 0.7763310185185185 | nan | nan | 32.0 | nan | 2000.0 | 119.31437090474653 | 0.6250496031746032 | 0.7342757936507938 | 0.8958333333333334 | nan | 2183.8381998240948 |
| gigp_high_probe | GPU | 3894.0 | gigp_high_probe | 233.253324 | 17.725445833333335 | 16.743850000000002 | 23.750294999999998 | retrieval_only | GIGP+ | 0.7784143518518518 | nan | nan | 32.0 | nan | 4000.0 | 56.41607039973371 | 0.6250496031746032 | 0.801140873015873 | 0.9166666666666666 | nan | 2712.27952092886 |
| gigp_max_probe | GPU | 3992.0 | gigp_max_probe | 233.253324 | 18.00122777777778 | 16.874299999999998 | 24.904155 | retrieval_only | GIGP+ | 0.7791087962962963 | nan | nan | 64.0 | nan | 4000.0 | 55.55176637642926 | 0.6194940476190476 | 0.7931547619047619 | 0.9166666666666666 | nan | 2889.7714242339134 |
| gigp_extreme | GPU | 5348.0 | gigp_extreme | 233.253324 | 34.076170138888884 | 31.7684 | 47.198455 | retrieval_only | GIGP+ | 0.7791087962962963 | nan | nan | 64.0 | nan | 8000.0 | 29.34602086807772 | 0.6319940476190476 | 0.8056547619047618 | 0.9166666666666666 | nan | 3685.607278719544 |
| gigp_near_full | GPU | 8250.0 | gigp_near_full | 233.253324 | 63.61869375 | 60.72475 | 83.392535 | retrieval_only | GIGP+ | 0.7992476851851851 | nan | nan | 128.0 | nan | 16000.0 | 15.718650306302465 | 0.6569940476190477 | 0.8625992063492064 | 0.9375 | nan | 5169.4627944380045 |
| gigp_fullish | GPU | 14066.0 | gigp_fullish | 233.253324 | 116.8667652777778 | 115.557 | 152.62154999999998 | retrieval_only | GIGP+ | 0.7992476851851851 | nan | nan | 256.0 | nan | 32000.0 | 8.556752620157871 | 0.6569940476190477 | 0.8593253968253969 | 0.9375 | nan | 8517.48364046216 |

## Recall-matched measured pairs

| target_recall | PLAID_config | GIGP_config | PLAID_recall_at_100 | GIGP_recall_at_100 | recall_abs_diff | PLAID_latency_ms | GIGP_latency_ms | PLAID_p50_ms | GIGP_p50_ms | PLAID_p95_ms | GIGP_p95_ms | PLAID_qps | GIGP_qps | GIGP_speedup_vs_PLAID | within_0.005 | within_0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.85 | plaid_low | gigp_fullish | 0.8396825396825397 | 0.8593253968253969 | 0.019642857142857184 | 9.22321332877295 | 116.8667652777778 | 9.113064035773276 | 115.557 | 10.507835634052752 | 152.62154999999998 | 108.42208288518891 | 8.556752620157871 | 0.07892075481725293 | False | False |
| 0.87 | plaid_base | gigp_near_full | 0.870436507936508 | 0.8625992063492064 | 0.00783730158730156 | 9.83156307807399 | 63.61869375 | 9.765875525772572 | 60.72475 | 10.715734772384168 | 83.392535 | 101.71322627529749 | 15.718650306302465 | 0.1545389019885997 | False | True |
| 0.9 | plaid_more_docs | gigp_near_full | 0.8756448412698412 | 0.8625992063492064 | 0.013045634920634819 | 11.57193166565978 | 63.61869375 | 11.60645205527544 | 60.72475 | 12.94174175709486 | 83.392535 | 86.41599595403284 | 15.718650306302465 | 0.18189514722093428 | False | False |
| 0.92 | plaid_more_docs | gigp_near_full | 0.8756448412698412 | 0.8625992063492064 | 0.013045634920634819 | 11.57193166565978 | 63.61869375 | 11.60645205527544 | 60.72475 | 12.94174175709486 | 83.392535 | 86.41599595403284 | 15.718650306302465 | 0.18189514722093428 | False | False |

## Interpretation

Only rows marked `within_0.005` satisfy the strict Recall@100 matching rule. Speedup is `PLAID retrieval-only mean / GIGP+ retrieval-only mean`; it is not a cross-GPU or interpolated estimate. The final conclusion should use only measured rows, not a smoothed curve.
