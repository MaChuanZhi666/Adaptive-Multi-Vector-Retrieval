# PLAID、GIGP+、WARP 同卡结果

- 数据集：LoTTE lifestyle/dev，50,000 documents，48 queries，TopK=100。
- GPU：NVIDIA RTX A6000。
- WARP 正式 query 记录：144 条；index build：535.453 秒。
- PLAID/GIGP+ 复用已有索引；其本次 build time 未测。
- 展示各方法本次实测最高 Recall@100 点，不进行插值。
- WARP latency 包含 query encoding + retrieval + 结果物化；PLAID/GIGP+ 报 retrieval-only。WARP 平均 query encoding 为 23.963 ms，平均 retrieval 为 31.235 ms。延迟/QPS图只描述各自观测口径，不是严格 speedup。

| Method | Config | Mean ms | P50 ms | P95 ms | QPS | Success@10 | MRR@10 | Recall@10 | Recall@100 | Index MB | GPU peak MiB | Build s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PLAID | plaid_more_docs | 11.572 | 11.606 | 12.942 | 86.42 | 0.9375 | 0.7780 | 0.7233 | 0.8756 | 310.25 | 613.9 | not measured |
| GIGP+ | gigp_near_full | 63.619 | 60.725 | 83.393 | 15.72 | 0.9375 | 0.7992 | 0.6570 | 0.8626 | 233.25 | 8250.0 | not measured |
| WARP-XTR | WARP default | 55.201 | 51.771 | 73.784 | 18.12 | 0.9167 | 0.7505 | 0.6736 | 0.8762 | 1361.81 | 13100.0 | 535.453 |


WARP Recall@100 与 PLAID 最高实测点相差 0.0005，与 GIGP+ 最高实测点相差 0.0136。同为 A6000、同一数据集与 query 数，但计时边界不同。PLAID/GIGP+ 本次 build time 标记为 not measured，不用 0 替代。
