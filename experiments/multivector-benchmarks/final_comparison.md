# 多向量检索最终实验对比

## 实验设置

正式结果均以 LoTTE lifestyle/dev 为目标数据集。已完成的正式实验使用 50,000 个 passages、48 个 queries、TopK=100、3 次重复。PLAID/WARP GPU 的 warm-up 为 5 个查询，CPU 和 GIGP+ GPU 的 warm-up 为 3 个查询。

`Build_time` 统一换算为秒，`Index_size_MB` 使用十进制 MB，`GPU_memory` 使用 MiB。CPU 实验复用了预建索引，因此没有把 GPU 建索引时间当作 CPU build time。

## 最终结果

| 方法 | 设备 | Mean / p50 / p95 (ms) | QPS | Success@10 | MRR@10 | Recall@10 | Build (s) | Index (MB) | GPU peak (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PLAID | Tesla T4 | 28.670 / 28.096 / 31.690 | 34.880 | 0.9167 | 0.7648 | 0.7108 | 744.890 | 311.921 | 1498.1 |
| PLAID | CPU | 364.717 / 290.967 / 576.915 | 2.742 | 0.9167 | 0.7648 | 0.7108 | missing | 311.921 | n.a. |
| WARP | Tesla T4 | 83.401 / 72.480 / 145.809 | 11.990 | 0.9167 | 0.7478 | 0.6694 | 1630.373 | 1346.386 | missing |
| WARP | CPU | 673.524 / 617.788 / 1076.633 | 1.485 | 0.9167 | 0.7500 | 0.6590 | missing | 1346.386 | n.a. |
| GIGP+ | RTX A6000 | 5.920 / 5.819 / 7.280 | 168.913 | 0.9167 | 0.7972 | 0.6570 | 38.321 | 233.253 | 1156 |
| GIGP+ | CPU | missing | missing | missing | missing | missing | missing | missing | n.a. |

## 查询性能

在各自记录到的运行配置上，GIGP+ GPU 的 mean latency 最低，PLAID GPU 次之，WARP GPU 最高。CPU 结果中，PLAID 的 mean latency 低于 WARP。这里不能据此写成严格的算法 speedup：GIGP+ 使用 RTX A6000，PLAID/WARP 使用 Tesla T4，而且 GIGP+ 统计的是 native steady-state search，不含模型编码、进程启动和索引加载；PLAID/WARP 的 GPU latency 包含 query encoding。

WARP 的 p95 与 mean 距离较大，说明这组记录的尾部延迟比 PLAID 和 GIGP+ 更明显。该判断仅描述本次观测数据，不外推到其他硬件或参数。

## 检索质量

所有已完成配置的 Success@10 都是 0.9167。GIGP+ GPU 的 MRR@10 为 0.7972，是已记录配置中的最高值；其 Recall@10 为 0.6570。PLAID 的 Recall@10 为 0.7108，是已记录配置中的最高值。WARP GPU/CPU 的质量指标接近，但并非完全一致，因此最终汇报应保留设备和实现配置标签。

## 索引与资源

已记录的索引大小为：GIGP+ 233.253 MB、PLAID 311.921 MB、WARP 1346.386 MB。已记录 build time 为：GIGP+ 38.321 秒、PLAID 744.890 秒、WARP 1630.373 秒。这些构建过程使用的硬件、代码路径和计时边界不同，适合汇报原始记录，不适合解释为受控构建速度排名。

PLAID GPU 峰值显存为约 1498.1 MiB，GIGP+ 为 1156 MiB。WARP GPU 峰值显存没有在现有正式结果中记录。

## PLAID stage latency

找到了源码级 stage latency 结果。48 queries × 3 repetitions 共 144 条记录：prune 7.670 ms、decompress 4.618 ms、MaxSim/GEMM 1.363 ms、sorting 0.159 ms；stage run 的完整端到端 mean 为 29.313 ms。sorting 位于 `score_pids()` 返回之后，不属于 `score_pids_total_ms`。

## 数据来源

- PLAID GPU：`outputs/colbert_benchmark_results_repacked.zip` 中的 `summary.csv`、`index_stats.csv` 和 `metadata.json`。
- WARP GPU：`warp_benchmark_v2_results.zip` 中的 `summary.csv`、`repetition_results.csv` 和 `metadata.json`。
- PLAID/WARP CPU：`plaid_warp_cpu_benchmark_results_fixed (5).zip` 中的 `summary.csv`、per-query CSV 和 `metadata.json`。
- PLAID stage latency：`plaid_stage_latency_results.zip`。
- GIGP+ GPU：服务器 `gigp_final_50k_outputs` 目录以及本地 `gigp_gpu_50k_results.tar.gz`。

服务器指定的三个目录中找到了完整的 GIGP+ 50k 正式结果和早期 smoke/stability 结果，但没有找到 PLAID/WARP 的正式 GPU/CPU benchmark 包。因此 PLAID/WARP 数字来自此前从 Colab 下载并保存在本地的原始结果包。

## 完整性与后续

PLAID GPU/CPU、WARP GPU/CPU、GIGP+ GPU 的 latency、QPS 和质量指标完整。缺失项为 WARP GPU peak memory、PLAID/WARP CPU build time，以及正式 GIGP+ CPU 50k 结果。当前三种方法的 GPU 图表可以用于阶段性汇报，但必须保留硬件和 latency scope 限制。若导师要求严格同硬件算法比较，仍需在同一 GPU 和统一 latency scope 下补测；否则不需要为了整理现有结论重新运行实验。
