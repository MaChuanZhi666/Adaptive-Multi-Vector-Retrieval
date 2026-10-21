# 硬件与计时口径说明

GIGP+ GPU 正式实验使用 NVIDIA RTX A6000 48 GB；PLAID 和 WARP GPU 正式实验使用 Tesla T4。两组 GPU 结果不是同硬件条件下的公平比较，因此图表只展示已记录的 latency、QPS 和质量指标，不计算或宣称跨方法 speedup。

计时范围也不同：

- PLAID GPU：端到端查询阶段，包含 query encoding。
- WARP GPU：query encoding、retrieval 和 ranked TopK materialization。
- GIGP+ GPU：native steady-state search，不包含模型编码、进程启动和索引加载。
- PLAID/WARP CPU：端到端 query encoding 与 retrieval，使用预建索引。

因此，当前结果适合说明“各方法在各自已测配置下的观测表现”。若需要严格的算法结论，应在同一 GPU、相同软件环境、相同 warm-up 和完全一致的 latency boundary 下重新测量。

资源字段方面，PLAID GPU 峰值显存约 1498.1 MiB，GIGP+ 为 1156 MiB；现有 WARP GPU 正式结果未记录峰值显存。CPU 结果未单独测量 build time，因为使用了预建 GPU 索引。
