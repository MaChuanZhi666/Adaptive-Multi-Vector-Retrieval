# Final comparison v2 notes

- 正式 comparison 表含 PLAID-GPU、PLAID-CPU、WARP-GPU、WARP-CPU、GIGP+-GPU、IGP-CPU 六组结果；统一为 LoTTE lifestyle/dev、50,000 documents、48 queries、TopK=100。
- GIGP+ 正式 benchmark 表使用 5.9202 ms/query；stage 图使用独立 profiling run 的 5.4530 ms/query，两者计时口径不同，未互相覆盖。
- PLAID/WARP GPU 使用 Tesla T4，GIGP+ GPU 使用 RTX A6000。图中保留硬件标注，不计算跨 GPU speedup。
- IGP-CPU 是官方 `IGP.DocRetrieval` CPU backend，不命名为 GIGP+ CPU。它复用量化/残差等索引工件，但未直接读取 GPU 专用 `ivf.index`；CPU build time 与 backend-specific index footprint 未测量，CSV 中保持空白。
- PLAID 与 GIGP+ stage 图为 implementation-specific stage breakdown。probe、candidate/filter 等边界存在内部耦合，不解释为跨实现严格等价。
- WARP 内部 stage 标为 `native coupled / not instrumented`。这表示原生实现边界未暴露，不是实验失败或数据缺失。
- 未纳入 lotte-500-gnd、expanded-1000、expanded-5000 等 smoke/stability 数据。
