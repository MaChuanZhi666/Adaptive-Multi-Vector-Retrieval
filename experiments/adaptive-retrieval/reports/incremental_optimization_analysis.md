# PLAID incremental PoC：性能瓶颈与优化分析

范围：只检查当前隔离 PoC 源码和已有 45-query × 3-repeat GPU runtime 产物；没有改算法代码、没有重跑 benchmark、没有重建 index。下列计时来自 RTX A6000 / PyTorch 1.12.1。对 full1024，现有计时器只把 `score_pids + 最终排序` 记成一个整体，不能伪造 filter、decompression、MaxSim、TopK 的独立耗时。

## 结论先行

增量路径已正确，但当前实现尚未变快：单次 Full1024 平均 **28.20 ms**，增量 400→512→1024 为 **42.13 ms**，慢 **13.94 ms**。二者的 query encoding 与 candidate generation 几乎相同（Full1024：20.145 + 0.985 ms；incremental 共享一次：19.941 + 0.977 ms），差距主要在三个增量阶段的 scoring/refinement 调用累计：**21.216 ms vs Full1024 整段 7.067 ms**。

这不是 centroid scoring 或同一 PID 的精确 MaxSim 被重复计算造成的。主要嫌疑是小批次分段执行的固定开销、各阶段反复做 selection / tensor-list 转换和重组、多个强制 CUDA 同步，以及每阶段重新排序结果。PoC 的阶段计时还没有把所有 Python/state/top-k merge 工作单列出来，因此需把这些视为由源码与总量差推断的候选瓶颈，而非逐项 profiler 结论。

## Pipeline 对照

| 阶段 | Full1024（一次） | Incremental 400→512→1024 | 重算/当前状态 |
|---|---|---|---|
| Query encoding | 20.145 ms | 19.941 ms，一次 | 未重复；restart 情景才每个 budget 重做 |
| Candidate generation | 0.985 ms | 0.977 ms，一次 | centroid × query-vector 得分、选 centroid、IVF lookup、PID 排序/去重只调用一次 |
| 第一阶段 thresholded approximate score | 藏在 `score_pids` 的 7.067 ms 整体里 | 400 时 2.584 ms；512/1024 时 0 | 每个 raw candidate 计算一次，得分向量缓存；没有按阶段重算分数 |
| 第一阶段 top-ndocs selection | 藏在 Full `score_pids` 总计时中 | 每阶段约 0.073 ms，累计 0.219 ms | 对完整缓存分数向量分别 top-k 400、512、1024；selection 重做 3 次 |
| 第二阶段 full-centroid approximate score | 同样在 Full 整体计时中 | 新 PID 计时：3.766 + 1.347 + 2.553 = **7.665 ms** | 对每个 PID 只计算一次并按 PID 缓存；但每次阶段都把当前前缀的 Python cache 重组为 tensor，再做 top-k |
| 第二阶段 top-ndocs/4 selection | Full 整体计时中 | 每阶段约 0.076/0.074/0.083 ms，累计 0.233 ms | 每阶段重做 top-k，k 为 100/128/256；这是预算变化所需的排序选择，但可优化实现 |
| Residual lookup/decompression | Full GPU `lookup_pids` 内部，没有独立计时 | 新增 refine PID 计时合计 **3.330 ms** | 400 首批 100 个，512 新增最多 28 个，1024 再新增最多 128 个；已有 PID 不重复 decode |
| MaxSim/GEMM | Full `colbert_score_packed` 内部，没有独立计时 | 新增 PID MaxSim 合计 **3.564 ms** | 每个 PID 的完整 scalar score 只算一次并缓存；不保存 token 级 partial MaxSim |
| 最终 TopK / merge | Full score 后 `scores.sort(descending=True)` | 每阶段对当前 refine scores 再 `sort`；没有持久 Top100 merge | 相同旧分数在每阶段重新 stack、排序；返回 400/512/1024 checkpoint 时确实要产出当阶段 Top100 |

Full1024 的 7.067 ms 是以下所有工作的一体化时间：两个 approximate filter、两次预算 selection、decompression、MaxSim、排序，以及函数开销。不得拿它直接与某个增量子项比较。增量的已分项累计为约 17.60 ms；三个阶段 `score_filter_refine_sort` 总计 21.216 ms，尚有约 **3.62 ms** 没被现有 component timer 归类（如 clone/state 管理、Python PID list/dict、score tensor stack、Top100 sort、额外同步/launch 等的组合）。

## 哪些计算实际重复

### 没有发现按 query 或 PID 重复的核心打分

- **Centroid scoring：** `CandidateGeneration.get_cells()` 做 centroid/query 矩阵乘在共享 `retrieve()` 中一次完成；incremental 调用复用 `centroid_scores`，不是每级重算。
- **第一阶段 approximate PID scores：** 首次 400 扫描 raw candidate PIDs 得到 `stage1_approx_scores`，后两次使用缓存。
- **第二阶段 approximate PID scores：** `stage2_approx_by_pid` 只为新进入 stage1 前缀的 PID 计算。已计算 PID 的 approximate score 不重新算，但每阶段对 cache 值再构造整段 `stage2_scores`。
- **Residual decompression 与精确 MaxSim：** `maxsim_by_pid` 缓存每个已 refine PID 的最终 scalar；只对新 PID 调 `lookup_pids` 和 `colbert_score_packed`。解压后的向量不保留，避免占用大块 GPU 内存；由于 PID 不重算，当前并无因丢弃向量而产生的重复解压。
- **没有 MaxSim partial：** 每个新 PID 一次完成 query-doc MaxSim 并只保存最后 scalar。没有缓存 token×doc-token 相似度矩阵或 query-token maxima 的分阶段部分值；但在此 PoC 的预算扩展里，scalar 足够，MaxSim 本身无需“续算”。

### 确实重复的阶段管理与排序工作

- 第一阶段对同一个 `stage1_approx_scores` 分别运行 400/512/1024 的 `torch.topk`。
- 每个预算对当前 stage1 前缀创建 PID Python list，查询 Python dict，stack 所有已知第二阶段分数，再 top-k。虽然 per-PID dot/reduce 不重算，tensor 前缀的重组和 full-prefix selection 会重做。
- 每阶段对当前 refine PID 列表构造 `refine_list` 并 stack 所有 exact score；随后 runtime 再把当前完整 refine list `scores.sort()` 以拿 Top100。旧 PID 的 MaxSim 分数没有重算，但旧分数参与了新的 stack/sort。
- 每个阶段内部有多次 `torch.cuda.synchronize()`，包含 selection 前后、approx scoring 前后、MaxSim 前后。强同步有利于诊断计时，却阻止异步流水与 kernel 合并，放大 28/128/100/128 这样的分批工作固定成本。

## 量化性能差距

| 路径 | 平均 E2E | score/filter/refine/sort | Recall@100 |
|---|---:|---:|---:|
| Full1024 | 28.199 ms | 7.067 ms（未拆分） | 0.9006 |
| Incremental 400→512→1024 | 42.134 ms | 21.216 ms（三次阶段调用累计） | 0.9006 |

增量虽共享 encoder 和 retrieval，但其 scoring 部分比 Full1024 多约 14.15 ms；编码+候选生成差异约 -0.21 ms，因此最终 E2E 慢约 13.94 ms。parity 在 405 个增量 query-stage/repetition 样本上全部通过：100/100 Top100 overlap、排序一致、最大 score 差为 0。

作为背景，若真的独立从头运行 400、512、1024 三次，总耗时 83.21 ms；incremental 42.13 ms 节省约 49.4%。这只说明比“重复重启三次”便宜，不代表比一次 Full1024 便宜。

## 修改建议（优先级从高到低）

1. **把强制同步移出热路径。** 搜索阶段内部不逐算子 `cuda.synchronize()`；用 CUDA events 或仅在外部阶段边界同步计时。先单独 profile，确认同步与小 kernel 的真实影响。预期首先减少 CPU-GPU 等待/launch 空隙；量级无法由现有 CSV 精确预测。
2. **用 GPU tensor 状态替代 PID Python list/dict。** 保存 PID→cache-slot 映射（dense array/hash map）以及 contiguous score/cache tensor，避免每级 `.cpu().tolist()`、Python membership、逐 scalar 插 dict、再 `torch.stack`。一次缓存后按索引取前缀或新增项。预期减少每级 CPU 开销和内存分配，当前未独立计时，暂不给虚假的毫秒承诺。
3. **只在新分数上维护 Top100。** refine 集为 nested 时，维护上阶段 Top100；下一阶段只合并“上阶段 Top100 + 新增 PID 的 exact scores”，对最多 100+28 或 100+128 个值做 GPU `topk(100)`。这避免重 stack 全部 128/256 个旧+新分数和全量 sort。必须保留与 restart 的逐 query parity gate，并验证边界 tie-break 与嵌套性。
4. **减少重复的预算 selection/前缀装配。** 对第一阶段 approximate scores 保留稳定降序索引；第二阶段只在必要时取已缓存 contiguous scores。若预算按递增固定，仅在新的 400/512/1024 决策点更新选择，避免将已知 PID cache 逐个回装 tensor。当前 `topk` 本身只有约 0.07 ms/次，不是唯一瓶颈，收益可能小于同步和 Python cache 改造。
5. **对“预期继续到最大预算”的请求提供 fast path。** 若在线规则在 400 阶段不需要精确结果（或只需 cheap approximate signal），先决定是否升级，再对最终 refine PID 一次性解压/MaxSim。这样可回到接近 Full1024 的批处理效率；代价是 400/512 中间 checkpoint 不再是已完成 exact Top100。不能同时免费获得每阶段精确 Top100 和单次 1024 的批处理成本。
6. **只在复测证明有重复解压后再缓存 D_packed。** 当前 exact scalar cache 已避免重复解压/MaxSim；缓存解压向量大幅增加 GPU memory，预期收益低。更合适的做法是分配一次大 scratch buffer，把每阶段真正新增 PID 批量写入，减少 allocator 开销。

## 预计收益与建议验证

- 理论性能下限：不要求中间 exact 输出时，最大预算的一次性 filter→decompress→MaxSim→TopK 可接近 Full1024 的 **28.2 ms**；若保留 400/512 exact checkpoint，至少还要支付各阶段 selection、同步和多批次执行成本，不能预设能达到单次 Full1024。
- 当前 incremental 相对 Full1024 的实测额外耗时为 **13.94 ms**。先做同步/计时清理与 tensor cache / incremental Top100 merge 的局部 profile；在未重测前，预计节省只能标作待验证，不能承诺可收回全部 13.94 ms。
- 下一轮小样本需同时报告 CUDA events 和 wall-clock；按 query 输出每阶段新增 PID 数、cache hit 数、filter/lookup、decompression、MaxSim、merge/top-k、Python 管理开销。Full1024 也需用同样边界埋点拆分，才能回答每个阶段的真实 speedup。
- 比较对象至少保留：Full1024、增量到 400 后停、增量到 512 后停、增量到 1024、以及只有当机制需要时的“共享 encode/retrieve + 直接 1024 refine” fast path。仍只做小样本，不跑全量 benchmark。

## 源码与数据

- 增量状态和缓存：`source/baseline/ColBERT/colbert/search/index_storage.py`，`score_pids_incremental()`。
- Full filter/refine 路径：同文件 `score_pids()`；预算两级筛选后 `lookup_pids()` 与 `colbert_score_packed()`。
- Centroid scoring / IVF PID 去重：`source/baseline/ColBERT/colbert/search/candidate_generation.py` 的 `get_cells()` / `generate_candidates()`。
- 测时边界：`server_incremental_runtime.py` 中 `measured_full()`、`run_incremental()`、`top_result()`。
- 实测原始数据：`plaid_incremental_runtime.csv`；组件/场景汇总：`plaid_incremental_stage_savings.csv`。

## 最终回答

1. incremental 400→512→1024 的 centroid scoring、第一阶段 approx PID scoring、同 PID 的第二阶段 approx scoring、decompression 和精确 MaxSim 都有缓存/一次计算语义；没有发现这些核心工作对相同 PID 被重复算。
2. 重复主要是阶段级 top-k 选择、cache 前缀组装/stack、每阶段再次构建 exact score tensor 并全量排序，以及多次强制同步和小批次启动成本。
3. 最值得先优化的是同步/热路径 Python 状态管理和增量 Top100 merge；不建议先缓存 decompressed embedding 或 partial MaxSim。
4. 当前正确性证据良好，但性能还未达到目标：比单次 Full1024 慢 13.94 ms。任何“接近或优于 Full1024”的结论都必须等同边界 profile 与优化后小样本复测，不可从本 PoC 外推。
