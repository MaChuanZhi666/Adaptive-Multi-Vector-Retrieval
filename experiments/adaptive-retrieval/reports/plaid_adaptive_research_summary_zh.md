# 理解 PLAID 检索中的 Query-Dependent Budget Sensitivity

## 摘要

PLAID 的检索预算对不同 query 的影响并不相同。对一部分 query，相关文档 PID 已经进入 candidate generation，但随后在 `ndocs` 截断边界被移除；提高预算可以恢复部分 Recall@100。另一部分 query 在 candidate generation 阶段就缺少相关 PID，因此提高 `ndocs` 无法修复。还有一部分 query 在提高预算后没有观察到质量提升。

本研究依次测试了静态 query feature、运行时动态信号，以及 approximate-then-exact 两阶段原型。结果证明 query-dependent budget sensitivity 确实存在，也证明了 opt-in 两阶段接口可以做到 parity 正确。但目前还没有得到可靠且低成本的 adaptive controller：静态规则存在跨数据集泛化问题；运行时信号较弱、受配额机制影响或不稳定；两阶段 approximate 路径在独立集上比 Always1024 更慢且 Recall 更低。

> **Query-dependent budget sensitivity exists, but reliable and low-cost adaptive control remains unsolved.**
>
> **Query 确实存在不同的预算敏感性，但可靠且低成本的自适应控制仍未解决。**

因此，本工作应被定位为机制分析与工程可行性研究，而不是一个已经成功的 adaptive PLAID 系统。

## 1. 背景

### 多向量检索与 PLAID

在多向量检索中，query 和 document 都由多个 token embedding 表示。Late-interaction scorer（例如 MaxSim）逐个比较 query token 与 document token 的匹配程度，并聚合最强的 token-level 匹配。这种方式具有较强的细粒度匹配能力，但如果对大量候选文档进行计算，代价会很高。

PLAID 通过基于 centroid 的索引和 pruning 降低成本。整体流程可以概括为：query embedding 探索 centroid 并生成 candidate PID；随后利用近似 centroid score 对候选进行排序和过滤；最后对保留下来的 document residual 表示进行解压缩，并执行精确的 late-interaction scoring。

### 预算权衡

`ndocs` 控制 candidate/refinement 阶段的预算。更大的 `ndocs` 允许更多候选进入精确评分，可能恢复相关文档，但会增加计算成本。更小的 `ndocs` 可能更快，但会在 exact scoring 之前移除相关 PID。

candidate generation 是另一个独立阶段：如果某个 PID 从未被生成，增加 `ndocs` 也无法把它找回来。

因此，固定预算天然存在质量/速度权衡；adaptive budget 要想有效，必须在不超过节省下来的 refinement 成本的前提下，在线判断哪些 query 值得使用更大的预算。

## 2. 主要观察：不同 query 对 `ndocs` 的响应不同

固定预算和逐 query 结果显示出明显的预算敏感性差异。在两阶段 runtime cohort 中，400 相比 1024 存在 Recall@100 损失的 query 数量为：old 集 3/9，independent 集 9/36。

在 independent 的 9 个 hard query 中，与自身 1024 结果达到相同 Recall@100 的 query 数量为：预算 200–400 时 0/9，512 时 4/9，768 时 8/9，1024 时 9/9。

另一方面，也有不少 query 在低预算和高预算下 Recall@100 没有变化。这说明更大预算对部分 query 是必要的，但对另一部分 query 可能是浪费。困难在于：必须在不花费过多额外计算的情况下，提前识别前一类 query。

上述数字来自不同的小规模实验 cohort。27-query mechanism cohort、36-query two-stage runtime cohort 和 94-query frozen-rule validation 分别报告，不进行合并统计。

## 3. 机制分析

直接进行机制追踪的 cohort 包括 old=9 个 query、independent=27 个 query。下面的分类同时区分“质量确实恢复”和“PID 层面的 pruning 证据”。独立集的 mixed query 同时计入 truncation-responsive 和 candidate-generation-limited，因此两列不能直接相加。

| Cohort | 截断敏感 | candidate-generation 缺失 | 预算不敏感且候选完整 | 结果未知 |
|---|---:|---:|---:|---:|
| old，n=9 | 3（33.3%） | 0 | 6（66.7%） | 0 |
| independent，n=27 | 9（33.3%，其中 1 个 mixed） | 3（11.1%，其中 1 个与前者重叠） | 14（51.9%） | 2（7.4%） |

### A. Truncation-limited query

相关 PID 已经出现在 candidate-generation 输出中，但在 `ndocs` pruning 边界被移除。对 independent hard subset 的 9 个 query，有 8 个直接观察到“candidate 中存在、截断后被移除”；query 106 同时包含 candidate-generation failure。

提高 `ndocs` 可以恢复被截断的 PID，并在部分 query 上提升 Recall@100。但 PID 被移除并不一定会造成 query-level Recall@100 下降：如果其他相关结果仍然足够，聚合指标可能不变。

### B. Candidate-generation-limited query

至少一个相关 PID 在 candidate generation 阶段就不存在。这类 PID 不可能通过提高 `ndocs` 找回。

在 mechanism cohort 中，old 为 0/9，independent 为 3/27；在较大的 two-stage independent cohort 中为 5/36。这些数字表示“至少一个相关 PID 缺失”，并不表示所有相关文档都缺失。

query 106 是 mixed case：部分质量可以通过增加预算恢复，但仍存在 candidate-generation 缺失。因此 candidate generation 需要独立于 `ndocs` adaptation 进行改进。

### C. Budget-insensitive query

在 trace cohort 中，old 有 6/9、independent 有 14/27 的 query 在低预算到高预算之间没有观察到质量提升，且 candidate coverage 完整。

这里的“easy”只表示在当前指标和预算范围内对预算不敏感，不代表这些 query 在所有意义上都简单。另有两个 independent query 的质量字段缺失，因此没有强行归入 easy。

## 4. Adaptive 尝试

### A. 静态 query-side predictor

**想法：** 使用 query-side feature，尤其是 token-level centroid gap 的 lower-tail 特征（如 `gap_p10`、`worst_token_gap`），预测 query 是否会受到小预算影响，并对预测为风险的 query 提升预算。

**实验：** 在 old 和 independent query 集上分析 feature 与 frozen rule，并进行 offline policy replay 和 94-query held-out validation。

**结果：** 在小规模 mechanism comparison 中，`gap_p10` 的方向比 `gap_mean` 更稳定，oriented AUC 约为 old 0.944、independent 0.722。但在更大范围的 validation 中，`gap_p10` AUC 从 old48 的约 0.807 降到 independent94 的约 0.642。第一版 frozen rule 的 hard-query recall 为 3/9，precision 为 14.3%，FN=6、FP=18；后续 v2 frozen rule 在 independent94 上 hard-query recall 为 0/9，FN=9、FP=11。

**失败原因：** hard query 数量太少导致阈值不稳定；feature 分布和 feature-label 关系发生变化；以 recall 为优先的阈值会带来大量 false promotion。方向一致并不足以形成可用的跨数据集 operating point。

### B. Runtime approximate signal

**想法：** 在检索过程中观察 candidate growth、cutoff behavior 或 ranking stability，如果近似结果已经稳定则提前停止。

**实验：** 通过不同 `ndocs` checkpoint 比较 candidate/TopK 变化，并在两阶段 runner 中记录 cutoff margin、tail slope、candidate overlap 和 candidate growth，然后只进行一次 exact finalize。

**结果：** candidate growth 主要由预算配额决定，AUC 约为 0.5。Top75 churn 在 independent 上有一定区分能力，AUC 约 0.652，但在 old control 上方向反转，AUC 约 0.167。一个冻结的 churn threshold 虽然捕获了 9/9 个 independent hard query，却同时提升了 24/27 个 easy query。后续两阶段数据中，400 cutoff-margin 的 AUC 从 old 的 0.806 降至 independent 的 0.527。400/512 阶段的 candidate overlap/growth 在当前实现中是常量。

此外，native filter 没有暴露内部 cutoff score，因此没有伪造 cutoff margin 或 slope。依赖未来 exact 结果计算的 approximate/exact Top100 overlap 也不能作为在线 early signal。

**失败原因：** 可用的 early signal 要么受配额机制控制、要么区分能力弱、要么跨数据集不稳定、要么只有在 exact scoring 之后才能得到。

### C. Two-stage approximate → exact

**想法：** 在 400/512 运行 cheap approximate checkpoint，根据状态决定停止或扩大预算，最后只做一次 exact decompression/MaxSim。

**实验：** opt-in API 的 6-query parity gate 通过 18/18；优化后的 incremental PoC 也在测试 query 上通过 exact Top100 parity。随后在 45 个 frozen query、每个 3 次重复的 runtime 实验中测试从 old 集冻结的 rule。

**结果：** independent36 上，adaptive policy 将 26/36 个 query 提升到 400 以上，Recall@100 为 0.886，低于 Always1024 的 0.908；平均延迟为 31.405 ms，高于 Always1024 的 28.578 ms。三个实际会从更大预算获益的 query（100、128、142）被停在 400；另有 20 个被提升预算的 query 没有观察到 Recall@100 增益。approximate checkpoint 加 decision/state work 约耗时 8.25 ms/query，而减少的 final refine 只节省约 4.33 ms/query。

单独的 400→512→1024 incremental timing 约为 224.33 ms，而单次 full1024 约为 81.27 ms；该比较使用其自身的 timing protocol。

**失败原因：** 工程上可以做到 parity 正确，但 approximate 阶段还不够便宜；同时 policy 既存在 false promotion，也存在 missed promotion。因此 parity 只能证明工程可行性，不能证明 adaptive speed/quality tradeoff 成功。

## 5. 最终研究结论

1. **Query-dependent budget sensitivity 确实存在。** 部分 query 的 Recall@100 会随 `ndocs` 增加而提升，另一部分不会。
2. **主要可恢复的质量损失来自 candidate generation 之后的 truncation。** 另有 candidate-generation failure 构成独立的质量上限，预算提升无法消除。
3. **简单 adaptive 难以泛化。** hard query 样本少，feature 跨数据集发生变化，若干动态 feature 是机制上固定的，recall-first 阈值会带来大量 false promotion。
4. **Two-stage 架构工程上可行，但目前没有收益。** parity 已通过，但 approximate checkpoint 成本超过了节省的 exact-refine 成本，独立集质量也低于 Always1024。

> **Query-dependent budget sensitivity exists, but reliable and low-cost adaptive control remains unsolved.**
>
> **Query 确实存在不同的预算敏感性，但可靠且低成本的自适应控制仍未解决。**

本研究的价值在于分离了两种 failure mechanism，量化了简单 signal 的跨数据集限制，并建立了 parity-checked 的工程基线。应将其汇报为机制与可行性研究，而不是 adaptive retrieval 成功。

## 6. 后续工作

- **降低 approximate checkpoint 成本：** 复用 candidate ordering 和 search state，减少重复 filtering 与 synchronization，并继续以 exact parity 作为硬门槛。
- **提升 candidate coverage：** 将 candidate-generation expansion 与 `ndocs` adaptation 分开研究，单独报告其开销和残余错误。
- **扩大 held-out validation：** 预先注册一个简单、可解释的 signal 和评估协议，在新的、更大的 held-out collection 上验证；不要在已有 independent 集上继续调阈值。

## 适合组会结尾的表述

“我们的实验表明，PLAID 的不同 query 对检索预算的敏感性不同。我们已经定位到一种可恢复的 truncation 机制，但也发现 candidate-generation failure 是预算提升无法解决的独立问题。现有静态和运行时 signal 还不能稳定、低成本地跨数据集泛化；parity 正确的两阶段原型目前也比 full1024 更慢。因此，这项工作不是一个已经成功的 adaptive controller，而是一项机制诊断和工程可行性研究。下一步应优先降低 checkpoint 成本、独立改进 candidate coverage，并在更大的 held-out 集上验证。”
