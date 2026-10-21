# PLAID / WARP / IGP 方法调研

## 1. PLAID

PLAID 是 ColBERTv2 的高效 late-interaction 检索引擎，不是一个独立神经模型。

核心思想：

- ColBERTv2 的 token embedding 经过 centroid + residual compression；
- PLAID 先用 centroid interaction 做轻量候选过滤；
- 再用 centroid pruning 去掉低贡献 centroid；
- 最后只对更少候选做重构和 MaxSim 精排。

适合本任务的原因：

- 你已经跑通；
- 官方 ColBERT 仓库支持；
- GPU 可跑；
- 结果已经能说明速度—质量权衡。

当前限制：

- 官方实现里 `score_pids()` 把 prune / decompress / MaxSim 耦合在一起；
- 不改源码时只能准确测 `score_pids_ms`，不能严格拆出三个子阶段。

参考依据：

- PLAID 论文摘要称其通过 centroid interaction 和 centroid pruning 降低 ColBERTv2 late interaction 延迟，并报告相对 vanilla ColBERTv2 可在 GPU 上最高约 7x、CPU 上最高约 45x 加速。
- Hugging Face Papers / Papers With Code: https://huggingface.co/papers/2205.09707

## 2. WARP

WARP 是面向多向量检索的高效检索引擎，尤其和 XTR 类模型更匹配。

核心思想：

- 仍然是 IVF 类多向量检索；
- 每个 query token 在倒排/centroid 空间中检索；
- 用更激进的近似、error-aware merge、优化 kernel 减少每次 query 的工作量；
- 更偏向“少做工作但尽量保持质量”。

适合本任务的原因：

- 和 PLAID 很像，适合做同框架对比；
- 有公开实现 `xtr-warp-rs`；
- PyLate 文档中提供了 WARP 接口；
- 支持 CPU/GPU 方向的评测。

需要注意：

- WARP 不一定适合所有 ColBERT 模型；
- PyLate 文档说明 WARP 对 token-score 分布有要求，配合 XTR 训练目标效果更稳；
- 如果直接用普通 ColBERT 模型，召回可能下降。

参考依据：

- xtr-warp-rs 官方仓库： https://github.com/pau-mensa/xtr-warp-rs
- PyLate WARP 文档： https://github.com/lightonai/pylate/blob/main/docs/documentation/retrieval.md
- WARP 论文页面： https://www.alphaxiv.org/abs/2501.17788v3

## 3. IGP / GIGP+

IGP 是基于 proximity graph index 的多向量检索方法。

核心思想：

- 不是只靠传统 IVF 大量召回候选；
- 使用 proximity graph 做更高质量的候选生成；
- 目标是用更少候选达到较高召回；
- 这样 candidate generation 和 refinement 阶段都可能更省。

适合本任务的原因：

- 学长明确提到了 IVF 类、多向量索引选型；
- IGP 正是多向量检索索引优化方向；
- 后续 GIGP+ 又进一步考虑 GPU/CPU 协同。

需要注意：

- 公开实现偏 C++/CUDA，环境比 PLAID/WARP 麻烦；
- 官方 README 写明 Python 3.8、模型下载、数据目录移动、pybind/eigen/spdlog 等依赖；
- 更适合 Linux 服务器或稳定 CUDA 环境，不建议直接在普通 Colab 免费版硬啃。

参考依据：

- IGP/GIGP+ 官方实现仓库： https://github.com/DBGroup-SUSTech/multi-vector-retrieval
- SIGIR 2025 IGP 页面： https://sigir2025.dei.unipd.it/detailed-program/paper?paper=e0c641195b27425bb056ac56f8953d24
- DBLP IGP 条目： https://dblp.org/rec/conf/sigir/BianY025.html

## 建议选型顺序

1. PLAID：已跑通，作为 baseline；
2. WARP：公开实现相对好接，下一步优先；
3. IGP：方法重要，但工程重，放在 Linux/服务器环境；
4. GIGP+：作为 IGP 的 GPU/CPU 协同后续扩展。

