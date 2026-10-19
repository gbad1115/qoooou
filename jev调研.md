
# JEv 在 Derisk 中的适用性调研与评估

> 调研日期：2026-10-19  
> 适用项目：`agent-infra` / Derisk 运行时风控管道  
> 状态：方案评估，尚未引入线上代码

## 1. 结论

建议把 JEv 作为 **L1 的 shadow candidate（旁路候选）和高置信意图对齐门**进行验证，不建议直接替换 L2，也不建议修改 L0。

推荐的落地顺序是：

1. 在不影响决策的 shadow 模式记录 JEv 的 typed decision、概率、延迟和错误。
2. 先只验证高置信 `aligned` 的 allow 短路，并要求通过语言分片、校准和回放验收。
3. 再单独验证高置信 `misaligned` 的 block；block precision 达标前不进入生产短路。
4. 所有低置信、上下文不足、协议错误和服务异常继续 `defer_to_l2`。

JEv 对 Derisk 的主要价值是将一部分“用户意图与当前操作是否对齐”的判断从较慢的 L2 中分流出来，并输出适合代码路由的结构化结果。它不能替代 L2 的复杂语义分析，也不能单独满足 Derisk 对证据、来源轮次和冲突点的审计要求。

## 2. 调研范围与资料

本次调研覆盖 JEv/System One 的公开资料，以及可自托管的相关开源模型：

- [Typesafe：Introducing System One Models and JEv](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [JEv Introduction](https://docs.typesafe.ai/introduction.md)
- [JEv Models](https://docs.typesafe.ai/models.md)
- [JEv 的已知限制（jev-1.13）](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
- [LLM guardrails 示例](https://docs.typesafe.ai/cookbooks/llm_guardrails.md)
- [Typesafe API](https://docs.typesafe.ai/api.md)
- [Open Alternative to Jev](https://github.com/ikermoel/open-alternative-jev)
- [GLiClass](https://github.com/Knowledgator/GLiClass)
- [ModernBERT](https://github.com/AnswerDotAI/ModernBERT)
- [Granite Guardian 3.0](https://huggingface.co/ibm-granite/granite-guardian-3.0-2b)

公开资料中的效果和限制是供应方或项目方报告，不能直接等同于 Derisk 数据集上的准确率。上线结论必须以 Derisk 的离线回放和线上 shadow 数据为准。

## 3. JEv/System One 能力概述

JEv 属于面向判断任务的 typed evaluation 模型。调用方给出待判断内容和明确的选项，模型返回结构化选择及其置信信息，适合把判断结果接入程序路由。对 Derisk 来说，最有用的是把以下判断建模为有限选项：

| Derisk 判断 | JEv 候选输出 | 用途 |
|---|---|---|
| 意图与操作是否一致 | `aligned` / `misaligned` / `insufficient_context` | L1 allow、block 或回退 L2 |
| 是否需要继续分析 | `decisive` / `defer` | 控制 L2 流量 |

`Choice` 适合表达明确的多选判断；`Noul` 更适合对自然语言判断进行概率化输出。实际接入时应先固定选项顺序、选项文本和版本号，避免把选项顺序变化误当成模型效果变化。

当前 Derisk 的 L1 prompt 和协议没有要求模型输出固定的“冲突类型”枚举；L1 只输出 alignment、拒判原因、reason 和 Evidence，其中 `mismatch_point` 是描述具体冲突的自由文本，而不是分类字段。JEv 的输出本身不应被视为 Derisk 的最终解释，接入时仍需处理 `evidence`、`intent_source_turns`、`mismatch_point` 以及最终的 `decided_by`。

## 4. 与 Derisk 当前架构的映射

Derisk 当前在线链路为：

```text
Pre-L0 → Adapter → L0 → L1（默认关闭）→ L2
```

入口和编排分别位于 [`service.py`](../../src/derisk/pipeline/service.py)、[`orchestrator.py`](../../src/derisk/pipeline/orchestrator.py)，架构约束见 [`architecture.md`](../../architecture.md)。各层与 JEv 的关系如下：

| 层 | 当前职责 | JEv 评估结论 |
|---|---|---|
| Pre-L0 | 协议识别、空对话处理和源审计 | 不接入。JEv 不应改变输入协议或空源语义 |
| L0 | 确定性规则和白名单，首条命中即终止 | 不替换。规则结果可解释、延迟稳定，适合继续保留 |
| L1 | 小模型意图、参数、目标对齐；可短路或回退 L2 | 最适合成为 JEv 的接入点 |
| L2 | 完整上下文的复杂语义分析和取证 | 不替换。处理不确定、长上下文和复杂冲突 |

建议 JEv 读取与当前 L1 相同的规范化 `RiskInput`/模型视图，避免旁路重新选择消息、重新截断上下文或读取未经脱敏的原始载荷。建议增加独立的适配器和结果记录，不改写现有 L1/L2 协议，便于对照回放。

推荐的控制流为：

```text
L0 未终止
  → JEv shadow 判断
  → 协议/置信度/语言覆盖检查
      → 高置信 aligned：候选 allow
      → 高置信 misaligned：候选 block
      → 其余：defer_to_l2
  → L2 对候选和 defer 样本进行最终对照
```

“候选 allow/block”在 shadow 阶段只记录，不改变真实响应。进入主动短路后，仍应保留 L2 回放样本和抽样复核能力。

## 5. 适用场景

JEv 适合以下类型的 L1 判断：

- 当前用户任务是否允许当前工具操作，且判断可以归纳为少量稳定选项。
- 目标、参数或副作用是否明显偏离用户最近的有效意图。
- 对已经过输入预算、脱敏和会话选择的短上下文做快速分流。
- 需要概率、置信度和 `defer` 分支，以便由代码决定是否继续 L2。
- 作为 L2 的旁路监控器，发现模型版本、语言或数据切片上的漂移。

以下场景不适合由 JEv 单独决定：

- 需要跨多轮对话恢复隐含授权或复杂业务前提。
- 需要解释证据来自哪一轮、哪一个工具调用，或需要生成可审计冲突点。
- 需要同时理解长 Skill、工具能力、环境权限和副作用链路。
- 输入中存在上下文冲突、对抗性指令、引用文本或多个任务段落。
- 需要判断安全政策、越权、数据外发等多种风险的组合结果。

这些请求应继续交给 L2，JEv 只能提供候选标签或辅助特征。

## 6. 主要风险与约束

### 6.1 语义和审计风险

- JEv 返回的是判断结果和置信信息，不天然返回 Derisk 所需的证据来源、`intent_source_turns` 和 `mismatch_point`。
- 高置信度不等于事实正确；若直接把概率当作决策依据，容易把系统校准误差放大成 allow 或 block。
- 官方限制说明提到 literal reading、上下文过大、对抗内容和选项顺序偏差等问题。Derisk 的工具调用、引用文本和多轮上下文正好会触发这些边界情况。
- 官方资料明确英语效果最好，CJK 准确率可能较低。中文、英文和中英混合必须分别评估，不能用英文总体结果代替。

### 6.2 可靠性、合规和运维风险

- 使用外部 API 会引入网络、速率限制、服务可用性、成本和数据出境审查要求。
- 需要记录 429、529、超时、连接错误和协议错误，并在任何异常时回退 L2；不能因为 JEv 不可用而默认 block 或绕过现有审计。
- 生产配置应固定模型版本（例如 `jev-1.14.0`），禁止直接依赖 `jev-latest`；模型、选项、提示和适配器版本应进入审计记录。
- 发送给外部服务的载荷必须沿用 Derisk 的脱敏和预算边界；默认不发送未脱敏的原始操作参数、Skill 正文或凭据。
- 若 JEv 自托管，需要承担 GPU、模型服务、升级、容量和监控成本；开源实现还可能存在模型和 HTTP 接口不兼容的问题。

### 6.3 成本和延迟风险

当前仓库的 L2 教师试跑显示，400 个样本的总耗时 p50 约 5.8 秒、p90 约 16.5 秒、p99 约 47.5 秒；该数据包含网络和模型等待，不能直接推导 JEv 延迟，但说明 L1 分流存在潜在价值。JEv 是否真正降低端到端延迟，要以同一批样本的 p95、L2 defer 比例和失败重试情况测量。

## 7. 开源方案对比

| 方案 | 形态 | 优点 | 局限 | 对 Derisk 的定位 |
|---|---|---|---|---|
| Open Alternative to Jev | Apache-2.0，自托管 typed decisions，支持 Qwen/Hugging Face/vLLM | API 思路接近 JEv，可离线研究，数据不出域 | Alpha，依赖 GPU；没有官方 HTTP drop-in；有 option order 和 packed interference 风险 | 离线基线和自托管实验，不建议直接线上替换 |
| GLiClass | 轻量 zero-shot/multi-label 分类 | 可输出多标签分数，适合做语义特征或路由 | score 不等于校准概率；不提供 Derisk 证据和来源轮次 | 低成本候选特征、离线对照 |
| ModernBERT | 编码器基础模型 | 可在中文/多语言标注数据上训练专用分类器，推理成本可控 | 需要标注、训练、校准和部署；不是即插即用 JEv | 中长期自训练 L1 分类器 |
| Granite Guardian 3.0 2B | 安全/越狱/groundedness 检测模型 | 可自托管，适合作为安全旁路 | 英文为主，任务与 Agent 意图/参数对齐不同 | 安全信号旁路，不作为意图对齐主模型 |

开源方案都不能自动解决“证据审计”和“低误阻断”问题。是否采用，应该与 JEv 使用同一套回放样本、校准方法和故障回退标准比较。

## 8. 建议的验证路径

### 阶段 A：Shadow

- JEv 读取与 L1 相同的 `RiskInput`，不影响现有 allow/block 响应。
- 记录 `model_version`、选项版本、输入语言切片、decision、各类概率、延迟、token/cost、HTTP 状态、错误码和 `defer_to_l2` 原因。
- 对每个样本保留 L0、当前 L2、JEv 和人工/审核标签的关联 ID；不要把当前 `review_status=pending` 的结果当作 ground truth。
- 验证选项顺序、重复请求、超时和异常响应的确定性。

### 阶段 B：高置信 allow

- 只对离线回放已证明安全的 `aligned` 区间启用 allow 短路。
- 对中文、英文、中英混合、短上下文、长上下文、引用/代码、工具调用和多任务分别设门槛。
- 低于门槛、缺少有效证据、上下文过大或语言覆盖不足时，统一回退 L2。

### 阶段 C：高置信 block

- 先做人工复核和 L2 对照，确认每个 block 都有稳定、可审计的具体冲突证据。
- 只有在 block precision 达到生产门槛、误阻断受控且具备快速回滚开关后，才允许 JEv 直接 block。
- JEv block 仍由本地生成审计字段；不能把 JEv 的自由文本当作证据来源。

### 阶段 D：长期运行

- 监控语言分布、置信度分布、ECE、L2 defer 比例、模型版本漂移和成本。
- 定期抽样 JEv 高置信 allow/block 与 L2 结果不一致的样本。
- 任何模型、选项、提示或输入视图变化都重新跑回放和校准，不使用历史阈值直接上线。

## 9. 验收指标

建议将以下指标作为进入主动短路前的最低门槛，按语言和场景切片报告：

| 指标 | 建议门槛或要求 | 说明 |
|---|---|---|
| block precision | `≥ 99.5%` | 以人工确认或高质量审核标签为准；协议有效不等于语义正确 |
| false block | `≤ 0.1%` | 以全部可评估请求及关键语言切片分别计算 |
| 语言覆盖 | 中文、英文、中英混合分别达标 | 不能只报告总体平均值 |
| 概率校准 | ECE `≤ 0.05` | 同时观察 Brier score 和 KL，按版本比较 |
| 新增延迟 | p95 `≤ 500 ms` | 测量端到端新增耗时，不只测模型服务时间 |
| 可用性 | 记录 429/529、超时、连接和协议错误 | 所有异常必须可回退 L2 |
| 流量收益 | 记录 `defer_to_l2` 比例和真实 L2 节省 | 不能只看 JEv 单次耗时 |
| 成本 | 记录请求、token 和总成本 | 按 allow、block、defer 和语言切片拆分 |
| 审计完整性 | 100% 记录模型/选项/输入视图版本和结果 | 能关联到 L2 或人工复核样本 |

其中，`block precision` 和 `false block` 是进入主动 block 的硬门槛；延迟和成本门槛不能以牺牲安全精度为代价放宽。
