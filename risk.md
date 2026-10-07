# Derisk L1 Qwen3-8B RL 训练计划

训练背景见 [Derisk L1 Training Docs](./README.md)，数据契约见 [L1 样本构建总规范](./l1-data-build-spec.md)。输入与线上统一使用 canonical `RiskModelView`，输出统一使用当前三态选择性判断协议。

## 1. 设计结论与实现状态

L1 是选择性分类器，不是低成本二选一 L2。训练标签使用稳定的事实语义：

```text
aligned       -> allow 候选
misaligned    -> block 候选
indeterminate -> 正常 defer_to_l2
```

三个层次不得混用：

| 状态 | 含义 | 是否模型正常结果 |
|---|---|---|
| `indeterminate` | 输入事实不足或存在多种合理解释的模型事实标签 | 是 |
| `defer_to_l2` | 策略层将 indeterminate 路由给 L2 的动作 | 不适用 |
| `fallback` | 服务异常、超时、输出不可解析、Schema 或 Evidence/source-turn 校验失败 | 否 |

选择性分类文献通常把模型不做终判称为 `abstain` 或 reject option；本文用 `indeterminate` 表示可监督的事实标签，用 `defer_to_l2` 表示具体路由目的。单独的 `defer` 容易同时指模型状态和路由动作，协议字段中不使用该缩写。

当前线上 L1 已使用 `L1SelectiveDecision` 输出 `aligned/misaligned/indeterminate`，并由策略层映射为 `allow/block/defer_to_l2`。协议、Prompt、Evidence/source-turn 校验和训练 Schema 必须保持同源；本计划描述当前训练目标，不再把三态协议当作尚未上线的未来方案。

## 2. 当前目标标签

```json
{
  "alignment": "aligned|misaligned|indeterminate",
  "risk_categories": ["intent_mismatch|parameter_mismatch|target_mismatch|unauthorized_action"],
  "indeterminate_reason": null,
  "reason": "...",
  "evidence": {
    "user_intent": "...",
    "intent_source_turns": ["seq:1"],
    "agent_action": "...",
    "mismatch_point": null
  }
}
```

组合约束：

- `aligned`：`risk_categories=[]`、`indeterminate_reason=null`、`mismatch_point=null`。
- `misaligned`：至少一个共享风险类别、`indeterminate_reason=null`，并给出非空 mismatch point。
- `indeterminate`：`risk_categories=[]`、`mismatch_point=null`，且 `indeterminate_reason` 为 `insufficient_context` 或 `ambiguous_intent`。
- `user_intent`、`agent_action`、`reason` 必须非空。
- `intent_source_turns` 只能引用当前 segment 的真实业务 user turn；当前 segment 无业务 user turn 时为空数组。
- 禁止输出管道字段 `allow/block/defer_to_l2/fallback/outcome`、Markdown、`<think>` 或额外字段。

## 3. indeterminate 的严格边界

`indeterminate` 表示依据当前可见输入无法确定对齐关系，不是模型“信心不足”的泛称，也不是垃圾桶：

| 场景 | 处理 |
|---|---|
| 缺少完成有副作用操作所需的授权上下文 | `indeterminate_reason=insufficient_context` |
| “继续”“方案二”等指代存在多个合理解释 | `indeterminate_reason=ambiguous_intent` |
| 多裁判意见不一致 | disagreement/quarantine，不自动标 indeterminate |
| 裁判格式错误或 Evidence 非法 | invalid/quarantine |
| 样本疑似 OOD | 单独切片评测，不自动改标签 |
| 操作危险但用户明确授权 | `aligned` |
| 存在明确实质冲突 | `misaligned` |

模型置信度不是标签。未来即使增加 confidence 或校准分数，也只能作为策略层额外 `defer_to_l2` 条件，不能替代 `indeterminate` 的事实定义。

## 4. 输入与 Prompt

训练与部署必须消费同一份 canonical `RiskModelView`，并像线上一样排除内部 `schema_version`：

- `conversation_messages`
- `operation_name`
- `operation_parameters`
- `raw_command`
- `operation_effect`
- `context_facts`
- `intent_context`
- `business_context`

不得继续维护 `task_goal/user_prompt/conversation_summary/operation_semantic` 等第二套模型投影。Prompt 使用 `system + user` 双消息，关闭 thinking，部署保持 `temperature=0`、`max_tokens=128`。L1 system prompt 已包含三态边界，并明确区分正常 `defer_to_l2` 与技术 fallback；后续修改必须同步更新训练和线上合同测试。

## 5. 历史数据迁移

历史 v2/v3/`v20260723` 的三分类顶层语义可以复用，降低重建复杂度，但不能原样训练：

- 重建 canonical `RiskModelView`。
- 为 Evidence 补齐并验证 `intent_source_turns`。
- 旧 `other` 重新拆分或进入 review；不能无条件映射为 `ambiguous_intent`。
- 旧 `injection` 根据实际事实重判为共享 risk category、aligned 或 indeterminate。
- 旧 uncertain 复核其确实属于输入不足/语义多解后迁移为 indeterminate；裁判分歧和无效输出移入 quarantine。
- 使用新 Prompt 重裁判高价值切片，不能只做字段替换。

这样保留了现有数据资产，同时避免把旧裁判噪声和过时输入一起继承。

## 6. RL Parquet Schema

```text
data_source: "derisk_l1_intent_alignment"
ability: "selective_intent_alignment"
prompt: list<struct<role: string, content: string>>
extra_info: {
  case_id, split, primary_source, slice_tags,
  operation_effect, alignment, risk_categories,
  indeterminate_reason
}
reward_model.ground_truth: {
  alignment, risk_categories, indeterminate_reason, reason,
  evidence: {
    user_intent, intent_source_turns,
    agent_action, mismatch_point
  }
}
```

`extra_info` 只用于追溯与切片，reward 不得从中读取标签。

## 7. GRPO Reward

训练路线保持 Qwen3-8B + verl GRPO。建议 Reward：

| 项 | 建议权重 |
|---|---:|
| 单一 JSON 可解析 | 0.10 |
| 标签 Schema 与组合约束合法 | 0.20 |
| alignment 精确匹配 | 0.30 |
| risk_categories 集合匹配 | 0.10 |
| indeterminate_reason 精确匹配 | 0.05 |
| Evidence 完整且 source turns 合法 | 0.20 |
| reason 简短且相关 | 0.05 |

应重点惩罚把 indeterminate 强行判成 aligned/misaligned、aligned 与 misaligned 互判、非法 source turn、额外策略字段和截断 JSON。不能为了降低 `defer_to_l2` 率而奖励模型少输出 indeterminate；覆盖率由评测和策略门槛治理。

## 8. 评测与发布门禁

除 JSON/schema valid rate、三分类 accuracy/macro F1 外，必须报告：

- aligned 短路 precision 与 coverage。
- misaligned 短路 precision/recall 与 coverage。
- indeterminate recall，以及应终判样本被过度拒判的比例。
- 正常 `defer_to_l2` rate，按原因和切片拆分。
- 技术 fallback rate，按 transport/timeout/parse/schema/evidence 拆分。
- `defer_to_l2` 后 L2 的最终 allow/block 分布和 L1/L2 分歧。
- Evidence/source-turn valid rate。
- P50/P95 延迟、输入/输出 Token 和净 L2 流量节省。

发布前还必须通过 canonical 输入 parity、Prompt parity、session/case/near-duplicate 隔离、test real-only 和多样性门禁。历史 checkpoint 可作 baseline，但不具备新拒判协议时不能作为目标版本。
