# LLM 清洗/重置数据集的目的与结果

## 1. 清洗不是改写

LPM 中的“LLM 清洗”不应理解为让大模型把剧情文本整理得更顺、更自然、更适合阅读。那会破坏原始观测数据。

LLM 清洗的目标是：

$$
D_{\mathrm{raw}}
\rightarrow
D_{\mathrm{canonical}}
\rightarrow
D_{\mathrm{enriched}}
\rightarrow
D_{\mathrm{lpm}}
$$

其中：

- $D_{\mathrm{raw}}$ 是原始提取结果，不修改。
- $D_{\mathrm{canonical}}$ 是确定性脚本生成的规范事件流。
- $D_{\mathrm{enriched}}$ 是 LLM 辅助生成的结构化观测解释。
- $D_{\mathrm{lpm}}$ 是用于分析和训练的回应单元、轨迹索引和测试样本。

LLM 只参与 $D_{\mathrm{enriched}}$，不能覆盖 $D_{\mathrm{raw}}$ 或 $D_{\mathrm{canonical}}$。

## 2. LLM 清洗要解决什么问题

当前提取数据已经有基本结构，但存在几个不适合直接建模的问题：

1. `text_type` 不可靠，不能直接判断对话/旁白。
2. 旁白可能是主角视角观测，不能当作心理真值。
3. 选择支和跳转结构有少量缺失或退化。
4. 说话人有低频、匿名、别名和临时角色。
5. 部分事件需要区分直接发言、动作描写、场景描写、主观解释。

这些问题不是靠“重写文本”解决，而是靠生成结构化字段解决。

## 3. LLM 清洗的正式输出

第一版清洗流程应输出四类文件。

### 3.1 规范事件流

路径建议：

$$
\texttt{Xi/LPM/data/canonical/events.jsonl}
$$

每行一个 canonical event。字段见：

$$
\texttt{Xi/LPM/docs/event\_schema\_zh.md}
$$

这部分主要由确定性脚本生成。

### 3.2 LLM 事件补充层

路径建议：

$$
\texttt{Xi/LPM/data/enriched/event\_annotations.jsonl}
$$

每行对应一个 `event_id`。LLM 只输出闭集字段，例如：

- `llm_observation_subtype`
- `viewpoint_mediated`
- `observer`
- `actor_candidates`
- `addressed_to_candidates`
- `cleaning_decision`
- `issue_codes`
- `confidence`

这部分不能包含改写后的剧情文本。

### 3.3 角色输入-回应单元

路径建议：

$$
\texttt{Xi/LPM/data/lpm/response\_units.jsonl}
$$

每行一个：

$$
(E_k^c, Y_k^c)
$$

核心字段：

| 字段 | 说明 |
|---|---|
| `unit_id` | 回应单元 ID |
| `target_character` | 目标角色 |
| `trajectory_id` | 场景或路线 ID |
| `input_event_ids` | 外部输入事件列表 |
| `response_event_id` | 目标回应事件 |
| `input_mix` | 输入通道统计 |
| `flags` | 构造异常或注意事项 |

### 3.4 清洗报告

路径建议：

$$
\texttt{Xi/LPM/data/reports/cleaning\_report.json}
$$

报告至少包含：

- 总事件数。
- 各 `observation_channel` 分布。
- 各 `llm_observation_subtype` 分布。
- 被 `drop` 或 `needs_review` 的数量。
- 分支缺失数量。
- 低频说话人数量。
- LLM 失败率。
- 抽样复核记录。

## 4. LLM Prompt 的目标

LLM prompt 不应问：

> 这个段落适合做什么？

这类问题太松散，输出不可训练、不可统计。

LLM prompt 应问：

> 给定一个 canonical event，在不改写原文的前提下，判断它属于哪种观测子类、是否存在视角中介、涉及哪些角色、是否需要复核。

输入应是单事件或小窗口：

$$
(e_{t-r}, \ldots, e_t, \ldots, e_{t+r})
$$

输出必须是严格 JSON，且字段闭集。

## 5. LLM 输出示意

事件级输出：

$$
\begin{aligned}
&\texttt{event\_id}: \texttt{"..."} \\
&\texttt{llm\_observation\_subtype}: \texttt{"subjective\_interpretation"} \\
&\texttt{viewpoint\_mediated}: \texttt{true} \\
&\texttt{observer}: \texttt{"protagonist"} \\
&\texttt{actor\_candidates}: [\texttt{"..."}] \\
&\texttt{addressed\_to\_candidates}: [] \\
&\texttt{cleaning\_decision}: \texttt{"keep"} \\
&\texttt{issue\_codes}: [\texttt{"viewpoint\_bias"}] \\
&\texttt{confidence}: 0.82
\end{aligned}
$$

注意：这里不产生 `text_rewritten` 字段。

## 6. 当前 annotations.jsonl 的地位

当前：

$$
\texttt{Xi/LPM/data/demo\_passages/annotations.jsonl}
$$

只是 DeepSeek API 与 JSON 输出的 smoke test。它不是正式清洗结果，也不是正式标注体系。

原因：

- 字段过于自由。
- `recommended_uses` 不可训练。
- `observation_channels` 不是闭集。
- `quality_score` 缺少复验标准。
- 输出对象是段落评审，不是事件级清洗。

后续正式清洗应废弃这种段落评审式标注，改为事件级闭集结构抽取。

