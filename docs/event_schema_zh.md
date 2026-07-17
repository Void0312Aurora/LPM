# LPM 事件 Schema 草案

## 1. 轨迹的基本构成

当前 GalGame 数据的基本结构不是单角色轨迹，而是场景内的交叉叙事序列：

$$
\tau_s = (e_1, e_2, \ldots, e_T)
$$

其中 $s$ 是场景，$e_t$ 是一个被观测到的事件。事件来自不同通道：

- 角色直接发言。
- 旁白或主角视角描述。
- 选择支。
- 场景跳转。

多个场景再通过跳转或路线顺序组合成更长剧情：

$$
\Gamma = (\tau_{s_1}, \tau_{s_2}, \ldots)
$$

因此，LPM 的第一层数据对象应是 canonical event，而不是“训练样本”。

## 2. 一个事件至少需要哪些字段

第一版 canonical event 不应试图承载心理解释。它只描述一个可追溯、可排序、可建模的观测事件。

最小字段分为五组，共 18 个核心字段。

### 2.1 身份与来源

| 字段 | 类型 | 说明 |
|---|---|---|
| `event_id` | string | 全局唯一 ID |
| `source_dataset` | string | 来源数据集，例如 `tree_dialogues_v2` |
| `source_file` | string | 来源文件路径 |
| `source_line` | int/null | 来源 JSONL 行号，若适用 |
| `source_path` | string | 原始对象路径，例如 `content[15]` |

这些字段保证任何派生事件都能回到原始提取数据。

### 2.2 轨迹位置

| 字段 | 类型 | 说明 |
|---|---|---|
| `trajectory_id` | string | 场景级轨迹 ID，通常等于 `scene` |
| `scene` | string | 场景名 |
| `local_pos` | int | 在场景 `content` 中的位置 |
| `text_index` | int/null | 原始文本 index，选择/跳转可为空 |

`local_pos` 是主排序键。`text_index` 只能作为对齐键，因为当前数据中存在少量重复和顺序回退。

### 2.3 事件类型

| 字段 | 类型 | 说明 |
|---|---|---|
| `raw_type` | string | 原始类型，例如 `text` 或 `choice` |
| `event_kind` | enum | 规范类型 |
| `observation_channel` | enum | 观测通道 |

`event_kind` 第一版闭集：

- `text`
- `choice`
- `jump`
- `unknown`

`observation_channel` 第一版闭集：

- `direct_dialogue`
- `narration`
- `choice`
- `scene_jump`
- `system_or_debug`
- `unknown`

其中 `direct_dialogue` 可由 `speaker != null` 确定性生成；`narration` 可由 `speaker == null` 确定性生成。不要使用当前 `text_type` 作为主判据。

在 `clean_event_v0` 中，`type` 保留 `narration` 与 `monologue` 的结构区分。该区分不是原始数据字段；具体判定应由清洗/标注环节写入，基础构造脚本不应凭启发式擅自改写。

### 2.4 参与者与文本

| 字段 | 类型 | 说明 |
|---|---|---|
| `speaker_raw` | string/null | 原始说话人 |
| `speaker_norm` | string/null | 去空白并执行受控异常标点归一后的说话人 |
| `text_raw` | string/null | 原始文本 |
| `text_norm` | string/null | 仅做空白/控制字符清理与受控异常标点归一后的文本 |

`text_norm` 不允许改写、润色、补全剧情。它只允许执行：

- 去除首尾空白。
- 删除控制字符。
- 将明显混入的异常同义标点归一到常规标点，例如阿拉伯问号 `U+061F` 归一为 `?`。
- 统一空字符串为 `null` 或过滤。

### 2.5 结构信息与异常

| 字段 | 类型 | 说明 |
|---|---|---|
| `choice` | object/null | 选择支信息 |
| `jump` | object/null | 场景跳转信息 |
| `flags` | list[string] | 异常或注意事项 |

`choice` 可包含：

- `var_id`
- `options`
- `option_value`
- `has_leads_to`
- `leads_to`

`jump` 可包含：

- `target_scene`
- `branch_index`

`flags` 第一版可包含：

- `duplicate_text_index`
- `index_order_regression`
- `empty_text`
- `control_char_removed`
- `text_punctuation_normalized`
- `speaker_punctuation_normalized`
- `degenerate_choice`
- `branch_missing_target`
- `low_frequency_speaker`
- `debug_or_system_scene`

## 3. LLM 不应负责的字段

以下字段必须由确定性脚本生成：

- `event_id`
- `source_*`
- `trajectory_id`
- `scene`
- `local_pos`
- `text_index`
- `raw_type`
- `speaker_raw`
- `text_raw`
- `choice`
- `jump`
- 由确定性规则可判断的 `flags`

LLM 不应重新决定这些字段。否则派生数据会失去可追溯性。

## 4. LLM 可以补充的事件级字段

LLM 的角色不是“美化文本”，而是在必要时补充确定性脚本难以判断的观测解释字段。

这些字段应进入单独的 enrichment 层，而不是覆盖 canonical event。

| 字段 | 类型 | 说明 |
|---|---|---|
| `llm_observation_subtype` | enum | 更细观测子类 |
| `viewpoint_mediated` | bool/null | 是否明显经过主角视角中介 |
| `observer` | enum | 观测者 |
| `actor_candidates` | list[string] | 事件中被描述或行动的角色候选 |
| `addressed_to_candidates` | list[string] | 对话指向对象候选 |
| `cleaning_decision` | enum | 保留、丢弃、需复核 |
| `issue_codes` | list[string] | 问题码 |
| `confidence` | number | 0 到 1 的置信度 |

`llm_observation_subtype` 第一版闭集：

- `direct_utterance`
- `internal_monologue`
- `action_description`
- `scene_description`
- `subjective_interpretation`
- `choice_option`
- `scene_jump`
- `unclear`

`observer` 第一版闭集：

- `protagonist`
- `speaker`
- `external_narrator`
- `unknown`

`cleaning_decision` 第一版闭集：

- `keep`
- `drop`
- `needs_review`

`issue_codes` 第一版闭集：

- `viewpoint_bias`
- `ambiguous_actor`
- `ambiguous_speaker`
- `fragment_without_context`
- `system_or_debug`
- `branch_incomplete`
- `text_noise`
- `translation_or_alignment_risk`

## 5. 事件与回应单元的关系

事件本身不是最终训练样本。角色输入-回应单元由事件构造：

$$
E_k^c = e_{t_{k-1}^c + 1 : t_k^c - 1}
$$

$$
Y_k^c = e_{t_k^c}
$$

其中 $Y_k^c$ 必须是 `observation_channel = direct_dialogue` 且 `speaker_norm = c` 的事件。

因此，清洗的第一目标是生成可靠事件流；第二目标才是从事件流生成回应单元。

## 6. 当前结论

一个 LPM 事件不需要大量心理字段。第一版需要 18 个核心字段来保证：

- 可追溯。
- 可排序。
- 可区分通道。
- 可保留分支/跳转。
- 可标记异常。
- 可构造 $E_k^c \rightarrow Y_k^c$。

LLM 补充字段应独立于 canonical event，并保持闭集、可统计、可复核。
