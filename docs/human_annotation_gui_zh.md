# LPM 人工事件标注器

## 1. 目的

人工标注器用于替代高风险的全自动 LLM event fill。它不修改：

- `data/canonical/events.jsonl`
- `data/clean_v0/events/*.jsonl`
- `data/clean_v0/graph.json`

每次人工保存都会追加到独立的审计日志。导出时再从当前 `clean_v0` 基线与最新人工 patch 生成可信事件集。

## 2. 确定性字段边界

以下字段始终只读：

- `id`
- `source`
- `trajectory`
- `order`
- `kind`
- `text`
- choice option 的索引和原始文本

当 canonical event 中存在明确的 `speaker_norm` 时，该事件按直接对白处理，并锁定：

- `type = utterance`
- `action_type = speak`
- `agent = speaker_norm`

人工不能覆盖这些字段，只标注 `target`、`participants`、质量状态和备注。这一规则用于避免已在 32 条 LLM 试验中出现的“明确 speaker 被模型改错”问题。

对于没有源 speaker 的普通事件，人工可判断：

- `type`: `utterance/action/monologue/narration/other`
- `agent`
- `target`
- `participants`
- `valid`
- `flag`

`action_type` 由 `type` 自动推导，不单独编辑。`participants` 导出时自动包含明确的 `agent` 和 `target`。`monologue` 的 `target` 固定为空。

所有结构化字段都必须从预制选项中选择：

- `action_type`：只读预制下拉，由 `type` 自动映射为 `speak/action/other`。
- `agent`、`target`、`participants`：从 canonical speaker 词表中单选或多选。
- `flag`：从闭集 flag 中选择。
- choice option 内的 `agent`、`target`、`participants`、`flag` 使用相同选择器。

选择器中的搜索框只负责过滤已有选项，不会创建新角色名或新 flag。当前上下文角色和受控常用角色默认显示；组合 speaker 和低频角色不会占据默认列表，但仍可通过搜索找到。纯符号、纯编号和脚本占位符会从可选词表中排除。如果必要角色不在预制词表中，应先标记 `needs_review`，随后单独审查并扩充受控词表，而不是在单条标注中临时造词。

## 3. 启动 GUI

从仓库根目录运行：

```bash
python scripts/annotation/annotate_clean_events.py serve --annotator void0312
```

默认地址：

```text
http://127.0.0.1:8765/
```

不自动打开浏览器：

```bash
python scripts/annotation/annotate_clean_events.py serve \
  --annotator void0312 \
  --no-open
```

如果暂时不加载日文/英文原始对齐：

```bash
python scripts/annotation/annotate_clean_events.py serve \
  --annotator void0312 \
  --no-aligned-evidence
```

服务只绑定 `127.0.0.1`，不应直接暴露到公网。

## 4. GUI 工作流

界面分为三栏：

1. 左侧选择 trajectory，并查看每条轨迹的标注覆盖率。
2. 中间查看连续事件流；默认约显示 33 条，可切换为约 17/65/129 条。
3. 右侧填写语义字段，并选择 `验收/待复核/跳过`。

上下文窗口在轨迹开头或结尾会自动向另一侧补足，不再因为缺少前文或后文而只显示半个窗口。事件流区域内部可滚动，切换事件后会自动将当前条目置于可见位置。

日文/中文/英文原始对齐不再作为常驻的“上游证据”区占据主界面。仅当当前事件存在对齐记录时，事件流标题旁会出现“查看原始对齐”按钮；它只用于处理说话人、译文或语义歧义，关闭后返回原标注位置。

点击 `选择 Agent/Target/Participants` 会打开预制选项面板。可以多选、清空或删除已选 chip；被源 speaker 锁定的 Agent 只能查看，不能移除。

快捷键：

- `Ctrl/Command + S`：保存当前事件。
- `Ctrl/Command + Enter`：保存并进入下一事件。
- `Alt + Left/Right`：上一条/下一条。

`needs_review` 必须填写人工备注。人工备注是唯一允许自由输入的标注内容。已经由确定性构造产生的 base flags 只能保留，不能在 GUI 中删除。

## 5. 存储与并发保护

默认审计日志：

```text
data/human_annotations/history.jsonl
```

日志采用 append-only 形式。每次保存包含：

- 唯一 `annotation_id`
- `trajectory/event_id`
- 基础事件 SHA-256
- revision
- annotator
- status
- note
- patch
- 填充结果 SHA-256
- UTC 时间

同一事件在多个浏览器标签页中修改时，旧 revision 会得到 HTTP 409 冲突，GUI 会重新载入最新版本，避免静默覆盖。

记录格式见：

```text
schemas/human_annotation_v0.schema.json
```

## 6. 统计与导出

查看覆盖情况：

```bash
python scripts/annotation/annotate_clean_events.py stats
```

导出最新快照：

```bash
python scripts/annotation/annotate_clean_events.py export
```

GUI 顶部也提供“导出快照”按钮。默认输出：

```text
data/enriched/human_clean_event_v0/
├── annotations.jsonl
├── accepted_events.jsonl
├── needs_review.jsonl
└── summary.json
```

其中：

- `annotations.jsonl`：每个事件的最新人工记录。
- `accepted_events.jsonl`：通过 `clean_event_v0` schema 的人工验收事件；每行同时保留 annotation metadata 和完整 event。
- `needs_review.jsonl`：仍需复核的记录。
- `summary.json`：覆盖率、过期基线和 schema 错误统计。

如果基础事件已经变化，SHA-256 不再匹配，该记录会进入 `stale`，不会被静默导出为可信事件。

## 7. 当前边界

第一版没有提供：

- 多用户远程协作和账号系统。
- 自动仲裁多位标注者的冲突。
- 将未完全标注的 trajectory 直接物化为训练全集。
- 将 LLM 建议作为默认值。

这些限制是有意的。当前目标是先建立可审计、可恢复、不会污染 deterministic canonical 层的人工标注闭环。
