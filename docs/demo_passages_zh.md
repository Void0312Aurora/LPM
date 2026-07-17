# 示范段落抽取与 LLM 辅助标注

## 1. 目的

LPM 需要一组可反复检查的示范段落，用于验证事件 schema、观测算子假设、角色输入-回应单元、分支处理和后续模型行为。示范段落不是训练全集，也不是人工挑选的“好看文本”。它应覆盖数据结构中的典型现象和边界情况。

示范段落应来自原始观测数据，不能由 LLM 改写或补写。

## 2. 数据层级

原始数据保持不可变：

$$
D_{\mathrm{raw}}
$$

示范段落是派生视图：

$$
D_{\mathrm{raw}}
\rightarrow
D_{\mathrm{demo}}
$$

LLM 可以参与的是对 $D_{\mathrm{demo}}$ 的结构化评审，而不是生成或改写 $D_{\mathrm{raw}}$ 的内容。

## 3. 候选段落类型

第一版抽取脚本覆盖以下候选类型：

- `choice_window`：包含选择支的窗口，用于测试分支/路径条件。
- `narration_mediated`：旁白占比较高的窗口，用于测试视角观测算子。
- `multi_speaker_dialogue`：多说话人对话密集窗口，用于测试交叉序列建模。
- `consecutive_same_speaker`：连续发言窗口，用于测试空外部输入或弱外部输入情况。
- `long_scene_middle`：长场景中的中段窗口，用于测试长轨迹切片。
- `target_response_unit`：主要角色在外部输入之后回应的窗口，用于测试 $E_k^c \rightarrow Y_k^c$ 单元。

## 4. DeepSeek 标注边界

`Xi/LPM/.env` 中的 API key 可用于 DeepSeek 调用。脚本默认读取：

- `DEEPSEEK_API_KEY`
- `API_KEY`

默认模型为：

$$
\mathrm{deepseek\text{-}v4\text{-}flash}
$$

也可切换为：

$$
\mathrm{deepseek\text{-}v4\text{-}pro}
$$

LLM 标注只允许输出结构化评审，例如：

- 段落是否适合做 LPM schema 测试。
- 适合测试哪些理论点。
- 是否存在视角中介。
- 是否存在分支、连续发言、旁白主导等结构特征。
- 作为示范段落的质量评分。

LLM 不应改写原文、不应补充剧情、不应把旁白当作心理真值。

## 5. 使用方式

只抽取候选段落，不调用 LLM：

$$
\texttt{python Xi/LPM/scripts/demo/extract_demo_passages.py}
$$

调用 DeepSeek 对前 8 条候选做结构化标注：

$$
\texttt{python Xi/LPM/scripts/demo/extract\_demo\_passages.py --annotate --llm-limit 8}
$$

使用更强模型：

$$
\texttt{python Xi/LPM/scripts/demo/extract\_demo\_passages.py --annotate --model deepseek-v4-pro}
$$

输出默认位于：

$$
\texttt{Xi/LPM/data/demo\_passages/}
$$

其中：

- `candidates.jsonl`：确定性抽取候选。
- `annotations.jsonl`：可选 LLM 标注结果。

## 6. 审计要求

每条候选和标注都必须保留来源信息：

- `scene`
- `start_pos`
- `end_pos`
- `candidate_type`
- `selection_reason`
- `metrics`
- `source_dataset`

后续若将这些段落用于测试集，应固定版本，不随训练脚本动态重采样。
