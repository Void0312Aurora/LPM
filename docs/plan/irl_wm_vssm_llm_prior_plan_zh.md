# LPM 理论计划：基于 LLM 先验的 WM/VSSM 辅助 IRL

## 1. 理论命题

LPM 的核心目标不是直接训练一个会续写角色对白的模型，而是在部分可观测叙事中解释角色行为背后的偏好结构。更合适的理论表述是：

$$
\text{LLM-prior-guided IRL over WM/VSSM belief dynamics}
$$

中文表述为：

$$
\text{基于大语言模型先验的世界模型辅助逆强化学习潜人格建模}
$$

其中：

- IRL 给出 persona 的解释对象：角色在状态中的 reward / preference / utility structure。
- WM/VSSM 给出 IRL 的状态空间：从带偏置观测中形成 latent belief dynamics。
- LLM prior 给出小数据条件下的先验约束：语义、动作流形、策略合理性和弱 reward prior。

因此，persona 不应首先被定义为表层语言风格，而应被定义为：

$$
\mathrm{persona}_c
\approx
\left(
R_c(s,a),
\pi_c(a\mid s),
p(s_t\mid s_{t-1},a_{t-1},o_t)
\right)
$$

其中 $R_c$ 表示角色 $c$ 的偏好结构，$\pi_c$ 表示由偏好诱导的行为倾向，$s_t$ 表示由 WM/VSSM 在叙事观测中维持的 belief state。

## 2. 与直接 SFT 的理论差异

Behavior cloning / SFT 学习的是：

$$
P(Y_t\mid X_t,c)
$$

这类模型回答的问题是：

$$
\text{在给定上下文中，角色大概会说什么}
$$

IRL 路线回答的问题是：

$$
\text{在某个 belief state 中，角色为何偏向某类行为}
$$

两者差异不在于是否使用语言模型，而在于解释对象不同。SFT 的目标是拟合观测行为分布；IRL 的目标是从观测行为中反推使这些行为合理的 reward structure。

这一区分对 LPM 很关键。若只做 SFT，模型可能学习到：

- 局部上下文续写。
- 角色口癖和表层风格。
- 场景或路线模板。
- 叙事结构泄漏。

这些都不能说明模型获得了可解释的 latent persona。

## 3. 部分可观测叙事设定

GalGame 文本并不直接暴露真实世界状态或角色心理状态。它暴露的是带视角偏置的观测：

$$
o_t \sim \mathcal{A}_v(S_t)
$$

其中 $S_t$ 是未观测叙事状态，$\mathcal{A}_v$ 是视角相关观测算子。旁白、动作描写、主观解释和直接对白具有不同认识论地位。

因此，LPM 更接近 POMDP / belief modeling，而不是完全可观测序列建模：

$$
o_{1:t}
\rightarrow
s_t
\rightarrow
a_t
\rightarrow
Y_t
$$

这里 $s_t$ 不是心理标签，而是对行为选择和 reward inference 有解释力的 latent belief。

## 4. 两级 WM/VSSM 状态

在低数据条件下，不宜一开始引入过细的层级。但纯角色局部状态也会遗漏叙事进展。因此第一版理论状态采用两级分解：

$$
s_t = (g_t, b_t^c)
$$

其中：

- $g_t$ 是 global narrative state，表示剧情、场景、路线、当前叙事阶段等慢变量。
- $b_t^c$ 是 character belief state，表示角色 $c$ 在当前叙事中形成的局部信念和行为相关状态。

对应的 WM/VSSM 形式可以写作：

$$
g_t \sim p_\theta(g_t\mid g_{t-1}, o_t)
$$

$$
b_t^c \sim p_\theta(b_t^c\mid b_{t-1}^c,g_t,o_t,a_{t-1}^c)
$$

这个分解的理论意义是区分：

- 角色行为受共享剧情状态影响的部分。
- 角色行为受个体偏好与信念影响的部分。

如果不引入 $g_t$，角色 reward 可能被迫解释剧情阶段；如果不引入 $b_t^c$，模型会退化为全局叙事预测，而不是角色建模。

## 5. 动作空间：连续 action embedding 优先

IRL 需要 action，但在自然语言任务中，直接把 token 或完整句子当作 action 会使问题不可控。另一方面，在约 4 万 token 级别的低数据设定下，预先定义完整闭集 action ontology 也不稳定。

因此，第一阶段理论上更合适的是连续 action embedding：

$$
a_t = A_{\mathrm{LLM}}(Y_t)
$$

其中 $A_{\mathrm{LLM}}$ 将自然语言回应 $Y_t$ 映射到低维连续行为空间。这个空间可以理解为由 LLM 语义先验诱导出的行为流形。

连续 action embedding 的理论优势是：

- 避免闭集类别在小数据下定义不全。
- 保留回应之间的语义距离和行为相似性。
- 允许 IRL 在平滑 action manifold 上比较偏好。
- 后续仍可从连续空间聚类或解释出离散行为类别。

这并不否定闭集 action 的价值。闭集类别更适合解释和审计；连续 embedding 更适合作为第一版 IRL 的低数据动作表征。二者的关系应是：

$$
\text{continuous action manifold first}
\rightarrow
\text{interpretable action categories later}
$$

## 6. Reward 分解

基于两级状态，角色 reward 可写成：

$$
R_c(s_t,a_t)
=
R_{\mathrm{shared}}(g_t,a_t)
+
R_{\mathrm{char}}(c,b_t^c,a_t)
+
R_{\mathrm{int}}(g_t,b_t^c,a_t)
$$

其中：

- $R_{\mathrm{shared}}$ 解释一般叙事合理性。
- $R_{\mathrm{char}}$ 解释角色特异偏好。
- $R_{\mathrm{int}}$ 解释角色偏好与当前剧情阶段之间的交互。

这一分解避免把所有行为差异都压进角色人格。某些行为是剧情阶段要求，某些行为是角色稳定倾向，某些行为来自二者交互。

LPM 的潜人格更接近：

$$
\mathrm{persona}_c
\approx
R_{\mathrm{char}}(c,\cdot,\cdot)
+
\text{its interaction with } g_t
$$

而不是整个 $R_c$ 的所有部分。

## 7. IRL 的低数据形式：pairwise reward ranking

完整 MaxEnt IRL 可以写成：

$$
P(\tau_c\mid R_c)
\propto
\exp
\left(
\sum_t R_c(s_t,a_t)
\right)
$$

但在数据极少时，直接估计轨迹分布和 partition function 不稳定。更合理的第一版理论近似是 pairwise reward ranking：

$$
R_c(s_t,a_t^+)
>
R_c(s_t,a_t^-)
$$

其中 $a_t^+$ 是观测到的角色行为，$a_t^-$ 是在同一状态下较不合理的替代行为。

这种形式的理论含义不是“做一个排序器”这么简单，而是把 IRL 从绝对 reward 估计转为偏好不等式约束：

$$
(s_t,a_t^+) \succ_c (s_t,a_t^-)
$$

可构造的负行为包括：

- 同一时刻其他角色的行为。
- 同一角色在错位历史中的行为。
- 未来或过去状态下的回应。
- LLM policy prior 生成但与角色轨迹不一致的候选。

这种 ranking IRL 更符合小数据设定，因为它利用相对偏好而不是完整环境交互。

## 8. LLM 先验的理论位置

LLM prior 不应被当作真值来源，而应作为低数据条件下的先验分布和正则项。

### 8.1 观测先验

LLM 提供：

$$
p(e_t\mid o_t)
$$

用于把文本观测映射为语义表示。但它不能把旁白直接转化为角色心理真值。

### 8.2 动作流形先验

LLM 提供：

$$
p(a_t\mid Y_t)
$$

或等价地提供从语言回应到连续 action embedding 的映射。这是小数据下构造可学习行为空间的关键。

### 8.3 策略先验

LLM 提供初始行为合理性：

$$
\pi_0(a\mid s,c)
$$

它约束模型不要生成语言上荒谬或语义断裂的行为，但不能替代角色特异 reward。

### 8.4 Reward 先验

LLM 可以提供弱 reward prior：

$$
p(R_c)
$$

例如角色一致性、叙事合理性、语义连贯性。该先验只能作为约束，不能作为 reward truth。否则学到的是通用 LLM 叙事偏好，而不是角色偏好。

## 9. WM/VSSM 的理论作用

WM/VSSM 在该路线中不是最终人格模型，而是 IRL 的 belief substrate。

它承担三项理论职责：

1. 将带视角偏置的观测转化为 belief state。
2. 将长程叙事历史压缩为对未来行为有用的状态。
3. 表达同一观测下的潜在不确定性。

因此 WM/VSSM 不应只被评价为 reconstruction model。它的理论要求是：

$$
I(s_t; a_{t:t+k})
\text{ should be high}
$$

同时：

$$
I(s_t; \text{scene id shortcut})
\text{ should be controlled}
$$

也就是说，$s_t$ 应携带对后续行为和 reward inference 有用的信息，而不是只记住场景身份。

## 10. 信息瓶颈与 state bypass

LPM 的关键风险是 state bypass：LLM 或 reward model 只依赖局部文本，绕过 $s_t$。

这不是单纯工程问题，而是理论可辨识性问题。如果局部上下文已经足以解释行为，则 $s_t$ 没有被证明为必要变量。

因此需要在理论目标中加入信息瓶颈思想：

$$
s_t
\text{ should be sufficient for behavior-relevant history}
$$

但：

$$
s_t
\text{ should not be an unconstrained memory dump}
$$

可表达为：

$$
\max I(s_t; a_t, o_{t+1:t+k})
-
\beta I(s_t; o_{1:t})
$$

同时保留局部上下文基线：

$$
R_c(s_t,a_t)
\quad \text{vs.} \quad
R_c(x_t^{\mathrm{local}},a_t)
$$

若引入 $s_t$ 后不能稳定改善相对偏好判断，则 WM/VSSM 对 IRL 没有提供必要状态信息。

## 11. 理论目标

整体理论目标可以写成：

$$
\mathcal{J}
=
\lambda_{\mathrm{rank}}\mathcal{J}_{\mathrm{IRL-rank}}
+
\lambda_{\mathrm{wm}}\mathcal{J}_{\mathrm{belief}}
+
\lambda_{\mathrm{prior}}\mathcal{J}_{\mathrm{LLM-prior}}
+
\lambda_{\mathrm{ib}}\mathcal{J}_{\mathrm{IB}}
+
\lambda_{\mathrm{aux}}\mathcal{J}_{\mathrm{aux}}
$$

其中：

- $\mathcal{J}_{\mathrm{IRL-rank}}$ 表示角色行为相对偏好约束。
- $\mathcal{J}_{\mathrm{belief}}$ 表示 WM/VSSM belief dynamics 的预测与一致性约束。
- $\mathcal{J}_{\mathrm{LLM-prior}}$ 表示语言、动作流形、策略和弱 reward 先验。
- $\mathcal{J}_{\mathrm{IB}}$ 表示防止 state 变成无限记忆或被绕过的信息瓶颈。
- $\mathcal{J}_{\mathrm{aux}}$ 表示辅助可辨识性约束。

这里的目标函数不是实现 recipe，而是表达本路线的理论依赖关系：IRL 是解释目标，WM/VSSM 是状态假设，LLM prior 是低数据先验，信息瓶颈是可辨识性保护。

## 12. 辅助约束的角色

Multi-task / auxiliary prediction 在这里不是主算法，而是帮助状态获得正确归纳偏置。可考虑的辅助理论约束包括：

- 预测未来观测通道分布。
- 预测后续角色行为 embedding。
- 区分直接对白与视角中介旁白。
- 预测分支或选择对后续状态的影响。
- 区分角色特异 reward 与共享叙事 reward。

这些辅助目标的作用不是扩大任务清单，而是减少 reward 和 state 的不可识别性。

## 13. 反事实与对照

为了证明该理论路线有意义，必须比较以下反事实：

| 对照                       | 理论问题                              |
| -------------------------- | ------------------------------------- |
| SFT / behavior cloning     | 表层模仿是否已经足够                  |
| retrieval baseline         | 相似历史检索是否已经解释行为          |
| contrastive representation | 不显式 reward 是否也能完成匹配        |
| VSSM without IRL           | 只学 belief dynamics 是否足够         |
| IRL without VSSM           | 没有状态模型时 reward 是否退化        |
| LLM prior only             | 模型是否只依赖通用 LLM 偏好           |
| wrong global state         | reward 是否依赖剧情阶段               |
| wrong character belief     | reward 是否依赖角色特异状态           |
| shuffled action embedding  | 连续 action manifold 是否携带行为语义 |

这些对照不是工程验证项，而是理论判别：它们回答该路线中的每个变量是否真的承担了不可替代的解释功能。

## 14. 主要理论风险

### 14.1 Reward 不可识别

同一行为可以由多个 reward 解释。pairwise ranking 只能提供相对约束，不能完全消除不可识别性。需要依赖错配状态、错配角色和错配剧情阶段来加强判别。

### 14.2 动作流形不可解释

连续 action embedding 有利于低数据学习，但可能降低可解释性。后续需要检查 embedding 空间是否形成稳定行为方向，而不是仅仅编码表面语义。

### 14.3 全局状态吞噬角色状态

如果 $g_t$ 过强，$R_{\mathrm{char}}$ 可能没有实际贡献；如果 $b_t^c$ 过强，模型又可能把剧情阶段误解释为角色偏好。两级状态需要通过错配实验判别各自作用。

### 14.4 LLM prior 污染 reward

LLM 的通用叙事偏好可能被误认为角色偏好。因此 LLM prior 必须保持弱约束地位，不能直接定义 $R_c$。

### 14.5 旁白误用

旁白是视角中介观测，可以帮助更新 belief，但不能作为角色心理或 reward 的直接标签。

## 15. 与现有文档的关系

`docs/theory_zh.md` 中已有形式：

$$
z_k^c = F_\theta(z_{k-1}^c,E_k^c,u_c)
$$

本计划将其重解释为：

$$
s_t=(g_t,b_t^c)
$$

$$
R_c(s_t,a_t)
\rightarrow
\pi_c(a_t\mid s_t)
\rightarrow
Y_t
$$

即：原先的 latent response process 被提升为 belief-state-conditioned reward / policy system。

`docs/temp/temp-01.md` 的 RSSM/VSSM 结构可作为 belief dynamics 的基础，但应避免把 $z_t$ 直接命名为心理状态。`docs/temp/temp-02.md` 则补充了低数据条件下更合理的理论取向：

- 连续 action embedding 先于闭集 action ontology。
- 状态分解从 global narrative state 与 character belief state 两级开始。
- IRL 先采用 pairwise reward ranking 作为小数据近似。
- state bypass 应通过信息瓶颈与局部上下文对照来处理。

## 16. 当前理论决策

基于上述分析，当前计划采用以下理论默认值：

1. Persona 的核心解释对象是角色 reward / preference structure，而不是语言风格。
2. WM/VSSM 是 IRL 的 belief dynamics，不是最终人格本体。
3. Action 第一阶段采用连续 action embedding，而不是预定义闭集标签。
4. State 第一阶段采用 $s_t=(g_t,b_t^c)$ 两级结构。
5. IRL 第一阶段采用 pairwise reward ranking，而不是完整 MaxEnt / AIRL。
6. LLM prior 是弱先验和归纳偏置，不是 reward 真值。
7. State bypass 是可辨识性问题，需要纳入理论目标，而非只在事后检查。

这些默认值确定的是理论路线。具体数据表示、训练脚本和模型实现应在这些理论对象稳定之后再展开。
