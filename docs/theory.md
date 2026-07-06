# Latent Persona Modeling: Theoretical Notes

## 1. Motivation

The GalGame dataset should not be treated as independent dialogue samples. Its
natural unit is a narrative sequence: short scene-level dialogue trajectories
compose longer route-level story trajectories. These trajectories are mostly
cross-speaker sequences, not single-character monologues.

The target problem is therefore not ordinary supervised fine-tuning:

$$
x_t \rightarrow y_t
$$

The target problem is to learn latent regularities that explain how a character
responds inside an externally driven narrative process. Surface speech style is
only one observed projection of this process.

## 2. Observation Operator

Let the unobserved narrative state be:

$$
S_t
$$

The dataset does not expose $S_t$ directly. It exposes observations:

$$
O_t = \mathcal{A}_v(S_t)
$$

Here $\mathcal{A}_v$ is a viewpoint-dependent observation operator. In this
dataset, the dominant viewpoint is the protagonist's perspective. This matters:
dialogue, narration, actions, and psychological descriptions do not have the
same epistemic status.

For direct dialogue, the observation is close to a character output channel:

$$
O_t^{\mathrm{dialogue}} \approx y_t^c
$$

For narration, action, expression, and psychological description, the observation
is mediated by viewpoint and narrative convention:

$$
O_t^{\mathrm{narration}} =
\mathcal{A}_{\mathrm{protagonist}}(S_t)
$$

This means narration is not a ground-truth psychological label. It is evidence
about latent state under a biased observation channel.

## 3. Dataset As Cross Narrative Sequence

For each scene $s$, define an ordered sequence:

$$
\tau_s = (o_1, o_2, \ldots, o_T)
$$

Each event may include:

$$
o_t = (a_t, y_t, r_t, p_t, b_t, j_t)
$$

where:

- $a_t$ is the speaker or event owner.
- $y_t$ is dialogue or narration text.
- $r_t$ is the record type, such as dialogue, narration, or choice.
- $p_t$ is the position inside the scene.
- $b_t$ is branch or choice information when present.
- $j_t$ is scene jump or route-flow information when present.

Visual and scene resource fields, such as background, sprite, and event CG, are
kept as metadata:

$$
m_t = (\mathrm{scene}_t, \mathrm{bg}_t, \mathrm{sprite}_t, \mathrm{eventcg}_t)
$$

They are not first-order variables in the initial theory. In short scene
trajectories, background and CG often remain constant and can become weak
shortcuts or scene identifiers. They should enter only if later evidence shows
that they add predictive structure beyond text, ordering, and branch flow.

## 4. Character State Under External Input

Because the sequence is cross-speaker, a character's state is not a closed
single-agent process. It is driven by external input from other speakers,
narration, choices, and story events.

For a target character $c$, let:

$$
T_c = \{t \mid a_t = c\}
$$

Let $t_k^c$ be the position of the $k$-th observed output of character $c$.
The external input received between two outputs is:

$$
E_k^c = o_{t_{k-1}^c + 1 : t_k^c - 1}
$$

The observed response is:

$$
Y_k^c = o_{t_k^c}
$$

The latent character state evolves as:

$$
z_k^c = F_\theta(z_{k-1}^c, E_k^c, u_c)
$$

The character response is generated through:

$$
Y_k^c \sim p_\theta(Y \mid E_k^c, z_k^c, u_c)
$$

Here $u_c$ is a slow character factor and $z_k^c$ is a dynamic state. Neither is
pre-assigned to human categories such as emotion, intent, or relationship. Their
meaning should be induced by their usefulness in explaining observed response
behavior.

## 5. What "Persona" Means Here

In this project, persona is not a static prompt, label list, or collection of
speech habits. A more useful operational definition is:

$$
\mathrm{persona}
\approx
\left(u_c, F_\theta, p_\theta(Y \mid E, z, u_c)\right)
$$

That is:

- $u_c$ captures stable cross-context invariance.
- $F_\theta$ captures response-state dynamics under external input.
- $p_\theta$ captures how latent state and input become observable behavior.

This is closer to a response operator than to a style profile.

## 6. Avoided Assumptions

The initial theory avoids several tempting shortcuts:

- It does not divide latent space into predefined psychological subspaces.
- It does not treat narration as direct mind-state supervision.
- It does not require visual resources to be explicit generation conditions.
- It does not assume that local future prediction is valid without external
  input.
- It does not treat random negative samples as reliable counterfactuals.

These restrictions are not aesthetic. They reduce the chance that the model
learns artifacts of annotation, viewpoint, scene identity, or surface style.

## 7. Algorithmic Design Space

The algorithmic design space follows from the formulation above.

### 7.1 Event Encoding

Design an encoder for external input blocks:

$$
e_k^c = \mathrm{Enc}_\phi(E_k^c)
$$

The important choice is how much structure to preserve: speaker order, narration
position, branch boundary, and scene position may be useful. Background and CG
should initially remain optional metadata rather than mandatory conditioning.

### 7.2 State Transition

The core transition is:

$$
z_k^c = F_\theta(z_{k-1}^c, e_k^c, u_c)
$$

Possible transition families include recurrent networks, state-space models,
gated residual updates, or continuous-time dynamics. This choice should be made
after measuring the actual distribution of response gaps and scene lengths.

### 7.3 Response Likelihood

The response objective is:

$$
\mathcal{L}_{\mathrm{resp}}
=
- \sum_{c,k}
\log p_\theta(Y_k^c \mid E_k^c, z_k^c, u_c)
$$

This is not the same as naive next-turn SFT because the response is conditioned
on a persistent state that updates only through the character's externally
observed experience.

### 7.4 Viewpoint-Aware Observation Likelihood

Different observation channels can be modeled with different likelihoods:

$$
p_\theta(O_t \mid S_t)
=
p_\theta(O_t^{\mathrm{dialogue}} \mid S_t)
p_\theta(O_t^{\mathrm{narration}} \mid S_t, v)
p_\theta(O_t^{\mathrm{branch}} \mid S_t)
$$

The key point is not to make this exact factorization final. The key point is
to preserve the asymmetry between direct character output and viewpoint-mediated
narrative observation.

### 7.5 Preventing State Bypass

If a language model can generate from the visible context alone, it may ignore
$z_k^c$. Any downstream architecture must therefore make the latent state
functionally necessary. Candidate mechanisms include:

$$
z_k^c \rightarrow \mathrm{soft\ prompt}
$$

$$
z_k^c \rightarrow \mathrm{adapter\ gate}
$$

$$
z_k^c \rightarrow \Delta W_{\mathrm{LoRA}}
$$

$$
z_k^c \rightarrow \mathrm{cross\ attention\ memory}
$$

The appropriate mechanism is an engineering question, but the theoretical
requirement is clear: latent state must carry information that is not trivially
recoverable from a short text window.

## 8. Immediate Research Questions

The next theoretical and empirical questions are:

1. What is the distribution of $|E_k^c|$ for major characters?
2. How often does a character produce consecutive outputs with nearly empty
   external input?
3. How much narration appears inside $E_k^c$, and how much of it is viewpoint
   mediated?
4. How often do branches alter the later response distribution of the same
   character?
5. Does a persistent state improve held-out scene response modeling beyond a
   short-context baseline?

These questions are meant to guide model design without committing to a final
training recipe too early.

## 9. Boundary With SFT and QLoRA

SFT or QLoRA may still be useful as a carrier mechanism, but they are not the
theory. In LPM, the central object is the latent response process:

$$
(E_k^c, z_{k-1}^c, u_c)
\rightarrow
z_k^c
\rightarrow
Y_k^c
$$

QLoRA becomes relevant only after there is a concrete decision about how
$z_k^c$ conditions the language model. The theoretical work should first decide
what state is being learned and why it cannot collapse into surface imitation.

