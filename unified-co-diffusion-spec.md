# original pure thoughts:
I think that is a major flaw still, it also that this archetecture requires this dublication of inputs.

i would much rather avoid that there are implicit information flows, and avoid having this kind of cross interaction. I also really like the SEDD paper because they propose a better loss function for discreet diffusion and allow editing that can self correct (i.e. recorrecting an already unmasked token) 

i would propose the following architecture:
apply both noises at once,  while preserving continuous information:
1. we apply the continuous corruption to everywhere including masked positions
2. we know what the discreet token was before corruption, so we perform discreet corruption by first calculating how much of our original token still exists at the position by evaluating the logit (embedding vector dot product then softmax_0) let call this c for confidence remnant, now we corrupt  by
so our process does both both corruptions in parallel i.e. we advance zx_t -> zx_(t+1)
1. z_(t+1) = continuous forward ( zx_t )
2. separately we keep track of x_t corruption x_(t+1) = discreet forward ( x_t ) SEDD style = 1 Hamming Distance corruption
3. get our remaining discreet token confidence remnant as explained above c_t := c(z_(t+1), x_t)
4.  zx_(t+1) = z_(t+1)- c_t * x_t + c_t * x_(t+1)  

as for the backward prediction
1. the continuous diffuser predicts the noise delta ez_t(|zx_(t+1)  := z_(t+1)-> zx_t i.e. its input is zx but it only predict the z noise, not including the discreet corruption 
2. the discreet diffuser predicts the ex_t(x_(t+1)|zx_(t+1) - ez_t) := x_(t+1)  -> x_t i.e. it is given the discreet tokens with a 1-hamming distance corruption, and it is also given the continuous state with the corruption already removed; it is to predict the 1-hamming distance corruption

the continuous diffuser and the discreet diffuser are thereby completely decoupled, but are co-trained. as the discreet diffuser will still discard its calculation intermediate we would not overcome this limitation, HOWEVER we can make it matter a lot less by making the discreet diffuser very small, possibly even a single layer, thereby forcing all reasoning to happen in the continuous space.

yes i explicitly require that the direct token embedding space and the space in which continuous diffusion happens are the same space. i think this is a desirable property.

on the other hand you are correct that some contextualization is required, i.e. positional embeddings. And the continuous noise must somehow not destroy the positional embeddings. luckily this is easily solvable as the continuous diffusion backwards pass predicts the noise delta and not the clear state. therefore we can just add the contextualization onto the input i.e. the diffusion process happens in a space without contextualization that is required for attention to work,while the model gets this required contextualization. This works because we actually never lose the contextualization. in the diffusion space the whole space is diffused and not on a per-token basis, so the index preserves ordering information, while for the model which treats the space on a per-token basis the positional embeddings provide the contextualization so that attention does not become order invariant. 

Step 4 Issue: This treats embeddings as additive
This is a important point that I need to solve, but not based on the issues that you are bringing up. position embeddings are added onto token-embeddings, thereby we can switch our the embedded token by substracting it and adding another
The real issue here is Token meaning overlap. I know how to solve it though: We have to normalize the token's confidence remnant. the normalisation constant for this is to embed the pure token, and immediately perform the softmax_0 logit on it. this provides us with the maximal value that confidence remnant can ever take for this token, thereby solving it. 
Learning blending function cannot work, as the best strategy is to not blend at all stopping discreet corruption. 

Problem: You're asking it to predict only the Gaussian noise component, but the input zx_(t+1) has both continuous noise AND discrete corruption embedded (via your step 4).
that is a  none issue, ML models are build to handle such complex tasks. 

yes i did not write the formula for the reversing the discrete corruption based on the prediction, but you just do the reverse of the corruption, you do have to take the confidence remnant  of the to be replaced corrupted token (before correcting the continuous corruption), but as the corruption process sets this to be the same i.e. see zx_(t+1) = z_(t+1) - c_t * x_t + c_t * x_(t+1) both here are c_t , so we can recover c_t from the corrupted token, because of our normalization fix earlier. 

For your suggestions:
Option A is out of the window as its exactly what I want to avoid

As to my explanation we make the Discrete Head extremely small! however importantly we must provide it not only with the noise removed z_(t-1) but we should also provide it with a rich internal state of the continuous diffuser to encourage the computation for both to happen there. i.e. not the output layer as there we are reduced to predicting the correction noise, but maybe 2 or 3 layers deeper. with other words, we perform cross attention onto this hidden state once.

also remember: the space in which diffusion happens in tied to the token embedding space.
also all layers of the diffuser have access to the continuous corruption t. 
further the model should support conditioning. e.g. text coming before that is fixed and not diffused on 

1. sorry I mean softmax_1 see: https://www.evanmiller.org/attention-is-off-by-one.html i.e. that attention can choose to not attend anything without requiring a dummy token
2. you got it
3. i thought SEDD main breakthrough is that with hamming dist-1 one can use the score loss, so if so of course we need to use that
4. should be state of the art for continuus diffusion. importantly as we diffuse inside of same latent space as the token encoding maybe its better to predict the denoised result after undoing the timestep rather than the error delta. of course doing the error delta would be okay as well, and should be done if science has shown that predicting the error delta achieves better result. in that case we probably need to let the discrete head also attend the first layer of the continuous one, as otherwise in higher layers the space is already no longer containing state information
5. i would assume that at very high rate of continuous diffusion noise the discrete denoiser has a much harder time, so we should adjust the noise ratio depending on T shouldnt we? i would assume that near t=0  a good default for the hyperparameters should be that they are equally weighted. for large T the change should be based on information theory
6.  
7. how much full noise is may be a hyperparameter, but indeed the standard would be full gaussian 
8. however given that we do a mixed approach and start with all maxed, probably the full noise case is all mask's embedding with a lot of noise added. maybe e.g. alpha = 0.8, so out max continuous noise in that case would indeed be only alpha 0.8. so to answer your question one could say we are doing absorbing and uniform at the same time, but uniform in latent space and absorbing in discrete space
9. it should be a suitable size, similarily how in transformers its now known that nearly all work can be done in the encoder, and the decoder only having e.g. 2 - 4 layers
10. probably in training we always do one-one. this is mostly relevant for inference only, so we do not need to care so much

# Unified Embedding-Space Co-Diffusion (UECD)

## Motivation & Problem Statement

Current co-discrete-continuous diffusion (CCDD, [arXiv:2510.03206](https://arxiv.org/abs/2510.03206)) suffers from three architectural flaws:

1. **Input duplication.** The model receives both discrete tokens x_t and continuous representations z_t as separate input streams, roughly doubling the input. These are informationally redundant at t=0 (z_0 is just the embedding of x_0) yet diverge under independent noise schedules, creating an implicit coupling problem.

2. **Implicit cross-modal information flow.** CCDD's "factored but conditioned" reverse process relies on the shared transformer trunk to *implicitly* learn when and how to transfer information between modalities. The architecture (MDiT/MMDiT/MoEDiT variants) provides no structural guarantee about the nature of this coupling.

3. **Separate representation spaces.** CCDD targets a *pretrained external embedding space* (e.g. Qwen3-Embedding) that is distinct from the model's own token embedding space. This introduces a decoding gap: continuous outputs must be mapped back to tokens through an external projection, adding fragility.

UECD eliminates all three issues by operating entirely within the token embedding space and structurally decoupling the continuous and discrete denoisers while co-training them.

---

## Foundation: SEDD

Score Entropy Discrete Diffusion ([arXiv:2310.16834](https://arxiv.org/abs/2310.16834), ICML 2024 Oral, [code](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)) provides the discrete diffusion foundation.

### Concrete Score

For sequences x = x¹…xᵈ ∈ {1,…,n}ᵈ, the concrete score is the ratio of marginal probabilities between states differing by Hamming distance 1:

$$
s_\theta(x^1 \dots x^i \dots x^d, t)_{i, \hat{x}^i} \approx \frac{p_t(x^1 \dots \hat{x}^i \dots x^d)}{p_t(x^1 \dots x^i \dots x^d)}
$$

### Score Entropy Loss

The score entropy replaces the L2 loss of concrete score matching with a Bregman divergence that naturally enforces positivity of the ratios:

$$
\mathcal{L}_{\mathrm{SE}} = \mathbb{E}_{x \sim p}\left[\sum_{y \neq x} w_{xy}\left(s_\theta(x)_y - \frac{p(y)}{p(x)} \log s_\theta(x)_y + K\!\left(\frac{p(y)}{p(x)}\right)\right)\right]
$$

where K(a) = a(log a − 1) is a normalising constant ensuring L_SE ≥ 0.

The practically used form is the **denoising score entropy** (Theorem 3.4 in SEDD), which replaces the intractable marginal ratios p(y)/p(x) with the tractable transition density ratios p(y|x₀)/p(x|x₀):

$$
\mathcal{L}_{\mathrm{DSE}} = \mathbb{E}_{\substack{x_0 \sim p_0 \\ x \sim p(\cdot|x_0)}}\left[\sum_{y \neq x} w_{xy}\left(s_\theta(x)_y - \frac{p(y|x_0)}{p(x|x_0)} \log s_\theta(x)_y\right)\right]
$$

### Diffusion-Weighted Form (ELBO)

Weighting by the forward diffusion rate matrix Q_t yields the DWDSE, which provides a valid ELBO:

$$
\mathcal{L}_{\mathrm{DWDSE}}(x_0) = \int_0^T \mathbb{E}_{x_t \sim p_{t|0}(\cdot|x_0)} \sum_{y \neq x_t} Q_t(x_t, y) \left(s_\theta(x_t,t)_y - \frac{p_{t|0}(y|x_0)}{p_{t|0}(x_t|x_0)} \log s_\theta(x_t,t)_y + K(\cdots)\right) dt
$$

### Key SEDD Properties

- The forward process corrupts tokens independently per position using a rate matrix Q_t^tok.
- Under independent per-position corruption, only Hamming-distance-1 ratios are needed.
- Absorbing (mask-based) corruption empirically outperforms uniform corruption.
- The Tweedie τ-leaping sampler enables self-correction: already-unmasked tokens can be re-corrupted and re-predicted, unlike purely absorbing approaches.
- The concrete score parameterisation naturally supports arbitrary prompting and infilling.

---

## UECD Architecture

### Core Principle

**The token embedding space and the continuous diffusion space are the same space.** Every position in the sequence holds a single vector in ℝᵈ_model, which simultaneously carries continuous state and discrete token identity.

### Notation

| Symbol | Meaning |
|--------|---------|
| x_t | Discrete token sequence at time t (bookkeeping) |
| z_t | Continuous state at time t (embedding vectors with Gaussian noise) |
| zx_t | Joint state = the actual vectors in embedding space at time t |
| embed(x) | Token embedding lookup (no positional encoding) |
| c_t | Confidence remnant: how much of a token's identity remains readable |
| e_z | Predicted continuous noise/velocity |
| α_t, σ_t | Noise schedule parameters for continuous diffusion |

---

### Forward Process (Corruption)

Given joint state zx_t, advancing to zx_{t+1}:

**Step 1 — Continuous corruption (all positions):**

$$
z_{t+1} = \text{continuous\_forward}(zx_t)
$$

Standard Gaussian noise addition applied to the entire embedding tensor, including positions whose discrete token has already been masked.

**Step 2 — Discrete corruption (tracked separately):**

$$
x_{t+1} = \text{discrete\_forward}(x_t)
$$

SEDD-style CTMC with absorbing state ([MASK] token). One position may flip x_t^i → [MASK] per step (1-Hamming-distance corruption).

**Step 3 — Confidence remnant:**

Compute how much of the original token x_t^i is still readable from the continuously-corrupted vector z_{t+1}^i:

$$
\text{logits}_y = \text{embed}(y) \cdot z_{t+1}^i \quad \forall y \in \text{vocab}
$$

$$
c_t^i = \text{softmax}_1(\text{logits})_{x_t^i}
$$

where softmax₁ is the off-by-one softmax ([Evan Miller, 2023](https://www.evanmiller.org/attention-is-off-by-one.html)):

$$
\text{softmax}_1(\text{logits})_j = \frac{\exp(\text{logits}_j)}{1 + \sum_k \exp(\text{logits}_k)}
$$

This allows c_t to reach near-zero when the token identity is truly destroyed, without being forced to assign mass to some token.

**Normalisation fix for token meaning overlap:**

Raw c_t is not directly comparable across tokens because embeddings are non-orthogonal. Normalise by the self-confidence of a pure token:

$$
c_{\max}(x) = \text{softmax}_1\!\big(\text{embed}(\cdot) \cdot \text{embed}(x)\big)_x
$$

$$
\hat{c}_t^i = \frac{c_t^i}{c_{\max}(x_t^i)}
$$

This provides the maximal value c can ever take for token x, making ĉ_t a properly normalised "fraction of token identity remaining."

**Step 4 — Blend to produce joint state:**

$$
zx_{t+1}^i = z_{t+1}^i - \hat{c}_t^i \cdot \text{embed}(x_t^i) + \hat{c}_t^i \cdot \text{embed}(x_{t+1}^i)
$$

At positions where no discrete corruption occurred (x_{t+1}^i = x_t^i), this reduces to zx_{t+1}^i = z_{t+1}^i (no blending). At positions where a token was replaced, the old token's contribution is surgically swapped for the new token's, scaled by exactly how readable the old identity still was.

---

### Terminal State (t = T)

- **Discrete:** All positions are [MASK] (absorbing).
- **Continuous:** embed([MASK]) · α_T + σ_T · ε, where ε ~ N(0, I).
- The continuous noise at maximum corruption does not go to pure Gaussian. A hyperparameter α controls the signal retention, e.g. α_T ≈ 0.2, meaning ~20% of the [MASK] embedding signal is retained. This provides a stable anchor for the continuous denoiser's starting point.

---

### Backward Process (Denoising)

Going from t+1 → t:

**Step 1 — Continuous denoiser (large backbone):**

The continuous denoiser sees the full corrupted state zx_{t+1} and predicts the continuous corruption. All layers receive the continuous timestep t.

The prediction target is v-prediction ([Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512)):

$$
v_t = \alpha_t \cdot \varepsilon - \sigma_t \cdot x_0
$$

$$
\mathcal{L}_{\text{cont}} = \| v_t - v_\theta(zx_{t+1}, t) \|^2
$$

From v, both x₀ and ε can be recovered:

$$
\hat{x}_0 = \alpha_t \cdot zx_{t+1} - \sigma_t \cdot v_\theta
$$

$$
\hat{\varepsilon} = \alpha_t \cdot v_\theta + \sigma_t \cdot zx_{t+1}
$$

v-prediction provides balanced gradient magnitudes across timesteps — it behaves like ε-prediction at high noise (where predicting noise is easier) and like x₀-prediction at low noise (where predicting the clean state is easier).

**Step 2 — Discrete denoiser (small head):**

The discrete denoiser receives:

1. The continuous-noise-corrected state: zx_{t+1} − ê_z (i.e., the state with continuous corruption already removed, still containing discrete corruption).
2. A rich intermediate hidden state from the continuous backbone via **one cross-attention layer**. This hidden state is taken from 2–3 layers before the continuous backbone's output, where representations are still semantically rich (not yet reduced to noise-prediction space).

The discrete head is intentionally small: 2–4 transformer layers, analogous to how modern encoder-decoder transformers push nearly all computation into the encoder. This forces all reasoning to happen in the continuous backbone's latent space.

The discrete head outputs concrete scores and is trained with the SEDD denoising score entropy loss (see Foundation section above).

**Step 3 — Recovering c_t for discrete correction:**

To undo the blend from forward step 4, we need ĉ_t. Since the normalisation fix makes ĉ_t deterministically computable from (zx_{t+1}, x_{t+1}), and x_{t+1} is known, ĉ_t can be recovered by running the same confidence-remnant computation on the corrupted token.

**Step 4 — Apply discrete correction:**

$$
zx_t = (zx_{t+1} - \hat{e}_z) - \hat{c}_t \cdot \text{embed}(x_{t+1}) + \hat{c}_t \cdot \text{embed}(\hat{x}_t)
$$

where x̂_t is the discrete head's prediction (the corrected token).

---

### Positional Embeddings

The diffusion process operates in embedding space **without** positional embeddings baked in. Positional embeddings (e.g. RoPE) are added as contextualisation at the model's input, not as part of the diffused state.

This works because:

- In the diffusion space, position is preserved by sequence index — it is a sequence, not a set.
- The model requires positional embeddings so that attention is not order-invariant.
- The continuous denoiser predicts the noise/velocity delta, never the full clean state with positional context. Therefore positional embeddings are never targets and never get corrupted.
- The diffusion process is over the *whole* embedding tensor (not per-token independently for the continuous part), so spatial ordering is inherently preserved.

---

### Conditioning (Prefix / Fixed Context)

For conditional generation (e.g. text completion), a fixed prefix is provided:

- Prefix token embeddings are concatenated to the front of the sequence.
- They are **not diffused**: no continuous noise, no discrete corruption.
- They participate in attention (the backbone sees them as context), but their embeddings are clamped.
- At each denoising step, only the non-prefix positions are updated.

This is analogous to SEDD's arbitrary prompting/infilling via clamping.

---

### Loss Function

The total loss is a timestep-dependent weighted sum:

$$
\mathcal{L} = w_{\text{cont}}(t) \cdot \mathcal{L}_{\text{cont}} + w_{\text{disc}}(t) \cdot \mathcal{L}_{\text{disc}}
$$

where L_cont is the v-prediction MSE and L_disc is the SEDD denoising score entropy.

**Weighting schedule:**

Near t = 0 (low noise), both losses are equally weighted. As t increases, the continuous noise increasingly destroys the signal the discrete head depends on, so the discrete loss should be downweighted:

$$
w_{\text{disc}}(t) \propto \frac{\text{SNR}(t)}{1 + \text{SNR}(t)}, \qquad w_{\text{cont}}(t) \propto \frac{1}{1 + \text{SNR}(t)}
$$

where SNR(t) = α_t² / σ_t² is the signal-to-noise ratio of the continuous process. This naturally downweights discrete gradients when continuous noise makes them uninformative.

---

### Training Regime

- Both noise processes share the same timestep t (single t sampled per training step).
- Continuous and discrete noise schedules may have different rate functions mapping t to their respective noise levels, but are synchronised in time.
- In training, the noise schedule coupling is one-to-one. Asynchronous schedules are an inference-time concern only.

---

## Summary of Key Tricks

1. **Shared embedding space.** Token embeddings = continuous diffusion space. No external encoder, no decoding gap.

2. **softmax₁ confidence remnant.** Using the off-by-one softmax allows the model to express "no token identity remains here" (c → 0) without requiring a dummy token. Pure softmax would always force probability mass onto some token.

3. **Self-confidence normalisation.** Dividing c_t by c_max(x_t) compensates for non-orthogonal token embeddings. Without this, high-overlap tokens (e.g. synonyms with similar embeddings) would have artificially deflated confidence remnants.

4. **Surgical blending (Step 4).** The formula zx = z − c·embed(x_old) + c·embed(x_new) performs an exact token-identity swap scaled by readability. It is a no-op when no discrete corruption occurs and fully replaces identity when c is at maximum.

5. **Decoupled but co-trained.** The continuous and discrete denoisers have no weight sharing and no bidirectional information flow. Information flows one way: continuous backbone → (via cross-attention from an intermediate layer) → discrete head. This avoids the implicit coupling problem of CCDD while still allowing the discrete head to leverage the backbone's reasoning.

6. **Tiny discrete head.** By making the discrete head 2–4 layers (vs. the full backbone), essentially all reasoning is forced into continuous space. The discrete head becomes a thin token-identity decoder, not a reasoning engine.

7. **Cross-attention tap point.** The discrete head attends to a hidden state 2–3 layers before the backbone's output. At the output layer, representations are already reduced to noise/velocity predictions. Deeper layers retain richer semantic structure.

8. **Positional embeddings as input-only contextualisation.** They are added to model input, not to the diffused state. The noise prediction never includes positional components. This cleanly separates the diffusion target from the contextualisation needed for attention.

9. **Partial signal retention at t=T.** The terminal state is not pure Gaussian but retains ~20% of embed([MASK]) signal (α_T ≈ 0.2). This gives the continuous denoiser a stable anchor to begin generation from.

10. **SNR-based loss weighting.** The discrete loss is downweighted at high noise levels where continuous corruption makes discrete predictions unreliable. Equal weighting at t=0 transitions to continuous-dominated weighting at t=T.

---

## References

| Paper | Link |
|-------|------|
| SEDD: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (Lou, Meng, Ermon, 2023) | [arXiv:2310.16834](https://arxiv.org/abs/2310.16834) |
| CCDD: Coevolutionary Continuous Discrete Diffusion (Zhou et al., 2025) | [arXiv:2510.03206](https://arxiv.org/abs/2510.03206) |
| v-prediction: Progressive Distillation for Fast Sampling (Salimans & Ho, 2022) | [arXiv:2202.00512](https://arxiv.org/abs/2202.00512) |
| softmax₁ / Attention Is Off By One (Evan Miller, 2023) | [evanmiller.org](https://www.evanmiller.org/attention-is-off-by-one.html) |
| RADD: Your Absorbing Discrete Diffusion Secretly Models Conditional Distributions (2024) | [arXiv:2406.03736](https://arxiv.org/abs/2406.03736) |
| MDLM: Simple and Effective Masked Diffusion Language Models (Sahoo et al., 2024) | [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) |
| Zero Terminal SNR / Common Noise Schedules Are Flawed (Lin et al., 2024) | [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf) |
