"""UECD backward process (sampling / denoising).

Iteratively denoises from t=T to t=0:
1. Continuous backbone predicts v -> recover predicted eps.
2. Discrete head predicts concrete scores from noise-corrected state.
3. Discrete correction via confidence-remnant blend reversal.
4. DDPM-style continuous step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.losses import Graph, sample_categorical
from model.uecd.confidence import confidence_remnant
from model.uecd.continuous_backbone import ContinuousBackbone
from model.uecd.continuous_noise import ContinuousNoiseSchedule
from model.uecd.discrete_head import DiscreteHead


@dataclass
class SamplingConfig:
    steps: int = 128
    noise_removal: bool = True
    eps: float = 1e-5


@torch.no_grad()
def sample(
    continuous_backbone: ContinuousBackbone,
    discrete_head: DiscreteHead,
    embedding_weight: torch.Tensor,
    continuous_noise: ContinuousNoiseSchedule,
    graph: Graph,
    mask_token_id: int,
    batch_size: int,
    seq_len: int,
    config: SamplingConfig,
    device: torch.device,
    prefix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate token sequences via UECD backward process.

    Args:
        continuous_backbone: The continuous denoiser.
        discrete_head: The discrete denoiser head.
        embedding_weight: Embedding matrix, shape (V, D).
        continuous_noise: Continuous noise schedule.
        graph: SEDD diffusion graph (Absorbing).
        mask_token_id: Index of [MASK] token.
        batch_size: Number of sequences to generate.
        seq_len: Length of generated sequences.
        config: Sampling configuration.
        device: Target device.
        prefix: Optional prefix tokens, shape (B, P). Not diffused.

    Returns:
        Generated token ids, shape (B, L) or (B, P+L) if prefix given.
    """
    dim = embedding_weight.shape[1]

    # Terminal state: all [MASK], with partial signal retention
    x_t = torch.full(
        (batch_size, seq_len), mask_token_id, dtype=torch.long, device=device
    )
    mask_embed = embedding_weight[mask_token_id]  # (D,)

    t_terminal = torch.ones(batch_size, device=device)
    alpha_T, sigma_T = continuous_noise(t_terminal)
    eps_init = torch.randn(batch_size, seq_len, dim, device=device)
    zx_t = (
        alpha_T[:, None, None] * mask_embed[None, None, :]
        + sigma_T[:, None, None] * eps_init
    )

    # Prefix handling
    prefix_len = 0
    if prefix is not None:
        prefix_len = prefix.shape[1]
        prefix_embed = F.embedding(prefix, embedding_weight)

    timesteps = torch.linspace(1.0, config.eps, config.steps + 1, device=device)

    for i in range(config.steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = t_now.expand(batch_size)

        # Prepend prefix for model input
        if prefix is not None:
            model_input = torch.cat([prefix_embed, zx_t], dim=1)
            full_x_t = torch.cat([prefix, x_t], dim=1)
        else:
            model_input = zx_t
            full_x_t = x_t

        # Step 1: Continuous backbone
        backbone_out = continuous_backbone(model_input, t_batch)
        v_pred = backbone_out.v_pred
        tap_hidden = backbone_out.tap_hidden

        # Recover predicted eps
        eps_hat = continuous_noise.predict_eps(model_input, v_pred, t_batch)

        # Strip prefix from outputs for diffusion updates
        if prefix is not None:
            eps_hat_gen = eps_hat[:, prefix_len:]
            tap_hidden_gen = tap_hidden[:, prefix_len:]
            v_pred_gen = v_pred[:, prefix_len:]
        else:
            eps_hat_gen = eps_hat
            tap_hidden_gen = tap_hidden
            v_pred_gen = v_pred

        # Step 2: Discrete head
        noise_corrected = zx_t - eps_hat_gen
        log_score = discrete_head(noise_corrected, tap_hidden_gen, t_batch, x_t)
        score = log_score.exp()

        # Step 3: Discrete update — staggered score + analytic transition
        sigma_disc_now = t_now
        sigma_disc_next = t_next
        dsigma = sigma_disc_now - sigma_disc_next

        # Flatten (B, L) -> (B*L) for SEDD graph operations that expect (B, V)
        B, L, V = score.shape
        score_flat = score.reshape(B * L, V)
        x_t_flat = x_t.reshape(B * L)
        dsigma_flat = dsigma.expand(B * L)

        stag_score = graph.staggered_score(score_flat, dsigma_flat)
        transp = graph.transp_transition(x_t_flat, dsigma_flat[:, None])
        probs = stag_score * transp
        x_new = sample_categorical(probs).reshape(B, L)

        # Step 3b: Confidence remnant for blend correction
        c_hat = confidence_remnant(zx_t, x_t, embedding_weight)
        c_hat_3d = c_hat.unsqueeze(-1)

        # Step 4: Discrete correction in embedding space
        x_t_embed = F.embedding(x_t, embedding_weight)
        x_new_embed = F.embedding(x_new, embedding_weight)
        zx_t = zx_t - eps_hat_gen - c_hat_3d * x_t_embed + c_hat_3d * x_new_embed

        # DDPM-style continuous step to t_next
        t_next_batch = t_next.expand(batch_size)
        alpha_next, sigma_next = continuous_noise(t_next_batch)
        x0_hat = continuous_noise.predict_x0(zx_t + eps_hat_gen, v_pred_gen, t_batch)

        if i < config.steps - 1:
            noise = torch.randn_like(zx_t)
        else:
            noise = torch.zeros_like(zx_t)

        zx_t = alpha_next[:, None, None] * x0_hat + sigma_next[:, None, None] * noise
        x_t = x_new

    # Final noise removal: one more discrete denoising step
    if config.noise_removal:
        if prefix is not None:
            model_input = torch.cat([prefix_embed, zx_t], dim=1)
        else:
            model_input = zx_t

        t_final = timesteps[-1].expand(batch_size)
        backbone_out = continuous_backbone(model_input, t_final)
        eps_hat = continuous_noise.predict_eps(
            model_input, backbone_out.v_pred, t_final
        )

        if prefix is not None:
            eps_hat_gen = eps_hat[:, prefix_len:]
            tap_hidden_gen = backbone_out.tap_hidden[:, prefix_len:]
        else:
            eps_hat_gen = eps_hat
            tap_hidden_gen = backbone_out.tap_hidden

        noise_corrected = zx_t - eps_hat_gen
        log_score = discrete_head(noise_corrected, tap_hidden_gen, t_final, x_t)

        # Denoise: get most likely token
        score = log_score.exp()
        B, L, V = score.shape
        score_flat = score.reshape(B * L, V)
        x_t_flat = x_t.reshape(B * L)
        t_flat = t_final[0].expand(B * L)

        stag_score = graph.staggered_score(score_flat, t_flat)
        transp = graph.transp_transition(x_t_flat, t_flat[:, None])
        probs = stag_score * transp
        if graph.absorb:
            probs = probs[..., :-1]
        x_t = sample_categorical(probs).reshape(B, L)

    if prefix is not None:
        return torch.cat([prefix, x_t], dim=1)
    return x_t
