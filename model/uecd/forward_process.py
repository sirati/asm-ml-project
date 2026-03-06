"""UECD forward process: joint continuous-discrete corruption.

Given clean token ids x0, produces corrupted joint state zx_t at timestep t
by applying continuous Gaussian noise and discrete absorbing corruption,
then blending via the confidence remnant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.uecd.confidence import confidence_remnant
from model.uecd.continuous_noise import ContinuousNoiseSchedule


@dataclass
class ForwardResult:
    zx_t: torch.Tensor  # (B, L, D) — joint corrupted state
    z_t: torch.Tensor  # (B, L, D) — continuous-only corrupted state
    x_t: torch.Tensor  # (B, L) — discrete token state (with [MASK])
    eps: torch.Tensor  # (B, L, D) — sampled Gaussian noise
    c_hat: torch.Tensor  # (B, L) — normalized confidence remnant
    x0_embed: torch.Tensor  # (B, L, D) — clean token embeddings


def forward_process(
    x0: torch.Tensor,
    t: torch.Tensor,
    embedding_weight: torch.Tensor,
    continuous_noise: ContinuousNoiseSchedule,
    mask_token_id: int,
) -> ForwardResult:
    """Apply joint continuous-discrete corruption.

    Args:
        x0: Clean token ids, shape (B, L).
        t: Timestep per sample, shape (B,), in [0, 1].
        embedding_weight: Embedding matrix, shape (V, D).
        continuous_noise: Continuous noise schedule.
        mask_token_id: Index of the [MASK] absorbing token.

    Returns:
        ForwardResult with all intermediate states needed for loss computation.
    """
    x0_embed = F.embedding(x0, embedding_weight)  # (B, L, D)

    # Step 1: Continuous corruption
    eps = torch.randn_like(x0_embed)
    z_t = continuous_noise.add_noise(x0_embed, eps, t)  # (B, L, D)

    # Step 2: Discrete corruption (absorbing CTMC)
    # Probability of masking at time t: 1 - exp(-sigma_discrete(t))
    # We use t directly as the masking probability for simplicity,
    # matching the absorbing forward process where mask_prob increases with t.
    mask_prob = t  # (B,)
    mask_draws = torch.rand_like(x0.float()) < mask_prob.unsqueeze(-1)
    x_t = torch.where(mask_draws, mask_token_id, x0)  # (B, L)

    # Step 3: Confidence remnant
    c_hat = confidence_remnant(z_t, x0, embedding_weight)  # (B, L)

    # Step 4: Blend to produce joint state
    # zx_t = z_t - c_hat * embed(x0) + c_hat * embed(x_t)
    x_t_embed = F.embedding(x_t, embedding_weight)  # (B, L, D)
    c_hat_3d = c_hat.unsqueeze(-1)  # (B, L, 1)
    zx_t = z_t - c_hat_3d * x0_embed + c_hat_3d * x_t_embed

    return ForwardResult(
        zx_t=zx_t,
        z_t=z_t,
        x_t=x_t,
        eps=eps,
        c_hat=c_hat,
        x0_embed=x0_embed,
    )
