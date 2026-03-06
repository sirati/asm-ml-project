"""UECD loss function.

Combines v-prediction MSE (continuous) with SEDD denoising score entropy
(discrete), weighted by SNR-based schedule that downweights discrete loss
at high noise where continuous corruption makes discrete predictions unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model.losses import Graph
from model.uecd.continuous_noise import ContinuousNoiseSchedule


@dataclass
class UECDLossResult:
    total: torch.Tensor  # (B,) — weighted combined loss per sample
    continuous: torch.Tensor  # (B,) — v-prediction MSE per sample
    discrete: torch.Tensor  # (B,) — score entropy per sample
    w_cont: torch.Tensor  # (B,) — continuous weight
    w_disc: torch.Tensor  # (B,) — discrete weight


def snr_weights(
    snr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SNR-based loss weighting.

    w_disc(t) = SNR / (1 + SNR)  — downweighted at high noise
    w_cont(t) = 1 / (1 + SNR)   — upweighted at high noise

    At t=0 (high SNR): both ~equal.
    At t=T (low SNR): continuous dominates.
    """
    w_disc = snr / (1.0 + snr)
    w_cont = 1.0 / (1.0 + snr)
    return w_cont, w_disc


def continuous_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    prefix_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """V-prediction MSE loss, averaged over sequence and embedding dims.

    Args:
        v_pred: Predicted v, shape (B, L, D).
        v_target: Target v, shape (B, L, D).
        prefix_mask: Optional bool mask, True for prefix positions to exclude.
            Shape (B, L).

    Returns:
        Per-sample loss, shape (B,).
    """
    mse = (v_pred - v_target).square()  # (B, L, D)
    if prefix_mask is not None:
        mse = mse.masked_fill(prefix_mask.unsqueeze(-1), 0.0)
        num_active = (~prefix_mask).sum(dim=-1).clamp(min=1).float()
        return mse.sum(dim=(-1, -2)) / (num_active * mse.shape[-1])
    return mse.mean(dim=(-1, -2))


def discrete_loss(
    log_score: torch.Tensor,
    graph: Graph,
    sigma: torch.Tensor,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    dsigma: torch.Tensor,
    prefix_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """SEDD denoising score entropy loss (DWDSE weighting).

    Args:
        log_score: Log concrete scores from discrete head, shape (B, L, V).
        graph: Diffusion graph (Absorbing).
        sigma: Discrete noise level, shape (B,).
        x_t: Corrupted discrete tokens, shape (B, L).
        x0: Clean discrete tokens, shape (B, L).
        dsigma: Rate of discrete noise, shape (B,).
        prefix_mask: Optional bool mask for prefix exclusion, shape (B, L).

    Returns:
        Per-sample loss, shape (B,).
    """
    entropy = graph.score_entropy(log_score, sigma[:, None], x_t, x0)  # (B, L)
    if prefix_mask is not None:
        entropy = entropy.masked_fill(prefix_mask, 0.0)
    weighted = dsigma[:, None] * entropy
    return weighted.sum(dim=-1)


def uecd_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    log_score: torch.Tensor,
    graph: Graph,
    continuous_noise: ContinuousNoiseSchedule,
    t: torch.Tensor,
    sigma_disc: torch.Tensor,
    dsigma_disc: torch.Tensor,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    prefix_mask: torch.Tensor | None = None,
) -> UECDLossResult:
    """Compute the full UECD loss.

    Args:
        v_pred: V-prediction from continuous backbone, shape (B, L, D).
        v_target: V-prediction target, shape (B, L, D).
        log_score: Log concrete scores from discrete head, shape (B, L, V).
        graph: SEDD diffusion graph.
        continuous_noise: Continuous noise schedule (for SNR).
        t: Shared timestep, shape (B,).
        sigma_disc: Discrete noise level, shape (B,).
        dsigma_disc: Discrete noise rate, shape (B,).
        x_t: Corrupted discrete tokens, shape (B, L).
        x0: Clean tokens, shape (B, L).
        prefix_mask: Optional bool mask for prefix positions.

    Returns:
        UECDLossResult with per-sample losses and weights.
    """
    snr = continuous_noise.snr(t)
    w_cont, w_disc = snr_weights(snr)

    l_cont = continuous_loss(v_pred, v_target, prefix_mask)
    l_disc = discrete_loss(
        log_score, graph, sigma_disc, x_t, x0, dsigma_disc, prefix_mask
    )

    total = w_cont * l_cont + w_disc * l_disc

    return UECDLossResult(
        total=total,
        continuous=l_cont,
        discrete=l_disc,
        w_cont=w_cont,
        w_disc=w_disc,
    )
