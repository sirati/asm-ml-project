"""Discrete denoiser head for UECD.

A small transformer (2-4 layers) that receives:
1. The continuous-noise-corrected state (zx_t - predicted_eps) as input.
2. A rich intermediate hidden state from the continuous backbone via
   one cross-attention layer.

Outputs concrete scores (log-ratios) for SEDD-style discrete denoising.
Intentionally tiny to force all reasoning into the continuous backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import (
    ConditionedCrossLayer,
    ConditionedLayer,
    LayerNorm,
    TimestepEmbedder,
)
from model.layers_pkg import (
    LayerBackend,
    LayerConfig,
    make_cross_layer,
    make_self_layer,
)


@dataclass
class DiscreteHeadConfig:
    hidden_size: int = 768
    cond_dim: int = 128
    num_layers: int = 3
    num_tokens: int = 50257
    scale_by_sigma: bool = True
    layer_config: LayerConfig = field(
        default_factory=lambda: LayerConfig(
            backend=LayerBackend.ATTN,
            num_heads=12,
            dropout=0.1,
            widening_factor=4,
        )
    )


class DiscreteHead(nn.Module):
    """Small transformer head for discrete token denoising.

    Architecture:
    - One cross-attention layer attending to the continuous backbone's tap hidden state
    - 2-4 self-attention layers
    - Linear output to vocab logits (concrete scores)
    """

    def __init__(self, config: DiscreteHeadConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        # +1 for [MASK] absorbing token
        vocab_size = config.num_tokens + 1

        self.sigma_map = TimestepEmbedder(config.cond_dim)

        # Cross-attention from discrete input to backbone tap
        self.cross_attn = ConditionedCrossLayer(
            dim,
            config.cond_dim,
            make_cross_layer(dim, dim, config.layer_config),
        )

        # Self-attention layers
        self.self_layers = nn.ModuleList(
            [
                ConditionedLayer(
                    dim,
                    config.cond_dim,
                    make_self_layer(dim, config.layer_config),
                )
                for _ in range(config.num_layers)
            ]
        )

        # Output projection to concrete scores
        self.norm_out = LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)
        self.output_proj.weight.data.zero_()
        self.output_proj.bias.data.zero_()

        self.scale_by_sigma = config.scale_by_sigma
        self.vocab_size = vocab_size

    def forward(
        self,
        noise_corrected: torch.Tensor,
        tap_hidden: torch.Tensor,
        t: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noise_corrected: zx_t - estimated continuous noise, shape (B, L, D).
            tap_hidden: Intermediate hidden from continuous backbone, shape (B, L, D).
            t: Timestep per sample, shape (B,).
            x_t: Current discrete token state (for scatter-zero), shape (B, L).

        Returns:
            Log concrete scores, shape (B, L, V).
        """
        c = F.silu(self.sigma_map(t))

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = self.cross_attn(noise_corrected, tap_hidden, c)

            for layer in self.self_layers:
                x = layer(x, c)

        log_score = self.output_proj(self.norm_out(x))

        if self.scale_by_sigma:
            esigm1_log = (
                torch.where(t < 0.5, torch.expm1(t), t.exp() - 1)
                .log()
                .to(log_score.dtype)[:, None, None]
            )
            log_score = log_score - esigm1_log - np.log(self.vocab_size - 1)

        # Zero out scores for current token (SEDD convention)
        log_score = torch.scatter(
            log_score, -1, x_t[..., None], torch.zeros_like(log_score[..., :1])
        )
        return log_score
