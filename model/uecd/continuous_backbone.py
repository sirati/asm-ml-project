"""Continuous denoiser backbone for UECD.

Takes continuous joint state zx_t (embedding vectors) as input and predicts
the v-target for continuous denoising. Exposes an intermediate hidden state
(tap point) for the discrete head's cross-attention.

Key differences from model.backbone.DiffusionBackbone:
- Input is continuous embedding vectors, not discrete token indices.
- No output logits/score — outputs v-prediction in embedding space.
- Exposes a hidden state tap point 2-3 layers before output.
- Positional embeddings are input-only contextualisation (not diffused).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import (
    ConditionedLayer,
    FinalLayer,
    LayerNorm,
    TimestepEmbedder,
)
from model.layers_pkg import LayerBackend, LayerConfig, make_self_layer


@dataclass
class ContinuousBackboneConfig:
    hidden_size: int = 768
    cond_dim: int = 128
    num_layers: int = 12
    tap_offset: int = 3
    layer_config: LayerConfig = field(
        default_factory=lambda: LayerConfig(
            backend=LayerBackend.ATTN,
            num_heads=12,
            dropout=0.1,
            widening_factor=4,
        )
    )


@dataclass
class ContinuousBackboneOutput:
    v_pred: torch.Tensor  # (B, L, D) — v-prediction
    tap_hidden: torch.Tensor  # (B, L, D) — intermediate hidden for discrete head


class ContinuousBackbone(nn.Module):
    """Transformer backbone for continuous v-prediction denoising.

    The tap point is at layer (num_layers - tap_offset), providing
    semantically rich representations before they are reduced to
    noise/velocity predictions in the final layers.
    """

    def __init__(self, config: ContinuousBackboneConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.tap_layer_idx = config.num_layers - config.tap_offset

        self.input_proj = nn.Linear(dim, dim)
        self.sigma_map = TimestepEmbedder(config.cond_dim)

        self.layers = nn.ModuleList(
            [
                ConditionedLayer(
                    dim,
                    config.cond_dim,
                    make_self_layer(dim, config.layer_config),
                )
                for _ in range(config.num_layers)
            ]
        )

        self.output_layer = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        # Zero-init output for stable training start
        self.output_layer[-1].weight.data.zero_()
        self.output_layer[-1].bias.data.zero_()

    def forward(self, zx_t: torch.Tensor, t: torch.Tensor) -> ContinuousBackboneOutput:
        """
        Args:
            zx_t: Joint corrupted state, shape (B, L, D).
            t: Timestep per sample, shape (B,).

        Returns:
            ContinuousBackboneOutput with v_pred and tap_hidden.
        """
        x = self.input_proj(zx_t)
        c = F.silu(self.sigma_map(t))

        tap_hidden = None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i, layer in enumerate(self.layers):
                x = layer(x, c)
                if i == self.tap_layer_idx:
                    tap_hidden = x

        v_pred = self.output_layer(x)

        return ContinuousBackboneOutput(
            v_pred=v_pred,
            tap_hidden=tap_hidden,
        )
