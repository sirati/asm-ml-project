"""Perceiver-IO specialist module for edit diffusion.

Adapted from vendor/perceiver-io/perceiver/model/core/modules.py.
Each specialist handles a subset of edit operations (e.g. INSERT-N tokens).
Uses pluggable layer backends via model.layers for both cross-attention
and self-attention, configurable at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from model.layers import (
    LayerBackend,
    LayerConfig,
    make_self_layer,
    make_cross_layer,
)


# ---------------------------------------------------------------------------
# Learnable latent array (from perceiver-io TrainableQueryProvider)
# ---------------------------------------------------------------------------

class TrainableLatentArray(nn.Module):
    """Learnable latent array used as queries in Perceiver-IO encoder."""

    def __init__(self, num_latents: int, num_channels: int, init_scale: float = 0.02):
        super().__init__()
        self.latents = nn.Parameter(torch.empty(num_latents, num_channels))
        with torch.no_grad():
            self.latents.normal_(0.0, init_scale)

    @property
    def num_channels(self) -> int:
        return self.latents.shape[-1]

    def forward(self, batch_size: int) -> torch.Tensor:
        """Returns latent array expanded to batch size: (B, N, D)."""
        return self.latents.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpecialistConfig:
    num_latents: int = 64
    num_latent_channels: int = 256
    num_self_attention_layers: int = 6
    init_scale: float = 0.02

    cross_attn_config: LayerConfig = field(default_factory=lambda: LayerConfig(
        backend=LayerBackend.ATTN, num_heads=4, dropout=0.0, widening_factor=1,
    ))

    self_attn_config: LayerConfig = field(default_factory=lambda: LayerConfig(
        backend=LayerBackend.ATTN, num_heads=4, dropout=0.0, widening_factor=1,
    ))


# ---------------------------------------------------------------------------
# Perceiver-IO Specialist
# ---------------------------------------------------------------------------

class PerceiverSpecialist(nn.Module):
    """Perceiver-IO specialist for handling a class of edit operations.

    Architecture:
    1. Cross-attention: learnable latent queries attend to backbone hidden states
    2. Self-attention: latent representations refine through self-attention layers
    3. Output cross-attention: output queries attend to refined latents

    Layer backends (attn, flash_attn, mamba) are configurable via SpecialistConfig.
    """

    def __init__(
        self,
        config: SpecialistConfig,
        input_channels: int,
        output_channels: Optional[int] = None,
    ):
        super().__init__()
        if output_channels is None:
            output_channels = input_channels

        d_latent = config.num_latent_channels

        self.latent_array = TrainableLatentArray(
            config.num_latents, d_latent, config.init_scale
        )

        # Encoder: cross-attend from latents to input
        self.encoder_cross_attn = make_cross_layer(
            d_q=d_latent, d_kv=input_channels, config=config.cross_attn_config,
        )

        # Self-attention block on latents
        self.self_attn_layers = nn.ModuleList([
            make_self_layer(d_latent, config.self_attn_config)
            for _ in range(config.num_self_attention_layers)
        ])

        # Decoder: cross-attend from output queries back to latents
        self.decoder_cross_attn = make_cross_layer(
            d_q=input_channels, d_kv=d_latent, config=config.cross_attn_config,
        )

        # Project to output dimension if needed
        self.output_proj = (
            nn.Linear(input_channels, output_channels)
            if output_channels != input_channels
            else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Backbone hidden states of shape (B, L, D_input).
            output_queries: Optional query tensor of shape (B, M, D_input).
                If None, uses hidden_states as output queries (same-length output).

        Returns:
            Output tensor of shape (B, M, D_output).
        """
        batch_size = hidden_states.shape[0]

        # Encode: latents attend to input
        latents = self.latent_array(batch_size)
        latents = self.encoder_cross_attn(latents, hidden_states)

        # Self-attend on latents
        for layer in self.self_attn_layers:
            latents = layer(latents)

        # Decode: output queries attend to latents
        if output_queries is None:
            output_queries = hidden_states

        output = self.decoder_cross_attn(output_queries, latents)
        return self.output_proj(output)
