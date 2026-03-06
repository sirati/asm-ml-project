"""Shared encoder-decoder backbone for edit diffusion.

Adapted from vendor/sedd/model/transformer.py and vendor/sedd/model/rotary.py.
Key changes from original SEDD:
- Encoder/decoder split with configurable asymmetric depths
- Decoder cross-attends to top N layers of encoder
- Pluggable layer backends via model.layers
- Removes HuggingFace Hub mixin, OmegaConf dependency
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers_pkg import (
    CrossSequenceLayer,
    LayerBackend,
    LayerConfig,
    SequenceLayer,
    make_cross_layer,
    make_self_layer,
)

# ---------------------------------------------------------------------------
# Embeddings and normalization
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP embedding (from SEDD)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: float = 10000.0
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


# ---------------------------------------------------------------------------
# Conditioning injection layer
# ---------------------------------------------------------------------------


class ConditionedLayer(nn.Module):
    """Wraps a SequenceLayer with adaptive layer norm conditioning.

    Injects a conditioning vector (e.g. timestep embedding) via
    AdaLN-style shift/scale/gate modulation, as in SEDD's DDiTBlock.
    """

    def __init__(self, dim: int, cond_dim: int, layer: SequenceLayer):
        super().__init__()
        self.layer = layer
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x_mod = self.norm(x) * (1 + scale) + shift
        return x + self.layer(x_mod)


class ConditionedCrossLayer(nn.Module):
    """Wraps a CrossSequenceLayer with conditioning on the query side."""

    def __init__(self, d_q: int, cond_dim: int, layer: CrossSequenceLayer):
        super().__init__()
        self.layer = layer
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * d_q)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
        self.norm = LayerNorm(d_q)

    def forward(
        self, x_q: torch.Tensor, x_kv: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        q_mod = self.norm(x_q) * (1 + scale) + shift
        return x_q + self.layer(q_mod, x_kv)


# ---------------------------------------------------------------------------
# Output layer
# ---------------------------------------------------------------------------


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    hidden_size: int = 768
    cond_dim: int = 128
    num_tokens: int = 50257
    graph_type: str = "absorb"
    scale_by_sigma: bool = True

    # Encoder
    encoder_layers: int = 10
    encoder_layer_config: LayerConfig = field(
        default_factory=lambda: LayerConfig(
            backend=LayerBackend.ATTN,
            num_heads=12,
            dropout=0.1,
            widening_factor=4,
        )
    )

    # Decoder
    decoder_layers: int = 4
    decoder_layer_config: LayerConfig = field(
        default_factory=lambda: LayerConfig(
            backend=LayerBackend.ATTN,
            num_heads=12,
            dropout=0.1,
            widening_factor=4,
        )
    )

    # How many top encoder layers the decoder cross-attends to.
    # e.g. encoder_cross_layers=2 means the bottom 2 decoder layers
    # each cross-attend to one of the top 2 encoder layer outputs.
    encoder_cross_layers: int = 2


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------


class DiffusionBackbone(nn.Module):
    """Encoder-decoder backbone for edit diffusion.

    Architecture:
    - Encoder: N self-attention layers processing the input sequence
    - Decoder: M self-attention layers, where the bottom `encoder_cross_layers`
      layers each cross-attend to the output of one of the top encoder layers.

    Example with encoder_layers=10, decoder_layers=4, encoder_cross_layers=2:
      Encoder layers 1..10 produce hidden states.
      Decoder layer 1: self-attn + cross-attn to encoder layer 9 output
      Decoder layer 2: self-attn + cross-attn to encoder layer 10 output
      Decoder layer 3: self-attn only
      Decoder layer 4: self-attn only
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size

        absorb = config.graph_type == "absorb"
        vocab_size = config.num_tokens + (1 if absorb else 0)
        self.absorb = absorb

        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.sigma_map = TimestepEmbedder(config.cond_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                ConditionedLayer(
                    dim,
                    config.cond_dim,
                    make_self_layer(dim, config.encoder_layer_config),
                )
                for _ in range(config.encoder_layers)
            ]
        )

        # Decoder self-attention layers
        self.decoder_self_layers = nn.ModuleList(
            [
                ConditionedLayer(
                    dim,
                    config.cond_dim,
                    make_self_layer(dim, config.decoder_layer_config),
                )
                for _ in range(config.decoder_layers)
            ]
        )

        # Decoder cross-attention layers (for the bottom `encoder_cross_layers` decoder layers)
        n_cross = min(
            config.encoder_cross_layers, config.decoder_layers, config.encoder_layers
        )
        self.decoder_cross_layers = nn.ModuleList(
            [
                ConditionedCrossLayer(
                    dim,
                    config.cond_dim,
                    make_cross_layer(dim, dim, config.decoder_layer_config),
                )
                for _ in range(n_cross)
            ]
        )
        self.n_cross = n_cross

        self.output_layer = FinalLayer(dim, vocab_size, config.cond_dim)
        self.scale_by_sigma = config.scale_by_sigma

    def _encode(self, x: torch.Tensor, c: torch.Tensor) -> list[torch.Tensor]:
        """Run encoder and return hidden states from all layers."""
        all_hidden = []
        for layer in self.encoder_layers:
            x = layer(x, c)
            all_hidden.append(x)
        return all_hidden

    def _decode(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        encoder_hidden: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run decoder with cross-attention to top encoder layers."""
        # The top `n_cross` encoder outputs, ordered from deepest to shallowest
        top_encoder = encoder_hidden[-self.n_cross :]

        for i, self_layer in enumerate(self.decoder_self_layers):
            x = self_layer(x, c)
            if i < self.n_cross:
                x = self.decoder_cross_layers[i](x, top_encoder[i], c)
        return x

    def forward(self, indices: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Full forward pass: embed -> encode -> decode -> output logits."""
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder_hidden = self._encode(x, c)
            x_dec = self._decode(x, c, encoder_hidden)
            x_out = self.output_layer(x_dec, c)

        if self.scale_by_sigma:
            assert self.absorb
            esigm1_log = (
                torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
                .log()
                .to(x_out.dtype)[:, None, None]
            )
            x_out = x_out - esigm1_log - np.log(x_out.shape[-1] - 1)

        x_out = torch.scatter(
            x_out, -1, indices[..., None], torch.zeros_like(x_out[..., :1])
        )
        return x_out

    def forward_hidden(
        self, indices: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning decoder hidden states (before output head).

        Useful for downstream heads (tagger, specialists).
        """
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder_hidden = self._encode(x, c)
            x_dec = self._decode(x, c, encoder_hidden)

        return x_dec
