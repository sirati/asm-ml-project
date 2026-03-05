"""Pluggable layer backends for attention and sequence processing.

All sequence-processing layers conform to a common interface so they can
be swapped via configuration. Each backend is registered by name and
can be selected at model construction time.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Layer protocol — all backends must implement this
# ---------------------------------------------------------------------------

class SequenceLayer(nn.Module, abc.ABC):
    """A single sequence-processing layer (self-attention, mamba, etc).

    Takes (B, L, D) -> (B, L, D).
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class CrossSequenceLayer(nn.Module, abc.ABC):
    """A cross-attention-like layer.

    Takes query (B, N, D_q) and context (B, L, D_kv) -> (B, N, D_q).
    """

    @abc.abstractmethod
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

class LayerBackend(str, Enum):
    ATTN = "attn"
    FLASH_ATTN = "flash_attn"
    MAMBA = "mamba"


@dataclass
class LayerConfig:
    backend: LayerBackend = LayerBackend.ATTN
    num_heads: int = 8
    dropout: float = 0.0
    widening_factor: int = 1
    qkv_bias: bool = True
    # mamba-specific (for future use)
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2


# ---------------------------------------------------------------------------
# Standard attention backend
# ---------------------------------------------------------------------------

class _AttnSelfLayer(SequenceLayer):
    """Standard PyTorch self-attention + MLP."""

    def __init__(self, dim: int, config: LayerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout,
            bias=config.qkv_bias, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, config.widening_factor * dim),
            nn.GELU(),
            nn.Linear(config.widening_factor * dim, dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.dropout(self.attn(x_norm, x_norm, x_norm, need_weights=False)[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class _AttnCrossLayer(CrossSequenceLayer):
    """Standard PyTorch cross-attention + MLP."""

    def __init__(self, d_q: int, d_kv: int, config: LayerConfig):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_q)
        self.kv_norm = nn.LayerNorm(d_kv)
        self.attn = nn.MultiheadAttention(
            d_q, config.num_heads, dropout=config.dropout,
            bias=config.qkv_bias, batch_first=True,
            kdim=d_kv, vdim=d_kv,
        )
        self.norm2 = nn.LayerNorm(d_q)
        self.mlp = nn.Sequential(
            nn.Linear(d_q, config.widening_factor * d_q),
            nn.GELU(),
            nn.Linear(config.widening_factor * d_q, d_q),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(x_q)
        kv = self.kv_norm(x_kv)
        x_q = x_q + self.dropout(self.attn(q, kv, kv, need_weights=False)[0])
        x_q = x_q + self.dropout(self.mlp(self.norm2(x_q)))
        return x_q


# ---------------------------------------------------------------------------
# Flash Attention backend
# ---------------------------------------------------------------------------

class _FlashAttnSelfLayer(SequenceLayer):
    """Flash Attention 2 self-attention with softmax-1 + MLP."""

    def __init__(self, dim: int, config: LayerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, config.widening_factor * dim),
            nn.GELU(),
            nn.Linear(config.widening_factor * dim, dim),
        )
        self.dropout_p = config.dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from utils.flash_utils import flash_attn_softmax1

        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm)
        q, k, v = [
            rearrange(t, "b l (h d) -> b l h d", h=self.num_heads)
            for t in qkv.chunk(3, dim=-1)
        ]
        dp = self.dropout_p if self.training else 0.0
        attn_out = flash_attn_softmax1(q, k, v, dropout_p=dp)
        attn_out = rearrange(attn_out, "b l h d -> b l (h d)")
        x = x + self.dropout(self.out_proj(attn_out))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class _FlashAttnCrossLayer(CrossSequenceLayer):
    """Flash Attention 2 cross-attention with softmax-1 + MLP."""

    def __init__(self, d_q: int, d_kv: int, config: LayerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.q_norm = nn.LayerNorm(d_q)
        self.kv_norm = nn.LayerNorm(d_kv)
        self.q_proj = nn.Linear(d_q, d_q, bias=config.qkv_bias)
        self.k_proj = nn.Linear(d_kv, d_q, bias=config.qkv_bias)
        self.v_proj = nn.Linear(d_kv, d_q, bias=config.qkv_bias)
        self.out_proj = nn.Linear(d_q, d_q)
        self.norm2 = nn.LayerNorm(d_q)
        self.mlp = nn.Sequential(
            nn.Linear(d_q, config.widening_factor * d_q),
            nn.GELU(),
            nn.Linear(config.widening_factor * d_q, d_q),
        )
        self.dropout_p = config.dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        from utils.flash_utils import flash_attn_softmax1

        q = rearrange(self.q_proj(self.q_norm(x_q)), "b n (h d) -> b n h d", h=self.num_heads)
        kv_norm = self.kv_norm(x_kv)
        k = rearrange(self.k_proj(kv_norm), "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(self.v_proj(kv_norm), "b l (h d) -> b l h d", h=self.num_heads)

        dp = self.dropout_p if self.training else 0.0
        attn_out = flash_attn_softmax1(q, k, v, dropout_p=dp)
        attn_out = rearrange(attn_out, "b n h d -> b n (h d)")
        x_q = x_q + self.dropout(self.out_proj(attn_out))
        x_q = x_q + self.dropout(self.mlp(self.norm2(x_q)))
        return x_q


# ---------------------------------------------------------------------------
# Mamba backend (stub for future use)
# ---------------------------------------------------------------------------

class _MambaSelfLayer(SequenceLayer):
    """Mamba-2 self-processing layer (requires mamba-ssm)."""

    def __init__(self, dim: int, config: LayerConfig):
        super().__init__()
        from mamba_ssm import Mamba2
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2(d_model=dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, config.widening_factor * dim),
            nn.GELU(),
            nn.Linear(config.widening_factor * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MambaCrossLayer(CrossSequenceLayer):
    """Mamba-based cross-attention: concatenate context + query, run Mamba, take query slice."""

    def __init__(self, d_q: int, d_kv: int, config: LayerConfig):
        super().__init__()
        from mamba_ssm import Mamba2
        self.q_norm = nn.LayerNorm(d_q)
        self.kv_norm = nn.LayerNorm(d_kv)
        self.proj_kv = nn.Linear(d_kv, d_q)
        self.mamba = Mamba2(d_model=d_q, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)
        self.norm2 = nn.LayerNorm(d_q)
        self.mlp = nn.Sequential(
            nn.Linear(d_q, config.widening_factor * d_q),
            nn.GELU(),
            nn.Linear(config.widening_factor * d_q, d_q),
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(x_q)
        kv = self.proj_kv(self.kv_norm(x_kv))
        combined = torch.cat([kv, q], dim=1)
        out = self.mamba(combined)[:, x_kv.shape[1]:]
        x_q = x_q + out
        x_q = x_q + self.mlp(self.norm2(x_q))
        return x_q


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

_SELF_LAYER_REGISTRY: dict[LayerBackend, type[SequenceLayer]] = {
    LayerBackend.ATTN: _AttnSelfLayer,
    LayerBackend.FLASH_ATTN: _FlashAttnSelfLayer,
    LayerBackend.MAMBA: _MambaSelfLayer,
}

_CROSS_LAYER_REGISTRY: dict[LayerBackend, type[CrossSequenceLayer]] = {
    LayerBackend.ATTN: _AttnCrossLayer,
    LayerBackend.FLASH_ATTN: _FlashAttnCrossLayer,
    LayerBackend.MAMBA: _MambaCrossLayer,
}


def make_self_layer(dim: int, config: LayerConfig) -> SequenceLayer:
    cls = _SELF_LAYER_REGISTRY.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown self-layer backend: {config.backend}")
    return cls(dim, config)


def make_cross_layer(d_q: int, d_kv: int, config: LayerConfig) -> CrossSequenceLayer:
    cls = _CROSS_LAYER_REGISTRY.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown cross-layer backend: {config.backend}")
    return cls(d_q, d_kv, config)
