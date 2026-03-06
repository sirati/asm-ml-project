from __future__ import annotations

import torch
import torch.nn as nn

from model.layers_pkg.config import LayerConfig
from model.layers_pkg.protocols import CrossSequenceLayer, SequenceLayer
from model.memory import (
    MIXED_BF16,
    DTypeConfig,
    LayerMemoryEstimate,
    MemoryMode,
    layernorm_activation_bytes,
    layernorm_param_count,
    linear_param_count,
    mlp_activation_bytes,
    mlp_param_count,
)


class AttnSelfLayer(SequenceLayer):
    """Standard PyTorch self-attention + MLP."""

    def __init__(self, dim: int, config: LayerConfig):
        super().__init__()
        self.dim = dim
        self.config = config
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            config.num_heads,
            dropout=config.dropout,
            bias=config.qkv_bias,
            batch_first=True,
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

    def estimate_memory(
        self,
        batch: int,
        seq_len: int,
        mode: MemoryMode,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> LayerMemoryEstimate:
        dim, cfg = self.dim, self.config
        # Q, K, V, out projections
        mha_p = 4 * linear_param_count(dim, dim, cfg.qkv_bias)
        mlp_p = mlp_param_count(dim, cfg.widening_factor)
        ln_p = 2 * layernorm_param_count(dim)
        total_params = mha_p + mlp_p + ln_p
        param_bytes = total_params * dtypes.param

        if mode == MemoryMode.INFERENCE:
            act = batch * seq_len * dim * dtypes.activation
            return LayerMemoryEstimate(param_bytes, act, 0)

        if checkpoint:
            act = batch * seq_len * dim * dtypes.activation
        else:
            # Attention matrix stored in fp32 by pytorch MHA backward
            attn_matrix = (
                batch * cfg.num_heads * seq_len * seq_len * dtypes.layernorm_act
            )
            mlp_act = mlp_activation_bytes(
                batch, seq_len, dim, cfg.widening_factor, dtypes
            )
            ln_act = 2 * layernorm_activation_bytes(batch, seq_len, dim, dtypes)
            residuals = 2 * batch * seq_len * dim * dtypes.activation
            act = attn_matrix + mlp_act + ln_act + residuals

        grad = total_params * dtypes.gradient
        return LayerMemoryEstimate(param_bytes, act, grad)


class AttnCrossLayer(CrossSequenceLayer):
    """Standard PyTorch cross-attention + MLP."""

    def __init__(self, d_q: int, d_kv: int, config: LayerConfig):
        super().__init__()
        self.d_q = d_q
        self.d_kv = d_kv
        self.config = config
        self.q_norm = nn.LayerNorm(d_q)
        self.kv_norm = nn.LayerNorm(d_kv)
        self.attn = nn.MultiheadAttention(
            d_q,
            config.num_heads,
            dropout=config.dropout,
            bias=config.qkv_bias,
            batch_first=True,
            kdim=d_kv,
            vdim=d_kv,
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

    def estimate_memory(
        self,
        batch: int,
        seq_len_q: int,
        seq_len_kv: int,
        mode: MemoryMode,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> LayerMemoryEstimate:
        d_q, d_kv, cfg = self.d_q, self.d_kv, self.config
        # Q proj from d_q, K/V proj from d_kv, out proj from d_q
        mha_p = (
            linear_param_count(d_q, d_q, cfg.qkv_bias)
            + 2 * linear_param_count(d_kv, d_q, cfg.qkv_bias)
            + linear_param_count(d_q, d_q)
        )
        mlp_p = mlp_param_count(d_q, cfg.widening_factor)
        ln_p = (
            layernorm_param_count(d_q)
            + layernorm_param_count(d_kv)
            + layernorm_param_count(d_q)
        )
        total_params = mha_p + mlp_p + ln_p
        param_bytes = total_params * dtypes.param

        if mode == MemoryMode.INFERENCE:
            act = batch * seq_len_q * d_q * dtypes.activation
            return LayerMemoryEstimate(param_bytes, act, 0)

        if checkpoint:
            act = (
                batch * seq_len_q * d_q + batch * seq_len_kv * d_kv
            ) * dtypes.activation
        else:
            attn_matrix = (
                batch * cfg.num_heads * seq_len_q * seq_len_kv * dtypes.layernorm_act
            )
            mlp_act = mlp_activation_bytes(
                batch, seq_len_q, d_q, cfg.widening_factor, dtypes
            )
            ln_act = (
                layernorm_activation_bytes(batch, seq_len_q, d_q, dtypes)
                + layernorm_activation_bytes(batch, seq_len_kv, d_kv, dtypes)
                + layernorm_activation_bytes(batch, seq_len_q, d_q, dtypes)
            )
            residuals = 2 * batch * seq_len_q * d_q * dtypes.activation
            act = attn_matrix + mlp_act + ln_act + residuals

        grad = total_params * dtypes.gradient
        return LayerMemoryEstimate(param_bytes, act, grad)
