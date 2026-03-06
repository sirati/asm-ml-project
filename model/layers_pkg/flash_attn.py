from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

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


class FlashAttnSelfLayer(SequenceLayer):
    """Flash Attention 2 self-attention with softmax-1 + MLP."""

    def __init__(self, dim: int, config: LayerConfig):
        super().__init__()
        self.dim = dim
        self.config = config
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

    def estimate_memory(
        self,
        batch: int,
        seq_len: int,
        mode: MemoryMode,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> LayerMemoryEstimate:
        dim, cfg = self.dim, self.config
        qkv_p = linear_param_count(dim, 3 * dim, cfg.qkv_bias)
        out_p = linear_param_count(dim, dim, bias=True)
        mlp_p = mlp_param_count(dim, cfg.widening_factor)
        ln_p = 2 * layernorm_param_count(dim)
        total_params = qkv_p + out_p + mlp_p + ln_p
        param_bytes = total_params * dtypes.param

        if mode == MemoryMode.INFERENCE:
            act = batch * seq_len * dim * dtypes.activation
            return LayerMemoryEstimate(param_bytes, act, 0)

        if checkpoint:
            act = batch * seq_len * dim * dtypes.activation
        else:
            # Flash-attn stores Q, K, V, output in bf16 + logsumexp in fp32
            qkvo = 4 * batch * seq_len * dim * dtypes.activation
            logsumexp = batch * cfg.num_heads * seq_len * dtypes.logsumexp
            mlp_act = mlp_activation_bytes(
                batch, seq_len, dim, cfg.widening_factor, dtypes
            )
            ln_act = 2 * layernorm_activation_bytes(batch, seq_len, dim, dtypes)
            residuals = 2 * batch * seq_len * dim * dtypes.activation
            act = qkvo + logsumexp + mlp_act + ln_act + residuals

        grad = total_params * dtypes.gradient
        return LayerMemoryEstimate(param_bytes, act, grad)


class FlashAttnCrossLayer(CrossSequenceLayer):
    """Flash Attention 2 cross-attention with softmax-1 + MLP."""

    def __init__(self, d_q: int, d_kv: int, config: LayerConfig):
        super().__init__()
        self.d_q = d_q
        self.d_kv = d_kv
        self.config = config
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

        q = rearrange(
            self.q_proj(self.q_norm(x_q)), "b n (h d) -> b n h d", h=self.num_heads
        )
        kv_norm = self.kv_norm(x_kv)
        k = rearrange(self.k_proj(kv_norm), "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(self.v_proj(kv_norm), "b l (h d) -> b l h d", h=self.num_heads)

        dp = self.dropout_p if self.training else 0.0
        attn_out = flash_attn_softmax1(q, k, v, dropout_p=dp)
        attn_out = rearrange(attn_out, "b n h d -> b n (h d)")
        x_q = x_q + self.dropout(self.out_proj(attn_out))
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
        q_p = linear_param_count(d_q, d_q, cfg.qkv_bias)
        k_p = linear_param_count(d_kv, d_q, cfg.qkv_bias)
        v_p = linear_param_count(d_kv, d_q, cfg.qkv_bias)
        out_p = linear_param_count(d_q, d_q, bias=True)
        mlp_p = mlp_param_count(d_q, cfg.widening_factor)
        ln_p = (
            layernorm_param_count(d_q)
            + layernorm_param_count(d_kv)
            + layernorm_param_count(d_q)
        )
        total_params = q_p + k_p + v_p + out_p + mlp_p + ln_p
        param_bytes = total_params * dtypes.param

        if mode == MemoryMode.INFERENCE:
            act = batch * seq_len_q * d_q * dtypes.activation
            return LayerMemoryEstimate(param_bytes, act, 0)

        if checkpoint:
            act = (
                batch * seq_len_q * d_q + batch * seq_len_kv * d_kv
            ) * dtypes.activation
        else:
            # Q, output in bf16 at query length; K, V in bf16 at kv length
            qo = 2 * batch * seq_len_q * d_q * dtypes.activation
            kv_stored = 2 * batch * seq_len_kv * d_q * dtypes.activation
            logsumexp = batch * cfg.num_heads * seq_len_q * dtypes.logsumexp
            mlp_act = mlp_activation_bytes(
                batch, seq_len_q, d_q, cfg.widening_factor, dtypes
            )
            ln_act = (
                layernorm_activation_bytes(batch, seq_len_q, d_q, dtypes)
                + layernorm_activation_bytes(batch, seq_len_kv, d_kv, dtypes)
                + layernorm_activation_bytes(batch, seq_len_q, d_q, dtypes)
            )
            residuals = 2 * batch * seq_len_q * d_q * dtypes.activation
            act = qo + kv_stored + logsumexp + mlp_act + ln_act + residuals

        grad = total_params * dtypes.gradient
        return LayerMemoryEstimate(param_bytes, act, grad)
