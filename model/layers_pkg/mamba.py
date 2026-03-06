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


def _mamba2_param_count(dim: int, config: LayerConfig) -> int:
    """Approximate parameter count for a single Mamba2 block."""
    d_inner = dim * config.expand
    nheads = max(1, d_inner // 64)  # mamba2 default headdim=64
    ngroups = max(1, nheads // 8)
    in_proj = dim * (2 * d_inner + 2 * ngroups * config.d_state + nheads)
    conv = d_inner * config.d_conv
    out_proj = d_inner * dim
    scalars = 3 * nheads  # dt_bias, A, D
    return in_proj + conv + out_proj + scalars


def _mamba2_training_activation_bytes(
    dim: int,
    config: LayerConfig,
    batch: int,
    seq_len: int,
    dtypes: DTypeConfig,
) -> int:
    """Activation bytes stored by Mamba2 for backward pass."""
    d_inner = dim * config.expand
    nheads = max(1, d_inner // 64)
    chunk_size = 256  # mamba2 default
    n_chunks = (seq_len + chunk_size - 1) // chunk_size

    # Expanded activations (bf16)
    expanded = batch * seq_len * d_inner * dtypes.activation
    # Conv1d state (bf16)
    conv_state = batch * d_inner * config.d_conv * dtypes.activation
    # SSM states per chunk (bf16)
    headdim = d_inner // nheads
    ssm_states = (
        batch * n_chunks * nheads * headdim * config.d_state * dtypes.activation
    )
    return expanded + conv_state + ssm_states


def _mamba2_inference_activation_bytes(
    dim: int,
    config: LayerConfig,
    batch: int,
    dtypes: DTypeConfig,
) -> int:
    """Mamba is recurrent at inference — only stores state, not full sequence."""
    d_inner = dim * config.expand
    return batch * d_inner * config.d_state * dtypes.activation


class MambaSelfLayer(SequenceLayer):
    """Mamba-2 self-processing layer with feedforward."""

    def __init__(self, dim: int, config: LayerConfig):
        super().__init__()
        from mamba_ssm import Mamba2

        self.dim = dim
        self.config = config
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2(
            d_model=dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
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

    def estimate_memory(
        self,
        batch: int,
        seq_len: int,
        mode: MemoryMode,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> LayerMemoryEstimate:
        dim, cfg = self.dim, self.config
        mamba_p = _mamba2_param_count(dim, cfg)
        mlp_p = mlp_param_count(dim, cfg.widening_factor)
        ln_p = 2 * layernorm_param_count(dim)
        total_params = mamba_p + mlp_p + ln_p
        param_bytes = total_params * dtypes.param

        if mode == MemoryMode.INFERENCE:
            state = _mamba2_inference_activation_bytes(dim, cfg, batch, dtypes)
            act = batch * dim * dtypes.activation + state
            return LayerMemoryEstimate(param_bytes, act, 0)

        if checkpoint:
            act = batch * seq_len * dim * dtypes.activation
        else:
            mamba_act = _mamba2_training_activation_bytes(
                dim, cfg, batch, seq_len, dtypes
            )
            mlp_act = mlp_activation_bytes(
                batch, seq_len, dim, cfg.widening_factor, dtypes
            )
            ln_act = 2 * layernorm_activation_bytes(batch, seq_len, dim, dtypes)
            residuals = 2 * batch * seq_len * dim * dtypes.activation
            act = mamba_act + mlp_act + ln_act + residuals

        grad = total_params * dtypes.gradient
        return LayerMemoryEstimate(param_bytes, act, grad)


class MambaCrossLayer(CrossSequenceLayer):
    """Mamba-based cross-attention: concatenate context + query, run Mamba, take query slice."""

    def __init__(self, d_q: int, d_kv: int, config: LayerConfig):
        super().__init__()
        from mamba_ssm import Mamba2

        self.d_q = d_q
        self.d_kv = d_kv
        self.config = config
        self.q_norm = nn.LayerNorm(d_q)
        self.kv_norm = nn.LayerNorm(d_kv)
        self.proj_kv = nn.Linear(d_kv, d_q)
        self.mamba = Mamba2(
            d_model=d_q,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
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
        out = self.mamba(combined)[:, x_kv.shape[1] :]
        x_q = x_q + out
        x_q = x_q + self.mlp(self.norm2(x_q))
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
        mamba_p = _mamba2_param_count(d_q, cfg)
        kv_proj_p = linear_param_count(d_kv, d_q)
        mlp_p = mlp_param_count(d_q, cfg.widening_factor)
        ln_p = (
            layernorm_param_count(d_q)
            + layernorm_param_count(d_kv)
            + layernorm_param_count(d_q)
        )
        total_params = mamba_p + kv_proj_p + mlp_p + ln_p
        param_bytes = total_params * dtypes.param

        combined_len = seq_len_q + seq_len_kv

        if mode == MemoryMode.INFERENCE:
            state = _mamba2_inference_activation_bytes(d_q, cfg, batch, dtypes)
            act = batch * d_q * dtypes.activation + state
            return LayerMemoryEstimate(param_bytes, act, 0)

        if checkpoint:
            act = (
                batch * seq_len_q * d_q + batch * seq_len_kv * d_kv
            ) * dtypes.activation
        else:
            mamba_act = _mamba2_training_activation_bytes(
                d_q, cfg, batch, combined_len, dtypes
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
            act = mamba_act + mlp_act + ln_act + residuals

        grad = total_params * dtypes.gradient
        return LayerMemoryEstimate(param_bytes, act, grad)


class MambaOnlySelfLayer(SequenceLayer):
    """Two Mamba-2 blocks with no feedforward network."""

    def __init__(self, dim: int, config: LayerConfig):
        super().__init__()
        from mamba_ssm import Mamba2

        self.dim = dim
        self.config = config
        self.norm1 = nn.LayerNorm(dim)
        self.mamba1 = Mamba2(
            d_model=dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mamba2 = Mamba2(
            d_model=dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba1(self.norm1(x))
        x = x + self.mamba2(self.norm2(x))
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
        one_mamba_p = _mamba2_param_count(dim, cfg)
        ln_p = layernorm_param_count(dim)
        total_params = 2 * (one_mamba_p + ln_p)
        param_bytes = total_params * dtypes.param

        if mode == MemoryMode.INFERENCE:
            state = 2 * _mamba2_inference_activation_bytes(dim, cfg, batch, dtypes)
            act = batch * dim * dtypes.activation + state
            return LayerMemoryEstimate(param_bytes, act, 0)

        if checkpoint:
            act = batch * seq_len * dim * dtypes.activation
        else:
            one_mamba_act = _mamba2_training_activation_bytes(
                dim, cfg, batch, seq_len, dtypes
            )
            ln_act = layernorm_activation_bytes(batch, seq_len, dim, dtypes)
            residuals = 2 * batch * seq_len * dim * dtypes.activation
            act = 2 * (one_mamba_act + ln_act) + residuals

        grad = total_params * dtypes.gradient
        return LayerMemoryEstimate(param_bytes, act, grad)
