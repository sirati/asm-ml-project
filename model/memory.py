"""Shared types and utilities for analytical VRAM estimation.

Layer-specific estimation logic lives in each layer class.
This module provides the common types, dtype-aware byte calculations,
and aggregation structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class MemoryMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"


@dataclass
class DTypeConfig:
    """Byte sizes for different tensor roles, reflecting mixed-precision training.

    Typical mixed-precision setup:
    - Parameters stored in bf16 (2 bytes)
    - Activations in bf16 (2 bytes)
    - Gradients in bf16 (2 bytes)
    - Optimizer states (AdamW m, v) in fp32 (4 bytes each)
    - Master weights in fp32 (4 bytes) — kept by optimizer
    - LayerNorm runs in fp32 (4 bytes for intermediate activations)
    - Flash-attn logsumexp stored in fp32 (4 bytes)
    """

    param: int = 2  # bf16
    activation: int = 2  # bf16
    gradient: int = 2  # bf16
    optimizer_state: int = 4  # fp32 (per state, AdamW has 2)
    master_weight: int = 4  # fp32 copy of params kept by optimizer
    layernorm_act: int = 4  # fp32 intermediate in layernorm
    logsumexp: int = 4  # fp32 logsumexp in flash-attn backward

    @staticmethod
    def from_dtype(dtype: torch.dtype) -> int:
        return {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float64: 8,
        }.get(dtype, 2)


# Singleton for the common bf16 mixed-precision setup
MIXED_BF16 = DTypeConfig()


@dataclass
class LayerMemoryEstimate:
    param_bytes: int = 0
    activation_bytes: int = 0
    gradient_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.param_bytes + self.activation_bytes + self.gradient_bytes

    def __iadd__(self, other: LayerMemoryEstimate) -> LayerMemoryEstimate:
        self.param_bytes += other.param_bytes
        self.activation_bytes += other.activation_bytes
        self.gradient_bytes += other.gradient_bytes
        return self


@dataclass
class MemoryEstimate:
    param_bytes: int = 0
    optimizer_bytes: int = 0
    activation_bytes: int = 0
    gradient_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.param_bytes
            + self.optimizer_bytes
            + self.activation_bytes
            + self.gradient_bytes
        )

    def summary(self) -> str:
        def _gb(b: int) -> str:
            return f"{b / (1024**3):.3f} GB"

        return (
            f"Parameters:  {_gb(self.param_bytes)}\n"
            f"Optimizer:   {_gb(self.optimizer_bytes)}\n"
            f"Activations: {_gb(self.activation_bytes)}\n"
            f"Gradients:   {_gb(self.gradient_bytes)}\n"
            f"Total:       {_gb(self.total_bytes)}"
        )


def linear_param_count(in_dim: int, out_dim: int, bias: bool = True) -> int:
    return in_dim * out_dim + (out_dim if bias else 0)


def layernorm_param_count(dim: int) -> int:
    return 2 * dim  # weight + bias


def mlp_param_count(dim: int, widening_factor: int, bias: bool = True) -> int:
    return linear_param_count(dim, widening_factor * dim, bias) + linear_param_count(
        widening_factor * dim, dim, bias
    )


def mlp_activation_bytes(
    batch: int,
    seq_len: int,
    dim: int,
    widening_factor: int,
    dtypes: DTypeConfig,
) -> int:
    """MLP stores the widened intermediate for GELU backward."""
    return batch * seq_len * widening_factor * dim * dtypes.activation


def layernorm_activation_bytes(
    batch: int,
    seq_len: int,
    dim: int,
    dtypes: DTypeConfig,
) -> int:
    """LayerNorm casts to fp32 internally; stores fp32 normalized output for backward."""
    return batch * seq_len * dim * dtypes.layernorm_act


def optimizer_bytes_for_params(param_count: int, dtypes: DTypeConfig) -> int:
    """AdamW: master weights (fp32) + m (fp32) + v (fp32) = 3 fp32 copies."""
    return param_count * (dtypes.master_weight + 2 * dtypes.optimizer_state)
