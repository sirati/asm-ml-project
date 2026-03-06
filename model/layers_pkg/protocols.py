from __future__ import annotations

import abc

import torch
import torch.nn as nn

from model.memory import MIXED_BF16, DTypeConfig, LayerMemoryEstimate, MemoryMode


class SequenceLayer(nn.Module, abc.ABC):
    """A single sequence-processing layer (self-attention, mamba, etc).

    Takes (B, L, D) -> (B, L, D).
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def estimate_memory(
        self,
        batch: int,
        seq_len: int,
        mode: MemoryMode,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> LayerMemoryEstimate: ...


class CrossSequenceLayer(nn.Module, abc.ABC):
    """A cross-attention-like layer.

    Takes query (B, N, D_q) and context (B, L, D_kv) -> (B, N, D_q).
    """

    @abc.abstractmethod
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def estimate_memory(
        self,
        batch: int,
        seq_len_q: int,
        seq_len_kv: int,
        mode: MemoryMode,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> LayerMemoryEstimate: ...
