from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.memory import MIXED_BF16, DTypeConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # PE buffer is stored in fp32 (registered buffer, not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len]

    def estimate_buffer_bytes(self, dtypes: DTypeConfig = MIXED_BF16) -> int:
        """PE buffer is fp32 (not trained, not in optimizer)."""
        return self.max_len * self.dim * 4  # always fp32 buffer

    def estimate_activation_bytes(
        self,
        batch: int,
        seq_len: int,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> int:
        """After broadcasting and adding to embeddings, result is in activation dtype."""
        return batch * seq_len * self.dim * dtypes.activation
