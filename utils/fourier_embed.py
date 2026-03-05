"""Fourier positional encoding for continuous N values.

Used to encode the N parameter in INSERT-N and DELETE-1-AND-INSERT-N
edit operations. Based on sinusoidal positional encoding
(https://arxiv.org/pdf/2502.09741).
"""

import math

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """Fourier embedding for continuous scalar values.

    Maps a scalar value (e.g. the N in INSERT-N) to a d_model-dimensional
    vector using sinusoidal frequencies, then projects through an MLP.
    """

    def __init__(
        self, d_model: int, frequency_dim: int = 256, max_period: float = 10000.0
    ):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def _encode(self, values: torch.Tensor) -> torch.Tensor:
        """Raw sinusoidal encoding of continuous values.

        Args:
            values: Tensor of shape (...) with continuous scalar values.

        Returns:
            Tensor of shape (..., frequency_dim).
        """
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=values.device)
            / half
        )
        args = values.unsqueeze(-1).float() * freqs
        out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.frequency_dim % 2:
            out = torch.cat([out, torch.zeros_like(out[..., :1])], dim=-1)
        return out

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode continuous values into d_model-dimensional embeddings.

        Args:
            values: Tensor of shape (...) with continuous scalar values.

        Returns:
            Tensor of shape (..., d_model).
        """
        return self.mlp(self._encode(values))
