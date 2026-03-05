"""Edit operation tagger head.

Predicts per-token edit operations and N values from backbone hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn

from utils.fourier_embed import FourierEmbedding


class EditOp(IntEnum):
    KEEP = 0
    DELETE = 1
    INSERT = 2
    DELETE_AND_INSERT = 3


@dataclass
class TaggerConfig:
    d_model: int = 768
    frequency_dim: int = 256
    max_n: int = 32


class EditTagger(nn.Module):
    """Predicts edit operations and N values from backbone hidden states.

    For each token position, predicts:
    - One of 4 edit operations (KEEP, DELETE, INSERT, DELETE_AND_INSERT)
    - A continuous N value (for INSERT and DELETE_AND_INSERT) encoded
      via Fourier embedding

    The N value is predicted as a continuous scalar and discretized
    during inference.
    """

    def __init__(self, config: TaggerConfig):
        super().__init__()
        self.op_head = nn.Linear(config.d_model, len(EditOp))
        self.n_head = nn.Linear(config.d_model, 1)
        self.n_embedder = FourierEmbedding(config.d_model, config.frequency_dim)
        self.max_n = config.max_n

    def forward(self, hidden_states: torch.Tensor) -> EditTaggerOutput:
        """
        Args:
            hidden_states: Backbone output of shape (B, L, D).

        Returns:
            EditTaggerOutput with op_logits and n_values.
        """
        op_logits = self.op_head(hidden_states)  # (B, L, 4)
        n_values = self.n_head(hidden_states).squeeze(-1)  # (B, L)
        n_values = torch.clamp(n_values, min=0.0, max=float(self.max_n))
        return EditTaggerOutput(op_logits=op_logits, n_values=n_values)

    def embed_n(self, n_values: torch.Tensor) -> torch.Tensor:
        """Embed N values into d_model-dimensional vectors.

        Args:
            n_values: Tensor of shape (...) with continuous N values.

        Returns:
            Tensor of shape (..., d_model).
        """
        return self.n_embedder(n_values)


@dataclass
class EditTaggerOutput:
    op_logits: torch.Tensor  # (B, L, 4)
    n_values: torch.Tensor   # (B, L)

    @property
    def op_predictions(self) -> torch.Tensor:
        """Discrete operation predictions: (B, L) int."""
        return self.op_logits.argmax(dim=-1)

    @property
    def n_predictions(self) -> torch.Tensor:
        """Discretized N values: (B, L) int."""
        return self.n_values.round().long().clamp(min=1)
