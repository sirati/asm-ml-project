"""Confidence remnant computation for UECD.

Measures how much of a token's identity remains readable from a
continuously-corrupted embedding vector, using softmax_1 (off-by-one softmax)
and self-confidence normalization for non-orthogonal embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax1(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Off-by-one softmax (Evan Miller, 2023).

    softmax_1(x)_j = exp(x_j) / (1 + sum_k exp(x_k))

    Allows all outputs to approach zero when no class is confident,
    unlike standard softmax which must always sum to 1.
    """
    exp_logits = torch.exp(logits - logits.max(dim=dim, keepdim=True).values)
    return exp_logits / (1.0 + exp_logits.sum(dim=dim, keepdim=True))


def confidence_remnant(
    z: torch.Tensor,
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute normalized confidence remnant ĉ_t.

    How much of each token's identity is still readable from the
    continuously-corrupted vector z.

    Args:
        z: Corrupted embedding vectors, shape (B, L, D).
        token_ids: Original token indices, shape (B, L).
        embedding_weight: Token embedding matrix, shape (V, D).

    Returns:
        Normalized confidence ĉ_t of shape (B, L), in [0, 1].
    """
    raw_c = _raw_confidence(z, token_ids, embedding_weight)
    c_max = _self_confidence(token_ids, embedding_weight)
    return raw_c / c_max.clamp(min=1e-8)


def _raw_confidence(
    z: torch.Tensor,
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """Raw softmax_1 confidence: P(token_id | z) via dot-product logits."""
    # (B, L, V)
    logits = torch.matmul(z, embedding_weight.T)
    # (B, L, V)
    probs = softmax1(logits, dim=-1)
    # Gather the probability of the actual token at each position
    return probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def _self_confidence(
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """Self-confidence c_max(x): confidence of a pure (uncorrupted) token embedding.

    c_max(x) = softmax_1(embed(.) @ embed(x))_x

    This is the maximum confidence a token can ever have, used to normalize
    raw confidence into a proper [0, 1] range that is comparable across tokens
    with different embedding overlaps.
    """
    # (B, L, D)
    pure_embed = F.embedding(token_ids, embedding_weight)
    # (B, L, V)
    logits = torch.matmul(pure_embed, embedding_weight.T)
    probs = softmax1(logits, dim=-1)
    return probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
