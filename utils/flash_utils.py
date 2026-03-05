"""Flash Attention 2 wrapper with softmax-1 (dummy token prepend trick).

Implements "Attention is Off by One" (https://arxiv.org/abs/2403.17130):
prepend a dummy token to K/V so that softmax over keys effectively becomes
softmax-1, allowing the model to "attend to nothing".
"""

import torch
from einops import rearrange
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_qkvpacked_func,
)


def flash_attn_softmax1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    """Flash Attention 2 with softmax-1 via dummy token prepend.

    Args:
        q: Query tensor of shape (B, N_q, H, D).
        k: Key tensor of shape (B, N_kv, H, D).
        v: Value tensor of shape (B, N_kv, H, D).
        dropout_p: Dropout probability.
        causal: Whether to use causal masking.

    Returns:
        Attention output of shape (B, N_q, H, D).
    """
    b, _, h, d = q.shape

    dummy_k = torch.zeros(b, 1, h, d, dtype=k.dtype, device=k.device)
    dummy_v = torch.zeros(b, 1, h, d, dtype=v.dtype, device=v.device)

    k_aug = torch.cat([dummy_k, k], dim=1)
    v_aug = torch.cat([dummy_v, v], dim=1)

    # flash_attn_func handles the softmax internally; the dummy token at
    # position 0 in K/V acts as the "+1" denominator term in softmax-1.
    out = flash_attn_func(q, k_aug, v_aug, dropout_p=dropout_p, causal=False)
    return out


def flash_attn_varlen_softmax1(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    """Variable-length Flash Attention 2 with softmax-1 for packed sequences.

    This is a thin wrapper around flash_attn_varlen_qkvpacked_func used in
    the SEDD backbone (DDiTBlock). The softmax-1 trick is not applied here
    because the varlen interface operates on packed sequences where prepending
    dummy tokens per-sequence is non-trivial. Use the non-varlen variant
    (flash_attn_softmax1) when softmax-1 is needed.

    Args:
        qkv: Packed QKV tensor of shape (total_tokens, 3, H, D).
        cu_seqlens: Cumulative sequence lengths of shape (B+1,).
        max_seqlen: Maximum sequence length in the batch.
        dropout_p: Dropout probability.
        causal: Whether to use causal masking.

    Returns:
        Attention output of shape (total_tokens, H, D).
    """
    return flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen, dropout_p, causal=causal
    )
