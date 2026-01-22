#!/usr/bin/env python3
import math
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    batch_size: int = 2
    seq_len: int = 128
    num_heads: int = 4
    head_dim: int = 64
    dtype: torch.dtype = torch.float16
    causal: bool = False
    dropout_p: float = 0.0


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    try:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    except Exception as e:
        print(f"Failed to import flash_attn_unpadded_func: {e}")
        return 1

    cfg = Config()
    device = torch.device("cuda")

    total_tokens = cfg.batch_size * cfg.seq_len
    q = torch.randn(
        total_tokens,
        cfg.num_heads,
        cfg.head_dim,
        device=device,
        dtype=cfg.dtype,
        requires_grad=True,
    )
    k = torch.randn(
        total_tokens,
        cfg.num_heads,
        cfg.head_dim,
        device=device,
        dtype=cfg.dtype,
        requires_grad=True,
    )
    v = torch.randn(
        total_tokens,
        cfg.num_heads,
        cfg.head_dim,
        device=device,
        dtype=cfg.dtype,
        requires_grad=True,
    )

    cu_seqlens = torch.arange(
        0,
        (cfg.batch_size + 1) * cfg.seq_len,
        cfg.seq_len,
        device=device,
        dtype=torch.int32,
    )
    max_seqlen = cfg.seq_len

    out = flash_attn_unpadded_func(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p=cfg.dropout_p,
        softmax_scale=None,
        causal=cfg.causal,
    )
    print(f"Output shape: {tuple(out.shape)}, dtype: {out.dtype}")

    loss = out.sum()
    print(f"Loss: {float(loss)}")
    loss.backward()

    q_grad = q.grad.norm().item() if q.grad is not None else math.nan
    k_grad = k.grad.norm().item() if k.grad is not None else math.nan
    v_grad = v.grad.norm().item() if v.grad is not None else math.nan
    print(f"Grad norms (q, k, v): {q_grad:.6f}, {k_grad:.6f}, {v_grad:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
