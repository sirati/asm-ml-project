#!/usr/bin/env python3
import math
import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    try:
        from flash_attn.flash_attn_interface import flash_attn_func
    except Exception as e:
        print(f"Failed to import flash_attn_func: {e}")
        return 1

    # Parameters
    batch_size = 2
    seq_len = 128
    head_dim = 64
    num_heads = 4
    dropout_p = 0.0
    causal = False
    dtype = torch.float16
    device = torch.device("cuda")

    # Create random Q, K, V with shape [B, S, H] per head and run per-head
    # flash_attn_func expects [B, S, H] for single head; we loop heads to exercise more paths
    q_all = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k_all = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v_all = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    outs = []
    for h in range(num_heads):
        q = q_all[:, :, h, :].contiguous()
        k = k_all[:, :, h, :].contiguous()
        v = v_all[:, :, h, :].contiguous()
        out = flash_attn_func(
            q, k, v, dropout_p=dropout_p, softmax_scale=None, causal=causal
        )
        outs.append(out)

    out_all = torch.stack(outs, dim=2)  # [B, S, num_heads, head_dim]
    print(f"Output shape: {tuple(out_all.shape)}, dtype: {out_all.dtype}")

    # Simple scalar loss to drive backward
    loss = out_all.sum()
    print(f"Loss: {float(loss)}")

    loss.backward()

    # Validate gradients exist and have non-zero norms
    q_grad_norm = q_all.grad.norm().item() if q_all.grad is not None else math.nan
    k_grad_norm = k_all.grad.norm().item() if k_all.grad is not None else math.nan
    v_grad_norm = v_all.grad.norm().item() if v_all.grad is not None else math.nan
    print(
        f"Grad norms (q, k, v): {q_grad_norm:.6f}, {k_grad_norm:.6f}, {v_grad_norm:.6f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
