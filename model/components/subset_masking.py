from __future__ import annotations

from dataclasses import dataclass

import torch

from model.memory import MIXED_BF16, DTypeConfig


@dataclass
class SubsetMaskResult:
    masked_encoder_out: torch.Tensor  # (B * N, subset_len, D)
    decoder_input: torch.Tensor  # (B * N, subset_len, D)
    target_ids: torch.Tensor  # (B * N, subset_len)


def subset_seq_len(seq_len: int, num_splits: int) -> int:
    """The length of each subset: floor((seq_len-1) / num_splits) + 1 (for the always-included last token)."""
    return (seq_len - 1) // num_splits + 1


def estimate_subset_activation_bytes(
    batch: int,
    seq_len: int,
    dim: int,
    num_splits: int,
    dtypes: DTypeConfig = MIXED_BF16,
) -> int:
    """Memory for the three sub-batched tensors created by create_subset_masks."""
    sub_len = subset_seq_len(seq_len, num_splits)
    sub_batch = batch * num_splits
    # masked_encoder_out (bf16) + decoder_input (bf16) + target_ids (int64=8 bytes)
    encoder_bytes = sub_batch * sub_len * dim * dtypes.activation
    decoder_bytes = sub_batch * sub_len * dim * dtypes.activation
    target_bytes = sub_batch * sub_len * 8  # int64
    return encoder_bytes + decoder_bytes + target_bytes


def create_subset_masks(
    encoder_out: torch.Tensor,
    token_embeddings: torch.Tensor,
    token_ids: torch.Tensor,
    num_splits: int,
) -> SubsetMaskResult:
    """Split token indices (excluding the last) into N random subsets, always including the last token."""
    batch_size, seq_len, dim = encoder_out.shape
    num_available = seq_len - 1
    subset_size = num_available // num_splits

    device = encoder_out.device

    all_encoder = []
    all_decoder_input = []
    all_target = []

    for b in range(batch_size):
        perm = torch.randperm(num_available, device=device)
        for s in range(num_splits):
            idx = perm[s * subset_size : (s + 1) * subset_size]
            idx = torch.cat([idx, torch.tensor([seq_len - 1], device=device)])
            idx, _ = idx.sort()

            all_encoder.append(encoder_out[b, idx])
            all_decoder_input.append(token_embeddings[b, idx])
            all_target.append(token_ids[b, idx])

    return SubsetMaskResult(
        masked_encoder_out=torch.stack(all_encoder),
        decoder_input=torch.stack(all_decoder_input),
        target_ids=torch.stack(all_target),
    )
