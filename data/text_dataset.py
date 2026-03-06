from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from training.config import MinibatchEntry


class RandomTokenDataset(Dataset):
    """Synthetic random-token dataset for initial testing."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 2000):
        self.samples: list[torch.Tensor] = []
        for _ in range(num_samples):
            length = torch.randint(seq_len // 2, seq_len, (1,)).item()
            self.samples.append(torch.randint(0, vocab_size, (length,)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def pad_collate(batch: list[torch.Tensor], seq_len: int, pad_id: int) -> torch.Tensor:
    padded = torch.full((len(batch), seq_len), pad_id, dtype=torch.long)
    for i, tokens in enumerate(batch):
        length = min(len(tokens), seq_len)
        padded[i, :length] = tokens[:length]
    return padded


def create_minibatch_loaders(
    entries: list[MinibatchEntry],
    vocab_size: int,
    pad_id: int,
    num_samples: int = 2000,
) -> list[DataLoader]:
    """Create one DataLoader per MinibatchEntry (each with its own seq_len/batch_size)."""
    loaders = []
    for entry in entries:
        dataset = RandomTokenDataset(vocab_size, entry.seq_len, num_samples)
        loader = DataLoader(
            dataset,
            batch_size=entry.batch_size,
            shuffle=True,
            collate_fn=lambda b, sl=entry.seq_len: pad_collate(b, sl, pad_id),
            drop_last=True,
        )
        loaders.append(loader)
    return loaders
