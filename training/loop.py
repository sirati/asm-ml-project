from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import TrainConfig
from training.schedule import cosine_lr


@dataclass
class MinibatchResult:
    """Accumulated result across sub-batches of potentially different lengths."""

    total_loss: float
    total_tokens: int

    @property
    def mean_loss(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_loss / self.total_tokens


def _infinite_iter(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def train_loop(
    model: nn.Module,
    loaders: list[DataLoader],
    train_config: TrainConfig,
) -> None:
    """Training loop with gradient accumulation across variable-length sub-batches.

    Each minibatch step:
    1. For each loader (one per seq_len/batch_size combo), draw one batch.
    2. Forward each, accumulate sum-reduced loss.
    3. Divide total loss by total tokens for correct per-token average.
    4. Single backward + optimizer step.
    """
    device = torch.device(train_config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    data_iters = [_infinite_iter(loader) for loader in loaders]

    model.train()

    for step in range(train_config.max_steps):
        lr = cosine_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        result = _minibatch_step(model, data_iters, device)

        optimizer.zero_grad()
        result.normalized_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        if step % train_config.log_interval == 0:
            print(
                f"step {step:5d} | loss {result.metrics.mean_loss:.4f} "
                f"| tokens {result.metrics.total_tokens} | lr {lr:.2e}"
            )

    print("Training complete.")


@dataclass
class _StepResult:
    normalized_loss: torch.Tensor
    metrics: MinibatchResult


def _minibatch_step(
    model: nn.Module,
    data_iters: list[Iterator],
    device: torch.device,
) -> _StepResult:
    """Accumulate forward passes across sub-batches, return single loss tensor for backward."""
    accumulated_loss = torch.tensor(0.0, device=device, requires_grad=False)
    total_tokens = 0
    # We need the computation graph, so collect sum-reduced losses
    loss_parts: list[torch.Tensor] = []

    for data_iter in data_iters:
        batch = next(data_iter).to(device)
        output = model(batch)
        loss_parts.append(output.loss)
        accumulated_loss = accumulated_loss + output.loss.detach()
        total_tokens += output.num_tokens

    # Single combined loss normalized by total tokens across all sub-batches
    combined_loss = torch.stack(loss_parts).sum() / total_tokens

    return _StepResult(
        normalized_loss=combined_loss,
        metrics=MinibatchResult(
            total_loss=accumulated_loss.item(),
            total_tokens=total_tokens,
        ),
    )
