from __future__ import annotations

import math

from training.config import TrainConfig


def cosine_lr(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    decay_ratio = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
