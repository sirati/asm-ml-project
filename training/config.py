from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    warmup_steps: int = 100
    log_interval: int = 10
    grad_clip: float = 1.0
    device: str = "cuda"
    tokenizer_name: str = "gpt2"
    # Each entry: (seq_len, batch_size). A minibatch consists of one forward
    # pass per entry; gradients are accumulated across all before stepping.
    minibatch_spec: list[MinibatchEntry] = field(
        default_factory=lambda: [
            MinibatchEntry(seq_len=512, batch_size=4),
        ]
    )


@dataclass
class MinibatchEntry:
    seq_len: int
    batch_size: int
