from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LayerBackend(str, Enum):
    ATTN = "attn"
    FLASH_ATTN = "flash_attn"
    MAMBA = "mamba"
    MAMBA_ONLY = "mamba_only"


@dataclass
class LayerConfig:
    backend: LayerBackend = LayerBackend.ATTN
    num_heads: int = 8
    dropout: float = 0.0
    widening_factor: int = 1
    qkv_bias: bool = True
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
