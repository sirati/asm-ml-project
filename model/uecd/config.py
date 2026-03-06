"""UECD top-level configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from model.uecd.continuous_backbone import ContinuousBackboneConfig
from model.uecd.continuous_noise import ContinuousNoiseConfig
from model.uecd.discrete_head import DiscreteHeadConfig
from model.uecd.sampling import SamplingConfig


@dataclass
class UECDConfig:
    num_tokens: int = 50257
    hidden_size: int = 768

    continuous_noise: ContinuousNoiseConfig = field(
        default_factory=ContinuousNoiseConfig
    )
    backbone: ContinuousBackboneConfig = field(default_factory=ContinuousBackboneConfig)
    discrete_head: DiscreteHeadConfig = field(default_factory=DiscreteHeadConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Discrete noise schedule (reused from existing SEDD infrastructure)
    discrete_noise_type: str = "loglinear"
    discrete_noise_eps: float = 1e-3

    sampling_eps: float = 1e-3

    def __post_init__(self) -> None:
        self.backbone.hidden_size = self.hidden_size
        self.discrete_head.hidden_size = self.hidden_size
        self.discrete_head.num_tokens = self.num_tokens
