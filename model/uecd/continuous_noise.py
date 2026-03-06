"""Continuous noise schedule for UECD.

Provides alpha_t / sigma_t schedules for the continuous diffusion process
with v-prediction parameterisation. Uses cosine schedule by default with
partial signal retention at t=T (alpha_T > 0).
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


class ContinuousNoiseSchedule(abc.ABC, nn.Module):
    """Maps t in [0, 1] to (alpha_t, sigma_t) for continuous diffusion."""

    @abc.abstractmethod
    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (alpha_t, sigma_t) for given timesteps."""
        ...

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio: alpha_t^2 / sigma_t^2."""
        alpha, sigma = self(t)
        return alpha.square() / sigma.square().clamp(min=1e-8)

    def v_target(
        self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """v-prediction target: v_t = alpha_t * eps - sigma_t * x0."""
        alpha, sigma = self(t)
        while alpha.dim() < x0.dim():
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * eps - sigma * x0

    def add_noise(
        self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """z_t = alpha_t * x0 + sigma_t * eps."""
        alpha, sigma = self(t)
        while alpha.dim() < x0.dim():
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * x0 + sigma * eps

    def predict_x0(
        self, z_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover x0 from v-prediction: x0_hat = alpha_t * z_t - sigma_t * v."""
        alpha, sigma = self(t)
        while alpha.dim() < z_t.dim():
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * z_t - sigma * v

    def predict_eps(
        self, z_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover eps from v-prediction: eps_hat = alpha_t * v + sigma_t * z_t."""
        alpha, sigma = self(t)
        while alpha.dim() < z_t.dim():
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * v + sigma * z_t


class CosineSchedule(ContinuousNoiseSchedule):
    """Cosine noise schedule with partial signal retention at t=T.

    alpha_t = cos(t * pi/2) * (1 - alpha_T) + alpha_T
    sigma_t = sin(t * pi/2)

    At t=0: alpha=1, sigma=0 (clean)
    At t=1: alpha=alpha_T, sigma=1 (mostly noise, partial signal retained)
    """

    def __init__(self, alpha_t_max: float = 0.2):
        super().__init__()
        self.alpha_t_max = alpha_t_max

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.cos(t * math.pi / 2) * (1.0 - self.alpha_t_max) + self.alpha_t_max
        sigma = torch.sin(t * math.pi / 2)
        return alpha, sigma


class LinearSchedule(ContinuousNoiseSchedule):
    """Linear noise schedule with partial signal retention.

    alpha_t = 1 - t * (1 - alpha_T)
    sigma_t = t
    """

    def __init__(self, alpha_t_max: float = 0.2):
        super().__init__()
        self.alpha_t_max = alpha_t_max

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = 1.0 - t * (1.0 - self.alpha_t_max)
        sigma = t
        return alpha, sigma


@dataclass
class ContinuousNoiseConfig:
    schedule_type: str = "cosine"
    alpha_t_max: float = 0.2

    def build(self) -> ContinuousNoiseSchedule:
        if self.schedule_type == "cosine":
            return CosineSchedule(self.alpha_t_max)
        elif self.schedule_type == "linear":
            return LinearSchedule(self.alpha_t_max)
        raise ValueError(f"Unknown continuous schedule: {self.schedule_type}")
