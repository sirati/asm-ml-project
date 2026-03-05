"""Noise schedules and sampling strategies for discrete diffusion.

Adapted from vendor/sedd/noise_lib.py and vendor/sedd/sampling.py.
Removes Hydra config dependency and the external catsample dependency.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
import torch.nn as nn

from model.losses import Graph, get_score_fn, sample_categorical

# ---------------------------------------------------------------------------
# Noise schedules (adapted from vendor/sedd/noise_lib.py)
# ---------------------------------------------------------------------------


class Noise(abc.ABC, nn.Module):
    """Base noise schedule mapping t in [0, 1] to (sigma, dsigma)."""

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous rate of noise g(t)."""
        ...

    @abc.abstractmethod
    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative noise integral from 0 to t."""
        ...


class GeometricNoise(Noise):
    """Geometric interpolation between sigma_min and sigma_max.

    sigma(t) = sigma_min^(1-t) * sigma_max^t
    """

    def __init__(
        self, sigma_min: float = 1e-3, sigma_max: float = 1.0, learnable: bool = False
    ):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (
            self.sigmas[0] ** (1 - t)
            * self.sigmas[1] ** t
            * (self.sigmas[1].log() - self.sigmas[0].log())
        )

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise):
    """Log-linear noise schedule for absorbing diffusion.

    Total noise is -log(1 - (1 - eps) * t).
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1 - self.eps) * t)


@dataclass
class NoiseConfig:
    noise_type: str = "loglinear"
    sigma_min: float = 1e-4
    sigma_max: float = 20.0
    eps: float = 1e-3


def get_noise(config: NoiseConfig) -> Noise:
    if config.noise_type == "geometric":
        return GeometricNoise(config.sigma_min, config.sigma_max)
    elif config.noise_type == "loglinear":
        return LogLinearNoise(config.eps)
    raise ValueError(f"Unknown noise type: {config.noise_type}")


# ---------------------------------------------------------------------------
# Predictors (adapted from vendor/sedd/sampling.py)
# ---------------------------------------------------------------------------


class Predictor(abc.ABC):
    def __init__(self, graph: Graph, noise: Noise):
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(
        self,
        score_fn,
        x: torch.Tensor,
        t: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor: ...


class EulerPredictor(Predictor):
    def update_fn(
        self, score_fn, x: torch.Tensor, t: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)
        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        return self.graph.sample_rate(x, rev_rate)


class AnalyticPredictor(Predictor):
    def update_fn(
        self, score_fn, x: torch.Tensor, t: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)
        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Denoiser:
    def __init__(self, graph: Graph, noise: Noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma = self.noise(t)[0]
        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        if self.graph.absorb:
            probs = probs[..., :-1]
        return sample_categorical(probs)


_PREDICTORS = {
    "euler": EulerPredictor,
    "analytic": AnalyticPredictor,
}


# ---------------------------------------------------------------------------
# Sampling (adapted from vendor/sedd/sampling.py)
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    predictor: str = "euler"
    steps: int = 128
    noise_removal: bool = True


def get_pc_sampler(
    graph: Graph,
    noise: Noise,
    batch_dims: tuple[int, ...],
    config: SamplingConfig,
    eps: float = 1e-5,
    device: torch.device = torch.device("cpu"),
):
    """Build a predictor-corrector sampler.

    Returns a callable that takes a model and produces samples.
    """
    predictor_cls = _PREDICTORS.get(config.predictor)
    if predictor_cls is None:
        raise ValueError(f"Unknown predictor: {config.predictor}")

    predictor = predictor_cls(graph, noise)
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model: nn.Module) -> torch.Tensor:
        sampling_score_fn = get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, config.steps + 1, device=device)
        dt = (1 - eps) / config.steps

        for i in range(config.steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)

        if config.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)

        return x

    return pc_sampler
