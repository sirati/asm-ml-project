"""Score entropy loss and diffusion graph definitions.

Adapted from vendor/sedd/losses.py and vendor/sedd/graph_lib.py.
Removes Hydra config dependency and the external catsample dependency.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Categorical sampling (inlined from vendor/sedd/catsample.py)
# ---------------------------------------------------------------------------


def sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    """Gumbel-max trick for sampling from categorical probabilities."""
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


# ---------------------------------------------------------------------------
# Graph types (forward diffusion process)
# ---------------------------------------------------------------------------


class GraphType(Enum):
    UNIFORM = "uniform"
    ABSORBING = "absorbing"


class Graph(abc.ABC):
    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def absorb(self) -> bool: ...

    @abc.abstractmethod
    def rate(self, i: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def transp_rate(self, i: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: ...

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector)

    def reverse_rate(self, i: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        normalized_rate = self.transp_rate(i) * score
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(
            -1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True)
        )
        return normalized_rate

    def sample_rate(self, i: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    @abc.abstractmethod
    def staggered_score(
        self, score: torch.Tensor, dsigma: torch.Tensor
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def sample_limit(self, *batch_dims: int) -> torch.Tensor: ...

    @abc.abstractmethod
    def score_entropy(
        self,
        score: torch.Tensor,
        sigma: torch.Tensor,
        x: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def transp_transition(
        self, i: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor: ...


class Uniform(Graph):
    """Uniform transition: every token can become any other with equal probability."""

    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def absorb(self) -> bool:
        return False

    def rate(self, i: torch.Tensor) -> torch.Tensor:
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], -(self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i: torch.Tensor) -> torch.Tensor:
        return self.rate(i)

    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        trans = (
            torch.ones(*i.shape, self.dim, device=i.device)
            * (1 - (-sigma[..., None]).exp())
            / self.dim
        )
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def transp_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.transition(i, sigma)

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        return torch.where(move_indices, torch.randint_like(i, self.dim), i)

    def staggered_score(
        self, score: torch.Tensor, dsigma: torch.Tensor
    ) -> torch.Tensor:
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(
            dim=-1, keepdim=True
        ) + score / epow

    def sample_limit(self, *batch_dims: int) -> torch.Tensor:
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(
        self,
        score: torch.Tensor,
        sigma: torch.Tensor,
        x: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
        ratio = 1 - self.dim / (esigm1 + self.dim)

        neg_term = (
            score.mean(dim=-1)
            - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        )
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term,
        )

        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim,
        )

        sexp = score.exp()
        pos_term = (
            sexp.mean(dim=-1)
            - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        )
        return pos_term - neg_term + const


def _unsqueeze_as(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))


class Absorbing(Graph):
    """Absorbing state diffusion: tokens get masked to an absorbing state."""

    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim + 1

    @property
    def absorb(self) -> bool:
        return True

    def rate(self, i: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            (self.dim - 1) * torch.ones_like(i), num_classes=self.dim
        ) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i: torch.Tensor) -> torch.Tensor:
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        pass

    def transp_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = _unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0,
        )[..., None]
        return edge

    def sample_transition(self, i: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        return torch.where(move_indices, self.dim - 1, i)

    def staggered_score(
        self, score: torch.Tensor, dsigma: torch.Tensor
    ) -> torch.Tensor:
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims: int) -> torch.Tensor:
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(
        self,
        score: torch.Tensor,
        sigma: torch.Tensor,
        x: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        neg_term = ratio * torch.gather(
            score[rel_ind], -1, other_ind[..., None]
        ).squeeze(-1)
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy


def get_graph(graph_type: GraphType, num_tokens: int) -> Graph:
    if graph_type == GraphType.UNIFORM:
        return Uniform(num_tokens)
    elif graph_type == GraphType.ABSORBING:
        return Absorbing(num_tokens)
    raise ValueError(f"Unknown graph type: {graph_type}")


# ---------------------------------------------------------------------------
# Score function utilities (adapted from vendor/sedd/model/utils.py)
# ---------------------------------------------------------------------------


def get_model_fn(model: nn.Module, train: bool = False) -> Callable:
    def model_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if train:
            model.train()
        else:
            model.eval()
        return model(x, sigma)

    return model_fn


def get_score_fn(
    model: nn.Module, train: bool = False, sampling: bool = False
) -> Callable:
    if sampling:
        assert not train
    model_fn = get_model_fn(model, train=train)

    def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1)
        score = model_fn(x, sigma)
        if sampling:
            return score.exp()
        return score

    return score_fn


# ---------------------------------------------------------------------------
# Loss function (adapted from vendor/sedd/losses.py)
# ---------------------------------------------------------------------------


def score_entropy_loss(
    model: nn.Module,
    graph: Graph,
    noise_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    batch: torch.Tensor,
    sampling_eps: float = 1e-3,
) -> torch.Tensor:
    """Compute score entropy loss for a batch.

    Args:
        model: The score model that takes (x_t, sigma) and returns log-score.
        graph: The diffusion graph (Uniform or Absorbing).
        noise_fn: Callable that maps t -> (sigma, dsigma).
        batch: Input token indices of shape (B, L).
        sampling_eps: Minimum time value to avoid t=0 singularity.

    Returns:
        Per-example loss of shape (B,).
    """
    t = (1 - sampling_eps) * torch.rand(
        batch.shape[0], device=batch.device
    ) + sampling_eps
    sigma, dsigma = noise_fn(t)

    perturbed_batch = graph.sample_transition(batch, sigma[:, None])

    log_score_fn = get_score_fn(model, train=True, sampling=False)
    log_score = log_score_fn(perturbed_batch, sigma)
    loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

    return (dsigma[:, None] * loss).sum(dim=-1)
