"""UECD top-level model.

Unified Embedding-Space Co-Diffusion model combining:
- Shared token embedding (= continuous diffusion space)
- Continuous backbone (v-prediction denoiser)
- Discrete head (SEDD concrete score predictor)
- Joint forward process with confidence-remnant blending
- SNR-weighted combined loss
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from model.diffusion import LogLinearNoise, NoiseConfig, get_noise
from model.losses import Absorbing, GraphType, get_graph
from model.uecd.config import UECDConfig
from model.uecd.continuous_backbone import ContinuousBackbone
from model.uecd.continuous_noise import ContinuousNoiseSchedule
from model.uecd.discrete_head import DiscreteHead
from model.uecd.forward_process import forward_process
from model.uecd.loss import UECDLossResult, uecd_loss


@dataclass
class UECDOutput:
    loss: torch.Tensor
    num_tokens: int
    loss_detail: UECDLossResult


class UECDModel(nn.Module):
    """Unified Embedding-Space Co-Diffusion model.

    The token embedding is shared: it serves as both the discrete vocabulary
    lookup and the continuous diffusion space. No external encoder needed.
    """

    def __init__(self, config: UECDConfig):
        super().__init__()
        self.config = config

        # Shared embedding: vocab + 1 for [MASK] absorbing token
        vocab_size = config.num_tokens + 1
        self.mask_token_id = config.num_tokens
        self.embedding = nn.Embedding(vocab_size, config.hidden_size)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

        # Continuous noise schedule
        self.continuous_noise: ContinuousNoiseSchedule = config.continuous_noise.build()

        # Discrete noise schedule (reuses existing SEDD infrastructure)
        self.discrete_noise = get_noise(
            NoiseConfig(
                noise_type=config.discrete_noise_type,
                eps=config.discrete_noise_eps,
            )
        )

        # Discrete diffusion graph (absorbing)
        self.graph = get_graph(GraphType.ABSORBING, config.num_tokens)

        # Continuous backbone
        self.continuous_backbone = ContinuousBackbone(config.backbone)

        # Discrete head
        self.discrete_head = DiscreteHead(config.discrete_head)

    def forward(self, x0: torch.Tensor) -> UECDOutput:
        """Training forward pass.

        Args:
            x0: Clean token ids, shape (B, L).

        Returns:
            UECDOutput with total loss, token count, and loss breakdown.
        """
        batch_size, seq_len = x0.shape
        device = x0.device

        # Sample shared timestep
        t = (1 - self.config.sampling_eps) * torch.rand(
            batch_size, device=device
        ) + self.config.sampling_eps

        # Forward process: corrupt x0 -> zx_t
        fwd = forward_process(
            x0=x0,
            t=t,
            embedding_weight=self.embedding.weight,
            continuous_noise=self.continuous_noise,
            mask_token_id=self.mask_token_id,
        )

        # Continuous backbone: predict v
        backbone_out = self.continuous_backbone(fwd.zx_t, t)
        v_pred = backbone_out.v_pred

        # v-prediction target
        v_target = self.continuous_noise.v_target(fwd.x0_embed, fwd.eps, t)

        # Recover predicted eps for discrete head input
        eps_hat = self.continuous_noise.predict_eps(fwd.zx_t, v_pred, t)

        # Discrete head: predict concrete scores
        noise_corrected = fwd.zx_t - eps_hat
        log_score = self.discrete_head(
            noise_corrected, backbone_out.tap_hidden, t, fwd.x_t
        )

        # Discrete noise parameters (for SEDD loss weighting)
        sigma_disc, dsigma_disc = self.discrete_noise(t)

        # Combined loss
        loss_result = uecd_loss(
            v_pred=v_pred,
            v_target=v_target,
            log_score=log_score,
            graph=self.graph,
            continuous_noise=self.continuous_noise,
            t=t,
            sigma_disc=sigma_disc,
            dsigma_disc=dsigma_disc,
            x_t=fwd.x_t,
            x0=x0,
        )

        return UECDOutput(
            loss=loss_result.total.sum(),
            num_tokens=batch_size * seq_len,
            loss_detail=loss_result,
        )
