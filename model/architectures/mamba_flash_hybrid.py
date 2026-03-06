from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.positional import SinusoidalPositionalEncoding
from model.components.subset_masking import (
    SubsetMaskResult,
    create_subset_masks,
    estimate_subset_activation_bytes,
    subset_seq_len,
)
from model.layers_pkg import (
    LayerBackend,
    LayerConfig,
    SequenceLayer,
    make_cross_layer,
    make_self_layer,
)
from model.memory import (
    MIXED_BF16,
    DTypeConfig,
    LayerMemoryEstimate,
    MemoryEstimate,
    MemoryMode,
    layernorm_param_count,
    linear_param_count,
    optimizer_bytes_for_params,
)


@dataclass
class MambaFlashHybridConfig:
    vocab_size: int = 50257
    max_seq_len: int = 512
    hidden_size: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    widening_factor: int = 4
    num_encoder_stacks: int = 3
    mamba_layers_per_stack: int = 5
    num_decoder_cross_layers: int = 2
    num_decoder_self_layers: int = 2
    num_subset_splits: int = 4
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    @property
    def flash_attn_config(self) -> LayerConfig:
        return LayerConfig(
            backend=LayerBackend.FLASH_ATTN,
            num_heads=self.num_heads,
            dropout=self.dropout,
            widening_factor=self.widening_factor,
        )

    @property
    def mamba_only_config(self) -> LayerConfig:
        return LayerConfig(
            backend=LayerBackend.MAMBA_ONLY,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand=self.mamba_expand,
        )


class HybridEncoder(nn.Module):
    """Encoder with repeating blocks of [N mamba-only layers, 1 flash-attn layer]."""

    def __init__(self, config: MambaFlashHybridConfig):
        super().__init__()
        layers: list[SequenceLayer] = []
        for _ in range(config.num_encoder_stacks):
            for _ in range(config.mamba_layers_per_stack):
                layers.append(
                    make_self_layer(config.hidden_size, config.mamba_only_config)
                )
            layers.append(make_self_layer(config.hidden_size, config.flash_attn_config))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def estimate_memory(
        self,
        batch: int,
        seq_len: int,
        mode: MemoryMode,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> LayerMemoryEstimate:
        total = LayerMemoryEstimate()
        for layer in self.layers:
            total += layer.estimate_memory(batch, seq_len, mode, checkpoint, dtypes)
        return total


class FlashAttnDecoder(nn.Module):
    """Decoder: cross-attention layers followed by self-attention layers, all flash-attn with softmax-1."""

    def __init__(self, config: MambaFlashHybridConfig):
        super().__init__()
        dim = config.hidden_size
        fa_cfg = config.flash_attn_config

        self.cross_layers = nn.ModuleList(
            [
                make_cross_layer(dim, dim, fa_cfg)
                for _ in range(config.num_decoder_cross_layers)
            ]
        )
        self.self_layers = nn.ModuleList(
            [
                make_self_layer(dim, fa_cfg)
                for _ in range(config.num_decoder_self_layers)
            ]
        )

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        for cross_layer in self.cross_layers:
            x = cross_layer(x, encoder_out)
        for self_layer in self.self_layers:
            x = self_layer(x)
        return x

    def estimate_memory(
        self,
        batch: int,
        seq_len_q: int,
        seq_len_kv: int,
        mode: MemoryMode,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> LayerMemoryEstimate:
        total = LayerMemoryEstimate()
        for layer in self.cross_layers:
            total += layer.estimate_memory(
                batch, seq_len_q, seq_len_kv, mode, checkpoint, dtypes
            )
        for layer in self.self_layers:
            total += layer.estimate_memory(batch, seq_len_q, mode, checkpoint, dtypes)
        return total


@dataclass
class MambaFlashHybridOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    num_tokens: int


class MambaFlashHybridModel(nn.Module):
    def __init__(self, config: MambaFlashHybridConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.hidden_size, config.max_seq_len
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.encoder = HybridEncoder(config)
        self.decoder = FlashAttnDecoder(config)

        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> MambaFlashHybridOutput:
        tok_emb = self.token_embedding(input_ids)
        pos_enc = self.pos_encoding(input_ids.shape[1])
        x = self.embedding_dropout(tok_emb + pos_enc)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder_out = self.encoder(x)

            subset = create_subset_masks(
                encoder_out=encoder_out,
                token_embeddings=self.embedding_dropout(tok_emb + pos_enc),
                token_ids=input_ids,
                num_splits=self.config.num_subset_splits,
            )

            decoder_out = self.decoder(subset.decoder_input, subset.masked_encoder_out)

        logits = self.output_proj(self.output_norm(decoder_out))

        shift_logits = logits[:, :-1].contiguous()
        shift_targets = subset.target_ids[:, 1:].contiguous()
        num_tokens = shift_targets.numel()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_targets.view(-1),
            reduction="sum",
        )

        return MambaFlashHybridOutput(loss=loss, logits=logits, num_tokens=num_tokens)

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        training: bool = True,
        checkpoint: bool = False,
        dtypes: DTypeConfig = MIXED_BF16,
    ) -> MemoryEstimate:
        mode = MemoryMode.TRAINING if training else MemoryMode.INFERENCE
        dim = self.config.hidden_size
        N = self.config.num_subset_splits
        sub_len = subset_seq_len(seq_len, N)
        decoder_batch = batch_size * N

        # --- Parameters: embedding + output_norm (output_proj shares embedding weights) ---
        embedding_param_count = self.config.vocab_size * dim
        output_norm_param_count = layernorm_param_count(dim)
        extra_param_count = embedding_param_count + output_norm_param_count

        # --- Delegate to encoder and decoder ---
        enc_est = self.encoder.estimate_memory(
            batch_size, seq_len, mode, checkpoint, dtypes
        )
        dec_est = self.decoder.estimate_memory(
            decoder_batch, sub_len, sub_len, mode, checkpoint, dtypes
        )

        total_param_bytes = (
            enc_est.param_bytes + dec_est.param_bytes + extra_param_count * dtypes.param
        )

        # --- Activations ---
        # Embedding + positional encoding activation
        embed_act = self.pos_encoding.estimate_activation_bytes(
            batch_size, seq_len, dtypes
        )
        # PE buffer (fp32, not a parameter)
        pe_buffer = self.pos_encoding.estimate_buffer_bytes(dtypes)
        # Subset masking tensors
        subset_act = estimate_subset_activation_bytes(
            batch_size, seq_len, dim, N, dtypes
        )
        # Output logits: decoder_batch * sub_len * vocab_size * activation_dtype
        logits_act = (
            decoder_batch * sub_len * self.config.vocab_size * dtypes.activation
        )

        total_act = (
            enc_est.activation_bytes
            + dec_est.activation_bytes
            + embed_act
            + pe_buffer
            + subset_act
            + logits_act
        )

        # --- Gradients ---
        total_grad = enc_est.gradient_bytes + dec_est.gradient_bytes
        if training:
            total_grad += extra_param_count * dtypes.gradient

        # --- Optimizer: master weights (fp32) + m (fp32) + v (fp32) ---
        opt_bytes = 0
        if training:
            total_param_count = total_param_bytes // dtypes.param
            opt_bytes = optimizer_bytes_for_params(total_param_count, dtypes)

        return MemoryEstimate(
            param_bytes=total_param_bytes,
            optimizer_bytes=opt_bytes,
            activation_bytes=total_act,
            gradient_bytes=total_grad,
        )
