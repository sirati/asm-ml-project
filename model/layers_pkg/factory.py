from __future__ import annotations

from model.layers_pkg.attn import AttnCrossLayer, AttnSelfLayer
from model.layers_pkg.config import LayerBackend, LayerConfig
from model.layers_pkg.flash_attn import FlashAttnCrossLayer, FlashAttnSelfLayer
from model.layers_pkg.mamba import MambaCrossLayer, MambaOnlySelfLayer, MambaSelfLayer
from model.layers_pkg.protocols import CrossSequenceLayer, SequenceLayer

_SELF_LAYER_REGISTRY: dict[LayerBackend, type[SequenceLayer]] = {
    LayerBackend.ATTN: AttnSelfLayer,
    LayerBackend.FLASH_ATTN: FlashAttnSelfLayer,
    LayerBackend.MAMBA: MambaSelfLayer,
    LayerBackend.MAMBA_ONLY: MambaOnlySelfLayer,
}

_CROSS_LAYER_REGISTRY: dict[LayerBackend, type[CrossSequenceLayer]] = {
    LayerBackend.ATTN: AttnCrossLayer,
    LayerBackend.FLASH_ATTN: FlashAttnCrossLayer,
    LayerBackend.MAMBA: MambaCrossLayer,
}


def make_self_layer(dim: int, config: LayerConfig) -> SequenceLayer:
    cls = _SELF_LAYER_REGISTRY.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown self-layer backend: {config.backend}")
    return cls(dim, config)


def make_cross_layer(d_q: int, d_kv: int, config: LayerConfig) -> CrossSequenceLayer:
    cls = _CROSS_LAYER_REGISTRY.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown cross-layer backend: {config.backend}")
    return cls(d_q, d_kv, config)
