"""Higher-level attention building blocks.

Provides ModuleOutput, Residual, and init_parameters utilities
used by other model modules. The actual attention implementations
live in model.layers (pluggable backends).

Adapted from vendor/perceiver-io/perceiver/model/core/utils.py and
vendor/perceiver-io/perceiver/model/core/modules.py.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class ModuleOutput(OrderedDict):
    """Dict-like container for module outputs with attribute access."""

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, ModuleOutput):
            output.last_hidden_state = self.dropout(output.last_hidden_state) + args[0]
            return output
        return self.dropout(output) + args[0]


def init_parameters(module: nn.Module, init_scale: float) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)
