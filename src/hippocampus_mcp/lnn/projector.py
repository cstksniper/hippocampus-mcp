# -*- coding: utf-8 -*-
"""Projection layer (Projector).

A simple linear projection layer. This functionality is already merged into
use inside MemoryCore; the standalone interface is kept here so it can be
reused on its own or for dimension alignment (e.g. projecting external
embeddings into the memory core's input dimension).
"""

import torch
import torch.nn as nn


class Projector(nn.Module):
    """Linear projection layer.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a linear projection to the input.

        Args:
            x: Input tensor; its last dimension must be input_dim.

        Returns:
            The projected tensor; its last dimension is output_dim.
        """
        return self.linear(x)
