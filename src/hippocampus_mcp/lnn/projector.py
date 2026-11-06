# -*- coding: utf-8 -*-
"""投影层 (Projector).

简单的线性投影层. 该功能在 MemoryCore 中已合并使用, 此处保留独立接口
以便单独复用或做维度对齐 (例如将外部嵌入投影到记忆核的输入维度).
"""

import torch
import torch.nn as nn


class Projector(nn.Module):
    """线性投影层.

    Args:
        input_dim: 输入维度.
        output_dim: 输出维度.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入做线性投影.

        Args:
            x: 输入张量, 最后一维需为 input_dim.

        Returns:
            投影后的张量, 最后一维为 output_dim.
        """
        return self.linear(x)
