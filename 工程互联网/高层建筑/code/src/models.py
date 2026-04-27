from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvBiLSTMDualHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_split: Tuple[int, int],
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h1, h2 = head_split
        if h1 + h2 != output_dim:
            raise ValueError("head_split 与 output_dim 不一致")

        self.output_dim = output_dim
        self.head_split = head_split

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.shared = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_a = nn.Linear(128, h1)
        self.head_b = nn.Linear(128, h2) if h2 > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = self.shared(x)

        a = self.head_a(x)
        if self.head_b is None:
            return a

        b = self.head_b(x)
        return torch.cat([a, b], dim=-1)
