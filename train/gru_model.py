"""GRU model: small Conv1d frontend + 2-layer bidirectional GRU residual head.

Output is the predicted residual r_hat = TVT_true - pfens_s5 in FEET
(internally normalized by TVT_SCALE; scaled back in forward).
~300k params for ~600 training sequences -- deliberately small.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TVT_SCALE = 25.0


class RefinerGRU(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 64, drop: float = 0.2):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(n_ch, 64, kernel_size=5, padding=2), nn.GELU())
        self.gru = nn.GRU(64, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.GELU(), nn.Dropout(drop),
            nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x (B, C, L) float32, lengths (B,) int64 -> r_hat (B, L) in feet."""
        h = self.frontend(x).transpose(1, 2)                     # (B, L, 64)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=x.shape[-1])   # (B, L, 2H)
        return self.head(out).squeeze(-1) * TVT_SCALE
