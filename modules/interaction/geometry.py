# src/dvfi/models/future/future_encoder.py
from __future__ import annotations
from typing import Tuple
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from modules.future.keypoint import build_kinematics, select_keypoints


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()  # channel-first
        return self.weight.view(1, -1, 1) * x * rms


class ConvGLUBlock(nn.Module):
    def __init__(self, d_model: int, k: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.norm = RMSNorm(d_model)
        self.conv = nn.Conv1d(d_model, 2 * d_model, kernel_size=k, padding=pad, dilation=dilation)
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H]
        h = self.norm(x)
        h = self.conv(h)
        a, b = h.chunk(2, dim=1)
        h = a * torch.sigmoid(b)     # GLU
        h = self.proj(h)
        h = self.drop(h)
        return x + h


def sinusoidal_time_embedding(H: int, d_t: int, device=None) -> torch.Tensor:
    """
    Returns [d_t, H] sin-cos embedding (channel-first for 1D conv add).
    """
    pos = torch.arange(H, device=device).float()  # [H]
    div = torch.exp(torch.arange(0, d_t, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / d_t))
    pe = torch.zeros(d_t, H, device=device)
    pe[0::2, :] = torch.sin(pos.unsqueeze(0) * div.unsqueeze(1))
    pe[1::2, :] = torch.cos(pos.unsqueeze(0) * div.unsqueeze(1))
    return pe  # [d_t, H]


class FutureEncoder(nn.Module):
    """
    Encode absolute future trajectory Y_abs into a compact feature sequence f(t)
    evaluated at Tk key timestamps. Independent from history SSM.
    """
    def __init__(self,
                 d_in_feats: int = 8,     # from build_kinematics
                 d_time: int = 32,
                 d_model: int = 128,
                 layers: int = 2,
                 kernel_size: int = 5,
                 dilations=(1, 2),
                 dropout: float = 0.1,
                 Tk: int = 6,
                 time_embed: str = "sin"):
        super().__init__()
        self.Tk = Tk
        self.time_embed = time_embed
        self.d_model = d_model
        self.proj_in = nn.Conv1d(d_in_feats, d_model, kernel_size=1)
        self.t_proj = nn.Conv1d(d_time, d_model, kernel_size=1)
        self.blocks = nn.ModuleList([
            ConvGLUBlock(d_model, k=kernel_size,
                         dilation=dilations[i % len(dilations)],
                         dropout=dropout)
            for i in range(layers)
        ])

    def forward(self, Y_abs: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Args:
          Y_abs: [B, A, K, M, H, 2]
        Returns:
          f_seq: [B, A, K, M, Tk, d_model]   # features at Tk key timestamps
        """
        device = Y_abs.device
        B, A, K, M, H, _ = Y_abs.shape

        # 1) kinematic feats over full H
        feats = build_kinematics(Y_abs, dt=dt)            # [B,A,K,M,H,8]
        feats = feats.permute(0,1,2,3,5,4).contiguous()   # [B,A,K,M,8,H]

        # 2) project to model dim & add time embedding
        x = feats.view(B * A * K * M, feats.size(-2), H)  # [N, C=8, H]
        x = self.proj_in(x)                               # [N, d_model, H]

        if self.time_embed == "sin":
            pe = sinusoidal_time_embedding(H, self.t_proj.in_channels, device=device)  # [d_time, H]
            pe = self.t_proj(pe.unsqueeze(0)).squeeze(0)                               # [d_model, H]
            x = x + pe.unsqueeze(0).expand_as(x)
        else:
            # 可扩展 learned embedding
            pass

        # 3) temporal conv backbone over full H
        for blk in self.blocks:
            x = blk(x)                                    # [N, d_model, H]

        # 4) select Tk keypoints & gather
        idx = select_keypoints(Y_abs, Tk=self.Tk, method="uniform")  # [Tk]
        # gather along time axis
        # x: [N, C, H] -> [N, C, Tk]
        x_k = x.index_select(dim=2, index=idx)            # [N, d_model, Tk]

        # 5) reshape back
        x_k = x_k.permute(0, 2, 1).contiguous()           # [N, Tk, d_model]
        f_seq = x_k.view(B, A, K, M, self.Tk, self.d_model)
        return f_seq