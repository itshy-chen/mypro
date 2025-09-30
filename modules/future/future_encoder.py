from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .keypoint import build_kinematics, select_keypoints


class RMSNorm(nn.Module):
    """Root-mean-square normalisation over the channel dimension."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return self.weight.view(1, -1, 1) * x * rms


class ConvGLUBlock(nn.Module):
    """Depth-wise gated convolutional block used by the future encoder."""

    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.norm = RMSNorm(d_model)
        self.conv = nn.Conv1d(
            d_model,
            2 * d_model,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.conv(h)
        a, b = h.chunk(2, dim=1)
        h = a * torch.sigmoid(b)
        h = self.proj(h)
        h = self.drop(h)
        return x + h


def sinusoidal_time_embedding(H: int, d_t: int, device=None) -> torch.Tensor:
    """Construct a sinusoidal embedding with ``H`` positions and ``d_t`` dims."""

    position = torch.arange(H, device=device, dtype=torch.float32)
    div_term = torch.exp(
        torch.arange(0, d_t, 2, device=device, dtype=torch.float32)
        * (-(torch.log(torch.tensor(10000.0, device=device)) / d_t))
    )
    emb = torch.zeros(d_t, H, device=device)
    emb[0::2, :] = torch.sin(position.unsqueeze(0) * div_term.unsqueeze(1))
    emb[1::2, :] = torch.cos(position.unsqueeze(0) * div_term.unsqueeze(1))
    return emb


class FutureEncoder(nn.Module):
    """Encode absolute future trajectories into fixed-size feature sequences."""

    def __init__(
        self,
        *,
        d_in_feats: int = 8,
        d_time: int = 32,
        d_model: int = 128,
        layers: int = 2,
        kernel_size: int = 5,
        dilations: Tuple[int, ...] = (1, 2),
        dropout: float = 0.1,
        Tk: int = 6,
        time_embed: str = "sin",
    ) -> None:
        super().__init__()
        self.Tk = Tk
        self.time_embed = time_embed
        self.d_model = d_model

        self.proj_in = nn.Conv1d(d_in_feats, d_model, kernel_size=1)
        self.t_proj = nn.Conv1d(d_time, d_model, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                ConvGLUBlock(
                    d_model,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(layers)
            ]
        )

    def forward(self, Y_abs: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """Encode candidate futures ``Y_abs`` into ``[B, A, K, M, Tk, D]`` features."""

        device = Y_abs.device
        B, A, K, M, H, _ = Y_abs.shape

        feats = build_kinematics(Y_abs, dt=dt)
        feats = feats.permute(0, 1, 2, 3, 5, 4).contiguous()  # [B,A,K,M,C,H]

        x = feats.view(B * A * K * M, feats.size(-2), H)
        x = self.proj_in(x)

        if self.time_embed == "sin":
            pe = sinusoidal_time_embedding(H, self.t_proj.in_channels, device=device)
            pe = self.t_proj(pe.unsqueeze(0)).squeeze(0)
            x = x + pe.unsqueeze(0).expand_as(x)
        else:  # pragma: no cover - placeholder for learned embeddings
            raise NotImplementedError("Only sinusoidal time embedding is supported.")

        for block in self.blocks:
            x = block(x)

        idx = select_keypoints(Y_abs, Tk=self.Tk, method="uniform")
        x_k = x.index_select(dim=2, index=idx)

        x_k = x_k.permute(0, 2, 1).contiguous()
        f_seq = x_k.view(B, A, K, M, self.Tk, self.d_model)
        return f_seq
