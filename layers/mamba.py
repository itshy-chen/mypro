"""Temporal encoding layers built around Mamba blocks."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

try:  # pragma: no cover - optional dependency probing
    from torch.nn import RMSNorm as TorchRMSNorm  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - PyTorch < 2.0 fallback
    TorchRMSNorm = None

try:  # pragma: no cover - optional dependency probing
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - bubbled up at runtime
    Mamba = None  # type: ignore[assignment]
    _mamba_import_error = exc
else:
    _mamba_import_error = None


class RMSNorm(nn.Module):
    """Compatibility RMSNorm implementation."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if TorchRMSNorm is not None:
            self._impl: Optional[nn.Module] = TorchRMSNorm(d_model, eps=eps)
            self.weight = None  # type: ignore[assignment]
        else:
            self._impl = None
            self.weight = nn.Parameter(torch.ones(d_model))
            self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if self._impl is not None:
            return self._impl(x)

        # Manual RMSNorm.
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(norm + self.eps)
        return x_norm * self.weight


class MambaBlock(nn.Module):
    """A single RMSNorm → Mamba → Dropout → Residual block."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        if Mamba is None:  # pragma: no cover - handled during module init
            raise ImportError(
                "MambaBlock requires the `mamba_ssm` package."
            ) from _mamba_import_error

        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            mask = mask.bool()
        residual = x
        if mask is not None:
            residual = residual * mask.unsqueeze(-1).to(residual.dtype)
        normed = self.norm(x)
        if mask is None:
            out = self.mamba(normed)
        else:
            out = self._run_segmented(normed, mask)

        out = self.dropout(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return residual + out

    def _run_segmented(self, x: Tensor, mask: Tensor) -> Tensor:
        """Apply Mamba only on valid segments so invalid steps do not update state."""

        orig_shape = x.shape
        seq_len = orig_shape[-2]
        d_model = orig_shape[-1]

        x_flat = x.reshape(-1, seq_len, d_model)
        mask_flat = mask.reshape(-1, seq_len).bool()
        outputs = torch.zeros_like(x_flat)

        for b_idx in range(x_flat.size(0)):
            valid = mask_flat[b_idx]
            if not valid.any():
                continue

            seq = x_flat[b_idx : b_idx + 1]
            start = 0
            while start < seq_len:
                while start < seq_len and not bool(valid[start]):
                    start += 1
                if start >= seq_len:
                    break
                end = start
                while end < seq_len and bool(valid[end]):
                    end += 1

                seg = seq[:, start:end]
                seg_out = self.mamba(seg)
                outputs[b_idx, start:end] = seg_out[0]
                start = end

        return outputs.reshape(orig_shape)


class FFNBlock(nn.Module):
    """Position-wise feed-forward block with SiLU activation."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            mask = mask.bool()
        residual = x
        if mask is not None:
            residual = residual * mask.unsqueeze(-1).to(residual.dtype)
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return residual + x


class AttentionPooling(nn.Module):
    """Masked attention pooling over temporal dimension."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        *batch_dims, seq_len, _ = x.shape
        x_flat = x.reshape(-1, seq_len, x.shape[-1])
        if mask is not None:
            mask_flat = mask.reshape(-1, seq_len).bool()
        else:
            mask_flat = None

        keys = self.key_proj(x_flat)
        values = self.value_proj(x_flat)
        query = self.query_proj(self.query)
        logits = torch.matmul(keys, query)

        if mask_flat is not None:
            logits = logits.masked_fill(~mask_flat, float("-inf"))
            weights = torch.softmax(logits, dim=-1)
            weights = torch.where(mask_flat, weights, torch.zeros_like(weights))
            valid_rows = mask_flat.any(dim=-1)
            if (~valid_rows).any():
                weights[~valid_rows] = 0.0
        else:
            weights = torch.softmax(logits, dim=-1)

        pooled = torch.sum(weights.unsqueeze(-1) * values, dim=-2)
        if mask_flat is not None and (~valid_rows).any():
            pooled[~valid_rows] = 0.0
        pooled = self.out_proj(pooled)
        return pooled.reshape(*batch_dims, -1)


def fuse_lastvalid_attn(
    last: Tensor, attn: Tensor, projection: nn.Module
) -> Tensor:
    """Fuse last-valid and attention pooled features via a learnable projection."""

    fused = torch.cat([last, attn], dim=-1)
    return projection(fused)


class MambaTimeEncoder(nn.Module):
    """Temporal encoder that produces per-step and agent-level features."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [MambaBlock(d_model, dropout), FFNBlock(d_model, dropout)]
                )
                for _ in range(n_layers)
            ]
        )
        self.attn_pool = AttentionPooling(d_model)
        self.fusion = nn.Linear(2 * d_model, d_model)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if x.dim() not in (3, 4):
            raise ValueError("`x` must be of shape [B, L, F] or [B, N, L, F].")

        if mask is not None and mask.shape != x.shape[:-1]:
            raise ValueError("Mask shape must match input batch dimensions.")

        if mask is not None:
            mask = mask.bool()
            mask_expanded = mask.unsqueeze(-1).to(x.dtype)
        else:
            mask_expanded = None

        x = self.input_proj(x)
        for mamba_block, ffn_block in self.layers:
            x = mamba_block(x, mask)
            x = ffn_block(x, mask)
            if mask_expanded is not None:
                x = x * mask_expanded

        H_seq = x
        if mask_expanded is not None:
            H_seq = H_seq * mask_expanded

        if mask is None:
            last_valid = H_seq[..., -1, :]
        else:
            valid_counts = mask.long().sum(dim=-1)
            idx = (valid_counts - 1).clamp(min=0)
            gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(
                *idx.shape, 1, H_seq.size(-1)
            )
            last_valid = torch.take_along_dim(H_seq, gather_idx, dim=-2).squeeze(-2)
            last_valid = torch.where(
                valid_counts.unsqueeze(-1) > 0,
                last_valid,
                torch.zeros_like(last_valid),
            )

        attn_summary = self.attn_pool(H_seq, mask)
        h_i = fuse_lastvalid_attn(last_valid, attn_summary, self.fusion)
        return H_seq, h_i
