# src/dvfi/models/decoder/conv_decoder.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utility: RMSNorm over channel dim
# ---------------------------
class RMSNorm(nn.Module):
    """RMSNorm over last dim. For [N,T,C], reshape to [-1,C] then norm."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = x * rms
        return self.weight * x_norm


# ---------------------------
# FiLM/AdaLN adapter (channel-wise)
# ---------------------------
class FiLMAdapter(nn.Module):
    """Apply channel-wise affine: y = (1+gamma)*x + beta.
    gamma/beta are [N,C] and broadcast to time axis."""
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T], gamma/beta: [N, C]
        return (1.0 + gamma).unsqueeze(-1) * x + beta.unsqueeze(-1)


# ---------------------------
# Conv Block: RMSNorm -> Conv1d -> GLU -> Dropout -> Residual
# ---------------------------
class ConvGLUBlock(nn.Module):
    def __init__(self, d_model: int, k: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (k - 1) // 2  # keep length
        self.norm = RMSNorm(d_model)
        self.conv = nn.Conv1d(d_model, 2 * d_model, kernel_size=k, padding=pad, dilation=dilation)
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, gamma: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None,
                film: Optional[FiLMAdapter] = None) -> torch.Tensor:
        # x: [N, C, T]
        # Pre-norm in (N,T,C)
        n, c, t = x.shape
        x_ln = self.norm(x.transpose(1, 2)).transpose(1, 2)  # [N,C,T]
        if film is not None and gamma is not None and beta is not None:
            x_ln = film(x_ln, gamma, beta)                   # FiLM modulation
        h = self.conv(x_ln)                                  # [N,2C,T]
        a, b = h.chunk(2, dim=1)
        h = a * torch.sigmoid(b)                             # GLU
        h = self.proj(h)                                     # [N,C,T]
        h = self.drop(h)
        return x + h                                         # residual


# ---------------------------
# Time embedding (learned)
# ---------------------------
class LearnedTimeEmbedding(nn.Module):
    def __init__(self, H: int, d_t: int):
        super().__init__()
        self.emb = nn.Embedding(H, d_t)

    def forward(self, H: int, device=None) -> torch.Tensor:
        # returns [H, d_t]
        idx = torch.arange(H, device=device)
        return self.emb(idx)


# ---------------------------
# Initial future decoder (parallel K x M)
# ---------------------------
class ConvDecoderPoints(nn.Module):
    """
    Parallel Conv1D decoder that regresses Δxy for H steps and integrates to absolute (x,y).
    Used for r=1 initial future Y^(0) (no interaction context).
    Also reusable for r>=1 with context via forward_with_ctx (optional extension).

    forward_initial inputs:
        h_tilde: [B, A, D]        # history SSM + Map late-fusion
        m:       [B, A, Dm]       # map vector (for conditioning only)
        mode_emb:[K, Dk]          # self trajectory "mode" embeddings (K modes, e.g., go/left/right/merge)
        z:       [B, A, M, Dz]    # continuous style latents (shared across K)
    returns:
        Y0: [B, A, K, M, H, 2]    # absolute coordinates in agent-centric frame
        S0: [B, A, K, M]          # raw scores (you can rescore later with safety head)
    """
    def __init__(self,
                 H: int = 30,
                 d_in_h: int = 256,
                 d_in_m: int = 128,
                 d_in_k: int = 64,
                 d_in_z: int = 16,
                 d_model: int = 256,
                 d_time: int = 32,
                 layers: int = 4,
                 kernel_size: int = 5,
                 dilations: Tuple[int, ...] = (1, 2, 4, 1),
                 dropout: float = 0.1,
                 use_film: bool = True):
        super().__init__()
        self.H = H
        self.use_film = use_film

        # Condition fusion -> FiLM parameters
        d_cond = d_in_h + d_in_m + d_in_k + d_in_z
        d_xi = d_model
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_cond, d_xi),
            nn.SiLU(),
            nn.Linear(d_xi, d_xi),
        )
        if use_film:
            self.to_gamma = nn.Linear(d_xi, d_model)
            self.to_beta  = nn.Linear(d_xi, d_model)
            self.film = FiLMAdapter()
        else:
            self.to_gamma = self.to_beta = None
            self.film = None

        # Time embeddings -> seed token stream
        self.tok_proj = nn.Linear(d_time, d_model)
        self.t_embed = LearnedTimeEmbedding(H, d_time)

        # Conv blocks
        self.blocks = nn.ModuleList([
            ConvGLUBlock(d_model, k=kernel_size,
                         dilation=dilations[i % len(dilations)],
                         dropout=dropout)
            for i in range(layers)
        ])

        # Heads: Δxy and score
        self.out_head = nn.Linear(d_model, 2)    # per step
        self.score_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    # ------------- helpers -------------
    @staticmethod
    def _broadcast_km(h: torch.Tensor, K: int, M: int) -> torch.Tensor:
        # h: [B,A,D] -> [B,A,K,M,D]
        return (h.unsqueeze(2).unsqueeze(3)
                  .expand(-1, -1, K, M, -1))

    @staticmethod
    def _expand_mode(mode_emb: torch.Tensor, B: int, A: int, M: int) -> torch.Tensor:
        # mode_emb: [K, Dk] -> [B,A,K,M,Dk]
        K, Dk = mode_emb.shape
        return (mode_emb.view(1, 1, K, 1, Dk)
                        .expand(B, A, K, M, Dk))

    @staticmethod
    def _repeat_time(tokens: torch.Tensor, N: int) -> torch.Tensor:
        # tokens: [H, D] -> [N, D, H]
        H, D = tokens.shape
        x = tokens.unsqueeze(0).repeat(N, 1, 1)       # [N, H, D]
        return x.transpose(1, 2).contiguous()         # [N, D, H]

    @staticmethod
    def _delta_to_abs(delta_xy: torch.Tensor) -> torch.Tensor:
        # delta_xy: [N, H, 2] -> abs coords with origin at (0,0)
        return torch.cumsum(delta_xy, dim=1)

    # ------------- core -------------
    def forward_initial(self,
                        h_tilde: torch.Tensor,   # [B,A,D]
                        m: torch.Tensor,         # [B,A,Dm]
                        mode_emb: torch.Tensor,  # [K,Dk]
                        z: torch.Tensor          # [B,A,M,Dz]
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = h_tilde.device
        B, A, D = h_tilde.shape
        K, Dk = mode_emb.shape
        M = z.shape[2]
        H = self.H

        # Broadcast to [B,A,K,M,*]
        h_b = self._broadcast_km(h_tilde, K, M)           # [B,A,K,M,D]
        m_b = self._broadcast_km(m,       K, M)           # [B,A,K,M,Dm]
        q_b = self._expand_mode(mode_emb, B, A, M)        # [B,A,K,M,Dk]
        z_b = z.unsqueeze(2).expand(B, A, K, M, z.shape[-1])  # [B,A,K,M,Dz]

        # Condition fusion -> ξ -> (γ,β)
        cond = torch.cat([h_b, m_b, q_b, z_b], dim=-1)    # [B,A,K,M,Dsum]
        xi = self.cond_mlp(cond)                          # [B,A,K,M,D_model]
        if self.use_film:
            gamma = self.to_gamma(xi)                     # [B,A,K,M,D_model]
            beta  = self.to_beta(xi)                      # [B,A,K,M,D_model]
        else:
            gamma = beta = None

        # Build time tokens
        t_tok = self.t_embed(H, device=device)            # [H, d_time]
        t_tok = self.tok_proj(t_tok)                      # [H, d_model]

        # Flatten (B,A,K,M) -> N for convolution
        N = B * A * K * M
        x = self._repeat_time(t_tok, N)                   # [N, d_model, H]

        # Prepare FiLM params as [N, C]
        if self.use_film:
            gamma_n = gamma.view(N, -1)                   # [N, d_model]
            beta_n  = beta.view(N, -1)                    # [N, d_model]
        else:
            gamma_n = beta_n = None

        # Conv stack
        for blk in self.blocks:
            x = blk(x, gamma=gamma_n, beta=beta_n, film=self.film)  # [N,d_model,H]

        # Per-step Δxy and integrate to abs
        delta = self.out_head(x.transpose(1, 2))          # [N,H,2]
        y_abs = self._delta_to_abs(delta)                 # [N,H,2]

        # Score head: pool (mean + last)
        feat_mean = x.mean(dim=-1)                        # [N,d_model]
        feat_last = x[..., -1]                            # [N,d_model]
        feat_pool = torch.cat([feat_mean, feat_last], dim=-1)
        s = self.score_head(feat_pool).squeeze(-1)        # [N]

        # Reshape back to [B,A,K,M,...]
        Y0 = y_abs.view(B, A, K, M, H, 2).contiguous()
        S0 = s.view(B, A, K, M).contiguous()
        return Y0, S0

    # (Optional) r>=1 with interaction context; placeholder for completeness
    def forward_with_ctx(self,
                         h_tilde: torch.Tensor,      # [B,A,D]
                         c_r: torch.Tensor,          # [B,A,Dc]
                         m: torch.Tensor,            # [B,A,Dm]
                         mode_emb: torch.Tensor,     # [K,Dk]
                         z: torch.Tensor             # [B,A,M,Dz]
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simply concatenate c_r into the condition and reuse the same stack
        # You can extend cond_mlp to accept Dc by wrapping this module or adding a small adapter.
        raise NotImplementedError("Use forward_initial for r=1. Implement forward_with_ctx for r>=1 as needed.")
