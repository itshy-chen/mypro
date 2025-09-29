# src/dvfi/models/future/keypoint.py
from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def select_keypoints(y_abs: torch.Tensor, Tk: int, method: str = "uniform") -> torch.Tensor:
    """
    选择关键时刻索引（沿时间维 H）。
    Args:
        y_abs: [B, A, K, M, H, 2]  绝对坐标(以 agent 为中心)
        Tk:    关键时刻数，例如 6
        method:"uniform" | (预留 "arc")
    Returns:
        idx:   [Tk] 的整型索引（0..H-1 单调递增），可直接用于 gather
    """
    H = y_abs.shape[-2]
    if method != "uniform":
        raise NotImplementedError("Only 'uniform' is implemented for now.")
    if Tk <= 1:
        return torch.tensor([H - 1], device=y_abs.device, dtype=torch.long)
    # 包含起点与终点的等间隔采样
    grid = torch.linspace(0, H - 1, steps=Tk, device=y_abs.device)
    idx = torch.clamp(grid.round().long(), 0, H - 1)
    # 去重（极少数短序列 rounding 可能重复）
    _, unique_idx = torch.unique(idx, sorted=True, return_inverse=False, return_counts=False, return_index=True)
    idx = idx[unique_idx]
    # 若去重后数量小于 Tk，补齐（简单回填末尾）
    if idx.numel() < Tk:
        pad = torch.full((Tk - idx.numel(),), H - 1, device=y_abs.device, dtype=torch.long)
        idx = torch.cat([idx, pad], dim=0)
    return idx


def build_kinematics(y_abs: torch.Tensor, dt: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """
    从绝对轨迹构造时序运动学特征（SE(2) 对齐）。
    Features per-timestep C=8:
      [Δx, Δy, v_x, v_y, a_x, a_y, sin(heading), cos(heading)]
    Args:
        y_abs: [B, A, K, M, H, 2]
        dt:    采样周期(秒)
        eps:   数值稳定项
    Returns:
        feats: [B, A, K, M, H, 8]
    """
    # Δxy
    delta = y_abs[..., 1:, :] - y_abs[..., :-1, :]              # [...., H-1, 2]
    delta = torch.cat([torch.zeros_like(delta[..., :1, :]), delta], dim=-2)  # 对齐到 H

    # 速度/加速度（向量形式）
    v = delta / max(dt, eps)                                    # [...., H, 2]
    a = (v[..., 1:, :] - v[..., :-1, :]) / max(dt, eps)         # [...., H-1, 2]
    a = torch.cat([torch.zeros_like(a[..., :1, :]), a], dim=-2)

    # 航向（由速度方向得到）
    vx, vy = v[..., 0], v[..., 1]
    speed = torch.sqrt(vx * vx + vy * vy).clamp_min(eps)        # [...., H]
    ux, uy = vx / speed, vy / speed                             # 单位速度向量
    sin_hd, cos_hd = uy, ux                                     # sin=uy, cos=ux

    feats = torch.stack([
        delta[..., 0], delta[..., 1],
        v[..., 0],     v[..., 1],
        a[..., 0],     a[..., 1],
        sin_hd,        cos_hd,
    ], dim=-1)                                                  # [B,A,K,M,H,8]
    return feats
