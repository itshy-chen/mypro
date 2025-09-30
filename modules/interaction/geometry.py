from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class NeighbourInfo:
    """Container storing the output of :func:`future_knn`."""

    indices: torch.Tensor
    distances: torch.Tensor
    mask: torch.Tensor
    k: int


def _summarise_trajectory(Y_abs: torch.Tensor) -> torch.Tensor:
    """Collapse mode/style dimensions and return ``[B, A, H, 2]`` trajectories."""

    if Y_abs.dim() != 6:
        raise ValueError("`Y_abs` is expected to be of shape [B, A, K, M, H, 2].")
    return Y_abs.mean(dim=(2, 3))


def future_knn(
    Y_abs: torch.Tensor,
    *,
    K: int = 16,
    fallback_k: int = 8,
    eps: float = 1e-6,
) -> NeighbourInfo:
    """Select the ``K`` closest neighbours using Euclidean distance.

    When fewer than ``K`` neighbours are available we reduce the number of
    neighbours to ``fallback_k`` (or the actual count if smaller).  The function
    returns indices and distances that can later be used for gathering
    neighbour-specific information.
    """

    traj = _summarise_trajectory(Y_abs)
    B, A, H, _ = traj.shape
    device = traj.device

    if A <= 1:  # pragma: no cover - degenerate case
        empty = torch.empty(B, A, 0, device=device, dtype=torch.long)
        return NeighbourInfo(empty, empty.float(), empty.bool(), 0)

    anchor = traj[..., -1, :]  # [B, A, 2]
    dist = torch.cdist(anchor, anchor, p=2)
    dist = dist.clamp_min(eps)

    eye = torch.eye(A, device=device, dtype=torch.bool)
    dist = dist.masked_fill(eye.unsqueeze(0), float("inf"))

    available = A - 1
    if available < K:
        k_eff = min(max(fallback_k, 1), available) if available >= fallback_k else available
    else:
        k_eff = K

    if k_eff <= 0:
        empty = torch.empty(B, A, 0, device=device, dtype=torch.long)
        return NeighbourInfo(empty, empty.float(), empty.bool(), 0)

    distances, indices = dist.topk(k=k_eff, largest=False)
    mask = torch.isfinite(distances)
    distances = distances.masked_fill(~mask, 0.0)
    return NeighbourInfo(indices=indices, distances=distances, mask=mask, k=k_eff)


def _gather_neighbour_trajectories(
    traj: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather neighbour trajectories according to ``indices``."""

    B, A, H, C = traj.shape
    _, _, K = indices.shape
    expanded = traj.unsqueeze(1).expand(B, A, A, H, C)
    gather_idx = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, C)
    neighbours = torch.gather(expanded, 2, gather_idx)
    return neighbours  # [B, A, K, H, C]


def geom_soft_prior(
    Y_abs: torch.Tensor,
    neighbours: NeighbourInfo,
    *,
    temperature: float = 2.0,
) -> dict:
    """Compute a geometry-based soft prior over the selected neighbours.

    The bias is derived from the final-step distances between the target agent
    and each neighbour.  The function returns both the softmax weights and the
    relative trajectories that can act as additional conditioning features for
    attention modules.
    """

    traj = _summarise_trajectory(Y_abs)
    neigh_traj = _gather_neighbour_trajectories(traj, neighbours.indices)

    target_final = traj[..., -1, :].unsqueeze(2)  # [B, A, 1, 2]
    neighbour_final = neigh_traj[..., -1, :]  # [B, A, K, 2]

    rel_final = neighbour_final - target_final
    logits = -torch.norm(rel_final, dim=-1) / max(temperature, 1e-6)

    mask = neighbours.mask
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))

    weights = F.softmax(logits, dim=-1)

    rel_traj = neigh_traj - traj.unsqueeze(2)

    return {
        "weights": weights,
        "logits": logits,
        "relative_traj": rel_traj,
        "indices": neighbours.indices,
        "mask": mask,
        "k": neighbours.k,
    }
