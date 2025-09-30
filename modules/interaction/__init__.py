
"""Interaction modules used by the trajectory prediction pipeline."""

from .geometry import NeighbourInfo, future_knn, geom_soft_prior
from .cross_attention import (
    AgentCrossAttention,
    FutureFutureCrossAttention,
    HistoryFutureCrossAttention,
)

__all__ = [
    "AgentCrossAttention",
    "FutureFutureCrossAttention",
    "HistoryFutureCrossAttention",
    "NeighbourInfo",
    "future_knn",
    "geom_soft_prior",
]