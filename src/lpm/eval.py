from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def pairwise_metrics(pos_reward: torch.Tensor, neg_reward: torch.Tensor) -> dict[str, float]:
    margins = pos_reward - neg_reward
    return {
        "accuracy": float((margins > 0).float().mean().item()),
        "mean_margin": float(margins.mean().item()),
        "median_margin": float(margins.median().item()),
        "count": float(margins.numel()),
    }


def grouped_pairwise_metrics(
    pos_reward: torch.Tensor,
    neg_reward: torch.Tensor,
    groups: list[str],
) -> dict[str, dict[str, float]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    out: dict[str, dict[str, float]] = {}
    for group, indices in by_group.items():
        idx = torch.tensor(indices, dtype=torch.long)
        out[group] = pairwise_metrics(pos_reward[idx], neg_reward[idx])
    return out


def as_serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            result[key] = as_serializable_metrics(value)
        elif isinstance(value, float):
            result[key] = round(value, 6)
        else:
            result[key] = value
    return result
