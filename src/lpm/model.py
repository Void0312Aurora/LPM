from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, sizes: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                layers.append(nn.GELU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoLevelRewardRanker(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        action_dim: int,
        num_characters: int,
        character_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        state_noise_std: float = 0.0,
        state_mode: str = "full",
    ) -> None:
        super().__init__()
        if state_mode not in {"full", "global", "character", "action_only"}:
            raise ValueError(f"unsupported state_mode: {state_mode}")
        self.state_noise_std = state_noise_std
        self.state_mode = state_mode
        self.global_proj = MLP([input_dim, hidden_dim, state_dim], dropout=dropout)
        self.char_proj = MLP([input_dim, hidden_dim, state_dim], dropout=dropout)
        self.character_embedding = nn.Embedding(num_characters, character_dim)
        reward_input_dim = action_dim
        if state_mode in {"full", "global"}:
            reward_input_dim += state_dim
        if state_mode in {"full", "character"}:
            reward_input_dim += state_dim + character_dim
        self.reward = MLP([reward_input_dim, hidden_dim, hidden_dim // 2, 1], dropout=dropout)

    def encode_state(
        self,
        global_context: torch.Tensor,
        character_context: torch.Tensor,
        character_index: torch.Tensor,
    ) -> list[torch.Tensor]:
        pieces: list[torch.Tensor] = []
        if self.state_mode in {"full", "global"}:
            global_state = self.global_proj(global_context)
            pieces.append(global_state)
        if self.state_mode in {"full", "character"}:
            char_state = self.char_proj(character_context)
            char_id = self.character_embedding(character_index)
            pieces.extend([char_state, char_id])
        if self.training and self.state_noise_std > 0 and pieces:
            noisy: list[torch.Tensor] = []
            for piece in pieces:
                noisy.append(piece + torch.randn_like(piece) * self.state_noise_std)
            pieces = noisy
        return pieces

    def forward(
        self,
        global_context: torch.Tensor,
        character_context: torch.Tensor,
        character_index: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        pieces = self.encode_state(global_context, character_context, character_index)
        reward_input = torch.cat([*pieces, action], dim=-1) if pieces else action
        return self.reward(reward_input).squeeze(-1)


def softplus_ranking_loss(pos_reward: torch.Tensor, neg_reward: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    return torch.nn.functional.softplus(-(pos_reward - neg_reward - margin)).mean()
