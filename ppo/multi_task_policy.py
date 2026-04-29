from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'silu':
        return nn.SiLU()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Unsupported activation: {name}')


def _mlp(input_dim: int, hidden: int, layers: int, activation: str) -> nn.Sequential:
    layers = max(1, int(layers))
    modules: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(layers):
        modules.append(nn.Linear(in_dim, hidden))
        modules.append(_activation(activation))
        in_dim = hidden
    return nn.Sequential(*modules)


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int = 2):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'attention hidden dim {dim} must be divisible by heads {heads}.')
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, _, dim = x.shape
        out = torch.zeros((batch, dim), dtype=x.dtype, device=x.device)
        valid = mask.sum(dim=1) > 0.5
        if valid.any():
            xv = x[valid]
            mv = mask[valid]
            q = self.query.expand(xv.shape[0], -1, -1)
            pooled, _ = self.attn(q, xv, xv, key_padding_mask=(mv < 0.5))
            out[valid] = self.norm(pooled[:, 0, :])
        return out


class MultiTaskPolicy(nn.Module):
    def __init__(
        self,
        self_dim: int = 82,
        ally_dim: int = 8,
        enemy_dim: int = 10,
        hidden: int = 128,
        attention_heads: int = 2,
        self_layers: int = 2,
        entity_layers: int = 1,
        trunk_layers: int = 2,
        activation: str = 'tanh',
    ):
        super().__init__()
        hidden = int(hidden)
        attention_heads = int(attention_heads)
        self.policy_config = {
            'hidden': hidden,
            'attention_heads': attention_heads,
            'self_layers': int(self_layers),
            'entity_layers': int(entity_layers),
            'trunk_layers': int(trunk_layers),
            'activation': str(activation).lower(),
        }
        self.self_net = _mlp(self_dim, hidden, self_layers, activation)
        self.ally_proj = _mlp(ally_dim, hidden, entity_layers, activation)
        self.enemy_proj = _mlp(enemy_dim, hidden, entity_layers, activation)
        self.ally_pool = MaskedAttentionPool(hidden, attention_heads)
        self.enemy_pool = MaskedAttentionPool(hidden, attention_heads)
        self.trunk = _mlp(hidden * 3, hidden, trunk_layers, activation)
        self.actor_heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(3)])
        self.critic = nn.Linear(hidden, 1)

    def encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.trunk(
            torch.cat(
                [
                    self.self_net(obs['self_state']),
                    self.ally_pool(self.ally_proj(obs['ally_features']), obs['ally_mask']),
                    self.enemy_pool(self.enemy_proj(obs['enemy_features']), obs['enemy_mask']),
                ],
                dim=-1,
            )
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        feat = self.encode(obs)
        heads = [head(feat) for head in self.actor_heads]
        return heads, self.critic(feat).squeeze(-1)

    def act(self, obs: dict[str, torch.Tensor], action_mask, deterministic: bool = False):
        heads, value = self.forward(obs)
        logprob = torch.zeros((obs['self_state'].shape[0],), dtype=torch.float32, device=obs['self_state'].device)
        entropy = torch.zeros_like(logprob)
        actions = []
        active = {int(index) for index in action_mask}
        for index, logits in enumerate(heads):
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite actor logits detected.')
            dist = Categorical(logits=logits)
            if index in active:
                action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
                logprob = logprob + dist.log_prob(action)
                entropy = entropy + dist.entropy()
            else:
                action = torch.zeros((obs['self_state'].shape[0],), dtype=torch.long, device=logits.device)
            actions.append(action)
        return torch.stack(actions, dim=-1), logprob, entropy, value

    def evaluate_actions(self, obs: dict[str, torch.Tensor], action: torch.Tensor, action_mask):
        heads, value = self.forward(obs)
        action = action.long()
        logprob = torch.zeros((obs['self_state'].shape[0],), dtype=torch.float32, device=obs['self_state'].device)
        entropy = torch.zeros_like(logprob)
        active = {int(index) for index in action_mask}
        for index, logits in enumerate(heads):
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite actor logits detected during evaluate_actions.')
            if index not in active:
                continue
            dist = Categorical(logits=logits)
            logprob = logprob + dist.log_prob(action[:, index].clamp(0, 2))
            entropy = entropy + dist.entropy()
        return logprob, entropy, value
