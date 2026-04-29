from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch

from env.specs import load_action_spec, save_action_spec
from ppo.multi_task_policy import MultiTaskPolicy


def save_policy_bundle(bundle_dir, policy, action_spec, obs_meta):
    path = Path(bundle_dir)
    path.mkdir(parents=True, exist_ok=True)
    policy_config = dict(getattr(policy, 'policy_config', {}))
    torch.save({'state_dict': policy.state_dict(), 'policy_config': policy_config}, path / 'model.pt')
    meta = dict(obs_meta)
    meta.setdefault('policy_config', policy_config)
    save_action_spec(path / 'action_spec.json', action_spec)
    (path / 'observation_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_checkpoint(path: Path, device):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='TypedStorage is deprecated.*', category=UserWarning)
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=device)


def load_policy_bundle(bundle_dir, device='cpu'):
    path = Path(bundle_dir)
    action_spec = load_action_spec(path / 'action_spec.json')
    obs_meta = json.loads((path / 'observation_meta.json').read_text(encoding='utf-8'))
    ckpt = _load_checkpoint(path / 'model.pt', device)
    state_dict = ckpt['state_dict']
    policy_config = dict(obs_meta.get('policy_config') or ckpt.get('policy_config') or {})
    policy = MultiTaskPolicy(
        self_dim=int(obs_meta['self_state_dim']),
        ally_dim=int(obs_meta['ally_entity_dim']),
        enemy_dim=int(obs_meta['enemy_entity_dim']),
        **policy_config,
    )
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy, action_spec, obs_meta
