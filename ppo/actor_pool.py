from __future__ import annotations

import numpy as np
import torch

from ppo.registry_io import load_policy_bundle


class ActorPool:
    def __init__(self, action_registry, device='cpu'):
        self.device = device
        self.models = {}
        self.action_masks = {}
        self.obs_meta = {}
        for action_id in action_registry.list_actions():
            policy, action_spec, obs_meta = load_policy_bundle(action_registry.bundle_dir(action_id), device=device)
            self.models[action_id] = policy
            self.action_masks[action_id] = list(action_spec.action_mask)
            self.obs_meta[action_id] = obs_meta

    def batch_act(self, requests, deterministic=True):
        groups = {}
        for req in requests:
            groups.setdefault(req['action_id'], []).append(req)
        outputs = {}
        for action_id, group in groups.items():
            batch = {key: np.stack([g['obs'][key] for g in group], axis=0) for key in group[0]['obs'].keys()}
            batch_t = {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in batch.items()}
            with torch.no_grad():
                acts, _, _, _ = self.models[action_id].act(batch_t, self.action_masks[action_id], deterministic=deterministic)
            acts_np = acts.detach().cpu().numpy().astype(np.int64)
            for req, act in zip(group, acts_np):
                outputs[int(req['red_id'])] = act.tolist()
        return outputs
