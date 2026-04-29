from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F

from ppo.multi_task_policy import MultiTaskPolicy
from ppo.registry_io import save_policy_bundle
from ppo.swanlab_logger import SwanLabLogger

if not hasattr(torch.utils._pytree, 'register_pytree_node') and hasattr(torch.utils._pytree, '_register_pytree_node'):
    def _compat_register_pytree_node(node_type, flatten_fn, unflatten_fn, *args, **kwargs):
        return torch.utils._pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

    torch.utils._pytree.register_pytree_node = _compat_register_pytree_node

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total=None, desc='', disable=False):
            self.total = total
            self.desc = desc
            self.disable = disable

        def update(self, n=1):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def close(self):
            return None


@dataclass
class PPOConfig:
    episodes_per_env: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    ent_coef_final: float | None = None
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 256
    checkpoint_interval: int = 10
    progress_bar: bool = True
    device: str = 'cpu'
    seed: int = 0
    policy_hidden_dim: int = 128
    policy_attention_heads: int = 2
    policy_self_layers: int = 2
    policy_entity_layers: int = 1
    policy_trunk_layers: int = 2
    policy_activation: str = 'tanh'


class PPOTrainer:
    def __init__(self, runner, action_spec, output_dir, config: PPOConfig, use_swanlab=False, swanlab_project='tank-skill'):
        self.runner = runner
        self.action_spec = action_spec
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.device = torch.device(config.device)
        self.run_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.run_dir = self.output_dir / self.run_stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temp_obs = runner.reset()
        self.obs_keys = list(temp_obs.keys())
        self.self_dim = temp_obs['self_state'].shape[-1]
        self.ally_dim = temp_obs['ally_features'].shape[-1]
        self.enemy_dim = temp_obs['enemy_features'].shape[-1]
        self.max_allies = temp_obs['ally_features'].shape[1]
        self.max_enemies = temp_obs['enemy_features'].shape[1]
        self.policy = MultiTaskPolicy(
            self_dim=self.self_dim,
            ally_dim=self.ally_dim,
            enemy_dim=self.enemy_dim,
            hidden=config.policy_hidden_dim,
            attention_heads=config.policy_attention_heads,
            self_layers=config.policy_self_layers,
            entity_layers=config.policy_entity_layers,
            trunk_layers=config.policy_trunk_layers,
            activation=config.policy_activation,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.logger = SwanLabLogger(enabled=use_swanlab, project=swanlab_project)
        self.best_return = -1e18
        self.best_metrics = None
        self.history = []
        self.obs_meta = {
            'self_state_dim': self.self_dim,
            'ally_entity_dim': self.ally_dim,
            'enemy_entity_dim': self.enemy_dim,
            'max_allies': self.max_allies,
            'max_enemies': self.max_enemies,
            'radar_num_rays': self.action_spec.observation.radar_num_rays,
            'radar_max_distance': self.action_spec.observation.radar_max_distance,
            'policy_config': dict(self.policy.policy_config),
        }

    def _to_torch(self, obs_batch):
        out = {}
        for key, value in obs_batch.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f'Non-finite observation detected in key {key}.')
            out[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return out

    def _collect_episode_batch(self):
        num_envs = self.runner.num_envs
        obs = self.runner.reset()
        active = np.ones((num_envs,), dtype=np.bool_)
        neutral_action = np.ones((num_envs, 3), dtype=np.int64)
        trajectories = [
            {
                'obs': {key: [] for key in self.obs_keys},
                'actions': [],
                'logp': [],
                'values': [],
                'rewards': [],
                'dones': [],
            }
            for _ in range(num_envs)
        ]
        episode_returns = np.zeros((num_envs,), dtype=np.float32)
        episode_lengths = np.zeros((num_envs,), dtype=np.int32)
        episode_success = np.zeros((num_envs,), dtype=np.float32)
        episode_failure = np.zeros((num_envs,), dtype=np.float32)
        episode_truncated = np.zeros((num_envs,), dtype=np.float32)
        raw_reward_weights = getattr(self.action_spec.reward_weights, 'weights', self.action_spec.reward_weights)
        reward_weights = {
            name: float(weight)
            for name, weight in raw_reward_weights.items()
            if abs(float(weight)) > 1e-12
        }
        reward_components = {
            name: np.zeros((num_envs,), dtype=np.float32)
            for name in reward_weights.keys()
        }
        reward_events = {
            name: np.zeros((num_envs,), dtype=np.float32)
            for name in reward_weights.keys()
        }
        while active.any():
            obs_t = self._to_torch(obs)
            with torch.no_grad():
                actions, logp, _, value = self.policy.act(obs_t, self.action_spec.action_mask, deterministic=False)
            acts_np = actions.cpu().numpy().astype(np.int64)
            logp_np = logp.cpu().numpy().astype(np.float32)
            value_np = value.cpu().numpy().astype(np.float32)
            acts_np[~active] = neutral_action[~active]
            next_obs, rewards, dones, infos = self.runner.step(acts_np.tolist())
            for env_idx in np.flatnonzero(active):
                for key in self.obs_keys:
                    trajectories[env_idx]['obs'][key].append(np.array(obs[key][env_idx], copy=True))
                trajectories[env_idx]['actions'].append(np.array(acts_np[env_idx], copy=True))
                trajectories[env_idx]['logp'].append(float(logp_np[env_idx]))
                trajectories[env_idx]['values'].append(float(value_np[env_idx]))
                trajectories[env_idx]['rewards'].append(float(rewards[env_idx]))
                trajectories[env_idx]['dones'].append(float(dones[env_idx]))
                episode_returns[env_idx] += float(rewards[env_idx])
                episode_lengths[env_idx] += 1
                events = infos[env_idx].get('events', {})
                for name, weight in reward_weights.items():
                    value = float(events.get(name, 0.0))
                    reward_events[name][env_idx] += value
                    reward_components[name][env_idx] += value * weight
                if dones[env_idx]:
                    episode_success[env_idx] = float(infos[env_idx].get('success', False))
                    episode_failure[env_idx] = float(infos[env_idx].get('failure', False))
                    episode_truncated[env_idx] = float(infos[env_idx].get('truncated', False))
            obs = next_obs
            active &= ~dones
        return trajectories, {
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths,
            'episode_success': episode_success,
            'episode_failure': episode_failure,
            'episode_truncated': episode_truncated,
            'sampled_steps': int(episode_lengths.sum()),
            'reward_components': reward_components,
            'reward_events': reward_events,
        }

    def _prepare_training_batch(self, trajectories):
        obs_chunks = {key: [] for key in self.obs_keys}
        action_chunks = []
        logp_chunks = []
        adv_chunks = []
        ret_chunks = []
        for traj in trajectories:
            rewards = np.asarray(traj['rewards'], dtype=np.float32)
            values = np.asarray(traj['values'], dtype=np.float32)
            dones = np.asarray(traj['dones'], dtype=np.float32)
            advantages = np.zeros_like(rewards)
            lastgaelam = 0.0
            for t in reversed(range(len(rewards))):
                next_value = 0.0 if t == len(rewards) - 1 else float(values[t + 1])
                next_nonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.cfg.gamma * next_value * next_nonterminal - values[t]
                lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values
            for key in self.obs_keys:
                obs_chunks[key].append(np.asarray(traj['obs'][key], dtype=np.float32))
            action_chunks.append(np.asarray(traj['actions'], dtype=np.int64))
            logp_chunks.append(np.asarray(traj['logp'], dtype=np.float32))
            adv_chunks.append(advantages.astype(np.float32))
            ret_chunks.append(returns.astype(np.float32))
        b_obs = {
            key: torch.as_tensor(np.concatenate(obs_chunks[key], axis=0), dtype=torch.float32, device=self.device)
            for key in self.obs_keys
        }
        b_actions = torch.as_tensor(np.concatenate(action_chunks, axis=0), dtype=torch.long, device=self.device)
        b_logp = torch.as_tensor(np.concatenate(logp_chunks, axis=0), dtype=torch.float32, device=self.device)
        b_adv = torch.as_tensor(np.concatenate(adv_chunks, axis=0), dtype=torch.float32, device=self.device)
        b_ret = torch.as_tensor(np.concatenate(ret_chunks, axis=0), dtype=torch.float32, device=self.device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        return b_obs, b_actions, b_logp, b_adv, b_ret

    def _save_bundle_with_metrics(self, bundle_dir: Path, metrics: dict):
        save_policy_bundle(bundle_dir, self.policy, self.action_spec, self.obs_meta)
        (bundle_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    def _write_train_summary(self):
        summary = {
            'run_stamp': self.run_stamp,
            'best_avg_episode_return': self.best_return,
            'num_updates': len(self.history),
            'history': self.history,
            'best_metrics': self.best_metrics,
        }
        summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
        (self.output_dir / 'train_summary.json').write_text(summary_text, encoding='utf-8')
        (self.run_dir / 'train_summary.json').write_text(summary_text, encoding='utf-8')

    def _current_ent_coef(self, update: int) -> float:
        if self.cfg.ent_coef_final is None:
            return float(self.cfg.ent_coef)
        denom = max(1, int(self.cfg.episodes_per_env) - 1)
        ratio = min(1.0, max(0.0, float(update) / denom))
        return float(self.cfg.ent_coef) + (float(self.cfg.ent_coef_final) - float(self.cfg.ent_coef)) * ratio

    def train(self):
        num_envs = self.runner.num_envs
        global_step = 0
        progress = tqdm(
            total=self.cfg.episodes_per_env,
            desc=f'train:{self.action_spec.action_id}',
            disable=not self.cfg.progress_bar,
        )
        try:
            for update in range(self.cfg.episodes_per_env):
                trajectories, stats = self._collect_episode_batch()
                global_step += stats['sampled_steps']
                b_obs, b_actions, b_logp, b_adv, b_ret = self._prepare_training_batch(trajectories)
                batch_size = b_actions.shape[0]
                idx = np.arange(batch_size)
                loss_records = {'actor': [], 'critic': [], 'entropy': [], 'total': []}
                ent_coef = self._current_ent_coef(update)
                for _ in range(self.cfg.update_epochs):
                    np.random.shuffle(idx)
                    for start in range(0, batch_size, self.cfg.minibatch_size):
                        mb = idx[start:start + self.cfg.minibatch_size]
                        mb_obs = {key: value[mb] for key, value in b_obs.items()}
                        new_logp, entropy, value = self.policy.evaluate_actions(mb_obs, b_actions[mb], self.action_spec.action_mask)
                        ratio = (new_logp - b_logp[mb]).exp()
                        pg_loss = torch.max(
                            -b_adv[mb] * ratio,
                            -b_adv[mb] * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef),
                        ).mean()
                        v_loss = F.mse_loss(value, b_ret[mb])
                        loss = pg_loss + self.cfg.vf_coef * v_loss - ent_coef * entropy.mean()
                        self.opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                        self.opt.step()
                        loss_records['actor'].append(float(pg_loss.detach().cpu().item()))
                        loss_records['critic'].append(float(v_loss.detach().cpu().item()))
                        loss_records['entropy'].append(float(entropy.mean().detach().cpu().item()))
                        loss_records['total'].append(float(loss.detach().cpu().item()))
                metrics = {
                    'update': update,
                    'episodes_per_env_completed': update + 1,
                    'sampled_episodes': (update + 1) * num_envs,
                    'global_step': global_step,
                    'avg_episode_return': float(stats['episode_returns'].mean()),
                    'reward/total': float(stats['episode_returns'].mean()),
                    'avg_episode_length': float(stats['episode_lengths'].mean()),
                    'success_rate': float(stats['episode_success'].mean()),
                    'failure_rate': float(stats['episode_failure'].mean()),
                    'truncated_rate': float(stats['episode_truncated'].mean()),
                    'loss/actor': float(np.mean(loss_records['actor'])) if loss_records['actor'] else 0.0,
                    'loss/critic': float(np.mean(loss_records['critic'])) if loss_records['critic'] else 0.0,
                    'loss/entropy': float(np.mean(loss_records['entropy'])) if loss_records['entropy'] else 0.0,
                    'loss/entropy_coef': float(ent_coef),
                    'loss/total': float(np.mean(loss_records['total'])) if loss_records['total'] else 0.0,
                }
                for name, values in stats['reward_components'].items():
                    metrics[f'reward/{name}'] = float(values.mean())
                for name, values in stats['reward_events'].items():
                    metrics[f'event/{name}'] = float(values.mean())
                self.history.append(metrics)
                if metrics['avg_episode_return'] > self.best_return:
                    self.best_return = metrics['avg_episode_return']
                    self.best_metrics = dict(metrics)
                    self._save_bundle_with_metrics(self.output_dir / 'best', metrics)
                    self._save_bundle_with_metrics(self.run_dir / 'best', metrics)
                self._save_bundle_with_metrics(self.output_dir / 'last', metrics)
                if self.cfg.checkpoint_interval > 0 and (update + 1) % self.cfg.checkpoint_interval == 0:
                    self._save_bundle_with_metrics(self.run_dir / f'episode_{update + 1:04d}', metrics)
                self._write_train_summary()
                swanlab_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if key.startswith('reward/')
                    or key.startswith('loss/')
                    or key in {'success_rate', 'failure_rate', 'truncated_rate'}
                }
                self.logger.log(swanlab_metrics)
                progress.update(1)
                progress.set_postfix(
                    avg_return=f"{metrics['avg_episode_return']:.3f}",
                    avg_len=f"{metrics['avg_episode_length']:.1f}",
                    best=f"{self.best_return:.3f}",
                )
            final_metrics = dict(self.history[-1]) if self.history else {
                'update': -1,
                'episodes_per_env_completed': 0,
                'sampled_episodes': 0,
                'global_step': global_step,
                'avg_episode_return': 0.0,
                'avg_episode_length': 0.0,
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'truncated_rate': 0.0,
            }
            self._save_bundle_with_metrics(self.output_dir / 'final', final_metrics)
            self._save_bundle_with_metrics(self.run_dir / 'final', final_metrics)
            self._write_train_summary()
        finally:
            progress.close()
