from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from env.combat_env import SingleSkillEnv
from env.specs import ActionSpec, load_json
from parallel_runner.env_runner import ParallelEnvRunner
from ppo.action_registry import ActionRegistry
from ppo.ppo_trainer import PPOConfig, PPOTrainer


@dataclass
class TrainRuntimeConfig:
    output_dir: str | None = None
    registry_dir: str = 'actions/trained'
    num_envs: int = 8
    episodes_per_env: int = 32
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    ent_coef_final: float | None = None
    device: str = 'cpu'
    seed: int = 0
    controlled_red_id: int = 0
    checkpoint_interval: int = 10
    max_time: int | None = None
    progress_bar: bool = True
    use_swanlab: bool = False
    swanlab_project: str = 'tank-skill'
    policy_hidden_dim: int = 128
    policy_attention_heads: int = 2
    policy_self_layers: int = 2
    policy_entity_layers: int = 1
    policy_trunk_layers: int = 2
    policy_activation: str = 'tanh'

    def to_dict(self):
        return asdict(self)


def load_training_inputs(task_config_path: str | Path, scene_config_path: str | Path):
    task_config_path = Path(task_config_path)
    if 'actions' in task_config_path.parts and 'specs' in task_config_path.parts:
        raise ValueError('Training task configs must be loaded from configs/tasks, not actions/specs.')
    action_spec_data = load_json(task_config_path)
    scene_data = load_json(scene_config_path)
    from env.specs import SceneConfig

    scene_cfg = SceneConfig.from_dict(scene_data)
    action_spec = ActionSpec.from_dict(action_spec_data)
    action_spec.reset_templates = [copy.deepcopy(template) for template in scene_cfg.reset_templates]
    if not action_spec.reset_templates:
        raise ValueError('Scene config must provide at least one reset template.')
    return action_spec, action_spec_data, scene_data


def train_from_config_paths(task_config_path: str | Path, scene_config_path: str | Path, runtime_cfg: TrainRuntimeConfig):
    action_spec, action_spec_data, scene_data = load_training_inputs(task_config_path, scene_config_path)
    if runtime_cfg.max_time is not None:
        action_spec.termination.max_steps = int(runtime_cfg.max_time)
    registry = ActionRegistry(runtime_cfg.registry_dir)
    output_dir = Path(runtime_cfg.output_dir) if runtime_cfg.output_dir else registry.action_dir(action_spec.action_id)
    runtime_cfg.output_dir = str(output_dir)
    registry.register_training(action_spec, task_config_path, scene_config_path, output_dir=output_dir)
    env_kwargs_list = [
        {
            'action_spec': action_spec,
            'seed': runtime_cfg.seed + idx,
            'headless': True,
            'controlled_red_id': runtime_cfg.controlled_red_id,
        }
        for idx in range(runtime_cfg.num_envs)
    ]
    runner = ParallelEnvRunner(SingleSkillEnv, env_kwargs_list)
    try:
        ppo_cfg = PPOConfig(
            episodes_per_env=runtime_cfg.episodes_per_env,
            update_epochs=runtime_cfg.update_epochs,
            minibatch_size=runtime_cfg.minibatch_size,
            learning_rate=runtime_cfg.learning_rate,
            ent_coef=runtime_cfg.ent_coef,
            ent_coef_final=runtime_cfg.ent_coef_final,
            checkpoint_interval=runtime_cfg.checkpoint_interval,
            progress_bar=runtime_cfg.progress_bar,
            device=runtime_cfg.device,
            seed=runtime_cfg.seed,
            policy_hidden_dim=runtime_cfg.policy_hidden_dim,
            policy_attention_heads=runtime_cfg.policy_attention_heads,
            policy_self_layers=runtime_cfg.policy_self_layers,
            policy_entity_layers=runtime_cfg.policy_entity_layers,
            policy_trunk_layers=runtime_cfg.policy_trunk_layers,
            policy_activation=runtime_cfg.policy_activation,
        )
        trainer = PPOTrainer(
            runner,
            action_spec,
            output_dir,
            ppo_cfg,
            use_swanlab=runtime_cfg.use_swanlab,
            swanlab_project=runtime_cfg.swanlab_project,
        )
        registry.register_training(
            action_spec,
            task_config_path,
            scene_config_path,
            output_dir=output_dir,
            run_dir=trainer.run_dir,
        )
        (trainer.run_dir / 'action_spec.json').write_text(json.dumps(action_spec_data, ensure_ascii=False, indent=2), encoding='utf-8')
        (trainer.run_dir / 'scene_config.json').write_text(json.dumps(scene_data, ensure_ascii=False, indent=2), encoding='utf-8')
        (trainer.run_dir / 'resolved_action_spec.json').write_text(json.dumps(action_spec.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        (trainer.run_dir / 'resolved_training.json').write_text(json.dumps(runtime_cfg.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        trainer.train()
        registry.mark_ready(action_spec.action_id, output_dir=output_dir, run_dir=trainer.run_dir)
        return trainer.run_dir
    finally:
        runner.close()


def main():
    parser = argparse.ArgumentParser(description='Train a tank skill from task and scene JSON configs.')
    parser.add_argument('--task-config', required=True, help='Path to the task JSON under configs/tasks.')
    parser.add_argument('--scene-config', required=True, help='Path to the scene JSON.')
    parser.add_argument('--output-dir', default=None, help='Directory to save trained outputs. Defaults to actions/trained/<action_id>.')
    parser.add_argument('--registry-dir', default='actions/trained', help='Skill registry root.')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--episodes-per-env', type=int, default=32)
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--minibatch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy bonus coefficient.')
    parser.add_argument('--ent-coef-final', type=float, default=None, help='Optional final entropy coefficient for linear annealing.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--controlled-red-id', type=int, default=0)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--max-time', type=int, default=None, help='Override for termination.max_steps during training.')
    parser.add_argument('--no-progress-bar', action='store_true')
    parser.add_argument('--use-swanlab', action='store_true')
    parser.add_argument('--swanlab-project', default='tank-skill')
    parser.add_argument('--policy-hidden-dim', '--hidden-dim', type=int, default=128, help='Policy hidden width.')
    parser.add_argument('--policy-attention-heads', '--attention-heads', type=int, default=2, help='Entity attention heads.')
    parser.add_argument('--policy-self-layers', '--self-layers', type=int, default=2, help='MLP layers for self state.')
    parser.add_argument('--policy-entity-layers', '--entity-layers', type=int, default=1, help='MLP layers for ally/enemy entities.')
    parser.add_argument('--policy-trunk-layers', '--trunk-layers', type=int, default=2, help='MLP layers after entity pooling.')
    parser.add_argument('--policy-activation', '--activation', default='tanh', choices=['tanh', 'relu', 'gelu', 'silu'])
    args = parser.parse_args()
    runtime_cfg = TrainRuntimeConfig(
        output_dir=args.output_dir,
        registry_dir=args.registry_dir,
        num_envs=args.num_envs,
        episodes_per_env=args.episodes_per_env,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        ent_coef_final=args.ent_coef_final,
        device=args.device,
        seed=args.seed,
        controlled_red_id=args.controlled_red_id,
        checkpoint_interval=args.checkpoint_interval,
        max_time=args.max_time,
        progress_bar=not args.no_progress_bar,
        use_swanlab=args.use_swanlab,
        swanlab_project=args.swanlab_project,
        policy_hidden_dim=args.policy_hidden_dim,
        policy_attention_heads=args.policy_attention_heads,
        policy_self_layers=args.policy_self_layers,
        policy_entity_layers=args.policy_entity_layers,
        policy_trunk_layers=args.policy_trunk_layers,
        policy_activation=args.policy_activation,
    )
    run_dir = train_from_config_paths(args.task_config, args.scene_config, runtime_cfg)
    print(f'training_complete: {run_dir}')


if __name__ == '__main__':
    main()
