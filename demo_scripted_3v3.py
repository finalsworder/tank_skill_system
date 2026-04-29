from __future__ import annotations

import argparse
import copy
import time

from env.combat_env import CombatEnv
from env.specs import UnitPolicySpec, load_action_spec, load_scene_config


def spawn_center(spawn_region):
    if spawn_region.mode == 'point':
        return [float(spawn_region.point[0]), float(spawn_region.point[1])]
    rect = spawn_region.rect
    return [float(rect[0]) + float(rect[2]) / 2.0, float(rect[1]) + float(rect[3]) / 2.0]


def build_demo_policies(template):
    blue_targets = [spawn_center(region) for region in template.blue_spawns[:template.blue_count]]
    red_policies = {
        str(unit_id): UnitPolicySpec(
            'attack_point',
            {
                'target_x': float(blue_targets[min(unit_id, len(blue_targets) - 1)][0]),
                'target_y': float(blue_targets[min(unit_id, len(blue_targets) - 1)][1]),
            },
        )
        for unit_id in range(template.red_count)
    }
    blue_policies = {
        str(unit_id): UnitPolicySpec('aim_closest', {})
        for unit_id in range(template.blue_count)
    }
    return red_policies, blue_policies


def main():
    parser = argparse.ArgumentParser(description='Visualize a fully scripted 3v3 battle.')
    parser.add_argument('--scene-config', default='configs/scenes/attack_scene.json')
    parser.add_argument('--task-config', default='configs/tasks/attack_task.json')
    parser.add_argument('--template-id', default='attack_3v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--sleep', type=float, default=0.03)
    args = parser.parse_args()

    scene = load_scene_config(args.scene_config)
    action_spec = load_action_spec(args.task_config)
    template = next((copy.deepcopy(t) for t in scene.reset_templates if t.template_id == args.template_id), None)
    if template is None:
        raise ValueError(f'Template {args.template_id!r} not found in {args.scene_config}.')

    env = CombatEnv(seed=args.seed, headless=False)
    red_policies, blue_policies = build_demo_policies(template)
    env.reset(
        template,
        controlled_red_ids=[],
        red_script_policies=red_policies,
        blue_script_policies=blue_policies,
    )
    env.refresh_render_targets()

    for step in range(args.steps):
        env.step({}, action_spec, {})
        env.render()
        red_alive = sum(unit['alive'] for unit in env.units['red'].values())
        blue_alive = sum(unit['alive'] for unit in env.units['blue'].values())
        if red_alive == 0 or blue_alive == 0:
            print({'step': step + 1, 'red_alive': red_alive, 'blue_alive': blue_alive})
            break
        if args.sleep > 0:
            time.sleep(args.sleep)
    print(
        {
            'red_hp': {uid: unit['hp'] for uid, unit in env.units['red'].items()},
            'blue_hp': {uid: unit['hp'] for uid, unit in env.units['blue'].items()},
        }
    )


if __name__ == '__main__':
    main()
