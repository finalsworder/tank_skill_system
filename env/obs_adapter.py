from __future__ import annotations

from typing import Dict

import numpy as np

from env.geometry import angle_diff_deg


def _float_or_zero(value) -> float:
    return 0.0 if value is None else float(value)


def _bool_to_float(value) -> float:
    return 1.0 if bool(value) else 0.0


def _world_to_local(delta_x: float, delta_y: float, body_angle_deg: float) -> tuple[float, float]:
    angle = np.deg2rad(float(body_angle_deg))
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    return (
        delta_x * cos_a + delta_y * sin_a,
        -delta_x * sin_a + delta_y * cos_a,
    )


def _relative_position(x: float | None, y: float | None, self_x: float, self_y: float, self_body_angle: float) -> tuple[float, float]:
    if x is None or y is None:
        return (0.0, 0.0)
    return _world_to_local(float(x) - self_x, float(y) - self_y, self_body_angle)


def _relative_angle(angle_deg: float | None, self_body_angle: float) -> float:
    if angle_deg is None:
        return 0.0
    return float(angle_diff_deg(float(angle_deg), self_body_angle))


def assemble_rl_obs(raw_obs: Dict, action_spec) -> Dict[str, np.ndarray]:
    obs_cfg = action_spec.observation
    map_w = float(raw_obs['map_size'][0])
    map_h = float(raw_obs['map_size'][1])
    coord_scale = max(1.0, map_w, map_h)
    constants = raw_obs['constants']
    self_unit = raw_obs['self']
    task = raw_obs['task']
    self_x = float(self_unit['x'])
    self_y = float(self_unit['y'])
    self_body_angle = float(self_unit['body_angle'])

    radar = np.asarray(raw_obs.get('radar', []), dtype=np.float32)
    if radar.shape[0] < obs_cfg.radar_num_rays:
        radar = np.pad(radar, (0, obs_cfg.radar_num_rays - radar.shape[0]))
    elif radar.shape[0] > obs_cfg.radar_num_rays:
        radar = radar[: obs_cfg.radar_num_rays]

    target_rel_x, target_rel_y = _relative_position(
        task.get('target_x'),
        task.get('target_y'),
        self_x,
        self_y,
        self_body_angle,
    )
    base_self_state = np.asarray(
        [
            float(raw_obs.get('step_count', 0)) / max(1.0, float(action_spec.termination.max_steps)),
            float(self_unit['speed']) / max(1.0, float(constants['max_forward'])),
            float(self_unit['cooldown']) / max(1.0, float(constants['fire_cooldown'])),
            _bool_to_float(self_unit.get('under_observation', False)),
            _relative_angle(self_unit.get('turret_angle'), self_body_angle) / 180.0,
            target_rel_x / coord_scale,
            target_rel_y / coord_scale,
        ],
        dtype=np.float32,
    )
    self_state = np.concatenate([base_self_state, radar.astype(np.float32)], axis=0)

    ally_features = np.zeros((obs_cfg.max_allies, 8), dtype=np.float32)
    ally_mask = np.zeros((obs_cfg.max_allies,), dtype=np.float32)
    selected_ally_id = task.get('ally_id')
    for index, ally in enumerate(raw_obs['allies'][: obs_cfg.max_allies]):
        ally_rel_x, ally_rel_y = _relative_position(
            ally.get('x'),
            ally.get('y'),
            self_x,
            self_y,
            self_body_angle,
        )
        ally_features[index] = np.asarray(
            [
                ally_rel_x / coord_scale,
                ally_rel_y / coord_scale,
                _relative_angle(ally.get('body_angle'), self_body_angle) / 180.0,
                _relative_angle(ally.get('turret_angle'), self_body_angle) / 180.0,
                float(ally['hp']) / max(1.0, float(constants['initial_hp'])),
                _bool_to_float(ally.get('under_observation', False)),
                _bool_to_float(selected_ally_id is not None and int(selected_ally_id) == int(ally['id'])),
                _bool_to_float(ally['alive']),
            ],
            dtype=np.float32,
        )
        ally_mask[index] = 1.0

    enemy_features = np.zeros((obs_cfg.max_enemies, 10), dtype=np.float32)
    enemy_mask = np.zeros((obs_cfg.max_enemies,), dtype=np.float32)
    selected_enemy_id = task.get('enemy_id')
    enemies_by_id = {int(enemy['id']): enemy for enemy in raw_obs['enemies']}
    for enemy_id in range(obs_cfg.max_enemies):
        enemy = enemies_by_id.get(enemy_id)
        if enemy is None or not enemy.get('track_available', False):
            continue
        enemy_rel_x, enemy_rel_y = _relative_position(
            enemy.get('track_x'),
            enemy.get('track_y'),
            self_x,
            self_y,
            self_body_angle,
        )
        enemy_features[enemy_id] = np.asarray(
            [
                enemy_rel_x / coord_scale,
                enemy_rel_y / coord_scale,
                _relative_angle(enemy.get('track_body_angle'), self_body_angle) / 180.0,
                _relative_angle(enemy.get('track_turret_angle'), self_body_angle) / 180.0,
                _float_or_zero(enemy.get('track_hp')) / max(1.0, float(constants['initial_hp'])),
                _bool_to_float(enemy.get('visible', False)),
                _bool_to_float(enemy.get('memory_known', False)),
                _float_or_zero(enemy.get('aim_error_deg')) / 180.0,
                _bool_to_float(selected_enemy_id is not None and int(selected_enemy_id) == enemy_id),
                _bool_to_float(enemy.get('alive', False)),
            ],
            dtype=np.float32,
        )
        enemy_mask[enemy_id] = 1.0

    return {
        'self_state': self_state,
        'ally_features': ally_features,
        'ally_mask': ally_mask,
        'enemy_features': enemy_features,
        'enemy_mask': enemy_mask,
    }
