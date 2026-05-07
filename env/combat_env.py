from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.geometry import (
    angle_diff_deg,
    angle_to,
    any_line_of_sight,
    circle_circle_overlap,
    circle_rect_overlap,
    clamp,
    distance,
    oriented_square_vertices,
    raycast_distance,
    raycast_hit,
)
from env.obs_adapter import assemble_rl_obs
from env.reward_functions import compute_weighted_reward
from env.scripted_policies import SCRIPTED_POLICY_REGISTRY
from env.specs import ActionSpec, ObservationConfig, ResetTemplate, UnitPolicySpec
from env.termination import evaluate_termination

MAX_TEAM_SIZE = 3
UNIT_RADIUS = 10.0
INITIAL_HP = 5
MAX_FORWARD = 4.0
MAX_BACKWARD = 2.0
MAX_STRAFE = 4.0
MAX_BODY_TURN = 6.0
MAX_TURRET_TURN = 5.0
FIRE_RANGE = 1_000_000.0
FIRE_CONE_DEG = 6.0
CANNON_AUTO_SNAP_DEG = 5.0
FIRE_COOLDOWN = 12
TEAM_COLORS = {
    'red': (220, 60, 60),
    'blue': (60, 90, 220),
}
CANNON_LINE_COLORS = {
    'red': (180, 35, 35),
    'blue': (35, 60, 180),
}
OBSTACLE_COLOR = (120, 120, 120)


class CombatEnv:
    def __init__(self, seed: int = 0, headless: bool = True, render_scale: float = 0.8):
        self.rng = random.Random(seed)
        self.headless = headless
        self.render_scale = render_scale
        self.map_size = [1000.0, 1000.0]
        self.max_forward = MAX_FORWARD
        self.max_backward = MAX_BACKWARD
        self.max_strafe = MAX_STRAFE
        self.max_body_turn = MAX_BODY_TURN
        self.max_turret_turn = MAX_TURRET_TURN
        self.unit_radius = UNIT_RADIUS
        self.units = {'red': {}, 'blue': {}}
        self.red_memory = {}
        self.red_cont_obs_steps = {}
        self.share_team_memory = True
        self.obstacles = []
        self.step_count = 0
        self.last_step_records = {}
        self.current_template = None
        self.controlled_red_ids = []
        self.red_script_policies = {}
        self.blue_script_policies = {}
        self.render_targets = {'red': {}, 'blue': {}}
        self.turret_locks = {'red': {}, 'blue': {}}
        self._screen = None
        self._pygame = None

    def reset(
        self,
        reset_template: ResetTemplate,
        controlled_red_ids: List[int],
        red_script_policies: Optional[Dict[int, UnitPolicySpec]] = None,
        blue_script_policies: Optional[Dict[int, UnitPolicySpec]] = None,
    ):
        self.current_template = copy.deepcopy(reset_template)
        self.map_size = list(reset_template.map_size)
        self.obstacles = [
            (float(obstacle['x']), float(obstacle['y']), float(obstacle['w']), float(obstacle['h']))
            for obstacle in reset_template.obstacles
        ]
        self.controlled_red_ids = list(controlled_red_ids)
        self.red_script_policies = dict(red_script_policies or reset_template.ally_policies)
        self.blue_script_policies = dict(blue_script_policies or reset_template.enemy_policies)
        self.units = {'red': {}, 'blue': {}}
        self.turret_locks = {'red': {}, 'blue': {}}
        self.step_count = 0
        self.last_step_records = {
            'red_hits': {},
            'blue_hits': {},
            'red_collisions': set(),
            'newly_seen': {},
            'under_observation': {},
            'self_hit': set(),
            'destroyed_enemies': set(),
        }

        occupied = []
        red_count = max(1, min(MAX_TEAM_SIZE, int(reset_template.red_count)))
        blue_count = max(1, min(MAX_TEAM_SIZE, int(reset_template.blue_count)))
        for unit_id in range(red_count):
            position = self._sample_spawn(reset_template.red_spawns[:red_count], occupied)
            occupied.append(tuple(position))
            self.units['red'][unit_id] = self._make_unit(unit_id, position, reset_template.red_heading_range)
        for unit_id in range(blue_count):
            position = self._sample_spawn(reset_template.blue_spawns[:blue_count], occupied)
            occupied.append(tuple(position))
            self.units['blue'][unit_id] = self._make_unit(unit_id, position, reset_template.blue_heading_range)

        self.red_memory = {}
        self.red_cont_obs_steps = {red_id: 0 for red_id in self.units['red'].keys()}
        for red_id in self.units['red'].keys():
            self.red_memory[red_id] = {
                enemy_id: {
                    'known': False,
                    'x': 0.0,
                    'y': 0.0,
                    'body_angle': 0.0,
                    'turret_angle': 0.0,
                    'alive': 0,
                    'hp': 0.0,
                }
                for enemy_id in range(MAX_TEAM_SIZE)
            }
        for enemy_id in reset_template.initial_known_enemy_ids:
            self.reveal_enemy_to_red_memory(int(enemy_id))

        self._update_visibility_and_memory()
        self.refresh_render_targets()

    def _make_unit(self, unit_id: int, position: List[float], heading_range: List[float]):
        angle = self.rng.uniform(float(heading_range[0]), float(heading_range[1]))
        return {
            'id': unit_id,
            'x': position[0],
            'y': position[1],
            'body_angle': angle,
            'turret_angle': angle,
            'speed': 0.0,
            'alive': 1,
            'cooldown': 0,
            'hp': INITIAL_HP,
        }

    def _sample_spawn(self, spawn_regions, occupied):
        for _ in range(500):
            region = self.rng.choice(spawn_regions)
            if region.mode == 'point':
                position = list(region.point)
            else:
                position = [
                    self.rng.uniform(region.rect[0], region.rect[0] + region.rect[2]),
                    self.rng.uniform(region.rect[1], region.rect[1] + region.rect[3]),
                ]
            if any(circle_rect_overlap(tuple(position), UNIT_RADIUS, rect) for rect in self.obstacles):
                continue
            if any(circle_circle_overlap(tuple(position), UNIT_RADIUS, other_pos, UNIT_RADIUS * 2.5) for other_pos in occupied):
                continue
            return position
        raise RuntimeError('Failed to sample legal spawn outside obstacles.')

    def _turret_center(self, unit) -> Tuple[float, float]:
        return (float(unit['x']), float(unit['y']))

    def _tank_body_vertices(self, unit) -> List[Tuple[float, float]]:
        return oriented_square_vertices(
            (float(unit['x']), float(unit['y'])),
            UNIT_RADIUS,
            float(unit['body_angle']),
        )

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        px, py = point
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            crosses = (yi > py) != (yj > py)
            if crosses:
                x_at_y = (xj - xi) * (py - yi) / (yj - yi) + xi
                if px < x_at_y:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _ray_segment_distance(
        origin: Tuple[float, float],
        angle_deg: float,
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> Optional[float]:
        rad = math.radians(angle_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        sx, sy = b[0] - a[0], b[1] - a[1]
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-9:
            return None
        ox, oy = origin
        qx, qy = a[0] - ox, a[1] - oy
        t = (qx * sy - qy * sx) / denom
        u = (qx * dy - qy * dx) / denom
        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def _ray_tank_distance(self, origin: Tuple[float, float], angle_deg: float, unit: Dict[str, Any]) -> Optional[float]:
        vertices = self._tank_body_vertices(unit)
        if self._point_in_polygon(origin, vertices):
            return 0.0
        distances = [
            self._ray_segment_distance(origin, angle_deg, vertices[index], vertices[(index + 1) % len(vertices)])
            for index in range(len(vertices))
        ]
        valid = [value for value in distances if value is not None]
        return min(valid) if valid else None

    def _enemy_memory_record(self, enemy_id: int) -> Dict[str, Any]:
        enemy = self.units['blue'][enemy_id]
        return {
            'known': True,
            'x': float(enemy['x']),
            'y': float(enemy['y']),
            'body_angle': float(enemy['body_angle']),
            'turret_angle': float(enemy['turret_angle']),
            'alive': int(enemy['alive']),
            'hp': float(enemy['hp']),
        }

    def _memory_target_red_ids(self, observer_red_id: Optional[int] = None) -> List[int]:
        if self.share_team_memory or observer_red_id is None:
            return list(self.units['red'].keys())
        return [int(observer_red_id)]

    def reveal_enemy_to_red_memory(self, enemy_id: int, red_ids: Optional[List[int]] = None) -> None:
        if enemy_id not in self.units['blue']:
            return
        memory_record = self._enemy_memory_record(enemy_id)
        target_red_ids = self._memory_target_red_ids() if red_ids is None else [rid for rid in red_ids if rid in self.red_memory]
        for red_id in target_red_ids:
            self.red_memory[red_id][enemy_id] = dict(memory_record)

    def _visible(self, team, unit_id, enemy_team, enemy_id):
        viewer = self.units[team].get(unit_id)
        target = self.units[enemy_team].get(enemy_id)
        if viewer is None or target is None or not viewer['alive'] or not target['alive']:
            return False
        return any_line_of_sight(self._turret_center(viewer), self._tank_body_vertices(target), self.obstacles)

    def _enemy_circles(self, team: str) -> List[Tuple[Tuple[float, float], float, int]]:
        enemy_team = 'blue' if team == 'red' else 'red'
        return [
            ((float(unit['x']), float(unit['y'])), UNIT_RADIUS, int(unit['id']))
            for unit in self.units[enemy_team].values()
            if unit['alive']
        ]

    def _raycast_enemy_hit(self, team: str, origin: Tuple[float, float], angle_deg: float) -> Dict[str, Any]:
        return raycast_hit(
            origin,
            float(angle_deg),
            FIRE_RANGE,
            tuple(self.map_size),
            self.obstacles,
            self._enemy_circles(team),
        )

    def _auto_snap_cannon_angle(self, team: str, unit: Dict[str, Any]) -> float:
        current_angle = float(unit['turret_angle']) % 360.0
        origin = self._turret_center(unit)
        current_hit = self._raycast_enemy_hit(team, origin, current_angle)
        if current_hit['kind'] == 'circle':
            return current_angle

        best_angle = current_angle
        best_delta = CANNON_AUTO_SNAP_DEG + 1e-6
        enemy_team = 'blue' if team == 'red' else 'red'
        for enemy in self.units[enemy_team].values():
            if not enemy['alive']:
                continue
            desired_angle = angle_to(origin, (float(enemy['x']), float(enemy['y'])))
            delta = abs(angle_diff_deg(desired_angle, current_angle))
            if delta > CANNON_AUTO_SNAP_DEG or delta >= best_delta:
                continue
            candidate_hit = self._raycast_enemy_hit(team, origin, desired_angle)
            if candidate_hit['kind'] != 'circle' or int(candidate_hit['payload']) != int(enemy['id']):
                continue
            best_angle = desired_angle
            best_delta = delta
        return best_angle % 360.0

    def _auto_snap_cannon_lines(self, team: str) -> None:
        for unit in self.units[team].values():
            if unit['alive']:
                unit['turret_angle'] = self._auto_snap_cannon_angle(team, unit)

    def _locked_turret_angle(self, team: str, unit: Dict[str, Any], enemy_id: int) -> Optional[float]:
        enemy_team = 'blue' if team == 'red' else 'red'
        target = self.units[enemy_team].get(enemy_id)
        if target is None or not unit['alive'] or not target['alive']:
            return None

        origin = self._turret_center(unit)
        desired_angle = angle_to(origin, (float(target['x']), float(target['y'])))
        hit = self._raycast_enemy_hit(team, origin, desired_angle)
        if hit['kind'] != 'circle' or int(hit['payload']) != int(enemy_id):
            return None
        return desired_angle % 360.0

    def _capture_current_turret_locks(self) -> Dict[str, set[int]]:
        locked_now = {'red': set(), 'blue': set()}
        for team in ['red', 'blue']:
            for unit_id, unit in self.units[team].items():
                if not unit['alive']:
                    self.turret_locks[team].pop(int(unit_id), None)
                    continue
                locked_enemy_id = self.turret_locks[team].get(int(unit_id))
                if locked_enemy_id is not None:
                    locked_angle = self._locked_turret_angle(team, unit, int(locked_enemy_id))
                    if locked_angle is None:
                        self.turret_locks[team].pop(int(unit_id), None)
                    else:
                        unit['turret_angle'] = locked_angle
                        locked_now[team].add(int(unit_id))
                    continue
                hit = self._raycast_enemy_hit(team, self._turret_center(unit), float(unit['turret_angle']))
                if hit['kind'] == 'circle':
                    self.turret_locks[team][int(unit_id)] = int(hit['payload'])
                    locked_now[team].add(int(unit_id))
        return locked_now

    def _apply_turret_locks(self) -> None:
        for team in ['red', 'blue']:
            for unit_id, unit in self.units[team].items():
                enemy_id = self.turret_locks[team].get(int(unit_id))
                if enemy_id is None:
                    continue
                locked_angle = self._locked_turret_angle(team, unit, int(enemy_id))
                if locked_angle is None:
                    self.turret_locks[team].pop(int(unit_id), None)
                    continue
                unit['turret_angle'] = locked_angle

    def _discrete_to_continuous(self, action_raw, action_mask):
        action = [0.0, 0.0, 0.0]
        if 0 in action_mask:
            choice = int(action_raw[0])
            action[0] = -MAX_BACKWARD if choice == 0 else (MAX_FORWARD if choice == 2 else 0.0)
        if 1 in action_mask:
            choice = int(action_raw[1])
            action[1] = -MAX_STRAFE if choice == 0 else (MAX_STRAFE if choice == 2 else 0.0)
        if 2 in action_mask:
            choice = int(action_raw[2])
            action[2] = -MAX_TURRET_TURN if choice == 0 else (MAX_TURRET_TURN if choice == 2 else 0.0)
        return action

    def _apply_motion(self, unit, action, lock_turret: bool = False):
        if not unit['alive']:
            return False
        original_x, original_y = unit['x'], unit['y']
        forward = clamp(float(action[0]), -MAX_BACKWARD, MAX_FORWARD)
        strafe = clamp(float(action[1]), -MAX_STRAFE, MAX_STRAFE)
        turret_turn = clamp(float(action[2]), -MAX_TURRET_TURN, MAX_TURRET_TURN)

        speed_mag = math.hypot(forward, strafe)
        if speed_mag > MAX_FORWARD:
            scale = MAX_FORWARD / max(1e-6, speed_mag)
            forward *= scale
            strafe *= scale

        body_rad = math.radians(float(unit['body_angle']))
        dx = forward * math.cos(body_rad) - strafe * math.sin(body_rad)
        dy = forward * math.sin(body_rad) + strafe * math.cos(body_rad)
        desired_x = original_x + dx
        desired_y = original_y + dy

        def legal_position(x: float, y: float) -> bool:
            if x < UNIT_RADIUS or x > self.map_size[0] - UNIT_RADIUS:
                return False
            if y < UNIT_RADIUS or y > self.map_size[1] - UNIT_RADIUS:
                return False
            return not any(circle_rect_overlap((x, y), UNIT_RADIUS, rect) for rect in self.obstacles)

        collided = False
        if legal_position(desired_x, desired_y):
            unit['x'], unit['y'] = desired_x, desired_y
        elif legal_position(desired_x, original_y):
            unit['x'] = desired_x
            collided = True
        elif legal_position(original_x, desired_y):
            unit['y'] = desired_y
            collided = True
        else:
            collided = abs(dx) > 1e-6 or abs(dy) > 1e-6

        actual_dx = float(unit['x']) - float(original_x)
        actual_dy = float(unit['y']) - float(original_y)
        unit['speed'] = math.hypot(actual_dx, actual_dy)
        if unit['speed'] > 1e-6:
            unit['body_angle'] = math.degrees(math.atan2(actual_dy, actual_dx)) % 360.0
        if not lock_turret:
            unit['turret_angle'] = (float(unit['turret_angle']) + turret_turn) % 360.0
        return collided

    def _script_action(self, team, unit_id):
        policies = self.red_script_policies if team == 'red' else self.blue_script_policies
        policy = policies.get(str(unit_id), UnitPolicySpec('idle', {}))
        obs = self.build_rule_obs(team, unit_id, policy.params)
        return SCRIPTED_POLICY_REGISTRY[policy.name](obs, policy.params)

    def _extract_target_xy(self, values):
        if values is None or not isinstance(values, dict):
            return None
        target_x = values.get('target_x')
        target_y = values.get('target_y')
        if target_x is None or target_y is None:
            return None
        target_x = float(target_x)
        target_y = float(target_y)
        if target_x == 0.0 and target_y == 0.0:
            return None
        return (
            clamp(target_x, 0.0, self.map_size[0]),
            clamp(target_y, 0.0, self.map_size[1]),
        )

    def refresh_render_targets(self, task_values_by_red=None):
        self.render_targets = {'red': {}, 'blue': {}}
        for red_id, task_values in (task_values_by_red or {}).items():
            target = self._extract_target_xy(task_values)
            if target is not None:
                self.render_targets['red'][int(red_id)] = target
        for team, policies in [('red', self.red_script_policies), ('blue', self.blue_script_policies)]:
            for unit_id, policy in policies.items():
                target = self._extract_target_xy(policy.params)
                if target is not None:
                    self.render_targets[team][int(unit_id)] = target

    def _update_visibility_and_memory(self):
        newly_seen = {red_id: set() for red_id in self.units['red'].keys()}
        under_observation = {}
        visible_enemy_ids_by_red = {red_id: set() for red_id in self.units['red'].keys()}
        for red_id in self.units['red'].keys():
            for enemy_id in self.units['blue'].keys():
                if self._visible('red', red_id, 'blue', enemy_id):
                    visible_enemy_ids_by_red[red_id].add(enemy_id)
        for red_id, enemy_ids in visible_enemy_ids_by_red.items():
            for enemy_id in enemy_ids:
                target_red_ids = self._memory_target_red_ids(red_id)
                for target_red_id in target_red_ids:
                    if not self.red_memory[target_red_id][enemy_id]['known']:
                        newly_seen[target_red_id].add(enemy_id)
                self.reveal_enemy_to_red_memory(enemy_id, target_red_ids)
        for red_id in self.units['red'].keys():
            visible_from_blue = any(
                self._visible('blue', blue_id, 'red', red_id)
                for blue_id in self.units['blue'].keys()
            )
            under_observation[red_id] = visible_from_blue
            self.red_cont_obs_steps[red_id] = self.red_cont_obs_steps[red_id] + 1 if visible_from_blue else 0
        self.last_step_records['newly_seen'] = newly_seen
        self.last_step_records['under_observation'] = under_observation

    def _auto_fire_team(self, team):
        enemy_team = 'blue' if team == 'red' else 'red'
        hit_dict = self.last_step_records['red_hits'] if team == 'red' else self.last_step_records['blue_hits']
        for attacker in self.units[team].values():
            if not attacker['alive'] or attacker['cooldown'] > 0:
                continue
            origin = self._turret_center(attacker)
            angle = float(attacker['turret_angle'])

            hit = self._raycast_enemy_hit(team, origin, angle)
            if hit['kind'] != 'circle':
                continue
            target_id = int(hit['payload'])
            target = self.units[enemy_team].get(target_id)
            if target is None or not target['alive']:
                continue

            target['hp'] -= 1
            if target['hp'] <= 0:
                target['hp'] = 0
                target['alive'] = 0
                if team == 'red':
                    self.last_step_records['destroyed_enemies'].add(target['id'])

            hit_dict[attacker['id']] = target['id']
            if team == 'blue':
                self.last_step_records['self_hit'].add(target['id'])
            attacker['cooldown'] = FIRE_COOLDOWN

    def _task_xy(self, task_values):
        return float(task_values.get('target_x') or 0.0), float(task_values.get('target_y') or 0.0)

    def build_rule_obs(
        self,
        team,
        unit_id,
        task_values=None,
        obs_cfg: Optional[ObservationConfig] = None,
    ):
        unit = self.units[team][unit_id]
        enemy_team = 'blue' if team == 'red' else 'red'
        task_values = dict(task_values or {})
        target_xy = self._extract_target_xy(task_values)
        target_x, target_y = self._task_xy(task_values)
        self_under_observation = bool(self.last_step_records['under_observation'].get(unit_id, False)) if team == 'red' else False

        allies = []
        for ally in sorted(self.units[team].values(), key=lambda item: item['id']):
            if ally['id'] == unit_id:
                continue
            allies.append(
                {
                    'id': int(ally['id']),
                    'x': float(ally['x']),
                    'y': float(ally['y']),
                    'body_angle': float(ally['body_angle']),
                    'turret_angle': float(ally['turret_angle']),
                    'hp': float(ally['hp']),
                    'alive': bool(ally['alive']),
                    'under_observation': bool(self.last_step_records['under_observation'].get(ally['id'], False)) if team == 'red' else False,
                }
            )

        enemies = []
        for enemy in sorted(self.units[enemy_team].values(), key=lambda item: item['id']):
            visible = self._visible(team, unit_id, enemy_team, enemy['id'])
            memory_known = False
            tracked_x = None
            tracked_y = None
            tracked_body = None
            tracked_turret = None
            tracked_hp = None
            tracked_alive = False
            if visible:
                tracked_x = float(enemy['x'])
                tracked_y = float(enemy['y'])
                tracked_body = float(enemy['body_angle'])
                tracked_turret = float(enemy['turret_angle'])
                tracked_hp = float(enemy['hp'])
                tracked_alive = bool(enemy['alive'])
            elif team == 'red':
                memory = self.red_memory[unit_id][enemy['id']]
                memory_known = bool(memory['known'])
                if memory_known:
                    tracked_x = float(memory['x'])
                    tracked_y = float(memory['y'])
                    tracked_body = float(memory['body_angle'])
                    tracked_turret = float(memory['turret_angle'])
                    tracked_hp = float(memory['hp'])
                    tracked_alive = bool(memory['alive'])

            if tracked_x is not None and tracked_y is not None:
                aim_error = abs(angle_diff_deg(angle_to(self._turret_center(unit), (tracked_x, tracked_y)), unit['turret_angle']))
                tracked_distance = float(distance((unit['x'], unit['y']), (tracked_x, tracked_y)))
            else:
                aim_error = 0.0
                tracked_distance = None

            enemies.append(
                {
                    'id': int(enemy['id']),
                    'visible': bool(visible),
                    'memory_known': bool(memory_known),
                    'track_available': tracked_x is not None and tracked_y is not None,
                    'alive': bool(tracked_alive),
                    'distance': tracked_distance,
                    'aim_error_deg': float(aim_error),
                    'x': float(enemy['x']) if visible else None,
                    'y': float(enemy['y']) if visible else None,
                    'body_angle': float(enemy['body_angle']) if visible else None,
                    'turret_angle': float(enemy['turret_angle']) if visible else None,
                    'hp': float(enemy['hp']) if visible else None,
                    'track_x': tracked_x,
                    'track_y': tracked_y,
                    'track_body_angle': tracked_body,
                    'track_turret_angle': tracked_turret,
                    'track_hp': tracked_hp,
                }
            )

        return {
            'team': team,
            'step_count': int(self.step_count),
            'map_size': [float(self.map_size[0]), float(self.map_size[1])],
            'obstacles': [
                {'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h)}
                for x, y, w, h in self.obstacles
            ],
            'radar': []
            if obs_cfg is None
            else self._get_radar_distances(team, unit_id, obs_cfg).astype(np.float32).tolist(),
            'task': {
                'target_x': None if target_xy is None else float(target_xy[0]),
                'target_y': None if target_xy is None else float(target_xy[1]),
                'ally_id': task_values.get('ally_id'),
                'enemy_id': task_values.get('enemy_id'),
            },
            'self': {
                'id': int(unit['id']),
                'x': float(unit['x']),
                'y': float(unit['y']),
                'body_angle': float(unit['body_angle']),
                'turret_angle': float(unit['turret_angle']),
                'speed': float(unit['speed']),
                'alive': bool(unit['alive']),
                'cooldown': int(unit['cooldown']),
                'hp': float(unit['hp']),
                'under_observation': self_under_observation,
            },
            'allies': allies,
            'enemies': enemies,
            'constants': {
                'max_forward': MAX_FORWARD,
                'max_backward': MAX_BACKWARD,
                'max_strafe': MAX_STRAFE,
                'max_body_turn': MAX_BODY_TURN,
                'max_turret_turn': MAX_TURRET_TURN,
                'fire_range': FIRE_RANGE,
                'fire_cone_deg': FIRE_CONE_DEG,
                'cannon_auto_snap_deg': CANNON_AUTO_SNAP_DEG,
                'fire_cooldown': FIRE_COOLDOWN,
                'unit_radius': UNIT_RADIUS,
                'initial_hp': INITIAL_HP,
            },
        }

    def _get_radar_distances(self, team, unit_id, obs_cfg):
        unit = self.units[team][unit_id]
        radar_max = float(max(1.0, obs_cfg.radar_max_distance))
        distances = []
        for ray_id in range(obs_cfg.radar_num_rays):
            base_angle = ray_id * (360.0 / obs_cfg.radar_num_rays)
            angle = (unit['body_angle'] + base_angle) if obs_cfg.radar_body_relative else base_angle
            distances.append(
                raycast_distance(
                    (unit['x'], unit['y']),
                    angle,
                    radar_max,
                    self.map_size,
                    self.obstacles,
                    [],
                    boundary_margin=UNIT_RADIUS,
                    obstacle_margin=UNIT_RADIUS,
                )
                / radar_max
            )
        return np.asarray(distances, dtype=np.float32)

    def build_obs(self, red_id: int, action_spec: ActionSpec, task_values: Dict[str, Any]):
        raw_obs = self.build_rule_obs('red', red_id, task_values, action_spec.observation)
        return assemble_rl_obs(raw_obs, action_spec)

    def observation_meta(self, action_spec):
        return {
            'self_state_dim': 7 + int(action_spec.observation.radar_num_rays),
            'ally_entity_dim': 8,
            'enemy_entity_dim': 10,
            'max_allies': action_spec.observation.max_allies,
            'max_enemies': action_spec.observation.max_enemies,
            'radar_num_rays': action_spec.observation.radar_num_rays,
            'radar_max_distance': action_spec.observation.radar_max_distance,
            'obs_keys': ['self_state', 'ally_features', 'ally_mask', 'enemy_features', 'enemy_mask'],
        }

    def step(self, controlled_actions, action_spec, task_values_by_red):
        self.last_step_records = {
            'red_hits': {},
            'blue_hits': {},
            'red_collisions': set(),
            'newly_seen': {},
            'under_observation': {},
            'self_hit': set(),
            'destroyed_enemies': set(),
        }
        self.refresh_render_targets(task_values_by_red)

        red_actions = {}
        blue_actions = {}
        for red_id in self.units['red'].keys():
            if red_id in controlled_actions:
                red_actions[red_id] = self._discrete_to_continuous(controlled_actions[red_id], action_spec.action_mask)
            else:
                red_actions[red_id] = self._script_action('red', red_id)
        for blue_id in self.units['blue'].keys():
            blue_actions[blue_id] = self._script_action('blue', blue_id)

        locked_turrets = self._capture_current_turret_locks()
        for red_id, action in red_actions.items():
            if self._apply_motion(self.units['red'][red_id], action, red_id in locked_turrets['red']):
                self.last_step_records['red_collisions'].add(red_id)
        for blue_id, action in blue_actions.items():
            self._apply_motion(self.units['blue'][blue_id], action, blue_id in locked_turrets['blue'])

        for team in ['red', 'blue']:
            for unit in self.units[team].values():
                unit['cooldown'] = max(0, unit['cooldown'] - 1)

        self._update_visibility_and_memory()
        self._apply_turret_locks()
        self._auto_snap_cannon_lines('red')
        self._auto_snap_cannon_lines('blue')
        self._capture_current_turret_locks()
        self._auto_fire_team('red')
        self._auto_fire_team('blue')
        self._update_visibility_and_memory()
        self.step_count += 1

        outputs = {}
        for red_id, task_values in task_values_by_red.items():
            events = self._build_events(red_id, task_values, action_spec.termination)
            reward = compute_weighted_reward(events, action_spec.reward_weights.weights)
            done, success, info = evaluate_termination(events, action_spec.termination, self.step_count)
            info['events'] = events
            raw_obs = self.build_rule_obs('red', red_id, task_values, action_spec.observation)
            outputs[red_id] = (raw_obs, reward, done, info)
        return outputs

    def _build_events(self, red_id, task_values, termination_spec):
        red = self.units['red'][red_id]
        target_x, target_y = self._task_xy(task_values)
        has_target = target_x != 0.0 or target_y != 0.0
        current_distance = distance((red['x'], red['y']), (target_x, target_y)) if has_target else 0.0
        previous_distance = task_values.get('_prev_dist', current_distance)
        task_values['_prev_dist'] = current_distance
        progress = (previous_distance - current_distance) / 25.0 if has_target else 0.0
        in_target_radius = (target_x != 0.0 or target_y != 0.0) and current_distance <= float(termination_spec.reach_radius)
        task_values['_target_hold_steps'] = int(task_values.get('_target_hold_steps', 0)) + 1 if in_target_radius else 0
        reach_target = in_target_radius and task_values['_target_hold_steps'] >= max(1, int(termination_spec.hold_steps))

        map_diagonal = max(1.0, math.hypot(self.map_size[0], self.map_size[1]))
        distance_to_target = 0.0 if not has_target else min(1.0, current_distance / map_diagonal)
        target_proximity = 0.0 if not has_target else max(0.0, 1.0 - distance_to_target)

        enemy_id = task_values.get('enemy_id')
        selected_enemy_visible = False
        selected_enemy_memory_valid = False
        selected_enemy_destroyed_state = False
        aim_align_selected_enemy = 0.0
        progress_to_selected_enemy = 0.0
        distance_to_selected_enemy = 0.0
        selected_enemy_proximity = 0.0
        if enemy_id is not None and int(enemy_id) in self.units['blue']:
            enemy_id = int(enemy_id)
            selected_enemy_visible = self._visible('red', red_id, 'blue', enemy_id)
            selected_enemy_memory_valid = self.red_memory[red_id][enemy_id]['known']
            enemy = self.units['blue'][enemy_id]
            current_enemy_distance = distance((red['x'], red['y']), (enemy['x'], enemy['y']))
            previous_enemy_distance = float(task_values.get('_prev_selected_enemy_dist', current_enemy_distance))
            task_values['_prev_selected_enemy_dist'] = current_enemy_distance
            desired = angle_to((red['x'], red['y']), (enemy['x'], enemy['y']))
            aim_align_selected_enemy = max(
                0.0,
                math.cos(math.radians(abs(angle_diff_deg(desired, red['turret_angle'])))),
            )
            selected_enemy_destroyed_state = not bool(enemy['alive'])
            if enemy['alive']:
                progress_to_selected_enemy = (previous_enemy_distance - current_enemy_distance) / 25.0
                distance_to_selected_enemy = min(1.0, current_enemy_distance / map_diagonal)
                selected_enemy_proximity = max(0.0, 1.0 - current_enemy_distance / map_diagonal)
        else:
            task_values['_prev_selected_enemy_dist'] = 0.0

        ally_id = task_values.get('ally_id')
        selected_ally_hit = ally_id is not None and int(ally_id) in self.last_step_records['blue_hits'].values()
        selected_ally_alive = (
            1.0
            if ally_id is None or int(ally_id) not in self.units['red']
            else float(self.units['red'][int(ally_id)]['alive'])
        )
        progress_to_selected_ally = 0.0
        selected_ally_proximity = 0.0
        selected_ally_under_observation = 0.0
        cover_ally_hold = False
        if ally_id is not None and int(ally_id) in self.units['red'] and int(ally_id) != int(red_id):
            ally = self.units['red'][int(ally_id)]
            ally_distance = distance((red['x'], red['y']), (ally['x'], ally['y']))
            prev_ally_distance = float(task_values.get('_prev_selected_ally_dist', ally_distance))
            task_values['_prev_selected_ally_dist'] = ally_distance
            progress_to_selected_ally = (prev_ally_distance - ally_distance) / 25.0
            selected_ally_proximity = max(0.0, 1.0 - ally_distance / map_diagonal)
            selected_ally_under_observation = float(self.last_step_records['under_observation'].get(int(ally_id), False))
            in_cover_radius = bool(ally['alive']) and ally_distance <= float(termination_spec.reach_radius)
            task_values['_ally_cover_hold_steps'] = int(task_values.get('_ally_cover_hold_steps', 0)) + 1 if in_cover_radius else 0
            cover_ally_hold = task_values['_ally_cover_hold_steps'] >= max(1, int(termination_spec.hold_steps))
        else:
            task_values['_prev_selected_ally_dist'] = 0.0
            task_values['_ally_cover_hold_steps'] = 0

        hit_target = 0.0
        destroy_target = 0.0
        if red_id in self.last_step_records['red_hits'] and enemy_id is not None:
            hit_target = float(self.last_step_records['red_hits'][red_id] == int(enemy_id))
            destroy_target = float(int(enemy_id) in self.last_step_records['destroyed_enemies'])

        forward_speed = max(0.0, float(red['speed']) / MAX_FORWARD)
        if target_x == 0.0 and target_y == 0.0:
            heading_align = 0.0
        else:
            heading_align = max(
                0.0,
                math.cos(math.radians(abs(angle_diff_deg(angle_to((red['x'], red['y']), (target_x, target_y)), red['body_angle'])))),
            ) * forward_speed
        under_observation = bool(self.last_step_records['under_observation'].get(red_id, False))
        task_values['_hidden_hold_steps'] = int(task_values.get('_hidden_hold_steps', 0)) + 1 if not under_observation else 0
        hidden_for_hold = task_values['_hidden_hold_steps'] >= max(1, int(termination_spec.hold_steps))

        return {
            'progress_to_target_point': progress,
            'distance_to_target': -float(distance_to_target),
            'target_proximity': float(target_proximity),
            'arrive_target_point': float(in_target_radius),
            'heading_align_to_target_point': -float(heading_align),
            'forward_speed': float(forward_speed),
            'progress_to_selected_enemy': progress_to_selected_enemy,
            'distance_to_selected_enemy': -float(distance_to_selected_enemy),
            'selected_enemy_proximity': -float(selected_enemy_proximity),
            'aim_align_selected_enemy': -float(aim_align_selected_enemy),
            'selected_enemy_visible': -float(selected_enemy_visible),
            'selected_enemy_memory_valid': float(selected_enemy_memory_valid),
            'hit_selected_enemy': hit_target,
            'destroy_selected_enemy': destroy_target,
            'hit_any_enemy': float(red_id in self.last_step_records['red_hits']),
            'ally_selected_hit': float(selected_ally_hit),
            'ally_selected_alive': float(selected_ally_alive),
            'progress_to_selected_ally': progress_to_selected_ally,
            'selected_ally_proximity': selected_ally_proximity,
            'selected_ally_under_observation': -float(selected_ally_under_observation),
            'cover_ally_hold': bool(cover_ally_hold),
            'self_hit': -float(red_id in self.last_step_records['self_hit']),
            'self_destroyed': -float(not red['alive']),
            'discover_any_enemy': float(len(self.last_step_records['newly_seen'].get(red_id, set())) > 0),
            'discover_selected_enemy': float(enemy_id is not None and int(enemy_id) in self.last_step_records['newly_seen'].get(red_id, set())),
            'under_observation': -float(under_observation),
            'not_under_observation': float(not under_observation),
            'hidden_for_hold': bool(hidden_for_hold),
            'continuous_under_observation': -float(self.red_cont_obs_steps.get(red_id, 0)),
            'collision_penalty': -float(red_id in self.last_step_records['red_collisions']),
            'time_penalty': -1.0,
            'reach_target_point': bool(reach_target),
            'selected_enemy_destroyed': bool(selected_enemy_destroyed_state),
        }

    def _cannon_line_endpoint(self, team: str, unit: Dict[str, Any]) -> Tuple[float, float]:
        angle_rad = math.radians(float(unit['turret_angle']))
        origin = (
            float(unit['x']) + math.cos(angle_rad) * UNIT_RADIUS,
            float(unit['y']) + math.sin(angle_rad) * UNIT_RADIUS,
        )
        hit = raycast_hit(
            origin,
            float(unit['turret_angle']),
            max(self.map_size) * 2.0,
            tuple(self.map_size),
            self.obstacles,
            [],
        )
        best_distance = float(hit['distance'])
        for other_team, units in self.units.items():
            for other in units.values():
                if not other['alive']:
                    continue
                if other_team == team and int(other['id']) == int(unit['id']):
                    continue
                tank_distance = self._ray_tank_distance(origin, float(unit['turret_angle']), other)
                if tank_distance is not None and tank_distance < best_distance:
                    best_distance = tank_distance
        return (
            origin[0] + math.cos(angle_rad) * best_distance,
            origin[1] + math.sin(angle_rad) * best_distance,
        )

    def render(self):
        if self.headless:
            return
        if self._pygame is None:
            import pygame

            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode(
                (int(self.map_size[0] * self.render_scale), int(self.map_size[1] * self.render_scale))
            )
        pg = self._pygame
        screen = self._screen
        screen.fill((245, 245, 245))
        for x, y, w, h in self.obstacles:
            pg.draw.rect(
                screen,
                OBSTACLE_COLOR,
                pg.Rect(
                    int(x * self.render_scale),
                    int(y * self.render_scale),
                    int(w * self.render_scale),
                    int(h * self.render_scale),
                ),
            )
        for team, color in TEAM_COLORS.items():
            for unit_id, target in self.render_targets[team].items():
                if unit_id not in self.units[team] or not self.units[team][unit_id]['alive']:
                    continue
                pg.draw.circle(
                    screen,
                    color,
                    (int(target[0] * self.render_scale), int(target[1] * self.render_scale)),
                    int(UNIT_RADIUS * self.render_scale),
                )
        for team, color in CANNON_LINE_COLORS.items():
            for unit in self.units[team].values():
                if not unit['alive']:
                    continue
                angle_rad = math.radians(unit['turret_angle'])
                barrel_tip = (
                    unit['x'] + math.cos(angle_rad) * UNIT_RADIUS,
                    unit['y'] + math.sin(angle_rad) * UNIT_RADIUS,
                )
                endpoint = self._cannon_line_endpoint(team, unit)
                pg.draw.line(
                    screen,
                    color,
                    (int(barrel_tip[0] * self.render_scale), int(barrel_tip[1] * self.render_scale)),
                    (int(endpoint[0] * self.render_scale), int(endpoint[1] * self.render_scale)),
                    1,
                )
        for team, color in TEAM_COLORS.items():
            for unit in self.units[team].values():
                if not unit['alive']:
                    continue
                center_x = int(unit['x'] * self.render_scale)
                center_y = int(unit['y'] * self.render_scale)
                body_radius = max(1, int(UNIT_RADIUS * self.render_scale))
                pg.draw.circle(screen, color, (center_x, center_y), body_radius)
                pg.draw.circle(screen, (0, 0, 0), (center_x, center_y), body_radius, max(1, int(2 * self.render_scale)))
                barrel_len = UNIT_RADIUS * 2.0
                end_x = center_x + int(math.cos(math.radians(unit['turret_angle'])) * barrel_len * self.render_scale)
                end_y = center_y + int(math.sin(math.radians(unit['turret_angle'])) * barrel_len * self.render_scale)
                pg.draw.line(screen, (0, 0, 0), (center_x, center_y), (end_x, end_y), 2)
                hp_width = int(24 * self.render_scale)
                hp_fill = int(hp_width * unit['hp'] / INITIAL_HP)
                pg.draw.rect(screen, (0, 0, 0), pg.Rect(center_x - hp_width // 2, center_y - 18, hp_width, 4), 1)
                pg.draw.rect(screen, (40, 200, 40), pg.Rect(center_x - hp_width // 2, center_y - 18, hp_fill, 4))
        pg.event.pump()
        pg.display.flip()


class SingleSkillEnv:
    def __init__(self, action_spec: ActionSpec, seed: int = 0, headless: bool = True, controlled_red_id: int = 0):
        self.action_spec = action_spec
        self.seed = seed
        self.headless = headless
        self.controlled_red_id = controlled_red_id
        self.base_env = CombatEnv(seed=seed, headless=headless)
        self.rng = random.Random(seed)
        self.current_task_values = {}
        self.done = False
        self.last_obs = None
        self.last_info = {}

    def _sample_value(self, spec, max_valid=None):
        if spec.mode == 'none':
            return None
        if spec.mode == 'fixed':
            return spec.fixed
        if spec.mode == 'range':
            return self.rng.uniform(float(spec.low), float(spec.high))
        if spec.mode == 'choice':
            choices = list(spec.choices)
            if max_valid is not None:
                choices = [choice for choice in choices if int(choice) < max_valid]
            return self.rng.choice(choices)
        raise RuntimeError('Unsupported value spec mode.')

    def _sample_task_target(self):
        for _ in range(200):
            target_x = self._sample_value(self.action_spec.task_slots.target_x)
            target_y = self._sample_value(self.action_spec.task_slots.target_y)
            if target_x is None or target_y is None:
                return (target_x or 0.0, target_y or 0.0)
            point = (float(target_x), float(target_y))
            if not any(circle_rect_overlap(point, UNIT_RADIUS, rect) for rect in self.base_env.obstacles):
                return point
        raise RuntimeError('Failed to sample a target point outside obstacles.')

    def _assemble_obs(self, raw_obs):
        return assemble_rl_obs(raw_obs, self.action_spec)

    def reset(self):
        template = copy.deepcopy(self.rng.choice(self.action_spec.reset_templates))
        self.base_env.share_team_memory = bool(self.action_spec.intel.share_team_memory)
        self.base_env.reset(template, controlled_red_ids=[self.controlled_red_id])
        ally_id = self._sample_value(self.action_spec.task_slots.ally_id, max_valid=template.red_count)
        enemy_id = self._sample_value(self.action_spec.task_slots.enemy_id, max_valid=template.blue_count)
        ally_id = None if ally_id == self.controlled_red_id else ally_id
        target_x, target_y = self._sample_task_target()
        self.current_task_values = {
            'target_x': target_x,
            'target_y': target_y,
            'ally_id': ally_id,
            'enemy_id': enemy_id,
            '_target_hold_steps': 0,
        }
        self.done = False
        self.last_info = {}
        if self.action_spec.intel.reveal_selected_enemy_on_reset and enemy_id is not None:
            reveal_targets = None if self.base_env.share_team_memory else [self.controlled_red_id]
            self.base_env.reveal_enemy_to_red_memory(int(enemy_id), reveal_targets)
        self.base_env.refresh_render_targets({self.controlled_red_id: self.current_task_values})
        raw_obs = self.base_env.build_rule_obs('red', self.controlled_red_id, self.current_task_values, self.action_spec.observation)
        self.last_obs = self._assemble_obs(raw_obs)
        return self.last_obs

    def step(self, action):
        if self.done:
            info = dict(self.last_info)
            info['already_done'] = True
            return self.last_obs, 0.0, True, info
        outputs = self.base_env.step(
            {self.controlled_red_id: action},
            self.action_spec,
            {self.controlled_red_id: self.current_task_values},
        )
        raw_obs, reward, done, info = outputs[self.controlled_red_id]
        self.last_obs = self._assemble_obs(raw_obs)
        self.done = bool(done)
        self.last_info = dict(info)
        return self.last_obs, reward, done, info
