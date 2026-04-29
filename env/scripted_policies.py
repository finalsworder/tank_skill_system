from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from env.geometry import angle_diff_deg, angle_to, blocked_by_enemy_units, clamp, distance, has_line_of_sight, line_intersects_rect
from env.rrt import plan_rrt_path

Point = Tuple[float, float]
DIRECT_GOAL_DIST = 28.0
# Keep waypoint switching distance consistent with "stop distance" to avoid
# the (reach-stop) > (reach-switch) deadzone where units stop but never advance.
WAYPOINT_REACHED_DIST = DIRECT_GOAL_DIST
SEARCH_RADIUS = 65.0
STUCK_MOVEMENT_EPS = 1.0
STUCK_REPLAN_STEPS = 6


def _idle(obs: Dict, state: Dict) -> List[float]:
    return [0.0, 0.0, 0.0]


def _target_point(obs: Dict) -> Point:
    target_x = obs['task'].get('target_x')
    target_y = obs['task'].get('target_y')
    if target_x is None or target_y is None:
        return (float(obs['self']['x']), float(obs['self']['y']))
    return (float(target_x), float(target_y))


def _has_explicit_target(obs: Dict) -> bool:
    return obs['task'].get('target_x') is not None and obs['task'].get('target_y') is not None


def _visible_enemies(obs: Dict) -> List[Dict]:
    return [
        enemy
        for enemy in obs['enemies']
        if enemy['alive'] and enemy['visible'] and enemy['x'] is not None and enemy['y'] is not None
    ]


def _closest_visible_enemy(obs: Dict) -> Optional[Dict]:
    visible_enemies = _visible_enemies(obs)
    if not visible_enemies:
        return None
    return min(
        visible_enemies,
        key=lambda enemy: float(enemy['distance']) if enemy['distance'] is not None else float('inf'),
    )


def _enemy_blockers(obs: Dict, target_enemy_id: int) -> List[Point]:
    blockers = []
    for enemy in _visible_enemies(obs):
        if int(enemy['id']) == int(target_enemy_id):
            continue
        blockers.append((float(enemy['x']), float(enemy['y'])))
    return blockers


def _is_enemy_shootable(obs: Dict, enemy: Dict) -> bool:
    if not enemy['alive'] or not enemy['visible'] or enemy['x'] is None or enemy['y'] is None:
        return False
    origin = (float(obs['self']['x']), float(obs['self']['y']))
    target = (float(enemy['x']), float(enemy['y']))
    obstacles = [(rect['x'], rect['y'], rect['w'], rect['h']) for rect in obs['obstacles']]
    if not has_line_of_sight(origin, target, obstacles):
        return False
    blockers = _enemy_blockers(obs, int(enemy['id']))
    if blocked_by_enemy_units(origin, target, blockers, float(obs['constants']['unit_radius'])):
        return False
    return True


def _closest_shootable_enemy(obs: Dict) -> Optional[Dict]:
    shootable = [enemy for enemy in _visible_enemies(obs) if _is_enemy_shootable(obs, enemy)]
    if not shootable:
        return None
    return min(shootable, key=lambda enemy: float(enemy['distance']))


def _aim_turret(obs: Dict, target: Optional[Point]) -> float:
    if target is None:
        return 0.0
    desired = angle_to((obs['self']['x'], obs['self']['y']), target)
    delta = angle_diff_deg(desired, obs['self']['turret_angle'])
    return clamp(delta, -obs['constants']['max_turret_turn'], obs['constants']['max_turret_turn'])


def _drive_to_point(obs: Dict, waypoint: Point) -> Tuple[float, float]:
    goal_distance = distance((obs['self']['x'], obs['self']['y']), waypoint)
    if goal_distance <= DIRECT_GOAL_DIST:
        return 0.0, 0.0

    desired = angle_to((obs['self']['x'], obs['self']['y']), waypoint)
    body_delta = math.radians(angle_diff_deg(desired, float(obs['self']['body_angle'])))
    speed = min(float(obs['constants']['max_forward']), goal_distance * 0.18)
    forward = math.cos(body_delta) * speed
    strafe = math.sin(body_delta) * speed
    return (
        clamp(forward, -float(obs['constants']['max_backward']), float(obs['constants']['max_forward'])),
        clamp(strafe, -float(obs['constants']['max_strafe']), float(obs['constants']['max_strafe'])),
    )


def _waypoint_from_state(state: Dict, index: int) -> Optional[Point]:
    path = state.get('_rrt_path', [])
    if not isinstance(path, list) or index >= len(path):
        return None
    point = path[index]
    return (float(point[0]), float(point[1]))


def _segment_is_free(a: Point, b: Point, obs: Dict, *, clearance: float) -> bool:
    """Obstacle-only feasibility check for a straight segment."""
    map_size = obs['map_size']
    if not (
        clearance <= a[0] <= map_size[0] - clearance
        and clearance <= a[1] <= map_size[1] - clearance
        and clearance <= b[0] <= map_size[0] - clearance
        and clearance <= b[1] <= map_size[1] - clearance
    ):
        return False
    obstacles = [(rect['x'], rect['y'], rect['w'], rect['h']) for rect in obs['obstacles']]
    inflated = [(x - clearance, y - clearance, w + 2.0 * clearance, h + 2.0 * clearance) for x, y, w, h in obstacles]
    return not any(line_intersects_rect(a, b, rect) for rect in inflated)


def _prune_path(points: List[Point], *, min_separation: float = 10.0) -> List[Point]:
    """Drop near-duplicate points to reduce jitter in tight gaps."""
    if not points:
        return []
    out: List[Point] = [points[0]]
    for p in points[1:]:
        if distance(out[-1], p) >= min_separation:
            out.append(p)
    return out


def _select_lookahead_waypoint(obs: Dict, state: Dict, *, max_lookahead: int = 10) -> Optional[Point]:
    """Pick the farthest upcoming waypoint reachable by a straight segment.

    This stabilizes motion through narrow corridors without replanning.
    """
    path = state.get('_rrt_path', [])
    if not isinstance(path, list) or not path:
        return None
    idx = int(state.get('_rrt_path_index', 0))
    idx = max(0, min(idx, len(path) - 1))
    current = (float(obs['self']['x']), float(obs['self']['y']))
    clearance = float(obs['constants']['unit_radius']) + 10.0

    best_idx = idx
    end = min(len(path) - 1, idx + max_lookahead)
    for j in range(idx, end + 1):
        cand = (float(path[j][0]), float(path[j][1]))
        if _segment_is_free(current, cand, obs, clearance=clearance):
            best_idx = j
    state['_rrt_path_index'] = best_idx
    return (float(path[best_idx][0]), float(path[best_idx][1]))


def _update_stuck_state(obs: Dict, state: Dict) -> None:
    prev_pos = state.get('_script_prev_pos')
    prev_speed = float(state.get('_script_prev_speed', 0.0))
    if prev_pos is None or abs(prev_speed) <= 0.5:
        state['_stuck_steps'] = 0
        return
    moved = distance((obs['self']['x'], obs['self']['y']), (float(prev_pos[0]), float(prev_pos[1])))
    if moved <= STUCK_MOVEMENT_EPS:
        state['_stuck_steps'] = int(state.get('_stuck_steps', 0)) + 1
    else:
        state['_stuck_steps'] = 0
    if int(state.get('_stuck_steps', 0)) >= STUCK_REPLAN_STEPS:
        state['_rrt_force_replan'] = True
        state['_stuck_steps'] = 0


def _update_rrt_path(obs: Dict, goal: Point, state: Dict) -> Point:
    goal_key = [round(goal[0], 3), round(goal[1], 3)]
    path_index = int(state.get('_rrt_path_index', 0))
    waypoint = _waypoint_from_state(state, path_index)
    force_replan = bool(state.pop('_rrt_force_replan', False))
    need_replan = (
        force_replan
        or state.get('_rrt_goal') != goal_key
        or waypoint is None
    )
    if need_replan:
        rng = random.Random(
            int(obs['step_count']) * 997
            + int(obs['self']['id']) * 131
            + (0 if obs['team'] == 'red' else 1)
        )
        planned = plan_rrt_path(
            start=(obs['self']['x'], obs['self']['y']),
            goal=goal,
            map_size=obs['map_size'],
            obstacles=[(rect['x'], rect['y'], rect['w'], rect['h']) for rect in obs['obstacles']],
            rng=rng,
            clearance=float(obs['constants']['unit_radius']) + 10.0,
        )
        state['_rrt_goal'] = goal_key
        raw = [(float(x), float(y)) for x, y in planned] if planned else [(float(goal_key[0]), float(goal_key[1]))]
        state['_rrt_path'] = [[float(x), float(y)] for x, y in _prune_path(raw)]
        state['_rrt_path_index'] = 0
        state['_rrt_last_plan_step'] = int(obs['step_count'])

    waypoint = _select_lookahead_waypoint(obs, state)
    while (
        waypoint is not None
        and distance((obs['self']['x'], obs['self']['y']), waypoint) <= WAYPOINT_REACHED_DIST
        and int(state.get('_rrt_path_index', 0)) < len(state.get('_rrt_path', [])) - 1
    ):
        state['_rrt_path_index'] = int(state.get('_rrt_path_index', 0)) + 1
        waypoint = _select_lookahead_waypoint(obs, state)
    return goal if waypoint is None else waypoint


def _search_goal(obs: Dict, anchor: Point) -> Point:
    angle_deg = (int(obs['step_count']) * 9 + int(obs['self']['id']) * 120) % 360
    angle_rad = math.radians(angle_deg)
    return (
        clamp(
            anchor[0] + math.cos(angle_rad) * SEARCH_RADIUS,
            obs['constants']['unit_radius'],
            obs['map_size'][0] - obs['constants']['unit_radius'],
        ),
        clamp(
            anchor[1] + math.sin(angle_rad) * SEARCH_RADIUS,
            obs['constants']['unit_radius'],
            obs['map_size'][1] - obs['constants']['unit_radius'],
        ),
    )


def _aim_closest(obs: Dict, state: Dict) -> List[float]:
    enemy = _closest_visible_enemy(obs)
    turret = _aim_turret(obs, None if enemy is None else (enemy['x'], enemy['y']))
    state['_script_prev_pos'] = [float(obs['self']['x']), float(obs['self']['y'])]
    state['_script_prev_speed'] = 0.0
    return [0.0, 0.0, turret]


def _attack_point(obs: Dict, state: Dict) -> List[float]:
    _update_stuck_state(obs, state)
    target = _target_point(obs)
    visible_enemy = _closest_visible_enemy(obs)
    shootable_enemy = _closest_shootable_enemy(obs)

    if shootable_enemy is not None:
        turret = _aim_turret(obs, (shootable_enemy['x'], shootable_enemy['y']))
        state['_script_prev_pos'] = [float(obs['self']['x']), float(obs['self']['y'])]
        state['_script_prev_speed'] = 0.0
        return [0.0, 0.0, turret]

    movement_goal = target
    at_target = distance((obs['self']['x'], obs['self']['y']), target) <= DIRECT_GOAL_DIST
    if at_target:
        anchor = target if _has_explicit_target(obs) else (
            (float(visible_enemy['x']), float(visible_enemy['y'])) if visible_enemy is not None else target
        )
        movement_goal = _search_goal(obs, anchor)

    waypoint = _update_rrt_path(obs, movement_goal, state)
    forward, strafe = _drive_to_point(obs, waypoint)
    turret_delta = _aim_turret(obs, None if visible_enemy is None else (visible_enemy['x'], visible_enemy['y']))
    turret = clamp(turret_delta, -float(obs['constants']['max_turret_turn']), float(obs['constants']['max_turret_turn']))
    state['_script_prev_pos'] = [float(obs['self']['x']), float(obs['self']['y'])]
    state['_script_prev_speed'] = abs(float(forward))
    return [forward, strafe, turret]


SCRIPTED_POLICY_REGISTRY = {
    'idle': _idle,
    'aim_closest': _aim_closest,
    'attack_point': _attack_point,
}
