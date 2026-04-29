from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

from env.geometry import clamp, distance, line_intersects_rect

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]


def _expand_rect(rect: Rect, margin: float) -> Rect:
    x, y, w, h = rect
    return (x - margin, y - margin, w + margin * 2.0, h + margin * 2.0)


def _point_in_bounds(point: Point, map_size: Sequence[float], clearance: float) -> bool:
    return clearance <= point[0] <= map_size[0] - clearance and clearance <= point[1] <= map_size[1] - clearance


def _segment_is_free(a: Point, b: Point, map_size: Sequence[float], obstacles: Iterable[Rect], clearance: float) -> bool:
    if not _point_in_bounds(a, map_size, clearance) or not _point_in_bounds(b, map_size, clearance):
        return False
    inflated = [_expand_rect(rect, clearance) for rect in obstacles]
    return not any(line_intersects_rect(a, b, rect) for rect in inflated)


def _steer(start: Point, goal: Point, step_size: float) -> Point:
    total = distance(start, goal)
    if total <= step_size:
        return goal
    ratio = step_size / max(total, 1e-6)
    return (
        start[0] + (goal[0] - start[0]) * ratio,
        start[1] + (goal[1] - start[1]) * ratio,
    )


def _reconstruct_path(nodes: List[Point], parents: List[int], node_index: int) -> List[Point]:
    path: List[Point] = []
    current = node_index
    while current >= 0:
        path.append(nodes[current])
        current = parents[current]
    path.reverse()
    return path


def _shortcut_path(path: List[Point], rng: random.Random, map_size: Sequence[float], obstacles: Iterable[Rect], clearance: float, passes: int = 32) -> List[Point]:
    if len(path) <= 2:
        return path
    result = list(path)
    for _ in range(passes):
        if len(result) <= 2:
            break
        i = rng.randrange(0, len(result) - 2)
        j = rng.randrange(i + 2, len(result))
        if _segment_is_free(result[i], result[j], map_size, obstacles, clearance):
            result = result[: i + 1] + result[j:]
    return result


def plan_rrt_path(
    start: Point,
    goal: Point,
    map_size: Sequence[float],
    obstacles: Iterable[Rect],
    rng: random.Random,
    step_size: float = 70.0,
    max_iters: int = 450,
    goal_sample_rate: float = 0.22,
    clearance: float = 18.0,
    goal_tolerance: float = 45.0,
) -> List[Point]:
    obstacles = [tuple(rect) for rect in obstacles]
    start = (float(start[0]), float(start[1]))
    goal = (
        clamp(float(goal[0]), clearance, map_size[0] - clearance),
        clamp(float(goal[1]), clearance, map_size[1] - clearance),
    )
    if _segment_is_free(start, goal, map_size, obstacles, clearance):
        return [goal]

    nodes: List[Point] = [start]
    parents: List[int] = [-1]

    for _ in range(max_iters):
        if rng.random() < goal_sample_rate:
            sample = goal
        else:
            sample = (
                rng.uniform(clearance, map_size[0] - clearance),
                rng.uniform(clearance, map_size[1] - clearance),
            )
        nearest = min(range(len(nodes)), key=lambda idx: distance(nodes[idx], sample))
        new_point = _steer(nodes[nearest], sample, step_size)
        if distance(nodes[nearest], new_point) < 1e-6:
            continue
        if not _segment_is_free(nodes[nearest], new_point, map_size, obstacles, clearance):
            continue
        nodes.append(new_point)
        parents.append(nearest)
        new_index = len(nodes) - 1
        if distance(new_point, goal) <= goal_tolerance and _segment_is_free(new_point, goal, map_size, obstacles, clearance):
            nodes.append(goal)
            parents.append(new_index)
            return _shortcut_path(_reconstruct_path(nodes, parents, len(nodes) - 1), rng, map_size, obstacles, clearance)[1:]

    fallback = min(range(len(nodes)), key=lambda idx: distance(nodes[idx], goal))
    if _segment_is_free(nodes[fallback], goal, map_size, obstacles, clearance):
        nodes.append(goal)
        parents.append(fallback)
        return _shortcut_path(_reconstruct_path(nodes, parents, len(nodes) - 1), rng, map_size, obstacles, clearance)[1:]
    return []
