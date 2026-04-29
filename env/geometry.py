"""Geometry helpers shared by the combat environment and rule policies."""

from __future__ import annotations

import math
from typing import Any

Point = tuple[float, float]
Rect = tuple[float, float, float, float]

EPS = 1e-9


def normalize_angle(deg: float) -> float:
    return deg % 360.0


def angle_diff(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def angle_diff_deg(a: float, b: float) -> float:
    return angle_diff(a, b)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def direction_to(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def angle_to(a: Point, b: Point) -> float:
    return direction_to(a, b)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def point_in_rect(p: Point, rect: Rect) -> bool:
    x, y, w, h = rect
    return x <= p[0] <= x + w and y <= p[1] <= y + h


def rect_from_center_square(cx: float, cy: float, side: float) -> Rect:
    half = side / 2.0
    return (cx - half, cy - half, side, side)


def _expanded_rect(rect: Rect, margin: float) -> Rect:
    if margin <= 0.0:
        return rect
    x, y, w, h = rect
    return (x - margin, y - margin, w + 2.0 * margin, h + 2.0 * margin)


def _segment_intersects_rect(a: Point, b: Point, rect: Rect) -> bool:
    """Robust segment-vs-AABB intersection, including collinear contacts."""

    if point_in_rect(a, rect) or point_in_rect(b, rect):
        return True

    x, y, w, h = rect
    min_x, max_x = x, x + w
    min_y, max_y = y, y + h
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    t_min, t_max = 0.0, 1.0

    for origin, delta, lo, hi in ((a[0], dx, min_x, max_x), (a[1], dy, min_y, max_y)):
        if abs(delta) < EPS:
            if origin < lo or origin > hi:
                return False
            continue
        inv = 1.0 / delta
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False

    return True


def line_intersects_rect(a: Point, b: Point, rect: Rect) -> bool:
    return _segment_intersects_rect(a, b, rect)


def _ray_rect_entry_distance(origin: Point, direction: Point, rect: Rect) -> float | None:
    """Return the first forward distance from a ray to an AABB."""

    if point_in_rect(origin, rect):
        return 0.0

    x, y, w, h = rect
    min_x, max_x = x, x + w
    min_y, max_y = y, y + h
    t_min = -math.inf
    t_max = math.inf

    for origin_axis, delta, lo, hi in (
        (origin[0], direction[0], min_x, max_x),
        (origin[1], direction[1], min_y, max_y),
    ):
        if abs(delta) < EPS:
            if origin_axis < lo or origin_axis > hi:
                return None
            continue
        inv = 1.0 / delta
        t1 = (lo - origin_axis) * inv
        t2 = (hi - origin_axis) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None

    if t_max < 0.0:
        return None
    return max(t_min, 0.0)


def _ray_box_exit_distance(origin: Point, direction: Point, bounds: tuple[float, float, float, float]) -> float:
    """Distance to leave an axis-aligned box that contains the ray origin."""

    min_x, min_y, max_x, max_y = bounds
    ox, oy = origin
    if ox < min_x or ox > max_x or oy < min_y or oy > max_y:
        return 0.0

    candidates: list[float] = []
    if direction[0] > EPS:
        candidates.append((max_x - ox) / direction[0])
    elif direction[0] < -EPS:
        candidates.append((min_x - ox) / direction[0])
    if direction[1] > EPS:
        candidates.append((max_y - oy) / direction[1])
    elif direction[1] < -EPS:
        candidates.append((min_y - oy) / direction[1])

    forward = [t for t in candidates if t >= 0.0]
    if not forward:
        return 0.0
    return min(forward)


def ray_rect_distance(origin: Point, angle_deg: float, rect: Rect, margin: float = 0.0) -> float | None:
    rad = math.radians(angle_deg)
    direction = (math.cos(rad), math.sin(rad))
    return _ray_rect_entry_distance(origin, direction, _expanded_rect(rect, margin))


def ray_circle_distance(origin: Point, angle_deg: float, center: Point, radius: float) -> float | None:
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    ox, oy = origin[0] - center[0], origin[1] - center[1]
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = math.sqrt(max(disc, 0.0))
    t1 = (-b - root) / 2.0
    t2 = (-b + root) / 2.0
    if t2 < 0.0:
        return None
    return t1 if t1 >= 0.0 else t2


def _unpack_circle(circle: Any) -> tuple[Point, float, Any]:
    if len(circle) == 2:
        center, radius = circle
        payload = None
    else:
        center, radius, payload = circle
    return center, float(radius), payload


def raycast_hit(
    origin: Point,
    angle_deg: float,
    max_dist: float,
    map_size: tuple[int, int],
    obstacles: list[Rect],
    circles: list[Any] | None = None,
    *,
    boundary_margin: float = 0.0,
    obstacle_margin: float = 0.0,
) -> dict[str, Any]:
    """Return the first thing hit by a ray.

    Obstacles are axis-aligned rectangles. Circles are tuples of
    ``(center, radius)`` or ``(center, radius, payload)``.
    """

    rad = math.radians(angle_deg)
    direction = (math.cos(rad), math.sin(rad))
    best_distance = float(max_dist)
    best_kind = "max_range"
    best_payload: Any = None

    min_x = float(boundary_margin)
    min_y = float(boundary_margin)
    max_x = float(map_size[0]) - float(boundary_margin)
    max_y = float(map_size[1]) - float(boundary_margin)
    boundary_dist = _ray_box_exit_distance(origin, direction, (min_x, min_y, max_x, max_y))
    if boundary_dist <= best_distance:
        best_distance = boundary_dist
        best_kind = "boundary"
        best_payload = None

    for idx, rect in enumerate(obstacles):
        hit_distance = _ray_rect_entry_distance(origin, direction, _expanded_rect(rect, obstacle_margin))
        if hit_distance is not None and hit_distance < best_distance:
            best_distance = hit_distance
            best_kind = "obstacle"
            best_payload = idx

    for idx, circle in enumerate(circles or []):
        center, radius, payload = _unpack_circle(circle)
        hit_distance = ray_circle_distance(origin, angle_deg, center, radius)
        if hit_distance is not None and hit_distance < best_distance:
            best_distance = hit_distance
            best_kind = "circle"
            best_payload = payload if payload is not None else idx

    best_distance = max(0.0, min(float(max_dist), best_distance))
    point = (origin[0] + direction[0] * best_distance, origin[1] + direction[1] * best_distance)
    return {"distance": best_distance, "point": point, "kind": best_kind, "payload": best_payload}


def raycast_distance(
    origin: Point,
    angle_deg: float,
    max_dist: float,
    map_size: tuple[int, int],
    obstacles: list[Rect],
    circles: list[Any] | None = None,
    *,
    boundary_margin: float = 0.0,
    obstacle_margin: float = 0.0,
) -> float:
    return float(
        raycast_hit(
            origin,
            angle_deg,
            max_dist,
            map_size,
            obstacles,
            circles,
            boundary_margin=boundary_margin,
            obstacle_margin=obstacle_margin,
        )["distance"]
    )


def circle_rect_collision(center: Point, radius: float, rect: Rect) -> bool:
    x, y, w, h = rect
    cx = min(max(center[0], x), x + w)
    cy = min(max(center[1], y), y + h)
    return distance(center, (cx, cy)) <= radius


def circle_rect_overlap(center: Point, radius: float, rect: Rect) -> bool:
    return circle_rect_collision(center, radius, rect)


def circle_circle_overlap(a: Point, ar: float, b: Point, br: float) -> bool:
    return distance(a, b) <= ar + br


def has_line_of_sight(a: Point, b: Point, obstacles: list[Rect]) -> bool:
    return not any(line_intersects_rect(a, b, rect) for rect in obstacles)


def any_line_of_sight(origin: Point, targets: list[Point], obstacles: list[Rect]) -> bool:
    return any(has_line_of_sight(origin, target, obstacles) for target in targets)


def oriented_square_vertices(center: Point, radius: float, angle_deg: float) -> list[Point]:
    """Return the four body corners for a square tank footprint."""

    cx, cy = center
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    corners = []
    for lx, ly in ((radius, radius), (radius, -radius), (-radius, -radius), (-radius, radius)):
        corners.append((cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a))
    return corners


def blocked_by_enemy_units(a: Point, b: Point, blockers: list[Point], radius: float) -> bool:
    return any(point_segment_distance(blocker, a, b) <= radius for blocker in blockers)


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return distance(p, a)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj = (ax + t * dx, ay + t * dy)
    return distance(p, proj)
