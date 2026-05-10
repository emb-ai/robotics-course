"""2D VacMan control environment.

This module mirrors the gameplay logic from the JavaScript VacMan viewer while
dropping the Three.js presentation layer.  Coordinates use the viewer's world
frame: ``x`` to the right and ``z`` on the ground plane.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


DUST_SPACING = 0.25
SPEED = 1.0
AXLE = 0.5
CLEAN_R = 0.5
CATCH_R = 1.1
WALL_HALF_T = 0.15
VAC_RADIUS = 0.45
BASE_R = VAC_RADIUS * 1.25
BASE_VISUAL_R = VAC_RADIUS * 1.25
CAT_OCT_R = 0.5
CAT_WAYPOINT_R = 0.42
REPLAN_INTERVAL = 0.12
ARENA_PAD = 1.2
EPS = 1e-7
PATH_CLOSE_EPS = 1e-4
TARGET_REPLAN_R = 0.12
WAYPOINT_BLOCKED_EPS = 1e-4
MIN_BATTERY_CHARGE = 180.0
DUST_SWEEP_CHARGE_FACTOR = 1.8
ARENA_TRAVEL_CHARGE_FACTOR = 4.0
VACMAN_ACTION_SIZE = 2
VACMAN_OBSERVATION_KEYS = (
    "vacman",
    "catman",
    "base",
    "dust",
    "obstacles",
    "cat_cspace_obstacles",
    "arena",
    "dust_cell_size",
    "battery",
    "flags",
)


def default_cases_path() -> Path:
    return Path(__file__).resolve().parent / "open_cases.json"


def load_vacman_cases(path: Path | str | None = None) -> tuple[dict, ...]:
    """Load and validate the JSON cases used by the JavaScript viewer."""
    cases_path = default_cases_path() if path is None else Path(path)
    with cases_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("open_cases.json must contain a JSON array")
    cases = []
    for item in raw:
        validate_case(item)
        cases.append(item)
    return tuple(cases)


def fixed_vacman_cases(path: Path | str | None = None) -> tuple[str, ...]:
    """Return deterministic case ids for tests and examples."""
    return tuple(str(case.get("id", f"case_{i}")) for i, case in enumerate(load_vacman_cases(path)))


def validate_case(case_data: dict) -> None:
    if not isinstance(case_data, dict):
        raise ValueError("Invalid Vacman case")
    paths = case_data.get("paths")
    if not isinstance(paths, list):
        raise ValueError(f"Case {case_data.get('id', '?')!r}: paths must be a list")
    for pi, path in enumerate(paths):
        if not isinstance(path, list):
            raise ValueError(f"Case {case_data.get('id', '?')!r}: paths[{pi}] must be a list")
        for i, pt in enumerate(path):
            if (
                not isinstance(pt, list)
                or len(pt) < 2
                or not isinstance(pt[0], (int, float))
                or not isinstance(pt[1], (int, float))
            ):
                raise ValueError(
                    f"Case {case_data.get('id', '?')!r}: paths[{pi}][{i}] must be [x,z] numbers"
                )
    for key in ("base", "catman"):
        pt = case_data.get(key)
        if (
            not isinstance(pt, list)
            or len(pt) < 2
            or not isinstance(pt[0], (int, float))
            or not isinstance(pt[1], (int, float))
        ):
            raise ValueError(f"Case {case_data.get('id', '?')!r}: {key} [x,z] required")
    speed = case_data.get("catman_speed")
    if speed is not None and (not np.isfinite(speed) or float(speed) <= 0.0):
        raise ValueError(f"Case {case_data.get('id', '?')!r}: catman_speed must be positive")


def _pt(p: Iterable[float]) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.shape != (2,):
        raise ValueError(f"expected point with shape (2,), got {a.shape}")
    return a.copy()


def _pose(p: Iterable[float]) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"expected pose with shape (3,), got {a.shape}")
    return a.copy()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def cross(ax: float, az: float, bx: float, bz: float) -> float:
    return float(ax * bz - az * bx)


def convex_hull(points: Iterable[np.ndarray]) -> np.ndarray:
    pts = sorted((_pt(p) for p in points), key=lambda p: (float(p[0]), float(p[1])))
    if len(pts) < 3:
        return np.asarray(pts, dtype=float)

    def cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return cross(a[0] - o[0], a[1] - o[1], b[0] - o[0], b[1] - o[1])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross2(lower[-2], lower[-1], p) <= EPS:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross2(upper[-2], upper[-1], p) <= EPS:
            upper.pop()
        upper.append(p)

    upper.pop()
    lower.pop()
    return np.asarray(lower + upper, dtype=float)


def point_in_convex(px: float, pz: float, poly: np.ndarray) -> bool:
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        c = cross(b[0] - a[0], b[1] - a[1], px - a[0], pz - a[1])
        if abs(c) < EPS:
            continue
        csign = 1 if c > 0.0 else -1
        if sign == 0:
            sign = csign
        elif csign != sign:
            return False
    return True


def point_in_convex_strict(px: float, pz: float, poly: np.ndarray) -> bool:
    n = len(poly)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        c = cross(b[0] - a[0], b[1] - a[1], px - a[0], pz - a[1])
        if abs(c) <= EPS:
            return False
        csign = 1 if c > 0.0 else -1
        if sign == 0:
            sign = csign
        elif csign != sign:
            return False
    return True


def seg_intersect_proper(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])
    o2 = cross(b[0] - a[0], b[1] - a[1], d[0] - a[0], d[1] - a[1])
    o3 = cross(d[0] - c[0], d[1] - c[1], a[0] - c[0], a[1] - c[1])
    o4 = cross(d[0] - c[0], d[1] - c[1], b[0] - c[0], b[1] - c[1])
    return ((o1 > EPS and o2 < -EPS) or (o1 < -EPS and o2 > EPS)) and (
        (o3 > EPS and o4 < -EPS) or (o3 < -EPS and o4 > EPS)
    )


def point_on_segment(px: float, pz: float, a: np.ndarray, b: np.ndarray) -> bool:
    abx = b[0] - a[0]
    abz = b[1] - a[1]
    apx = px - a[0]
    apz = pz - a[1]
    if abs(cross(abx, abz, apx, apz)) > EPS:
        return False
    dot = apx * abx + apz * abz
    if dot < -EPS:
        return False
    len2 = abx * abx + abz * abz
    return bool(dot <= len2 + EPS)


def _segment_param(px: float, pz: float, a: np.ndarray, b: np.ndarray) -> float:
    dx = b[0] - a[0]
    dz = b[1] - a[1]
    len2 = dx * dx + dz * dz
    if len2 <= EPS:
        return 0.0
    return float(((px - a[0]) * dx + (pz - a[1]) * dz) / len2)


def _add_unique_param(params: list[float], t: float) -> None:
    if t <= EPS or t >= 1.0 - EPS:
        return
    if not any(abs(u - t) <= 1e-6 for u in params):
        params.append(float(t))


def segment_aabb(ax: float, az: float, bx: float, bz: float) -> tuple[float, float, float, float]:
    pad = 0.02
    return (
        min(ax, bx) - pad,
        max(ax, bx) + pad,
        min(az, bz) - pad,
        max(az, bz) + pad,
    )


def poly_aabb(poly: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(poly[:, 0])),
        float(np.max(poly[:, 0])),
        float(np.min(poly[:, 1])),
        float(np.max(poly[:, 1])),
    )


def aabb_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[1] and a[1] >= b[0] and a[2] <= b[3] and a[3] >= b[2]


def segment_visible(a_in: np.ndarray, b_in: np.ndarray, obstacles: Iterable[np.ndarray]) -> bool:
    a = _pt(a_in)
    b = _pt(b_in)
    obs = list(obstacles)
    if not obs:
        return True
    sa = segment_aabb(a[0], a[1], b[0], b[1])
    obs_boxes = [poly_aabb(poly) for poly in obs]
    for oi, poly in enumerate(obs):
        if not aabb_overlap(sa, obs_boxes[oi]):
            continue
        n = len(poly)
        if point_in_convex_strict(a[0], a[1], poly) or point_in_convex_strict(b[0], b[1], poly):
            return False
        touch_params: list[float] = []
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            if seg_intersect_proper(a, b, p1, p2):
                return False
            if point_on_segment(p1[0], p1[1], a, b):
                _add_unique_param(touch_params, _segment_param(p1[0], p1[1], a, b))
            if point_on_segment(a[0], a[1], p1, p2):
                _add_unique_param(touch_params, 0.0)
            if point_on_segment(b[0], b[1], p1, p2):
                _add_unique_param(touch_params, 1.0)
        touch_params.sort()
        params = [0.0, *touch_params, 1.0]
        for i in range(len(params) - 1):
            if params[i + 1] - params[i] <= 1e-6:
                continue
            t = 0.5 * (params[i] + params[i + 1])
            mx = a[0] + t * (b[0] - a[0])
            mz = a[1] + t * (b[1] - a[1])
            if point_in_convex_strict(mx, mz, poly):
                return False
    return True


def octagon_vertices(r: float) -> np.ndarray:
    return np.asarray(
        [[r * math.cos(math.pi / 8.0 + i * math.pi / 4.0), r * math.sin(math.pi / 8.0 + i * math.pi / 4.0)] for i in range(8)],
        dtype=float,
    )


def segment_to_rect(p0_in: np.ndarray, p1_in: np.ndarray, half_t: float) -> np.ndarray | None:
    p0 = _pt(p0_in)
    p1 = _pt(p1_in)
    dx = p1[0] - p0[0]
    dz = p1[1] - p0[1]
    length = float(np.hypot(dx, dz))
    if length < EPS:
        return None
    nx = (-dz / length) * half_t
    nz = (dx / length) * half_t
    return np.asarray(
        [
            [p0[0] + nx, p0[1] + nz],
            [p0[0] - nx, p0[1] - nz],
            [p1[0] - nx, p1[1] - nz],
            [p1[0] + nx, p1[1] + nz],
        ],
        dtype=float,
    )


def minkowski_sum_convex(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    pts = []
    for a in poly_a:
        for b in poly_b:
            pts.append(a + b)
    return convex_hull(pts)


def closest_point_on_seg(px: float, pz: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    abx = b[0] - a[0]
    abz = b[1] - a[1]
    t = clamp(((px - a[0]) * abx + (pz - a[1]) * abz) / (abx * abx + abz * abz + EPS), 0.0, 1.0)
    return np.asarray([a[0] + t * abx, a[1] + t * abz], dtype=float)


@dataclass
class WorldTransform:
    half_x: float
    half_z: float
    cx: float
    cz: float

    def to_world(self, jx: float, jz: float) -> np.ndarray:
        return np.asarray([self.cx - float(jx), float(jz) - self.cz], dtype=float)

    @staticmethod
    def to_world_heading(theta: float) -> float:
        return math.pi - float(theta)


def bbox_from_case(case_data: dict) -> tuple[float, float, float, float]:
    min_x = math.inf
    max_x = -math.inf
    min_z = math.inf
    max_z = -math.inf

    def add_pt(x: float, z: float) -> None:
        nonlocal min_x, max_x, min_z, max_z
        min_x = min(min_x, float(x))
        max_x = max(max_x, float(x))
        min_z = min(min_z, float(z))
        max_z = max(max_z, float(z))

    for path in case_data.get("paths", []):
        for pt in path:
            add_pt(pt[0], pt[1])
    if case_data.get("base") is not None:
        add_pt(case_data["base"][0], case_data["base"][1])
    if case_data.get("catman") is not None:
        add_pt(case_data["catman"][0], case_data["catman"][1])
    return min_x, max_x, min_z, max_z


def build_world_transform(case_data: dict) -> WorldTransform:
    min_x, max_x, min_z, max_z = bbox_from_case(case_data)
    cx = (min_x + max_x) / 2.0
    cz = (min_z + max_z) / 2.0
    width = max_x - min_x + 2.0 * ARENA_PAD
    depth = max_z - min_z + 2.0 * ARENA_PAD
    return WorldTransform(width / 2.0, depth / 2.0, cx, cz)


def collect_wall_rects(case_data: dict, transform: WorldTransform) -> list[np.ndarray]:
    rects: list[np.ndarray] = []
    for path in case_data.get("paths", []):
        if not path or len(path) < 2:
            continue
        for i in range(len(path) - 1):
            p0 = transform.to_world(path[i][0], path[i][1])
            p1 = transform.to_world(path[i + 1][0], path[i + 1][1])
            rect = segment_to_rect(p0, p1, WALL_HALF_T)
            if rect is not None:
                rects.append(rect)
        p_first = transform.to_world(path[0][0], path[0][1])
        p_last = transform.to_world(path[-1][0], path[-1][1])
        if dist(p_first, p_last) > PATH_CLOSE_EPS:
            rect = segment_to_rect(p_last, p_first, WALL_HALF_T)
            if rect is not None:
                rects.append(rect)
    return rects


def build_cspace_obstacles(wall_rects: Iterable[np.ndarray], oct_neg: np.ndarray) -> list[np.ndarray]:
    obstacles: list[np.ndarray] = []
    for rect in wall_rects:
        hull = minkowski_sum_convex(rect, oct_neg)
        if len(hull) >= 3:
            obstacles.append(hull)
    return obstacles


def merge_close_nodes(nodes: Iterable[np.ndarray], eps: float = 0.08) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p_in in nodes:
        p = _pt(p_in)
        if any(dist(p, q) < eps for q in out):
            continue
        out.append(p)
    return out


def build_visibility_graph(nodes: list[np.ndarray], obstacles: list[np.ndarray]) -> list[list[tuple[int, float]]]:
    n = len(nodes)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    obs_boxes = [poly_aabb(poly) for poly in obstacles]
    for i in range(n):
        ai = nodes[i]
        for j in range(i + 1, n):
            aj = nodes[j]
            seg_box = segment_aabb(ai[0], ai[1], aj[0], aj[1])
            maybe_blocked = any(aabb_overlap(seg_box, obs_box) for obs_box in obs_boxes)
            weight = dist(ai, aj)
            if not maybe_blocked or segment_visible(ai, aj, obstacles):
                adj[i].append((j, weight))
                adj[j].append((i, weight))
    return adj


def point_free_in_obstacles(p_in: np.ndarray, obstacles: Iterable[np.ndarray]) -> bool:
    p = _pt(p_in)
    return not any(point_in_convex_strict(p[0], p[1], poly) for poly in obstacles)


def astar(nodes: list[np.ndarray], adj: list[list[tuple[int, float]]], start_idx: int, goal_idx: int) -> list[np.ndarray] | None:
    n = len(nodes)
    g = np.full(n, np.inf, dtype=float)
    f = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    open_nodes = [start_idx]
    g[start_idx] = 0.0
    f[start_idx] = dist(nodes[start_idx], nodes[goal_idx])

    while open_nodes:
        best_pos = 0
        best = math.inf
        for pos, idx in enumerate(open_nodes):
            if f[idx] < best:
                best = f[idx]
                best_pos = pos
        u = open_nodes.pop(best_pos)
        if u == goal_idx:
            break
        for v, weight in adj[u]:
            ng = g[u] + weight
            if ng < g[v]:
                g[v] = ng
                parent[v] = u
                f[v] = ng + dist(nodes[v], nodes[goal_idx])
                if v not in open_nodes:
                    open_nodes.append(v)

    if parent[goal_idx] < 0 and goal_idx != start_idx:
        return None
    path: list[np.ndarray] = []
    cur = goal_idx
    max_steps = len(nodes) + 2
    for _ in range(max_steps):
        if cur < 0:
            return None
        path.append(nodes[cur])
        if cur == start_idx:
            break
        cur = int(parent[cur])
    if not path or dist(path[-1], nodes[start_idx]) > 1e-6:
        return None
    path.reverse()
    return [p.copy() for p in path]


def push_out_of_walls(px: float, pz: float, wall_rects: Iterable[np.ndarray], radius: float) -> np.ndarray:
    x = float(px)
    z = float(pz)
    for _ in range(4):
        for rect in wall_rects:
            n = len(rect)
            for i in range(n):
                a = rect[i]
                b = rect[(i + 1) % n]
                q = closest_point_on_seg(x, z, a, b)
                d = float(np.hypot(x - q[0], z - q[1]))
                if d < radius and d > EPS:
                    scale = radius / d
                    x = q[0] + (x - q[0]) * scale
                    z = q[1] + (z - q[1]) * scale
                elif d <= EPS and point_in_convex(x, z, rect):
                    center = np.mean(rect, axis=0)
                    dx = x - center[0]
                    dz = z - center[1]
                    length = float(np.hypot(dx, dz)) + EPS
                    x += (dx / length) * radius
                    z += (dz / length) * radius
    return np.asarray([x, z], dtype=float)


def cell_free_from_walls(wx: float, wz: float, wall_rects: Iterable[np.ndarray]) -> bool:
    return not any(point_in_convex(wx, wz, rect) for rect in wall_rects)


@dataclass
class CatCspaceContext:
    collision_obstacles: list[np.ndarray]
    planning_obstacles: list[np.ndarray]
    cspace_obstacles: list[np.ndarray]
    graph_nodes: list[np.ndarray]
    vis_adj: list[list[tuple[int, float]]]


def make_cat_cspace_context(wall_rects: list[np.ndarray]) -> CatCspaceContext:
    octagon = octagon_vertices(CAT_OCT_R)
    oct_neg = -octagon
    obstacles = build_cspace_obstacles(wall_rects, oct_neg)
    graph_nodes = merge_close_nodes(
        vertex
        for poly in obstacles
        for vertex in poly
        if point_free_in_obstacles(vertex, obstacles)
    )
    vis_adj = build_visibility_graph(graph_nodes, obstacles)
    return CatCspaceContext(
        collision_obstacles=obstacles,
        planning_obstacles=obstacles,
        cspace_obstacles=obstacles,
        graph_nodes=graph_nodes,
        vis_adj=vis_adj,
    )


@dataclass
class CatPlanner:
    graph_nodes: list[np.ndarray]
    vis_adj: list[list[tuple[int, float]]]
    visibility_obstacles: list[np.ndarray]
    path: list[np.ndarray] | None = None
    wp_idx: int = 0
    goal: np.ndarray | None = None
    partial: bool = False
    blocked_at_waypoint: bool = False

    def visible(self, a: np.ndarray, b: np.ndarray) -> bool:
        return segment_visible(a, b, self.visibility_obstacles)

    def move_visible(self, a: np.ndarray, b: np.ndarray) -> bool:
        return segment_visible(a, b, self.visibility_obstacles)

    def set_path_from_waypoints(self, waypoints: list[np.ndarray], goal: np.ndarray | None = None, partial: bool = False) -> None:
        self.path = [_pt(p) for p in waypoints]
        if len(self.path) == 1:
            self.path.append(self.path[0].copy())
        self.wp_idx = min(1, max(0, len(self.path) - 1))
        self.goal = _pt(goal) if goal is not None else (self.path[-1].copy() if self.path else None)
        self.partial = bool(partial)
        self.blocked_at_waypoint = False

    def _connect_visible_node(self, nodes: list[np.ndarray], adj: list[list[tuple[int, float]]], idx: int, count: int) -> None:
        for i in range(count):
            if self.visible(nodes[idx], nodes[i]):
                weight = dist(nodes[idx], nodes[i])
                adj[idx].append((i, weight))
                adj[i].append((idx, weight))

    @staticmethod
    def _reachable_from(adj: list[list[tuple[int, float]]], start_idx: int) -> np.ndarray:
        seen = np.zeros(len(adj), dtype=np.uint8)
        stack = [start_idx]
        seen[start_idx] = 1
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if seen[v]:
                    continue
                seen[v] = 1
                stack.append(v)
        return seen

    def _smooth_path(self, path: list[np.ndarray]) -> list[np.ndarray]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            nxt = i + 1
            for j in range(len(path) - 1, i + 1, -1):
                if self.move_visible(path[i], path[j]):
                    nxt = j
                    break
            out.append(path[nxt])
            i = nxt
        return out

    def plan_cat_path(self, agent: np.ndarray, target: np.ndarray) -> None:
        start = _pt(agent)
        goal = _pt(target)
        nodes = [n.copy() for n in self.graph_nodes]
        si = len(nodes)
        gi = len(nodes) + 1
        nodes.extend([start, goal])
        adj = [[(int(v), float(w)) for v, w in row] for row in self.vis_adj] + [[], []]
        self._connect_visible_node(nodes, adj, si, si)
        self._connect_visible_node(nodes, adj, gi, gi)
        goal_reach = self._reachable_from(adj, gi)

        def fallback_toward_visible() -> None:
            if self.visible(start, goal):
                self.set_path_from_waypoints([start, goal], goal, False)
                return
            best_k = -1
            best_d = math.inf
            for require_goal_reach in (True, False):
                for k, node in enumerate(self.graph_nodes):
                    if require_goal_reach and not goal_reach[k]:
                        continue
                    if not self.visible(start, node):
                        continue
                    d = dist(start, node) + dist(node, goal)
                    if d < best_d:
                        best_d = d
                        best_k = k
                if best_k >= 0:
                    break
            if best_k >= 0:
                self.set_path_from_waypoints([start, self.graph_nodes[best_k]], goal, True)
                return
            self.set_path_from_waypoints([start, start], goal, True)

        if dist(start, goal) < 0.4:
            if self.visible(start, goal):
                self.set_path_from_waypoints([start, goal], goal, False)
            else:
                fallback_toward_visible()
            return

        if len(adj[si]) == 0:
            fallback_toward_visible()
            return

        path = astar(nodes, adj, si, gi)
        path_ok = (
            path is not None
            and len(path) >= 2
            and dist(path[0], start) < 0.25
            and dist(path[-1], goal) < 0.25
        )
        if path_ok:
            self.set_path_from_waypoints(self._smooth_path(path), goal, False)
        else:
            fallback_toward_visible()

    def advance_waypoints(self, agent: np.ndarray, waypoint_radius: float) -> None:
        self.blocked_at_waypoint = False
        while self.path is not None and self.wp_idx < len(self.path):
            wp = self.path[self.wp_idx]
            d = dist(agent, wp)
            if d > waypoint_radius:
                break
            next_idx = self.wp_idx + 1
            if next_idx < len(self.path) and not self.move_visible(agent, self.path[next_idx]):
                self.blocked_at_waypoint = d <= min(WAYPOINT_BLOCKED_EPS, waypoint_radius * 0.25)
                break
            self.wp_idx += 1

    def current_waypoint(self) -> np.ndarray | None:
        if self.path is None or self.wp_idx >= len(self.path):
            return None
        return self.path[self.wp_idx]

    def path_usable(self, agent: np.ndarray) -> bool:
        if self.blocked_at_waypoint:
            return False
        wp = self.current_waypoint()
        if wp is None:
            return False
        return self.move_visible(agent, wp)

    def target_moved(self, target: np.ndarray, threshold: float = TARGET_REPLAN_R) -> bool:
        return self.partial or self.goal is None or dist(self.goal, target) > threshold

    def shortcut_path(self, agent: np.ndarray, target: np.ndarray) -> bool:
        if self.path is None or self.wp_idx >= len(self.path):
            return False
        start = _pt(agent)
        goal = _pt(target)
        if self.move_visible(start, goal):
            self.set_path_from_waypoints([start, goal], goal, False)
            return True
        best = self.wp_idx
        for i in range(len(self.path) - 1, self.wp_idx, -1):
            if self.move_visible(start, self.path[i]):
                best = i
                break
        self.wp_idx = best
        return self.path_usable(agent)

    def movement_safe(self, from_pt: np.ndarray, to_pt: np.ndarray) -> bool:
        return self.move_visible(from_pt, to_pt)

    def reset_path(self) -> None:
        self.path = None
        self.wp_idx = 0
        self.goal = None
        self.partial = False
        self.blocked_at_waypoint = False

    def diagnostics(self) -> dict:
        path = [] if self.path is None else [p.copy() for p in self.path]
        length = 0.0
        for i in range(1, len(path)):
            length += dist(path[i - 1], path[i])
        return {
            "wp_idx": self.wp_idx,
            "partial": self.partial,
            "blocked_at_waypoint": self.blocked_at_waypoint,
            "goal": None if self.goal is None else self.goal.copy(),
            "waypoint": None if self.current_waypoint() is None else self.current_waypoint().copy(),
            "path": path,
            "path_length": length,
        }


@dataclass
class DustSystem:
    arena_x: float
    arena_z: float
    half_x: float
    half_z: float
    base_pos: np.ndarray
    wall_rects: list[np.ndarray]
    grid_width: int = field(init=False)
    grid_height: int = field(init=False)
    cell_size_x: float = field(init=False)
    cell_size_z: float = field(init=False)
    cell_size_min: float = field(init=False)
    dust_grid: np.ndarray = field(init=False)
    total_dust: int = field(default=0, init=False)
    cleaned: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.grid_width = max(1, int(math.ceil(self.arena_x / DUST_SPACING)))
        self.grid_height = max(1, int(math.ceil(self.arena_z / DUST_SPACING)))
        self.cell_size_x = self.arena_x / self.grid_width
        self.cell_size_z = self.arena_z / self.grid_height
        self.cell_size_min = min(self.cell_size_x, self.cell_size_z)
        self.dust_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

    def fill_dust(self) -> None:
        self.dust_grid.fill(0)
        self.total_dust = 0
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                wx = ((gx + 0.5) / self.grid_width) * self.arena_x - self.half_x
                wz = ((gy + 0.5) / self.grid_height) * self.arena_z - self.half_z
                if np.hypot(wx - self.base_pos[0], wz - self.base_pos[1]) < BASE_VISUAL_R + self.cell_size_min * 0.5:
                    continue
                if not cell_free_from_walls(wx, wz, self.wall_rects):
                    continue
                self.dust_grid[gy, gx] = 1
                self.total_dust += 1

    def reset_cleaned(self) -> None:
        self.cleaned = 0

    def clean_at(self, wx: float, wz: float) -> bool:
        cx = int(math.floor(((wx + self.half_x) / self.arena_x) * self.grid_width))
        cy = int(math.floor(((wz + self.half_z) / self.arena_z) * self.grid_height))
        r_cells = int(math.ceil(CLEAN_R / self.cell_size_min)) + 1
        hit = False
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                px = cx + dx
                py = cy + dy
                if px < 0 or px >= self.grid_width or py < 0 or py >= self.grid_height:
                    continue
                wcx = ((px + 0.5) / self.grid_width) * self.arena_x - self.half_x
                wcz = ((py + 0.5) / self.grid_height) * self.arena_z - self.half_z
                if np.hypot(wcx - wx, wcz - wz) > CLEAN_R:
                    continue
                if self.dust_grid[py, px]:
                    self.dust_grid[py, px] = 0
                    self.cleaned += 1
                    hit = True
        return hit


@dataclass
class Agent:
    x: float
    z: float
    th: float

    def xy(self) -> np.ndarray:
        return np.asarray([self.x, self.z], dtype=float)

    def pose(self) -> np.ndarray:
        return np.asarray([self.x, self.z, self.th], dtype=float)

    def set_xy(self, p: np.ndarray) -> None:
        self.x = float(p[0])
        self.z = float(p[1])

    def set_pose(self, p: np.ndarray) -> None:
        self.x = float(p[0])
        self.z = float(p[1])
        self.th = float(p[2])


def differential_drive_twist(v_left: float, v_right: float, axle_length: float = AXLE) -> tuple[float, float]:
    return (float(v_left) + float(v_right)) / 2.0, (float(v_right) - float(v_left)) / axle_length


def step_planar_unicycle(agent: Agent, dt: float, v: float, omega: float) -> None:
    agent.x += v * math.cos(agent.th) * dt
    agent.z += v * math.sin(agent.th) * dt
    agent.th += omega * dt


def catman_linear_speed(case_speed: float | None = None) -> float:
    if case_speed is not None and np.isfinite(case_speed) and float(case_speed) > 0.0:
        return float(case_speed)
    return SPEED


def estimate_battery_charge(dust: DustSystem, arena_x: float, arena_z: float) -> float:
    dust_sweep = dust.total_dust * dust.cell_size_min * DUST_SWEEP_CHARGE_FACTOR
    arena_travel = math.hypot(arena_x, arena_z) * ARENA_TRAVEL_CHARGE_FACTOR
    return float(math.ceil(max(MIN_BATTERY_CHARGE, dust_sweep + arena_travel)))


@dataclass
class VacmanEnvConfig:
    cases_path: Path = field(default_factory=default_cases_path)
    case_id: str | None = None
    case_index: int = 0
    dt: float = 0.05
    action_limit: float = SPEED * 1.5
    time_limit: float | None = None
    render_width: int = 640
    render_height: int = 640
    debug: bool = False


class VacmanEnv:
    """Gym-like 2D Vacman environment.

    ``step(action)`` expects wheel speeds ``[v_left, v_right]``.  Observations
    are dictionaries because cases have variable-size maps and obstacle lists.
    """

    def __init__(self, config: VacmanEnvConfig | None = None):
        self.config = config if config is not None else VacmanEnvConfig()
        if self.config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.config.action_limit <= 0.0:
            raise ValueError("action_limit must be positive")
        self.cases = load_vacman_cases(self.config.cases_path)
        if not self.cases:
            raise ValueError("no Vacman cases available")

        self.case_index = 0
        self.case_data: dict = {}
        self.case_id = ""
        self.transform = WorldTransform(1.0, 1.0, 0.0, 0.0)
        self.wall_rects: list[np.ndarray] = []
        self.cspace = CatCspaceContext([], [], [], [], [])
        self.cat_planner = CatPlanner([], [], [])
        self.base_pos = np.zeros(2, dtype=float)
        self.base_heading = 0.0
        self.vac_start = np.zeros(2, dtype=float)
        self.cat_start = np.zeros(2, dtype=float)
        self.vac = Agent(0.0, 0.0, 0.0)
        self.cat = Agent(0.0, 0.0, 0.0)
        self.dust = DustSystem(1.0, 1.0, 0.5, 0.5, np.zeros(2), [])
        self.cat_speed = SPEED
        self.t = 0.0
        self.battery_capacity = 0.0
        self.battery_remaining = 0.0
        self.left_base = False
        self.vacman_moved = False
        self.replan_acc = 0.0
        self.cat_frame = 0
        self.cat_still_frames = 0
        self.last_action = np.zeros(VACMAN_ACTION_SIZE, dtype=float)
        self.status = "ready"
        self._last_cat_move_status = "inactive"
        self._last_chase_target: np.ndarray | None = None

        self._load_case(self._initial_case_index())

    @property
    def action_size(self) -> int:
        return VACMAN_ACTION_SIZE

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return VACMAN_OBSERVATION_KEYS

    @property
    def half_x(self) -> float:
        return self.transform.half_x

    @property
    def half_z(self) -> float:
        return self.transform.half_z

    @property
    def arena_x(self) -> float:
        return self.transform.half_x * 2.0

    @property
    def arena_z(self) -> float:
        return self.transform.half_z * 2.0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        del seed
        options = {} if options is None else dict(options)
        if "case_id" in options or "case_index" in options:
            self._load_case(self._case_index_from_options(options))
        else:
            self._load_case(self.case_index)

        self.vac = Agent(
            float(self.vac_start[0]),
            float(self.vac_start[1]),
            WorldTransform.to_world_heading(self.case_data["base"][2] if len(self.case_data["base"]) > 2 else 0.0),
        )
        self.cat = Agent(
            float(self.cat_start[0]),
            float(self.cat_start[1]),
            WorldTransform.to_world_heading(self.case_data["catman"][2] if len(self.case_data["catman"]) > 2 else 0.0),
        )
        if "vacman" in options:
            self.vac.set_pose(_pose(options["vacman"]))
        if "vacman_pose" in options:
            self.vac.set_pose(_pose(options["vacman_pose"]))
        if "catman" in options:
            self.cat.set_pose(_pose(options["catman"]))
            self.cat.set_xy(self.project_cat_body_point(self.cat.x, self.cat.z))
        if "catman_pose" in options:
            self.cat.set_pose(_pose(options["catman_pose"]))
            self.cat.set_xy(self.project_cat_body_point(self.cat.x, self.cat.z))

        self.t = 0.0
        self.left_base = False
        self.vacman_moved = False
        self.replan_acc = 0.0
        self.cat_frame = 0
        self.cat_still_frames = 0
        self.last_action = np.zeros(VACMAN_ACTION_SIZE, dtype=float)
        self.cat_planner.reset_path()
        self.dust.reset_cleaned()
        self.dust.fill_dust()
        self.battery_capacity = estimate_battery_charge(self.dust, self.arena_x, self.arena_z)
        self.battery_remaining = self.battery_capacity
        self.status = "playing"
        self._last_cat_move_status = "inactive"
        self._last_chase_target = None
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        u = self._clip_action(action)
        self.last_action = u
        dt = float(self.config.dt)

        self.t += dt
        self.battery_remaining = max(0.0, self.battery_remaining - dt)
        if self.battery_remaining <= 0.0:
            self.status = "out_of_charge"
            return self._observation(), -1.0, True, False, self._info()

        v_left, v_right = float(u[0]), float(u[1])
        if abs(v_left) > 1e-3 or abs(v_right) > 1e-3:
            self.vacman_moved = True
        cat_active = self.vacman_moved

        v, omega = differential_drive_twist(v_left, v_right)
        step_planar_unicycle(self.vac, dt, v, omega)
        pushed = push_out_of_walls(self.vac.x, self.vac.z, self.wall_rects, VAC_RADIUS)
        self.vac.x = clamp(pushed[0], -self.half_x + VAC_RADIUS, self.half_x - VAC_RADIUS)
        self.vac.z = clamp(pushed[1], -self.half_z + VAC_RADIUS, self.half_z - VAC_RADIUS)

        base_center_distance = dist(self.vac.xy(), self.base_pos)
        if base_center_distance > BASE_VISUAL_R:
            self.left_base = True
        self.dust.clean_at(self.vac.x, self.vac.z)

        if cat_active:
            self._advance_cat(dt)
        else:
            self._last_cat_move_status = "inactive"

        terminated = False
        truncated = False
        reward = 0.0
        if cat_active and dist(self.vac.xy(), self.cat.xy()) < CATCH_R:
            self.status = "caught"
            reward = -1.0
            terminated = True
        elif self.left_base and base_center_distance < BASE_VISUAL_R:
            self.status = "success"
            reward = self.dust_fraction()
            terminated = True
        elif self.config.time_limit is not None and self.t >= self.config.time_limit - 1e-12:
            self.status = "timeout"
            reward = -1.0
            truncated = True
        else:
            self.status = "playing"

        return self._observation(), reward, terminated, truncated, self._info()

    def dust_fraction(self) -> float:
        if self.dust.total_dust <= 0:
            return 0.0
        return float(self.dust.cleaned / self.dust.total_dust)

    def render(self, mode: str = "rgb_array"):
        if mode == "matplotlib":
            return self._render_matplotlib()
        if mode != "rgb_array":
            raise ValueError("render mode must be 'rgb_array' or 'matplotlib'")
        return self._render_rgb_array()

    def project_cat_body_point(self, x: float, z: float, from_pt: np.ndarray | None = None) -> np.ndarray:
        raw = self._clamp_cat_point(np.asarray([x, z], dtype=float))
        if self._free_for_cat(raw):
            return raw
        pushed = self._clamp_cat_point(push_out_of_walls(x, z, self.wall_rects, CAT_OCT_R))
        if self._free_for_cat(pushed):
            return pushed
        best = self._choose_nearest_free_cat_point(raw, from_pt, [pushed])
        return pushed if best is None else best

    def project_cat_chase_target(self, x: float, z: float, from_pt: np.ndarray) -> np.ndarray:
        raw = self._clamp_cat_point(np.asarray([x, z], dtype=float))
        if self._free_for_cat(raw):
            return raw

        pushed = self.project_cat_body_point(x, z, from_pt)
        best = pushed if self._free_for_cat(pushed) else None
        best_score = dist(best, raw) + 0.08 * dist(best, from_pt) if best is not None else math.inf
        rings = [0.18, 0.32, 0.5, 0.72, 0.96, 1.22, 1.52]
        samples = 40
        for r in rings:
            for i in range(samples):
                a = (i / samples) * math.pi * 2.0
                p = self._clamp_cat_point(np.asarray([x + math.cos(a) * r, z + math.sin(a) * r], dtype=float))
                if not self._free_for_cat(p):
                    continue
                score = dist(p, raw) + 0.08 * dist(p, from_pt)
                if score < best_score:
                    best = p
                    best_score = score
            if best is not None and best_score < r + 0.2:
                break
        return pushed if best is None else best

    def _initial_case_index(self) -> int:
        if self.config.case_id is not None:
            return self._case_index_by_id(self.config.case_id)
        return int(np.clip(self.config.case_index, 0, len(self.cases) - 1))

    def _case_index_from_options(self, options: dict) -> int:
        if "case_id" in options:
            return self._case_index_by_id(str(options["case_id"]))
        return int(np.clip(int(options["case_index"]), 0, len(self.cases) - 1))

    def _case_index_by_id(self, case_id: str) -> int:
        for i, case in enumerate(self.cases):
            if str(case.get("id", f"case_{i}")) == case_id:
                return i
        raise ValueError(f"unknown Vacman case_id: {case_id}")

    def _load_case(self, case_index: int) -> None:
        self.case_index = int(np.clip(case_index, 0, len(self.cases) - 1))
        self.case_data = self.cases[self.case_index]
        self.case_id = str(self.case_data.get("id", f"case_{self.case_index}"))
        self.transform = build_world_transform(self.case_data)
        self.wall_rects = collect_wall_rects(self.case_data, self.transform)
        self.cspace = make_cat_cspace_context(self.wall_rects)
        self.cat_planner = CatPlanner(
            graph_nodes=[p.copy() for p in self.cspace.graph_nodes],
            vis_adj=[list(row) for row in self.cspace.vis_adj],
            visibility_obstacles=[poly.copy() for poly in self.cspace.cspace_obstacles],
        )
        base_json = self.case_data["base"]
        cat_json = self.case_data["catman"]
        self.base_pos = self.transform.to_world(base_json[0], base_json[1])
        self.base_heading = WorldTransform.to_world_heading(base_json[2] if len(base_json) > 2 else 0.0)
        self.vac_start = self.base_pos.copy()
        cat_start_raw = self.transform.to_world(cat_json[0], cat_json[1])
        self.dust = DustSystem(self.arena_x, self.arena_z, self.half_x, self.half_z, self.base_pos.copy(), self.wall_rects)
        self.dust.fill_dust()
        self.cat_speed = catman_linear_speed(self.case_data.get("catman_speed"))
        self.cat_start = self.project_cat_body_point(cat_start_raw[0], cat_start_raw[1])

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        u = np.asarray(action, dtype=float)
        if u.shape != (VACMAN_ACTION_SIZE,):
            raise ValueError(f"action must have shape ({VACMAN_ACTION_SIZE},), got {u.shape}")
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(u, -self.config.action_limit, self.config.action_limit)

    def _advance_cat(self, dt: float) -> None:
        self.cat_frame += 1
        self.replan_acc += dt
        cat_before = self.cat.xy()
        cat_free = self.project_cat_body_point(self.cat.x, self.cat.z, self.cat.xy())
        self.cat.set_xy(cat_free)
        chase_target = self.project_cat_chase_target(self.vac.x, self.vac.z, self.cat.xy())
        self._last_chase_target = chase_target.copy()

        self.cat_planner.advance_waypoints(self.cat.xy(), CAT_WAYPOINT_R)
        usable_before_plan = self.cat_planner.path_usable(self.cat.xy())
        target_moved = self.cat_planner.target_moved(chase_target)
        if (not usable_before_plan) or (target_moved and self.replan_acc >= REPLAN_INTERVAL):
            self.replan_acc = 0.0
            self.cat_planner.plan_cat_path(self.cat.xy(), chase_target)
            self.cat_planner.advance_waypoints(self.cat.xy(), CAT_WAYPOINT_R)

        self.cat_planner.shortcut_path(self.cat.xy(), chase_target)
        if not self.cat_planner.path_usable(self.cat.xy()) and self.replan_acc >= REPLAN_INTERVAL:
            self.replan_acc = 0.0
            self.cat_planner.plan_cat_path(self.cat.xy(), chase_target)
            self.cat_planner.advance_waypoints(self.cat.xy(), CAT_WAYPOINT_R)

        wp = self.cat_planner.current_waypoint()
        usable_for_move = self.cat_planner.path_usable(self.cat.xy())
        move_status = "path-unusable" if wp is not None else "no-waypoint"
        if wp is not None and self.cat_planner.blocked_at_waypoint:
            move_status = "blocked-at-waypoint"
        if wp is not None and usable_for_move:
            delta = wp - self.cat.xy()
            d = float(np.hypot(delta[0], delta[1]))
            if d > 1e-4:
                self.cat.th = math.atan2(delta[1], delta[0])
                step = min(self.cat_speed * dt, d)
                nxt = self.cat.xy() + (delta / d) * step
                if self.cat_planner.movement_safe(self.cat.xy(), nxt):
                    self.cat.set_xy(nxt)
                    move_status = "moved"
                else:
                    move_status = "blocked-step"
                    self.replan_acc = REPLAN_INTERVAL
                    self.cat_planner.reset_path()
            else:
                move_status = "at-waypoint"

        projected = self.project_cat_body_point(self.cat.x, self.cat.z, cat_before)
        self.cat.set_xy(projected)
        self.cat_planner.advance_waypoints(self.cat.xy(), CAT_WAYPOINT_R)
        moved = dist(cat_before, self.cat.xy())
        target_dist = dist(self.cat.xy(), chase_target)
        if target_dist >= CATCH_R and moved < 1e-5:
            self.cat_still_frames += 1
        else:
            self.cat_still_frames = 0
        self._last_cat_move_status = move_status

    def _clamp_cat_point(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                clamp(p[0], -self.half_x + CAT_OCT_R, self.half_x - CAT_OCT_R),
                clamp(p[1], -self.half_z + CAT_OCT_R, self.half_z - CAT_OCT_R),
            ],
            dtype=float,
        )

    def _free_for_cat(self, p: np.ndarray) -> bool:
        return point_free_in_obstacles(p, self.cspace.collision_obstacles)

    def _choose_nearest_free_cat_point(
        self,
        raw_point: np.ndarray,
        from_pt: np.ndarray | None = None,
        preferred: list[np.ndarray] | None = None,
    ) -> np.ndarray | None:
        raw = self._clamp_cat_point(raw_point)
        anchor = self._clamp_cat_point(from_pt) if from_pt is not None else raw
        best: np.ndarray | None = None
        best_score = math.inf

        def consider(p_in: np.ndarray) -> None:
            nonlocal best, best_score
            q = self._clamp_cat_point(p_in)
            if not self._free_for_cat(q):
                return
            score = dist(q, raw) + 0.05 * dist(q, anchor)
            if score < best_score:
                best = q
                best_score = score

        for p in preferred or []:
            consider(p)
        consider(raw)
        rings = [0.06, 0.12, 0.22, 0.36, 0.54, 0.76, 1.02, 1.34, 1.72, 2.16]
        samples = 56
        for r in rings:
            for i in range(samples):
                a = (i / samples) * math.pi * 2.0
                consider(np.asarray([raw[0] + math.cos(a) * r, raw[1] + math.sin(a) * r], dtype=float))
            if best is not None and best_score < r + 0.12:
                break
        return best

    def _observation(self) -> dict:
        return {
            "vacman": self.vac.pose(),
            "catman": self.cat.pose(),
            "base": np.asarray([self.base_pos[0], self.base_pos[1], self.base_heading], dtype=float),
            "dust": self.dust.dust_grid.copy(),
            "obstacles": tuple(rect.copy() for rect in self.wall_rects),
            "cat_cspace_obstacles": tuple(poly.copy() for poly in self.cspace.cspace_obstacles),
            "arena": np.asarray([-self.half_x, self.half_x, -self.half_z, self.half_z], dtype=float),
            "dust_cell_size": np.asarray([self.dust.cell_size_x, self.dust.cell_size_z], dtype=float),
            "battery": np.asarray([self.battery_remaining, self.battery_capacity], dtype=float),
            "flags": np.asarray([float(self.left_base), float(self.vacman_moved)], dtype=float),
        }

    def _info(self) -> dict:
        info = {
            "t": self.t,
            "status": self.status,
            "case_id": self.case_id,
            "action": self.last_action.copy(),
            "cleaned": self.dust.cleaned,
            "total_dust": self.dust.total_dust,
            "dust_fraction": self.dust_fraction(),
            "battery_remaining": self.battery_remaining,
            "battery_capacity": self.battery_capacity,
            "left_base": self.left_base,
            "cat_active": self.vacman_moved,
            "caught": self.status == "caught",
            "success": self.status == "success",
            "vacman": self.vac.pose(),
            "catman": self.cat.pose(),
            "base_distance": dist(self.vac.xy(), self.base_pos),
            "cat_distance": dist(self.vac.xy(), self.cat.xy()),
        }
        if self.config.debug:
            info.update(
                graph_nodes=len(self.cspace.graph_nodes),
                graph_edges=sum(len(row) for row in self.cspace.vis_adj) // 2,
                cat_move_status=self._last_cat_move_status,
                chase_target=None if self._last_chase_target is None else self._last_chase_target.copy(),
                cat_planner=self.cat_planner.diagnostics(),
                wall_rects=tuple(rect.copy() for rect in self.wall_rects),
                cspace_obstacles=tuple(poly.copy() for poly in self.cspace.cspace_obstacles),
            )
        return info

    def _render_rgb_array(self) -> np.ndarray:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex-cache"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = self._render_matplotlib()
        fig.set_size_inches(self.config.render_width / 100, self.config.render_height / 100)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()
        plt.close(fig)
        return rgb

    def _render_matplotlib(self):
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex-cache"))
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="#0d1117")
        ax.set_facecolor("#03070c")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-self.half_x, self.half_x)
        ax.set_ylim(-self.half_z, self.half_z)
        ax.tick_params(colors="#d8e8e8")
        for spine in ax.spines.values():
            spine.set_color("#3dff8c")

        ys, xs = np.nonzero(self.dust.dust_grid)
        if len(xs):
            wx = ((xs + 0.5) / self.dust.grid_width) * self.arena_x - self.half_x
            wz = ((ys + 0.5) / self.dust.grid_height) * self.arena_z - self.half_z
            ax.scatter(wx, wz, s=4, c="#ffd21f", alpha=0.75, marker="s", linewidths=0)

        for rect in self.wall_rects:
            ax.add_patch(patches.Polygon(rect, closed=True, facecolor="#2636b8", edgecolor="#6f86ff", lw=1.0, alpha=0.95))

        base_color = "#3dff8c" if self.left_base else "#2fb76b"
        ax.add_patch(patches.Circle(self.base_pos, BASE_VISUAL_R, fill=False, edgecolor=base_color, lw=2.0, alpha=0.9))
        ax.scatter([self.base_pos[0]], [self.base_pos[1]], c=base_color, s=50, marker="s", alpha=0.8)

        if self.config.debug and self.cat_planner.path:
            path = np.asarray(self.cat_planner.path)
            ax.plot(path[:, 0], path[:, 1], color="#ff9fb0", lw=1.3, alpha=0.75)

        mouth = math.degrees(math.pi * 0.24)
        heading = math.degrees(self.vac.th)
        ax.add_patch(
            patches.Wedge(
                (self.vac.x, self.vac.z),
                VAC_RADIUS,
                heading + mouth,
                heading + 360.0 - mouth,
                facecolor="#ffd21f",
                edgecolor="#fff1a8",
                lw=1.0,
            )
        )
        ax.plot(
            [self.vac.x, self.vac.x + math.cos(self.vac.th) * VAC_RADIUS * 1.25],
            [self.vac.z, self.vac.z + math.sin(self.vac.th) * VAC_RADIUS * 1.25],
            color="#050505",
            lw=1.0,
        )

        ax.add_patch(
            patches.Rectangle(
                (self.cat.x - CAT_OCT_R / 2.0, self.cat.z - CAT_OCT_R / 2.0),
                CAT_OCT_R,
                CAT_OCT_R,
                facecolor="#ff5252",
                edgecolor="#ffd1d1",
                lw=1.0,
            )
        )
        ax.plot(
            [self.cat.x, self.cat.x + math.cos(self.cat.th) * CAT_OCT_R],
            [self.cat.z, self.cat.z + math.sin(self.cat.th) * CAT_OCT_R],
            color="#ffd1d1",
            lw=1.0,
        )

        title = (
            f"Vacman {self.case_id}: {self.status} | dust {self.dust.cleaned}/{self.dust.total_dust} "
            f"| charge {self.battery_remaining:.0f}/{self.battery_capacity:.0f}"
        )
        ax.set_title(title, color="#d8e8e8", fontsize=10)
        ax.set_xlabel("x", color="#d8e8e8")
        ax.set_ylabel("z", color="#d8e8e8")
        fig.tight_layout()
        return fig
