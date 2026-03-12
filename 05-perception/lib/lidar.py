from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.special import erf


@dataclass
class BeamModel:
    z_max: float = 10.0
    sigma_hit: float = 0.2
    lambda_short: float = 2.0
    alpha_hit: float = 0.85
    alpha_short: float = 0.05
    alpha_max: float = 0.05
    alpha_rand: float = 0.05

    def pdf(self, z: np.ndarray, z_true: float) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        p_hit = _truncated_gaussian(z, z_true, self.sigma_hit, 0.0, self.z_max)

        p_short = self.lambda_short * np.exp(-self.lambda_short * z)
        p_short[z < 0] = 0
        p_short[z > z_true] = 0
        if z_true > 0:
            norm = 1 - np.exp(-self.lambda_short * z_true)
            p_short = p_short / max(norm, 1e-12)

        p_max = _discrete_mass_as_density(z, self.z_max)

        p_rand = np.where((z >= 0) & (z <= self.z_max), 1.0 / self.z_max, 0.0)

        return (self.alpha_hit * p_hit +
                self.alpha_short * p_short +
                self.alpha_max * p_max +
                self.alpha_rand * p_rand)

    def pdf_components(self, z: np.ndarray, z_true: float) -> dict[str, np.ndarray]:
        z = np.asarray(z, dtype=np.float64)
        p_hit = _truncated_gaussian(z, z_true, self.sigma_hit, 0.0, self.z_max)

        p_short = self.lambda_short * np.exp(-self.lambda_short * z)
        p_short[(z < 0) | (z > z_true)] = 0
        if z_true > 0:
            norm = 1 - np.exp(-self.lambda_short * z_true)
            p_short /= max(norm, 1e-12)

        p_max = _discrete_mass_as_density(z, self.z_max)

        p_rand = np.where((z >= 0) & (z <= self.z_max), 1.0 / self.z_max, 0.0)

        return {"hit": p_hit, "short": p_short, "max": p_max, "rand": p_rand}


@dataclass
class LikelihoodField:
    occupancy_map: np.ndarray
    resolution: float = 0.05
    sigma: float = 0.3
    origin: tuple[float, float] = (0.0, 0.0)

    _dist_transform: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dist_transform = distance_transform_edt(1 - self.occupancy_map) * self.resolution

    def log_likelihood(self, endpoints: np.ndarray) -> float:
        """endpoints: (N, 2) in world coordinates."""
        grid_coords = ((endpoints - np.array(self.origin)) / self.resolution).astype(int)
        h, w = self.occupancy_map.shape
        valid = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < w) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < h)
        )
        d = np.full(len(endpoints), self.sigma * 3)
        d[valid] = self._dist_transform[grid_coords[valid, 1], grid_coords[valid, 0]]
        return float(np.sum(-d**2 / (2 * self.sigma**2)))


class OccupancyGrid:
    def __init__(
        self,
        size: tuple[int, int] = (200, 200),
        resolution: float = 0.05,
        origin: tuple[float, float] = (-5.0, -5.0),
        l_occ: float = 0.85,
        l_free: float = -0.4,
    ):
        self.size = size
        self.resolution = resolution
        self.origin = origin
        self.l_occ = l_occ
        self.l_free = l_free
        self.log_odds = np.zeros(size, dtype=np.float64)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def update(self, robot_pos: np.ndarray, endpoints: np.ndarray) -> None:
        """Update grid from a single scan. endpoints: (N, 2) in world frame."""
        rx, ry = self.world_to_grid(robot_pos[0], robot_pos[1])
        for ep in endpoints:
            ex, ey = self.world_to_grid(ep[0], ep[1])
            cells = _bresenham(rx, ry, ex, ey)
            for cx, cy in cells[:-1]:
                if 0 <= cx < self.size[0] and 0 <= cy < self.size[1]:
                    self.log_odds[cy, cx] += self.l_free
            if 0 <= ex < self.size[0] and 0 <= ey < self.size[1]:
                self.log_odds[ey, ex] += self.l_occ

    @property
    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    @staticmethod
    def generate_synthetic_scans(
        n_scans: int = 20,
        n_beams: int = 360,
        max_range: float = 4.0,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """Generate synthetic 2D scans in a box room with an obstacle."""
        if rng is None:
            rng = np.random.default_rng()

        walls = [
            (np.array([-4, -4]), np.array([4, -4])),
            (np.array([4, -4]), np.array([4, 4])),
            (np.array([4, 4]), np.array([-4, 4])),
            (np.array([-4, 4]), np.array([-4, -4])),
            (np.array([0, -1]), np.array([0, 1])),
            (np.array([-1, 2]), np.array([1, 2])),
        ]

        scans = []
        for i in range(n_scans):
            angle = 2 * np.pi * i / n_scans
            pos = np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle)])
            heading = angle + np.pi
            angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False) + heading
            endpoints = []
            for a in angles:
                d = np.array([np.cos(a), np.sin(a)])
                best_t = max_range
                for w_start, w_end in walls:
                    t = _ray_segment_intersect(pos, d, w_start, w_end)
                    if t is not None and 0 < t < best_t:
                        best_t = t
                best_t += rng.normal(0, 0.02)
                best_t = np.clip(best_t, 0, max_range)
                endpoints.append(pos + best_t * d)
            scans.append({"pos": pos, "heading": heading, "endpoints": np.array(endpoints)})
        return scans

    @staticmethod
    def synthetic_icp_clouds(
        n_points: int = 200,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate two 2D point clouds related by a known rigid transform + noise."""
        if rng is None:
            rng = np.random.default_rng()
        theta_true = 0.15
        t_true = np.array([0.3, -0.2])
        R_true = np.array([[np.cos(theta_true), -np.sin(theta_true)],
                           [np.sin(theta_true), np.cos(theta_true)]])

        source = rng.uniform(-2, 2, (n_points, 2))
        target = (R_true @ source.T).T + t_true + rng.normal(0, 0.02, (n_points, 2))
        return source, target


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """2D ICP point-to-point. Returns (R, t, errors_per_iter)."""
    src = source.copy()
    errors: list[float] = []

    R_total = np.eye(2)
    t_total = np.zeros(2)

    for _ in range(max_iter):
        # closest-point correspondences
        dists = np.linalg.norm(src[:, None, :] - target[None, :, :], axis=2)
        indices = np.argmin(dists, axis=1)
        matched = target[indices]

        err = np.sqrt(np.mean(np.sum((src - matched) ** 2, axis=1)))
        errors.append(err)

        # centroids
        src_mean = src.mean(axis=0)
        tgt_mean = matched.mean(axis=0)
        src_c = src - src_mean
        tgt_c = matched - tgt_mean

        W = src_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = tgt_mean - R @ src_mean

        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        if len(errors) > 1 and abs(errors[-1] - errors[-2]) < tol:
            break

    return R_total, t_total, errors


def icp_point_to_plane(
    source: np.ndarray,
    target: np.ndarray,
    normals: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """2D ICP point-to-plane (linearised). Returns (R, t, errors)."""
    src = source.copy()
    errors: list[float] = []
    R_total = np.eye(2)
    t_total = np.zeros(2)

    for _ in range(max_iter):
        dists = np.linalg.norm(src[:, None, :] - target[None, :, :], axis=2)
        indices = np.argmin(dists, axis=1)
        matched = target[indices]
        matched_n = normals[indices]

        diff = src - matched
        proj = np.sum(diff * matched_n, axis=1)
        err = np.sqrt(np.mean(proj**2))
        errors.append(err)

        # linearised: [n_x, n_y, (s x n)] @ [tx, ty, dtheta]^T = -proj
        A = np.column_stack([
            matched_n[:, 0],
            matched_n[:, 1],
            src[:, 0] * matched_n[:, 1] - src[:, 1] * matched_n[:, 0],
        ])
        b = -proj
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        tx, ty, dtheta = x
        R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                       [np.sin(dtheta), np.cos(dtheta)]])
        t = np.array([tx, ty])
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        if len(errors) > 1 and abs(errors[-1] - errors[-2]) < tol:
            break

    return R_total, t_total, errors


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham's line algorithm."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


def _ray_segment_intersect(
    origin: np.ndarray, direction: np.ndarray,
    seg_start: np.ndarray, seg_end: np.ndarray,
) -> float | None:
    """Intersect ray (origin + t*direction) with segment [seg_start, seg_end]."""
    v1 = origin - seg_start
    v2 = seg_end - seg_start
    v3 = np.array([-direction[1], direction[0]])
    denom = np.dot(v2, v3)
    if abs(denom) < 1e-10:
        return None
    t1 = np.cross(v2, v1) / denom
    t2 = np.dot(v1, v3) / denom
    if t1 >= 0 and 0 <= t2 <= 1:
        return t1
    return None


def _truncated_gaussian(
    z: np.ndarray,
    mean: float,
    sigma: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    base = np.exp(-0.5 * ((z - mean) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    cdf_hi = 0.5 * (1.0 + erf((hi - mean) / (np.sqrt(2.0) * sigma)))
    cdf_lo = 0.5 * (1.0 + erf((lo - mean) / (np.sqrt(2.0) * sigma)))
    norm = max(float(cdf_hi - cdf_lo), 1e-12)
    out = np.where((z >= lo) & (z <= hi), base / norm, 0.0)
    return out


def _discrete_mass_as_density(z: np.ndarray, location: float) -> np.ndarray:
    if len(z) < 2:
        return np.zeros_like(z)
    width = float(np.median(np.diff(z)))
    out = np.zeros_like(z)
    out[np.abs(z - location) <= width / 2] = 1.0 / width
    return out
