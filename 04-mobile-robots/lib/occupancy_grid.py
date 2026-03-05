from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


def bresenham_ray(
    x0: int, y0: int, x1: int, y1: int
) -> list[tuple[int, int]]:
    """Bresenham's line algorithm: integer grid cells from (x0,y0) to (x1,y1)."""
    cells = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


@dataclass
class OccupancyGrid:
    """2D occupancy grid with Bayesian log-odds update.

    Log-odds: l = log( P(occ) / P(free) )
    Probability of occupancy: P = 1 - 1 / (1 + exp(l))
    """
    width: float = 20.0       # metres
    height: float = 20.0      # metres
    resolution: float = 0.05  # metres per cell
    origin: np.ndarray = field(default_factory=lambda: np.array([-10.0, -10.0]))

    # Log-odds hyperparameters
    l_occ: float = 0.85   # inv sensor model: occupied beam endpoint
    l_free: float = -0.4  # inv sensor model: free along ray
    l_0: float = 0.0      # prior (unknown)
    l_min: float = -5.0
    l_max: float = 5.0

    _log_odds: np.ndarray = field(init=False)

    def __post_init__(self):
        nx = int(np.ceil(self.width / self.resolution))
        ny = int(np.ceil(self.height / self.resolution))
        self._log_odds = np.zeros((ny, nx), dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self._log_odds.shape

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        ix = int((x - self.origin[0]) / self.resolution)
        iy = int((y - self.origin[1]) / self.resolution)
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        x = self.origin[0] + (ix + 0.5) * self.resolution
        y = self.origin[1] + (iy + 0.5) * self.resolution
        return x, y

    def _in_bounds(self, ix: int, iy: int) -> bool:
        h, w = self._log_odds.shape
        return 0 <= ix < w and 0 <= iy < h

    def update_ray(
        self,
        robot_pose: np.ndarray,
        z: float,
        angle: float,
        z_max: float = 10.0,
    ) -> None:
        """Update log-odds along one LiDAR ray.

        robot_pose: [x, y, theta]
        z:          measured range (metres)
        angle:      beam angle in world frame
        """
        rx, ry, _ = robot_pose
        hit = z < z_max * 0.99   # True if beam returned before max range

        # End-point in world frame
        ex = rx + z * np.cos(angle)
        ey = ry + z * np.sin(angle)

        sx, sy = self.world_to_grid(rx, ry)
        ex_g, ey_g = self.world_to_grid(ex, ey)

        cells = bresenham_ray(sx, sy, ex_g, ey_g)

        # Free cells along ray (all except last if hit)
        free_cells = cells[:-1] if hit else cells
        for ix, iy in free_cells:
            if self._in_bounds(ix, iy):
                self._log_odds[iy, ix] = np.clip(
                    self._log_odds[iy, ix] + self.l_free - self.l_0,
                    self.l_min, self.l_max,
                )

        # Occupied endpoint
        if hit and self._in_bounds(ex_g, ey_g):
            self._log_odds[ey_g, ex_g] = np.clip(
                self._log_odds[ey_g, ex_g] + self.l_occ - self.l_0,
                self.l_min, self.l_max,
            )

    def update_scan(
        self,
        robot_pose: np.ndarray,
        ranges: np.ndarray,
        angles: np.ndarray,
        z_max: float = 10.0,
    ) -> None:
        abs_angles = robot_pose[2] + angles
        for r, a in zip(ranges, abs_angles):
            self.update_ray(robot_pose, r, a, z_max)

    def probability(self) -> np.ndarray:
        """Return P(occupied) for each cell."""
        return 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))

    def to_image(self) -> np.ndarray:
        """Return uint8 image: 128=unknown, 0=free, 255=occupied."""
        p = self.probability()
        img = np.full_like(p, 128, dtype=np.uint8)
        img[p > 0.65] = 255
        img[p < 0.35] = 0
        return img

    def is_occupied(self, x: float, y: float, threshold: float = 0.65) -> bool:
        ix, iy = self.world_to_grid(x, y)
        if not self._in_bounds(ix, iy):
            return True
        return self.probability()[iy, ix] > threshold

    def is_known(self, ix: int, iy: int, threshold: float = 0.1) -> bool:
        p = self.probability()[iy, ix]
        return abs(p - 0.5) > threshold
