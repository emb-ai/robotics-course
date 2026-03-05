from __future__ import annotations
import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

@dataclass
class AStar:
    """A*/Dijkstra path planner on a 2D occupancy grid.

    Set use_heuristic=False to get Dijkstra (uniform-cost search).
    """
    occupancy: np.ndarray
    resolution: float = 0.05
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    allow_diagonal: bool = True
    use_heuristic: bool = True

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        ix = int((x - self.origin[0]) / self.resolution)
        iy = int((y - self.origin[1]) / self.resolution)
        return iy, ix

    def _grid_to_world(self, iy: int, ix: int) -> np.ndarray:
        x = self.origin[0] + (ix + 0.5) * self.resolution
        y = self.origin[1] + (iy + 0.5) * self.resolution
        return np.array([x, y])

    def _heuristic(self, a: tuple, b: tuple) -> float:
        if not self.use_heuristic:
            return 0.0
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def _neighbors(self, node: tuple) -> list[tuple[tuple, float]]:
        r, c = node
        H, W = self.occupancy.shape
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.allow_diagonal:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        nbrs = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not self.occupancy[nr, nc]:
                cost = np.hypot(dr, dc)
                nbrs.append(((nr, nc), cost))
        return nbrs

    def plan(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        record_explored: bool = False,
    ) -> Optional[list[np.ndarray]] | tuple[Optional[list[np.ndarray]], list[tuple]]:
        """Return world-frame waypoints (or None).

        If record_explored=True, also return list of explored grid cells
        (useful for visualising search order).
        """
        start = self._world_to_grid(*start_xy)
        goal = self._world_to_grid(*goal_xy)

        H, W = self.occupancy.shape
        if (not (0 <= start[0] < H and 0 <= start[1] < W) or
                not (0 <= goal[0] < H and 0 <= goal[1] < W)):
            return (None, []) if record_explored else None

        open_set: list[tuple[float, tuple]] = []
        heapq.heappush(open_set, (0.0, start))
        came_from: dict = {}
        g: dict = {start: 0.0}
        explored: list[tuple] = []

        while open_set:
            _, current = heapq.heappop(open_set)
            explored.append(current)
            if current == goal:
                path = []
                node = current
                while node in came_from:
                    path.append(self._grid_to_world(*node))
                    node = came_from[node]
                path.append(self._grid_to_world(*start))
                path = list(reversed(path))
                return (path, explored) if record_explored else path

            for nbr, cost in self._neighbors(current):
                tentative_g = g[current] + cost
                if tentative_g < g.get(nbr, np.inf):
                    came_from[nbr] = current
                    g[nbr] = tentative_g
                    f = tentative_g + self._heuristic(nbr, goal)
                    heapq.heappush(open_set, (f, nbr))

        return (None, explored) if record_explored else None


# ---------------------------------------------------------------------------
# RRT
# ---------------------------------------------------------------------------

@dataclass
class RRT:
    """Kinematically-constrained RRT for Ackermann or diff-drive robot.

    For simplicity, samples random states and steers toward them.
    Works in 3-DOF pose space [x, y, theta].
    """
    kinematic_model: object      # AckermannModel or DiffDrive
    occupancy: np.ndarray        # H x W bool
    resolution: float = 0.05
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    dt: float = 0.1
    n_steps: int = 5             # steps per steer
    max_iter: int = 5000
    goal_tol: float = 0.3
    goal_bias: float = 0.1
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    _tree: list = field(init=False, default_factory=list)

    def _sample_state(self, goal: np.ndarray) -> np.ndarray:
        H, W = self.occupancy.shape
        if self.rng.random() < self.goal_bias:
            return goal
        x = self.rng.uniform(self.origin[0], self.origin[0] + W * self.resolution)
        y = self.rng.uniform(self.origin[1], self.origin[1] + H * self.resolution)
        th = self.rng.uniform(-np.pi, np.pi)
        return np.array([x, y, th])

    def _steer(self, from_state: np.ndarray, to_state: np.ndarray) -> np.ndarray:
        """Return state after steering toward to_state for n_steps."""
        dx, dy = to_state[0] - from_state[0], to_state[1] - from_state[1]
        target_heading = np.arctan2(dy, dx)
        diff = (target_heading - from_state[2] + np.pi) % (2 * np.pi) - np.pi

        state = from_state.copy()
        for _ in range(self.n_steps):
            if hasattr(self.kinematic_model, 'max_steer'):
                # Ackermann
                delta = np.clip(diff, -self.kinematic_model.max_steer, self.kinematic_model.max_steer)
                state = self.kinematic_model.forward(state, v=1.0, delta=delta, dt=self.dt)
            else:
                # DiffDrive
                omega = np.clip(diff / self.dt, -2.0, 2.0)
                state = self.kinematic_model.forward(state, v=1.0, omega=omega, dt=self.dt)
            diff = (target_heading - state[2] + np.pi) % (2 * np.pi) - np.pi
        return state

    def _collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if straight-line segment p1->p2 is collision-free."""
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        n = max(2, int(dist / self.resolution))
        H, W = self.occupancy.shape
        for t in np.linspace(0, 1, n):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            ix = int((x - self.origin[0]) / self.resolution)
            iy = int((y - self.origin[1]) / self.resolution)
            if 0 <= iy < H and 0 <= ix < W and self.occupancy[iy, ix]:
                return False
        return True

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[list[np.ndarray]]:
        self._tree = [(start, None)]
        for _ in range(self.max_iter):
            x_rand = self._sample_state(goal)
            # nearest node
            dists = [np.hypot(n[0][0] - x_rand[0], n[0][1] - x_rand[1]) for n in self._tree]
            nearest_idx = int(np.argmin(dists))
            x_near = self._tree[nearest_idx][0]
            x_new = self._steer(x_near, x_rand)

            if self._collision_free(x_near, x_new):
                self._tree.append((x_new, nearest_idx))
                if np.hypot(x_new[0] - goal[0], x_new[1] - goal[1]) < self.goal_tol:
                    # Reconstruct path
                    path = []
                    idx = len(self._tree) - 1
                    while idx is not None:
                        path.append(self._tree[idx][0])
                        idx = self._tree[idx][1]
                    return list(reversed(path))
        return None


# ---------------------------------------------------------------------------
# Potential Field
# ---------------------------------------------------------------------------

@dataclass
class PotentialField:
    """Artificial potential field local planner.

    F_total = F_attractive(goal) + sum(F_repulsive(obstacles))
    """
    k_att: float = 1.0
    k_rep: float = 2.0
    d0: float = 1.5       # influence radius of obstacles (metres)
    step_size: float = 0.1
    max_steps: int = 1000
    goal_tol: float = 0.2

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_positions: np.ndarray,  # (M, 2)
    ) -> list[np.ndarray]:
        pos = start[:2].copy()
        path = [pos.copy()]
        for _ in range(self.max_steps):
            f_att = self.k_att * (goal[:2] - pos)
            f_rep = np.zeros(2)
            for obs in obstacle_positions:
                d = np.hypot(pos[0] - obs[0], pos[1] - obs[1])
                if d < self.d0 and d > 1e-3:
                    f_rep += self.k_rep * (1.0/d - 1.0/self.d0) / d**2 * (pos - obs) / d
            f = f_att + f_rep
            norm_f = np.linalg.norm(f)
            if norm_f > 1e-9:
                pos = pos + self.step_size * f / norm_f
            path.append(pos.copy())
            if np.hypot(pos[0] - goal[0], pos[1] - goal[1]) < self.goal_tol:
                break
        return [np.array([p[0], p[1], 0.0]) for p in path]


# ---------------------------------------------------------------------------
# Frontier Explorer
# ---------------------------------------------------------------------------

@dataclass
class FrontierExplorer:
    """Frontier-based exploration: select nearest frontier cell as goal.

    A frontier cell is a free cell adjacent to an unknown cell.
    """
    occupancy_grid: object  # OccupancyGrid instance

    def find_frontiers(self) -> list[np.ndarray]:
        """Return list of world-frame frontier cell centres."""
        grid = self.occupancy_grid
        prob = grid.probability()
        H, W = prob.shape
        frontiers = []
        for iy in range(1, H - 1):
            for ix in range(1, W - 1):
                # free cell
                if prob[iy, ix] >= 0.35:
                    continue
                # adjacent to unknown
                neighbors = prob[iy-1:iy+2, ix-1:ix+2].ravel()
                if np.any(np.abs(neighbors - 0.5) < 0.1):
                    wx, wy = grid.grid_to_world(ix, iy)
                    frontiers.append(np.array([wx, wy]))
        return frontiers

    def next_goal(self, robot_pos: np.ndarray) -> Optional[np.ndarray]:
        frontiers = self.find_frontiers()
        if not frontiers:
            return None
        dists = [np.hypot(f[0] - robot_pos[0], f[1] - robot_pos[1]) for f in frontiers]
        return frontiers[int(np.argmin(dists))]


# ---------------------------------------------------------------------------
# Boustrophedon Coverage
# ---------------------------------------------------------------------------

@dataclass
class BoustrophedonCoverage:
    """Simple boustrophedon (back-and-forth) coverage path.

    Generates a lawnmower path over the known free space.
    """
    occupancy_grid: object  # OccupancyGrid instance
    stripe_width: float = 0.3    # metres between sweeping rows

    def plan(self) -> list[np.ndarray]:
        """Return ordered waypoints covering all free cells."""
        grid = self.occupancy_grid
        prob = grid.probability()
        H, W = prob.shape
        res = grid.resolution
        n_rows = max(1, int(self.stripe_width / res))
        path = []
        left_to_right = True
        for iy in range(0, H, n_rows):
            row_cells = []
            for ix in range(W):
                if prob[iy, ix] < 0.35:   # free cell
                    wx, wy = grid.grid_to_world(ix, iy)
                    row_cells.append(np.array([wx, wy, 0.0]))
            if row_cells:
                if not left_to_right:
                    row_cells = list(reversed(row_cells))
                path.extend(row_cells)
                left_to_right = not left_to_right
        return path
