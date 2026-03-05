from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


def _make_city_map(
    width: float = 60.0,
    height: float = 50.0,
    resolution: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple city-block map.

    Returns (occupancy bool H x W, origin [x0, y0]).
    """
    rng = rng or np.random.default_rng(42)
    W = int(width / resolution)
    H = int(height / resolution)
    occ = np.zeros((H, W), dtype=bool)

    # Border walls
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True

    # City blocks: horizontal and vertical roads with buildings between them
    road_spacing = int(10.0 / resolution)
    road_width   = int(2.0  / resolution)
    for r in range(0, H, road_spacing):
        occ[max(0, r - road_width // 2): r + road_width // 2, :] = False
    for c in range(0, W, road_spacing):
        occ[:, max(0, c - road_width // 2): c + road_width // 2] = False

    # Buildings in blocks
    block_size = road_spacing - road_width
    for r0 in range(road_width, H - road_width, road_spacing):
        for c0 in range(road_width, W - road_width, road_spacing):
            r1 = min(H - road_width, r0 + block_size)
            c1 = min(W - road_width, c0 + block_size)
            inset = max(1, int(0.5 / resolution))
            occ[r0 + inset: r1 - inset, c0 + inset: c1 - inset] = True

    origin = np.array([0.0, 0.0])
    return occ, origin


def _make_apartment_map(
    width: float = 12.0,
    height: float = 10.0,
    resolution: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple apartment floor plan."""
    W = int(width / resolution)
    H = int(height / resolution)
    occ = np.zeros((H, W), dtype=bool)

    # Border walls
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True

    # Interior walls (rooms)
    mid_col = W // 2
    occ[:, mid_col] = True
    occ[mid_col: mid_col + 2, mid_col] = False   # doorway

    mid_row = H // 2
    occ[mid_row, :mid_col] = True
    occ[mid_row, mid_col // 3: mid_col // 3 + 2] = False   # doorway

    # Furniture (obstacles)
    for (r0, c0, r1, c1) in [
        (2, 2, 5, 4),     # couch
        (H - 6, 2, H - 3, 5),  # bed
        (3, mid_col + 3, 6, mid_col + 6),  # table
    ]:
        occ[r0:r1, c0:c1] = True

    origin = np.array([0.0, 0.0])
    return occ, origin


def _cast_ray(
    pose: np.ndarray,
    angle: float,
    occupancy: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    z_max: float = 8.0,
) -> float:
    """Cast a single ray and return range to first occupied cell."""
    x, y = pose[0], pose[1]
    for d in np.arange(0.01, z_max, resolution):
        px = x + d * np.cos(angle)
        py = y + d * np.sin(angle)
        ix = int((px - origin[0]) / resolution)
        iy = int((py - origin[1]) / resolution)
        H, W = occupancy.shape
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            return z_max
        if occupancy[iy, ix]:
            return d
    return z_max


def simulate_lidar(
    pose: np.ndarray,
    occupancy: np.ndarray,
    resolution: float,
    origin: np.ndarray,
    n_rays: int = 90,
    fov: float = 2 * np.pi,
    z_max: float = 8.0,
    sigma: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a LiDAR scan from robot pose.

    Returns (ranges, angles) both shape (n_rays,).
    """
    rng = rng or np.random.default_rng()
    angles = pose[2] + np.linspace(-fov / 2, fov / 2, n_rays, endpoint=False)
    ranges = np.array([
        _cast_ray(pose, a, occupancy, resolution, origin, z_max)
        for a in angles
    ])
    ranges += rng.normal(0, sigma, n_rays)
    ranges = np.clip(ranges, 0, z_max)
    return ranges, angles - pose[2]


@dataclass
class Simulator:
    """2D top-down robot simulator."""
    occupancy: np.ndarray       # H x W bool
    resolution: float
    origin: np.ndarray
    kinematic_model: object
    dt: float = 0.1
    lidar_rays: int = 90
    lidar_fov: float = 2 * np.pi
    lidar_z_max: float = 8.0
    lidar_sigma: float = 0.02
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    pose: np.ndarray = field(init=False)

    def __post_init__(self):
        self.pose = np.zeros(3)

    def reset(self, pose: np.ndarray) -> None:
        self.pose = pose.copy()

    def _is_collision(self, pose: np.ndarray) -> bool:
        ix = int((pose[0] - self.origin[0]) / self.resolution)
        iy = int((pose[1] - self.origin[1]) / self.resolution)
        H, W = self.occupancy.shape
        if not (0 <= ix < W and 0 <= iy < H):
            return True
        return bool(self.occupancy[iy, ix])

    def step_velocity(self, v: float, omega: float) -> np.ndarray:
        new_pose = self.kinematic_model.forward(self.pose, v, omega, self.dt)
        if not self._is_collision(new_pose):
            self.pose = new_pose
        return self.pose.copy()

    def step_ackermann(self, v: float, delta: float) -> np.ndarray:
        new_pose = self.kinematic_model.forward(self.pose, v, delta, self.dt)
        if not self._is_collision(new_pose):
            self.pose = new_pose
        return self.pose.copy()

    def get_lidar_scan(self) -> tuple[np.ndarray, np.ndarray]:
        return simulate_lidar(
            self.pose, self.occupancy, self.resolution, self.origin,
            self.lidar_rays, self.lidar_fov, self.lidar_z_max, self.lidar_sigma,
            self.rng,
        )


@dataclass
class BusScenario:
    """Autonomous campus bus scenario.

    Ackermann model, known city map, EKF localization, A* route planning,
    pure pursuit control.
    """
    resolution: float = 0.2
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def build(self) -> dict:
        from .kinematic_models import AckermannModel
        from .motion_models import VelocityMotionModel
        from .kalman_filters import EKF
        from .observation_models import LikelihoodFieldModel, RangeBearingModel
        from .planners import AStar
        from .controllers import PurePursuit
        from .occupancy_grid import OccupancyGrid

        occ, origin = _make_city_map(resolution=self.resolution, rng=self.rng)
        kin = AckermannModel(wheelbase=2.7)
        sim = Simulator(occ, self.resolution, origin, kin, dt=0.1, rng=self.rng)

        # Landmarks: bus stop positions (world frame)
        H, W = occ.shape
        stops = [
            np.array([5.0,  5.0]),
            np.array([25.0, 5.0]),
            np.array([45.0, 5.0]),
            np.array([45.0, 25.0]),
            np.array([25.0, 25.0]),
            np.array([5.0,  25.0]),
        ]

        rb_model = RangeBearingModel(sigma_r=0.3, sigma_phi=np.deg2rad(3))
        motion_model = VelocityMotionModel(alpha_1=0.02, alpha_2=0.01, alpha_3=0.01, alpha_4=0.02)
        ekf = EKF(motion_model=kin, obs_model=rb_model)

        planner = AStar(occ, resolution=self.resolution, origin=origin)
        controller = PurePursuit(lookahead=3.0, wheelbase=2.7, min_speed=2.0)

        return {
            "sim": sim,
            "occupancy": occ,
            "resolution": self.resolution,
            "origin": origin,
            "kin": kin,
            "ekf": ekf,
            "landmarks": np.array(stops),
            "planner": planner,
            "controller": controller,
            "motion_model": motion_model,
            "rb_model": rb_model,
        }


@dataclass
class VacuumScenario:
    """Vacuum robot cleaning an unknown apartment.

    Diff-drive model, occupancy grid mapping, MCL localization,
    frontier exploration then boustrophedon coverage.
    """
    resolution: float = 0.05
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1))

    def build(self) -> dict:
        from .kinematic_models import DiffDrive
        from .motion_models import OdometryMotionModel
        from .observation_models import LikelihoodFieldModel
        from .particle_filter import MCL
        from .occupancy_grid import OccupancyGrid
        from .planners import AStar, FrontierExplorer, BoustrophedonCoverage
        from .controllers import PIDController

        occ, origin = _make_apartment_map(resolution=self.resolution)
        kin = DiffDrive(wheel_radius=0.05, wheel_base=0.30)
        sim = Simulator(occ, self.resolution, origin, kin, dt=0.1,
                        lidar_rays=180, lidar_fov=2 * np.pi, rng=self.rng)

        motion_model = OdometryMotionModel(alpha_1=0.05, alpha_2=0.01,
                                            alpha_3=0.01, alpha_4=0.05)
        H, W = occ.shape
        lf_model = LikelihoodFieldModel(sigma_hit=0.15, w_hit=0.9, z_max=8.0)

        mcl = MCL(motion_model=motion_model, obs_model=lf_model, n_particles=500, rng=self.rng)

        grid = OccupancyGrid(
            width=W * self.resolution,
            height=H * self.resolution,
            resolution=self.resolution,
            origin=origin,
        )

        planner = AStar(np.zeros_like(occ), resolution=self.resolution, origin=origin)
        explorer = FrontierExplorer(occupancy_grid=grid)
        coverage = BoustrophedonCoverage(occupancy_grid=grid, stripe_width=0.3)
        controller = PIDController(kp_lin=1.5, kp_ang=2.5, max_v=0.5, max_omega=1.5)

        return {
            "sim": sim,
            "true_occupancy": occ,
            "occupancy": occ,
            "resolution": self.resolution,
            "origin": origin,
            "kin": kin,
            "mcl": mcl,
            "lf_model": lf_model,
            "grid": grid,
            "planner": planner,
            "explorer": explorer,
            "coverage": coverage,
            "controller": controller,
            "motion_model": motion_model,
        }
