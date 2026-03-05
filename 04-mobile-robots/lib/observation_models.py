from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import distance_transform_edt


@dataclass
class BeamModel:
    """4-component mixture LiDAR beam model (Thrun et al. Ch. 6.3).

    Components:
        hit   : Gaussian around expected range
        short : Exponential (unexpected near obstacles)
        max   : Point mass at z_max (sensor failure / no return)
        rand  : Uniform over [0, z_max] (phantom / crosstalk)
    """
    z_max: float = 10.0
    sigma_hit: float = 0.2
    lambda_short: float = 1.0
    w_hit: float = 0.7
    w_short: float = 0.1
    w_max: float = 0.1
    w_rand: float = 0.1
    n_rays: int = 180

    def __post_init__(self):
        total = self.w_hit + self.w_short + self.w_max + self.w_rand
        self.w_hit   /= total
        self.w_short /= total
        self.w_max   /= total
        self.w_rand  /= total

    def beam_log_prob(self, z: float, z_expected: float) -> float:
        """Log probability of a single beam measurement z given z_expected."""
        z = np.clip(z, 0, self.z_max)

        # p_hit: truncated Gaussian
        if 0 <= z <= self.z_max:
            from math import erf
            eta = 1.0 / (
                0.5 * (1 + erf((self.z_max - z_expected) / (self.sigma_hit * np.sqrt(2))))
                - 0.5 * (1 + erf((0 - z_expected) / (self.sigma_hit * np.sqrt(2))))
            )
            p_hit = eta * np.exp(-0.5 * (z - z_expected)**2 / self.sigma_hit**2)
        else:
            p_hit = 0.0

        # p_short: truncated exponential
        if 0 <= z <= z_expected:
            eta_s = 1.0 / (1 - np.exp(-self.lambda_short * z_expected))
            p_short = eta_s * self.lambda_short * np.exp(-self.lambda_short * z)
        else:
            p_short = 0.0

        # p_max: point mass
        p_max = 1.0 if abs(z - self.z_max) < 1e-3 else 0.0

        # p_rand: uniform
        p_rand = 1.0 / self.z_max

        p = (self.w_hit * p_hit + self.w_short * p_short +
             self.w_max * p_max + self.w_rand * p_rand)
        return np.log(p + 1e-300)

    def scan_log_prob(self, z: np.ndarray, z_expected: np.ndarray) -> float:
        """Log probability of full scan (sum over independent beams)."""
        return sum(self.beam_log_prob(zi, ze) for zi, ze in zip(z, z_expected))

    def sample_beam(
        self,
        z_expected: float,
        rng: np.random.Generator | None = None,
    ) -> float:
        rng = rng or np.random.default_rng()
        component = rng.choice(4, p=[self.w_hit, self.w_short, self.w_max, self.w_rand])
        if component == 0:
            z = rng.normal(z_expected, self.sigma_hit)
        elif component == 1:
            z = rng.exponential(1.0 / self.lambda_short)
            while z > z_expected:
                z = rng.exponential(1.0 / self.lambda_short)
        elif component == 2:
            z = self.z_max
        else:
            z = rng.uniform(0, self.z_max)
        return float(np.clip(z, 0, self.z_max))


@dataclass
class LikelihoodFieldModel:
    """Likelihood-field (end-point) observation model.

    Precomputes a distance transform from the occupancy grid and uses
    it to evaluate p(z | x_t, m) efficiently.
    """
    sigma_hit: float = 0.2
    w_hit: float = 0.9
    w_rand: float = 0.1
    z_max: float = 10.0
    _distance_field: np.ndarray | None = field(default=None, repr=False)
    _resolution: float = 0.05
    _origin: np.ndarray = field(default_factory=lambda: np.zeros(2), repr=False)

    def build_from_grid(
        self,
        occupancy: np.ndarray,
        resolution: float = 0.05,
        origin: np.ndarray | None = None,
    ) -> None:
        """Build distance field from binary occupancy grid (True = occupied)."""
        self._resolution = resolution
        self._origin = origin if origin is not None else np.zeros(2)
        # distance_transform_edt: distance from each free cell to nearest obstacle
        free_mask = ~occupancy.astype(bool)
        dist_pixels = distance_transform_edt(free_mask)
        self._distance_field = dist_pixels * resolution

    @property
    def distance_field(self) -> np.ndarray:
        if self._distance_field is None:
            raise RuntimeError("Call build_from_grid() first.")
        return self._distance_field

    def _world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        idx = (xy - self._origin) / self._resolution
        return np.round(idx).astype(int)

    def _distance_at_point(self, x: float, y: float) -> float:
        h, w = self._distance_field.shape
        ix, iy = self._world_to_grid(np.array([x, y]))
        if 0 <= iy < h and 0 <= ix < w:
            return float(self._distance_field[iy, ix])
        return self.z_max

    def beam_log_prob(self, endpoint_world: np.ndarray) -> float:
        """Log p for one beam endpoint given in world coordinates."""
        d = self._distance_at_point(*endpoint_world)
        p_hit = np.exp(-0.5 * d**2 / self.sigma_hit**2)
        p_rand = 1.0 / self.z_max
        p = self.w_hit * p_hit + self.w_rand * p_rand
        return np.log(p + 1e-300)

    def scan_log_prob(self, endpoints_world: np.ndarray) -> float:
        """Sum log-probs over all beam endpoints (shape Nx2)."""
        return sum(self.beam_log_prob(ep) for ep in endpoints_world)

    def endpoints_from_pose(
        self,
        pose: np.ndarray,
        ranges: np.ndarray,
        angles: np.ndarray,
    ) -> np.ndarray:
        """Convert polar scan (ranges, angles) at robot pose to Cartesian endpoints."""
        x, y, th = pose
        abs_angles = th + angles
        ex = x + ranges * np.cos(abs_angles)
        ey = y + ranges * np.sin(abs_angles)
        return np.stack([ex, ey], axis=1)


@dataclass
class RangeBearingModel:
    """Range-bearing landmark observation model for EKF localization.

    Measurement: z = [range, bearing] to a point landmark.
    """
    sigma_r: float = 0.1
    sigma_phi: float = np.deg2rad(2.0)

    @property
    def R(self) -> np.ndarray:
        return np.diag([self.sigma_r**2, self.sigma_phi**2])

    def expected_measurement(
        self, pose: np.ndarray, landmark: np.ndarray
    ) -> np.ndarray:
        """h(x_t): [range, bearing] from robot pose to landmark [lx, ly]."""
        dx = landmark[0] - pose[0]
        dy = landmark[1] - pose[1]
        r = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx) - pose[2]
        return np.array([r, (phi + np.pi) % (2 * np.pi) - np.pi])

    def jacobian(self, pose: np.ndarray, landmark: np.ndarray) -> np.ndarray:
        """H_t = dh/dx  (2x3)."""
        dx = landmark[0] - pose[0]
        dy = landmark[1] - pose[1]
        q = dx**2 + dy**2
        r = np.sqrt(q)
        return np.array([
            [-dx / r,   -dy / r,   0],
            [ dy / q,  -dx / q,  -1],
        ])

    def sample(
        self,
        pose: np.ndarray,
        landmark: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        z_hat = self.expected_measurement(pose, landmark)
        noise = rng.multivariate_normal(np.zeros(2), self.R)
        z = z_hat + noise
        z[1] = (z[1] + np.pi) % (2 * np.pi) - np.pi
        return z

    def log_prob(self, z: np.ndarray, pose: np.ndarray, landmark: np.ndarray) -> float:
        z_hat = self.expected_measurement(pose, landmark)
        innov = z - z_hat
        innov[1] = (innov[1] + np.pi) % (2 * np.pi) - np.pi
        R_inv = np.diag([1 / self.sigma_r**2, 1 / self.sigma_phi**2])
        return -0.5 * innov @ R_inv @ innov - np.log(
            2 * np.pi * self.sigma_r * self.sigma_phi
        )
