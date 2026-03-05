from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class MCL:
    """Monte Carlo Localization (particle filter) for a 2D robot.

    Uses a motion model for prediction and a likelihood-field observation
    model for weighting.
    """
    motion_model: object        # OdometryMotionModel or VelocityMotionModel
    obs_model: object           # LikelihoodFieldModel
    n_particles: int = 500
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    particles: np.ndarray = field(init=False)   # (N, 3)
    weights: np.ndarray = field(init=False)     # (N,)

    def __post_init__(self):
        self.particles = np.zeros((self.n_particles, 3))
        self.weights = np.ones(self.n_particles) / self.n_particles

    def init_uniform(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        n = self.n_particles
        self.particles[:, 0] = self.rng.uniform(*x_range, n)
        self.particles[:, 1] = self.rng.uniform(*y_range, n)
        self.particles[:, 2] = self.rng.uniform(-np.pi, np.pi, n)
        self.weights = np.ones(n) / n

    def init_gaussian(self, mu: np.ndarray, Sigma: np.ndarray) -> None:
        self.particles = self.rng.multivariate_normal(mu, Sigma, self.n_particles)
        self.particles[:, 2] = _wrap_angle(self.particles[:, 2])
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _predict_velocity(self, v: float, omega: float, dt: float) -> None:
        for i in range(self.n_particles):
            self.particles[i] = self.motion_model.sample(
                self.particles[i], v, omega, dt, n=1, rng=self.rng
            )

    def _predict_odometry(
        self,
        delta_rot1: float,
        delta_trans: float,
        delta_rot2: float,
    ) -> None:
        for i in range(self.n_particles):
            self.particles[i] = self.motion_model.sample(
                self.particles[i], delta_rot1, delta_trans, delta_rot2,
                n=1, rng=self.rng,
            )

    def predict(self, *args, motion_type: str = "velocity", **kwargs) -> None:
        if motion_type == "velocity":
            self._predict_velocity(*args, **kwargs)
        else:
            self._predict_odometry(*args, **kwargs)

    def update_weights(self, ranges: np.ndarray, angles: np.ndarray) -> None:
        """Weight particles by likelihood field log-prob, then normalize."""
        log_ws = np.zeros(self.n_particles)
        for i, p in enumerate(self.particles):
            endpoints = self.obs_model.endpoints_from_pose(p, ranges, angles)
            log_ws[i] = self.obs_model.scan_log_prob(endpoints)
        # numerical stability
        log_ws -= log_ws.max()
        self.weights = np.exp(log_ws)
        self.weights /= self.weights.sum()

    def resample(self) -> None:
        """Low-variance (systematic) resampling."""
        n = self.n_particles
        positions = (self.rng.uniform() + np.arange(n)) / n
        cumsum = np.cumsum(self.weights)
        idx = np.searchsorted(cumsum, positions)
        idx = np.clip(idx, 0, n - 1)
        self.particles = self.particles[idx]
        self.weights = np.ones(n) / n

    def effective_sample_size(self) -> float:
        return 1.0 / np.sum(self.weights**2)

    def mean_pose(self) -> np.ndarray:
        x = np.sum(self.weights * self.particles[:, 0])
        y = np.sum(self.weights * self.particles[:, 1])
        sin_th = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_th = np.sum(self.weights * np.cos(self.particles[:, 2]))
        return np.array([x, y, np.arctan2(sin_th, cos_th)])


@dataclass
class KLDAdaptiveMCL(MCL):
    """MCL with KLD-sampling for adaptive particle count.

    Dynamically adjusts N based on the number of occupied histogram bins
    in (x, y, theta) space, targeting a KL-divergence bound.

    Parameters
    ----------
    bin_size_xy : float
        Cell size for position binning (metres).
    bin_size_th : float
        Cell size for heading binning (radians).
    epsilon : float
        KLD error bound.
    delta : float
        Confidence level (1 - failure probability).
    n_min : int
        Minimum number of particles.
    n_max : int
        Maximum number of particles (safety cap).
    """
    bin_size_xy: float = 0.5
    bin_size_th: float = np.deg2rad(10.0)
    epsilon: float = 0.05
    delta: float = 0.01
    n_min: int = 50
    n_max: int = 5000

    def _kld_n_required(self, k: int) -> int:
        """Compute required N given k occupied bins (Wilson-Hilferty approximation)."""
        if k <= 1:
            return self.n_min
        z = -0.5 * np.log(self.delta)   # approx chi2 quantile / 2
        # n = (k-1) / (2 * epsilon) * (1 - 2/(9*(k-1)) + sqrt(2/(9*(k-1))) * z)^3
        c = 1.0 - 2.0 / (9 * (k - 1)) + np.sqrt(2.0 / (9 * (k - 1))) * z
        n = max(self.n_min, int(np.ceil((k - 1) / (2 * self.epsilon) * c**3)))
        return min(n, self.n_max)

    def _bin_key(self, p: np.ndarray) -> tuple:
        ix = int(np.floor(p[0] / self.bin_size_xy))
        iy = int(np.floor(p[1] / self.bin_size_xy))
        it = int(np.floor((p[2] + np.pi) / self.bin_size_th))
        return (ix, iy, it)

    def resample_adaptive(
        self, ranges: np.ndarray, angles: np.ndarray,
        motion_type: str = "velocity", motion_args: tuple = (),
    ) -> int:
        """Run one predict-update-resample cycle with adaptive N.

        Returns the number of particles actually used.
        """
        new_particles = []
        occupied_bins: set = set()

        n_needed = self.n_min
        i = 0
        while len(new_particles) < n_needed and len(new_particles) < self.n_max:
            # Draw from posterior approximation (resample from current set)
            idx = self.rng.integers(0, len(self.particles))
            candidate = self.particles[idx].copy()

            # propagate through motion model
            if motion_type == "velocity":
                candidate = self.motion_model.sample(candidate, *motion_args, n=1, rng=self.rng)
            else:
                candidate = self.motion_model.sample(candidate, *motion_args, n=1, rng=self.rng)

            new_particles.append(candidate)
            key = self._bin_key(candidate)
            occupied_bins.add(key)
            k = len(occupied_bins)
            n_needed = self._kld_n_required(k)
            i += 1

        self.particles = np.array(new_particles)
        self.n_particles = len(self.particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

        # Now weight by observation
        self.update_weights(ranges, angles)
        # Resample to exactly n_needed
        idx = self.rng.choice(self.n_particles, size=self.n_particles,
                               replace=True, p=self.weights)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.n_particles) / self.n_particles
        return self.n_particles
