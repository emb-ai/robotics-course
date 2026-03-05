from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.stats import chi2


def _wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class KalmanFilter:
    """Standard linear Kalman filter.

    State model:   x_t = A x_{t-1} + B u_t + w,   w ~ N(0, Q)
    Observation:   z_t = H x_t + v,                v ~ N(0, R)
    """
    A: np.ndarray
    B: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    mu: np.ndarray = field(init=False)
    Sigma: np.ndarray = field(init=False)

    def __post_init__(self):
        n = self.A.shape[0]
        self.mu = np.zeros(n)
        self.Sigma = np.eye(n)

    def reset(self, mu: np.ndarray, Sigma: np.ndarray) -> None:
        self.mu = mu.copy()
        self.Sigma = Sigma.copy()

    def predict(self, u: np.ndarray) -> None:
        self.mu = self.A @ self.mu + self.B @ u
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.Q

    def update(self, z: np.ndarray) -> None:
        S = self.H @ self.Sigma @ self.H.T + self.R
        K = self.Sigma @ self.H.T @ np.linalg.inv(S)
        innov = z - self.H @ self.mu
        self.mu = self.mu + K @ innov
        n = self.A.shape[0]
        self.Sigma = (np.eye(n) - K @ self.H) @ self.Sigma

    def step(self, u: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.predict(u)
        self.update(z)
        return self.mu.copy(), self.Sigma.copy()


@dataclass
class EKF:
    """Extended Kalman Filter for unicycle robot with range-bearing landmarks.

    State: [x, y, theta]
    Uses DiffDrive kinematic model and RangeBearingModel.
    """
    motion_model: object    # DiffDrive instance (has forward, jacobian_state, jacobian_noise)
    obs_model: object       # RangeBearingModel instance
    Q_coeffs: np.ndarray = field(default_factory=lambda: np.diag([0.05, 0.01]))
    # noise proportionality: Q = V_t diag(alpha) V_t^T -- computed per step

    mu: np.ndarray = field(init=False)
    Sigma: np.ndarray = field(init=False)

    chi2_threshold: float = 5.991  # 95% confidence, 2 DOF

    def __post_init__(self):
        self.mu = np.zeros(3)
        self.Sigma = np.eye(3) * 0.1

    def reset(self, mu: np.ndarray, Sigma: np.ndarray) -> None:
        self.mu = mu.copy()
        self.Sigma = Sigma.copy()

    def predict(self, v: float, omega: float, dt: float) -> None:
        G = self.motion_model.jacobian_state(self.mu, v, omega, dt)
        V = self.motion_model.jacobian_noise(self.mu, v, omega, dt)
        M = np.diag([
            self.Q_coeffs[0, 0] * v**2 + self.Q_coeffs[0, 0] * omega**2,
            self.Q_coeffs[1, 1] * v**2 + self.Q_coeffs[1, 1] * omega**2,
        ])
        self.mu = self.motion_model.forward(self.mu, v, omega, dt)
        self.Sigma = G @ self.Sigma @ G.T + V @ M @ V.T

    def update(
        self,
        z: np.ndarray,
        landmark: np.ndarray,
    ) -> bool:
        """Update with a range-bearing measurement.

        Returns True if the association passed the chi-square gate.
        """
        z_hat = self.obs_model.expected_measurement(self.mu, landmark)
        H = self.obs_model.jacobian(self.mu, landmark)
        S = H @ self.Sigma @ H.T + self.obs_model.R
        innov = z - z_hat
        innov[1] = _wrap_angle(innov[1])

        # Mahalanobis chi-square gate
        mahal = innov @ np.linalg.inv(S) @ innov
        if mahal > self.chi2_threshold:
            return False

        K = self.Sigma @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ innov
        self.mu[2] = _wrap_angle(self.mu[2])
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma
        return True

    def associate(
        self,
        z: np.ndarray,
        landmarks: np.ndarray,
    ) -> int | None:
        """Nearest-neighbor data association via Mahalanobis distance.

        Returns index of best landmark or None if all fail the chi-square gate.
        """
        best_idx = None
        best_dist = np.inf
        for i, lm in enumerate(landmarks):
            z_hat = self.obs_model.expected_measurement(self.mu, lm)
            H = self.obs_model.jacobian(self.mu, lm)
            S = H @ self.Sigma @ H.T + self.obs_model.R
            innov = z - z_hat
            innov[1] = _wrap_angle(innov[1])
            d = innov @ np.linalg.inv(S) @ innov
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist > self.chi2_threshold:
            return None
        return best_idx

    def covariance_ellipse(self, n_sigma: float = 2.0) -> tuple[np.ndarray, float, float, float]:
        """Return (center_xy, width, height, angle_deg) for the 2-sigma position ellipse."""
        Sigma_xy = self.Sigma[:2, :2]
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_xy)
        eigenvalues = np.maximum(eigenvalues, 0)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width  = 2 * n_sigma * np.sqrt(eigenvalues[0])
        height = 2 * n_sigma * np.sqrt(eigenvalues[1])
        return self.mu[:2].copy(), width, height, angle
