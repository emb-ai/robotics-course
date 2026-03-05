from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import cho_factor, cho_solve
from typing import Optional


def _wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _pose_compound(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """p1 ⊕ p2 : compose two SE(2) poses."""
    x1, y1, h1 = p1
    x2, y2, h2 = p2
    c, s = np.cos(h1), np.sin(h1)
    return np.array([
        x1 + c * x2 - s * y2,
        y1 + s * x2 + c * y2,
        _wrap_angle(h1 + h2),
    ])


def _pose_inverse(p: np.ndarray) -> np.ndarray:
    x, y, h = p
    c, s = np.cos(h), np.sin(h)
    return np.array([-(c * x + s * y), -(-s * x + c * y), _wrap_angle(-h)])


def _relative_pose(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Measurement prediction: relative pose of p2 w.r.t. p1 frame."""
    return _pose_compound(_pose_inverse(p1), p2)


def _jacobian_relative(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jacobians of _relative_pose(p1, p2) w.r.t. p1 and p2 (each 3x3)."""
    x1, y1, h1 = p1
    x2, y2, h2 = p2
    dx, dy = x2 - x1, y2 - y1
    c, s = np.cos(h1), np.sin(h1)

    A = np.array([
        [-c, -s, -s * dx + c * dy],
        [ s, -c, -c * dx - s * dy],
        [ 0,  0, -1],
    ])
    B = np.array([
        [c,  s, 0],
        [-s, c, 0],
        [0,  0, 1],
    ])
    return A, B


@dataclass
class PoseEdge:
    i: int
    j: int
    z: np.ndarray       # measured relative pose (3,)
    Omega: np.ndarray   # information matrix (3x3) of this constraint


@dataclass
class PoseGraph:
    """Pose graph for 2D Graph SLAM.

    Nodes: SE(2) poses  [x, y, theta]
    Edges: relative pose constraints with information matrices.

    Solve via linearization:
        Omega_full @ mu = xi_full
    using Cholesky decomposition (scipy.linalg.cho_factor / cho_solve).
    """
    n_poses: int
    poses: np.ndarray = field(init=False)  # (n, 3)
    edges: list[PoseEdge] = field(default_factory=list)

    def __post_init__(self):
        self.poses = np.zeros((self.n_poses, 3))

    def init_from_odometry(self, odometry: np.ndarray) -> None:
        """Propagate initial estimates along odometry chain.

        odometry: (n-1, 3) relative poses
        """
        self.poses[0] = np.zeros(3)
        for k in range(len(odometry)):
            self.poses[k + 1] = _pose_compound(self.poses[k], odometry[k])

    def add_edge(
        self,
        i: int,
        j: int,
        z: np.ndarray,
        info: np.ndarray | None = None,
    ) -> None:
        """Add a relative pose constraint from node i to node j."""
        if info is None:
            info = np.eye(3) * 100.0
        self.edges.append(PoseEdge(i=i, j=j, z=z, Omega=info))

    def _build_linear_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble global information matrix Omega (3n x 3n) and vector xi (3n)."""
        dim = 3 * self.n_poses
        Omega = np.zeros((dim, dim))
        xi = np.zeros(dim)

        for edge in self.edges:
            i, j = edge.i, edge.j
            z_pred = _relative_pose(self.poses[i], self.poses[j])
            e = _wrap_angle_vec(edge.z - z_pred)

            A, B = _jacobian_relative(self.poses[i], self.poses[j])

            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * j, 3 * j + 3)

            # Contributions to information matrix (linearized)
            Omega[si, si] += A.T @ edge.Omega @ A
            Omega[si, sj] += A.T @ edge.Omega @ B
            Omega[sj, si] += B.T @ edge.Omega @ A
            Omega[sj, sj] += B.T @ edge.Omega @ B

            # Contributions to information vector
            xi[si] += A.T @ edge.Omega @ e
            xi[sj] += B.T @ edge.Omega @ e

        return Omega, xi

    def optimize(self, fix_first: bool = True, n_iters: int = 10) -> np.ndarray:
        """Linearize and solve the pose graph.

        Uses iterated Gauss-Newton via Cholesky (mu = Omega^{-1} xi).
        Returns optimized poses (n, 3).
        """
        for _ in range(n_iters):
            Omega, xi = self._build_linear_system()

            if fix_first:
                # Anchor first node by adding a strong prior
                Omega[:3, :3] += np.eye(3) * 1e6

            c, low = cho_factor(Omega + np.eye(Omega.shape[0]) * 1e-9)
            delta = cho_solve((c, low), xi)
            self.poses += delta.reshape(self.n_poses, 3)
            self.poses[:, 2] = _wrap_angle(self.poses[:, 2])

        return self.poses.copy()

    def build_info_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Public accessor for the linearised system at current poses."""
        return self._build_linear_system()


def _wrap_angle_vec(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    out[2] = _wrap_angle(v[2])
    return out
