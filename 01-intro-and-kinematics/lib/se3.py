import numpy as np
from .rotations import rodrigues, matrix_log_so3, skew


def se3_matrix(omega: np.ndarray, v: np.ndarray) -> np.ndarray:
    S = np.zeros((4, 4))
    S[:3, :3] = skew(omega)
    S[:3, 3] = v
    return S


def se3_exp(S: np.ndarray, theta: float) -> np.ndarray:
    omega = S[:3]
    v = S[3:]
    if np.linalg.norm(omega) < 1e-10:
        T = np.eye(4)
        T[:3, 3] = v * theta
        return T
    omega_mat = skew(omega)
    R = rodrigues(omega, theta)
    G = (
        np.eye(3) * theta
        + (1 - np.cos(theta)) * omega_mat
        + (theta - np.sin(theta)) * omega_mat @ omega_mat
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    omega_hat, theta = matrix_log_so3(R)
    if np.isclose(theta, 0):
        return np.concatenate([np.zeros(3), p])
    omega_mat = skew(omega_hat)
    G_inv = (
        np.eye(3) / theta
        - 0.5 * omega_mat
        + (1 / theta - 0.5 / np.tan(theta / 2)) * omega_mat @ omega_mat
    )
    v = G_inv @ p
    return np.concatenate([omega_hat * theta, v * theta])


def adjoint_matrix(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew(p) @ R
    return Ad


def se3_interpolate(T0: np.ndarray, T1: np.ndarray, t: float) -> np.ndarray:
    V = se3_log(np.linalg.inv(T0) @ T1)
    return T0 @ se3_exp(V, t)


if __name__ == "__main__":
    T = se3_exp(np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]), np.pi / 2)
    V = se3_log(T)
    T_recon = se3_exp(V, 1.0)
    print("se3 roundtrip error:", np.linalg.norm(T - T_recon))

    T_sb = se3_exp(np.array([0, 0, 1, 0.5, 0, 0]), np.pi / 4)
    V_b = np.array([0, 0, 1, 0.1, 0.2, 0])
    Ad = adjoint_matrix(T_sb)
    V_s = Ad @ V_b
    print("Adjoint V_b -> V_s:", V_s)
