import numpy as np
from scipy.spatial.transform import Rotation


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_zyx(alpha: float, beta: float, gamma: float) -> np.ndarray:
    return (rot_z(alpha) @ rot_y(beta) @ rot_x(gamma)).astype(float)


def skew(omega: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0],
    ])


def unskew(omega_hat: np.ndarray) -> np.ndarray:
    return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])


def rodrigues(omega_hat: np.ndarray, theta: float) -> np.ndarray:
    omega_hat_mat = skew(omega_hat)
    return (
        np.eye(3)
        + np.sin(theta) * omega_hat_mat
        + (1 - np.cos(theta)) * omega_hat_mat @ omega_hat_mat
    )


def matrix_log_so3(R: np.ndarray) -> tuple[np.ndarray, float]:
    trace_R = np.trace(R)
    if np.isclose(trace_R, 3):
        return np.zeros(3), 0.0
    theta = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0))
    if np.isclose(np.sin(theta), 0):
        if np.isclose(theta, np.pi):
            omega_hat = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
            omega_hat = omega_hat * np.sign(omega_hat[0] if omega_hat[0] != 0 else 1)
            return omega_hat, theta
    omega_hat_mat = (R - R.T) / (2 * np.sin(theta))
    omega_hat = unskew(omega_hat_mat)
    return omega_hat, theta


def rotation_from_quaternion(xyzw: np.ndarray) -> Rotation:
    return Rotation.from_quat(xyzw)


def rotation_from_matrix(R: np.ndarray) -> Rotation:
    return Rotation.from_matrix(R)


def slerp(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    r0 = Rotation.from_matrix(R0)
    r1 = Rotation.from_matrix(R1)
    key_times = [0.0, 1.0]
    key_rots = Rotation.concatenate([r0, r1])
    from scipy.spatial.transform import Slerp
    slerp_fn = Slerp(key_times, key_rots)
    return slerp_fn(t).as_matrix()


if __name__ == "__main__":
    from scipy.linalg import expm

    omega_hat = np.array([1, 1, 1]) / np.sqrt(3)
    theta = np.pi / 3
    R_rodrigues = rodrigues(omega_hat, theta)
    R_expm = expm(skew(omega_hat) * theta)
    print("Rodrigues vs expm diff:", np.linalg.norm(R_rodrigues - R_expm))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from .plotting import plot_frame_3d

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_frame_3d(np.eye(3), scale=1.0, ax=ax)
    plot_frame_3d(R_rodrigues, scale=0.8, ax=ax)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Rodrigues rotation frame")
    plt.show()
