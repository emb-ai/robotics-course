import numpy as np


def normalize_axis(omega: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(omega)
    if n < 1e-10:
        return np.zeros_like(omega)
    return omega / n


def wrap_angle(theta: float) -> float:
    return np.arctan2(np.sin(theta), np.cos(theta))
