from __future__ import annotations

import numpy as np

from .rotations import rot2d

_OMEGA = np.array([[0, -1], [1, 0]])


def se2_hat(omega: float, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2)
    xi_hat = np.zeros((3, 3))
    xi_hat[:2, :2] = omega * _OMEGA
    xi_hat[:2, 2] = v
    return xi_hat


def se2_exp(omega: float, v: np.ndarray, tau: float) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2)
    theta = omega * tau
    R = rot2d(theta)
    if np.abs(omega) < 1e-12:
        d = tau * v
    else:
        sin_t = np.sin(theta)
        one_minus_cos = 1 - np.cos(theta)
        A = np.array([[sin_t, -one_minus_cos], [one_minus_cos, sin_t]]) / omega
        d = A @ v
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = d
    return T
