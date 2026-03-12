from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EncoderModel:
    ticks_per_rev: int = 1000

    @property
    def delta(self) -> float:
        return 2 * np.pi / self.ticks_per_rev

    @property
    def quantization_variance(self) -> float:
        return self.delta**2 / 12

    def quantize(self, theta: np.ndarray) -> np.ndarray:
        return np.round(theta / self.delta) * self.delta


def simulate_encoder(
    model: EncoderModel,
    duration: float = 2.0,
    omega: float = 3.0,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate continuous rotation and encoder output.

    Returns (true_angle, quantized_angle, time).
    """
    t = np.arange(0, duration, dt)
    true_angle = omega * t
    quantized_angle = model.quantize(true_angle)
    return true_angle, quantized_angle, t
