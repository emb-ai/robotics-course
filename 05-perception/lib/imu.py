from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ImuModel:
    gyro_noise: float = 0.01
    accel_noise: float = 0.05
    gyro_bias_drift: float = 1e-4
    accel_bias_drift: float = 1e-3


@dataclass
class ImuDriftResult:
    t: np.ndarray
    accel_meas: np.ndarray
    gyro_meas: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    velocity: np.ndarray
    position: np.ndarray


def simulate_imu_drift(
    model: ImuModel,
    duration: float = 60.0,
    dt: float = 0.01,
    rng: np.random.Generator | None = None,
) -> ImuDriftResult:
    """Simulate a *stationary* IMU and integrate to show position drift.

    The true acceleration is zero (gravity-compensated) and true angular
    velocity is zero, so any integrated motion is purely from noise and
    bias drift.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = int(duration / dt)
    t = np.arange(n) * dt

    accel_bias = np.zeros((n, 3))
    gyro_bias = np.zeros((n, 3))
    for i in range(1, n):
        accel_bias[i] = accel_bias[i - 1] + rng.normal(0, model.accel_bias_drift, 3) * np.sqrt(dt)
        gyro_bias[i] = gyro_bias[i - 1] + rng.normal(0, model.gyro_bias_drift, 3) * np.sqrt(dt)

    accel_noise = rng.normal(0, model.accel_noise, (n, 3)) * np.sqrt(dt)
    gyro_noise = rng.normal(0, model.gyro_noise, (n, 3)) * np.sqrt(dt)

    accel_meas = accel_bias + accel_noise
    gyro_meas = gyro_bias + gyro_noise

    velocity = np.cumsum(accel_meas * dt, axis=0)
    position = np.cumsum(velocity * dt, axis=0)

    return ImuDriftResult(
        t=t,
        accel_meas=accel_meas,
        gyro_meas=gyro_meas,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
        velocity=velocity,
        position=position,
    )


# ---------------------------------------------------------------------------
# Live IMU drift tracker — accumulates real ImuSample packets
# ---------------------------------------------------------------------------

class ImuLiveDriftTracker:
    """Double-integrate live accelerometer readings to demonstrate position drift.

    Accepts ImuSample packets from phone_stream or ws_server and keeps a
    running history suitable for re-plotting.
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self._velocity: np.ndarray = np.zeros(3)
        self._position: np.ndarray = np.zeros(3)
        self._last_ts: float | None = None

        self.timestamps: list[float] = []
        self.accels: list[np.ndarray] = []
        self.gyros: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []

    def update(self, sample: object) -> np.ndarray:
        """Accept an ImuSample and return the current integrated position."""
        # Avoid importing live at module load time (circular-safe)
        ts = float(sample.timestamp)  # type: ignore[attr-defined]
        accel = np.asarray(sample.accel, dtype=np.float64)  # type: ignore[attr-defined]
        gyro = np.asarray(sample.gyro, dtype=np.float64)  # type: ignore[attr-defined]

        dt = self.dt if self._last_ts is None else max(ts - self._last_ts, 1e-6)
        self._last_ts = ts

        self._velocity += accel * dt
        self._position += self._velocity * dt

        self.timestamps.append(ts)
        self.accels.append(accel.copy())
        self.gyros.append(gyro.copy())
        self.velocities.append(self._velocity.copy())
        self.positions.append(self._position.copy())
        return self._position.copy()

    def reset(self) -> None:
        self._velocity[:] = 0
        self._position[:] = 0
        self._last_ts = None
        self.timestamps.clear()
        self.accels.clear()
        self.gyros.clear()
        self.velocities.clear()
        self.positions.clear()

    def to_drift_result(self) -> ImuDriftResult:
        """Convert accumulated history to ImuDriftResult for existing viz reuse."""
        n = len(self.timestamps)
        zeros = np.zeros((n, 3))
        return ImuDriftResult(
            t=np.array(self.timestamps),
            accel_meas=np.array(self.accels) if self.accels else zeros,
            gyro_meas=np.array(self.gyros) if self.gyros else zeros,
            accel_bias=zeros,
            gyro_bias=zeros,
            velocity=np.array(self.velocities) if self.velocities else zeros,
            position=np.array(self.positions) if self.positions else zeros,
        )
