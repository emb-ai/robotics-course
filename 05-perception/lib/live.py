from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Packet types
# ---------------------------------------------------------------------------

@dataclass
class RgbFrame:
    timestamp: float
    image: np.ndarray          # (H, W, 3) BGR uint8
    intrinsics: np.ndarray     # (3, 3) float64


@dataclass
class DepthFrame:
    timestamp: float
    depth_m: np.ndarray        # (H, W) float32, metres
    intrinsics: np.ndarray     # (3, 3) float64
    pose_T_cw: np.ndarray      # (4, 4) float64, world-to-camera


@dataclass
class ImuSample:
    timestamp: float
    accel: np.ndarray          # (3,) m/s^2 — user acceleration, gravity removed
    gyro: np.ndarray           # (3,) rad/s
    gravity: np.ndarray | None = None  # (3,) m/s^2


@dataclass
class ServoSample:
    timestamp: float
    position_ticks: int        # raw encoder ticks (0–4095)
    position_rad: float        # ticks * 2π / 4096
    speed: int | None = None   # raw signed speed
    load: int | None = None    # raw signed load


@dataclass
class CameraPoseSample:
    timestamp: float
    T_cw: np.ndarray           # (4, 4)
    intrinsics: np.ndarray     # (3, 3)


# ---------------------------------------------------------------------------
# Source modes
# ---------------------------------------------------------------------------

class SourceMode(Enum):
    LIVE = "live"
    REPLAY = "replay"
    SIMULATED = "simulated"


# ---------------------------------------------------------------------------
# SensorSource protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SensorSource(Protocol):
    mode: SourceMode

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...


# ---------------------------------------------------------------------------
# SyncBuffer — thread-safe ring buffer per packet type
# ---------------------------------------------------------------------------

class SyncBuffer:
    def __init__(self, maxlen: int = 200):
        self._maxlen = maxlen
        self._rgb: deque[RgbFrame] = deque(maxlen=maxlen)
        self._depth: deque[DepthFrame] = deque(maxlen=maxlen)
        self._imu: deque[ImuSample] = deque(maxlen=maxlen)
        self._servo: deque[ServoSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    # --- push ---

    def push_rgb(self, frame: RgbFrame) -> None:
        with self._cond:
            self._rgb.append(frame)
            self._cond.notify_all()

    def push_depth(self, frame: DepthFrame) -> None:
        with self._cond:
            self._depth.append(frame)
            self._cond.notify_all()

    def push_imu(self, sample: ImuSample) -> None:
        with self._cond:
            self._imu.append(sample)
            self._cond.notify_all()

    def push_servo(self, sample: ServoSample) -> None:
        with self._cond:
            self._servo.append(sample)
            self._cond.notify_all()

    # --- get latest (non-blocking) ---

    def get_latest_rgb(self) -> RgbFrame | None:
        with self._lock:
            return self._rgb[-1] if self._rgb else None

    def get_latest_depth(self) -> DepthFrame | None:
        with self._lock:
            return self._depth[-1] if self._depth else None

    def get_latest_imu(self) -> ImuSample | None:
        with self._lock:
            return self._imu[-1] if self._imu else None

    def get_latest_servo(self) -> ServoSample | None:
        with self._lock:
            return self._servo[-1] if self._servo else None

    # --- get recent N samples ---

    def get_recent_imu(self, n: int = 50) -> list[ImuSample]:
        with self._lock:
            buf = list(self._imu)
        return buf[-n:]

    def get_recent_servo(self, n: int = 50) -> list[ServoSample]:
        with self._lock:
            buf = list(self._servo)
        return buf[-n:]

    def get_recent_rgb(self, n: int = 10) -> list[RgbFrame]:
        with self._lock:
            buf = list(self._rgb)
        return buf[-n:]

    # --- blocking get with timeout ---

    def wait_for_imu(self, timeout: float = 2.0) -> ImuSample | None:
        with self._cond:
            deadline = time.monotonic() + timeout
            while not self._imu:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._imu[-1]

    def wait_for_rgb(self, timeout: float = 2.0) -> RgbFrame | None:
        with self._cond:
            deadline = time.monotonic() + timeout
            while not self._rgb:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._rgb[-1]

    def wait_for_depth(self, timeout: float = 5.0) -> DepthFrame | None:
        with self._cond:
            deadline = time.monotonic() + timeout
            while not self._depth:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._depth[-1]

    # --- stats ---

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "rgb": len(self._rgb),
                "depth": len(self._depth),
                "imu": len(self._imu),
                "servo": len(self._servo),
            }

    def clear(self) -> None:
        with self._lock:
            self._rgb.clear()
            self._depth.clear()
            self._imu.clear()
            self._servo.clear()


# ---------------------------------------------------------------------------
# LiveDemoSession — orchestrates one or more sources
# ---------------------------------------------------------------------------

class LiveDemoSession:
    def __init__(self, sources: list[SensorSource] | None = None):
        self._sources: list[SensorSource] = sources or []
        self.buffer = SyncBuffer()

    def add_source(self, source: SensorSource) -> None:
        self._sources.append(source)

    def start(self) -> None:
        for src in self._sources:
            src.start()

    def stop(self) -> None:
        for src in self._sources:
            src.stop()

    def is_running(self) -> bool:
        return any(src.is_running() for src in self._sources)

    # Convenience pass-throughs
    def get_latest_rgb(self) -> RgbFrame | None:
        return self.buffer.get_latest_rgb()

    def get_latest_depth(self) -> DepthFrame | None:
        return self.buffer.get_latest_depth()

    def get_latest_imu(self) -> ImuSample | None:
        return self.buffer.get_latest_imu()

    def get_latest_servo(self) -> ServoSample | None:
        return self.buffer.get_latest_servo()

    def __enter__(self) -> LiveDemoSession:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
