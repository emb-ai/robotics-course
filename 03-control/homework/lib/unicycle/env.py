"""MuJoCo-backed unicycle control environment.

The Python task uses the same simplified MJCF as the JavaScript viewer.  The
student policy sees the current target and the raw MuJoCo state, then returns
the three direct motor torques declared in the MJCF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised only when deps are missing.
    raise ImportError(
        "The unicycle environment requires the Python 'mujoco' package. "
        "Install the homework requirements before using lib.unicycle."
    ) from exc


UNICYCLE_QPOS_SIZE = 10
UNICYCLE_QVEL_SIZE = 9
UNICYCLE_ACTION_SIZE = 3
UNICYCLE_OBSERVATION_SIZE = 2 + UNICYCLE_QPOS_SIZE + UNICYCLE_QVEL_SIZE
UNICYCLE_ACTION_NAMES = ("pelvis_y", "pelvis_x", "wheel")
UNICYCLE_OBSERVATION_NAMES = (
    "target_x",
    "target_y",
    "root_x",
    "root_y",
    "root_z",
    "root_qw",
    "root_qx",
    "root_qy",
    "root_qz",
    "wheel_angle",
    "pelvis_y_angle",
    "pelvis_x_angle",
    "root_vx",
    "root_vy",
    "root_vz",
    "root_omega_x",
    "root_omega_y",
    "root_omega_z",
    "wheel_angular_velocity",
    "pelvis_y_angular_velocity",
    "pelvis_x_angular_velocity",
)
FIXED_TARGET_CASES = (
    (2.0, 0.0),
    (-2.0, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
    (2.4, -1.6),
)


def default_mjcf_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "unicycle_simplified.xml"


def fixed_target_cases() -> tuple[np.ndarray, ...]:
    """Return deterministic target locations for the student solution tests."""
    return tuple(np.asarray(target, dtype=float) for target in FIXED_TARGET_CASES)


@dataclass
class UnicycleEnvConfig:
    mjcf_path: Path = field(default_factory=default_mjcf_path)
    target_xy: tuple[float, float] = FIXED_TARGET_CASES[0]
    policy_dt: float = 0.02
    time_limit: float = 12.0
    success_radius: float = 0.45
    fall_height: float = 0.65
    debug: bool = False


class UnicycleEnv:
    """Small Gym-like environment for the simplified MuJoCo unicycle.

    Observation order is ``[target_x, target_y, qpos(10), qvel(9)]``:
    ``[target_x, target_y, root_x, root_y, root_z, root_qw, root_qx,
    root_qy, root_qz, wheel_angle, pelvis_y_angle, pelvis_x_angle,
    root_vx, root_vy, root_vz, root_omega_x, root_omega_y, root_omega_z,
    wheel_angular_velocity, pelvis_y_angular_velocity,
    pelvis_x_angular_velocity]``.

    The root free joint belongs to the wheel body. Its ``qpos`` block uses
    MuJoCo's position-plus-quaternion convention, not Euler angles.
    Action order is ``[tau_pelvis_y, tau_pelvis_x, tau_wheel]``.
    """

    def __init__(self, config: UnicycleEnvConfig | None = None):
        self.config = config if config is not None else UnicycleEnvConfig()
        self.model = mujoco.MjModel.from_xml_path(str(Path(self.config.mjcf_path)))
        self.data = mujoco.MjData(self.model)

        if self.model.nq != UNICYCLE_QPOS_SIZE or self.model.nv != UNICYCLE_QVEL_SIZE:
            raise ValueError(
                "Unexpected unicycle MJCF state size: "
                f"nq={self.model.nq}, nv={self.model.nv}; expected "
                f"{UNICYCLE_QPOS_SIZE}, {UNICYCLE_QVEL_SIZE}."
            )

        self.actuator_ids = np.array([self._actuator_id(name) for name in UNICYCLE_ACTION_NAMES], dtype=int)
        self.action_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].astype(float).copy()
        self.action_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].astype(float).copy()

        self.wheel_body_id = self._body_id("wheel")
        self.com_body_id = self._body_id("com")
        self.model_timestep = float(self.model.opt.timestep)
        self.policy_steps = self._policy_steps(self.config.policy_dt)
        self.policy_dt = self.policy_steps * self.model_timestep

        self.target_xy = np.asarray(self.config.target_xy, dtype=float).copy()
        self.t = 0.0
        self.last_action = np.zeros(UNICYCLE_ACTION_SIZE, dtype=float)
        self.status = "ready"
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del seed
        options = {} if options is None else dict(options)
        if "target_xy" in options:
            self.target_xy = self._as_target(options["target_xy"])
        else:
            self.target_xy = np.asarray(self.config.target_xy, dtype=float).copy()

        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        self.last_action = np.zeros(UNICYCLE_ACTION_SIZE, dtype=float)
        self.status = "running"
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        u = self._clip_action(action)
        self.last_action = u
        self.data.ctrl[self.actuator_ids] = u

        for _ in range(self.policy_steps):
            mujoco.mj_step(self.model, self.data)
        self.t += self.policy_dt

        reward = 0.0
        terminated = False
        truncated = False

        if self._is_success():
            self.status = "success"
            reward = 1.0
            terminated = True
        elif self._has_fallen():
            self.status = "failed"
            reward = -1.0
            terminated = True
        elif self.t >= self.config.time_limit - 1e-12:
            self.status = "timeout"
            reward = -1.0
            truncated = True
        else:
            self.status = "running"

        return self._observation(), reward, terminated, truncated, self._info()

    @property
    def observation_size(self) -> int:
        return UNICYCLE_OBSERVATION_SIZE

    @property
    def action_size(self) -> int:
        return UNICYCLE_ACTION_SIZE

    def wheel_xy(self) -> np.ndarray:
        return self.data.xpos[self.wheel_body_id, :2].astype(float).copy()

    def com_height(self) -> float:
        return float(self.data.xpos[self.com_body_id, 2])

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.wheel_xy() - self.target_xy))

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.target_xy.astype(float, copy=True),
                self.data.qpos.astype(float, copy=True),
                self.data.qvel.astype(float, copy=True),
            ]
        )

    def _info(self) -> dict:
        info = {
            "t": self.t,
            "status": self.status,
            "target_xy": self.target_xy.copy(),
            "wheel_xy": self.wheel_xy(),
            "distance_to_target": self.distance_to_target(),
            "action": self.last_action.copy(),
            "success": self.status == "success",
            "fallen": self._has_fallen(),
        }
        if self.config.debug:
            info.update(
                qpos=self.data.qpos.copy(),
                qvel=self.data.qvel.copy(),
                com_height=self.com_height(),
                model_timestep=self.model_timestep,
                policy_steps=self.policy_steps,
            )
        return info

    def _is_success(self) -> bool:
        return self.distance_to_target() <= self.config.success_radius and not self._has_fallen()

    def _has_fallen(self) -> bool:
        return self.com_height() < self.config.fall_height

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        u = np.asarray(action, dtype=float)
        if u.shape != (UNICYCLE_ACTION_SIZE,):
            raise ValueError(f"action must have shape ({UNICYCLE_ACTION_SIZE},), got {u.shape}")
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(u, self.action_low, self.action_high)

    def _as_target(self, target_xy: Iterable[float]) -> np.ndarray:
        target = np.asarray(target_xy, dtype=float)
        if target.shape != (2,):
            raise ValueError(f"target_xy must have shape (2,), got {target.shape}")
        return target.copy()

    def _policy_steps(self, policy_dt: float) -> int:
        if policy_dt <= 0.0:
            raise ValueError("policy_dt must be positive")
        steps = int(round(float(policy_dt) / self.model_timestep))
        return max(1, steps)

    def _actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"actuator not found in unicycle MJCF: {name}")
        return int(actuator_id)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"body not found in unicycle MJCF: {name}")
        return int(body_id)
