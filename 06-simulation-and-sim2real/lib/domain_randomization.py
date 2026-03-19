"""Domain randomization utilities for MuJoCo models.

Provides helpers to randomize visual, dynamics, and action parameters
of a MuJoCo model / data pair.  Designed for lecture demos — not a
production DR framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from numpy.typing import NDArray


@dataclass
class DRConfig:
    """Specifies which parameters to randomize and their ranges.

    Each range is ``(low, high)`` applied multiplicatively or additively
    depending on the parameter type.
    """

    # -- Dynamics --
    mass_scale: tuple[float, float] = (0.8, 1.2)
    friction_scale: tuple[float, float] = (0.5, 2.0)
    damping_scale: tuple[float, float] = (0.7, 1.5)
    gravity_offset: tuple[float, float] = (-0.5, 0.5)

    # -- Visual --
    geom_rgba_noise: float = 0.15
    light_pos_noise: float = 0.5
    light_diffuse_noise: float = 0.3

    # -- Action --
    action_noise_std: float = 0.0
    action_delay_steps: int = 0

    enabled: set[str] = field(default_factory=lambda: {
        "mass", "friction", "damping", "rgba", "light",
    })


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def randomize_dynamics(
    model: mujoco.MjModel,
    cfg: DRConfig,
    rng: np.random.Generator | None = None,
) -> dict:
    """Randomize dynamics parameters in-place.  Returns dict of applied scales."""
    rng = rng or np.random.default_rng()
    applied: dict = {}

    if "mass" in cfg.enabled:
        scales = rng.uniform(*cfg.mass_scale, size=model.nbody)
        scales[0] = 1.0  # world body
        model.body_mass[:] *= scales
        applied["mass_scales"] = scales

    if "friction" in cfg.enabled:
        scale = rng.uniform(*cfg.friction_scale)
        model.geom_friction[:, 0] *= scale
        applied["friction_scale"] = scale

    if "damping" in cfg.enabled:
        scale = rng.uniform(*cfg.damping_scale)
        model.dof_damping[:] *= scale
        applied["damping_scale"] = scale

    if "gravity" in cfg.enabled:
        offset = rng.uniform(*cfg.gravity_offset, size=3)
        offset[:2] = 0.0  # only perturb z-gravity
        model.opt.gravity[:] += offset
        applied["gravity_offset"] = offset

    return applied


def randomize_visuals(
    model: mujoco.MjModel,
    cfg: DRConfig,
    rng: np.random.Generator | None = None,
) -> dict:
    """Randomize visual parameters (geom RGBA, lights) in-place."""
    rng = rng or np.random.default_rng()
    applied: dict = {}

    if "rgba" in cfg.enabled:
        noise = rng.uniform(-cfg.geom_rgba_noise, cfg.geom_rgba_noise,
                            size=model.geom_rgba[:, :3].shape)
        model.geom_rgba[:, :3] = np.clip(model.geom_rgba[:, :3] + noise, 0.0, 1.0)
        applied["rgba_noise"] = noise

    if "light" in cfg.enabled and model.nlight > 0:
        pos_noise = rng.uniform(-cfg.light_pos_noise, cfg.light_pos_noise,
                                size=model.light_pos.shape)
        model.light_pos[:] += pos_noise
        diff_noise = rng.uniform(-cfg.light_diffuse_noise, cfg.light_diffuse_noise,
                                 size=model.light_diffuse.shape)
        model.light_diffuse[:] = np.clip(model.light_diffuse + diff_noise, 0.0, 1.0)
        applied["light_pos_noise"] = pos_noise

    return applied


def noisy_ctrl(
    ctrl: NDArray,
    cfg: DRConfig,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Add Gaussian noise to control signals."""
    if cfg.action_noise_std <= 0.0:
        return ctrl
    rng = rng or np.random.default_rng()
    return ctrl + rng.normal(0, cfg.action_noise_std, size=ctrl.shape)


class ActionDelayBuffer:
    """FIFO buffer that introduces a fixed delay to actions."""

    def __init__(self, delay_steps: int, nu: int):
        self.buffer = np.zeros((max(delay_steps, 1), nu))
        self.idx = 0
        self.delay = delay_steps

    def push(self, ctrl: NDArray) -> NDArray:
        if self.delay <= 0:
            return ctrl
        out = self.buffer[self.idx].copy()
        self.buffer[self.idx] = ctrl
        self.idx = (self.idx + 1) % self.delay
        return out


def snapshot_model_params(model: mujoco.MjModel) -> dict:
    """Save a snapshot of randomizable parameters for later restore."""
    return {
        "body_mass": model.body_mass.copy(),
        "geom_friction": model.geom_friction.copy(),
        "dof_damping": model.dof_damping.copy(),
        "gravity": model.opt.gravity.copy(),
        "geom_rgba": model.geom_rgba.copy(),
        "light_pos": model.light_pos.copy() if model.nlight > 0 else None,
        "light_diffuse": model.light_diffuse.copy() if model.nlight > 0 else None,
    }


def restore_model_params(model: mujoco.MjModel, snap: dict) -> None:
    """Restore parameters from a snapshot."""
    model.body_mass[:] = snap["body_mass"]
    model.geom_friction[:] = snap["geom_friction"]
    model.dof_damping[:] = snap["dof_damping"]
    model.opt.gravity[:] = snap["gravity"]
    model.geom_rgba[:] = snap["geom_rgba"]
    if snap["light_pos"] is not None:
        model.light_pos[:] = snap["light_pos"]
        model.light_diffuse[:] = snap["light_diffuse"]


def render_dr_grid(
    xml_source: str,
    cfg: DRConfig,
    n_variants: int = 5,
    *,
    width: int = 320,
    height: int = 240,
    seed: int = 42,
    include_nominal: bool = True,
) -> list[NDArray]:
    """Render nominal scene plus randomized variants side-by-side.

    Returns list of (H, W, 3) uint8 images.
    """
    rng = np.random.default_rng(seed)
    frames = []

    if include_nominal:
        model = mujoco.MjModel.from_xml_string(xml_source)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())
        renderer.close()

    for _ in range(n_variants):
        model = mujoco.MjModel.from_xml_string(xml_source)
        data = mujoco.MjData(model)
        randomize_dynamics(model, cfg, rng)
        randomize_visuals(model, cfg, rng)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())
        renderer.close()
    return frames
