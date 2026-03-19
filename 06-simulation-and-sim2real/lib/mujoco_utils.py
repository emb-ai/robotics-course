"""MuJoCo helpers: load models, step simulations, render frames."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray


def load_model(source: str | Path) -> mujoco.MjModel:
    """Load MjModel from an MJCF file path or an XML string."""
    if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith("<")):
        return mujoco.MjModel.from_xml_path(str(source))
    return mujoco.MjModel.from_xml_string(source)


def make_data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    width: int = 640,
    height: int = 480,
    camera: str | int = -1,
) -> NDArray:
    """Render a single RGB frame (H, W, 3) uint8."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    renderer.close()
    return img


def rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int,
    ctrl_fn=None,
    *,
    render: bool = False,
    render_every: int = 1,
    width: int = 640,
    height: int = 480,
    camera: str | int = -1,
) -> dict:
    """Run simulation for *n_steps*.

    Parameters
    ----------
    ctrl_fn : callable(model, data) -> None, optional
        Called before each step to set ``data.ctrl``.
    render : bool
        Collect RGB frames.
    render_every : int
        Render every N steps (reduces memory for long rollouts).

    Returns
    -------
    dict with keys ``qpos`` (T+1, nq), ``qvel`` (T+1, nv), ``time`` (T+1,),
    and optionally ``frames`` list of uint8 arrays.
    """
    qpos_list = [data.qpos.copy()]
    qvel_list = [data.qvel.copy()]
    time_list = [data.time]
    frames = []

    renderer = None
    if render:
        renderer = mujoco.Renderer(model, height=height, width=width)

    for i in range(n_steps):
        if ctrl_fn is not None:
            ctrl_fn(model, data)
        mujoco.mj_step(model, data)
        qpos_list.append(data.qpos.copy())
        qvel_list.append(data.qvel.copy())
        time_list.append(data.time)
        if render and (i % render_every == 0):
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()

    result = {
        "qpos": np.array(qpos_list),
        "qvel": np.array(qvel_list),
        "time": np.array(time_list),
    }
    if render:
        result["frames"] = frames
    return result


def describe_model(model: mujoco.MjModel) -> dict:
    """Return a summary dict of the model's bodies, joints, actuators, geoms."""
    bodies = []
    for i in range(model.nbody):
        bodies.append({
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}",
            "mass": float(model.body_mass[i]),
        })

    joints = []
    for i in range(model.njnt):
        joints.append({
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}",
            "type": ["free", "ball", "slide", "hinge"][model.jnt_type[i]],
            "range": model.jnt_range[i].tolist() if model.jnt_limited[i] else None,
            "damping": float(model.dof_damping[model.jnt_dofadr[i]]),
        })

    actuators = []
    for i in range(model.nu):
        actuators.append({
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}",
            "ctrl_range": model.actuator_ctrlrange[i].tolist() if model.actuator_ctrllimited[i] else None,
        })

    geoms = []
    for i in range(model.ngeom):
        geoms.append({
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}",
            "type": [
                "plane", "hfield", "sphere", "capsule", "ellipsoid",
                "cylinder", "box", "mesh",
            ][model.geom_type[i]],
            "friction": model.geom_friction[i].tolist(),
        })

    return {
        "nbody": model.nbody,
        "njnt": model.njnt,
        "nu": model.nu,
        "ngeom": model.ngeom,
        "bodies": bodies,
        "joints": joints,
        "actuators": actuators,
        "geoms": geoms,
        "timestep": float(model.opt.timestep),
    }


def print_model_summary(model: mujoco.MjModel) -> None:
    """Pretty-print model structure to stdout."""
    info = describe_model(model)
    print(f"Bodies: {info['nbody']}  |  Joints: {info['njnt']}  |  "
          f"Actuators: {info['nu']}  |  Geoms: {info['ngeom']}  |  dt={info['timestep']}")

    if info["joints"]:
        print("\nJoints:")
        for j in info["joints"]:
            rng = f"  range={j['range']}" if j["range"] else ""
            print(f"  [{j['id']}] {j['name']:20s}  type={j['type']:6s}  damping={j['damping']:.4f}{rng}")

    if info["actuators"]:
        print("\nActuators:")
        for a in info["actuators"]:
            rng = f"  ctrl_range={a['ctrl_range']}" if a["ctrl_range"] else ""
            print(f"  [{a['id']}] {a['name']:20s}{rng}")
