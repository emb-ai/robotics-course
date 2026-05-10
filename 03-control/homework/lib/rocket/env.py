"""
2D planar rocket with thrust vector control (TVC).

Provides dynamics, RK4 simulator, ascent reference trajectory generator,
and wind disturbance helpers.  The linearisation is left to students.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class RocketParams:
    m: float = 50.0           # total mass [kg] (constant – no fuel depletion)
    I: float = 100.0          # pitch moment of inertia about CoG [kg·m²]
    l: float = 1.5            # distance from CoG to nozzle gimbal pivot [m]
    g: float = 9.81           # gravitational acceleration [m/s²]
    T_max: float = 981.0      # maximum thrust [N]  (gives TWR = 2)
    T_min: float = 0.0        # minimum thrust [N]  (engine can shut off)
    delta_max: float = 0.26   # maximum gimbal deflection [rad] (≈15°)
    body_length: float = 5.0  # rocket body length [m] (visualisation only)
    body_width: float = 1.0   # rocket body width  [m] (visualisation only)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
def rocket_dynamics(
    state: np.ndarray,
    control: np.ndarray,
    params: RocketParams,
    wind_accel: float = 0.0,
    wind_angular_accel: float = 0.0,
) -> np.ndarray:
    """Continuous-time state derivative for the planar rocket.

    State  ``[x, z, vx, vz, theta, omega]``
    Control ``[T, delta]``   (thrust magnitude, gimbal angle)

    Sign convention
    ---------------
    * *theta* is measured from vertical; ``sin(theta + delta)`` gives the
      horizontal component of the thrust direction.
    * The torque is ``-(T l / I) sin(delta)`` – a positive gimbal deflection
      produces a *negative* angular acceleration (restoring torque).

    Parameters
    ----------
    state : (6,) array
    control : (2,) array – **not** clipped here; the simulator clips.
    params : RocketParams
    wind_accel : float
        Horizontal acceleration disturbance added to ``dvx``.
    wind_angular_accel : float
        Angular acceleration disturbance added to ``domega``.
    """
    x, z, vx, vz, theta, omega = state
    T, delta = control

    ang = theta + delta
    dx = vx
    dz = vz
    dvx = (T / params.m) * np.sin(ang) + wind_accel
    dvz = (T / params.m) * np.cos(ang) - params.g
    dtheta = omega
    domega = -(T * params.l / params.I) * np.sin(delta) + wind_angular_accel

    return np.array([dx, dz, dvx, dvz, dtheta, domega])


# ---------------------------------------------------------------------------
# RK4 simulator
# ---------------------------------------------------------------------------
ControllerFn = Callable[[float, np.ndarray], np.ndarray]
WindFn = Callable[[float], float]


def _clip_control(u: np.ndarray, p: RocketParams) -> np.ndarray:
    T = np.clip(u[0], p.T_min, p.T_max)
    delta = np.clip(u[1], -p.delta_max, p.delta_max)
    return np.array([T, delta])


def simulate(
    x0: np.ndarray,
    controller_fn: ControllerFn,
    t_final: float,
    dt: float,
    params: RocketParams,
    wind_fn: WindFn | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the rocket with a closed-loop controller using RK4.

    Parameters
    ----------
    x0 : (6,) initial state
    controller_fn : callable ``(t, state) -> [T, delta]``
    t_final : end time [s]
    dt : integration step [s]
    params : RocketParams
    wind_fn : optional ``t -> float`` horizontal acceleration disturbance

    Returns
    -------
    t : (N+1,) time grid
    states : (N+1, 6)
    controls : (N, 2)  – clipped controls actually applied
    """
    t_grid = np.arange(0.0, t_final + 0.5 * dt, dt)
    n_steps = len(t_grid) - 1
    states = np.zeros((n_steps + 1, 6))
    controls = np.zeros((n_steps, 2))
    states[0] = np.asarray(x0, dtype=float)

    for i in range(n_steps):
        t_i = t_grid[i]
        s_i = states[i]
        u_raw = controller_fn(t_i, s_i)
        u = _clip_control(np.asarray(u_raw, dtype=float), params)
        controls[i] = u
        w = wind_fn(t_i) if wind_fn is not None else 0.0

        def f(s, t_loc):
            w_loc = wind_fn(t_loc) if wind_fn is not None else 0.0
            return rocket_dynamics(s, u, params, w_loc)

        k1 = f(s_i, t_i)
        k2 = f(s_i + 0.5 * dt * k1, t_i + 0.5 * dt)
        k3 = f(s_i + 0.5 * dt * k2, t_i + 0.5 * dt)
        k4 = f(s_i + dt * k3, t_i + dt)
        states[i + 1] = s_i + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_grid, states, controls


# ---------------------------------------------------------------------------
# Ascent reference trajectory
# ---------------------------------------------------------------------------
def _smoothstep(t: np.ndarray, t_final: float) -> np.ndarray:
    """Quintic smoothstep  s(tau) = 6τ⁵ − 15τ⁴ + 10τ³,  tau = t/t_final."""
    tau = np.clip(t / t_final, 0.0, 1.0)
    return tau * tau * tau * (tau * (tau * 6.0 - 15.0) + 10.0)


def generate_ascent_reference(
    params: RocketParams,
    z_target: float,
    t_final: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth altitude-ramp reference for the ascent task.

    The lateral state (x, vx, theta, omega) is zero throughout; only
    z and vz follow a quintic smoothstep profile.

    Returns ``(t, states_ref, controls_ref)`` with shapes
    ``(N+1,)``, ``(N+1, 6)``, ``(N+1, 2)``.
    """
    t = np.arange(0.0, t_final + 0.5 * dt, dt)
    s = _smoothstep(t, t_final)

    z_ref = z_target * s
    vz_ref = np.gradient(z_ref, t, edge_order=2)
    az_ref = np.gradient(vz_ref, t, edge_order=2)

    states_ref = np.zeros((len(t), 6))
    states_ref[:, 1] = z_ref
    states_ref[:, 3] = vz_ref

    controls_ref = np.zeros((len(t), 2))
    controls_ref[:, 0] = params.m * (params.g + az_ref)

    return t, states_ref, controls_ref


# ---------------------------------------------------------------------------
# Wind helpers
# ---------------------------------------------------------------------------
def make_sinusoidal_wind(
    amplitude: float = 2.0,
    omega: float = 0.5,
) -> WindFn:
    """Return ``t -> A sin(ω t)`` horizontal acceleration disturbance."""
    def wind(t: float) -> float:
        return amplitude * np.sin(omega * t)
    return wind


# ---------------------------------------------------------------------------
# Gym-like environment
# ---------------------------------------------------------------------------
@dataclass
class WindConfig:
    """Smooth hidden wind disturbance configuration."""

    strength: float = 2.2
    seed: int = 98765
    time_scale: float = 1.35
    surface_fade_start: float = 0.25
    surface_fade_end: float = 8.0
    center_of_pressure_offset: float = 1.1
    rotation_coupling: float = 0.24


@dataclass
class RocketScenario:
    """Scenario definition for :class:`RocketEnv`.

    ``initial_state`` uses the internal dynamics order:
    ``[x, z, vx, vz, theta, omega]``.
    """

    kind: str
    initial_state: np.ndarray
    target_x: float = 0.0
    target_z: float = 0.0
    target_theta: float = 0.0
    time_limit: float = 20.0
    x_tolerance: float = 2.0
    z_tolerance: float = 4.0
    speed_tolerance: float = 2.8
    vertical_speed_tolerance: float = 2.4
    theta_tolerance: float = 0.15
    omega_tolerance: float = 0.65
    crash_angle_tolerance: float = 0.35


@dataclass
class RocketEnvConfig:
    params: RocketParams = field(default_factory=RocketParams)
    scenario: RocketScenario = field(default_factory=lambda: make_ascent_scenario())
    wind: WindConfig = field(default_factory=WindConfig)
    dt: float = 0.02
    render_width: int = 640
    render_height: int = 480
    debug: bool = False


def state_to_observation(state: np.ndarray, scenario: RocketScenario | None = None) -> np.ndarray:
    """Convert internal state order to env observation order.

    Internal: ``[x, z, vx, vz, theta, omega]``
    Observation:
    ``[target_x, target_z, target_theta, x, z, theta, vx, vz, omega]``
    """
    s = np.asarray(state, dtype=float)
    if scenario is None:
        target_x = target_z = target_theta = 0.0
    else:
        target_x = scenario.target_x
        target_z = scenario.target_z
        target_theta = scenario.target_theta
    return np.array(
        [target_x, target_z, target_theta, s[0], s[1], s[4], s[2], s[3], s[5]],
        dtype=float,
    )


def observation_to_state(obs: np.ndarray) -> np.ndarray:
    """Convert env observation order to internal dynamics state order."""
    o = np.asarray(obs, dtype=float)
    return np.array([o[3], o[4], o[6], o[7], o[5], o[8]], dtype=float)


def make_ascent_scenario(
    z_target: float = 100.0,
    t_final: float = 20.0,
    initial_state: np.ndarray | None = None,
    target_x: float = 0.0,
    target_theta: float = 0.0,
) -> RocketScenario:
    return RocketScenario(
        kind="ascent",
        initial_state=np.zeros(6) if initial_state is None else np.asarray(initial_state, dtype=float),
        target_x=target_x,
        target_z=z_target,
        target_theta=target_theta,
        time_limit=t_final,
        x_tolerance=20.0,
        z_tolerance=5.0,
        speed_tolerance=4.0,
        vertical_speed_tolerance=2.5,
        theta_tolerance=0.15,
    )


def make_landing_scenario(
    initial_state: np.ndarray | None = None,
    t_final: float = 20.0,
    target_x: float = 0.0,
    target_z: float = 0.0,
    target_theta: float = 0.0,
) -> RocketScenario:
    x0 = np.array([30.0, 100.0, -3.0, -10.0, 0.05, 0.0])
    return RocketScenario(
        kind="landing",
        initial_state=x0 if initial_state is None else np.asarray(initial_state, dtype=float),
        target_x=target_x,
        target_z=target_z,
        target_theta=target_theta,
        time_limit=t_final,
        x_tolerance=5.0,
        z_tolerance=3.0,
        speed_tolerance=2.8,
        vertical_speed_tolerance=2.4,
        theta_tolerance=0.15,
        omega_tolerance=0.65,
        crash_angle_tolerance=0.35,
    )


def _angle_error(a: float, b: float) -> float:
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _smoothstep_scalar(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = _clip01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def wind_surface_factor(z: float, config: WindConfig) -> float:
    return _smoothstep_scalar(config.surface_fade_start, config.surface_fade_end, max(0.0, float(z)))


def _hash_uint32(seed: int, index: int) -> int:
    h = (int(seed) ^ ((int(index) * 0x9E3779B9) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x7FEB352D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846CA68B) & 0xFFFFFFFF
    h ^= h >> 16
    return h & 0xFFFFFFFF


def _gradient_1d(seed: int, index: int) -> float:
    return (_hash_uint32(seed, index) / 0xFFFFFFFF) * 2.0 - 1.0


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _perlin_1d(seed: int, x: float) -> float:
    i0 = int(np.floor(x))
    t = x - i0
    g0 = _gradient_1d(seed, i0)
    g1 = _gradient_1d(seed, i0 + 1)
    v0 = g0 * t
    v1 = g1 * (t - 1.0)
    return ((1.0 - _fade(t)) * v0 + _fade(t) * v1) * 2.0


def _wind_time_offset(seed: int) -> float:
    return 3.5 + (_hash_uint32(seed, 0x51F15E) / 0xFFFFFFFF) * 19.0


def wind_base_accel(t: float, config: WindConfig) -> float:
    wind_t = float(t) + _wind_time_offset(config.seed)
    primary = _perlin_1d(config.seed, wind_t / config.time_scale)
    secondary = _perlin_1d(config.seed ^ 0x5BD1E995, wind_t / (config.time_scale * 0.52) + 19.7)
    raw = np.clip((0.86 * primary + 0.18 * secondary) * 1.35, -1.0, 1.0)
    return float(raw * config.strength)


def wind_accel_at(t: float, z: float, config: WindConfig) -> float:
    return wind_base_accel(t, config) * wind_surface_factor(z, config)


def wind_angular_accel(state: np.ndarray, wind_accel: float, params: RocketParams, config: WindConfig) -> float:
    theta = float(np.asarray(state)[4])
    return float(
        -config.rotation_coupling
        * (params.m / params.I)
        * wind_accel
        * config.center_of_pressure_offset
        * np.cos(theta)
    )


class RocketEnv:
    """Dependency-free Gym-like rocket environment.

    Observations are
    ``[target_x, target_z, target_theta, x, z, theta, vx, vz, omega]``.
    Wind is hidden from observations and from ``info`` unless
    ``config.debug`` is true.
    """

    def __init__(self, config: RocketEnvConfig | None = None):
        self.config = config if config is not None else RocketEnvConfig()
        self.params = self.config.params
        self.scenario = self.config.scenario
        self.wind = self.config.wind
        self.state = np.asarray(self.scenario.initial_state, dtype=float).copy()
        self.t = 0.0
        self.last_action = np.array([0.0, 0.0])
        self.status = "ready"

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.wind = WindConfig(**{**self.wind.__dict__, "seed": int(seed)})
        if options:
            if "scenario" in options:
                self.scenario = options["scenario"]
            if "wind" in options:
                self.wind = options["wind"]
        self.state = np.asarray(self.scenario.initial_state, dtype=float).copy()
        self.t = 0.0
        self.last_action = np.array([0.0, 0.0])
        self.status = "flying"
        return state_to_observation(self.state, self.scenario), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        u = _clip_control(np.asarray(action, dtype=float), self.params)
        self.last_action = u
        s0 = self.state.copy()
        dt = float(self.config.dt)

        def f(s: np.ndarray, t_loc: float) -> np.ndarray:
            wa = wind_accel_at(t_loc, s[1], self.wind)
            wo = wind_angular_accel(s, wa, self.params, self.wind)
            return rocket_dynamics(s, u, self.params, wa, wo)

        k1 = f(s0, self.t)
        k2 = f(s0 + 0.5 * dt * k1, self.t + 0.5 * dt)
        k3 = f(s0 + 0.5 * dt * k2, self.t + 0.5 * dt)
        k4 = f(s0 + dt * k3, self.t + dt)
        impact_state = s0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state = impact_state.copy()
        self.t += dt

        ground_contact = self.state[1] <= 0.0 and self.t > 0.5 * dt
        if ground_contact:
            self.state[1] = 0.0
            if self.state[3] < 0.0:
                self.state[3] = 0.0

        terminated, reward = self._evaluate_terminal(impact_state, ground_contact)
        truncated = False
        if not terminated and self.t >= self.scenario.time_limit - 1e-12:
            truncated = True
            self.status = "timeout"
            reward = -1.0

        return state_to_observation(self.state, self.scenario), reward, terminated, truncated, self._info()

    def render(self, mode: str = "rgb_array"):
        if mode == "matplotlib":
            return self._render_matplotlib()
        if mode != "rgb_array":
            raise ValueError("render mode must be 'rgb_array' or 'matplotlib'")
        return self._render_rgb_array()

    def _evaluate_terminal(self, impact_state: np.ndarray, ground_contact: bool) -> tuple[bool, float]:
        sc = self.scenario
        s = impact_state
        speed = float(np.hypot(s[2], s[3]))
        theta_ok = abs(_angle_error(s[4], sc.target_theta)) <= sc.theta_tolerance

        if sc.kind == "ascent":
            near_target = abs(s[1] - sc.target_z) <= sc.z_tolerance
            crossed_target = s[1] >= sc.target_z
            vz_ok = abs(s[3]) <= sc.vertical_speed_tolerance
            x_ok = abs(s[0] - sc.target_x) <= sc.x_tolerance
            if near_target and vz_ok and theta_ok and x_ok and self.t >= 0.75 * sc.time_limit:
                self.status = "success"
                return True, 1.0
            if s[1] > sc.target_z + sc.z_tolerance:
                self.status = "failed"
                return True, -1.0
            if ground_contact and self.t > 0.25:
                self.status = "crashed"
                return True, -1.0
            self.status = "flying"
            return False, 0.0

        near_pad = abs(s[1] - sc.target_z) <= sc.z_tolerance and self.t >= 0.75 * sc.time_limit
        if ground_contact or near_pad:
            x_ok = abs(s[0] - sc.target_x) <= sc.x_tolerance
            z_ok = abs(s[1] - sc.target_z) <= sc.z_tolerance
            speed_ok = speed <= sc.speed_tolerance and abs(s[3]) <= sc.vertical_speed_tolerance
            omega_ok = abs(s[5]) <= sc.omega_tolerance
            if x_ok and z_ok and speed_ok and theta_ok and omega_ok:
                self.status = "success"
                return True, 1.0
            if near_pad and not ground_contact:
                self.status = "flying"
                return False, 0.0
            self.status = "crashed"
            return True, -1.0

        if abs(s[4]) > np.pi / 2 or s[1] < -1.0:
            self.status = "crashed"
            return True, -1.0

        self.status = "flying"
        return False, 0.0

    def _info(self) -> dict:
        info = {
            "t": self.t,
            "status": self.status,
            "action": self.last_action.copy(),
        }
        if self.config.debug:
            base = wind_base_accel(self.t, self.wind)
            surface = wind_surface_factor(self.state[1], self.wind)
            info.update(
                wind_base_accel=base,
                wind_surface_factor=surface,
                wind_accel=base * surface,
                internal_state=self.state.copy(),
            )
        return info

    def _render_rgb_array(self) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = self._render_matplotlib()
        fig.set_size_inches(self.config.render_width / 100, self.config.render_height / 100)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()
        plt.close(fig)
        return rgb

    def _render_matplotlib(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor="#101628")
        ax.set_facecolor("#101628")
        ax.tick_params(colors="#d8e8e8")
        for spine in ax.spines.values():
            spine.set_color("#3dff8c")

        x, z, vx, vz, theta, omega = self.state
        sc = self.scenario
        view_z_max = max(sc.target_z + 15.0, z + 20.0, 40.0)
        view_x_span = max(abs(x - sc.target_x) * 2.4 + 20.0, 60.0)
        ax.set_xlim(sc.target_x - view_x_span / 2, sc.target_x + view_x_span / 2)
        ax.set_ylim(-5.0, view_z_max)
        ax.axhspan(-5.0, 0.0, color="#1f4b24")
        ax.axhline(0.0, color="#beffdc", lw=1.0, alpha=0.5)
        ax.scatter([sc.target_x], [sc.target_z], s=80, c="#3dff8c", alpha=0.55, marker="s")

        L = self.params.body_length
        nose = np.array([x + np.sin(theta) * L * 0.45, z + np.cos(theta) * L * 0.45])
        tail = np.array([x - np.sin(theta) * L * 0.35, z - np.cos(theta) * L * 0.35])
        ax.plot([tail[0], nose[0]], [tail[1], nose[1]], color="#f0f0f0", lw=4)
        ax.scatter([nose[0]], [nose[1]], c="#e03030", s=30)

        thrust_ratio = np.clip(self.last_action[0] / self.params.T_max, 0.0, 1.0)
        if thrust_ratio > 0.02:
            delta = self.last_action[1]
            flame_dir = theta + delta + np.pi
            flame = tail + np.array([np.sin(flame_dir), np.cos(flame_dir)]) * L * 0.5 * thrust_ratio
            ax.plot([tail[0], flame[0]], [tail[1], flame[1]], color="#ff7b22", lw=2)

        if self.config.debug:
            w = wind_accel_at(self.t, z, self.wind)
            ax.text(0.02, 0.95, f"wind {w:+.2f} m/s2", transform=ax.transAxes, color="#d8e8e8")
        ax.set_title(f"RocketEnv {self.scenario.kind}: {self.status}", color="#d8e8e8")
        ax.set_xlabel("x [m]", color="#d8e8e8")
        ax.set_ylabel("z [m]", color="#d8e8e8")
        fig.tight_layout()
        return fig
