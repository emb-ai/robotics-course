"""
Tests for kinetic energy computation (Task 1.1).

Run with:  pytest tests/test_kin_energy.py -v
"""
import numpy as np
import pytest

from lib.phys.phys_objects import RigidBody
from lib.phys.physics_world import PhysicsEngine
from lib.basic_structs import Vec3, Pose
from solutions.kin_energy import KinEnergyCallback
from solutions.ode_solvers import EulerMethod, RK4Method


# ── helpers ──────────────────────────────────────────────────────

def _box_inertia(mass: float, extents):
    """Inertia tensor of a uniform box (same formula as entity.get_box_inertia)."""
    x, y, z = extents
    return mass / 12 * np.array([
        [y**2 + z**2, 0, 0],
        [0, x**2 + z**2, 0],
        [0, 0, x**2 + y**2],
    ])


def _make_body(lin_momentum=(0, 0, 0), ang_momentum=(10, 10, 0),
               extents=(2, 12, 3), mass=1.0):
    """Create a RigidBody with the same defaults as kin_energy.main()."""
    inertia = _box_inertia(mass, extents)
    return RigidBody(
        name='rect',
        inv_mass=1.0 / mass,
        body_inertia_tensor_inv=np.linalg.inv(inertia),
        pose=Pose(),
        linear_momentum=Vec3(*lin_momentum),
        angular_momentum=Vec3(*ang_momentum),
    )


N_STEPS = 100
DT = 1e-2

_callback = KinEnergyCallback(draw_graph=False)
calc_kinetic_energy = _callback.calc_kinetic_energy


def _run_simulation(solver, n_steps=N_STEPS, dt=DT):
    """
    Step the physics engine *n_steps* times and return the kinetic-energy
    history list (length == n_steps).

    energies[i] is the kinetic energy **after** the (i+1)-th step,
    matching KinEnergyCallback.kin_e_history indexing.
    """
    body = _make_body()
    engine = PhysicsEngine(
        bodies=[body],
        dynamics_solver=solver,
        enable_collisions=False,
    )

    energies = []
    t = 0.0
    for _ in range(n_steps):
        engine.step(t, t + dt)
        e = calc_kinetic_energy(engine.bodies)
        energies.append(e)
        t += dt
    return energies



GT_CHECK_STEPS = [0, 9, 49, 99]

GT_ENERGY_EULER = [50.1043, 50.3618, 51.4929, 52.8988] 
GT_ENERGY_RK4   = [50.0754, 50.0754, 50.0754, 50.0754]

RTOL = 1e-4   # relative tolerance for ground-truth comparison


# ── sanity checks ────────────────────────────────────────────────

class TestKineticEnergySanity:
    """Basic sanity checks for calc_kinetic_energy."""

    def test_returns_scalar(self):
        """Energy must be a single number, not an array."""
        body = _make_body()
        e = calc_kinetic_energy([body])
        assert np.isscalar(e) or (isinstance(e, np.ndarray) and e.ndim == 0), \
            f"Expected a scalar, got {type(e).__name__} with value {e}"

    def test_returns_real_number(self):
        """Energy must be a real (not complex) number."""
        body = _make_body()
        e = calc_kinetic_energy([body])
        assert np.isreal(e), f"Expected a real number, got {e}"

    def test_returns_finite(self):
        """Energy must be finite (not NaN or ±inf)."""
        body = _make_body()
        e = calc_kinetic_energy([body])
        assert np.isfinite(float(e)), f"Expected finite value, got {e}"

    def test_nonnegative(self):
        """Kinetic energy is never negative."""
        body = _make_body()
        e = calc_kinetic_energy([body])
        assert float(e) >= 0, f"Expected non-negative energy, got {e}"

    def test_zero_for_body_at_rest(self):
        """A body with zero momentum must have zero kinetic energy."""
        body = _make_body(lin_momentum=(0, 0, 0), ang_momentum=(0, 0, 0))
        e = calc_kinetic_energy([body])
        assert float(e) == pytest.approx(0.0, abs=1e-10), \
            f"Energy of a body at rest should be 0, got {e}"

    def test_positive_for_moving_body(self):
        """A body with non-zero momentum must have positive energy."""
        body = _make_body(lin_momentum=(5, 0, 0), ang_momentum=(0, 0, 0))
        e = calc_kinetic_energy([body])
        assert float(e) > 0, f"Expected positive energy for a moving body, got {e}"

    def test_stays_scalar_during_simulation(self):
        """Energy must remain a finite scalar at every simulation step."""
        energies = _run_simulation(EulerMethod(), n_steps=10)
        for i, e in enumerate(energies):
            assert np.isscalar(e) or (isinstance(e, np.ndarray) and e.ndim == 0), \
                f"Step {i}: expected scalar, got {type(e).__name__}"
            assert np.isfinite(float(e)), \
                f"Step {i}: energy is not finite ({e})"
            assert float(e) >= 0, \
                f"Step {i}: energy is negative ({e})"


# ── ground-truth comparison ──────────────────────────────────────

class TestKineticEnergyGroundTruth:
    """Compare computed kinetic energy against known-good reference values."""

    @pytest.mark.parametrize("solver_name, solver, gt_values", [
        ("euler", EulerMethod(), GT_ENERGY_EULER),
        ("rk4",   RK4Method(),   GT_ENERGY_RK4),
    ])
    def test_energy_matches_ground_truth(self, solver_name, solver, gt_values):
        if any(v is None for v in gt_values):
            pytest.skip(
                f"Ground-truth values for '{solver_name}' not yet provided — "
                f"fill in GT_ENERGY_{solver_name.upper()} in test_kin_energy.py"
            )

        energies = _run_simulation(solver, n_steps=N_STEPS, dt=DT)

        for step_idx, gt_val in zip(GT_CHECK_STEPS, gt_values):
            actual = float(energies[step_idx])
            assert actual == pytest.approx(gt_val, rel=RTOL), \
                f"[{solver_name}] step {step_idx}: expected {gt_val}, got {actual}"
