"""
Tests for Task 2.1 (ConstraintsManagerImpulseBased) and
Task 2.2 (BallAndSocketJoint).

Run with:  pytest tests/test_joints.py -v
"""
import numpy as np
import pytest

from lib.phys.phys_objects import RigidBody
from lib.phys.physics_world import PhysicsEngine
from lib.basic_structs import Vec3, Pose
from lib.phys.constraints.constraints import BallAndSocketPoint
from lib.phys.forces import GravityForce
from lib.utils import skew_sym
from solutions.ode_solvers import RK4Method
from solutions.constraints import BallAndSocketJoint
from solutions.constraints_manager import ConstraintsManagerImpulseBased


# ── helpers ───────────────────────────────────────────────────────────────────

def _box_inertia(mass: float, extents):
    x, y, z = extents
    return mass / 12 * np.array([
        [y**2 + z**2, 0, 0],
        [0, x**2 + z**2, 0],
        [0, 0, x**2 + y**2],
    ])


def _make_body(name, pos=(0, 0, 0), lin_momentum=(0, 0, 0), ang_momentum=(0, 0, 0),
               extents=(1, 6, 2), mass=1.0):
    inertia = _box_inertia(mass, extents)
    return RigidBody(
        name=name,
        inv_mass=1.0 / mass,
        body_inertia_tensor_inv=np.linalg.inv(inertia),
        pose=Pose.from_pq(p=list(pos)),
        linear_momentum=Vec3(*lin_momentum),
        angular_momentum=Vec3(*ang_momentum),
    )


ROD_LENGTH  = 6.0          # matches default extents y-dimension
DT          = 1e-2
N_STEPS     = 200
WARMUP      = 20           # steps to skip before checking drift
DRIFT_TOL   = 1e-2         # max |C| in steady state

# Anchor at the bottom tip of rod1 (body frame)
ANCHOR_1_LOCAL = Vec3(0, -ROD_LENGTH / 2, 0)
# rod2 starts directly below rod1 so the constraint is initially satisfied
BODY2_POS      = (0.0, -ROD_LENGTH, 0.0)


def _make_joint_pair():
    """Two rods connected by a BallAndSocketJoint with C=0 at t=0."""
    body_1 = _make_body('rod1')
    body_2 = _make_body('rod2', pos=BODY2_POS)
    joint = BallAndSocketJoint(
        body_1=body_1,
        body_2=body_2,
        anchor_point_body_1_local=ANCHOR_1_LOCAL,
    )
    joint.update(0.0, DT)
    return body_1, body_2, joint


# ── Task 2.1 — ConstraintsManagerImpulseBased ─────────────────────────────────

class TestImpulseManagerInterface:
    """calc_forces returns a correctly-shaped, finite array."""

    def _setup(self):
        body = _make_body('rod', ang_momentum=(1, 0, 0))
        constraint = BallAndSocketPoint(
            body=body,
            body_fixed_point_local=Vec3(0, ROD_LENGTH / 2, 0),
        )
        manager = ConstraintsManagerImpulseBased(
            bodies=[body],
            constraints={'fixed': constraint},
        )
        manager.update_constraints(0.0, DT)
        return body, manager

    def test_returns_array(self):
        _, manager = self._setup()
        F_C = manager.calc_forces(0.0, DT)
        assert F_C is not None, "calc_forces returned None"
        assert isinstance(F_C, np.ndarray), \
            f"Expected ndarray, got {type(F_C).__name__}"

    def test_correct_shape(self):
        body, manager = self._setup()
        F_C = manager.calc_forces(0.0, DT)
        assert F_C.shape == (body.state_p_size,), \
            f"Expected shape ({body.state_p_size},), got {F_C.shape}"

    def test_finite_values(self):
        _, manager = self._setup()
        F_C = manager.calc_forces(0.0, DT)
        assert np.all(np.isfinite(F_C)), \
            f"calc_forces returned non-finite values: {F_C}"

    def test_returns_none_with_no_constraints(self):
        body = _make_body('rod')
        manager = ConstraintsManagerImpulseBased(bodies=[body], constraints={})
        assert manager.calc_forces(0.0, DT) is None, \
            "calc_forces should return None when there are no constraints"


class TestImpulseManagerDrift:
    """Constraint error must stay small over a full simulation."""

    def _run(self, n_steps=N_STEPS, beta=0.9):
        body = _make_body('rod', ang_momentum=(2, 1, 0))
        constraint = BallAndSocketPoint(
            body=body,
            body_fixed_point_local=Vec3(0, ROD_LENGTH / 2, 0),
        )
        manager = ConstraintsManagerImpulseBased(
            bodies=[body],
            constraints={'fixed': constraint},
            beta_baumgarte=beta,
        )
        engine = PhysicsEngine(
            bodies=[body],
            dynamics_solver=RK4Method(),
            constraints_manager=manager,
            forces={'gravity': GravityForce(g_vector=Vec3(0, -9.81, 0))},
            enable_collisions=False,
        )
        errors = []
        t = 0.0
        for _ in range(n_steps):
            engine.step(t, t + DT)
            constraint.update(t + DT, t + DT)
            errors.append(float(np.linalg.norm(constraint.get_C())))
            t += DT
        return errors

    def test_does_not_diverge(self):
        errors = self._run()
        assert np.all(np.isfinite(errors)), \
            "Constraint error became non-finite during simulation"

    def test_drift_bounded_in_steady_state(self):
        errors = self._run()
        max_err = max(errors[WARMUP:])
        assert max_err < DRIFT_TOL, \
            f"Constraint drift too large: max |C| = {max_err:.5f} (tol={DRIFT_TOL})"


# ── Task 2.2 — BallAndSocketJoint.get_C ───────────────────────────────────────

class TestBallAndSocketGetC:

    def test_returns_length_3(self):
        _, _, joint = _make_joint_pair()
        C = joint.get_C()
        C = np.asarray(C)
        assert C.shape == (3,), \
            f"get_C must return a length-3 vector, got shape {C.shape}"

    def test_zero_when_satisfied(self):
        """Reference configuration has C = 0."""
        _, _, joint = _make_joint_pair()
        C = joint.get_C()
        np.testing.assert_allclose(C, np.zeros(3), atol=1e-5,
            err_msg=f"C should be 0 in the reference configuration, got {C}")

    def test_nonzero_when_violated(self):
        """Translating body_2 off the constraint must produce non-zero C."""
        body_1, body_2, joint = _make_joint_pair()
        body_2.pose = Pose.from_pq(p=list(np.array(BODY2_POS) + [0.5, 0.0, 0.0]))
        joint.update(0.0, DT)
        C = joint.get_C()
        assert np.linalg.norm(C) > 1e-4, \
            f"Expected non-zero C after displacement, got {C}"

    def test_value_equals_translation(self):
        """
        For a pure translation of body_2 (no rotation), C should equal
        the displacement vector.
        """
        delta = np.array([0.3, 0.1, -0.2])
        body_1, body_2, joint = _make_joint_pair()
        body_2.pose = Pose.from_pq(p=list(np.array(BODY2_POS) + delta))
        joint.update(0.0, DT)
        C = joint.get_C()
        np.testing.assert_allclose(np.asarray(C), delta, atol=1e-5,
            err_msg=f"For a pure translation by {delta}, C should equal the displacement; "
                    f"got {C}")

    def test_finite(self):
        _, _, joint = _make_joint_pair()
        C = joint.get_C()
        assert np.all(np.isfinite(C)), \
            f"get_C returned non-finite values: {C}"


# ── Task 2.2 — BallAndSocketJoint.get_J_updates ───────────────────────────────

class TestBallAndSocketGetJUpdates:

    def test_returns_dict(self):
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        assert isinstance(J, dict), \
            f"get_J_updates must return a dict, got {type(J).__name__}"

    def test_contains_both_bodies(self):
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        assert body_1.name in J, \
            f"Missing key '{body_1.name}' in get_J_updates result"
        assert body_2.name in J, \
            f"Missing key '{body_2.name}' in get_J_updates result"

    def test_jacobian_shapes(self):
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        assert J[body_1.name].shape == (3, 6), \
            f"J for '{body_1.name}' should be (3, 6), got {J[body_1.name].shape}"
        assert J[body_2.name].shape == (3, 6), \
            f"J for '{body_2.name}' should be (3, 6), got {J[body_2.name].shape}"

    def test_finite_values(self):
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        assert np.all(np.isfinite(J[body_1.name])), \
            "J for body_1 contains non-finite values"
        assert np.all(np.isfinite(J[body_2.name])), \
            "J for body_2 contains non-finite values"

    def test_translational_block_body1(self):
        """At identity rotation, translational block of J_1 must be -I."""
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        np.testing.assert_allclose(J[body_1.name][:, :3], -np.eye(3), atol=1e-5,
            err_msg="Translational block of J_1 should be -I")

    def test_translational_block_body2(self):
        """At identity rotation, translational block of J_2 must be +I."""
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        np.testing.assert_allclose(J[body_2.name][:, :3], np.eye(3), atol=1e-5,
            err_msg="Translational block of J_2 should be +I")

    def test_rotational_block_body1(self):
        """At identity rotation, rotational block of J_1 must be +[r_f1]_×."""
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        expected = skew_sym(joint.body_1_fixed_point_world)
        np.testing.assert_allclose(J[body_1.name][:, 3:], expected, atol=1e-5,
            err_msg="Rotational block of J_1 should be +[r_f1]_×")

    def test_rotational_block_body2(self):
        """At identity rotation, rotational block of J_2 must be -[r_f2]_×."""
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        expected = -skew_sym(joint.body_2_fixed_point_world)
        np.testing.assert_allclose(J[body_2.name][:, 3:], expected, atol=1e-5,
            err_msg="Rotational block of J_2 should be -[r_f2]_×")

    def test_jv_zero_at_rest(self):
        """J @ V must be 0 when both bodies are at rest."""
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        jv = J[body_1.name] @ np.zeros(6) + J[body_2.name] @ np.zeros(6)
        np.testing.assert_allclose(jv, np.zeros(3), atol=1e-5,
            err_msg="JV should be 0 when both bodies are at rest")

    def test_jv_captures_relative_velocity(self):
        """
        When body_2 has a pure linear velocity v, JV should equal v
        (the relative velocity of the anchor points).
        """
        body_1, body_2, joint = _make_joint_pair()
        J = joint.get_J_updates()
        v = np.array([1.0, 2.0, -0.5])
        V1 = np.zeros(6)
        V2 = np.concatenate([v, np.zeros(3)])
        jv = J[body_1.name] @ V1 + J[body_2.name] @ V2
        np.testing.assert_allclose(jv, v, atol=1e-5,
            err_msg=f"JV should equal body_2 velocity {v} when body_1 is at rest; got {jv}")


# ── Task 2.2 — Integration test ───────────────────────────────────────────────

class TestBallAndSocketSimulation:
    """Joint constraint error must stay small during a full simulation."""

    def _run(self, n_steps=N_STEPS):
        body_1 = _make_body('rod1')
        body_2 = _make_body('rod2', pos=BODY2_POS, lin_momentum=(2, 0, 0))

        fixed = BallAndSocketPoint(
            body=body_1,
            body_fixed_point_local=Vec3(0, ROD_LENGTH / 2, 0),
        )
        joint = BallAndSocketJoint(
            body_1=body_1,
            body_2=body_2,
            anchor_point_body_1_local=ANCHOR_1_LOCAL,
        )
        manager = ConstraintsManagerImpulseBased(
            bodies=[body_1, body_2],
            constraints={'fixed': fixed, 'joint': joint},
            beta_baumgarte=0.9,
        )
        engine = PhysicsEngine(
            bodies=[body_1, body_2],
            dynamics_solver=RK4Method(),
            constraints_manager=manager,
            forces={'gravity': GravityForce(g_vector=Vec3(0, -9.81, 0))},
            enable_collisions=False,
        )
        errors = []
        t = 0.0
        for _ in range(n_steps):
            engine.step(t, t + DT)
            joint.update(t + DT, t + DT)
            errors.append(float(np.linalg.norm(joint.get_C())))
            t += DT
        return errors

    def test_does_not_diverge(self):
        errors = self._run()
        assert np.all(np.isfinite(errors)), \
            "Joint constraint error became non-finite during simulation"

    def test_drift_bounded_in_steady_state(self):
        errors = self._run()
        max_err = max(errors[WARMUP:])
        assert max_err < DRIFT_TOL, \
            f"Joint constraint drift too large: max |C| = {max_err:.5f} (tol={DRIFT_TOL})"
