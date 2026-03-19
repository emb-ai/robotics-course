"""
Tests for Task 3.1 (PenaltyMethod — normal contact force) and
Task 3.2 (PenaltyMethod — friction).

Run with:  pytest tests/test_penalty.py -v
"""
import numpy as np
import pytest

from lib.phys.phys_objects import RigidBody, Particle
from lib.phys.physics_world import PhysicsEngine
from lib.basic_structs import Vec3, Pose
from lib.phys.collisions.collision_detector import Collision, FCLCollisionDetector
from lib.phys.collisions.colliders import BoxCollider
from lib.phys.forces import GravityForce
from solutions.ode_solvers import RK4Method
from solutions.penalty import PenaltyMethod


# ── helpers ───────────────────────────────────────────────────────────────────

def _box_inertia(mass, extents):
    x, y, z = extents
    return mass / 12 * np.array([
        [y**2 + z**2, 0, 0],
        [0, x**2 + z**2, 0],
        [0, 0, x**2 + y**2],
    ])


def _make_rigid_body(name, pos=(0, 0, 0), vel=(0, 0, 0), omega=(0, 0, 0),
                     extents=(2, 2, 2), mass=1.0):
    """RigidBody with velocity and angular velocity set via momenta."""
    inertia = _box_inertia(mass, extents)
    lin_mom = np.array(vel, dtype=float) * mass
    ang_mom = inertia @ np.array(omega, dtype=float)
    # Explicitly create fresh accumulators — RigidBody's Vec3 defaults are
    # mutable and shared across instances if not overridden here.
    return RigidBody(
        name=name,
        inv_mass=1.0 / mass,
        body_inertia_tensor_inv=np.linalg.inv(inertia),
        pose=Pose.from_pq(p=list(pos)),
        linear_momentum=Vec3(*lin_mom),
        angular_momentum=Vec3(*ang_mom),
        force_accumulator=Vec3(0, 0, 0),
        torque_accumulator=Vec3(0, 0, 0),
    )


def _make_static_body(name, pos=(0, 0, 0), extents=(2, 2, 2)):
    """RigidBody with inv_mass=0 (static wall / floor)."""
    body = _make_rigid_body(name, pos=pos, extents=extents)
    body.inv_mass = 0.0
    body.body_inertia_tensor_inv = np.zeros((3, 3))
    return body


def _make_collision(depth=0.1, normal=(0, 1, 0),
                    vel_a=(0, 0, 0), omega_a=(0, 0, 0), point_a=(0, 0, 0),
                    vel_b=(0, 0, 0), omega_b=(0, 0, 0), point_b=(0, 0, 0),
                    mass_a=1.0, static_b=True):
    obj_a = _make_rigid_body('a', vel=vel_a, omega=omega_a, mass=mass_a)
    if static_b:
        obj_b = _make_static_body('b')
    else:
        obj_b = _make_rigid_body('b', vel=vel_b, omega=omega_b)
    return obj_a, obj_b, Collision(
        obj_a=obj_a, obj_b=obj_b,
        point_a=Vec3(*point_a), point_b=Vec3(*point_b),
        normal=Vec3(*normal), depth=float(depth),
    )


# Use simple parameters to make expected values easy to compute
K_S    = 100.0
K_D    = 10.0
MU     = 0.5
K_DRAG = 5.0

DT = 0.01


# ── Task 3.1 — Normal contact force ──────────────────────────────────────────

class TestNormalForceInterface:

    def test_no_collisions_leaves_accumulator_zero(self):
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        body = _make_rigid_body('a')
        handler.step([], 0.0, DT)
        np.testing.assert_array_equal(body.force_accumulator, np.zeros(3))

    def test_returns_none(self):
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        obj_a, obj_b, col = _make_collision()
        result = handler.step([col], 0.0, DT)
        assert result is None

    def test_finite_output(self):
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        obj_a, obj_b, col = _make_collision(depth=0.1, vel_a=(1, 2, 3))
        handler.step([col], 0.0, DT)
        assert np.all(np.isfinite(obj_a.force_accumulator))
        assert np.all(np.isfinite(obj_a.torque_accumulator))


class TestNormalForceMagnitudeAndDirection:

    def test_zero_depth_zero_velocity_zero_normal_force(self):
        """No penetration, no relative velocity → zero normal force."""
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        obj_a, obj_b, col = _make_collision(depth=0.0, vel_a=(0, 0, 0))
        handler.step([col], 0.0, DT)
        # only spring + damping → both zero; friction also zero (max_friction=0)
        np.testing.assert_allclose(obj_a.force_accumulator, np.zeros(3), atol=1e-6)

    def test_spring_term_magnitude(self):
        """depth > 0, zero velocity → |force_n| = k_s * depth."""
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        depth = 0.2
        normal = (0.0, 1.0, 0.0)
        obj_a, obj_b, col = _make_collision(depth=depth, normal=normal,
                                            vel_a=(0, 0, 0))
        handler.step([col], 0.0, DT)
        # friction is also zero (max_friction = mu * k_s * depth, v_rel = 0)
        expected = np.array([0.0, -K_S * depth, 0.0])
        np.testing.assert_allclose(obj_a.force_accumulator, expected, atol=1e-5)

    def test_spring_term_direction(self):
        """Normal force must point in -normal direction."""
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        normal = (1.0, 0.0, 0.0)
        obj_a, obj_b, col = _make_collision(depth=0.1, normal=normal,
                                            vel_a=(0, 0, 0))
        handler.step([col], 0.0, DT)
        F = np.array(obj_a.force_accumulator)
        # force should be anti-parallel to normal
        assert F[0] < 0, f"Force component along normal should be negative, got {F}"
        assert abs(F[1]) < 1e-5 and abs(F[2]) < 1e-5

    def test_damping_term_along_normal(self):
        """
        Zero depth, relative velocity along normal → force_n = -k_d * v_rel_n * normal.
        Friction is disabled (mu=0, k_drag=0) to isolate the damping term.
        """
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=0.0, k_drag=0.0)
        normal = (0.0, 1.0, 0.0)
        v_rel_n = 3.0          # obj_a moving in +y (into obj_b)
        obj_a, obj_b, col = _make_collision(depth=0.0, normal=normal,
                                            vel_a=(0, v_rel_n, 0))
        handler.step([col], 0.0, DT)
        expected_n = np.array([0.0, -K_D * v_rel_n, 0.0])
        np.testing.assert_allclose(obj_a.force_accumulator, expected_n, atol=1e-5,
            err_msg="Damping-only force should be -k_d * v_rel_n * normal")

    def test_combined_spring_and_damping(self):
        """depth > 0 and v_rel along normal → force_n = -(k_s*d + k_d*v_n) * normal.
        Friction is disabled (mu=0, k_drag=0) to isolate the normal force."""
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=0.0, k_drag=0.0)
        normal = (0.0, 1.0, 0.0)
        depth = 0.15
        v_n = 2.5
        obj_a, obj_b, col = _make_collision(depth=depth, normal=normal,
                                            vel_a=(0, v_n, 0))
        handler.step([col], 0.0, DT)
        module = K_S * depth + K_D * v_n
        expected_n = np.array([0.0, -module, 0.0])
        np.testing.assert_allclose(obj_a.force_accumulator, expected_n, atol=1e-4,
            err_msg=f"Expected force_n = {expected_n}, got {obj_a.force_accumulator}")

    def test_opposing_velocity_reduces_force(self):
        """
        When obj_a moves away (v_rel_n < 0), damping opposes the spring,
        so the total normal force magnitude is smaller than k_s * depth alone.
        """
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        depth = 0.1
        normal = (0.0, 1.0, 0.0)
        # obj_a moving away from obj_b: v_rel_n < 0
        obj_a_away, _, col_away = _make_collision(depth=depth, normal=normal,
                                                  vel_a=(0, -1.0, 0))
        obj_a_rest, _, col_rest = _make_collision(depth=depth, normal=normal,
                                                  vel_a=(0, 0.0, 0))
        handler.step([col_away], 0.0, DT)
        handler.step([col_rest], 0.0, DT)
        # spring-only normal force: k_s * depth
        f_rest = abs(float(obj_a_rest.force_accumulator[1]))
        f_away = abs(float(obj_a_away.force_accumulator[1]))
        assert f_away < f_rest, (
            f"Moving away should reduce normal force: got {f_away:.3f} vs {f_rest:.3f}")


class TestNormalForceAppliedToCorrectBody:

    def test_force_applied_only_to_obj_a(self):
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        obj_a, obj_b, col = _make_collision(depth=0.1, vel_a=(0, 1, 0))
        handler.step([col], 0.0, DT)
        # obj_a should be modified
        assert np.linalg.norm(obj_a.force_accumulator) > 0, \
            "Force should be applied to obj_a"
        # obj_b is static but its accumulator should not be touched
        np.testing.assert_array_equal(obj_b.force_accumulator, np.zeros(3),
            err_msg="Force should NOT be applied to obj_b")

    def test_multiple_collisions_accumulate(self):
        """Forces from multiple collisions must all be added to obj_a."""
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        obj_a1, _, col1 = _make_collision(depth=0.1, normal=(1, 0, 0))
        obj_a2, _, col2 = _make_collision(depth=0.1, normal=(0, 1, 0))
        # Give each collision the same obj_a
        col2.obj_a = col1.obj_a
        handler.step([col1, col2], 0.0, DT)
        F = np.array(col1.obj_a.force_accumulator)
        # Both x and y components should be nonzero (from two collisions)
        assert abs(F[0]) > 1e-5, "x-component should come from collision 1"
        assert abs(F[1]) > 1e-5, "y-component should come from collision 2"


# ── Torque ────────────────────────────────────────────────────────────────────

class TestTorque:

    def test_torque_equals_r_cross_force(self):
        """τ_n = point_a × force_n."""
        handler = PenaltyMethod(k_s=K_S, k_d=0.0, mu=0.0, k_drag=0.0)
        depth = 0.2
        normal = (0.0, 1.0, 0.0)
        point_a = (1.0, 0.0, 0.0)
        obj_a, _, col = _make_collision(depth=depth, normal=normal,
                                        point_a=point_a, vel_a=(0, 0, 0))
        handler.step([col], 0.0, DT)
        force_n = np.array([0.0, -K_S * depth, 0.0])
        expected_torque = np.cross(np.array(point_a), force_n)
        np.testing.assert_allclose(obj_a.torque_accumulator, expected_torque, atol=1e-5,
            err_msg=f"Expected torque {expected_torque}, got {obj_a.torque_accumulator}")

    def test_zero_torque_at_com(self):
        """Contact exactly at COM (point_a = 0) → zero torque."""
        handler = PenaltyMethod(k_s=K_S, k_d=0.0, mu=0.0, k_drag=0.0)
        obj_a, _, col = _make_collision(depth=0.1, normal=(0, 1, 0),
                                        point_a=(0, 0, 0))
        handler.step([col], 0.0, DT)
        np.testing.assert_allclose(obj_a.torque_accumulator, np.zeros(3), atol=1e-6,
            err_msg="Contact at COM should produce zero torque")


# ── Angular velocity contribution to contact velocity ─────────────────────────

class TestAngularVelocityContribution:

    def test_omega_a_affects_contact_velocity(self):
        """
        obj_a at rest (v_com = 0) but spinning: the contact point velocity
        is omega × point_a, which should contribute to v_rel and hence to the force.
        """
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=0.0, k_drag=0.0)
        normal   = (1.0, 0.0, 0.0)
        point_a  = (0.0, 1.0, 0.0)
        # omega = (0,0,-1): v_contact = (0,0,-1) × (0,1,0) = (1,0,0) along normal
        omega_a  = (0.0, 0.0, -1.0)

        obj_a_spin,  _, col_spin  = _make_collision(normal=normal, point_a=point_a,
                                                    omega_a=omega_a, depth=0.0)
        obj_a_still, _, col_still = _make_collision(normal=normal, point_a=point_a,
                                                    omega_a=(0, 0, 0), depth=0.0)
        handler.step([col_spin],  0.0, DT)
        handler.step([col_still], 0.0, DT)

        f_spin  = np.linalg.norm(obj_a_spin.force_accumulator)
        f_still = np.linalg.norm(obj_a_still.force_accumulator)
        assert f_spin > f_still, (
            f"Spinning body should produce larger contact force due to angular velocity; "
            f"got f_spin={f_spin:.4f}, f_still={f_still:.4f}")

    def test_omega_b_affects_contact_velocity(self):
        """
        obj_b spinning: its contact point velocity reduces v_rel,
        leading to a different (smaller) force than when obj_b is at rest.
        """
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=0.0, k_drag=0.0)
        normal  = (1.0, 0.0, 0.0)
        point_b = (0.0, 1.0, 0.0)
        # obj_a moves in +normal direction; obj_b spinning opposes v_rel
        # omega_b = (0,0,1): v_b_contact = (0,0,1) × (0,1,0) = (-1,0,0) along -normal
        # v_rel = v_a - v_b_contact → larger v_rel_n → larger damping force

        obj_a1, _, col_spin  = _make_collision(normal=normal, point_b=point_b,
                                               vel_a=(2, 0, 0), depth=0.0,
                                               static_b=False, omega_b=(0, 0, 1))
        obj_a2, _, col_still = _make_collision(normal=normal, point_b=point_b,
                                               vel_a=(2, 0, 0), depth=0.0,
                                               static_b=False, omega_b=(0, 0, 0))
        handler.step([col_spin],  0.0, DT)
        handler.step([col_still], 0.0, DT)

        f_spin  = np.linalg.norm(obj_a1.force_accumulator)
        f_still = np.linalg.norm(obj_a2.force_accumulator)
        assert not np.isclose(f_spin, f_still, rtol=1e-3), (
            "obj_b angular velocity should change the contact force")


# ── Task 3.2 — Friction ───────────────────────────────────────────────────────

class TestFriction:

    def test_viscous_regime_small_tangential_velocity(self):
        """
        When |k_drag * v_rel| < mu * |F_n|, friction = -k_drag * v_rel.
        """
        handler = PenaltyMethod(k_s=K_S, k_d=0.0, mu=MU, k_drag=K_DRAG)
        normal  = (0.0, 1.0, 0.0)
        depth   = 0.5              # large depth → large F_n → wide viscous regime
        # purely tangential velocity (no normal component)
        v_tang = 0.1               # small enough to stay in viscous regime
        obj_a, _, col = _make_collision(depth=depth, normal=normal,
                                        vel_a=(v_tang, 0.0, 0.0))
        handler.step([col], 0.0, DT)

        force_n_mag   = K_S * depth
        max_friction  = MU * force_n_mag
        friction_pred = K_DRAG * v_tang
        assert friction_pred < max_friction, "Test setup error: not in viscous regime"

        # total x-force should be -k_drag * v_tang (viscous friction)
        np.testing.assert_allclose(float(obj_a.force_accumulator[0]),
                                   -K_DRAG * v_tang, atol=1e-5,
                                   err_msg="Viscous friction should be -k_drag * v_tang")

    def test_coulomb_regime_large_tangential_velocity(self):
        """|friction| is clamped to mu * |F_n| for large tangential velocity."""
        handler = PenaltyMethod(k_s=K_S, k_d=0.0, mu=MU, k_drag=K_DRAG)
        normal = (0.0, 1.0, 0.0)
        depth  = 0.1
        # Very large tangential velocity → clearly Coulomb regime
        obj_a, _, col = _make_collision(depth=depth, normal=normal,
                                        vel_a=(1000.0, 0.0, 0.0))
        handler.step([col], 0.0, DT)

        force_n_mag  = K_S * depth
        max_friction = MU * force_n_mag
        total_force  = np.array(obj_a.force_accumulator)
        friction_mag = abs(float(total_force[0]))  # y is normal, x is tangential

        np.testing.assert_allclose(friction_mag, max_friction, rtol=1e-4,
            err_msg=f"Friction should be clamped to {max_friction}, got {friction_mag}")

    def test_friction_direction_opposes_velocity(self):
        """Friction must act in the direction opposite to the tangential velocity."""
        handler = PenaltyMethod(k_s=K_S, k_d=0.0, mu=MU, k_drag=K_DRAG)
        normal = (0.0, 1.0, 0.0)
        # obj_a sliding in +x direction; friction should be in -x direction
        obj_a, _, col = _make_collision(depth=0.1, normal=normal,
                                        vel_a=(5.0, 0.0, 0.0))
        handler.step([col], 0.0, DT)
        fx = float(obj_a.force_accumulator[0])
        assert fx < 0, f"Friction should oppose +x velocity, but got fx={fx}"

    def test_zero_normal_force_means_zero_friction(self):
        """
        If depth=0 and v_rel_n=0 (so force_n=0), friction is clamped to
        mu * 0 = 0, regardless of tangential velocity.
        """
        handler = PenaltyMethod(k_s=K_S, k_d=K_D, mu=MU, k_drag=K_DRAG)
        normal = (0.0, 1.0, 0.0)
        # tangential velocity only, no penetration, no normal velocity
        obj_a, _, col = _make_collision(depth=0.0, normal=normal,
                                        vel_a=(100.0, 0.0, 0.0))
        handler.step([col], 0.0, DT)
        np.testing.assert_allclose(obj_a.force_accumulator, np.zeros(3), atol=1e-6,
            err_msg="Zero normal force must produce zero friction (Coulomb clamping)")

    def test_friction_torque(self):
        """τ_friction = point_a × friction."""
        # Use zero normal velocity to isolate friction torque more easily
        handler = PenaltyMethod(k_s=K_S, k_d=0.0, mu=1.0, k_drag=K_DRAG)
        normal  = (0.0, 1.0, 0.0)
        depth   = 0.5
        point_a = (1.0, 0.0, 0.0)
        v_tang  = 0.1              # viscous regime
        obj_a, _, col = _make_collision(depth=depth, normal=normal,
                                        point_a=point_a, vel_a=(v_tang, 0.0, 0.0))
        handler.step([col], 0.0, DT)

        friction_force = np.array([-K_DRAG * v_tang, 0.0, 0.0])
        force_n        = np.array([0.0, -K_S * depth, 0.0])
        expected_torque = np.cross(np.array(point_a), force_n + friction_force)
        np.testing.assert_allclose(obj_a.torque_accumulator, expected_torque, atol=1e-4,
            err_msg=f"Expected torque {expected_torque}, got {obj_a.torque_accumulator}")


# ── Integration test ──────────────────────────────────────────────────────────

class TestPenaltyIntegration:
    """
    A box falls under gravity onto a static floor.
    Verify the simulation stays finite and the box bounces.
    """

    def _run(self, n_steps=400, dt=0.01):
        box_ext   = (2.0, 2.0, 2.0)
        floor_ext = (20.0, 0.2, 20.0)
        # Box center at y=3 (bottom at y=2), floor top at y=0 (center at y=-0.1)
        box = RigidBody(
            name='box',
            inv_mass=1.0,
            body_inertia_tensor_inv=np.linalg.inv(_box_inertia(1.0, box_ext)),
            pose=Pose.from_pq(p=[0.0, 3.0, 0.0]),
            linear_momentum=Vec3(0, 0, 0),
            angular_momentum=Vec3(0, 0, 0),
            collider=BoxCollider.create(box_ext, Pose.from_pq(p=[0.0, 3.0, 0.0])),
        )
        floor = RigidBody(
            name='floor',
            inv_mass=0.0,
            body_inertia_tensor_inv=np.zeros((3, 3)),
            pose=Pose.from_pq(p=[0.0, -0.1, 0.0]),
            linear_momentum=Vec3(0, 0, 0),
            angular_momentum=Vec3(0, 0, 0),
            collider=BoxCollider.create(floor_ext, Pose.from_pq(p=[0.0, -0.1, 0.0])),
        )
        handler   = PenaltyMethod(k_s=1e2, k_d=1.0, mu=0.5, k_drag=1.0)
        detector  = FCLCollisionDetector(bodies=[box, floor])
        engine    = PhysicsEngine(
            bodies=[box, floor],
            dynamics_solver=RK4Method(),
            forces={'gravity': GravityForce(g_vector=Vec3(0, -9.81, 0))},
            constraints_manager=None,
            collision_detector=detector,
            contact_forces_handler=handler,
            enable_collisions=True,
        )
        positions = []
        t = 0.0
        for _ in range(n_steps):
            engine.step(t, t + dt)
            positions.append(float(box.pose.p[1]))
            t += dt
        return positions

    def test_simulation_stays_finite(self):
        positions = self._run()
        assert np.all(np.isfinite(positions)), \
            "Box y-position became non-finite during simulation"

    def test_box_does_not_tunnel_through_floor(self):
        """Box should never fall far below the floor surface (y = 0)."""
        positions = self._run()
        min_y = min(positions)
        assert min_y > -2.0, \
            f"Box tunnelled through the floor: min y = {min_y:.3f}"

    def test_box_bounces(self):
        """After hitting the floor, the box should return upward at some point."""
        positions = self._run(n_steps=400)
        # The box starts at y=3 and falls. After the bounce, it should be
        # above y=0 at some point in the second half of the simulation.
        second_half = positions[200:]
        assert max(second_half) > 0.5, \
            f"Box did not bounce: max y in second half = {max(second_half):.3f}"
