"""Homework 2 starter diagnostic plugins."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

from .base import DiagnosticContext, DiagnosticPlugin, DiagnosticResult
from .homework_runtime import isolated_homework_imports, jsonable, load_submitted_module


class HW02KinEnergyTracePlugin:
    id = "hw02_kin_energy_trace"
    label = "HW2 kinetic-energy trace"
    timeout_sec = 15.0

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw02(context) and context.problem_id == "kin_energy"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            rows, error = _kin_energy_rows(context)
        artifacts = [
            _write_text(context, "kin_energy_trace.csv", _rows_to_csv(rows), "Kinetic energy trace CSV."),
            _write_text(context, "kin_energy_summary.md", _kin_energy_markdown(rows, error), "Kinetic energy diagnostic summary."),
        ]
        png = _line_plot_png(rows, "step", ["actual", "expected"], "Kinetic energy trace")
        if png:
            artifacts.append(_write_png(context, "kin_energy_trace.png", png, "Kinetic energy trace plot."))
        nonfinite = sum(1 for row in rows if not row.get("finite", True))
        max_abs = max((float(row.get("abs_error") or 0.0) for row in rows), default=0.0)
        summary = f"Kinetic energy diagnostic: {len(rows)} sample(s), max_abs_error={max_abs:.4g}."
        if error:
            summary = f"Kinetic energy diagnostic recorded error: {error}"
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary=summary,
            metrics={"samples": len(rows), "nonfinite_values": nonfinite, "max_abs_error": max_abs},
            artifacts=artifacts,
        )


class HW02ConstraintDriftPlugin:
    id = "hw02_constraint_drift"
    label = "HW2 constraint drift"
    timeout_sec = 15.0

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw02(context) and context.problem_id == "joints"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            rows, checks = _constraint_rows(context)
        artifacts = [
            _write_text(context, "constraint_drift.csv", _rows_to_csv(rows), "Constraint drift CSV."),
            _write_text(context, "constraint_drift_summary.md", _constraint_markdown(rows, checks), "Constraint drift summary."),
        ]
        png = _line_plot_png(rows, "step", ["drift"], "Constraint drift")
        if png:
            artifacts.append(_write_png(context, "constraint_drift.png", png, "Constraint drift plot."))
        errors = [item for item in checks if item.get("status") != "ok"]
        detail = ""
        if errors:
            first = errors[0]
            detail = f" First issue: {first.get('name')}: {first.get('error') or first.get('status')}."
        summary = f"Constraint drift diagnostic: {len(errors)} API/simulation issue(s).{detail}"
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary=summary,
            metrics={"issues": len(errors), "samples": len(rows)},
            artifacts=artifacts,
        )


class HW02PenaltyForcePlugin:
    id = "hw02_penalty_force"
    label = "HW2 penalty force"
    timeout_sec = 10.0

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw02(context) and context.problem_id == "penalty"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            report = _penalty_report(context)
        artifacts = [
            _write_text(
                context,
                "penalty_cases.json",
                json.dumps(jsonable(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                "Penalty force case report.",
            ),
            _write_text(context, "penalty_force_summary.md", _penalty_markdown(report), "Penalty force summary."),
        ]
        png = _penalty_plot_png(report)
        if png:
            artifacts.append(_write_png(context, "penalty_force.png", png, "Penalty normal/friction force plot."))
        issue_count = sum(1 for case in report["cases"] if case.get("error") or not case.get("finite", True))
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary=f"Penalty diagnostic: {len(report['cases'])} case(s), {issue_count} issue(s).",
            metrics={"cases": len(report["cases"]), "issues": issue_count},
            artifacts=artifacts,
        )


def default_plugins() -> list[DiagnosticPlugin]:
    return [HW02KinEnergyTracePlugin(), HW02ConstraintDriftPlugin(), HW02PenaltyForcePlugin()]


def _is_hw02(context: DiagnosticContext) -> bool:
    return context.homework_id == "02" or context.topic_slug == "02-dynamics"


def _write_text(context: DiagnosticContext, filename: str, text: str, description: str) -> dict[str, Any]:
    return context.artifact_store.write_text_artifact(
        context.student_id,
        context.problem_id,
        "diagnostic",
        filename,
        text if text.endswith("\n") else text + "\n",
        label=filename,
        description=description,
    )


def _write_png(context: DiagnosticContext, filename: str, data: bytes, description: str) -> dict[str, Any]:
    return context.artifact_store.write_bytes_artifact(
        context.student_id,
        context.problem_id,
        "diagnostic",
        filename,
        data,
        label=filename,
        description=description,
    )


def _kin_energy_rows(context: DiagnosticContext) -> tuple[list[dict[str, Any]], str | None]:
    import numpy as np

    rows = []
    error = None
    try:
        kin_mod = load_submitted_module(context, "kin_energy.py")
        callback = kin_mod.KinEnergyCallback(draw_graph=False)
        calc = callback.calc_kinetic_energy
        try:
            rows = _real_kin_energy_trace(context, calc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            expected = [0.0, 1.5, 6.0, 13.5]
            for step, value in enumerate(expected):
                actual = float(calc([_FakeBody(value)]))
                rows.append(_energy_row("fallback", step, actual, value))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        rows = [_energy_row("fallback", 0, float("nan"), 0.0)]
    for row in rows:
        row["finite"] = bool(np.isfinite(row["actual"]))
    return rows, error


def _real_kin_energy_trace(context: DiagnosticContext, calc: Any) -> list[dict[str, Any]]:
    import numpy as np
    from lib.basic_structs import Pose, Vec3
    from lib.phys.phys_objects import RigidBody
    from lib.phys.physics_world import PhysicsEngine

    solvers = load_submitted_module(context, "ode_solvers.py")

    def box_inertia(mass: float, extents: tuple[float, float, float]) -> Any:
        x, y, z = extents
        return mass / 12 * np.array([[y**2 + z**2, 0, 0], [0, x**2 + z**2, 0], [0, 0, x**2 + y**2]])

    def make_body() -> Any:
        inertia = box_inertia(1.0, (2, 12, 3))
        return RigidBody(
            name="rect",
            inv_mass=1.0,
            body_inertia_tensor_inv=np.linalg.inv(inertia),
            pose=Pose(),
            linear_momentum=Vec3(0, 0, 0),
            angular_momentum=Vec3(10, 10, 0),
        )

    rows = []
    expected_by_solver = {
        "euler": [50.1043, 50.3618, 51.4929, 52.8988],
        "rk4": [50.0754, 50.0754, 50.0754, 50.0754],
    }
    check_steps = [0, 9, 49, 99]
    for solver_name, solver in (("euler", solvers.EulerMethod()), ("rk4", solvers.RK4Method())):
        body = make_body()
        engine = PhysicsEngine(bodies=[body], dynamics_solver=solver, enable_collisions=False)
        t = 0.0
        values = []
        for _ in range(100):
            engine.step(t, t + 1e-2)
            values.append(float(calc(engine.bodies)))
            t += 1e-2
        for step, expected in zip(check_steps, expected_by_solver[solver_name]):
            rows.append(_energy_row(solver_name, step, values[step], expected))
    return rows


@dataclass
class _FakeBody:
    kinetic_energy: float


def _energy_row(solver: str, step: int, actual: float, expected: float) -> dict[str, Any]:
    import numpy as np

    abs_error = abs(actual - expected) if np.isfinite(actual) else float("inf")
    rel_error = abs_error / max(abs(expected), 1e-12) if np.isfinite(abs_error) else float("inf")
    return {
        "solver": solver,
        "step": int(step),
        "actual": float(actual),
        "expected": float(expected),
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
        "finite": bool(np.isfinite(actual)),
    }


def _constraint_rows(context: DiagnosticContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checks = []
    rows = []
    try:
        constraints_mod = load_submitted_module(context, "constraints.py")
        manager_mod = load_submitted_module(context, "constraints_manager.py")
    except Exception as exc:
        return [{"step": 0, "drift": float("nan"), "error": str(exc)}], [{"name": "import", "status": "error", "error": str(exc)}]

    joint_cls = getattr(constraints_mod, "BallAndSocketJoint", None)
    manager_cls = getattr(manager_mod, "ConstraintsManagerImpulseBased", None)
    for name, obj in (("BallAndSocketJoint", joint_cls), ("ConstraintsManagerImpulseBased", manager_cls)):
        checks.append({"name": name, "status": "ok" if callable(obj) else "missing"})

    try:
        joint = joint_cls()
        try:
            c = joint.get_C()
            import numpy as np

            drift = float(np.linalg.norm(c))
            rows.append({"step": 0, "drift": drift, "error": ""})
        except Exception as exc:
            checks.append({"name": "BallAndSocketJoint.get_C", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            rows.append({"step": 0, "drift": float("nan"), "error": f"{type(exc).__name__}: {exc}"})
        try:
            updates = joint.get_J_updates()
            checks.append({"name": "BallAndSocketJoint.get_J_updates", "status": "ok", "shape": str(type(updates))})
        except Exception as exc:
            checks.append({"name": "BallAndSocketJoint.get_J_updates", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    except Exception:
        try:
            rows.extend(_real_constraint_drift(context, joint_cls, manager_cls))
        except Exception as exc:
            checks.append({"name": "simulation", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            rows.append({"step": 0, "drift": float("nan"), "error": f"{type(exc).__name__}: {exc}"})
    return rows or [{"step": 0, "drift": float("nan"), "error": "no rows"}], checks


def _real_constraint_drift(context: DiagnosticContext, joint_cls: Any, manager_cls: Any) -> list[dict[str, Any]]:
    import numpy as np
    from lib.basic_structs import Pose, Vec3
    from lib.phys.constraints.constraints import BallAndSocketPoint
    from lib.phys.forces import GravityForce
    from lib.phys.phys_objects import RigidBody
    from lib.phys.physics_world import PhysicsEngine

    solvers = load_submitted_module(context, "ode_solvers.py")

    def box_inertia(mass: float, extents: tuple[float, float, float]) -> Any:
        x, y, z = extents
        return mass / 12 * np.array([[y**2 + z**2, 0, 0], [0, x**2 + z**2, 0], [0, 0, x**2 + y**2]])

    def body(name: str, pos=(0, 0, 0), lin=(0, 0, 0)) -> Any:
        inertia = box_inertia(1.0, (1, 6, 2))
        return RigidBody(
            name=name,
            inv_mass=1.0,
            body_inertia_tensor_inv=np.linalg.inv(inertia),
            pose=Pose.from_pq(p=list(pos)),
            linear_momentum=Vec3(*lin),
            angular_momentum=Vec3(0, 0, 0),
        )

    body_1 = body("rod1")
    body_2 = body("rod2", pos=(0, -6, 0), lin=(2, 0, 0))
    fixed = BallAndSocketPoint(body=body_1, body_fixed_point_local=Vec3(0, 3, 0))
    joint = joint_cls(body_1=body_1, body_2=body_2, anchor_point_body_1_local=Vec3(0, -3, 0))
    manager = manager_cls(bodies=[body_1, body_2], constraints={"fixed": fixed, "joint": joint}, beta_baumgarte=0.9)
    engine = PhysicsEngine(
        bodies=[body_1, body_2],
        dynamics_solver=solvers.RK4Method(),
        constraints_manager=manager,
        forces={"gravity": GravityForce(g_vector=Vec3(0, -9.81, 0))},
        enable_collisions=False,
    )
    rows = []
    t = 0.0
    for step in range(40):
        engine.step(t, t + 1e-2)
        joint.update(t + 1e-2, t + 1e-2)
        rows.append({"step": step, "drift": float(np.linalg.norm(joint.get_C())), "error": ""})
        t += 1e-2
    return rows


def _penalty_report(context: DiagnosticContext) -> dict[str, Any]:
    try:
        penalty_mod = load_submitted_module(context, "penalty.py")
        handler_cls = penalty_mod.PenaltyMethod
        handler = handler_cls(k_s=100.0, k_d=10.0, mu=0.5, k_drag=5.0)
    except Exception as exc:
        return {"cases": [{"name": "import", "error": f"{type(exc).__name__}: {exc}"}]}

    cases = []
    for name, collision in (
        ("normal_spring", _fake_collision(depth=0.2, normal=(0, 1, 0), vel=(0, 0, 0))),
        ("damping", _fake_collision(depth=0.0, normal=(0, 1, 0), vel=(0, 3, 0))),
        ("friction", _fake_collision(depth=0.2, normal=(0, 1, 0), vel=(2, 0, 0))),
    ):
        try:
            handler.step([collision], 0.0, 0.01)
            import numpy as np

            force = np.asarray(collision.obj_a.force_accumulator, dtype=float)
            torque = np.asarray(collision.obj_a.torque_accumulator, dtype=float)
            cases.append(
                {
                    "name": name,
                    "force": force.tolist(),
                    "torque": torque.tolist(),
                    "finite": bool(np.all(np.isfinite(force)) and np.all(np.isfinite(torque))),
                    "normal_direction_ok": bool(force[1] <= 1e-9) if name != "friction" else True,
                    "friction_clipped": bool(abs(force[0]) <= 0.5 * max(abs(force[1]), 1e-12) + 1e-6) if name == "friction" else None,
                    "error": "",
                }
            )
        except Exception as exc:
            cases.append({"name": name, "error": f"{type(exc).__name__}: {exc}", "finite": False})
    return {"cases": cases}


@dataclass
class _Body:
    force_accumulator: Any
    torque_accumulator: Any
    linear_momentum: Any
    angular_momentum: Any


@dataclass
class _Collision:
    obj_a: Any
    obj_b: Any
    point_a: Any
    point_b: Any
    normal: Any
    depth: float


def _fake_collision(depth: float, normal: tuple[float, float, float], vel: tuple[float, float, float]) -> _Collision:
    import numpy as np

    obj_a = _Body(np.zeros(3), np.zeros(3), np.asarray(vel, dtype=float), np.zeros(3))
    obj_b = _Body(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    return _Collision(obj_a, obj_b, np.zeros(3), np.zeros(3), np.asarray(normal, dtype=float), depth)


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = sorted({key for row in rows for key in row})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _line_plot_png(rows: list[dict[str, Any]], x_key: str, y_keys: list[str], title: str) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for key in y_keys:
            xs = [row.get(x_key, i) for i, row in enumerate(rows) if isinstance(row.get(key), (int, float))]
            ys = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
            if xs and ys:
                ax.plot(xs, ys, marker="o", label=key)
        ax.set_title(title)
        ax.set_xlabel(x_key)
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _kin_energy_markdown(rows: list[dict[str, Any]], error: str | None) -> str:
    lines = ["# Kinetic energy diagnostic", "", f"Samples: {len(rows)}"]
    if error:
        lines.append(f"Error: {error}")
    if rows:
        max_abs = max(float(row.get("abs_error") or 0.0) for row in rows)
        lines.append(f"Max absolute error: {max_abs:.6g}")
    return "\n".join(lines) + "\n"


def _constraint_markdown(rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> str:
    lines = ["# Constraint drift diagnostic", ""]
    for check in checks:
        line = f"- {check.get('name')}: {check.get('status')}"
        if check.get("error"):
            line += f" ({check['error']})"
        lines.append(line)
    numeric = [float(row["drift"]) for row in rows if isinstance(row.get("drift"), (int, float))]
    if numeric:
        lines.append(f"Max drift: {max(numeric):.6g}")
    return "\n".join(lines) + "\n"


def _penalty_markdown(report: dict[str, Any]) -> str:
    lines = ["# Penalty force diagnostic", ""]
    for case in report.get("cases", []):
        if case.get("error"):
            lines.append(f"- {case['name']}: {case['error']}")
        else:
            lines.append(f"- {case['name']}: force={case.get('force')}, torque={case.get('torque')}")
    return "\n".join(lines) + "\n"


def _penalty_plot_png(report: dict[str, Any]) -> bytes | None:
    rows = []
    for case in report.get("cases", []):
        force = case.get("force")
        if force:
            rows.append({"case": len(rows), "normal": abs(force[1]), "tangent": abs(force[0])})
    return _line_plot_png(rows, "case", ["normal", "tangent"], "Penalty force components")
