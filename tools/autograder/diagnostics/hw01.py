"""Homework 1 diagnostic plugins."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .base import DiagnosticContext, DiagnosticPlugin, DiagnosticResult
from .homework_runtime import (
    call_with_timeout,
    hidden_dir,
    homework_dir,
    isolated_homework_imports,
    jsonable,
    load_module_from_path,
    load_reference_module,
    load_submitted_module,
)


HW01_TOPIC = "01-intro-and-kinematics"
SO101_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class HW01BeadsComparePlugin:
    id = "hw01_beads_compare"
    label = "HW1 beads radius comparison"
    timeout_sec = 30.0

    def __init__(self, case_timeout_sec: float = 3.0):
        self.case_timeout_sec = case_timeout_sec

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw01(context) and context.problem_id == "beads"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            try:
                student_mod = load_submitted_module(context, "beads.py")
                reference_mod = load_reference_module(context, "beads.py")
            except Exception as exc:
                return _simple_result(context, self.id, "beads_summary.md", f"Beads diagnostic skipped: {exc}")

            student_func = getattr(student_mod, "optimal_bead_config", None)
            reference_func = getattr(reference_mod, "optimal_bead_config", None)
            if not callable(student_func) or not callable(reference_func):
                return _simple_result(
                    context,
                    self.id,
                    "beads_summary.md",
                    "Beads diagnostic: missing `optimal_bead_config` in submitted or reference file.",
                )

            lib = _try_import("lib.beads")
            cases = _beads_cases(context)
            results = [
                _score_beads_case(index, source, lengths, student_func, reference_func, lib, self.case_timeout_sec)
                for index, source, lengths in cases
            ]
            worst = max(results, key=lambda item: item["score"])

            artifacts = [
                _write_json(context, "beads", "beads_case.json", worst, "Worst beads diagnostic case."),
                _write_text(context, "beads", "beads_summary.md", _beads_markdown(worst), "Beads diagnostic summary."),
            ]
            png = _beads_plot_png(worst, lib)
            if png:
                artifacts.append(
                    context.artifact_store.write_bytes_artifact(
                        context.student_id,
                        context.problem_id,
                        "diagnostic",
                        "beads_radius.png",
                        png,
                        label="beads_radius.png",
                        description="Beads student/reference radius comparison.",
                    )
                )
            summary = _beads_summary_line(worst)
            return DiagnosticResult(
                plugin_id=self.id,
                problem_id=context.problem_id,
                status="ok",
                summary=summary,
                metrics={
                    "delta": worst.get("delta"),
                    "student_radius": worst.get("student_radius"),
                    "reference_radius": worst.get("reference_radius"),
                    "violations": len(worst.get("violations") or []),
                },
                artifacts=artifacts,
            )


class HW01BroomRacingTrajectoryPlugin:
    id = "hw01_broom_racing_trajectory"
    label = "HW1 broom trajectory comparison"
    timeout_sec = 45.0

    def __init__(self, case_timeout_sec: float = 8.0):
        self.case_timeout_sec = case_timeout_sec

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw01(context) and context.problem_id == "broom_racing"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            try:
                student_mod = load_submitted_module(context, "broom_racing.py")
                reference_mod = load_reference_module(context, "broom_racing.py")
            except Exception as exc:
                return _simple_result(context, self.id, "broom_summary.md", f"Broom diagnostic skipped: {exc}")

            missing = [
                name
                for name in ("gate_pass", "catch_snitch", "catch_ball_and_gate")
                if not callable(getattr(student_mod, name, None))
            ]
            if missing:
                return _simple_result(
                    context,
                    self.id,
                    "broom_summary.md",
                    f"Broom diagnostic: missing required function(s): {', '.join(missing)}.",
                )

            lib = _broom_lib()
            cases = _broom_cases(context)
            results = [
                _score_broom_case(index, case, student_mod, reference_mod, lib, self.case_timeout_sec)
                for index, case in enumerate(cases)
            ]
            worst = _worst_broom_by_task(results)
            payload = {"worst_cases": worst, "case_count": len(cases)}
            artifacts = [
                _write_json(context, "broom_racing", "broom_worst_cases.json", payload, "Worst broom cases by task."),
                _write_text(
                    context,
                    "broom_racing",
                    "broom_summary.md",
                    _broom_markdown(worst),
                    "Broom racing diagnostic summary.",
                ),
            ]
            for item in worst:
                png = _broom_plot_png(item, student_mod, reference_mod, lib, self.case_timeout_sec)
                if png:
                    artifacts.append(
                        context.artifact_store.write_bytes_artifact(
                            context.student_id,
                            context.problem_id,
                            "diagnostic",
                            f"broom_{item['task']}_trajectory.png",
                            png,
                            label=f"{item['task']}_trajectory.png",
                            description=f"Broom trajectory plot for {item['task']}.",
                        )
                    )
                trace_png = _broom_trace_plot_png(item)
                if trace_png:
                    artifacts.append(
                        context.artifact_store.write_bytes_artifact(
                            context.student_id,
                            context.problem_id,
                            "diagnostic",
                            f"broom_{item['task']}_curvature_pitch.png",
                            trace_png,
                            label=f"{item['task']}_curvature_pitch.png",
                            description=f"Broom curvature and pitch trace for {item['task']}.",
                        )
                    )
            failed = [item for item in worst if item.get("error") or item.get("constraint_errors")]
            summary = f"Broom diagnostic: {len(worst)} task(s), {len(failed)} with errors or constraint violations."
            return DiagnosticResult(
                plugin_id=self.id,
                problem_id=context.problem_id,
                status="ok",
                summary=summary,
                metrics={"tasks": len(worst), "problematic_tasks": len(failed)},
                artifacts=artifacts,
            )


class HW01SO101IKDebugPlugin:
    id = "hw01_so101_ik_debug"
    label = "HW1 SO101 IK failure classification"
    timeout_sec = 30.0

    def __init__(self, case_timeout_sec: float = 3.0):
        self.case_timeout_sec = case_timeout_sec

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw01(context) and context.problem_id == "so101_ik"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            try:
                student_mod = load_submitted_module(context, "so101_ik.py")
            except Exception as exc:
                return _simple_result(context, self.id, "so101_summary.md", f"SO101 diagnostic skipped: {exc}")

            cases = _so101_cases(context)
            failures = []
            rows = []
            for index, source, pose, analytical_expected, numerical_expected in cases:
                for solver_name, expected, func_name in (
                    ("numerical", numerical_expected, "numerical_ik_so101_downturned"),
                    ("analytical", analytical_expected, "analytical_ik_so101_downturned"),
                ):
                    func = getattr(student_mod, func_name, None)
                    record = _classify_so101_solver(index, source, pose, solver_name, expected, func, self.case_timeout_sec)
                    rows.append(record)
                    if record["classification"] != "ok":
                        failures.append(record)
                formula_record = _classify_so101_formula(index, source, pose, student_mod)
                rows.append(formula_record)
                if formula_record["classification"] != "ok":
                    failures.append(formula_record)

            csv_text = _dicts_to_csv(rows)
            artifacts = [
                _write_json(context, "so101_ik", "so101_failures.json", {"failures": failures}, "SO101 failures."),
                _write_text(context, "so101_ik", "so101_residuals.csv", csv_text, "SO101 residual/classification table."),
                _write_text(context, "so101_ik", "so101_summary.md", _so101_markdown(failures), "SO101 diagnostic summary."),
            ]
            png = _so101_plot_png(rows)
            if png:
                artifacts.append(
                    context.artifact_store.write_bytes_artifact(
                        context.student_id,
                        context.problem_id,
                        "diagnostic",
                        "so101_residuals.png",
                        png,
                        label="so101_residuals.png",
                        description="SO101 residual and classification plot.",
                    )
                )
            numerical_failures = sum(1 for item in failures if item["solver"] == "numerical")
            analytical_failures = sum(1 for item in failures if item["solver"] in {"analytical", "formula"})
            return DiagnosticResult(
                plugin_id=self.id,
                problem_id=context.problem_id,
                status="ok",
                summary=f"SO101 diagnostic: numerical={numerical_failures}, analytical/formula={analytical_failures} issue(s).",
                metrics={"failures": len(failures), "numerical_failures": numerical_failures},
                artifacts=artifacts,
            )


def default_plugins() -> list[DiagnosticPlugin]:
    return [
        HW01BeadsComparePlugin(),
        HW01BroomRacingTrajectoryPlugin(),
        HW01SO101IKDebugPlugin(),
    ]


def _is_hw01(context: DiagnosticContext) -> bool:
    return context.homework_id == "01" or context.topic_slug == HW01_TOPIC


def _try_import(name: str) -> Any | None:
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


def _write_json(context: DiagnosticContext, problem_id: str, filename: str, payload: Any, description: str) -> dict[str, Any]:
    return context.artifact_store.write_text_artifact(
        context.student_id,
        problem_id,
        "diagnostic",
        filename,
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        label=filename,
        description=description,
    )


def _write_text(context: DiagnosticContext, problem_id: str, filename: str, text: str, description: str) -> dict[str, Any]:
    return context.artifact_store.write_text_artifact(
        context.student_id,
        problem_id,
        "diagnostic",
        filename,
        text if text.endswith("\n") else text + "\n",
        label=filename,
        description=description,
    )


def _simple_result(context: DiagnosticContext, plugin_id: str, filename: str, summary: str) -> DiagnosticResult:
    artifact = _write_text(context, context.problem_id, filename, f"# Diagnostic\n\n{summary}\n", summary)
    return DiagnosticResult(
        plugin_id=plugin_id,
        problem_id=context.problem_id,
        status="skipped" if "skipped" in summary.lower() else "ok",
        summary=summary,
        artifacts=[artifact],
    )


def _beads_cases(context: DiagnosticContext) -> list[tuple[int, str, Any]]:
    np = __import__("numpy")
    cases: list[tuple[int, str, Any]] = [
        (0, "public", np.array([2.0, 2.0, 2.0])),
        (1, "public", np.array([2.0, 2.0, 3.0, 2.0, 2.0])),
        (2, "public", np.array([2.0, 2.0, 3.0, 2.0, 2.0, 3.0])),
        (3, "public", np.full(30, 3.0)),
        (4, "public", np.random.Generator(np.random.PCG64(0xBAD_5EED)).uniform(2.0, 4.0, size=30)),
        (5, "public", np.random.Generator(np.random.PCG64(0x600D_5EED)).uniform(2.0, 4.0, size=30)),
    ]
    hdir = hidden_dir(context)
    hidden_path = hdir / "beads.py" if hdir else None
    if hidden_path and hidden_path.is_file():
        try:
            hidden_mod = load_module_from_path("_diagnostic_hidden_beads", hidden_path)
            for case in getattr(hidden_mod, "HIDDEN_CASES", []):
                cases.append((len(cases), "hidden", np.asarray(case, dtype=float)))
        except Exception:
            pass
    return cases


def _score_beads_case(
    index: int,
    source: str,
    link_lengths: Any,
    student_func: Callable[..., Any],
    reference_func: Callable[..., Any],
    lib: Any | None,
    timeout_sec: float,
) -> dict[str, Any]:
    np = __import__("numpy")
    lengths = np.asarray(link_lengths, dtype=float)
    base = {
        "case_index": index,
        "source": source,
        "links": int(len(lengths)),
        "link_lengths": lengths.tolist(),
        "student_radius": None,
        "reference_radius": None,
        "delta": None,
        "violations": [],
        "error": None,
        "score": float("-inf"),
    }
    student = call_with_timeout(student_func, lengths.copy(), timeout_sec=timeout_sec)
    if not student.ok:
        base["error"] = student.error
        base["score"] = float("inf")
        return base
    reference = call_with_timeout(reference_func, lengths.copy(), timeout_sec=timeout_sec)
    if not reference.ok:
        base["error"] = f"reference: {reference.error}"
        base["score"] = float("inf")
        return base
    student_angles = np.asarray(student.value, dtype=float)
    reference_angles = np.asarray(reference.value, dtype=float)
    expected_shape = (len(lengths) - 1, 2)
    if student_angles.shape != expected_shape:
        base["violations"] = [f"angles shape {student_angles.shape} != expected {expected_shape}"]
        base["score"] = float("inf")
        return base
    if not np.all(np.isfinite(student_angles)):
        base["violations"] = ["angles contain nonfinite values"]
        base["score"] = float("inf")
        return base
    violations = _beads_violations(lib, lengths, student_angles)
    student_radius = _beads_radius(lib, lengths, student_angles)
    reference_radius = _beads_radius(lib, lengths, reference_angles)
    delta = student_radius - reference_radius
    base.update(
        {
            "student_radius": float(student_radius),
            "reference_radius": float(reference_radius),
            "delta": float(delta),
            "violations": violations,
            "student_angles": student_angles.tolist(),
            "reference_angles": reference_angles.tolist(),
            "score": float("inf") if violations else float(delta),
        }
    )
    return base


def _beads_violations(lib: Any | None, lengths: Any, angles: Any) -> list[str]:
    if lib is not None and hasattr(lib, "bead_configuration_violations"):
        try:
            min_dist = getattr(lib, "BEAD_MIN_NONADJACENT_CENTER_DIST", 2.0) - 5e-6
            return list(lib.bead_configuration_violations(lengths, angles, tol=1e-2, min_center_dist=min_dist))
        except Exception as exc:
            return [f"violation check failed: {type(exc).__name__}: {exc}"]
    np = __import__("numpy")
    if np.max(np.abs(angles)) > math.pi:
        return ["joint angles exceed pi in fallback check"]
    return []


def _beads_radius(lib: Any | None, lengths: Any, angles: Any) -> float:
    if lib is not None and hasattr(lib, "bounding_sphere_radius"):
        return float(lib.bounding_sphere_radius(lengths, angles))
    np = __import__("numpy")
    return float(np.linalg.norm(np.cumsum(np.asarray(lengths, dtype=float))) / 2.0)


def _beads_markdown(worst: dict[str, Any]) -> str:
    lines = ["# Beads diagnostic", ""]
    lines.append(f"- case: `{worst['source']} #{worst['case_index']}` ({worst['links']} links)")
    if worst.get("error"):
        lines.append(f"- error: {worst['error']}")
    else:
        lines.append(f"- student radius: {worst['student_radius']:.6f}")
        lines.append(f"- reference radius: {worst['reference_radius']:.6f}")
        lines.append(f"- delta: {worst['delta']:.6f}")
        if worst.get("violations"):
            lines.append("- violations: " + "; ".join(worst["violations"]))
    return "\n".join(lines) + "\n"


def _beads_summary_line(worst: dict[str, Any]) -> str:
    label = f"{worst['source']} case {worst['case_index']}"
    if worst.get("error"):
        return f"Beads diagnostic: {label} failed with {worst['error']}."
    if worst.get("violations"):
        return f"Beads diagnostic: {label} has {len(worst['violations'])} feasibility violation(s)."
    return f"Beads diagnostic: {label} delta={worst['delta']:.4f}, links={worst['links']}."


def _beads_plot_png(worst: dict[str, Any], lib: Any | None) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)
        if worst.get("student_radius") is not None:
            ax.bar(["student", "reference"], [worst["student_radius"], worst["reference_radius"]], color=["tab:blue", "tab:orange"])
            ax.set_ylabel("bounding sphere radius")
        else:
            lengths = np.asarray(worst.get("link_lengths") or [], dtype=float)
            ax.plot(lengths, marker="o")
            ax.set_ylabel("link length")
        ax.set_title("Worst beads case")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


@dataclass
class _Configuration:
    x: float
    y: float
    z: float
    theta: float = 0.0
    phi: float = 0.0

    def position(self):
        np = __import__("numpy")
        return np.asarray([self.x, self.y, self.z], dtype=float)


@dataclass
class _XYZConfiguration:
    x: float
    y: float
    z: float

    def position(self):
        np = __import__("numpy")
        return np.asarray([self.x, self.y, self.z], dtype=float)


def _broom_lib() -> dict[str, Any]:
    lib = _try_import("lib.broom_racing")
    return {
        "Configuration": getattr(lib, "Configuration", _Configuration),
        "XYZConfiguration": getattr(lib, "XYZConfiguration", _XYZConfiguration),
        "curve_length": getattr(lib, "curve_length", _fallback_curve_length),
        "check_all": getattr(lib, "check_all", _fallback_check_all),
        "check_constraints": getattr(lib, "check_constraints", None),
    }


def _broom_cases(context: DiagnosticContext) -> list[dict[str, Any]]:
    cases_path = homework_dir(context) / "tests" / "broom_racing_open_test_cases.json"
    cases = []
    if cases_path.is_file():
        cases.extend(json.loads(cases_path.read_text(encoding="utf-8")))
    hdir = hidden_dir(context)
    hidden_path = hdir / "broom_racing_hidden_cases.json" if hdir else None
    if hidden_path and hidden_path.is_file():
        try:
            for item in json.loads(hidden_path.read_text(encoding="utf-8")):
                item = dict(item)
                item["_source"] = "hidden"
                cases.append(item)
        except Exception:
            pass
    return cases


def _configuration(lib: dict[str, Any], values: list[float]) -> Any:
    return lib["Configuration"](*(float(v) for v in values))


def _xyz_configuration(lib: dict[str, Any], values: list[float]) -> Any:
    return lib["XYZConfiguration"](*(float(v) for v in values))


def _score_broom_case(
    index: int,
    case: dict[str, Any],
    student_mod: Any,
    reference_mod: Any,
    lib: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
    task = str(case.get("task", ""))
    source = str(case.get("_source", "public"))
    start = _configuration(lib, case["start"])
    ref_func = getattr(reference_mod, f"{task}_ref", None) or getattr(reference_mod, task, None)
    student_func = getattr(student_mod, task, None)
    record = {"task": task, "case_index": index, "source": source, "error": None, "score": float("inf")}
    if not callable(student_func) or not callable(ref_func):
        record["error"] = "missing student or reference function"
        return record
    try:
        args, goal, goal_xyz = _broom_args(lib, task, start, case)
    except Exception as exc:
        record["error"] = f"case setup failed: {type(exc).__name__}: {exc}"
        return record
    student = call_with_timeout(student_func, *args, timeout_sec=timeout_sec)
    if not student.ok:
        record["error"] = student.error
        return record
    reference = call_with_timeout(ref_func, *args, timeout_sec=timeout_sec)
    if not reference.ok:
        record["error"] = f"reference: {reference.error}"
        return record
    try:
        student_length = float(lib["curve_length"](student.value))
        reference_length = float(lib["curve_length"](reference.value))
        ok, errors = lib["check_all"](student.value, start, goal=goal, goal_xyz=goal_xyz)
        samples = _sample_curve(student.value)
        nonfinite = not samples["finite"]
        ratio_delta = student_length / max(reference_length, 1e-12) - 1.0
        record.update(
            {
                "student_length": student_length,
                "reference_length": reference_length,
                "ratio_delta": ratio_delta,
                "constraint_errors": list(errors) + (["nonfinite curve samples"] if nonfinite else []),
                "score": float("inf") if (not ok or nonfinite) else ratio_delta,
                "samples": samples,
                "case": case,
            }
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _broom_args(lib: dict[str, Any], task: str, start: Any, case: dict[str, Any]) -> tuple[tuple[Any, ...], Any | None, Any | None]:
    if task == "gate_pass":
        goal = _configuration(lib, case["goal"])
        return (start, goal), goal, None
    if task == "catch_snitch":
        goal_xyz = _xyz_configuration(lib, case["goal_xyz"])
        return (start, goal_xyz), None, goal_xyz
    if task == "catch_ball_and_gate":
        intermediate = _xyz_configuration(lib, case["goal_xyz"])
        final = _configuration(lib, case["goal"])
        return (start, intermediate, final), final, None
    raise ValueError(f"unknown task {task}")


def _fallback_curve_length(curve_fn: Callable[..., Any], n_points: int = 200) -> float:
    import numpy as np

    samples = _sample_curve(curve_fn, n_points)
    pts = np.column_stack([samples["x"], samples["y"], samples["z"]])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _fallback_check_all(curve_fn: Callable[..., Any], start: Any, goal: Any = None, goal_xyz: Any = None, **_: Any) -> tuple[bool, list[str]]:
    import numpy as np

    errors = []
    samples = _sample_curve(curve_fn)
    if not samples["finite"]:
        errors.append("curve has nonfinite samples")
    first = np.asarray([samples["x"][0], samples["y"][0], samples["z"][0]], dtype=float)
    if np.linalg.norm(first - start.position()) > 0.1:
        errors.append("start position error")
    target = goal.position() if goal is not None else goal_xyz.position() if goal_xyz is not None else None
    if target is not None:
        last = np.asarray([samples["x"][-1], samples["y"][-1], samples["z"][-1]], dtype=float)
        if np.linalg.norm(last - target) > 0.1:
            errors.append("goal position error")
    return not errors, errors


def _sample_curve(curve_fn: Callable[..., Any], n_points: int = 200) -> dict[str, Any]:
    import numpy as np

    values = {"s": [], "x": [], "y": [], "z": [], "theta": [], "phi": []}
    finite = True
    for s in np.linspace(0.0, 1.0, n_points):
        c = curve_fn(np.atleast_1d(s))
        values["s"].append(float(s))
        for key in ("x", "y", "z", "theta", "phi"):
            val = float(np.asarray(getattr(c, key, 0.0)).flat[0])
            values[key].append(val)
            finite = finite and bool(np.isfinite(val))
    values["finite"] = finite
    return values


def _worst_broom_by_task(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    worst = []
    for task in ("gate_pass", "catch_snitch", "catch_ball_and_gate"):
        task_results = [item for item in results if item.get("task") == task]
        if task_results:
            worst.append(max(task_results, key=lambda item: (float(item.get("score", float("-inf"))), int(item.get("case_index", -1)))))
    return worst


def _broom_markdown(worst: list[dict[str, Any]]) -> str:
    lines = ["# Broom racing diagnostic", ""]
    for item in worst:
        line = f"- {item['task']}: {item['source']} case {item['case_index']}"
        if item.get("error"):
            line += f", error {item['error']}"
        else:
            line += (
                f", student {item.get('student_length', float('nan')):.6f}, "
                f"ref {item.get('reference_length', float('nan')):.6f}, "
                f"ratio_delta {item.get('ratio_delta', float('nan')):.6f}"
            )
            if item.get("constraint_errors"):
                line += ", constraints: " + "; ".join(item["constraint_errors"])
        lines.append(line)
    return "\n".join(lines) + "\n"


def _broom_plot_png(item: dict[str, Any], student_mod: Any, reference_mod: Any, lib: dict[str, Any], timeout_sec: float) -> bytes | None:
    if item.get("error") or "case" not in item:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        task = item["task"]
        case = item["case"]
        start = _configuration(lib, case["start"])
        args, _, _ = _broom_args(lib, task, start, case)
        student_func = getattr(student_mod, task)
        ref_func = getattr(reference_mod, f"{task}_ref", None) or getattr(reference_mod, task)
        s_curve = call_with_timeout(student_func, *args, timeout_sec=timeout_sec).value
        r_curve = call_with_timeout(ref_func, *args, timeout_sec=timeout_sec).value
        s = _sample_curve(s_curve)
        r = _sample_curve(r_curve)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(s["x"], s["y"], label="student", color="tab:blue")
        ax.plot(r["x"], r["y"], label="reference", color="tab:orange")
        ax.scatter([start.x], [start.y], c="black", label="start")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"{task} trajectory")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _broom_trace_plot_png(item: dict[str, Any]) -> bytes | None:
    samples = item.get("samples")
    if not samples:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        s = np.asarray(samples["s"])
        theta = np.unwrap(np.asarray(samples["theta"]))
        phi = np.asarray(samples["phi"])
        curvature = np.gradient(theta, s, edge_order=1) ** 2 + np.gradient(phi, s, edge_order=1) ** 2
        curvature = np.sqrt(np.maximum(curvature, 0.0))
        fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        axes[0].plot(s, curvature, color="tab:purple")
        axes[0].set_ylabel("curvature proxy")
        axes[1].plot(s, phi, color="tab:green")
        axes[1].set_ylabel("pitch")
        axes[1].set_xlabel("s")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _so101_cases(context: DiagnosticContext) -> list[tuple[int, str, list[float], bool, bool]]:
    cases_path = homework_dir(context) / "tests" / "so101_ik_open_test_cases.json"
    cases = []
    if cases_path.is_file():
        for idx, item in enumerate(json.loads(cases_path.read_text(encoding="utf-8"))):
            ana = bool(item.get("solvable_analytical", item.get("solvable", False)))
            num = bool(item.get("solvable_numerical", item.get("solvable", False)))
            cases.append((idx, "public", item["pose"], ana, num))
    hdir = hidden_dir(context)
    hidden_path = hdir / "test_so101_ik.py" if hdir else None
    if hidden_path and hidden_path.is_file():
        try:
            hidden_mod = load_module_from_path("_diagnostic_hidden_so101", hidden_path)
            for pose, solvable, _q_ref in getattr(hidden_mod, "HIDDEN_CASES", []):
                cases.append((len(cases), "hidden", list(pose), False, bool(solvable)))
        except Exception:
            pass
    return cases


def _classify_so101_solver(
    index: int,
    source: str,
    pose: list[float],
    solver: str,
    expected_solution: bool,
    func: Callable[..., Any] | None,
    timeout_sec: float,
) -> dict[str, Any]:
    record = {
        "case_index": index,
        "source": source,
        "solver": solver,
        "classification": "ok",
        "expected_solution": expected_solution,
        "error": "",
        "joint_count": 0,
        "nonfinite": False,
    }
    if not callable(func):
        record.update(classification="missing_api", error="function is missing")
        return record
    result = call_with_timeout(func, *pose, timeout_sec=timeout_sec)
    if not result.ok:
        record.update(classification="timeout" if result.timeout else "exception", error=result.error or "")
        return record
    if result.value is None:
        if expected_solution:
            record.update(classification="returned_none", error="expected a solution")
        return record
    q = _so101_joint_array(result.value)
    record["joint_count"] = len(q)
    if len(q) != len(SO101_JOINT_NAMES):
        record.update(classification="bad_shape", error=f"expected {len(SO101_JOINT_NAMES)} joints, got {len(q)}")
        return record
    import numpy as np

    nonfinite = not bool(np.all(np.isfinite(q)))
    record["nonfinite"] = nonfinite
    if nonfinite:
        record.update(classification="nonfinite", error="joint vector has nonfinite values")
    elif not expected_solution:
        record.update(classification="unexpected_solution", error="expected None for unsolvable pose")
    return record


def _so101_joint_array(value: Any) -> list[float]:
    import numpy as np

    if isinstance(value, dict):
        return [float(value.get(name, np.nan)) for name in SO101_JOINT_NAMES]
    return [float(v) for v in np.asarray(value, dtype=float).reshape(-1).tolist()]


def _classify_so101_formula(index: int, source: str, pose: list[float], student_mod: Any) -> dict[str, Any]:
    record = {"case_index": index, "source": source, "solver": "formula", "classification": "ok", "error": ""}
    func = getattr(student_mod, "so101_downturned_ik_symbolic", None)
    if not callable(func):
        record.update(classification="missing_api", error="symbolic function is missing")
        return record
    try:
        import sympy

        x, y, z, yaw = sympy.symbols("x y z yaw", real=True)
        formulas = func(x, y, z, yaw)
        values = list(formulas.values()) if isinstance(formulas, dict) else list(formulas)
        if len(values) != len(SO101_JOINT_NAMES):
            record.update(classification="bad_shape", error=f"expected {len(SO101_JOINT_NAMES)} formulas")
            return record
        if any(value.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo) for value in values if hasattr(value, "has")):
            record.update(classification="nonfinite_formula", error="formula contains nonfinite symbolic value")
    except Exception as exc:
        record.update(classification="formula_exception", error=f"{type(exc).__name__}: {exc}")
    return record


def _dicts_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = sorted({key for row in rows for key in row})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _so101_markdown(failures: list[dict[str, Any]]) -> str:
    lines = ["# SO101 IK diagnostic", "", f"Failures: {len(failures)}"]
    for item in failures[:20]:
        lines.append(
            f"- {item['solver']} {item['source']} case {item['case_index']}: "
            f"{item['classification']} {item.get('error', '')}".strip()
        )
    return "\n".join(lines) + "\n"


def _so101_plot_png(rows: list[dict[str, Any]]) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        counts: dict[str, int] = {}
        for row in rows:
            counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(counts), list(counts.values()), color="tab:red")
        ax.set_ylabel("cases")
        ax.set_title("SO101 classification counts")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None

