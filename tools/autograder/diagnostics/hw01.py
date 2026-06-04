"""Homework 1 diagnostic plugins."""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
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
    reference_file,
    submitted_file,
)


HW01_TOPIC = "01-intro-and-kinematics"
SO101_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
BROOM_LENGTH_RTOL = 1e-3
BROOM_LENGTH_MARGIN = 1e-3


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
            video_error = None
            try:
                video = _beads_video_mp4(context, worst)
            except Exception as exc:
                video_error = str(exc)
                artifacts.append(
                    _write_text(
                        context,
                        "beads",
                        "beads_video_error.md",
                        f"# Beads video\n\nVideo artifact generation failed: {video_error}\n",
                        "Beads video renderer failure.",
                    )
                )
            else:
                artifacts.append(
                    context.artifact_store.write_bytes_artifact(
                        context.student_id,
                        context.problem_id,
                        "diagnostic",
                        "beads_compare.mp4",
                        video,
                        label="beads_compare.mp4",
                        description="Beads student/reference comparison video.",
                    )
                )
            summary = _beads_summary_line(worst)
            if video_error:
                summary = f"{summary} Beads video failed: {video_error}"
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
    timeout_sec = 240.0

    def __init__(self, case_timeout_sec: float = 60.0):
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
            cases_to_score = _broom_cases_to_score(context, cases)
            results = [
                _score_broom_case(index, case, student_mod, reference_mod, lib, self.case_timeout_sec)
                for index, case in cases_to_score
            ]
            worst = _worst_broom_by_task(results)
            payload = {"worst_cases": worst, "case_count": len(cases), "scored_case_count": len(cases_to_score)}
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
                png = _broom_plot_png(item)
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
            failed = [
                item
                for item in worst
                if item.get("error") or item.get("constraint_errors") or item.get("length_failed")
            ]
            summary = (
                f"Broom diagnostic: {len(worst)} task(s), "
                f"{len(failed)} with errors, constraint violations, or length overruns."
            )
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


def _beads_video_mp4(context: DiagnosticContext, worst: dict[str, Any]) -> bytes:
    if worst.get("error"):
        raise RuntimeError(f"cannot render beads video for failed case: {worst['error']}")
    missing = [
        key
        for key in ("link_lengths", "student_angles", "reference_angles")
        if key not in worst
    ]
    if missing:
        raise RuntimeError(f"cannot render beads video; missing {', '.join(missing)}")
    script = (
        Path(context.repo_root)
        / "dev"
        / HW01_TOPIC
        / "homework"
        / "utility"
        / "beads"
        / "run_beads_compare_video.py"
    )
    if not script.is_file():
        raise FileNotFoundError(f"beads video renderer not found: {script}")

    with tempfile.TemporaryDirectory(prefix="beads_video_") as tmp:
        out_path = Path(tmp) / "beads_compare.mp4"
        cmd = [
            sys.executable,
            str(script),
            "--link-lengths-json",
            json.dumps(worst["link_lengths"]),
            "--student-angles-json",
            json.dumps(worst["student_angles"]),
            "--reference-angles-json",
            json.dumps(worst["reference_angles"]),
            "--fps",
            "12",
            "--segment-seconds",
            "2.5",
            "--width",
            "960",
            "--height",
            "540",
            "--matplotlib-only",
            "--out",
            str(out_path),
        ]
        env = os.environ.copy()
        extra_pythonpath = [
            str(Path(context.normalized_submission_dir).parent),
            str(homework_dir(context)),
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            extra_pythonpath.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(extra_pythonpath)
        completed = subprocess.run(
            cmd,
            cwd=script.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=75,
            check=False,
        )
        if completed.returncode != 0:
            output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
            if len(output) > 800:
                output = output[:800] + "..."
            raise RuntimeError(f"renderer exited with code {completed.returncode}: {output}")
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError("renderer did not produce beads_compare.mp4")
        return out_path.read_bytes()


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


def _broom_cases_to_score(context: DiagnosticContext, cases: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    failed_cases = _broom_failed_length_cases_from_junit(context, cases)
    if failed_cases:
        return failed_cases
    return list(enumerate(cases))


def _broom_failed_length_cases_from_junit(
    context: DiagnosticContext,
    cases: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    junit_path = _broom_junit_path(context)
    if junit_path is None:
        return []
    try:
        from xml.etree import ElementTree

        root = ElementTree.fromstring(junit_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []

    best_by_task: dict[str, tuple[float, int, dict[str, Any]]] = {}
    for testcase in root.iter("testcase"):
        name = str(testcase.get("name", ""))
        task = _broom_task_from_test_name(name)
        if not task:
            continue
        failure = testcase.find("failure")
        if failure is None:
            continue
        message = str(failure.get("message", ""))
        if "Student length" not in message or "ref" not in message:
            continue
        case = _broom_case_from_failure_text(failure.text or "")
        if case is None:
            continue
        match = _match_broom_case(cases, case)
        if match is None:
            continue
        score = _broom_length_failure_score(message)
        previous = best_by_task.get(task)
        if previous is None or score > previous[0]:
            best_by_task[task] = (score, match[0], match[1])
    return [(index, case) for _, index, case in best_by_task.values()]


def _broom_junit_path(context: DiagnosticContext) -> Path | None:
    student_path_id = str(context.student_result.get("student_path_id") or "")
    candidates = []
    if student_path_id:
        student_dir = context.run_dir / "students" / student_path_id
        candidates.extend(
            [
                student_dir / "test_results" / "test_broom_racing" / "pytest.xml",
                student_dir / "pytest.xml",
            ]
        )
    try:
        student_dir = context.artifact_store.student_dir(context.student_id)
        candidates.extend(
            [
                student_dir / "test_results" / "test_broom_racing" / "pytest.xml",
                student_dir / "pytest.xml",
            ]
        )
    except Exception:
        pass
    for path in candidates:
        if path.is_file():
            return path
    return None


def _broom_task_from_test_name(name: str) -> str | None:
    for task in ("gate_pass", "catch_snitch", "catch_ball_and_gate"):
        if name.startswith(f"test_{task}_length_vs_reference"):
            return task
    return None


def _broom_case_from_failure_text(text: str) -> dict[str, Any] | None:
    match = re.search(r"case = (\{.*?\})(?:\n|$)", text)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _broom_length_failure_score(message: str) -> float:
    match = re.search(r"Student length ([0-9.eE+-]+) > ref ([0-9.eE+-]+)", message)
    if not match:
        return 0.0
    student = float(match.group(1))
    reference = float(match.group(2))
    return student / max(reference, 1e-12) - 1.0


def _match_broom_case(cases: list[dict[str, Any]], target: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    target_key = _broom_case_key(target)
    for index, case in enumerate(cases):
        if _broom_case_key(case) == target_key:
            return index, case
    return None


def _broom_case_key(case: dict[str, Any]) -> str:
    payload = {key: value for key, value in case.items() if not str(key).startswith("_")}
    return json.dumps(payload, sort_keys=True)


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
    record = {
        "task": task,
        "case_index": index,
        "source": source,
        "error": None,
        "score": float("inf"),
        "case": case,
    }
    if not callable(student_func) or not callable(ref_func):
        record["error"] = "missing student or reference function"
        return record
    try:
        _broom_args(lib, task, start, case)
    except Exception as exc:
        record["error"] = f"case setup failed: {type(exc).__name__}: {exc}"
        return record

    evaluated = call_with_timeout(
        _evaluate_broom_case,
        task,
        case,
        student_func,
        ref_func,
        lib,
        timeout_sec=timeout_sec,
    )
    if not evaluated.ok:
        record["error"] = evaluated.error
        reference_only = call_with_timeout(
            _evaluate_broom_reference_case,
            task,
            case,
            ref_func,
            lib,
            timeout_sec=timeout_sec,
        )
        if reference_only.ok and isinstance(reference_only.value, dict):
            record.update(reference_only.value)
        return record

    value = evaluated.value
    if isinstance(value, dict):
        record.update(value)
    else:
        record["error"] = f"unexpected worker result: {type(value).__name__}"
    return record


def _evaluate_broom_case(
    task: str,
    case: dict[str, Any],
    student_func: Callable[..., Any],
    ref_func: Callable[..., Any],
    lib: dict[str, Any],
) -> dict[str, Any]:
    record = _evaluate_broom_reference_case(task, case, ref_func, lib)
    start = _configuration(lib, case["start"])
    args, goal, goal_xyz = _broom_args(lib, task, start, case)
    try:
        student_curve = student_func(*args)
        student_samples = _sample_curve(student_curve)
        student_length = float(lib["curve_length"](student_curve))
        ok, errors = lib["check_all"](student_curve, start, goal=goal, goal_xyz=goal_xyz)
        constraint_errors = list(errors)
        if not student_samples["finite"]:
            constraint_errors.append("nonfinite curve samples")
        reference_length = float(record.get("reference_length", 0.0) or 0.0)
        ratio_delta = student_length / max(reference_length, 1e-12) - 1.0
        length_limit = reference_length * (1.0 + BROOM_LENGTH_RTOL) + BROOM_LENGTH_MARGIN
        length_failed = student_length > length_limit
        record.update(
            {
                "error": None,
                "student_length": student_length,
                "ratio_delta": ratio_delta,
                "length_limit": length_limit,
                "length_failed": length_failed,
                "constraint_errors": constraint_errors,
                "student_samples": student_samples,
                "samples": student_samples,
                "score": float("inf") if (not ok or constraint_errors) else ratio_delta,
            }
        )
    except Exception as exc:
        record.update({"error": f"{type(exc).__name__}: {exc}", "score": float("inf")})
    return record


def _evaluate_broom_reference_case(
    task: str,
    case: dict[str, Any],
    ref_func: Callable[..., Any],
    lib: dict[str, Any],
) -> dict[str, Any]:
    start = _configuration(lib, case["start"])
    args, _, _ = _broom_args(lib, task, start, case)
    ref_curve = ref_func(*args)
    reference_samples = _sample_curve(ref_curve)
    return {
        "reference_length": float(lib["curve_length"](ref_curve)),
        "reference_samples": reference_samples,
    }


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


def _broom_plot_png(item: dict[str, Any]) -> bytes | None:
    student = item.get("student_samples")
    reference = item.get("reference_samples")
    if not student and not reference:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        task = item["task"]
        case = item.get("case") or {}
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        projections = (("x", "y", "xy"), ("x", "z", "xz"), ("y", "z", "yz"))
        for ax, (a, b, title) in zip(axes, projections):
            if reference:
                ax.plot(reference[a], reference[b], label="reference", color="tab:orange", linewidth=1.8)
            if student:
                ax.plot(student[a], student[b], label="student", color="tab:blue", linewidth=1.8)
            _plot_broom_case_markers(ax, case, a, b)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(a)
            ax.set_ylabel(b)
            ax.set_title(title)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4), fontsize=8)
        detail = ""
        if item.get("student_length") is not None and item.get("reference_length") is not None:
            detail = f" student={item['student_length']:.4f}, ref={item['reference_length']:.4f}"
        elif item.get("error"):
            detail = f" {item['error']}"
        fig.suptitle(f"{task} trajectory{detail}", fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _plot_broom_case_markers(ax: Any, case: dict[str, Any], a: str, b: str) -> None:
    coords = {"x": 0, "y": 1, "z": 2}
    ai = coords[a]
    bi = coords[b]
    start = case.get("start")
    if start:
        ax.scatter([start[ai]], [start[bi]], c="black", marker="o", s=35, label="start", zorder=5)
    goal = case.get("goal")
    if goal:
        ax.scatter([goal[ai]], [goal[bi]], c="tab:red", marker="x", s=45, label="goal", zorder=5)
    goal_xyz = case.get("goal_xyz")
    if goal_xyz:
        label = "ball" if case.get("task") == "catch_ball_and_gate" else "goal_xyz"
        ax.scatter([goal_xyz[ai]], [goal_xyz[bi]], c="tab:green", marker="s", s=35, label=label, zorder=5)


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
