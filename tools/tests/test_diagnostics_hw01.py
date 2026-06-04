"""Tests for Homework 1 diagnostics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import autograder.diagnostics.hw01 as hw01
from autograder.batch.artifacts import BatchArtifactStore
from autograder.diagnostics.base import DiagnosticContext
from autograder.diagnostics.hw01 import (
    HW01BeadsComparePlugin,
    HW01BroomRacingTrajectoryPlugin,
    HW01SO101IKDebugPlugin,
)


TOPIC = "01-intro-and-kinematics"


def _context(tmp_path, repo_root: Path, problem_id: str, filename: str, source: str) -> DiagnosticContext:
    store = BatchArtifactStore(tmp_path / "batches", "run-01")
    store.write_submitted_file("Alice", filename, source)
    return DiagnosticContext(
        repo_root=repo_root,
        run_dir=store.run_dir,
        topic_slug=TOPIC,
        homework_id="01",
        student_id="Alice",
        problem_id=problem_id,
        test_file=f"test_{problem_id}.py",
        submitted_files=[filename],
        normalized_submission_dir=store.student_dir("Alice") / "submitted",
        student_result={"student_id": "Alice", "submitted_files": [filename]},
        problem_result={"status": "failed", "message": "failed"},
        artifact_store=store,
    )


def _write_hw01_reference(repo_root: Path, filename: str, source: str) -> None:
    ref_dir = repo_root / "dev" / TOPIC / "homework" / "reference_solution"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "__init__.py").write_text("", encoding="utf-8")
    (ref_dir / filename).write_text(source, encoding="utf-8")


def _load_beads_video_script():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "dev" / TOPIC / "homework" / "utility" / "beads" / "run_beads_compare_video.py"
    spec = importlib.util.spec_from_file_location("_test_beads_video_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hw01_beads_compare_writes_case_summary_plot_and_video(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_hw01_reference(
        repo_root,
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )
    context = _context(
        tmp_path,
        repo_root,
        "beads",
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.full((len(link_lengths) - 1, 2), 0.2)\n",
    )
    monkeypatch.setattr(hw01, "_beads_video_mp4", lambda _context, _worst: b"fake mp4", raising=False)

    result = HW01BeadsComparePlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("beads_case.json") for path in paths)
    assert any(path.endswith("beads_summary.md") for path in paths)
    assert any(path.endswith("beads_radius.png") for path in paths)
    assert any(path.endswith("beads_compare.mp4") for path in paths)
    report_ref = next(ref for ref in result.artifacts if ref["path"].endswith("beads_case.json"))
    report = json.loads((context.run_dir / report_ref["path"]).read_text())
    assert {"student_radius", "reference_radius", "delta", "links"} <= set(report)


def test_hw01_beads_compare_writes_video_when_renderer_succeeds(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_hw01_reference(
        repo_root,
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )
    context = _context(
        tmp_path,
        repo_root,
        "beads",
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.full((len(link_lengths) - 1, 2), 0.2)\n",
    )
    monkeypatch.setattr(hw01, "_beads_video_mp4", lambda _context, _worst: b"fake mp4", raising=False)

    result = HW01BeadsComparePlugin(case_timeout_sec=1.0).run(context)

    video_ref = next(ref for ref in result.artifacts if ref["path"].endswith("beads_compare.mp4"))
    assert video_ref["label"] == "beads_compare.mp4"
    assert (context.run_dir / video_ref["path"]).read_bytes() == b"fake mp4"


def test_hw01_beads_compare_reports_video_failure_when_renderer_fails(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_hw01_reference(
        repo_root,
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )
    context = _context(
        tmp_path,
        repo_root,
        "beads",
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.full((len(link_lengths) - 1, 2), 0.2)\n",
    )

    def fail_video(_context, _worst):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(hw01, "_beads_video_mp4", fail_video, raising=False)

    result = HW01BeadsComparePlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    assert result.error is None
    assert "video failed" in result.summary.lower()
    assert any(ref["path"].endswith("beads_case.json") for ref in result.artifacts)
    error_ref = next(ref for ref in result.artifacts if ref["path"].endswith("beads_video_error.md"))
    assert "renderer unavailable" in (context.run_dir / error_ref["path"]).read_text(encoding="utf-8")
    assert not any(ref["path"].endswith(".mp4") for ref in result.artifacts)


def test_hw01_beads_video_uses_explicit_link_lengths(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    _write_hw01_reference(
        repo_root,
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )
    script = repo_root / "dev" / TOPIC / "homework" / "utility" / "beads" / "run_beads_compare_video.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# test renderer\n", encoding="utf-8")
    context = _context(
        tmp_path,
        repo_root,
        "beads",
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.full((len(link_lengths) - 1, 2), 0.2)\n",
    )
    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.write_bytes(b"video")
        return hw01.subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hw01.subprocess, "run", fake_run)
    worst = {
        "case_index": 50,
        "link_lengths": [2.0, 3.0, 4.0],
        "student_angles": [[0.1, 0.2], [0.3, 0.4]],
        "reference_angles": [[0.0, 0.0], [0.0, 0.0]],
        "error": None,
    }

    video = hw01._beads_video_mp4(context, worst)

    cmd = captured["cmd"]
    assert video == b"video"
    assert "--link-lengths-json" in cmd
    assert json.loads(cmd[cmd.index("--link-lengths-json") + 1]) == [2.0, 3.0, 4.0]
    assert "--student-angles-json" in cmd
    assert json.loads(cmd[cmd.index("--student-angles-json") + 1]) == [[0.1, 0.2], [0.3, 0.4]]
    assert "--reference-angles-json" in cmd
    assert json.loads(cmd[cmd.index("--reference-angles-json") + 1]) == [[0.0, 0.0], [0.0, 0.0]]
    assert "--pytest-case-index" not in cmd


def test_beads_video_matplotlib_fallback_uses_ffmpeg_without_imageio(tmp_path, monkeypatch):
    script = _load_beads_video_script()
    np = __import__("numpy")
    out_mp4 = tmp_path / "beads.mp4"
    encoded = {}

    monkeypatch.setitem(sys.modules, "imageio", None)
    monkeypatch.setitem(sys.modules, "imageio.v2", None)

    def fake_run_ffmpeg(frames_dir, out_path, fps):
        frames = sorted(Path(frames_dir).glob("frame_*.png"))
        assert frames
        encoded["fps"] = fps
        Path(out_path).write_bytes(b"mp4")

    monkeypatch.setattr(script, "_run_ffmpeg_frames", fake_run_ffmpeg)

    script._matplotlib_fallback(
        cum_s=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        cum_r=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        out_mp4=out_mp4,
        fps=2,
        frames_per_segment=1,
        width=160,
        height=120,
    )

    assert out_mp4.read_bytes() == b"mp4"
    assert encoded == {"fps": 2}


def test_hw01_beads_compare_records_slow_student_timeout(tmp_path):
    repo_root = tmp_path / "repo"
    _write_hw01_reference(
        repo_root,
        "beads.py",
        "import numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )
    context = _context(
        tmp_path,
        repo_root,
        "beads",
        "beads.py",
        "import time, numpy as np\n"
        "def optimal_bead_config(link_lengths):\n"
        "    time.sleep(2)\n"
        "    return np.zeros((len(link_lengths) - 1, 2))\n",
    )

    result = HW01BeadsComparePlugin(case_timeout_sec=0.05).run(context)

    assert result.status == "ok"
    assert "timeout" in result.summary.lower()
    report_ref = next(ref for ref in result.artifacts if ref["path"].endswith("beads_case.json"))
    report = json.loads((context.run_dir / report_ref["path"]).read_text())
    assert report["error"] and "timeout" in report["error"].lower()


def test_hw01_broom_trajectory_writes_worst_cases_and_plots(tmp_path):
    repo_root = tmp_path / "repo"
    tests_dir = repo_root / TOPIC / "homework" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "broom_racing_open_test_cases.json").write_text(
        json.dumps(
            [
                {"task": "gate_pass", "start": [0, 0, 0, 0, 0], "goal": [1, 0, 0, 0, 0]},
                {"task": "catch_snitch", "start": [0, 0, 0, 0, 0], "goal_xyz": [1, 1, 0]},
                {
                    "task": "catch_ball_and_gate",
                    "start": [0, 0, 0, 0, 0],
                    "goal_xyz": [0.5, 0.5, 0],
                    "goal": [1, 1, 0, 0, 0],
                },
            ]
        ),
        encoding="utf-8",
    )
    curve_source = (
        "import numpy as np\n"
        "class Curve:\n"
        "    def __init__(self, sx, sy, sz, gx, gy, gz):\n"
        "        self.sx=sx; self.sy=sy; self.sz=sz; self.gx=gx; self.gy=gy; self.gz=gz\n"
        "    def __call__(self, s):\n"
        "        t=float(np.asarray(s).flat[0])\n"
        "        return type('C', (), {'x': self.sx + (self.gx-self.sx)*t, 'y': self.sy + (self.gy-self.sy)*t, 'z': self.sz + (self.gz-self.sz)*t, 'theta': 0.0, 'phi': 0.0})()\n"
        "def gate_pass_ref(start, goal): return Curve(start.x,start.y,start.z,goal.x,goal.y,goal.z)\n"
        "def catch_snitch_ref(start, goal): return Curve(start.x,start.y,start.z,goal.x,goal.y,goal.z)\n"
        "def catch_ball_and_gate_ref(start, intermediate, final): return Curve(start.x,start.y,start.z,final.x,final.y,final.z)\n"
    )
    _write_hw01_reference(repo_root, "broom_racing.py", curve_source)
    context = _context(
        tmp_path,
        repo_root,
        "broom_racing",
        "broom_racing.py",
        curve_source.replace("_ref", ""),
    )

    result = HW01BroomRacingTrajectoryPlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("broom_worst_cases.json") for path in paths)
    assert any(path.endswith("broom_summary.md") for path in paths)
    assert any(path.endswith(".png") for path in paths)


def test_hw01_broom_trajectory_plots_nested_curve_closures_that_fail_length(tmp_path):
    repo_root = tmp_path / "repo"
    tests_dir = repo_root / TOPIC / "homework" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "broom_racing_open_test_cases.json").write_text(
        json.dumps([{"task": "gate_pass", "start": [0, 0, 0, 0, 0], "goal": [1, 0, 0, 0, 0]}]),
        encoding="utf-8",
    )
    reference_source = (
        "import numpy as np\n"
        "def _curve(start, goal):\n"
        "    def curve(s):\n"
        "        t=float(np.asarray(s).flat[0])\n"
        "        return type('C', (), {'x': start.x + (goal.x-start.x)*t, 'y': start.y + (goal.y-start.y)*t, 'z': start.z + (goal.z-start.z)*t, 'theta': 0.0, 'phi': 0.0})()\n"
        "    return curve\n"
        "def gate_pass_ref(start, goal): return _curve(start, goal)\n"
        "def catch_snitch_ref(start, goal): return _curve(start, goal)\n"
        "def catch_ball_and_gate_ref(start, intermediate, final): return _curve(start, final)\n"
    )
    student_source = (
        "import numpy as np\n"
        "def _long_curve(start, goal):\n"
        "    def curve(s):\n"
        "        t=float(np.asarray(s).flat[0])\n"
        "        bump=4.0*t*(1.0-t)\n"
        "        return type('C', (), {'x': start.x + (goal.x-start.x)*t, 'y': start.y + (goal.y-start.y)*t + bump, 'z': start.z + (goal.z-start.z)*t, 'theta': 0.0, 'phi': 0.0})()\n"
        "    return curve\n"
        "def gate_pass(start, goal): return _long_curve(start, goal)\n"
        "def catch_snitch(start, goal): return _long_curve(start, goal)\n"
        "def catch_ball_and_gate(start, intermediate, final): return _long_curve(start, final)\n"
    )
    _write_hw01_reference(repo_root, "broom_racing.py", reference_source)
    context = _context(tmp_path, repo_root, "broom_racing", "broom_racing.py", student_source)

    result = HW01BroomRacingTrajectoryPlugin(case_timeout_sec=1.0).run(context)

    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("broom_gate_pass_trajectory.png") for path in paths)
    report_ref = next(ref for ref in result.artifacts if ref["path"].endswith("broom_worst_cases.json"))
    report = json.loads((context.run_dir / report_ref["path"]).read_text())
    worst = report["worst_cases"][0]
    assert worst["error"] is None
    assert worst["student_length"] > worst["reference_length"] * (1 + 1e-3) + 1e-3
    assert "student_samples" in worst
    assert "reference_samples" in worst


def test_hw01_broom_trajectory_loads_reference_package_relative_imports(tmp_path):
    repo_root = tmp_path / "repo"
    tests_dir = repo_root / TOPIC / "homework" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "broom_racing_open_test_cases.json").write_text(
        json.dumps([{"task": "gate_pass", "start": [0, 0, 0, 0, 0], "goal": [1, 0, 0, 0, 0]}]),
        encoding="utf-8",
    )
    _write_hw01_reference(
        repo_root,
        "dubins3d.py",
        "import numpy as np\n"
        "class Curve:\n"
        "    def __call__(self, s):\n"
        "        t=float(np.asarray(s).flat[0])\n"
        "        return type('C', (), {'x': t, 'y': 0.0, 'z': 0.0, 'theta': 0.0, 'phi': 0.0})()\n"
        "def make_curve(*args): return Curve()\n",
    )
    _write_hw01_reference(
        repo_root,
        "broom_racing.py",
        "from .dubins3d import make_curve\n"
        "def gate_pass_ref(start, goal): return make_curve(start, goal)\n"
        "def catch_snitch_ref(start, goal): return make_curve(start, goal)\n"
        "def catch_ball_and_gate_ref(start, intermediate, final): return make_curve(start, intermediate, final)\n",
    )
    context = _context(
        tmp_path,
        repo_root,
        "broom_racing",
        "broom_racing.py",
        "import numpy as np\n"
        "class Curve:\n"
        "    def __call__(self, s):\n"
        "        t=float(np.asarray(s).flat[0])\n"
        "        return type('C', (), {'x': t, 'y': 0.0, 'z': 0.0, 'theta': 0.0, 'phi': 0.0})()\n"
        "def gate_pass(start, goal): return Curve()\n"
        "def catch_snitch(start, goal): return Curve()\n"
        "def catch_ball_and_gate(start, intermediate, final): return Curve()\n",
    )

    result = HW01BroomRacingTrajectoryPlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    assert "attempted relative import" not in result.summary
    assert any(ref["path"].endswith("broom_worst_cases.json") for ref in result.artifacts)


def test_hw01_so101_debug_classifies_missing_and_nonfinite_cases(tmp_path):
    repo_root = tmp_path / "repo"
    tests_dir = repo_root / TOPIC / "homework" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "so101_ik_open_test_cases.json").write_text(
        json.dumps([{"pose": [0.2, 0.0, 0.04, 0.0], "solvable_analytical": True, "solvable_numerical": True}]),
        encoding="utf-8",
    )
    context = _context(
        tmp_path,
        repo_root,
        "so101_ik",
        "so101_ik.py",
        "import sympy\n"
        "def numerical_ik_so101_downturned(x, y, z, yaw): return None\n"
        "def analytical_ik_so101_downturned(x, y, z, yaw): return {'shoulder_pan': float('nan')}\n"
        "def so101_downturned_ik_symbolic(x, y, z, yaw): return {'shoulder_pan': sympy.nan}\n",
    )

    result = HW01SO101IKDebugPlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    assert "numerical" in result.summary.lower()
    paths = [ref["path"] for ref in result.artifacts]
    assert any(path.endswith("so101_failures.json") for path in paths)
    assert any(path.endswith("so101_summary.md") for path in paths)


def test_hw01_so101_debug_uses_homework_layout_for_file_relative_assets(tmp_path):
    repo_root = tmp_path / "repo"
    hw_dir = repo_root / TOPIC / "homework"
    tests_dir = hw_dir / "tests"
    assets_dir = hw_dir / "assets" / "so101"
    tests_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "robot.urdf").write_text("<robot name='dummy' />\n", encoding="utf-8")
    (tests_dir / "so101_ik_open_test_cases.json").write_text(
        json.dumps([{"pose": [0.2, 0.0, 0.04, 0.0], "solvable_analytical": True, "solvable_numerical": True}]),
        encoding="utf-8",
    )
    source = (
        "from pathlib import Path\n"
        "import sympy\n"
        "SO101_JOINT_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')\n"
        "URDF_PATH = Path(__file__).resolve().parent.parent / 'assets' / 'so101' / 'robot.urdf'\n"
        "def _q(): return {name: 0.0 for name in SO101_JOINT_NAMES}\n"
        "def numerical_ik_so101_downturned(x, y, z, yaw):\n"
        "    open(URDF_PATH, 'rb').read()\n"
        "    return _q()\n"
        "def analytical_ik_so101_downturned(x, y, z, yaw):\n"
        "    open(URDF_PATH, 'rb').read()\n"
        "    return _q()\n"
        "def so101_downturned_ik_symbolic(x, y, z, yaw):\n"
        "    return {name: sympy.Integer(0) for name in SO101_JOINT_NAMES}\n"
    )
    context = _context(tmp_path, repo_root, "so101_ik", "so101_ik.py", source)

    result = HW01SO101IKDebugPlugin(case_timeout_sec=1.0).run(context)

    assert result.status == "ok"
    report_ref = next(ref for ref in result.artifacts if ref["path"].endswith("so101_failures.json"))
    report = json.loads((context.run_dir / report_ref["path"]).read_text())
    assert not report["failures"]
