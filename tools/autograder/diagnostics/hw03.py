"""Homework 3 starter diagnostic plugins."""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any

from .base import DiagnosticContext, DiagnosticPlugin, DiagnosticResult
from .homework_runtime import isolated_homework_imports, jsonable, load_submitted_module


class HW03RocketEpisodePlugin:
    id = "hw03_rocket_episode"
    label = "HW3 rocket episode"
    timeout_sec = 25.0

    def __init__(self, max_steps: int = 600):
        self.max_steps = max_steps

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw03(context) and context.problem_id == "rocket"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            rows, summary, metrics = _run_rocket(context, self.max_steps)
        return _episode_result(
            context,
            self.id,
            "rocket_episode.csv",
            "rocket_episode.png",
            "rocket_summary.md",
            rows,
            summary,
            metrics,
            "Rocket episode trace.",
        )


class HW03VacmanPathPlugin:
    id = "hw03_vacman_path"
    label = "HW3 Vacman path"
    timeout_sec = 25.0

    def __init__(self, max_steps: int = 800):
        self.max_steps = max_steps

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw03(context) and context.problem_id == "vacman"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            report, rows, summary, metrics = _run_vacman(context, self.max_steps)
        artifacts = [
            _write_text(
                context,
                "vacman_episode.json",
                json.dumps(jsonable(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                "Vacman episode report.",
            ),
            _write_text(context, "vacman_summary.md", _summary_markdown("Vacman", summary, metrics), "Vacman summary."),
        ]
        png = _vacman_plot_png(rows, report)
        if png:
            artifacts.append(_write_png(context, "vacman_path.png", png, "Vacman path and coverage plot."))
        return DiagnosticResult(
            plugin_id=self.id,
            problem_id=context.problem_id,
            status="ok",
            summary=summary,
            metrics=metrics,
            artifacts=artifacts,
        )


class HW03UnicycleTracePlugin:
    id = "hw03_unicycle_trace"
    label = "HW3 unicycle trace"
    timeout_sec = 25.0

    def __init__(self, max_steps: int = 600):
        self.max_steps = max_steps

    def supports(self, context: DiagnosticContext) -> bool:
        return _is_hw03(context) and context.problem_id == "unicycle"

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        with isolated_homework_imports(context):
            rows, summary, metrics, skipped = _run_unicycle(context, self.max_steps)
        if skipped:
            artifact = _write_text(context, "unicycle_summary.md", _summary_markdown("Unicycle", summary, metrics), "Unicycle summary.")
            return DiagnosticResult(
                plugin_id=self.id,
                problem_id=context.problem_id,
                status="skipped",
                summary=summary,
                metrics=metrics,
                artifacts=[artifact],
            )
        return _episode_result(
            context,
            self.id,
            "unicycle_trace.csv",
            "unicycle_trace.png",
            "unicycle_summary.md",
            rows,
            summary,
            metrics,
            "Unicycle trace.",
        )


def default_plugins() -> list[DiagnosticPlugin]:
    return [HW03RocketEpisodePlugin(), HW03VacmanPathPlugin(), HW03UnicycleTracePlugin()]


def _is_hw03(context: DiagnosticContext) -> bool:
    return context.homework_id == "03" or context.topic_slug == "03-control"


def _run_rocket(context: DiagnosticContext, max_steps: int) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    import numpy as np
    from lib.rocket import RocketEnv, RocketEnvConfig, WindConfig, make_ascent_scenario

    student_mod = load_submitted_module(context, "rocket.py")
    controller = student_mod.RocketController()
    env = RocketEnv(RocketEnvConfig(scenario=make_ascent_scenario(), wind=WindConfig(seed=98765)))
    if hasattr(controller, "reset"):
        controller.reset()
    obs, info = env.reset()
    rows = []
    summary = "Rocket episode reached step limit."
    for step in range(min(max_steps, int(np.ceil(env.scenario.time_limit / env.config.dt)))):
        try:
            action_raw = controller(obs)
        except Exception as exc:
            summary = f"Rocket controller raised {type(exc).__name__}: {exc}"
            rows.append(_rocket_row(step, env, [float("nan"), float("nan")], summary))
            break
        action = np.asarray(action_raw, dtype=float)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(_rocket_row(step, env, action, ""))
        if terminated or truncated:
            summary = f"Rocket status={info['status']}, reward={reward}, t={info['t']:.2f}."
            break
    metrics = {
        "steps": len(rows),
        "final_status": info.get("status", "unknown"),
        "final_t": float(info.get("t", 0.0)),
    }
    return rows, summary, metrics


def _rocket_row(step: int, env: Any, action: Any, error: str) -> dict[str, Any]:
    state = getattr(env, "state", [float("nan")] * 6)
    return {
        "step": step,
        "t": float(getattr(env, "t", 0.0)),
        "x": float(state[0]),
        "z": float(state[1]),
        "vx": float(state[2]),
        "vz": float(state[3]),
        "theta": float(state[4]),
        "omega": float(state[5]),
        "thrust": float(action[0]) if len(action) else float("nan"),
        "gimbal": float(action[1]) if len(action) > 1 else float("nan"),
        "error": error,
    }


def _run_vacman(context: DiagnosticContext, max_steps: int) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any]]:
    import json as json_lib
    import numpy as np
    from lib.vacman import VacmanEnv, VacmanEnvConfig, load_vacman_cases

    student_mod = load_submitted_module(context, "vacman.py")
    controller = student_mod.VacmanController()
    case = load_vacman_cases()[0]
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
        json_lib.dump([case], f)
        f.flush()
        env = VacmanEnv(VacmanEnvConfig(cases_path=Path(f.name), case_index=0, debug=True))
    if hasattr(controller, "reset"):
        controller.reset()
    obs, info = env.reset()
    rows = []
    summary = "Vacman episode reached step limit."
    for step in range(max_steps):
        try:
            action = np.asarray(controller(obs), dtype=float)
        except Exception as exc:
            summary = f"Vacman controller raised {type(exc).__name__}: {exc}"
            rows.append(_vacman_row(step, info, [float("nan"), float("nan")], summary))
            break
        obs, _reward, terminated, truncated, info = env.step(action)
        rows.append(_vacman_row(step, info, action, ""))
        if terminated or truncated:
            summary = f"Vacman status={info['status']}, cleaned={info['cleaned']}, t={info['t']:.2f}."
            break
    metrics = {
        "steps": len(rows),
        "cleaned": int(info.get("cleaned", 0)),
        "baseline": 0,
        "battery_remaining": float(info.get("battery_remaining", 0.0)),
        "base_distance": float(info.get("base_distance", 0.0)),
        "cat_distance": float(info.get("cat_distance", 0.0)),
        "final_status": info.get("status", "unknown"),
    }
    report = dict(metrics)
    report["path"] = rows
    return report, rows, summary, metrics


def _vacman_row(step: int, info: dict[str, Any], action: Any, error: str) -> dict[str, Any]:
    vac = info.get("vacman", [float("nan"), float("nan"), float("nan")])
    cat = info.get("catman", [float("nan"), float("nan"), float("nan")])
    return {
        "step": step,
        "t": float(info.get("t", 0.0)),
        "vac_x": float(vac[0]),
        "vac_z": float(vac[1]),
        "cat_x": float(cat[0]),
        "cat_z": float(cat[1]),
        "cleaned": int(info.get("cleaned", 0)),
        "battery": float(info.get("battery_remaining", 0.0)),
        "base_distance": float(info.get("base_distance", 0.0)),
        "cat_distance": float(info.get("cat_distance", 0.0)),
        "v_left": float(action[0]) if len(action) else float("nan"),
        "v_right": float(action[1]) if len(action) > 1 else float("nan"),
        "error": error,
    }


def _run_unicycle(context: DiagnosticContext, max_steps: int) -> tuple[list[dict[str, Any]], str, dict[str, Any], bool]:
    try:
        import numpy as np
        from lib.unicycle import FIXED_TARGET_CASES, UnicycleEnv, UnicycleEnvConfig
    except Exception as exc:
        return [], f"Unicycle diagnostic skipped: {type(exc).__name__}: {exc}", {"skipped": True}, True

    try:
        student_mod = load_submitted_module(context, "unicycle.py")
        controller = student_mod.UnicycleController()
        target = FIXED_TARGET_CASES[0]
        env = UnicycleEnv(UnicycleEnvConfig(target_xy=target, debug=True))
        if hasattr(controller, "reset"):
            controller.reset()
        obs, info = env.reset(options={"target_xy": target})
        rows = []
        summary = "Unicycle episode reached step limit."
        for step in range(max_steps):
            try:
                action = np.asarray(controller(obs), dtype=float)
            except Exception as exc:
                summary = f"Unicycle controller raised {type(exc).__name__}: {exc}"
                rows.append(_unicycle_row(step, info, [float("nan")] * 3, summary))
                break
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append(_unicycle_row(step, info, action, ""))
            if terminated or truncated:
                summary = f"Unicycle status={info['status']}, reward={reward}, distance={info['distance_to_target']:.3f}."
                break
        metrics = {
            "steps": len(rows),
            "final_status": info.get("status", "unknown"),
            "distance_to_target": float(info.get("distance_to_target", 0.0)),
        }
        return rows, summary, metrics, False
    except Exception as exc:
        rows = [{"step": 0, "t": 0.0, "distance_to_target": float("nan"), "error": f"{type(exc).__name__}: {exc}"}]
        return rows, f"Unicycle diagnostic recorded error: {type(exc).__name__}: {exc}", {"error": True}, False


def _unicycle_row(step: int, info: dict[str, Any], action: Any, error: str) -> dict[str, Any]:
    wheel = info.get("wheel_xy", [float("nan"), float("nan")])
    qpos = info.get("qpos", [])
    qvel = info.get("qvel", [])
    return {
        "step": step,
        "t": float(info.get("t", 0.0)),
        "wheel_x": float(wheel[0]),
        "wheel_y": float(wheel[1]),
        "distance_to_target": float(info.get("distance_to_target", float("nan"))),
        "com_height": float(info.get("com_height", float("nan"))),
        "root_qw": float(qpos[3]) if len(qpos) > 3 else float("nan"),
        "root_omega_x": float(qvel[3]) if len(qvel) > 3 else float("nan"),
        "tau_pelvis_y": float(action[0]) if len(action) else float("nan"),
        "tau_pelvis_x": float(action[1]) if len(action) > 1 else float("nan"),
        "tau_wheel": float(action[2]) if len(action) > 2 else float("nan"),
        "error": error,
    }


def _episode_result(
    context: DiagnosticContext,
    plugin_id: str,
    csv_name: str,
    png_name: str,
    md_name: str,
    rows: list[dict[str, Any]],
    summary: str,
    metrics: dict[str, Any],
    plot_title: str,
) -> DiagnosticResult:
    artifacts = [
        _write_text(context, csv_name, _rows_to_csv(rows), f"{plot_title} CSV."),
        _write_text(context, md_name, _summary_markdown(plot_title, summary, metrics), f"{plot_title} summary."),
    ]
    png = _trace_plot_png(rows, plot_title)
    if png:
        artifacts.append(_write_png(context, png_name, png, f"{plot_title} plot."))
    return DiagnosticResult(
        plugin_id=plugin_id,
        problem_id=context.problem_id,
        status="ok",
        summary=summary,
        metrics=metrics,
        artifacts=artifacts,
    )


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


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = sorted({key for row in rows for key in row})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(jsonable(rows))
    return buf.getvalue()


def _summary_markdown(title: str, summary: str, metrics: dict[str, Any]) -> str:
    lines = [f"# {title}", "", summary, "", "## Metrics"]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _trace_plot_png(rows: list[dict[str, Any]], title: str) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        x = [row.get("t", row.get("step", i)) for i, row in enumerate(rows)]
        plotted = False
        for key in ("z", "distance_to_target", "base_distance", "cat_distance", "theta", "thrust"):
            y = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
            if len(y) == len(x) and y:
                ax.plot(x, y, label=key)
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("time/step")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def _vacman_plot_png(rows: list[dict[str, Any]], report: dict[str, Any]) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        if rows:
            ax.plot([row["vac_x"] for row in rows], [row["vac_z"] for row in rows], marker="o", label="vacman")
            ax.plot([row["cat_x"] for row in rows], [row["cat_z"] for row in rows], marker="x", label="catman")
        ax.set_title(f"Vacman cleaned={report.get('cleaned', 0)}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None

