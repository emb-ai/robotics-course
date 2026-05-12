"""Diagnostics plugin registry and fail-soft execution."""

from __future__ import annotations

import queue
import threading
from typing import Any

from .base import DiagnosticContext, DiagnosticPlugin, DiagnosticResult


DIAGNOSTIC_PROBLEM_STATUSES = {"failed", "error", "timeout", "missing", "skipped"}


class DiagnosticRegistry:
    def __init__(self, plugins: list[DiagnosticPlugin] | None = None):
        self._plugins: dict[str, DiagnosticPlugin] = {}
        for plugin in plugins or []:
            self.register(plugin)

    def register(self, plugin: DiagnosticPlugin) -> None:
        plugin_id = str(getattr(plugin, "id", ""))
        if not plugin_id:
            raise ValueError("diagnostic plugin id is required")
        if plugin_id in self._plugins:
            raise ValueError(f"duplicate diagnostic plugin id: {plugin_id}")
        self._plugins[plugin_id] = plugin

    def select(self, context: DiagnosticContext) -> list[DiagnosticPlugin]:
        selected: list[DiagnosticPlugin] = []
        for plugin in self._plugins.values():
            try:
                if plugin.supports(context):
                    selected.append(plugin)
            except Exception:
                continue
        return selected

    def run(self, context: DiagnosticContext) -> list[DiagnosticResult]:
        return [self._run_plugin(plugin, context) for plugin in self.select(context)]

    def _run_plugin(
        self,
        plugin: DiagnosticPlugin,
        context: DiagnosticContext,
    ) -> DiagnosticResult:
        plugin_id = str(getattr(plugin, "id", "unknown"))
        timeout_sec = float(getattr(plugin, "timeout_sec", 10.0) or 10.0)
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def target() -> None:
            try:
                result_queue.put(("ok", plugin.run(context)))
            except Exception as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout_sec)
        if thread.is_alive():
            return DiagnosticResult(
                plugin_id=plugin_id,
                problem_id=context.problem_id,
                status="timeout",
                summary=f"Diagnostic plugin timed out after {timeout_sec:g}s.",
                error="timeout",
            )

        status, payload = result_queue.get()
        if status == "error":
            return DiagnosticResult(
                plugin_id=plugin_id,
                problem_id=context.problem_id,
                status="error",
                summary="Diagnostic plugin failed.",
                error=str(payload),
            )

        try:
            result = payload
            if isinstance(result, DiagnosticResult):
                return result
            if isinstance(result, dict):
                return DiagnosticResult(**result)
            return DiagnosticResult(
                plugin_id=plugin_id,
                problem_id=context.problem_id,
                status="error",
                error=f"Plugin returned unsupported result: {type(result).__name__}",
            )
        except Exception as exc:
            return DiagnosticResult(
                plugin_id=plugin_id,
                problem_id=context.problem_id,
                status="error",
                summary="Diagnostic plugin failed.",
                error=str(exc),
            )


def should_run_diagnostics(problem_result: dict[str, Any]) -> bool:
    return str(problem_result.get("status", "")).lower() in DIAGNOSTIC_PROBLEM_STATUSES
