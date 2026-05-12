"""Fail-soft diagnostics for reports-only batch grading."""

from .base import DiagnosticContext, DiagnosticPlugin, DiagnosticResult
from .generic import default_plugins
from .registry import DiagnosticRegistry
from .run_diagnostics import run_diagnostics_for_student

__all__ = [
    "DiagnosticContext",
    "DiagnosticPlugin",
    "DiagnosticRegistry",
    "DiagnosticResult",
    "default_plugins",
    "run_diagnostics_for_student",
]
