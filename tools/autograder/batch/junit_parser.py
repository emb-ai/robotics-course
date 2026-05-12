"""Parse pytest JUnit XML into per-problem batch results."""

from __future__ import annotations

from pathlib import PurePosixPath
from xml.etree import ElementTree

from .models import ProblemRunResult


def parse_junit_results(
    pytest_xml: str,
    problem_ids: dict[str, str],
    points: dict[str, float],
    *,
    missing_results: list[ProblemRunResult] | None = None,
) -> list[ProblemRunResult]:
    """Map JUnit testcase outcomes to one binary result per configured test file."""

    results_by_file: dict[str, list[str]] = {}
    messages_by_file: dict[str, list[str]] = {}
    if pytest_xml.strip():
        try:
            root = ElementTree.fromstring(pytest_xml)
        except ElementTree.ParseError:
            root = None
        if root is not None:
            for case in root.iter("testcase"):
                test_file = _test_file(case)
                if not test_file:
                    continue
                status = "passed"
                message = ""
                if case.find("error") is not None:
                    status = "error"
                    message = case.find("error").get("message", "")  # type: ignore[union-attr]
                elif case.find("failure") is not None:
                    status = "failed"
                    message = case.find("failure").get("message", "")  # type: ignore[union-attr]
                elif case.find("skipped") is not None:
                    status = "skipped"
                    message = case.find("skipped").get("message", "")  # type: ignore[union-attr]
                results_by_file.setdefault(test_file, []).append(status)
                if message:
                    messages_by_file.setdefault(test_file, []).append(message)

    merged: list[ProblemRunResult] = list(missing_results or [])
    missing_files = {result.test_file for result in merged}
    for test_file, problem_id in problem_ids.items():
        if test_file in missing_files:
            continue
        statuses = results_by_file.get(test_file)
        if not statuses:
            continue
        status = _file_status(statuses)
        max_points = float(points.get(problem_id, 0))
        merged.append(
            ProblemRunResult(
                problem_id=problem_id,
                test_file=test_file,
                status=status,
                points_possible=max_points,
                points_earned=max_points if status == "passed" else 0,
                message="; ".join(messages_by_file.get(test_file, [])),
            )
        )
    return merged


def _test_file(case: ElementTree.Element) -> str | None:
    file_attr = case.get("file")
    if file_attr:
        return PurePosixPath(file_attr.replace("\\", "/")).name
    classname = case.get("classname", "")
    if classname:
        return f"{classname.split('.')[-1]}.py"
    return None


def _file_status(statuses: list[str]) -> str:
    if "error" in statuses:
        return "error"
    if "failed" in statuses:
        return "failed"
    if "passed" in statuses:
        return "passed"
    return "skipped"
