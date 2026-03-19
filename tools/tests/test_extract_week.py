"""Tests for shared/week_helpers.py extract_week (no telegram dependency)."""

from shared.week_helpers import extract_week


def test_grade_01():
    assert extract_week("/grade 01") == "01"


def test_grade_1_padded():
    assert extract_week("/grade 1") == "01"


def test_week_01():
    assert extract_week("week 01") == "01"


def test_week01():
    assert extract_week("week01") == "01"


def test_no_match():
    assert extract_week("hello") is None
    assert extract_week("") is None


def test_none_caption():
    assert extract_week(None) is None


def test_case_insensitive():
    assert extract_week("/GRADE 01") == "01"
    assert extract_week("Week 01") == "01"
