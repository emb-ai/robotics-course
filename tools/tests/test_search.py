"""Tests for oracle/tools/file_tools.py search_code."""

from pathlib import Path
from unittest.mock import patch


from oracle.tools.file_tools import search_code


def test_search_code_empty_dir(tmp_path):
    result = search_code("anything", ".", base=tmp_path)
    assert isinstance(result, str)


def test_search_code_mock_rg(tmp_path):
    (tmp_path / "foo.py").write_text("def bar(): pass")
    with patch("oracle.tools.file_tools.subprocess.run") as m:
        m.return_value = type("R", (), {
            "returncode": 1,
            "stdout": "",
        })()
        # _rtk_available check also calls subprocess.run; force rtk unavailable
        m.side_effect = FileNotFoundError
        result = search_code("bar", ".", base=tmp_path)
        assert isinstance(result, str)


def test_search_code_char_limit(tmp_path):
    (tmp_path / "foo.py").write_text("x" * 3000)
    with patch("oracle.tools.file_tools._rtk_available", return_value=False):
        with patch("oracle.tools.file_tools.subprocess.run") as m:
            m.return_value = type("R", (), {
                "returncode": 0,
                "stdout": "x" * 3000,
            })()
            result = search_code("x", ".", base=tmp_path, max_chars=500)
            assert len(result) <= 520
