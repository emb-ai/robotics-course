"""Tests for allowlist_db/store.py."""

import pytest


@pytest.fixture
def allowlist_db_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOWLIST_DB_PATH", str(tmp_path / "allowlist.db"))
    import allowlist_db.schema as mod
    mod._schema_initialized.clear()
    yield tmp_path / "allowlist.db"


def test_is_allowed_empty_allowlist_allows_all(allowlist_db_path):
    """Empty DB = allow all."""
    from allowlist_db.store import is_allowed

    assert is_allowed(12345) is True
    assert is_allowed(99999) is True


def test_env_override_allows(monkeypatch, allowlist_db_path):
    """ALLOWED_TELEGRAM_IDS overrides DB."""
    monkeypatch.setenv("ALLOWED_TELEGRAM_IDS", "111,222")
    from allowlist_db.store import is_allowed

    assert is_allowed(111) is True
    assert is_allowed(222) is True
    assert is_allowed(333) is False


def test_add_remove_list(allowlist_db_path):
    from allowlist_db.store import add_user, remove_user, list_users, is_allowed

    add_user(111)
    add_user(222)
    assert is_allowed(111) is True
    assert is_allowed(222) is True
    assert is_allowed(333) is False

    users = list_users()
    assert 111 in users
    assert 222 in users

    remove_user(111)
    assert is_allowed(111) is False
    assert is_allowed(222) is True
    users = list_users()
    assert 111 not in users
    assert 222 in users
