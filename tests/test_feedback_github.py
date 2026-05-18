"""Tests for filing customer bug reports into the central GitHub tracker."""
from __future__ import annotations

import pytest

from clsplusplus.api import _file_github_bug_issue
from clsplusplus.config import Settings

pytestmark = [pytest.mark.unit, pytest.mark.regression]


async def test_noop_without_token():
    # No token configured → must not raise and must not call out.
    await _file_github_bug_issue(
        Settings(github_issue_token=""),
        comment="login broke",
        context="bug · /signup",
        score=2,
        user_email="a@b.com",
    )


async def test_files_labeled_issue(monkeypatch):
    captured: dict = {}

    class _Resp:
        def raise_for_status(self):
            pass

    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    await _file_github_bug_issue(
        Settings(github_issue_token="ghp_test", github_issue_repo="acme/repo"),
        comment="login button does nothing",
        context="bug · /signup",
        score=2,
        user_email="dev@acme.com",
    )

    assert captured["url"] == "https://api.github.com/repos/acme/repo/issues"
    assert captured["json"]["title"].startswith("[prod-cx]")
    assert set(captured["json"]["labels"]) == {
        "bug", "prod-cx", "source:user", "priority:P1",
    }
    assert "dev@acme.com" in captured["json"]["body"]
    assert "2/5" in captured["json"]["body"]


async def test_swallows_errors(monkeypatch):
    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise RuntimeError("github down")

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    # Best-effort — a tracker failure must not raise.
    await _file_github_bug_issue(
        Settings(github_issue_token="ghp_test"),
        comment="x",
        context="bug · /p",
        score=1,
        user_email="a@b.com",
    )
