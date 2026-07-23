"""Tests for scoped, short-lived git push tokens (7ge5 enforcement)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from swarm.agentgit.__main__ import main
from swarm.agentgit.push_tokens import (
    PushGrant,
    check_push,
    mint_push_grant,
    parse_prepush_stdin,
    verify_grant,
)

KEY = "authority-secret-key"


def _grant(patterns=("refs/heads/agent/*",), ttl=900, now=None):
    return mint_push_grant(
        task_id="task-1", agent_id="codex",
        allowed_ref_patterns=patterns, authority_key=KEY, ttl_seconds=ttl, now=now,
    )


# ---------------------------------------------------------------------------
# grant lifecycle
# ---------------------------------------------------------------------------
def test_grant_round_trips_through_token():
    grant = _grant()
    restored = PushGrant.from_token(grant.to_token())
    assert restored == grant


def test_valid_grant_verifies():
    ok, errors = verify_grant(_grant(), authority_key=KEY)
    assert ok and errors == []


def test_wrong_key_fails_verification():
    ok, errors = verify_grant(_grant(), authority_key="attacker-key")
    assert not ok
    assert any("signature" in e for e in errors)


def test_tampered_scope_fails_verification():
    grant = _grant()
    widened = PushGrant(**{**grant.__dict__,
                           "allowed_ref_patterns": ("refs/heads/*",)})
    ok, _ = verify_grant(widened, authority_key=KEY)
    assert not ok  # signature no longer covers the widened scope


def test_expired_grant_fails_verification():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    ok, errors = verify_grant(_grant(ttl=60, now=past), authority_key=KEY)
    assert not ok
    assert any("expired" in e for e in errors)


def test_mint_requires_a_pattern():
    with pytest.raises(ValueError, match="at least one"):
        mint_push_grant(task_id="t", agent_id="a",
                        allowed_ref_patterns=[], authority_key=KEY)


# ---------------------------------------------------------------------------
# push decision — fail closed
# ---------------------------------------------------------------------------
def test_unprotected_ref_allowed_without_grant():
    ok, _ = check_push(["refs/heads/agent/task-1"], grant=None, authority_key=KEY)
    assert ok


def test_protected_ref_denied_without_grant():
    ok, reasons = check_push(["refs/heads/main"], grant=None, authority_key=KEY)
    assert not ok
    assert any("main" in r for r in reasons)


def test_grant_scopes_only_its_patterns():
    grant = _grant(patterns=("refs/heads/main",))
    ok, _ = check_push(["refs/heads/main"], grant=grant, authority_key=KEY)
    assert ok
    # A grant for main does not authorize a tag push.
    ok2, _ = check_push(["refs/tags/v1"], grant=grant, authority_key=KEY)
    assert not ok2


def test_expired_grant_authorizes_nothing():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    grant = _grant(patterns=("refs/heads/main",), ttl=60, now=past)
    ok, reasons = check_push(["refs/heads/main"], grant=grant, authority_key=KEY)
    assert not ok
    assert any("expired" in r for r in reasons)


def test_mixed_push_denied_if_any_ref_unscoped():
    grant = _grant(patterns=("refs/heads/agent/*",))
    ok, _ = check_push(
        ["refs/heads/agent/task-1", "refs/heads/main"],
        grant=grant, authority_key=KEY,
    )
    assert not ok  # agent/* is fine, main is not — whole push denied


def test_grant_confines_even_unprotected_refs():
    """A valid grant is the whole authority envelope: refs outside its
    patterns are denied even when they are not protected refs."""
    grant = _grant(patterns=("refs/heads/agent/task-1",))
    ok, reasons = check_push(
        ["refs/heads/agent/other-task"],  # unprotected, but out of scope
        grant=grant, authority_key=KEY,
    )
    assert not ok
    assert any("outside grant scope" in r for r in reasons)


def test_naive_expires_at_fails_closed():
    """A crafted timezone-naive expiry denies cleanly instead of raising."""
    grant = _grant(patterns=("refs/heads/agent/*",))
    naive = PushGrant(**{**grant.__dict__, "expires_at": "2999-01-01T00:00:00"})
    # re-sign so only the naive timestamp is the defect under test
    from swarm.agentgit.push_tokens import _sign

    naive = PushGrant(**{**naive.__dict__, "signature": _sign(naive._payload(), KEY)})
    ok, errors = verify_grant(naive, authority_key=KEY)
    assert not ok
    assert any("timezone-aware" in e for e in errors)


# ---------------------------------------------------------------------------
# pre-push stdin parsing
# ---------------------------------------------------------------------------
def test_parse_prepush_extracts_remote_refs():
    stdin = (
        "refs/heads/agent/x abc123 refs/heads/agent/x 000000\n"
        "refs/heads/main def456 refs/heads/main 111111\n"
    )
    assert parse_prepush_stdin(stdin) == ["refs/heads/agent/x", "refs/heads/main"]


def test_parse_prepush_ignores_deletions():
    stdin = "(delete) 0000000000000000000000000000000000000000 refs/heads/old 111\n"
    assert parse_prepush_stdin(stdin) == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_mint_verify_roundtrip(capsys, monkeypatch):
    monkeypatch.setenv("AGENTGIT_PUSH_AUTHORITY_KEY", KEY)
    rc = main([
        "push-token", "mint", "--task", "task-1", "--agent", "codex",
        "--ref-pattern", "refs/heads/agent/*",
    ])
    assert rc == 0
    token = capsys.readouterr().out.strip()
    assert json.loads(token)["task_id"] == "task-1"

    rc = main(["push-token", "verify", "--token", token])
    assert rc == 0
    assert "VALID" in capsys.readouterr().out


def test_cli_prepush_denies_protected(capsys, monkeypatch):
    monkeypatch.setenv("AGENTGIT_PUSH_AUTHORITY_KEY", KEY)
    import io

    monkeypatch.setattr(
        "sys.stdin", io.StringIO("refs/heads/main abc refs/heads/main 111\n")
    )
    rc = main(["push-token", "prepush"])  # no token
    assert rc == 1
    assert "denied" in capsys.readouterr().err


def test_cli_prepush_allows_scoped_grant(capsys, monkeypatch):
    monkeypatch.setenv("AGENTGIT_PUSH_AUTHORITY_KEY", KEY)
    token = _grant(patterns=("refs/heads/main",)).to_token()
    monkeypatch.setenv("AGENTGIT_PUSH_TOKEN", token)
    import io

    monkeypatch.setattr(
        "sys.stdin", io.StringIO("refs/heads/main abc refs/heads/main 111\n")
    )
    rc = main(["push-token", "prepush"])
    assert rc == 0


def test_cli_no_key_fails_closed(capsys, monkeypatch):
    monkeypatch.delenv("AGENTGIT_PUSH_AUTHORITY_KEY", raising=False)
    rc = main(["push-token", "mint", "--ref-pattern", "refs/heads/x"])
    assert rc == 1
    assert "no authority key" in capsys.readouterr().out
