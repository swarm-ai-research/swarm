"""Short-lived, ref-scoped git push tokens — enforce "block push to protected refs".

The MVP policy is advisory: it inspects a diff after the fact, so nothing
physically stops a buggy or compromised agent from pushing to ``main``. This
module closes that gap for the push path with a capability the agent cannot
forge or over-scope:

- A **push grant** authorizes pushing *only* to refs matching an explicit
  allow-list of patterns (``refs/heads/agent/*``), and only until it expires.
- The grant is HMAC-sealed by an authority key the agent does not hold, so it
  cannot mint its own grant, widen the ref scope, or extend the expiry.
- A ``pre-push`` gate (:func:`check_push`) verifies the grant against the refs
  actually being pushed and **fails closed**: no token, expired token,
  tampered token, or an out-of-scope ref all deny the push.

"tasks have power, not accounts": the grant is bound to a task id and a short
TTL, so authority is scoped to the unit of delegated work rather than to a
standing credential. This is a policy-layer enforcement primitive — it gates
the push at the client via the git ``pre-push`` hook; pair it with server-side
branch protection for defense in depth.
"""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple

TOKEN_SCHEME = "agentgit-push.v1"
# Protected refs an agent must never push to without an explicit grant that
# names them. The default allow-list deliberately excludes these prefixes.
DEFAULT_PROTECTED_PATTERNS = ("refs/heads/main", "refs/heads/master", "refs/tags/*")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PushGrant:
    """A signed, expiring authorization to push to specific ref patterns."""

    task_id: str
    agent_id: str
    allowed_ref_patterns: Tuple[str, ...]
    issued_at: str
    expires_at: str
    signature: str = ""

    def _payload(self) -> dict:
        return {
            "scheme": TOKEN_SCHEME,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "allowed_ref_patterns": list(self.allowed_ref_patterns),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    def to_token(self) -> str:
        """Serialize to a compact, signed token string."""

        return json.dumps({**self._payload(), "signature": self.signature},
                          sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_token(cls, token: str) -> "PushGrant":
        data = json.loads(token)
        return cls(
            task_id=data.get("task_id", ""),
            agent_id=data.get("agent_id", ""),
            allowed_ref_patterns=tuple(data.get("allowed_ref_patterns", [])),
            issued_at=data.get("issued_at", ""),
            expires_at=data.get("expires_at", ""),
            signature=data.get("signature", ""),
        )


def _sign(payload: dict, authority_key: str) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(authority_key.encode("utf-8"), body, hashlib.sha256).hexdigest()


def mint_push_grant(
    *,
    task_id: str,
    agent_id: str,
    allowed_ref_patterns: Sequence[str],
    authority_key: str,
    ttl_seconds: int = 900,
    now: Optional[datetime] = None,
) -> PushGrant:
    """Mint a signed push grant. Only an authority-key holder can call this.

    ``allowed_ref_patterns`` are fnmatch patterns against full ref names
    (``refs/heads/agent/task-1``). TTL defaults to 15 minutes.
    """

    if not allowed_ref_patterns:
        raise ValueError("a push grant must authorize at least one ref pattern")
    issued = now or _now()
    grant = PushGrant(
        task_id=task_id,
        agent_id=agent_id,
        allowed_ref_patterns=tuple(allowed_ref_patterns),
        issued_at=issued.isoformat(),
        expires_at=(issued + timedelta(seconds=ttl_seconds)).isoformat(),
    )
    signature = _sign(grant._payload(), authority_key)
    return PushGrant(**{**grant.__dict__, "signature": signature})


def verify_grant(
    grant: PushGrant,
    *,
    authority_key: str,
    now: Optional[datetime] = None,
) -> Tuple[bool, List[str]]:
    """Verify a grant's signature and expiry. Returns ``(ok, errors)``."""

    errors: List[str] = []
    expected = _sign(grant._payload(), authority_key)
    if not hmac.compare_digest(expected, grant.signature or ""):
        errors.append("signature verification failed")
    try:
        expires = datetime.fromisoformat(grant.expires_at)
    except ValueError:
        errors.append("unparseable expires_at")
        return False, errors
    if expires.tzinfo is None:
        # A crafted naive timestamp must deny cleanly, not raise on compare.
        errors.append("expires_at is not timezone-aware")
        return False, errors
    if (now or _now()) >= expires:
        errors.append("grant expired")
    return not errors, errors


def _ref_allowed(ref: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(ref, pat) for pat in patterns)


def check_push(
    refs: Sequence[str],
    *,
    grant: Optional[PushGrant],
    authority_key: str,
    protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATTERNS,
    now: Optional[datetime] = None,
) -> Tuple[bool, List[str]]:
    """Decide whether a push to ``refs`` is allowed. Fails closed.

    Two regimes:

    - **No grant supplied**: only unprotected refs may be pushed. Any
      protected ref is denied.
    - **Grant supplied**: the grant is the agent's *whole* authority
      envelope — every pushed ref (protected or not) must match one of the
      grant's patterns. An invalid or expired grant authorizes nothing, so
      with one presented every ref is denied ("fails closed").

    The second regime is deliberately tighter than "protected refs need a
    grant": an agent operating under a task grant must not push outside its
    task's refs at all, even to otherwise-unprotected branches.
    """

    reasons: List[str] = []
    if grant is not None:
        ok, gerrors = verify_grant(grant, authority_key=authority_key, now=now)
        if not ok:
            reasons.extend(f"grant: {e}" for e in gerrors)
            reasons.extend(
                f"push to {ref!r} denied (grant invalid)" for ref in refs
            )
            return (not refs), reasons
        allowed = True
        for ref in refs:
            if not _ref_allowed(ref, grant.allowed_ref_patterns):
                allowed = False
                reasons.append(
                    f"push to {ref!r} denied (outside grant scope "
                    f"{list(grant.allowed_ref_patterns)})"
                )
        return allowed, reasons

    allowed = True
    for ref in refs:
        if _ref_allowed(ref, protected_patterns):
            allowed = False
            reasons.append(
                f"push to protected ref {ref!r} denied (no grant presented)"
            )
    return allowed, reasons


def parse_prepush_stdin(text: str) -> List[str]:
    """Extract the local refs being pushed from git ``pre-push`` hook stdin.

    Each line is ``<local ref> <local sha> <remote ref> <remote sha>``; a
    deletion has an all-zero local sha and is ignored (deleting a protected
    ref is a separate concern handled by server-side protection).
    """

    refs: List[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        local_ref, local_sha, remote_ref, _remote_sha = parts
        if set(local_sha) == {"0"}:
            continue  # deletion
        refs.append(remote_ref)
    return refs
