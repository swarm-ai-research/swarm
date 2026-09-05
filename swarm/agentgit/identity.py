"""Cryptographic agent identity and delegation chains for AgentGit.

The MVP signs provenance bundles with a shared HMAC key (see
``swarm.attestation.signer``). That proves a bundle was sealed by *someone
holding the key* — it cannot prove *which agent* produced a change, because
anyone with the key can impersonate any ``agent_id`` string.

This module adds verifiable identity using Ed25519 (asymmetric) signatures:

- An :class:`AgentIdentity` carries a DID derived from a public key, plus the
  owner/org and model/runtime/version provenance and the tools the agent is
  allowed to use.
- A :class:`DelegationChain` of signed :class:`DelegationLink` objects encodes
  ``human -> org policy -> task agent``. Each link is signed by its issuer, and
  permissions may only narrow down the chain. Because we use ``did:key``, every
  issuer's public key is embedded in its DID, so a verifier needs no external
  key registry: it extracts the key from the DID and checks the signature.

DID format (simplified ``did:key``): ``did:key:ed25519:<hex public key>``. This
is intentionally simpler than the full multibase/multicodec ``did:key`` spec;
it is unambiguous and self-describing for our purposes.

Context binding (bead zsof). A link signed over only issuer, subject,
permissions, and dates is a *bearer credential*: valid anywhere until it
expires, liftable from one run and replayed in another (the replayability
finding of arXiv:2608.09867, reproduced in our own delegation layer). Links
now also sign a random ``nonce`` and an optional ``audience`` — the task,
scenario, or session the grant is for. ``DelegationChain.verify`` takes the
``context`` it is being presented in and rejects a bound link outside it; a
:class:`NonceRegistry` rejects a nonce reused under a different payload.
Links may also carry signed ``wards`` (bead illq.3, see
``swarm.agentgit.wards``); verify checks they only narrow down the chain,
the same way permissions do. Legacy links (no nonce, no audience, no wards)
keep verifying so existing bundles are unaffected; pass
``require_context=True`` to refuse them.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

DID_PREFIX = "did:key:ed25519:"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    """Stable, signature-ready encoding of a mapping."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def did_from_public_key(public_key_hex: str) -> str:
    return f"{DID_PREFIX}{public_key_hex}"


def public_key_from_did(did: str) -> str:
    if not did.startswith(DID_PREFIX):
        raise ValueError(f"Unsupported DID method: {did!r}")
    return did[len(DID_PREFIX) :]


class AgentKeypair:
    """An Ed25519 keypair backing one agent (or org/human) identity."""

    def __init__(self, signing_key: SigningKey) -> None:
        self._signing_key = signing_key
        self._verify_key = signing_key.verify_key

    @classmethod
    def generate(cls) -> "AgentKeypair":
        return cls(SigningKey.generate())

    @classmethod
    def from_seed_hex(cls, seed_hex: str) -> "AgentKeypair":
        """Deterministically rebuild a keypair from a 32-byte hex seed."""

        seed = bytes.fromhex(seed_hex)
        if len(seed) != 32:
            raise ValueError("Ed25519 seed must be 32 bytes (64 hex chars)")
        return cls(SigningKey(seed))

    @property
    def seed_hex(self) -> str:
        """Hex-encoded private seed. Treat as a secret."""

        return bytes(self._signing_key).hex()

    @property
    def public_key_hex(self) -> str:
        return bytes(self._verify_key).hex()

    @property
    def did(self) -> str:
        return did_from_public_key(self.public_key_hex)

    def sign(self, message: bytes) -> str:
        """Return a detached signature as hex."""

        signature: bytes = self._signing_key.sign(message).signature
        return signature.hex()


def verify_signature(did: str, message: bytes, signature_hex: str) -> bool:
    """Verify a detached hex signature against the key embedded in ``did``."""

    try:
        verify_key = VerifyKey(bytes.fromhex(public_key_from_did(did)))
        verify_key.verify(message, bytes.fromhex(signature_hex))
        return True
    except (BadSignatureError, ValueError):
        return False


@dataclass(frozen=True)
class AgentIdentity:
    """Verifiable identity for one agent, owner, or org."""

    did: str
    owner: str
    org: str
    model: str = ""
    runtime: str = ""
    version: str = ""
    allowed_tools: List[str] = field(default_factory=list)

    @classmethod
    def for_keypair(
        cls,
        keypair: AgentKeypair,
        *,
        owner: str,
        org: str,
        model: str = "",
        runtime: str = "",
        version: str = "",
        allowed_tools: Optional[List[str]] = None,
    ) -> "AgentIdentity":
        return cls(
            did=keypair.did,
            owner=owner,
            org=org,
            model=model,
            runtime=runtime,
            version=version,
            allowed_tools=list(allowed_tools or []),
        )

    @property
    def public_key_hex(self) -> str:
        return public_key_from_did(self.did)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "did": self.did,
            "owner": self.owner,
            "org": self.org,
            "model": self.model,
            "runtime": self.runtime,
            "version": self.version,
            "allowed_tools": list(self.allowed_tools),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentIdentity":
        return cls(
            did=data["did"],
            owner=data.get("owner", ""),
            org=data.get("org", ""),
            model=data.get("model", ""),
            runtime=data.get("runtime", ""),
            version=data.get("version", ""),
            allowed_tools=list(data.get("allowed_tools", [])),
        )


@dataclass(frozen=True)
class DelegationLink:
    """One signed grant: ``issuer`` delegates ``permissions`` to ``subject``.

    ``nonce`` makes every link unique; ``audience`` binds it to the context
    (task / scenario / session id) it may be presented in; ``wards`` is an
    optional signed ``WardSet.to_dict()`` bounding the subject. All three are
    part of the signed payload when present, and omitted from it when absent
    so legacy signatures still verify byte-for-byte.
    """

    issuer_did: str
    subject_did: str
    permissions: List[str]
    issued_at: str
    not_after: Optional[str] = None
    signature: str = ""
    nonce: str = ""
    audience: Optional[str] = None
    wards: Optional[Dict[str, Any]] = None

    @property
    def is_bound(self) -> bool:
        return self.audience is not None

    def _signing_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "issuer_did": self.issuer_did,
            "subject_did": self.subject_did,
            "permissions": sorted(self.permissions),
            "issued_at": self.issued_at,
            "not_after": self.not_after,
        }
        if self.nonce:
            payload["nonce"] = self.nonce
        if self.audience is not None:
            payload["audience"] = self.audience
        if self.wards is not None:
            payload["wards"] = self.wards
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self._signing_payload())

    def payload_digest(self) -> str:
        """Hash of the signed payload; what a nonce is registered against."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {**self._signing_payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelegationLink":
        wards = data.get("wards")
        return cls(
            issuer_did=data["issuer_did"],
            subject_did=data["subject_did"],
            permissions=list(data.get("permissions", [])),
            issued_at=data["issued_at"],
            not_after=data.get("not_after"),
            signature=data.get("signature", ""),
            nonce=str(data.get("nonce", "") or ""),
            audience=data.get("audience"),
            wards=dict(wards) if isinstance(wards, dict) else None,
        )


class NonceRegistry:
    """Remembers which payload each nonce was signed over.

    Re-verifying the *same* link is normal (every bundle check does it), so a
    nonce seen again under the same payload digest is fine. The same nonce
    under a different payload is a reuse: someone minted a new grant with a
    recycled nonce, or edited a link after signing. Optionally persisted as
    JSON so the memory survives across processes.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self._path = Path(path) if path is not None else None
        self._seen: Dict[str, str] = {}
        if self._path is not None and self._path.exists():
            try:
                data = json.loads(self._path.read_text())
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict):
                self._seen = {str(k): str(v) for k, v in data.items()}

    def __len__(self) -> int:
        return len(self._seen)

    def check(self, link: DelegationLink) -> Optional[str]:
        """Register ``link``'s nonce; return an error string on reuse."""

        if not link.nonce:
            return None
        digest = link.payload_digest()
        prior = self._seen.get(link.nonce)
        if prior is not None and prior != digest:
            return f"nonce {link.nonce[:8]}… reused under a different payload"
        if prior is None:
            self._seen[link.nonce] = digest
            self._persist()
        return None

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._seen, sort_keys=True))


def sign_link(
    issuer: AgentKeypair,
    *,
    subject_did: str,
    permissions: List[str],
    issued_at: Optional[str] = None,
    not_after: Optional[str] = None,
    audience: Optional[str] = None,
    wards: Optional[Union[Dict[str, Any], Any]] = None,
    nonce: Optional[str] = None,
) -> DelegationLink:
    """Create a link signed by ``issuer`` granting ``permissions`` to subject.

    A fresh 128-bit ``nonce`` is minted unless one is supplied. ``audience``
    binds the link to the context it may be presented in (a task, scenario,
    or session id); leave it None only for long-lived upper links such as
    ``human -> org``. ``wards`` may be a ``WardSet`` or its ``to_dict()``.
    """

    wards_dict: Optional[Dict[str, Any]]
    if wards is None:
        wards_dict = None
    elif isinstance(wards, dict):
        wards_dict = dict(wards)
    else:
        wards_dict = wards.to_dict()
    unsigned = DelegationLink(
        issuer_did=issuer.did,
        subject_did=subject_did,
        permissions=list(permissions),
        issued_at=issued_at or _now_iso(),
        not_after=not_after,
        nonce=nonce if nonce is not None else secrets.token_hex(16),
        audience=audience,
        wards=wards_dict,
    )
    signature = issuer.sign(unsigned.canonical_bytes())
    return DelegationLink(
        issuer_did=unsigned.issuer_did,
        subject_did=unsigned.subject_did,
        permissions=unsigned.permissions,
        issued_at=unsigned.issued_at,
        not_after=unsigned.not_after,
        signature=signature,
        nonce=unsigned.nonce,
        audience=unsigned.audience,
        wards=unsigned.wards,
    )


@dataclass(frozen=True)
class DelegationChain:
    """An ordered ``human -> org -> ... -> agent`` chain of signed links."""

    links: List[DelegationLink] = field(default_factory=list)

    @property
    def root_did(self) -> Optional[str]:
        return self.links[0].issuer_did if self.links else None

    @property
    def subject_did(self) -> Optional[str]:
        return self.links[-1].subject_did if self.links else None

    def effective_permissions(self) -> List[str]:
        """Permissions granted to the final subject (the last link)."""

        return list(self.links[-1].permissions) if self.links else []

    def effective_wards(self) -> Optional[Dict[str, Any]]:
        """Signed wards on the final link, if any (``WardSet.to_dict()`` form)."""

        return dict(self.links[-1].wards) if self.links and self.links[-1].wards else None

    def verify(
        self,
        *,
        expected_subject_did: Optional[str] = None,
        now: Optional[datetime] = None,
        context: Optional[str] = None,
        require_context: bool = False,
        nonces: Optional[NonceRegistry] = None,
    ) -> Tuple[bool, List[str]]:
        """Verify signatures, connectivity, narrowing, expiry, and binding.

        ``context`` is where the chain is being presented (task / scenario /
        session id). Every bound link must name exactly this context; a bound
        link presented with no context is refused. Unbound (legacy) links pass
        unless ``require_context`` is set, which insists the final link be
        bound. ``nonces`` records each link's nonce against its payload and
        refuses reuse under a different payload. Wards, when signed on
        consecutive links, may only narrow (``swarm.agentgit.wards.never_widens``).
        """

        errors: List[str] = []
        if not self.links:
            return False, ["delegation chain is empty"]

        # Lazy: wards imports this module for DelegationChain.
        from swarm.agentgit.wards import WardSet, never_widens

        now = now or datetime.now(timezone.utc)
        prev_permissions: Optional[set[str]] = None
        prev_subject: Optional[str] = None
        prev_wards: Optional[WardSet] = None

        for index, link in enumerate(self.links):
            if not verify_signature(link.issuer_did, link.canonical_bytes(), link.signature):
                errors.append(f"link {index}: invalid issuer signature")

            if link.is_bound:
                if context is None:
                    errors.append(
                        f"link {index}: bound to audience {link.audience!r} but "
                        "presented with no context"
                    )
                elif link.audience != context:
                    errors.append(
                        f"link {index}: bound to audience {link.audience!r}, "
                        f"presented in context {context!r}"
                    )

            if nonces is not None:
                reuse = nonces.check(link)
                if reuse:
                    errors.append(f"link {index}: {reuse}")

            if link.wards is not None:
                try:
                    wards = WardSet.from_dict(link.wards)
                except (TypeError, ValueError, AttributeError):
                    errors.append(f"link {index}: malformed wards block")
                    wards = None
                if wards is not None:
                    if prev_wards is not None:
                        widened_dims = never_widens(prev_wards, wards)
                        if widened_dims:
                            errors.append(
                                f"link {index}: wards widen beyond parent grant: "
                                + "; ".join(widened_dims)
                            )
                    prev_wards = wards

            if prev_subject is not None and link.issuer_did != prev_subject:
                errors.append(
                    f"link {index}: issuer {link.issuer_did} does not match "
                    f"prior subject {prev_subject}"
                )

            permissions = set(link.permissions)
            if prev_permissions is not None and not permissions <= prev_permissions:
                widened = sorted(permissions - prev_permissions)
                errors.append(
                    f"link {index}: permissions widen beyond parent grant: {widened}"
                )

            if link.not_after is not None:
                # verify() is a security boundary over untrusted bundle data;
                # a malformed timestamp must surface as an error, never raise.
                try:
                    expiry = datetime.fromisoformat(link.not_after)
                except ValueError:
                    errors.append(f"link {index}: malformed not_after {link.not_after!r}")
                else:
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=timezone.utc)
                    if now > expiry:
                        errors.append(
                            f"link {index}: delegation expired at {link.not_after}"
                        )

            prev_permissions = permissions
            prev_subject = link.subject_did

        if expected_subject_did is not None and self.subject_did != expected_subject_did:
            errors.append(
                f"chain subject {self.subject_did} does not match expected "
                f"agent {expected_subject_did}"
            )

        if require_context and not self.links[-1].is_bound:
            errors.append(
                "final link is an unbound bearer credential; context binding required"
            )

        return not errors, errors

    def to_dict(self) -> Dict[str, Any]:
        return {"links": [link.to_dict() for link in self.links]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelegationChain":
        return cls(links=[DelegationLink.from_dict(item) for item in data.get("links", [])])
