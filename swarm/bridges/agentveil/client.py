"""Mock-mode AVP client for the SWARM–AgentVeil bridge (v1).

Implements the four core AVP SDK methods (resolve_did, can_trust,
submit_attestation, get_reputation) in deterministic, replayable mock mode.

v1 is mock-only; live registry HTTP lands in v2.

Mitigations implemented:
- B5 (signature algorithm confusion): whitelist did:key only
- D6 (non-deterministic replay): fully deterministic mock fixtures
- E3 (privacy leak via write-back): client accepts only pre-hashed evidence
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Return dataclasses (v1 return types, distinct from event payloads)


@dataclass
class DIDDocument:
    """W3C DID document with public key.

    In v1, this is minimal; v2 will include full DID resolution with
    controller, authentication methods, key rotation history, etc.
    """

    did: str
    pubkey_hex: str  # Ed25519 public key (hex-encoded, 32 bytes = 64 hex chars)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "did": self.did,
            "pubkey_hex": self.pubkey_hex,
        }


@dataclass
class TrustDecision:
    """Trust decision returned by can_trust().

    Matches the TrustDecisionEvent payload shape from events.py,
    but is a plain dataclass (not the event itself).
    """

    allowed: bool
    tier: str  # newcomer | basic | trusted | elite
    risk_level: str  # low | medium | high
    confidence: float  # AVP Bayesian confidence, in [0, 1]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "tier": self.tier,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class AttestationReceipt:
    """Receipt for submit_attestation().

    In v1 mock mode, always returns accepted=True with a deterministic ID.
    """

    accepted: bool
    attestation_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "attestation_id": self.attestation_id,
        }


@dataclass
class ReputationSnapshot:
    """Current reputation of an agent.

    Matches the ReputationSnapshotEvent payload shape from events.py,
    but is a plain dataclass (not the event itself).
    """

    score: float  # AVP Bayesian score, in [0, 1]
    confidence: float  # in [0, 1]
    tier: str  # newcomer | basic | trusted | elite
    attestation_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "tier": self.tier,
            "attestation_count": self.attestation_count,
        }


# Client


class AVPClient:
    """Mock-mode AVP client (v1).

    Wraps the four core AVP SDK methods with deterministic, replayable
    behavior. All methods are driven by optional fixtures; unknown DIDs
    get sensible defaults.

    Fixtures format:
        fixtures = {
            "did:key:alice": {
                "tier": "trusted",
                "allowed": True,
                "risk_level": "low",
                "confidence": 0.9,
                "reputation_score": 0.8,
                "attestation_count": 10,
                "pubkey_hex": "<hex_string>",
            },
            ...
        }

    For DIDs not in fixtures, sensible defaults are generated.

    All validation happens at the client layer:
    - DIDs must start with "did:key:" (B5 mitigation)
    - min_tier must be one of {"newcomer", "basic", "trusted", "elite"}
    - outcome_sign must be +1 or -1
    - Deterministic pubkey for unknown DIDs via SHA256(did)
    """

    VALID_TIERS = {"newcomer", "basic", "trusted", "elite"}
    TIER_ORDINALS = {"newcomer": 0, "basic": 1, "trusted": 2, "elite": 3}

    def __init__(
        self,
        mock_mode: bool = True,
        fixtures: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Initialize AVPClient.

        Args:
            mock_mode: If False, raises NotImplementedError (v2 feature).
                Defaults to True for v1.
            fixtures: Optional DID → attribute dict for deterministic behavior.
                If a DID is not in fixtures, sensible defaults are generated.
        """
        self._mock_mode = mock_mode
        self._fixtures = fixtures or {}
        self._submitted: list[AttestationReceipt] = []

    def resolve_did(self, did: str) -> DIDDocument:
        """Resolve a DID to its document and public key.

        Args:
            did: W3C DID string (must start with "did:key:")

        Returns:
            DIDDocument with did and pubkey_hex

        Raises:
            ValueError: if DID is not "did:key:" format (B5 mitigation)
            NotImplementedError: if mock_mode=False
        """
        if not self._mock_mode:
            raise NotImplementedError("Live registry support lands in v2")

        self._validate_did(did)

        # Check fixtures first
        if did in self._fixtures:
            fixture = self._fixtures[did]
            pubkey_hex = fixture.get("pubkey_hex")
            if pubkey_hex:
                return DIDDocument(did=did, pubkey_hex=pubkey_hex)

        # Generate deterministic pubkey for unknown DIDs (D6 mitigation)
        pubkey_hex = self._deterministic_pubkey(did)
        return DIDDocument(did=did, pubkey_hex=pubkey_hex)

    def can_trust(self, did: str, min_tier: str = "basic") -> TrustDecision:
        """Check if a DID is trusted at or above min_tier.

        Args:
            did: W3C DID string (must start with "did:key:")
            min_tier: Minimum acceptable tier.
                One of {"newcomer", "basic", "trusted", "elite"}.
                Defaults to "basic".

        Returns:
            TrustDecision with allowed, tier, risk_level, confidence, reason

        Raises:
            ValueError: if DID is invalid or min_tier is not recognized
            NotImplementedError: if mock_mode=False
        """
        if not self._mock_mode:
            raise NotImplementedError("Live registry support lands in v2")

        self._validate_did(did)
        self._validate_tier(min_tier)

        # Check fixtures
        if did in self._fixtures:
            fixture = self._fixtures[did]
            tier = fixture.get("tier", "newcomer")
            # Fixtures are the source of deterministic replay data, but they
            # must not bypass the tier whitelist — a typo like "trustd" would
            # otherwise produce TrustDecision payloads outside the documented
            # {newcomer, basic, trusted, elite} set.
            self._validate_tier(tier)
            allowed = fixture.get("allowed", self._tier_meets_min(tier, min_tier))
            risk_level = fixture.get("risk_level", self._tier_to_risk_level(tier))
            confidence = fixture.get("confidence", 0.8)
            reason = fixture.get("reason", f"Fixture tier {tier}")
            return TrustDecision(
                allowed=allowed,
                tier=tier,
                risk_level=risk_level,
                confidence=confidence,
                reason=reason,
            )

        # Default behavior for unknown DIDs
        tier = "newcomer"
        allowed = min_tier == "newcomer"  # only allow if min_tier is newcomer
        risk_level = "high"
        confidence = 0.5
        reason = f"Unknown DID {did}; default tier={tier}"

        return TrustDecision(
            allowed=allowed,
            tier=tier,
            risk_level=risk_level,
            confidence=confidence,
            reason=reason,
        )

    def submit_attestation(
        self,
        subject_did: str,
        outcome_sign: int,
        evidence_hash: str,
    ) -> AttestationReceipt:
        """Submit an attestation for a subject DID.

        Per failure-mode E3, the client accepts only a pre-hashed evidence
        (SHA-256 hash of interaction_id || outcome_sign), not the raw p.
        The bridge pre-computes this hash and passes it here.

        Args:
            subject_did: DID being attested about (must start with "did:key:")
            outcome_sign: +1 (positive attestation) or -1 (negative)
            evidence_hash: Hex-encoded SHA-256 hash of (interaction_id || outcome_sign)

        Returns:
            AttestationReceipt with accepted and attestation_id

        Raises:
            ValueError: if DID is invalid, outcome_sign is not ±1, or evidence_hash
                is malformed
            NotImplementedError: if mock_mode=False
        """
        if not self._mock_mode:
            raise NotImplementedError("Live registry support lands in v2")

        self._validate_did(subject_did)
        self._validate_outcome_sign(outcome_sign)

        # Verify evidence_hash is a valid hex string (simple format check)
        if not self._is_valid_hex(evidence_hash):
            raise ValueError(
                f"evidence_hash must be valid hex; got {evidence_hash!r}"
            )

        # Generate deterministic attestation_id from subject_did, outcome_sign, hash
        combined = f"{subject_did}||{outcome_sign}||{evidence_hash}".encode()
        attestation_id_hash = hashlib.sha256(combined).hexdigest()[:16]
        attestation_id = f"mock-{attestation_id_hash}"

        receipt = AttestationReceipt(accepted=True, attestation_id=attestation_id)
        self._submitted.append(receipt)
        logger.debug(f"Attestation submitted: {attestation_id}")

        return receipt

    def get_reputation(self, did: str) -> ReputationSnapshot:
        """Fetch the current reputation of a DID.

        Args:
            did: W3C DID string (must start with "did:key:")

        Returns:
            ReputationSnapshot with score, confidence, tier, attestation_count

        Raises:
            ValueError: if DID is invalid
            NotImplementedError: if mock_mode=False
        """
        if not self._mock_mode:
            raise NotImplementedError("Live registry support lands in v2")

        self._validate_did(did)

        # Check fixtures
        if did in self._fixtures:
            fixture = self._fixtures[did]
            return ReputationSnapshot(
                score=float(fixture.get("reputation_score", 0.5)),
                confidence=float(fixture.get("confidence", 0.8)),
                tier=fixture.get("tier", "newcomer"),
                attestation_count=int(fixture.get("attestation_count", 0)),
            )

        # Default for unknown DIDs
        return ReputationSnapshot(
            score=0.5,
            confidence=0.5,
            tier="newcomer",
            attestation_count=0,
        )

    # Private helpers

    @staticmethod
    def _validate_did(did: str) -> None:
        """Validate that DID is in "did:key:" format (B5 mitigation)."""
        if not did.startswith("did:key:"):
            raise ValueError(
                f"Only did:key DIDs supported; got {did!r} (B5 mitigation)"
            )

    @staticmethod
    def _validate_tier(tier: str) -> None:
        """Validate tier is in the allowed set."""
        if tier not in AVPClient.VALID_TIERS:
            raise ValueError(
                f"Invalid tier {tier!r}; must be one of {AVPClient.VALID_TIERS}"
            )

    @staticmethod
    def _validate_outcome_sign(outcome_sign: int) -> None:
        """Validate outcome_sign is +1 or -1."""
        if outcome_sign not in (+1, -1):
            raise ValueError(f"outcome_sign must be +1 or -1; got {outcome_sign}")

    # SHA-256 hex digests are always 64 characters. The E3 contract is that
    # callers pass `SHA-256(interaction_id || outcome_sign)`, so anything
    # shorter or longer than 64 hex chars is malformed and must be rejected
    # — otherwise a string like "deadbeef" (8 chars) or a raw hex-encoded
    # payload would silently get an attestation receipt.
    _SHA256_HEX_LEN = 64

    @staticmethod
    def _is_valid_hex(s: str) -> bool:
        """Check if string is a valid lowercase SHA-256 hex digest."""
        if not isinstance(s, str) or len(s) != AVPClient._SHA256_HEX_LEN:
            return False
        try:
            int(s, 16)
        except ValueError:
            return False
        # E3: enforce the canonical lowercase form so hex matching is total.
        return s == s.lower()

    @staticmethod
    def _deterministic_pubkey(did: str) -> str:
        """Generate a deterministic Ed25519 pubkey from a DID (D6 mitigation).

        Uses SHA256(did) to generate the first 32 bytes (256 bits) of a
        deterministic fake public key. This is NOT a real key but is
        deterministic and replayable.
        """
        h = hashlib.sha256(did.encode()).hexdigest()
        # Ed25519 pubkeys are 32 bytes = 64 hex chars; use first 64 chars
        return h[:64]

    @staticmethod
    def _tier_meets_min(tier: str, min_tier: str) -> bool:
        """Check if tier meets minimum."""
        return AVPClient.TIER_ORDINALS.get(tier, -1) >= AVPClient.TIER_ORDINALS.get(
            min_tier, -1
        )

    @staticmethod
    def _tier_to_risk_level(tier: str) -> str:
        """Map tier to risk level."""
        mapping = {
            "newcomer": "high",
            "basic": "medium",
            "trusted": "low",
            "elite": "low",
        }
        return mapping.get(tier, "unknown")

    # Introspection (for testing/debugging)

    @property
    def submitted_attestations(self) -> list[AttestationReceipt]:
        """Return list of submitted attestations (for testing)."""
        return self._submitted.copy()


if __name__ == "__main__":
    # Smoke test
    client = AVPClient(
        fixtures={
            "did:key:alice": {
                "tier": "trusted",
                "allowed": True,
                "risk_level": "low",
                "confidence": 0.95,
                "reputation_score": 0.85,
                "attestation_count": 12,
            }
        }
    )

    # Test resolve_did
    doc = client.resolve_did("did:key:alice")
    print(f"DID: {doc.did}, pubkey_hex: {doc.pubkey_hex[:16]}...")

    # Test can_trust
    decision = client.can_trust("did:key:alice", min_tier="basic")
    print(
        f"Trust decision for alice: allowed={decision.allowed}, "
        f"tier={decision.tier}, confidence={decision.confidence}"
    )

    # Test unknown DID
    unknown = client.can_trust("did:key:unknown", min_tier="basic")
    print(f"Unknown DID: allowed={unknown.allowed}, tier={unknown.tier}")

    # Test submit_attestation
    evidence_hash = hashlib.sha256(b"interaction123||+1").hexdigest()
    receipt = client.submit_attestation("did:key:alice", +1, evidence_hash)
    print(f"Attestation submitted: {receipt.attestation_id}")

    # Test get_reputation
    rep = client.get_reputation("did:key:alice")
    print(f"Reputation: score={rep.score}, tier={rep.tier}, count={rep.attestation_count}")

    print("\nAll smoke tests passed!")
