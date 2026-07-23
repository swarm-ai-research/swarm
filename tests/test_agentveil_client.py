"""Regression + mitigation tests for AVPClient.

Covers:
  - Codex P2 fix: _is_valid_hex enforces 64-char lowercase SHA-256.
  - Codex P2 fix: fixture tiers route through _validate_tier.
  - B5 (did:key whitelist): only "did:key:…" DIDs are accepted by every
    method that takes a DID.
  - D6 (deterministic pubkey): unknown DIDs get a SHA-256(did)-derived
    pubkey; same DID → same pubkey across calls and instances.
  - E3 (pre-hashed evidence): submit_attestation accepts only canonical
    SHA-256 hex evidence; raw payloads and short/long hex are rejected.
  - Input validators: DID format, tier set, outcome_sign ±1, hex
    evidence — exhaustively parameterised.
"""

from __future__ import annotations

import hashlib

import pytest

from swarm.bridges.agentveil.client import AVPClient

VALID_DID = "did:key:z6MkqaliceTestDID0000000000000000000000000"


@pytest.fixture
def client() -> AVPClient:
    return AVPClient()


def _sha256_hex(payload: bytes = b"x") -> str:
    return hashlib.sha256(payload).hexdigest()


class TestEvidenceHashLengthEnforced:
    """E3: evidence_hash must be a 64-char lowercase SHA-256 hex digest."""

    def test_valid_sha256_accepted(self, client):
        r = client.submit_attestation(VALID_DID, +1, _sha256_hex())
        assert r.accepted is True
        assert r.attestation_id.startswith("mock-")

    @pytest.mark.parametrize(
        "bad_hex",
        [
            "",                  # empty
            "deadbeef",          # 8 chars (valid hex, wrong length)
            "a" * 63,            # 63 chars (one short)
            "a" * 65,            # 65 chars (one long)
            "a" * 128,           # 128 chars (SHA-512 length)
            _sha256_hex().upper(),  # uppercase form (canonicalization)
        ],
    )
    def test_wrong_length_or_case_rejected(self, client, bad_hex):
        with pytest.raises(ValueError, match="evidence_hash"):
            client.submit_attestation(VALID_DID, +1, bad_hex)

    def test_non_hex_rejected(self, client):
        with pytest.raises(ValueError, match="evidence_hash"):
            client.submit_attestation(VALID_DID, +1, "z" * 64)


class TestFixtureTierValidated:
    """Mock fixtures must route through _validate_tier so typo'd tiers
    can't leak into TrustDecision payloads."""

    def test_valid_fixture_tier_accepted(self):
        c = AVPClient(fixtures={VALID_DID: {"tier": "elite"}})
        d = c.can_trust(VALID_DID, min_tier="basic")
        assert d.tier == "elite"
        assert d.allowed is True

    def test_typoed_fixture_tier_rejected(self):
        c = AVPClient(fixtures={VALID_DID: {"tier": "trustd"}})
        with pytest.raises(ValueError, match="Invalid tier"):
            c.can_trust(VALID_DID, min_tier="basic")

    def test_empty_fixture_tier_falls_back_to_newcomer(self):
        c = AVPClient(fixtures={VALID_DID: {}})
        d = c.can_trust(VALID_DID, min_tier="newcomer")
        assert d.tier == "newcomer"


# ---------------------------------------------------------------- B5

class TestB5DIDKeyWhitelist:
    """B5: only did:key:… DIDs accepted by every DID-taking method."""

    @pytest.fixture
    def client(self):
        return AVPClient()

    @pytest.mark.parametrize("method", ["resolve_did", "get_reputation"])
    def test_method_rejects_non_did_key(self, client, method):
        bad_dids = [
            "did:web:example.com",
            "did:peer:abc",
            "did:ion:123",
            "alice",
            "",
            "did:key",          # missing trailing colon
            "DID:KEY:alice",    # case sensitive
        ]
        for did in bad_dids:
            with pytest.raises(ValueError, match="did:key"):
                getattr(client, method)(did)

    def test_can_trust_rejects_non_did_key(self, client):
        with pytest.raises(ValueError, match="did:key"):
            client.can_trust("did:web:example.com", min_tier="basic")

    def test_submit_attestation_rejects_non_did_key(self, client):
        good_hex = _sha256_hex()
        with pytest.raises(ValueError, match="did:key"):
            client.submit_attestation("did:web:example.com", +1, good_hex)

    def test_valid_did_key_accepted(self, client):
        doc = client.resolve_did(VALID_DID)
        assert doc.did == VALID_DID


# ---------------------------------------------------------------- D6

class TestD6DeterministicPubkey:
    """D6: SHA-256(did) → deterministic pubkey for unknown DIDs."""

    def test_same_did_same_pubkey_across_calls(self):
        c = AVPClient()
        d1 = c.resolve_did(VALID_DID)
        d2 = c.resolve_did(VALID_DID)
        assert d1.pubkey_hex == d2.pubkey_hex

    def test_same_did_same_pubkey_across_instances(self):
        c1 = AVPClient()
        c2 = AVPClient()
        assert c1.resolve_did(VALID_DID).pubkey_hex == c2.resolve_did(VALID_DID).pubkey_hex

    def test_pubkey_matches_sha256_of_did(self):
        c = AVPClient()
        doc = c.resolve_did(VALID_DID)
        expected = hashlib.sha256(VALID_DID.encode()).hexdigest()[:64]
        assert doc.pubkey_hex == expected

    def test_different_dids_different_pubkeys(self):
        c = AVPClient()
        other_did = "did:key:z6MkbobSecondDID00000000000000000000000000"
        assert c.resolve_did(VALID_DID).pubkey_hex != c.resolve_did(other_did).pubkey_hex

    def test_pubkey_is_32_byte_ed25519_shape(self):
        # Ed25519 pubkeys are 32 bytes = 64 hex chars.
        c = AVPClient()
        doc = c.resolve_did(VALID_DID)
        assert len(doc.pubkey_hex) == 64
        int(doc.pubkey_hex, 16)  # valid hex

    def test_fixture_pubkey_overrides_deterministic(self):
        override = "ab" * 32
        c = AVPClient(fixtures={VALID_DID: {"pubkey_hex": override}})
        assert c.resolve_did(VALID_DID).pubkey_hex == override


# ---------------------------------------------------------------- E3

class TestE3PreHashedEvidence:
    """E3: only canonical SHA-256 hex evidence accepted; never raw payload."""

    @pytest.fixture
    def client(self):
        return AVPClient()

    def test_canonical_sha256_accepted(self, client):
        r = client.submit_attestation(VALID_DID, +1, _sha256_hex(b"payload"))
        assert r.accepted is True

    def test_raw_payload_rejected(self, client):
        # A caller that accidentally passes the raw payload instead of
        # SHA-256 of it: even if the raw bytes happen to be hex-coded
        # they're the wrong length.
        raw_hex = b"interaction_id_or_whatever_string".hex()
        with pytest.raises(ValueError, match="evidence_hash"):
            client.submit_attestation(VALID_DID, +1, raw_hex)

    def test_sha512_rejected(self, client):
        # SHA-512 hex (128 chars) is exactly the wrong-length trap E3
        # is supposed to catch.
        bad = hashlib.sha512(b"x").hexdigest()
        assert len(bad) == 128
        with pytest.raises(ValueError, match="evidence_hash"):
            client.submit_attestation(VALID_DID, +1, bad)

    def test_canonical_attestation_id_is_deterministic(self, client):
        # Same (did, sign, hash) → same attestation_id (replayable).
        ev = _sha256_hex(b"payload")
        r1 = client.submit_attestation(VALID_DID, +1, ev)
        c2 = AVPClient()
        r2 = c2.submit_attestation(VALID_DID, +1, ev)
        assert r1.attestation_id == r2.attestation_id

    def test_attestation_id_changes_with_evidence_hash(self, client):
        ev1 = _sha256_hex(b"payload-a")
        ev2 = _sha256_hex(b"payload-b")
        r1 = client.submit_attestation(VALID_DID, +1, ev1)
        r2 = client.submit_attestation(VALID_DID, +1, ev2)
        assert r1.attestation_id != r2.attestation_id


# ---------------------------------------------------------------- Validators

class TestInputValidators:
    """Exhaustive parameterisation of the four input validators."""

    @pytest.fixture
    def client(self):
        return AVPClient()

    # ----- outcome_sign

    @pytest.mark.parametrize("bad_sign", [0, 2, -2, +0.5, "+1", None])
    def test_outcome_sign_must_be_plus_or_minus_one(self, client, bad_sign):
        with pytest.raises((ValueError, TypeError)):
            client.submit_attestation(VALID_DID, bad_sign, _sha256_hex())

    @pytest.mark.xfail(
        reason="bool subclasses int in Python: True == +1 so True sneaks "
               "through `outcome_sign in (+1, -1)`. Worth adding an "
               "isinstance(..., bool) gate to _validate_outcome_sign.",
        strict=True,
    )
    def test_outcome_sign_rejects_true(self, client):
        with pytest.raises((ValueError, TypeError)):
            client.submit_attestation(VALID_DID, True, _sha256_hex())

    @pytest.mark.parametrize("good_sign", [+1, -1])
    def test_outcome_sign_valid_values(self, client, good_sign):
        r = client.submit_attestation(VALID_DID, good_sign, _sha256_hex())
        assert r.accepted is True

    # ----- min_tier

    @pytest.mark.parametrize("good_tier", ["newcomer", "basic", "trusted", "elite"])
    def test_min_tier_whitelist(self, client, good_tier):
        d = client.can_trust(VALID_DID, min_tier=good_tier)
        assert d.tier in {"newcomer", "basic", "trusted", "elite"}

    @pytest.mark.parametrize("bad_tier", ["", "ELITE", "premium", "trustd", "0", "newcomer "])
    def test_min_tier_rejects_unknown(self, client, bad_tier):
        with pytest.raises(ValueError, match="Invalid tier"):
            client.can_trust(VALID_DID, min_tier=bad_tier)

    # ----- DID format already exercised in B5 above; one more positive case.

    def test_did_format_with_long_suffix_accepted(self, client):
        long_did = "did:key:z" + "ab" * 32
        doc = client.resolve_did(long_did)
        assert doc.did == long_did

    # ----- evidence hex already exercised in E3 + the original
    # TestEvidenceHashLengthEnforced block; add the type-coercion case.

    def test_evidence_hash_must_be_string(self, client):
        for bad in (None, 12345, b"a" * 64):
            with pytest.raises((ValueError, TypeError)):
                client.submit_attestation(VALID_DID, +1, bad)
