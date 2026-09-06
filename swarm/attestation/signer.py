"""Receipt signing and verification.

Receipts are sealed with a per-signer **Ed25519** key. The receipt's
``signer_id`` *is* the signing key's DID (``did:key:ed25519:<hex>``), so a
verifier needs no secret and no key registry: it reads the public key out of
``signer_id`` and checks the signature. Changing ``signer_id`` changes the key
the verifier checks against, so the field cannot be spoofed.

This replaced a shared HMAC-SHA256 secret (bead ``jxyi``): under HMAC anyone
able to *verify* a receipt could also *mint* one under any ``agent_id`` — the
single-global-key finding from Matthew Green's "Let's talk about encrypted
reasoning" (2026-05-29), reproduced in our own attestation layer. Receipts
sealed before the change (a ``signer_id`` that is not a DID) still verify when
the verifier is given the old key as ``legacy_hmac_key``, so stored event logs
and git notes remain replayable; new receipts are never sealed with HMAC.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import TYPE_CHECKING, Iterable, Optional

from swarm.attestation.receipt import AdmissibilityReceipt, ReceiptStatus

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, see _identity()
    from swarm.agentgit.identity import AgentKeypair


def _identity():  # type: ignore[no-untyped-def]
    """Import ``swarm.agentgit.identity`` lazily.

    ``swarm.agentgit`` imports this module (via ``bundle``), so a top-level
    import here would be circular.
    """

    from swarm.agentgit import identity

    return identity


def is_did_signer(signer_id: Optional[str]) -> bool:
    """True when ``signer_id`` names an Ed25519 DID (i.e. not a legacy receipt)."""

    return bool(signer_id) and str(signer_id).startswith(_identity().DID_PREFIX)


class ReceiptSigner:
    """Signs admissibility receipts with a per-signer Ed25519 key.

    Parameters
    ----------
    seed_hex:
        32-byte hex Ed25519 seed. If *None* and no ``keypair`` is given, a
        fresh key is generated (useful for single-run simulations).
    keypair:
        An existing :class:`~swarm.agentgit.identity.AgentKeypair` to sign
        with. Mutually exclusive with ``seed_hex``.

    The signer's identity is ``did`` — that string is written into every
    sealed receipt's ``signer_id`` and is the only thing a verifier needs.
    """

    def __init__(
        self,
        seed_hex: Optional[str] = None,
        *,
        keypair: Optional["AgentKeypair"] = None,
    ) -> None:
        if seed_hex is not None and keypair is not None:
            raise ValueError("pass seed_hex or keypair, not both")
        identity = _identity()
        if keypair is None:
            keypair = (
                identity.AgentKeypair.generate()
                if seed_hex is None
                else identity.AgentKeypair.from_seed_hex(seed_hex)
            )
        self._keypair = keypair

    @property
    def did(self) -> str:
        """The signer's DID; embedded in each sealed receipt as ``signer_id``."""

        return self._keypair.did

    @property
    def public_key_hex(self) -> str:
        return self._keypair.public_key_hex

    @property
    def signer_id(self) -> str:
        """Alias for :attr:`did` — the value written into ``receipt.signer_id``."""

        return self.did

    def seal(self, receipt: AdmissibilityReceipt) -> AdmissibilityReceipt:
        """Seal a receipt by signing its canonical bytes.

        The receipt's ``status`` is moved to ``SEALED`` and the ``signature``
        and ``signer_id`` fields are populated.  Returns a *new* receipt
        instance (receipts are treated as immutable after sealing).
        """
        if receipt.status != ReceiptStatus.PENDING:
            raise ValueError(
                f"Cannot seal receipt in status {receipt.status!r}; "
                "only PENDING receipts may be sealed"
            )

        # Sign the canonical (pre-signature) bytes. signer_id is excluded from
        # the canonical form but is still bound: it names the verifying key.
        canonical = receipt.canonical_bytes()
        sig = self._keypair.sign(canonical)

        return receipt.model_copy(
            update={
                "status": ReceiptStatus.SEALED,
                "signature": sig,
                "signer_id": self.did,
            }
        )


class ReceiptVerifier:
    """Verifies sealed admissibility receipts. Needs no secret.

    Parameters
    ----------
    trusted_signers:
        Optional set of DIDs. When given, a receipt is valid only if its
        ``signer_id`` is one of them — a correct signature from an unknown
        key is rejected. When *None*, any well-formed DID signer is accepted
        (integrity without authentication).
    legacy_hmac_key:
        Hex key for receipts sealed under the pre-``jxyi`` HMAC scheme (a
        ``signer_id`` that is not a DID). Without it, legacy receipts fail.
    """

    def __init__(
        self,
        trusted_signers: Optional[Iterable[str]] = None,
        *,
        legacy_hmac_key: Optional[str] = None,
    ) -> None:
        self._trusted = (
            None if trusted_signers is None else frozenset(trusted_signers)
        )
        self._legacy_key = (
            None if legacy_hmac_key is None else bytes.fromhex(legacy_hmac_key)
        )

    @property
    def trusted_signers(self) -> Optional[frozenset[str]]:
        return self._trusted

    def is_legacy(self, receipt: AdmissibilityReceipt) -> bool:
        """True when the receipt was sealed under the old shared-HMAC scheme."""

        return not is_did_signer(receipt.signer_id)

    def verify(self, receipt: AdmissibilityReceipt) -> bool:
        """Return *True* if the receipt's signature is valid.

        Checks:
        1. The receipt is in SEALED or VERIFIED status and carries a signature.
        2. ``signer_id`` is a trusted DID (when a trust set is configured).
        3. The Ed25519 signature over the canonical bytes verifies against the
           key embedded in ``signer_id`` — or, for a legacy receipt, the HMAC
           matches under ``legacy_hmac_key``.
        """
        if receipt.status not in (ReceiptStatus.SEALED, ReceiptStatus.VERIFIED):
            return False
        if receipt.signature is None or not receipt.signer_id:
            return False

        # Canonical form excludes signature, signer_id, and status.
        canonical = receipt.canonical_bytes()

        if self.is_legacy(receipt):
            if self._legacy_key is None:
                return False
            expected = hmac.new(self._legacy_key, canonical, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, receipt.signature)

        if self._trusted is not None and receipt.signer_id not in self._trusted:
            return False
        return bool(
            _identity().verify_signature(receipt.signer_id, canonical, receipt.signature)
        )

    def verify_and_mark(
        self, receipt: AdmissibilityReceipt
    ) -> AdmissibilityReceipt:
        """Verify and transition the receipt to VERIFIED if valid.

        Returns a copy with ``status=VERIFIED`` on success, or
        ``status=REJECTED`` if verification fails.
        """
        if self.verify(receipt):
            return receipt.model_copy(update={"status": ReceiptStatus.VERIFIED})
        return receipt.model_copy(update={"status": ReceiptStatus.REJECTED})
