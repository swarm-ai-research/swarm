"""Receipt signing and verification.

Uses HMAC-SHA256 for receipt sealing.  The signer holds the shared secret;
the verifier only needs the same key to confirm receipt integrity.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional

from swarm.attestation.receipt import AdmissibilityReceipt, ReceiptStatus


class ReceiptSigner:
    """Signs admissibility receipts with HMAC-SHA256.

    Parameters
    ----------
    secret_key:
        Hex-encoded shared secret.  If *None*, a fresh 256-bit key is
        generated (useful for single-run simulations).
    signer_id:
        Identity string embedded in every receipt (e.g. ``"platform"``).
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        signer_id: str = "swarm-platform",
    ) -> None:
        if secret_key is None:
            secret_key = secrets.token_hex(32)
        self._key = bytes.fromhex(secret_key)
        self.signer_id = signer_id

    @property
    def secret_key_hex(self) -> str:
        """Return the hex-encoded key (for sharing with verifiers)."""
        return self._key.hex()

    def seal(self, receipt: AdmissibilityReceipt) -> AdmissibilityReceipt:
        """Seal a receipt by computing and attaching its HMAC signature.

        The receipt's ``status`` is moved to ``SEALED`` and the ``signature``
        and ``signer_id`` fields are populated.  Returns a *new* receipt
        instance (receipts are treated as immutable after sealing).
        """
        if receipt.status != ReceiptStatus.PENDING:
            raise ValueError(
                f"Cannot seal receipt in status {receipt.status!r}; "
                "only PENDING receipts may be sealed"
            )

        # Compute HMAC over the canonical (pre-signature) bytes
        canonical = receipt.canonical_bytes()
        sig = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()

        return receipt.model_copy(
            update={
                "status": ReceiptStatus.SEALED,
                "signature": sig,
                "signer_id": self.signer_id,
            }
        )


class ReceiptVerifier:
    """Verifies sealed admissibility receipts.

    Parameters
    ----------
    secret_key:
        Hex-encoded shared secret (same key used by the signer).
    """

    def __init__(self, secret_key: str) -> None:
        self._key = bytes.fromhex(secret_key)

    def verify(self, receipt: AdmissibilityReceipt) -> bool:
        """Return *True* if the receipt's signature is valid.

        Checks:
        1. The receipt is in SEALED or VERIFIED status.
        2. The HMAC over the canonical bytes matches the stored signature.
        """
        if receipt.status not in (ReceiptStatus.SEALED, ReceiptStatus.VERIFIED):
            return False
        if receipt.signature is None:
            return False

        # Recompute over canonical (excludes signature, signer_id, & status)
        canonical = receipt.canonical_bytes()
        expected = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, receipt.signature)

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
