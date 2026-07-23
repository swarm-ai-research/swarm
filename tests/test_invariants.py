"""Tests backing INVARIANTS.md — each proves a declared repo invariant holds.

If you change one of these, update INVARIANTS.md in the same commit;
scripts/check_invariants.py (CI) verifies the manifest's test references
still collect.
"""
import pytest
from pydantic import ValidationError

from swarm.models.interaction import SoftInteraction


class TestPBounds:
    """INV-1: p must remain in [0, 1] everywhere it is surfaced or logged."""

    def test_p_above_one_rejected(self):
        with pytest.raises(ValidationError):
            SoftInteraction(initiator="a", counterparty="b", p=1.5)

    def test_p_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            SoftInteraction(initiator="a", counterparty="b", p=-0.1)

    def test_boundary_values_accepted(self):
        assert SoftInteraction(initiator="a", counterparty="b", p=0.0).p == 0.0
        assert SoftInteraction(initiator="a", counterparty="b", p=1.0).p == 1.0
