"""Tests for the observable -> Beta belief proxy."""

import pytest

from beta_swarm.proxy import BetaProxyComputer, ProxyObservables


def test_more_evidence_sharpens_belief():
    proxy = BetaProxyComputer()
    sparse = ProxyObservables(task_progress_delta=0.5)
    rich = ProxyObservables(
        task_progress_delta=0.5, rework_count=2, verifier_rejections=1
    )
    assert (
        proxy.compute_belief(rich).concentration
        > proxy.compute_belief(sparse).concentration
    )


def test_positive_signal_raises_mean():
    proxy = BetaProxyComputer()
    good = proxy.compute_belief(ProxyObservables(task_progress_delta=1.0))
    bad = proxy.compute_belief(
        ProxyObservables(task_progress_delta=-1.0, verifier_rejections=3)
    )
    assert good.mean > bad.mean


def test_no_evidence_stays_at_base_concentration():
    proxy = BetaProxyComputer(base_concentration=2.0)
    b = proxy.compute_belief(ProxyObservables())
    assert b.concentration == pytest.approx(2.0)


def test_v_hat_clamped():
    proxy = BetaProxyComputer()
    v = proxy.compute_v_hat(ProxyObservables(task_progress_delta=5.0))
    assert -1.0 <= v <= 1.0


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        BetaProxyComputer(w_progress=-1.0)
