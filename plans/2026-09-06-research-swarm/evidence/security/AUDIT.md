# Wave 1 security verification

Role: independent Auditor. Date: 2026-09-06. Repository HEAD: `7dcb672b28872373ba991f0dd3680ce6a3ad5c56`. Read-only review; no tracked files, issue statuses, ownership, or claims changed. Existing unrelated `.beads/interactions.jsonl` change and untracked `.beads/hooks/`, `plans/` observed before and after. Tests used `/opt/anaconda3/bin/python` (Python 3.13).

## Executed gates

- PASS: `python -m pytest tests/test_attestation.py tests/test_agentgit_identity.py tests/test_agentgit_push_tokens.py tests/test_agentgit_review.py -q -o addopts='' -o cache_dir=/tmp/swarm-wave1-security/pytest-cache` — **123 passed**, 109 datetime deprecation warnings, 9.20 seconds. Log: `regressions.log`.
- PASS: same existing qkzc regression individually with a synthetic non-default `AGENTGIT_SIGNING_KEY` and `-n 2` — **1 passed**, 1 datetime warning, 2.10 seconds. Log: `qkzc-env-xdist.log`. Test node: `tests/test_agentgit_review.py::test_cli_attest_review_folds_panel_into_signed_bundle`.
- PASS as characterization, NOT as secure-default acceptance: `PYTHONPATH=/Users/raelisavitt/swarm python /tmp/swarm-wave1-security/residual_probe.py` — exit 0; six explicit behavior assertions. Output: `residual-probe.json`.

## Bead findings

### jxyi: receipt signing

Implementing commit: `e4438251ff4b953e0ad58555d7a9cb4eeade4910`.

The stated acceptance criteria are supported at HEAD: `ReceiptSigner` uses an `AgentKeypair`; signer identity derives from its Ed25519 DID; verification uses the public key embedded in that DID. Tests execute verifier-without-secret, signer-ID spoofing refusal, inability to mint as an existing signer, unknown signer rejection with an explicit trust set, and legacy HMAC behavior.

Limits: `ReceiptVerifier()` without trusted signers establishes cryptographic integrity, not authorization of a claimed agent identity. Receipt `agent_id` remains a separate payload field and is not itself constrained to equal the signing DID. Explicit legacy HMAC compatibility retains the old shared-key trust model for old receipts. These are boundaries to state in research methods, not evidence the narrower signer-ID acceptance failed. Do not claim all provenance forgery is impossible.

### zsof: context-bound delegation

Implementing commit: `ff8a2a6862ea1bbb0e2f1b569bdcb44f45e2fedc`.

PASS: nonce and audience enter the signed payload when present; default signing creates a nonce; a signed bound link rejects a foreign context or absent context; signed wards are covered; provided registry rejects the same nonce with a different signed payload and persists that memory. Existing tests verify these behaviors.

**Acceptance remains PARTIAL as written.** The issue says `DelegationChain.verify()` rejects nonce reuse. Its default is `nonces=None`, so a different valid payload reusing a nonce passes by default. `swarm/agentgit/bundle.py:_verify_identity` passes task context but no nonce registry and no `require_context=True`. Unbound grants therefore still pass normal verification in arbitrary contexts. Same-link replay is explicitly allowed even when a registry is supplied, so the implementation does not establish single-use grants. The probe reproduces all three boundaries. The foreign-context bound-link criterion itself is met; it would be inaccurate to say this rejection depends on require_context (that flag instead refuses unbound final links).

Before closure: owner should either strengthen and wire secure defaults with replay policy tests, or explicitly amend acceptance to permit reusable credentials, legacy unbound grants, and caller-managed nonce collision checking. Scope context identifiers to the intended task/session/repository threat model; task_id equality alone does not establish global uniqueness. No claim of cross-process atomic nonce storage was tested here.

### qkzc: ambient signing environment

Fixing commit: `e4438251ff4b953e0ad58555d7a9cb4eeade4910`.

PASS: the named test removes `AGENTGIT_SIGNING_KEY` with monkeypatch before CLI invocation, aligning its implicit verification key with signing. Both normal suite execution and a synthetic non-default ambient signing key with xdist pass. This supports the specific latent test fragility fix; it is not a whole-suite CI claim.

## Negative specification and handoff

Insufficient outcomes: (1) source inspection without executed regression logs; (2) security closure inferred from passing tests whose fixture opts into protections absent in production callers; (3) treating same-link repeat verification as prevented replay, or public-key integrity as identity authorization. No negative-spec stamp was written to beads because this task explicitly forbids task mutations.

Recommended status wording: “jxyi and qkzc current-HEAD regression gates pass; zsof bound-context defenses pass, but default nonce/legacy-context enforcement leaves the original acceptance partially unmet.” All three beads were open at read time; none was closed. No new dependency, external service, remote CI run, full-suite gate, or production deployment was performed.
