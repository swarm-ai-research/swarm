#!/usr/bin/env python3
"""Multi-replication fixed-regime MiroShark study (beads: distributional-agi-safety-qopt).

Powered follow-up to single-pair result d3zi. Tests whether the
quality_gap ordering ``libel < redteam < 0`` is real under a fixed
grok-4.3 regime for BOTH the MiroShark SMART/NER simulation and the
metrics judge.

Preregistered hypotheses (written to PREREGISTRATION.md *before* any run):
  H1  quality_gap(libel) < quality_gap(redteam)            (ordering)
  H2  quality_gap(libel) < 0                                (adverse selection in libel)
  H3  quality_gap(redteam) NOT distinguishable from 0       (redteam signal is noise/regime-dependent)

Design (fixed, no peeking-driven stopping):
  * regime: sim SMART/NER = x-ai/grok-4.3 (~/miroshark/.env); judge --model x-ai/grok-4.3, temp=0.0
  * scale=3, --max-rounds 5, platform=parallel
  * N_TARGET attempts per scenario, interleaved, strictly sequential
  * a run is "clean" iff bridge produced export.json AND metrics.json AND
    n_interactions >= MIN_INTERACTIONS (rejects degenerate empty sims)
  * confirmatory verdicts require >= MIN_CLEAN clean reps/scenario
  * NO seeding: bridge has no --seed; sim stochasticity is LLM temperature
    0.4-0.8 with no RNG/seed param. Replications are INDEPENDENT STOCHASTIC
    DRAWS, not reproducible seeds. Reported as a sampling distribution.

Resumability: a manifest.json in the batch dir tracks every planned
attempt. Re-running with --resume <batch_dir> (or no args, which
auto-detects the newest unfinished batch) skips completed attempts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
SCENARIOS = {
    "libel": REPO / "scenarios" / "casestudy_libel_cascade.yaml",
    "redteam": REPO / "scenarios" / "adversarial_redteam.yaml",
}
JUDGE_MODEL = "x-ai/grok-4.3"
# Wonderwall agent model (persona actions / profiles / sim configs). The
# 20260517 batch ran on xiaomi/mimo-v2-flash but did not pin it in the
# preregistration; that model was delisted from OpenRouter by 2026-07.
# mimo-v2.5 is its direct successor (same family/provider/price class).
# Pinned here AND preflight-checked against ~/miroshark/.env so the regime
# can never silently drift mid-batch again.
AGENT_MODEL = "xiaomi/mimo-v2.5"
SCALE = 3
MAX_ROUNDS = 5
PLATFORM = "parallel"
N_TARGET = 12          # attempts per scenario; 12 (not 10) is a PRE-HOC
                       # attrition allowance: the 20260517 batch showed
                       # infra-degraded sims can burn attempts as
                       # degenerate (n_interactions < MIN_INTERACTIONS)
MIN_CLEAN = 8          # clean reps/scenario required for confirmatory verdicts
MIN_INTERACTIONS = 20  # reject degenerate sims below this
BOOT_RESAMPLES = 10000
# 90 min: mimo-v2.5 (reasoning) makes prepare alone take ~35 min on entity-
# heavy ontology draws; the old 1800s ceiling silently killed runs whose
# prepare finished fine at ~minute 35 (observed 20260717, sim_2bb11c42d910).
# Orchestration-only knob — does not alter sim/judge semantics.
BRIDGE_TIMEOUT_S = 5400
POLL_TIMEOUT_S = 3600   # per-phase (graph/prepare/run) window inside the bridge
JUDGE_TIMEOUT_S = 1200


# ---------------------------------------------------------------------------
# env / key handling (self-contained so a restart needs no shell setup)
# ---------------------------------------------------------------------------
def _build_env() -> dict[str, str]:
    """Pin the judge's OPENROUTER_API_KEY to the canonical miroshark regime
    key (SMART_API_KEY in ~/miroshark/.env), OVERRIDING any ambient shell
    value. The ambient OPENROUTER_API_KEY may be a dead/invalid key whose
    401s silently degrade every judge call to p=0.5 (uncertain) while still
    being counted as an 'llm' source — the failure mode that poisoned the
    first batch attempt. Using the same OpenRouter account for sim and judge
    is also what "fixed regime" requires.
    """
    env = dict(os.environ)
    envfile = Path.home() / "miroshark" / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if line.startswith("SMART_API_KEY="):
                env["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip()
            elif line.startswith("MIROSHARK_INTERNAL_KEY="):
                # backend fail-closed auth guard (miroshark 375ef84);
                # the bridge CLI forwards this as x-miroshark-internal-key
                env["MIROSHARK_INTERNAL_KEY"] = line.split("=", 1)[1].strip()
    env["MIROSHARK_POLL_TIMEOUT_S"] = str(POLL_TIMEOUT_S)
    return env


def _validate_regime(env: dict[str, str]) -> None:
    """Fail fast if ~/miroshark/.env disagrees with the pinned regime, so a
    batch can never mix agent/SMART/NER models across replications."""
    envfile = Path.home() / "miroshark" / ".env"
    vals: dict[str, str] = {}
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    expected = {
        "WONDERWALL_MODEL_NAME": AGENT_MODEL,
        "LLM_MODEL_NAME": AGENT_MODEL,
        "SMART_MODEL_NAME": "x-ai/grok-4.3",
        "NER_MODEL_NAME": "x-ai/grok-4.3",
    }
    bad = {k: (vals.get(k), want) for k, want in expected.items()
           if vals.get(k) != want}
    if bad:
        for k, (got, want) in bad.items():
            print(f"[fatal] regime drift: {k}={got!r}, pinned regime wants "
                  f"{want!r}", file=sys.stderr)
        sys.exit(2)
    print(f"[regime] agents={AGENT_MODEL} smart/ner=x-ai/grok-4.3 "
          f"judge={JUDGE_MODEL} — env matches pinned regime")


def _validate_key(env: dict[str, str]) -> None:
    """Fail fast if the OpenRouter key is rejected, so we never burn a whole
    batch on 401s. (The first batch attempt produced 20× all-p=0.5 runs
    before this guard existed.)"""
    import urllib.request
    key = env.get("OPENROUTER_API_KEY", "")
    if not key:
        print("[fatal] no OPENROUTER_API_KEY (set SMART_API_KEY in "
              "~/miroshark/.env)", file=sys.stderr)
        sys.exit(2)
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status != 200:
                raise RuntimeError(f"status {r.status}")
        print("[keycheck] OpenRouter key OK")
    except Exception as exc:  # noqa: BLE001
        print(f"[fatal] OpenRouter key rejected ({exc}); judge would "
              f"silently degrade to all-p=0.5. Aborting.", file=sys.stderr)
        sys.exit(2)
    # Account balance, not just key validity: the 20260703 batch burned its
    # first 7 clean runs and then 402-cascaded when total account credits
    # ran out mid-batch (~$0.7-0.9/run at scale=3 max_rounds=5 incl. judge).
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())["data"]
        remaining = d["total_credits"] - d["total_usage"]
        print(f"[keycheck] OpenRouter balance ${remaining:.2f}")
        if remaining < 5.0:
            print(f"[fatal] balance ${remaining:.2f} < $5 floor; a batch "
                  f"would 402-cascade mid-run. Top up at "
                  f"https://openrouter.ai/settings/credits first.",
                  file=sys.stderr)
            sys.exit(2)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not read credit balance ({exc}); continuing",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def _load_manifest(batch: Path) -> dict[str, Any]:
    return json.loads((batch / "manifest.json").read_text())


def _save_manifest(batch: Path, m: dict[str, Any]) -> None:
    tmp = batch / "manifest.json.tmp"
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(batch / "manifest.json")


def _new_batch() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    batch = REPO / "runs" / f"{ts}_multiseed_miroshark"
    batch.mkdir(parents=True, exist_ok=False)

    prereg = f"""# PREREGISTRATION — multi-replication fixed-regime MiroShark study

Beads: distributional-agi-safety-qopt
Created (UTC): {datetime.now(timezone.utc).isoformat()}
Author: written before any simulation in this batch was started.

## Fixed regime
- MiroShark SMART/NER model: x-ai/grok-4.3 (~/miroshark/.env, pinned)
- Wonderwall agent model: {AGENT_MODEL} (pinned; preflight-checked against
  ~/miroshark/.env every execution). NOTE: the 20260517 batch ran agents on
  xiaomi/mimo-v2-flash without pinning it; that model was delisted from
  OpenRouter before this batch. {AGENT_MODEL} is its direct successor. This
  is a known, documented regime difference vs the 20260517 indicative batch;
  all runs WITHIN this batch use one agent model.
- Metrics judge model: {JUDGE_MODEL}, temperature=0.0
- scale={SCALE}, --max-rounds {MAX_ROUNDS}, platform={PLATFORM}
- grok-4.1-fast is deprecated and is NOT used anywhere in this batch.

## Sampling plan (no peeking-driven stopping)
- N_TARGET = {N_TARGET} attempts per scenario (libel, redteam) = {2 * N_TARGET} bridge runs.
- N_TARGET includes a PRE-HOC attrition allowance (12 vs the prior 10): the
  20260517 batch showed attempts can be burned by degenerate sims
  (n_interactions < {MIN_INTERACTIONS}). Chosen before any run in THIS batch;
  the stopping rule below is unchanged.
- Attempts are interleaved (libel rep1, redteam rep1, libel rep2, ...) and run
  strictly sequentially (one MiroShark backend; Wonderwall multiprocessing).
- A run is CLEAN iff: bridge produced export.json AND metrics.json was written
  AND n_interactions >= {MIN_INTERACTIONS}.
- Confirmatory verdicts require >= {MIN_CLEAN} clean reps per scenario.
- Stopping rule: run exactly N_TARGET attempts per scenario regardless of
  interim results. No adaptive stopping, no interim hypothesis tests.

## No reproducible seeds (documented irreproducibility)
The bridge has no --seed flag. MiroShark simulation stochasticity is LLM
sampling temperature (0.4-0.8) with no seed parameter exposed to the API;
the judge is temperature=0.0. Replications are therefore INDEPENDENT
STOCHASTIC DRAWS, not reproducible seeds. Results are reported as a
sampling distribution with bootstrap 95% CIs; the batch is not
bit-reproducible from scenario+seed.

## Hypotheses
- H1: quality_gap(libel) < quality_gap(redteam).
      Test: bootstrap 95% CI of (mean_libel - mean_redteam) + Welch t-test.
      Supported iff CI upper bound < 0.
- H2: quality_gap(libel) < 0.
      Test: bootstrap 95% CI of mean quality_gap(libel).
      Supported iff CI upper bound < 0.
- H3: quality_gap(redteam) is NOT distinguishable from 0.
      Test: bootstrap 95% CI of mean quality_gap(redteam).
      Supported (fail to reject) iff CI contains 0.

## Primary outcome
Per run: quality_gap, spread, accept_rate, n_interactions (from metrics.json
soft_metrics). Aggregate: per-scenario mean +/- bootstrap 95% CI
({BOOT_RESAMPLES} percentile resamples) and the full per-run distribution.
"""
    (batch / "PREREGISTRATION.md").write_text(prereg)
    sha = hashlib.sha256(prereg.encode()).hexdigest()

    # Interleaved plan: libel rep1, redteam rep1, libel rep2, ...
    plan = []
    for rep in range(1, N_TARGET + 1):
        for scen in ("libel", "redteam"):
            plan.append({
                "scenario": scen,
                "rep": rep,
                "status": "pending",   # pending|sim_done|judged|failed
                "run_dir": None,
                "quality_gap": None,
                "spread": None,
                "accept_rate": None,
                "n_interactions": None,
                "error": None,
            })
    manifest = {
        "beads": "distributional-agi-safety-qopt",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha,
        "regime": {
            "sim_model": "x-ai/grok-4.3",
            "agent_model": AGENT_MODEL,
            "judge_model": JUDGE_MODEL,
            "judge_temperature": 0.0,
            "scale": SCALE,
            "max_rounds": MAX_ROUNDS,
            "platform": PLATFORM,
        },
        "n_target": N_TARGET,
        "min_clean": MIN_CLEAN,
        "min_interactions": MIN_INTERACTIONS,
        "plan": plan,
    }
    _save_manifest(batch, manifest)
    print(f"[prereg] {batch/'PREREGISTRATION.md'} sha256={sha[:16]}…")
    return batch


def _find_resumable() -> Optional[Path]:
    root = REPO / "runs"
    cands = sorted(root.glob("*_multiseed_miroshark"), reverse=True)
    for c in cands:
        if (c / "manifest.json").exists():
            m = _load_manifest(c)
            if any(e["status"] in ("pending", "failed") for e in m["plan"]):
                return c
            return c  # newest; will just summarize
    return None


# ---------------------------------------------------------------------------
# run one attempt
# ---------------------------------------------------------------------------
def _kill_stale_sims() -> None:
    """After a bridge failure, any live run_parallel_simulation process is
    stale by definition (attempts run strictly sequentially): the bridge's
    subprocess timeout kills only the bridge client, NOT the backend-spawned
    Wonderwall sim, which keeps churning CPU. Observed 20260716-17: each
    leftover sim slows subsequent runs past BRIDGE_TIMEOUT_S, compounding
    one hang into a cascade of timeouts."""
    subprocess.run(["pkill", "-9", "-f", "run_parallel_simulation"],
                   check=False, capture_output=True)


def _run_bridge(scenario_path: Path, env: dict[str, str]) -> Path:
    cmd = [
        sys.executable, "-m", "swarm.bridges.miroshark", str(scenario_path),
        "--scale", str(SCALE), "--max-rounds", str(MAX_ROUNDS),
        "--platform", PLATFORM,
    ]
    res = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True,
        timeout=BRIDGE_TIMEOUT_S,
    )
    if res.returncode != 0:
        raise RuntimeError(f"bridge rc={res.returncode}: {res.stderr[-800:]}")
    run_dir = Path(res.stdout.strip().splitlines()[-1].strip())
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    if not (run_dir / "export.json").exists():
        raise RuntimeError(f"bridge produced no export.json in {run_dir}")
    return run_dir


def _run_judge(run_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        sys.executable, "-m", "swarm.bridges.miroshark.metrics", str(run_dir),
        "--model", JUDGE_MODEL,
    ]
    res = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True,
        timeout=JUDGE_TIMEOUT_S,
    )
    if res.returncode != 0:
        raise RuntimeError(f"judge rc={res.returncode}: {res.stderr[-800:]}")
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        raise RuntimeError(f"judge produced no metrics.json in {run_dir}")
    return json.loads(mpath.read_text())


def _execute(batch: Path) -> None:
    env = _build_env()
    _validate_key(env)
    _validate_regime(env)
    m = _load_manifest(batch)
    total = len(m["plan"])
    for i, e in enumerate(m["plan"]):
        tag = f"{e['scenario']}#{e['rep']}"
        if e["status"] == "judged":
            print(f"[{i+1}/{total}] {tag} already judged — skip")
            continue
        if e["status"] == "failed":
            print(f"[{i+1}/{total}] {tag} previously failed — retrying")
        print(f"[{i+1}/{total}] {tag} bridge…", flush=True)
        try:
            if e["status"] != "sim_done" or not e["run_dir"]:
                rd = _run_bridge(SCENARIOS[e["scenario"]], env)
                e["run_dir"] = str(rd)
                e["status"] = "sim_done"
                _save_manifest(batch, m)
            else:
                rd = Path(e["run_dir"])
                print(f"           reusing sim {rd.name}")
            print(f"[{i+1}/{total}] {tag} judge ({JUDGE_MODEL})…", flush=True)
            metrics = _run_judge(rd, env)
            sm = metrics["soft_metrics"]
            e["quality_gap"] = sm["quality_gap"]
            e["spread"] = sm["spread"]
            e["accept_rate"] = metrics["accept_rate"]
            e["n_interactions"] = metrics["n_interactions"]
            clean = metrics["n_interactions"] >= MIN_INTERACTIONS
            e["status"] = "judged"
            e["clean"] = clean
            e["error"] = None
            print(f"           qg={sm['quality_gap']:+.5f} "
                  f"spread={sm['spread']:+.5f} acc={metrics['accept_rate']:.3f} "
                  f"n={metrics['n_interactions']} clean={clean}")
        except Exception as exc:  # noqa: BLE001
            e["status"] = "failed"
            e["error"] = str(exc)[-500:]
            print(f"[{i+1}/{total}] {tag} FAILED: {e['error']}", file=sys.stderr)
            _kill_stale_sims()
        _save_manifest(batch, m)


# ---------------------------------------------------------------------------
# stats + summary
# ---------------------------------------------------------------------------
def _boot_ci(xs: list[float], stat, resamples: int = BOOT_RESAMPLES) -> tuple:
    import random
    rng = random.Random(12345)  # deterministic resampling of fixed data only
    n = len(xs)
    boots = []
    for _ in range(resamples):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int(0.025 * resamples)]
    hi = boots[int(0.975 * resamples)]
    return lo, hi


def _welch_t(a: list[float], b: list[float]) -> tuple:
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = (va / na + vb / nb) ** 0.5
    if se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    return t, df


def _summarize(batch: Path) -> None:
    m = _load_manifest(batch)
    by = {"libel": [], "redteam": []}
    rows = []
    for e in m["plan"]:
        rows.append(e)
        if e["status"] == "judged" and e.get("clean"):
            by[e["scenario"]].append(e["quality_gap"])

    lines = []
    lines.append("# Multi-replication fixed-regime MiroShark study — SUMMARY\n")
    lines.append("Beads: distributional-agi-safety-qopt  ")
    lines.append(f"Batch: `{batch.name}`  ")
    lines.append(f"Preregistration sha256: `{m['preregistration_sha256'][:16]}…`  ")
    lines.append(f"Regime: sim={m['regime']['sim_model']}, "
                 f"judge={m['regime']['judge_model']} (T=0.0), "
                 f"scale={m['regime']['scale']}, max_rounds={m['regime']['max_rounds']}, "
                 f"platform={m['regime']['platform']}\n")
    lines.append("> No reproducible seeds: replications are independent stochastic "
                 "draws (sim LLM temperature 0.4–0.8, no seed param). Reported as a "
                 "sampling distribution, not bit-reproducible.\n")

    # per-run table
    lines.append("## Per-run results\n")
    lines.append("| scenario | rep | status | clean | quality_gap | spread | accept_rate | n_int |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in rows:
        qg = f"{e['quality_gap']:+.5f}" if e["quality_gap"] is not None else "—"
        sp = f"{e['spread']:+.5f}" if e["spread"] is not None else "—"
        ar = f"{e['accept_rate']:.3f}" if e["accept_rate"] is not None else "—"
        ni = e["n_interactions"] if e["n_interactions"] is not None else "—"
        lines.append(f"| {e['scenario']} | {e['rep']} | {e['status']} | "
                      f"{e.get('clean','—')} | {qg} | {sp} | {ar} | {ni} |")

    # aggregates
    lines.append("\n## Aggregates (clean runs only)\n")
    agg = {}
    for scen in ("libel", "redteam"):
        xs = by[scen]
        if len(xs) >= 2:
            mean = statistics.mean(xs)
            sd = statistics.stdev(xs)
            lo, hi = _boot_ci(xs, statistics.mean)
            agg[scen] = (mean, sd, lo, hi, len(xs))
            lines.append(f"- **{scen}**: n={len(xs)}  mean quality_gap = "
                          f"{mean:+.5f}  (sd={sd:.5f})  "
                          f"bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}]")
        else:
            agg[scen] = None
            lines.append(f"- **{scen}**: n={len(xs)} — insufficient clean runs "
                          f"(need >= 2 for CI, >= {MIN_CLEAN} for confirmatory).")

    # verdicts
    lines.append("\n## Hypothesis verdicts\n")
    powered = (agg["libel"] and agg["libel"][4] >= MIN_CLEAN
               and agg["redteam"] and agg["redteam"][4] >= MIN_CLEAN)
    if not powered:
        lines.append(f"⚠️ **Underpowered**: need >= {MIN_CLEAN} clean reps per "
                      f"scenario for confirmatory verdicts. Verdicts below are "
                      f"INDICATIVE ONLY.\n")

    if agg["libel"]:
        _, _, llo, lhi, _ = agg["libel"]
        h2 = lhi < 0
        lines.append(f"- **H2** (quality_gap(libel) < 0): "
                      f"CI [{llo:+.5f}, {lhi:+.5f}] → "
                      f"{'SUPPORTED' if h2 else 'NOT supported'} "
                      f"(CI upper {'<' if h2 else '>='} 0)")
    if agg["redteam"]:
        _, _, rlo, rhi, _ = agg["redteam"]
        h3 = rlo <= 0 <= rhi
        lines.append(f"- **H3** (quality_gap(redteam) ~ 0, not distinguishable): "
                      f"CI [{rlo:+.5f}, {rhi:+.5f}] → "
                      f"{'SUPPORTED (fail to reject 0)' if h3 else 'NOT supported (0 excluded)'}")
    if agg["libel"] and agg["redteam"]:
        a, b = by["libel"], by["redteam"]
        diff = statistics.mean(a) - statistics.mean(b)
        # bootstrap CI of difference of means (independent resampling per arm)
        import random
        rng = random.Random(2024)
        boots = []
        for _ in range(BOOT_RESAMPLES):
            sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
            sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
            boots.append(statistics.mean(sa) - statistics.mean(sb))
        boots.sort()
        dlo = boots[int(0.025 * BOOT_RESAMPLES)]
        dhi = boots[int(0.975 * BOOT_RESAMPLES)]
        t, df = _welch_t(a, b)
        h1 = dhi < 0
        lines.append(f"- **H1** (quality_gap(libel) < quality_gap(redteam)): "
                      f"Δ(libel−redteam) = {diff:+.5f}, "
                      f"bootstrap 95% CI [{dlo:+.5f}, {dhi:+.5f}], "
                      f"Welch t={t:.3f} (df={df:.1f}) → "
                      f"{'SUPPORTED' if h1 else 'NOT supported'} "
                      f"(CI upper {'<' if h1 else '>='} 0)")

    lines.append("\n## Acceptance criteria status\n")
    nc_l = len(by["libel"])
    nc_r = len(by["redteam"])
    lines.append(f"- (1) >= {MIN_CLEAN} clean reps/scenario: "
                  f"libel={nc_l}, redteam={nc_r} → "
                  f"{'MET' if nc_l >= MIN_CLEAN and nc_r >= MIN_CLEAN else 'NOT MET'}")
    lines.append("- (2) summary with per-run table + CIs + verdicts: this file")
    lines.append("- (3) blog post update: pending operator decision based on verdicts")
    lines.append("- (4) archive raw run dirs to swarm-artifacts: pending (gitignored locally)")

    (batch / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"[summary] wrote {batch/'SUMMARY.md'}")
    print("\n".join(lines[-18:]))


# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="multiseed_miroshark")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume an existing batch dir (default: auto-detect newest)")
    ap.add_argument("--new", action="store_true",
                    help="force a new batch even if a resumable one exists")
    ap.add_argument("--summarize-only", action="store_true",
                    help="recompute SUMMARY.md from manifest, run nothing")
    args = ap.parse_args(argv)

    if args.summarize_only:
        batch = args.resume or _find_resumable()
        if not batch:
            print("no batch found", file=sys.stderr)
            return 2
        _summarize(batch)
        return 0

    if args.new:
        batch = _new_batch()
    elif args.resume:
        batch = args.resume
    else:
        batch = _find_resumable() or _new_batch()

    print(f"[batch] {batch}")
    _execute(batch)
    _summarize(batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
