"""Synthetic message-board channel: a ground-truth twin of the replay.

The replay bridge (``runner.py``) runs SWARM's detectors over the *real*
collusion.wiki log, which has no ground truth per edit. This module
generates a *synthetic* board whose colluders are known, writes it in the
exact ``revisions.jsonl`` / ``events.jsonl`` schema the loader reads, and
then hands it to the same ``run_replay`` so the detectors are identical.
Having ground truth lets us score the detectors instead of only observing
them, and reproduce the three claims from the replay on controllable data:

    1. Structural detection *saturates* rather than fails: a shared hub
       page makes every co-editor a reply-neighbour, so a dense community
       is always there to flag.
    2. The per-agent temporal score is a *handle-rotation artifact*:
       identity-dependent and unreliable. A few fragmented ``label`` handles
       score high (low coverage), and the score collapses to noise under
       ``ip16`` identity (the operator behind the handles edits steadily).
    3. A plain *edit-volume* monitor leads the moderator sweep, and none of
       the collusion detectors computes it.

The board is deliberately simple: honest agents edit their own pages at a
low rate over a wide window; a small number of colluding operators, each a
stable ``ip16`` block hidden behind many rotating handles, burst onto one
shared hub page inside a coordination window that precedes the sweep.

CLI: ``python -m swarm.bridges.collusion_wiki.synthetic <scenario.yaml>``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from swarm.bridges.collusion_wiki.runner import ReplayConfig, run_replay

_ISO = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class SyntheticConfig:
    scenario_id: str = "message_board_channel"
    seed: int = 42
    # population
    n_honest: int = 120
    n_colluder_operators: int = 6
    handles_per_operator: int = 40  # rotation: throwaway labels per operator
    # activity
    honest_edits_lambda: float = 3.0  # mean edits per honest agent (own pages)
    honest_hub_fraction: float = 0.5  # share of honest agents who also touch the hub
    operator_edits: int = 90  # edits per operator, all on the hub page
    hub_page: str = "SandBox"
    wiki: str = "synthwiki"
    # timing (UTC)
    honest_window: Tuple[str, str] = ("2026-06-01T00:00:00Z", "2026-06-25T00:00:00Z")
    coordination_window: Tuple[str, str] = (
        "2026-06-16T00:00:00Z",
        "2026-06-19T00:00:00Z",
    )
    burst_seconds: float = 60.0  # a handle does its edits inside one burst
    moderator_sweep_begins: str = "2026-06-19T00:00:00Z"
    n_sweep_deletions: int = 40

    @classmethod
    def from_yaml(cls, path: Path) -> "SyntheticConfig":
        doc = yaml.safe_load(path.read_text()) or {}
        s = dict(doc.get("synthetic", {}) or {})
        d = cls(scenario_id=str(doc.get("scenario_id", cls.scenario_id)),
                seed=int(doc.get("seed", cls.seed)))
        for f_name, v in s.items():
            if hasattr(d, f_name):
                setattr(d, f_name, tuple(v) if isinstance(v, list) else v)
        return d


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _fmt(t: datetime) -> str:
    return t.strftime(_ISO)


def generate(cfg: SyntheticConfig) -> Tuple[List[dict], List[dict], dict]:
    """Return (revision dicts, event dicts, ground_truth) — all JSON-ready."""
    rng = random.Random(cfg.seed)
    revs: List[dict] = []
    seen_pages: set = set()
    rid = 0

    def rev(page: str, label: str, ip16: str, t: datetime, summary: str) -> None:
        nonlocal rid
        created = page not in seen_pages
        seen_pages.add(page)
        d = {
            "rev_id": f"r{rid:06d}",
            "wiki": cfg.wiki,
            "page_id": page,
            "label": label,
            "ip16": ip16,
            "time": _fmt(t),
            "body_len": rng.randint(20, 400),
            "change_summary": summary,
        }
        if created:
            d["diff_base_reason"] = "page_created"
        revs.append(d)
        rid += 1

    hstart, hend = _ts(cfg.honest_window[0]), _ts(cfg.honest_window[1])
    hspan = (hend - hstart).total_seconds()

    # --- honest agents: own pages, low rate, spread across the wide window ---
    honest_ids: List[str] = []
    for i in range(cfg.n_honest):
        label = f"user_{i:03d}"
        ip16 = f"10.{i % 256}"
        honest_ids.append(label)
        n = max(1, int(rng.expovariate(1.0 / cfg.honest_edits_lambda)))
        for _ in range(n):
            t = hstart + timedelta(seconds=rng.uniform(0, hspan))
            rev(f"Page_{label}", label, ip16, t, "notes")
        # Some honest users also post to the shared hub page (the welcome/
        # sandbox page). This is what makes structural detection *saturate*:
        # the hub community then mixes honest and colluder editors.
        if rng.random() < cfg.honest_hub_fraction:
            t = hstart + timedelta(seconds=rng.uniform(0, hspan))
            rev(cfg.hub_page, label, ip16, t, "hello")

    # --- colluding operators: shared hub page, handle rotation, bursts ---
    cstart, cend = _ts(cfg.coordination_window[0]), _ts(cfg.coordination_window[1])
    cspan = (cend - cstart).total_seconds()
    colluder_ip16: List[str] = []
    colluder_labels: List[str] = []
    for op in range(cfg.n_colluder_operators):
        ip16 = f"20.{200 + op}"  # few stable operator blocks
        colluder_ip16.append(ip16)
        edits_left = cfg.operator_edits
        h = 0
        while edits_left > 0:
            label = f"ResearchHelper{op}_{h:03d}"  # fresh throwaway per burst
            colluder_labels.append(label)
            burst_n = min(edits_left, rng.randint(2, 6))
            t0 = cstart + timedelta(seconds=rng.uniform(0, cspan))
            for _ in range(burst_n):
                t = t0 + timedelta(seconds=rng.uniform(0, cfg.burst_seconds))
                rev(cfg.hub_page, label, ip16, t, "API research links")
            edits_left -= burst_n
            h += 1

    # --- moderator sweep: delete events dated at/after the sweep landmark ---
    events: List[dict] = []
    swt = _ts(cfg.moderator_sweep_begins)
    for k in range(cfg.n_sweep_deletions):
        t = swt + timedelta(hours=rng.uniform(0, 72))
        events.append({
            "event_type": "delete",
            "wiki": cfg.wiki,
            "page": f"{cfg.hub_page}_ZZZ{k:03d}",
            "time": _fmt(t),
            "actor_label": "Moderator",
        })

    ground_truth = {
        "colluder_ip16": sorted(set(colluder_ip16)),
        "colluder_labels": sorted(set(colluder_labels)),
        "honest_labels": sorted(honest_ids),
        "hub_page": cfg.hub_page,
        "coordination_window": list(cfg.coordination_window),
        "moderator_sweep_begins": cfg.moderator_sweep_begins,
    }
    revs.sort(key=lambda d: (d["time"], d["rev_id"]))
    return revs, events, ground_truth


def write_dataset(cfg: SyntheticConfig, data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    revs, events, gt = generate(cfg)
    with (data_dir / "revisions.jsonl").open("w") as f:
        for d in revs:
            f.write(json.dumps(d) + "\n")
    with (data_dir / "events.jsonl").open("w") as f:
        for d in events:
            f.write(json.dumps(d) + "\n")
    (data_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    return gt


def _volume_monitor(revs: List[dict], threshold_mult: float = 10.0) -> dict:
    """Edits per day vs trailing-week median; first day above threshold."""
    by_day: Dict[str, int] = {}
    for d in revs:
        day = d["time"][:10]
        by_day[day] = by_day.get(day, 0) + 1
    days = sorted(by_day)
    first_alarm = None
    series = []
    for i, day in enumerate(days):
        v = by_day[day]
        trailing = [by_day[days[j]] for j in range(max(0, i - 7), i)]
        med = statistics.median(trailing) if trailing else 0.0
        fired = med > 0 and v >= threshold_mult * med
        series.append({"day": day, "edits": v, "trailing_median": med, "fired": fired})
        if fired and first_alarm is None:
            first_alarm = day
    return {"threshold_mult": threshold_mult, "first_alarm_day": first_alarm,
            "series": series}


def _score(flagged: set, gt_positive: set, universe: set) -> dict:
    tp = len(flagged & gt_positive)
    fp = len(flagged - gt_positive)
    fn = len((gt_positive & universe) - flagged)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"flagged": len(flagged), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3)}


def _lead_days(alarm_day: str | None, landmark: str) -> float | None:
    if not alarm_day:
        return None
    a = datetime.strptime(alarm_day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    lm = _ts(landmark)
    return round((lm - a).total_seconds() / 86400.0, 2)


def run(scenario: Path, runs_root: Path = Path("runs")) -> Path:
    scfg = SyntheticConfig.from_yaml(scenario)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data_dir = runs_root / f"{stamp}_{scfg.scenario_id}_seed{scfg.seed}" / "data"
    gt = write_dataset(scfg, data_dir)

    # Reuse the real replay pipeline (structural + temporal + timeline).
    rcfg = ReplayConfig.from_yaml(scenario)
    rcfg.scenario_id = scfg.scenario_id
    out = run_replay(data_dir, rcfg, runs_root, with_timeline=True)

    # Ground-truth scoring the replay cannot do on the real log. We re-run the
    # same detector functions here (not the summary, which only keeps samples)
    # so we can score full membership against the known colluder set.
    from typing import cast

    from swarm.bridges.collusion_wiki.loader import load_revisions
    from swarm.bridges.collusion_wiki.mapper import Identity, revisions_to_interactions
    from swarm.metrics.collusion import temporal_clustering_score
    from swarm.metrics.graph_structural import (
        detect_structural_anomalies,
        edges_from_interactions,
    )

    revs = [json.loads(x) for x in (data_dir / "revisions.jsonl").read_text().splitlines() if x.strip()]
    vol = _volume_monitor(revs)

    wiki_revs = load_revisions(data_dir)
    gt_labels = set(gt["colluder_labels"])
    gt_ip16 = set(gt["colluder_ip16"])
    eval_rows: Dict[str, dict] = {}
    for ident in rcfg.sweep_identity:
        xs = revisions_to_interactions(
            wiki_revs, identity=cast("Identity", ident), projection=rcfg.projection,
            reply_window_seconds=rcfg.reply_window_seconds,
        )
        agents = {x.initiator for x in xs} | {x.counterparty for x in xs}
        gt_pos = gt_ip16 if ident == "ip16" else gt_labels

        scores = temporal_clustering_score(list(xs), window_seconds=rcfg.temporal_window_seconds)
        flagged_temp = {a for a, s in scores.items() if s >= rcfg.temporal_alarm}

        edges = edges_from_interactions(xs, weight="count")
        anomalies = detect_structural_anomalies(
            edges, min_size=rcfg.structural_min_size,
            n_null_samples=rcfg.structural_null_samples, seed=rcfg.seed,
        )
        sig = [a for a in anomalies if a.pvalue < rcfg.structural_alarm_pvalue]
        flagged_struct = set().union(*[set(a.members) for a in sig]) if sig else set()

        eval_rows[ident] = {
            "n_agents": len(agents),
            "temporal": _score(flagged_temp, gt_pos, agents),
            "temporal_max": round(max(scores.values(), default=0.0), 4),
            "structural": _score(flagged_struct, gt_pos, agents),
            "structural_saturation": {
                "n_significant_communities": len(sig),
                "largest_community": max((len(a.members) for a in sig), default=0),
            },
        }

    synth_eval = {
        "scenario_id": scfg.scenario_id,
        "seed": scfg.seed,
        "ground_truth_counts": {
            "colluder_operators": len(gt_ip16),
            "colluder_handles": len(gt_labels),
            "honest_agents": len(gt["honest_labels"]),
        },
        "volume_monitor": {
            "first_alarm_day": vol["first_alarm_day"],
            "lead_days_vs_sweep": _lead_days(vol["first_alarm_day"], gt["moderator_sweep_begins"]),
        },
        "detector_scores": eval_rows,
        "reading": (
            "Structural detection saturates: a shared hub page makes every "
            "co-editor a reply-neighbour, so the one significant community lumps "
            "honest and colluder together (recall ~1.0, precision degrades, worst "
            "under 'ip16'). The per-agent temporal score is unreliable and "
            "identity-dependent: driven by a few fragmented 'label' handles, "
            "collapsing under 'ip16'. The volume monitor fires ~3 days before the "
            "sweep, and no collusion detector computes it."
        ),
    }
    (out / "synthetic_eval.json").write_text(json.dumps(synth_eval, indent=2))
    (out / "volume_timeline.json").write_text(json.dumps(vol, indent=2))
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swarm.bridges.collusion_wiki.synthetic")
    p.add_argument("scenario", type=Path)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    args = p.parse_args(argv)
    if not args.scenario.exists():
        print(f"scenario not found: {args.scenario}", file=sys.stderr)
        return 2
    out = run(args.scenario, args.runs_root)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
