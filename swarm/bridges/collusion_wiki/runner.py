"""Run SWARM's collusion detectors over a collusion.wiki replay.

Outputs a self-contained run folder::

    runs/<ts>_casestudy_wiki_backchannel_seed<seed>/
        summary.json     headline numbers per identity mode
        timeline.csv     per-step detector state (for the detection-lag plot)
        pairs_<id>.csv   flagged pairs per identity mode
        groups_<id>.csv  flagged groups per identity mode
        structural_<id>.csv  structural anomalies per identity mode
        config.json      the resolved ReplayConfig

The timeline answers the question the blog post could only assert: at a
given threshold, on which day would each detector first have fired,
relative to the moderator sweep (Jun 19) and the OpenAI visit (Jun 21)?
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from swarm.bridges.collusion_wiki.loader import (
    WikiRevision,
    load_events,
    load_revisions,
)
from swarm.bridges.collusion_wiki.mapper import (
    Identity,
    Projection,
    revisions_to_interactions,
)
from swarm.metrics.collusion import CollusionDetector, temporal_clustering_score
from swarm.metrics.graph_structural import (
    detect_structural_anomalies,
    edges_from_interactions,
)
from swarm.models.interaction import SoftInteraction


@dataclass
class ReplayConfig:
    scenario_id: str = "casestudy_wiki_backchannel"
    identity: Identity = "label"
    projection: Projection = "agent"
    reply_window_seconds: Optional[float] = None
    temporal_window_seconds: float = 60.0
    structural_min_size: int = 3
    structural_null_samples: int = 50
    timeline_step_hours: float = 24.0
    timeline_null_samples: int = 20
    temporal_alarm: float = 0.7  # the 0.7 bar from the graph_structural sweep
    structural_alarm_pvalue: float = 0.05
    landmarks: Dict[str, str] = field(default_factory=dict)
    sweep_identity: List[str] = field(default_factory=lambda: ["label"])
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: Path) -> "ReplayConfig":
        with path.open() as f:
            doc = yaml.safe_load(f) or {}
        rp = doc.get("replay", {}) or {}
        sw = doc.get("sweep", {}) or {}
        return cls(
            scenario_id=str(doc.get("scenario_id", cls.scenario_id)),
            identity=rp.get("identity", "label"),
            projection=rp.get("projection", "agent"),
            reply_window_seconds=rp.get("reply_window_seconds"),
            temporal_window_seconds=float(rp.get("temporal_window_seconds", 60.0)),
            structural_min_size=int(rp.get("structural_min_size", 3)),
            structural_null_samples=int(rp.get("structural_null_samples", 50)),
            timeline_step_hours=float(rp.get("timeline_step_hours", 24.0)),
            timeline_null_samples=int(rp.get("timeline_null_samples", 20)),
            temporal_alarm=float(rp.get("temporal_alarm", 0.7)),
            structural_alarm_pvalue=float(rp.get("structural_alarm_pvalue", 0.05)),
            landmarks=dict(rp.get("landmarks", {}) or {}),
            sweep_identity=list(sw.get("identity", [rp.get("identity", "label")])),
            seed=int(doc.get("seed", 0)),
        )


# ---------------------------------------------------------------------------
# single-pass detectors
# ---------------------------------------------------------------------------


def _temporal(interactions: Sequence[SoftInteraction], window: float) -> Dict[str, Any]:
    scores = temporal_clustering_score(list(interactions), window_seconds=window)
    if not scores:
        return {"max": 0.0, "mean": 0.0, "n_agents": 0, "top": []}
    vals = sorted(scores.items(), key=lambda kv: -kv[1])
    return {
        "max": float(vals[0][1]),
        "mean": float(sum(scores.values()) / len(scores)),
        "n_agents": len(scores),
        "top": [(a, round(s, 4)) for a, s in vals[:10]],
    }


def _structural(
    interactions: Sequence[SoftInteraction], cfg: ReplayConfig, n_null: int
) -> List[Dict[str, Any]]:
    edges = edges_from_interactions(interactions, weight="count")
    anomalies = detect_structural_anomalies(
        edges, min_size=cfg.structural_min_size, n_null_samples=n_null, seed=cfg.seed
    )
    rows: List[Dict[str, Any]] = []
    for a in anomalies:
        rows.append(
            {
                "size": len(a.members),
                "n_internal_edges": a.n_internal_edges,
                "density": round(a.density, 4),
                "k_core": a.k_core,
                "reciprocity": round(a.reciprocity, 4),
                "reciprocity_z": round(a.reciprocity_z, 3),
                "pvalue": round(a.pvalue, 4),
                "members_sample": sorted(a.members)[:8],
            }
        )
    rows.sort(key=lambda r: (r["pvalue"], -r["size"]))
    return rows


def _pairwise(interactions: Sequence[SoftInteraction], cfg: ReplayConfig):
    det = CollusionDetector(seed=cfg.seed)
    return det.analyze(list(interactions))


# ---------------------------------------------------------------------------
# timeline / detection lag
# ---------------------------------------------------------------------------


def _timeline(
    interactions: Sequence[SoftInteraction], cfg: ReplayConfig
) -> List[Dict[str, Any]]:
    """Cumulative-to-date detector state at each step boundary."""
    if not interactions:
        return []
    xs = sorted(interactions, key=lambda x: x.timestamp)
    t0 = xs[0].timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    t_end = xs[-1].timestamp
    step = timedelta(hours=cfg.timeline_step_hours)
    rows: List[Dict[str, Any]] = []
    t = t0 + step
    i = 0
    while t <= t_end + step:
        while i < len(xs) and xs[i].timestamp < t:
            i += 1
        window = xs[:i]
        if not window:
            t += step
            continue
        # temporal over the last step only (a rolling alarm, not cumulative)
        recent = [x for x in window if x.timestamp >= t - step]
        temp = _temporal(recent, cfg.temporal_window_seconds)
        struct = _structural(window, cfg, cfg.timeline_null_samples)
        best_p = min((r["pvalue"] for r in struct), default=1.0)
        best_size = max((r["size"] for r in struct if r["pvalue"] == best_p), default=0)
        rows.append(
            {
                "step_end": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "n_interactions_to_date": len(window),
                "n_interactions_in_step": len(recent),
                "temporal_max": round(temp["max"], 4),
                "temporal_alarm": temp["max"] >= cfg.temporal_alarm,
                "structural_best_pvalue": best_p,
                "structural_best_size": best_size,
                "structural_alarm": best_p < cfg.structural_alarm_pvalue,
            }
        )
        t += step
    return rows


def _first_alarm(rows: Sequence[Dict[str, Any]], key: str) -> Optional[str]:
    for r in rows:
        if r[key]:
            return str(r["step_end"])
    return None


def _lag_days(alarm: Optional[str], landmark: Optional[str]) -> Optional[float]:
    if not alarm or not landmark:
        return None
    a = datetime.strptime(alarm, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    b = datetime.strptime(landmark, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return round((a - b).total_seconds() / 86400.0, 2)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})


def analyze_identity(
    revisions: Sequence[WikiRevision], cfg: ReplayConfig, identity: Identity
) -> Dict[str, Any]:
    """Full detector pass for one identity mode (no timeline)."""
    xs = revisions_to_interactions(
        revisions,
        identity=identity,
        projection=cfg.projection,
        reply_window_seconds=cfg.reply_window_seconds,
    )
    agents = {x.initiator for x in xs} | {x.counterparty for x in xs}
    temp = _temporal(xs, cfg.temporal_window_seconds)
    struct = _structural(xs, cfg, cfg.structural_null_samples)
    rep = _pairwise(xs, cfg)
    return {
        "identity": identity,
        "n_interactions": len(xs),
        "n_agents": len(agents),
        "temporal": temp,
        "structural": {
            "n_anomalies": len(struct),
            "n_significant": sum(
                1 for r in struct if r["pvalue"] < cfg.structural_alarm_pvalue
            ),
            "best": struct[0] if struct else None,
            "rows": struct,
        },
        "pairwise": {
            "ecosystem_collusion_risk": round(rep.ecosystem_collusion_risk, 4),
            "n_flagged_pairs": rep.n_flagged_pairs,
            "n_flagged_groups": rep.n_flagged_groups,
            "max_pair_collusion_score": round(rep.max_pair_collusion_score, 4),
            "pairs": [
                {
                    "agent_a": p.agent_a,
                    "agent_b": p.agent_b,
                    "n": p.interaction_count,
                    "score": round(p.collusion_score, 4),
                    "burstiness": round(p.interaction_burstiness, 4),
                }
                for p in sorted(rep.suspicious_pairs, key=lambda p: -p.collusion_score)
            ],
            "groups": [
                {
                    "size": len(g.members),
                    "score": round(g.collusion_score, 4),
                    "method": g.detection_method,
                    "members_sample": sorted(g.members)[:8],
                }
                for g in sorted(rep.suspicious_groups, key=lambda g: -g.collusion_score)
            ],
        },
        "_interactions": xs,
    }


def run_replay(
    data_dir: Path,
    cfg: ReplayConfig,
    runs_root: Path = Path("runs"),
    *,
    with_timeline: bool = True,
) -> Path:
    t_start = time.time()
    revisions = load_revisions(data_dir)
    deletions = load_events(data_dir, types={"delete"})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = runs_root / f"{stamp}_{cfg.scenario_id}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)

    per_identity: Dict[str, Any] = {}
    for ident in cfg.sweep_identity:
        res = analyze_identity(revisions, cfg, ident)  # type: ignore[arg-type]
        xs = res.pop("_interactions")
        per_identity[ident] = res
        _write_csv(out / f"pairs_{ident}.csv", res["pairwise"]["pairs"])
        _write_csv(out / f"groups_{ident}.csv", res["pairwise"]["groups"])
        _write_csv(out / f"structural_{ident}.csv", res["structural"]["rows"])
        if ident == cfg.identity and with_timeline:
            rows = _timeline(xs, cfg)
            _write_csv(out / "timeline.csv", rows)
            per_identity[ident]["timeline"] = {
                "n_steps": len(rows),
                "first_temporal_alarm": _first_alarm(rows, "temporal_alarm"),
                "first_structural_alarm": _first_alarm(rows, "structural_alarm"),
                "lag_days": {
                    f"{det}_vs_{lm}": _lag_days(_first_alarm(rows, f"{det}_alarm"), when)
                    for det in ("temporal", "structural")
                    for lm, when in cfg.landmarks.items()
                },
            }

    deletion_days: Dict[str, int] = {}
    for e in deletions:
        k = e.time.strftime("%Y-%m-%d")
        deletion_days[k] = deletion_days.get(k, 0) + 1

    summary = {
        "scenario_id": cfg.scenario_id,
        "seed": cfg.seed,
        "data_dir": str(data_dir),
        "n_revisions": len(revisions),
        "n_deletions": len(deletions),
        "time_range": [
            revisions[0].time.isoformat() if revisions else None,
            revisions[-1].time.isoformat() if revisions else None,
        ],
        "wikis": sorted({r.wiki for r in revisions}),
        "deletions_by_day": dict(sorted(deletion_days.items())),
        "p_note": "p fixed at 0.5: the log carries no per-edit quality signal, "
        "so quality asymmetry contributes nothing; detectors run on frequency, "
        "acceptance, timing and topology only.",
        "per_identity": per_identity,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return out
