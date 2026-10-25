"""What counts as convincing evidence of collusion? A null-model audit.

Bead 50bb (Apart AI Collusion Research Sprint, RQ 2.3). Builds on 19n0 and 1a2w
(docs/research/collusion-detector-false-positives.md).

The detector question is split into the two jobs evidence is asked to do:

* adjudication -- someone names a group; is its internal traffic enriched
  beyond what ordinary activity implies?  The group is given, so this isolates
  the null model.
* screening -- find the group.  Adds candidate generation (densest subgraph +
  label propagation, as in ``detect_structural_anomalies``).

Data is real, non-collusive multi-agent coordination: AI Village chat after the
2026-03-24 cutover, edges = "message names exactly one other agent" (19n0
variant A).  Positive controls are PLANTED into that real background: a
random group of k agents redirects a fraction f of its outbound messages to
co-present members.  f=0 is the honest background.

Nulls compared (all keep each speaker's message count and timestamps):

* ``configuration`` -- stub-matching degree null, the published default.
* ``reciprocity``   -- double-edge swap preserving mutual dyads and edge count
                       (1a2w, with the 50bb edge-leak fix).
* ``uniform``       -- addressee drawn uniformly from all agents (19n0's C).
* ``copresent``     -- addressee drawn uniformly from agents present that day.
* ``day_perm``      -- addressee labels permuted among messages of the same
                       day: every agent's per-day sent AND received counts are
                       exact.  The schedule-preserving null.

The first two score the binary presence graph (``density_pvalue``); the last
three score message counts, T(S) = #messages with speaker and addressee in S.

Protocol (sprint brief, RQ 2.3): every decision threshold is chosen on the
calibration months (odd-indexed) and frozen before the test months are scored.

    python experiments/collusion_evidence_audit.py [--quick]

Dataset is gated (aidigestorg/ai-village); analysis only.  Cite AI Digest.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Pin `swarm` to THIS checkout (bead hjyp); see ai_village_detector_fp.py.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_village_detector_fp as av  # noqa: E402

from swarm.metrics.graph_structural import (  # noqa: E402
    DiGraph,
    densest_subgraph,
    density_pvalue,
    edges_from_interactions,
    label_propagation,
)

ALPHA = 0.05
MESSAGE_NULLS = ("uniform", "copresent", "day_perm")
GRAPH_NULLS = ("configuration", "reciprocity")
NULLS = GRAPH_NULLS + MESSAGE_NULLS


# --------------------------------------------------------------------------
# Window: one month of (day, speaker, addressee) as integer arrays
# --------------------------------------------------------------------------


class Window:
    def __init__(self, records):
        agents = sorted({n for _, a, b in records for n in (a, b)})
        self.agents = agents
        self.index = {n: i for i, n in enumerate(agents)}
        days = sorted({ts[:10] for ts, _, _ in records})
        day_ix = {d: i for i, d in enumerate(days)}
        order = sorted(range(len(records)), key=lambda i: records[i][0])
        self.day = np.array([day_ix[records[i][0][:10]] for i in order])
        self.src = np.array([self.index[records[i][1]] for i in order])
        self.dst = np.array([self.index[records[i][2]] for i in order])
        self.n_agents = len(agents)
        self.n_days = len(days)
        self.bounds = np.searchsorted(self.day, np.arange(self.n_days + 1))
        self.present = []
        for d in range(self.n_days):
            lo, hi = self.bounds[d], self.bounds[d + 1]
            self.present.append(np.unique(np.concatenate([self.src[lo:hi], self.dst[lo:hi]])))

    def with_dst(self, dst):
        w = object.__new__(Window)
        w.__dict__.update(self.__dict__)
        w.dst = dst
        return w

    def edges(self):
        recs = [("2026-01-01T00:00:00", self.agents[s], self.agents[t])
                for s, t in zip(self.src, self.dst, strict=True)]
        return edges_from_interactions(av.to_interactions(recs), weight="count")


def internal(src, dst, member):
    """T(S): messages whose speaker and addressee are both in S."""
    return int(np.count_nonzero(member[src] & member[dst]))


# --------------------------------------------------------------------------
# Message-level nulls
# --------------------------------------------------------------------------


def null_uniform(w, rng):
    r = rng.integers(0, w.n_agents - 1, size=len(w.src))
    return r + (r >= w.src)


def null_copresent(w, rng):
    out = np.empty_like(w.dst)
    for d in range(w.n_days):
        lo, hi = w.bounds[d], w.bounds[d + 1]
        pres = w.present[d]
        src = w.src[lo:hi]
        # uniform over present agents other than the speaker
        k = rng.integers(0, len(pres) - 1, size=hi - lo)
        pos = np.searchsorted(pres, src)
        k = k + (k >= pos)
        out[lo:hi] = pres[k]
    return out


def null_day_perm(w, rng):
    """Permute addressees within each day; repair self-addressing by swaps
    that create none, so per-day in- and out-counts are exact."""
    out = w.dst.copy()
    for d in range(w.n_days):
        lo, hi = w.bounds[d], w.bounds[d + 1]
        if hi - lo < 2:
            continue
        seg = out[lo:hi]
        seg[:] = seg[rng.permutation(hi - lo)]
        src = w.src[lo:hi]
        for i in np.flatnonzero(seg == src):
            for _ in range(50):
                j = int(rng.integers(0, hi - lo))
                if seg[j] != src[i] and seg[i] != src[j]:
                    seg[i], seg[j] = seg[j], seg[i]
                    break
    return out


MESSAGE_NULL_FNS = {"uniform": null_uniform, "copresent": null_copresent,
                    "day_perm": null_day_perm}


def null_samples(w, name, n, seed):
    rng = np.random.default_rng(seed)
    return [MESSAGE_NULL_FNS[name](w, rng) for _ in range(n)]


def enrichment(w, member, samples):
    """(observed T, ratio obs/mean null, ratio 95% interval, p)."""
    obs = internal(w.src, w.dst, member)
    nulls = np.array([internal(w.src, d, member) for d in samples], dtype=float)
    mean = nulls.mean()
    ratio = obs / mean if mean > 0 else float("inf") if obs else 1.0
    with np.errstate(divide="ignore"):
        per = obs / np.maximum(nulls, 1e-9)
    lo, hi = np.percentile(per, [2.5, 97.5])
    p = (np.count_nonzero(nulls >= obs) + 1) / (len(nulls) + 1)
    return obs, float(ratio), (float(lo), float(hi)), float(p)


# --------------------------------------------------------------------------
# Positive control: plant a coalition into the real background
# --------------------------------------------------------------------------


def plant(w, member_ix, f, rng):
    """Redirect fraction f of members' outbound-to-outsider messages to a
    co-present fellow member.  Speakers, timestamps and volume unchanged."""
    if f == 0:
        return w
    dst = w.dst.copy()
    member = np.zeros(w.n_agents, bool)
    member[member_ix] = True
    for d in range(w.n_days):
        lo, hi = w.bounds[d], w.bounds[d + 1]
        pres_members = [m for m in member_ix if m in set(w.present[d].tolist())]
        for i in range(lo, hi):
            s = w.src[i]
            if member[s] and not member[dst[i]] and rng.random() < f:
                others = [m for m in pres_members if m != s]
                if others:
                    dst[i] = others[int(rng.integers(0, len(others)))]
    return w.with_dst(dst)


def pick_group(w, k, rng, min_sent=20):
    sent = np.bincount(w.src, minlength=w.n_agents)
    pool = np.flatnonzero(sent >= min_sent)
    if len(pool) < k:
        return None
    return sorted(rng.choice(pool, size=k, replace=False).tolist())


# --------------------------------------------------------------------------
# Scoring one (window, group) under every null
# --------------------------------------------------------------------------


def score_group(w, member_ix, *, n_msg, n_graph, seed, graph_cache=None):
    member = np.zeros(w.n_agents, bool)
    member[member_ix] = True
    names = {w.agents[i] for i in member_ix}
    out = {}
    for name in MESSAGE_NULLS:
        obs, ratio, ci, p = enrichment(w, member, null_samples(w, name, n_msg, seed))
        out[name] = {"obs": obs, "ratio": ratio, "ci": ci, "p": p}
    g = graph_cache if graph_cache is not None else DiGraph.from_edges(w.edges())
    for name in GRAPH_NULLS:
        p = density_pvalue(g, names, n_samples=n_graph, seed=seed, null=name)
        out[name] = {"p": p}
    return out


def candidates(w, min_size=3):
    g = DiGraph.from_edges(w.edges())
    cands = []
    dense, _ = densest_subgraph(g)
    if len(dense) >= min_size:
        cands.append(dense)
    comm = collections.defaultdict(set)
    for node, lab in label_propagation(g, seed=0).items():
        comm[lab].add(node)
    for members in comm.values():
        if len(members) >= min_size and members not in cands:
            cands.append(members)
    return g, cands


def jaccard(a, b):
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--n-msg", type=int, default=500)
    ap.add_argument("--n-graph", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.quick:
        args.groups, args.n_msg, args.n_graph = 3, 50, 20
    fs = (0.0, 0.1, 0.2, 0.4)

    t0 = time.time()
    records = av.build_edges_source("name")
    by_month = collections.defaultdict(list)
    for r in records:
        by_month[r[0][:7]].append(r)
    months = sorted(by_month)
    calib = months[0::2]
    test = months[1::2]
    print(f"loaded {len(records):,} msgs, months {months}; calib {calib}; test {test}"
          f" ({time.time()-t0:.0f}s)", flush=True)

    rows = []          # adjudication
    screen = []        # screening
    background = []    # the detector's own candidates on the honest graph
    for mi, month in enumerate(months):
        w = Window(by_month[month])
        split = "calib" if month in calib else "test"

        # Candidates the published pipeline proposes on the honest window.
        g0, cands0 = candidates(w)
        for c in cands0:
            ix = [w.index[n] for n in c]
            sc = score_group(w, ix, n_msg=args.n_msg, n_graph=args.n_graph,
                             seed=7, graph_cache=g0)
            background.append({"month": month, "split": split, "size": len(c),
                               "size_fraction": len(c) / w.n_agents, "scores": sc})

        for gi in range(args.groups):
            rng = np.random.default_rng(10_000 * mi + gi)
            grp = pick_group(w, args.k, rng)
            if grp is None:
                continue
            for f in fs:
                wp = plant(w, grp, f, np.random.default_rng(rng.integers(1 << 31)))
                sc = score_group(wp, grp, n_msg=args.n_msg, n_graph=args.n_graph,
                                 seed=gi, graph_cache=g0 if f == 0 else None)
                rows.append({"month": month, "split": split, "group": gi, "f": f,
                             "members": [w.agents[i] for i in grp],
                             "redirected": int(np.count_nonzero(wp.dst != w.dst)),
                             "member_msgs": int(np.isin(w.src, grp).sum()),
                             "scores": sc})
                if f > 0:
                    gp, cands = candidates(wp)
                    target = {w.agents[i] for i in grp}
                    best = max((jaccard(c, target) for c in cands), default=0.0)
                    screen.append({"month": month, "split": split, "f": f,
                                   "n_candidates": len(cands),
                                   "best_jaccard": best,
                                   "candidate_sizes": [len(c) for c in cands]})
        print(f"  {month} [{split}] agents={w.n_agents} msgs={len(w.src)} "
              f"done ({time.time()-t0:.0f}s)", flush=True)

    summary = summarise(rows, background, screen, calib)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"runs/{stamp}_collusion_evidence_audit{'_quick' if args.quick else ''}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "adjudication.json").write_text(json.dumps(rows, indent=1))
    (out / "background_candidates.json").write_text(json.dumps(background, indent=1))
    (out / "screening.json").write_text(json.dumps(screen, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "config.json").write_text(json.dumps(vars(args) | {"fs": fs, "alpha": ALPHA,
                                                             "calib": calib, "test": test}))
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")


def summarise(rows, background, screen, calib):
    def flag_p(s, null):
        return s[null]["p"] < ALPHA

    # Freeze the effect-size bar on calibration months: the 95th percentile of
    # day_perm enrichment among honest (f=0) named groups.
    honest_cal = [r["scores"]["day_perm"]["ratio"] for r in rows
                  if r["f"] == 0 and r["split"] == "calib"]
    r_star = float(np.percentile(honest_cal, 95)) if honest_cal else float("nan")

    def frozen(s):
        d = s["day_perm"]
        return d["p"] < ALPHA and d["ratio"] >= r_star

    rules = {f"p<{ALPHA} under {n}": (lambda s, n=n: flag_p(s, n)) for n in NULLS}
    rules[f"frozen: day_perm p<{ALPHA} and ratio>={r_star:.2f}"] = frozen

    table = {}
    for split in ("calib", "test"):
        for rule, fn in rules.items():
            for f in sorted({r["f"] for r in rows}):
                sel = [r for r in rows if r["split"] == split and r["f"] == f]
                if not sel:
                    continue
                k = sum(fn(r["scores"]) for r in sel)
                table.setdefault(split, {}).setdefault(rule, {})[str(f)] = {
                    "flagged": k, "n": len(sel), "rate": k / len(sel),
                    "wilson95": wilson(k, len(sel))}

    bg = {}
    for split in ("calib", "test"):
        sel = [b for b in background if b["split"] == split]
        bg[split] = {rule: {"flagged": sum(fn(b["scores"]) for b in sel), "n": len(sel)}
                     for rule, fn in rules.items()}
        bg[split]["mean_size_fraction"] = (float(np.mean([b["size_fraction"] for b in sel]))
                                           if sel else None)

    sc = {}
    for f in sorted({s["f"] for s in screen}):
        sel = [s for s in screen if s["f"] == f]
        sc[str(f)] = {"n": len(sel),
                      "best_jaccard_mean": float(np.mean([s["best_jaccard"] for s in sel])),
                      "share_jaccard_ge_0.5": float(np.mean([s["best_jaccard"] >= 0.5 for s in sel]))}

    ratios = collections.defaultdict(list)
    for r in rows:
        for n in MESSAGE_NULLS:
            ratios[f"{n} f={r['f']}"].append(r["scores"][n]["ratio"])
    return {"r_star": r_star, "rules": table, "background_candidates": bg,
            "screening": sc,
            "median_ratio": {k: float(np.median(v)) for k, v in ratios.items()}}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - h), min(1.0, c + h))


if __name__ == "__main__":
    main()
