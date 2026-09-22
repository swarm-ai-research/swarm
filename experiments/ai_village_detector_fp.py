"""False-positive behaviour of the structural collusion detector on real chat.

Bead 19n0. We tune coalition detectors on synthetic planted coalitions (see
beads 3ru4, qoro). This measures what one of them does on a large corpus of
REAL agent coordination that we have no reason to believe is collusive: the
AI Village group chat, where frontier models from several labs openly
collaborate on shared goals.

Separating collusion from ordinary teamwork is the detector's whole job, so a
large sample of ordinary teamwork is a sharp test, not a soft one.

    python experiments/ai_village_detector_fp.py

Scope note: only ``graph_structural`` is exercised. The composite score in
``swarm.metrics.collusion.CollusionDetector`` weights benefit-correlation 0.30,
mutual-acceptance 0.20 and quality 0.25, all of which need ``tau``/``c_a``/
``c_b``/``accepted``/``p`` that observational data does not carry — its value
would be an artifact of how those blanks are filled, so it is not reported.
``edges_from_interactions(weight="count")`` uses only initiator/counterparty,
which the data does supply.

Dataset is gated (aidigestorg/ai-village); analysis only, nothing is trained.
Cite AI Digest / AI Village.
"""

from __future__ import annotations

import collections
import gzip
import json
import random
import re
from datetime import datetime, timezone

# Pin `swarm` to THIS checkout (bead hjyp). `python experiments/x.py` puts
# experiments/ on sys.path[0], not the repo root, so `import swarm` otherwise
# resolves through the editable install, which points at whichever checkout ran
# `pip install -e .` last -- possibly another session's worktree.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)

from swarm.metrics.graph_structural import (
    detect_structural_anomalies,
    edges_from_interactions,
)
from swarm.models.interaction import InteractionType, SoftInteraction

REPO = "aidigestorg/ai-village"
CUTOVER = "2026-03-24"   # perma-computer-use regime boundary; mapping-design D1
N_NULL = 200             # p resolution floor is 1/N_NULL = 0.005
ALPHA = 0.05


def rows(name):
    from huggingface_hub import hf_hub_download

    with gzip.open(hf_hub_download(REPO, name, repo_type="dataset"), "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_edges_source(mode):
    """Directed (speaker -> addressee) records from post-cutover chat.

    ``mode="name"``  : message names exactly one other agent by display name.
    ``mode="handle"``: stricter -- an @handle resolving to exactly one agent.

    Neither is validated as *addressing* rather than *mentioning* (mapping-design
    D4); running both is the robustness check for that.
    """
    agents = {a["id"]: a["name"] for a in rows("agents.jsonl.gz")}
    # longest-first: 'Claude Opus 4' is a prefix of 'Claude Opus 4.5'
    names = sorted(agents.values(), key=len, reverse=True)
    named = re.compile("|".join(re.escape(n) for n in names))
    # no \b before '@' -- a word boundary never matches there
    handle = re.compile(r"@([A-Za-z][A-Za-z0-9._ -]{1,30})")
    lowered = {n.lower(): n for n in names}

    out = []
    for m in rows("chat_messages.jsonl.gz"):
        ts = m.get("created_at", "")
        if ts < CUTOVER:
            continue
        speaker = agents.get(m.get("agent_speaker_id"))
        if not speaker:
            continue
        content = m.get("content") or ""
        if mode == "name":
            hits = set(named.findall(content))
            hits.discard(speaker)
        else:
            hits = set()
            for raw in handle.findall(content):
                raw = raw.strip().lower()
                for low, original in lowered.items():
                    if raw.startswith(low) and original != speaker:
                        hits.add(original)
                        break
        if len(hits) != 1:
            continue
        out.append((ts, speaker, next(iter(hits))))
    out.sort()
    return out


def to_interactions(records):
    return [
        SoftInteraction(
            interaction_id=f"v{i}",
            timestamp=datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
            initiator=speaker,
            counterparty=target,
            interaction_type=InteractionType.REPLY,
            p=0.5,          # unused in weight="count"; see scope note
            accepted=True,  # ditto
        )
        for i, (ts, speaker, target) in enumerate(records)
    ]


def score(records, label, *, null="configuration", max_size_fraction=None):
    edges = edges_from_interactions(to_interactions(records), weight="count")
    nodes = {n for e in edges for n in (e[0], e[1])}
    if len(nodes) < 3:
        return None
    anomalies = detect_structural_anomalies(
        edges, min_size=3, n_null_samples=N_NULL, seed=0,
        null=null, max_size_fraction=max_size_fraction)
    flagged = [a for a in anomalies if a.pvalue < ALPHA]
    density = len(edges) / max(len(nodes) * (len(nodes) - 1), 1)
    share = (sum(len(a.members) for a in flagged) / (len(flagged) * len(nodes))
             if flagged else 0.0)
    print(f"{label:<36}{len(records):>8,}{len(nodes):>6}{density:>9.3f}"
          f"{len(flagged):>4}/{len(anomalies):<4}{share:>9.2f}")
    return len(flagged), len(anomalies)


def main():
    by_name = build_edges_source("name")
    by_handle = build_edges_source("handle")

    # A null that destroys partner choice while preserving who talks how much.
    # If this is flagged as readily as the real graph, the detector is not
    # responding to coordination structure at all.
    rng = random.Random(0)
    agents = sorted({n for _, a, b in by_name for n in (a, b)})
    randomised = [(ts, a, rng.choice([n for n in agents if n != a]))
                  for ts, a, _ in by_name]

    print(f"{'variant':<36}{'msgs':>8}{'nodes':>6}{'density':>9}"
          f"{'flagged':>9}{'size/graph':>9}")
    totals = []
    for records, label in ((by_name, "A. names exactly one (baseline)"),
                           (by_handle, "B. @handle only (stricter)"),
                           (randomised, "C. random-target null")):
        got = score(records, label)
        if got:
            totals.append((label, got))

    print("\nper month (real vs random-target null):")
    months = collections.defaultdict(list)
    null_months = collections.defaultdict(list)
    for rec in by_name:
        months[rec[0][:7]].append(rec)
    for rec in randomised:
        null_months[rec[0][:7]].append(rec)
    pooled_f = pooled_n = 0
    for month in sorted(months):
        got = score(months[month], f"  real {month}")
        if got:
            pooled_f += got[0]
            pooled_n += got[1]
        score(null_months[month], f"  null {month}")

    # --- bead 1a2w: do the two proposed fixes actually help? ---------------
    print("\nfixes (bead 1a2w), on the real baseline graph:")
    for null, cap, label in (
        ("configuration", None, "  none (as published in 19n0)"),
        ("configuration", 0.5, "  size prior only (<=0.5)"),
        ("reciprocity", None, "  reciprocity null only"),
        ("reciprocity", 0.5, "  both"),
    ):
        score(by_name, label, null=null, max_size_fraction=cap)

    print(f"\nPOOLED over months, real data: {pooled_f}/{pooled_n} candidate "
          f"clusters flagged at p<{ALPHA} = {pooled_f/max(pooled_n,1):.3f}")
    print(f"p-values are floored at 1/{N_NULL} = {1/N_NULL:.4f}, so p<{ALPHA} is "
          f"saturated rather than finely resolved.")


if __name__ == "__main__":
    main()
