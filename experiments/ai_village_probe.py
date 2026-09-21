"""Reproduce every figure in docs/research/ai-village-mapping-design.md.

The AI Village dataset (https://huggingface.co/datasets/aidigestorg/ai-village,
AI Digest) is gated: request access, then `hf auth login`. Nothing here trains
or fine-tunes anything -- it counts rows and joins tables, which is inside the
dataset's research-use terms. Cite AI Digest / AI Village in any resulting work.

    python experiments/ai_village_probe.py regimes     # D1 table
    python experiments/ai_village_probe.py addressing  # D4 rates
    python experiments/ai_village_probe.py outcomes    # D5 candidates + join

`outcomes` and `regimes` stream the 2.5 GB turns file and take a few minutes.
"""

from __future__ import annotations

import collections
import difflib
import gzip
import json
import re
import sys

REPO = "aidigestorg/ai-village"
CUTOVER = "2026-03-24"      # perma-computer-use rollout; see CHANGELOG.md
DOC_END = "2026-07-03"      # last documented scaffolding change
MAIN_ROOM = "18a3b2fb-9d2e-4ce7-b9b1-52e09c5408a8"


def rows(name):
    """Stream one gzipped JSONL table, downloading it on first use."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, name, repo_type="dataset")
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def regime(ts: str) -> str:
    if ts < CUTOVER:
        return "pre"
    return "documented" if ts < DOC_END else "undocumented"


COLS = ("pre", "documented", "undocumented")


def _table(title, counters):
    print(f"\n{title}")
    print(f"{'':<32}{'pre':>12}{'documented':>13}{'undocumented':>14}")
    for label, c in counters:
        print(f"{label:<32}" + "".join(f"{c[k]:>13,}" for k in COLS))


def cmd_regimes():
    """D1: the event vocabulary is near-disjoint across the cutover."""
    actions = collections.Counter()
    ev = collections.Counter()
    for e in rows("events.jsonl.gz"):
        r = regime(e.get("created_at", ""))
        ev[r] += 1
        at = (e.get("data") or {}).get("actionType", "?")
        actions[(at, r)] += 1

    sess = collections.Counter()
    for s in rows("computer_use_sessions.jsonl.gz"):
        sess[regime(s.get("created_at", ""))] += 1

    chat = collections.Counter()
    for m in rows("chat_messages.jsonl.gz"):
        chat[regime(m.get("created_at", ""))] += 1

    turns = collections.Counter()
    err = collections.Counter()
    rep = collections.Counter()
    prev: dict[str, str] = {}
    for t in rows("computer_use_turns.jsonl.gz"):
        r = regime(t.get("created_at", ""))
        turns[r] += 1
        if t.get("error"):
            err[r] += 1
        act = t.get("agent_action")
        if act is not None:
            sid = t.get("session_id")
            key = json.dumps(act, sort_keys=True)
            if prev.get(sid) == key:
                rep[r] += 1
            prev[sid] = key

    _table("corpus by regime", [
        ("computer-use turns", turns), ("sessions", sess),
        ("chat messages", chat), ("events", ev),
        ("turns with error", err), ("repeat-of-prev action", rep),
    ])
    print(f"\n{'turn error rate':<32}"
          + "".join(f"{100*err[k]/max(turns[k],1):>12.2f}%" for k in COLS))
    print(f"{'repeat-action rate':<32}"
          + "".join(f"{100*rep[k]/max(turns[k],1):>12.2f}%" for k in COLS))

    print("\nactionType, pre vs post cutover (the disjointness claim)")
    def post(a):
        return actions[(a, "documented")] + actions[(a, "undocumented")]

    names = {a for a, _ in actions}
    for a in sorted(names, key=lambda x: -(actions[(x, "pre")] + post(x)))[:14]:
        print(f"  {a:<34}{actions[(a,'pre')]:>10,}{post(a):>10,}")


def cmd_addressing():
    """D4: how far 'names exactly one agent' can stand in for addressing."""
    names = sorted((a["name"] for a in rows("agents.jsonl.gz")),
                   key=len, reverse=True)   # longest-first: 'Opus 4' prefixes 'Opus 4.5'
    named_pat = re.compile("|".join(re.escape(n) for n in names))
    # No \b before '@' -- a word boundary never matches there, silently killing the pattern.
    at_pat = re.compile(r"@[A-Za-z][A-Za-z0-9._-]{1,30}")

    total = main = named = at_hits = 0
    per_msg = collections.Counter()
    speakers = set()
    for m in rows("chat_messages.jsonl.gz"):
        total += 1
        if m.get("room_id") != MAIN_ROOM:
            continue
        main += 1
        if m.get("agent_speaker_id"):
            speakers.add(m["agent_speaker_id"])
        hits = set(named_pat.findall(m.get("content") or ""))
        if hits:
            named += 1
            per_msg[len(hits)] += 1
        if at_pat.search(m.get("content") or ""):
            at_hits += 1

    print(f"main-room messages: {main:,} of {total:,}")
    print(f"distinct agent speakers: {len(speakers)}")
    print(f"names >=1 agent: {named:,} ({100*named/main:.1f}%)")
    print(f"names exactly one: {per_msg[1]:,}  <- candidate directed edges")
    print(f"contains @handle: {at_hits:,} ({100*at_hits/main:.1f}%)")


def cmd_outcomes():
    """D5: score the three outcome-variable candidates."""
    verdicts = {}
    goals = collections.defaultdict(list)
    for e in rows("events.jsonl.gz"):
        d = e.get("data") or {}
        ts = e.get("created_at", "")
        if d.get("actionType") == "OUTREACH_APPROVAL_RESPONSE":
            verdicts[d.get("outreachApprovalRequestId")] = {
                "agent": d.get("agentId"),
                "approval": bool(d.get("approval")),
                "admin": bool(d.get("adminComment")),
                "ts": ts,
            }
        elif d.get("actionType") == "CONSOLIDATE" and CUTOVER <= ts < DOC_END:
            if d.get("nextSessionGoal") and d.get("agentId"):
                goals[d["agentId"]].append((ts, d["nextSessionGoal"]))

    # -- candidate 3: consolidation-goal repetition (rejected) ---------------
    def norm(text):
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    pairs = collections.Counter()
    n_pairs = 0
    for seq in goals.values():
        seq.sort()
        for i in range(1, len(seq)):
            x, y = norm(seq[i - 1][1]), norm(seq[i][1])
            if not (x and y):
                continue
            n_pairs += 1
            r = difflib.SequenceMatcher(None, x, y).ratio()
            band = ("identical" if r >= .95 else "near-repeat" if r >= .80
                    else "related" if r >= .50 else "different")
            pairs[band] += 1
    print(f"consolidation-goal pairs (primary window): {n_pairs:,}")
    for band in ("identical", "near-repeat", "related", "different"):
        print(f"  {band:<14}{pairs[band]:>8,} ({100*pairs[band]/max(n_pairs,1):5.1f}%)")

    # -- candidate 1: outreach approvals (accepted) --------------------------
    ok = sum(1 for v in verdicts.values() if v["approval"])
    admin = collections.Counter((v["approval"], v["admin"]) for v in verdicts.values())
    print(f"\noutreach verdicts: {len(verdicts)}  approved={ok} "
          f"({100*ok/max(len(verdicts),1):.1f}%)")
    print(f"  adminComment by verdict (approval, has_comment): {dict(admin)}")

    agents = {v["agent"] for v in verdicts.values()}
    lo = min(v["ts"] for v in verdicts.values())[:10]
    hi = max(v["ts"] for v in verdicts.values())[:10]
    print(f"  window {lo}..{hi}  agents={len(agents)}")

    sess_agent = {s["id"]: s["agent_id"] for s in rows("computer_use_sessions.jsonl.gz")
                  if s.get("agent_id") in agents}

    span = collections.defaultdict(lambda: {"n": 0, "err": 0, "lo": None, "hi": None})
    for t in rows("computer_use_turns.jsonl.gz"):
        sid = t.get("session_id")
        if sid not in sess_agent:
            continue
        ts = t.get("created_at", "")
        if not (lo <= ts[:10] <= hi):
            continue
        a = span[sid]
        a["n"] += 1
        if t.get("error"):
            a["err"] += 1
        if a["lo"] is None or ts < a["lo"]:
            a["lo"] = ts
        if a["hi"] is None or ts > a["hi"]:
            a["hi"] = ts

    by_agent = collections.defaultdict(list)
    for sid, a in span.items():
        if a["lo"]:
            by_agent[sess_agent[sid]].append((a["lo"], a["hi"], sid))
    for v in by_agent.values():
        v.sort()

    joined = []
    for v in verdicts.values():
        for start, end, sid in by_agent.get(v["agent"], []):
            if start <= v["ts"] <= end:
                joined.append((sid, v["approval"]))
                break
    hit = len(joined)
    print(f"\nJOIN: {hit}/{len(verdicts)} verdicts fall inside a session with turns "
          f"({100*hit/max(len(verdicts),1):.1f}%)")
    if joined:
        cls = collections.Counter(x[1] for x in joined)
        clusters = collections.Counter(sess_agent[s] for s, _ in joined)
        top = clusters.most_common(1)[0][1]
        ns = sorted(span[s]["n"] for s, _ in joined)
        print(f"  approved={cls[True]} rejected={cls[False]}")
        print(f"  median turns/session={ns[len(ns)//2]}")
        print(f"  agents={len(clusters)}  largest cluster={top} "
              f"({100*top/hit:.0f}% of rows) <- effective n is well below {hit}")


COMMANDS = {"regimes": cmd_regimes, "addressing": cmd_addressing, "outcomes": cmd_outcomes}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}}")
    COMMANDS[sys.argv[1]]()
