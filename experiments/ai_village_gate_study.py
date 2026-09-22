"""Does a session's proxy score predict whether its outreach was approved?

This is the decisive test for bead vu70. The AI Village supplies no interaction
outcomes, but it does supply real accept/reject verdicts from a human reviewer
(OUTREACH_APPROVAL_RESPONSE). If session observables carry no information about
those verdicts, that is a null result and no bridge is warranted.

Per D3 of docs/research/ai-village-mapping-design.md there is no admissible
task-progress observable, so that channel is held at zero and three are fitted.

    python experiments/ai_village_gate_study.py

Dataset is gated; see experiments/ai_village_probe.py. Analysis only -- nothing
here trains a model. Cite AI Digest / AI Village.
"""

from __future__ import annotations

import collections
import gzip
import json
import random
import re
import statistics

# Pin `swarm` to THIS checkout (bead hjyp). `python experiments/x.py` puts
# experiments/ on sys.path[0], not the repo root, so `import swarm` otherwise
# resolves through the editable install, which points at whichever checkout ran
# `pip install -e .` last -- possibly another session's worktree.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)

from swarm.core.proxy import ProxyComputer, ProxyObservables

REPO = "aidigestorg/ai-village"
MAIN_ROOM = "18a3b2fb-9d2e-4ce7-b9b1-52e09c5408a8"
REPLY_WINDOW_S = 1800.0


def rows(name):
    from huggingface_hub import hf_hub_download

    with gzip.open(hf_hub_download(REPO, name, repo_type="dataset"), "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def ts_of(s):
    """'2026-04-16 12:00:00.123456' -> epoch-ish seconds, for deltas only."""
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def collect():
    verdicts = {}
    for e in rows("events.jsonl.gz"):
        d = e.get("data") or {}
        if d.get("actionType") == "OUTREACH_APPROVAL_RESPONSE":
            verdicts[d.get("outreachApprovalRequestId")] = {
                "agent": d.get("agentId"),
                "approval": bool(d.get("approval")),
                "ts": e.get("created_at", ""),
            }
    lo = min(v["ts"] for v in verdicts.values())[:10]
    hi = max(v["ts"] for v in verdicts.values())[:10]
    agents = {v["agent"] for v in verdicts.values()}

    sess_agent = {s["id"]: s["agent_id"] for s in rows("computer_use_sessions.jsonl.gz")
                  if s.get("agent_id") in agents}

    span = collections.defaultdict(
        lambda: {"n": 0, "err": 0, "rep": 0, "lo": None, "hi": None})
    prev = {}
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
        act = t.get("agent_action")
        if act is not None:
            key = json.dumps(act, sort_keys=True)
            if prev.get(sid) == key:
                a["rep"] += 1
            prev[sid] = key
        if a["lo"] is None or ts < a["lo"]:
            a["lo"] = ts
        if a["hi"] is None or ts > a["hi"]:
            a["hi"] = ts

    names = sorted((a["name"] for a in rows("agents.jsonl.gz")), key=len, reverse=True)
    name_of = {a["id"]: a["name"] for a in rows("agents.jsonl.gz")}
    pat = re.compile("|".join(re.escape(n) for n in names))
    chat = [m for m in rows("chat_messages.jsonl.gz")
            if m.get("room_id") == MAIN_ROOM and lo <= m.get("created_at", "")[:10] <= hi]
    chat.sort(key=lambda m: m.get("created_at", ""))
    return verdicts, sess_agent, span, chat, pat, name_of


def engagement(agent_id, start, end, chat, pat, name_of):
    """Fraction of uniquely-addressed messages that drew a naming reply, in [-1, 1]."""
    sent = [m for m in chat
            if m.get("agent_speaker_id") == agent_id and start <= m.get("created_at", "") <= end]
    addressed = []
    for m in sent:
        hits = set(pat.findall(m.get("content") or ""))
        hits.discard(name_of.get(agent_id, ""))
        if len(hits) == 1:
            addressed.append((m, next(iter(hits))))
    if not addressed:
        return 0.0, 0
    me = name_of.get(agent_id, "")
    replies = 0
    for m, target in addressed:
        t0 = ts_of(m["created_at"])
        for r in chat:
            if r.get("created_at", "") <= m["created_at"]:
                continue
            if ts_of(r["created_at"]) - t0 > REPLY_WINDOW_S:
                break
            if name_of.get(r.get("agent_speaker_id")) == target and me in (r.get("content") or ""):
                replies += 1
                break
    return 2.0 * replies / len(addressed) - 1.0, len(addressed)


def cluster_bootstrap(records, stat, iters=5000, seed=0):
    """Resample whole agents, not rows -- 20 clusters, one holding 35% of rows."""
    rng = random.Random(seed)
    by_agent = collections.defaultdict(list)
    for r in records:
        by_agent[r["agent"]].append(r)
    keys = list(by_agent)
    out = []
    for _ in range(iters):
        draw = []
        for _ in range(len(keys)):
            draw.extend(by_agent[rng.choice(keys)])
        val = stat(draw)
        if val is not None:
            out.append(val)
    out.sort()
    if not out:
        return None
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def auc(records, key, sign):
    """P(a random approved row scores above a random rejected one)."""
    pos = [sign * r[key] for r in records if r["approval"]]
    neg = [sign * r[key] for r in records if not r["approval"]]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def gap(records):
    a = [r["p"] for r in records if r["approval"]]
    b = [r["p"] for r in records if not r["approval"]]
    if not a or not b:
        return None
    return statistics.fmean(a) - statistics.fmean(b)


def main():
    verdicts, sess_agent, span, chat, pat, name_of = collect()
    by_agent = collections.defaultdict(list)
    for sid, a in span.items():
        if a["lo"]:
            by_agent[sess_agent[sid]].append((a["lo"], a["hi"], sid))
    for v in by_agent.values():
        v.sort()

    proxy = ProxyComputer()
    records, unjoined = [], []
    for v in verdicts.values():
        sid = None
        for start, end, cand in by_agent.get(v["agent"], []):
            if start <= v["ts"] <= end:
                sid = cand
                break
        if sid is None:
            unjoined.append(v)
            continue
        a = span[sid]
        eng, n_addr = engagement(v["agent"], a["lo"], a["hi"], chat, pat, name_of)
        obs = ProxyObservables(
            task_progress_delta=0.0,          # D3: no admissible source
            rework_count=a["rep"],
            verifier_rejections=a["err"],
            counterparty_engagement_delta=eng,
        )
        p = proxy.compute_p(proxy.compute_v_hat(obs))
        records.append({"agent": v["agent"], "approval": v["approval"], "p": p,
                        "n": a["n"], "err": a["err"], "rep": a["rep"],
                        "eng": eng, "n_addr": n_addr})

    print(f"joined {len(records)} / {len(verdicts)} verdicts "
          f"({100*len(records)/len(verdicts):.1f}%)")

    # --- non-join bias: is the joined subset unbiased on the verdict? --------
    j = sum(r["approval"] for r in records) / len(records)
    u = sum(v["approval"] for v in unjoined) / max(len(unjoined), 1)
    print(f"\nNON-JOIN BIAS  approval rate joined={j:.3f} (n={len(records)})  "
          f"unjoined={u:.3f} (n={len(unjoined)})  diff={j-u:+.3f}")

    # --- confound: does session length differ by verdict? -------------------
    for label, key in (("turns", "n"), ("errors", "err"), ("repeats", "rep"),
                       ("engagement", "eng"), ("addressed msgs", "n_addr")):
        a = [r[key] for r in records if r["approval"]]
        b = [r[key] for r in records if not r["approval"]]
        print(f"  {label:<16} approved median={statistics.median(a):>8.3f}   "
              f"rejected median={statistics.median(b):>8.3f}")

    # --- the actual test ----------------------------------------------------
    g = gap(records)
    ci = cluster_bootstrap(records, gap)
    print(f"\nquality_gap = E[p|approved] - E[p|rejected] = {g:+.4f}")
    if ci:
        print(f"  agent-cluster bootstrap 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        verdict = "SEPARATES" if (ci[0] > 0 or ci[1] < 0) else "NO SEPARATION (null)"
        print(f"  -> {verdict}")
    clusters = collections.Counter(r["agent"] for r in records)
    print(f"  clusters={len(clusters)}  largest={clusters.most_common(1)[0][1]}"
          f" ({100*clusters.most_common(1)[0][1]/len(records):.0f}% of rows)")

    # --- per-channel diagnosis ---------------------------------------------
    # A null on the composed proxy could mean "no signal" or "signal cancelled
    # by the composition". Score each channel on its own to tell them apart.
    # Every CI here must be agent-clustered: the unclustered error-rate AUC is
    # 0.604, which looks like discrimination and is not once 20 clusters (one
    # holding 35% of rows) are accounted for.
    for r in records:
        r["errate"] = r["err"] / max(r["n"], 1)
        r["reprate"] = r["rep"] / max(r["n"], 1)
    print(f"\n{'channel':<26}{'AUC':>7}{'  95% CI':>20}   verdict")
    channels = (("p", 1, "composed proxy p"), ("errate", -1, "error rate (neg)"),
                ("reprate", -1, "repeat rate (neg)"), ("eng", 1, "engagement"),
                ("n", -1, "session length (neg)"))
    for key, sign, label in channels:
        point = auc(records, key, sign)
        ci = cluster_bootstrap(records, lambda rs, k=key, g=sign: auc(rs, k, g), iters=2000)
        if point is None or ci is None:
            continue
        call = ("discriminates" if ci[0] > 0.5
                else "INVERTED" if ci[1] < 0.5 else "null")
        print(f"{label:<26}{point:>7.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]   {call}")
    print("\nNote: 20 clusters resolve only AUC >= ~0.64. A null here is absence of"
          "\nevidence at this sample size, not evidence the proxy is uninformative.")


if __name__ == "__main__":
    main()
