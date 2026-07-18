#!/usr/bin/env python3
"""
Deep analysis of the Moltbook CAPTCHA scenario through the SWARM framework.

This script:
1. Runs a fresh simulation of the moltbook_captcha scenario
2. Inspects the orchestrator's internal state for per-agent details
3. Reconstructs all interactions from the event log
4. Computes SWARM soft metrics on all interactions
5. Analyzes per-agent-type performance (diligent, spam, pretender, collusive)
6. Evaluates governance mechanism effectiveness (CAPTCHA, rate limits, collusion detection)
7. Prints a comprehensive evaluation report
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from swarm.core.payoff import SoftPayoffEngine
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction
from swarm.scenarios.loader import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
AGENT_TYPE_MAP = {
    "diligent_moltbook": "Diligent",
    "spam_bot": "Spam Bot",
    "human_pretender": "Human Pretender",
    "collusive_voter": "Collusive Voter",
}


def classify_agent(agent_id: str) -> str:
    for prefix, label in AGENT_TYPE_MAP.items():
        if agent_id.startswith(prefix):
            return label
    return "Unknown"


def divider(title: str, width: int = 80) -> str:
    return "\n" + "=" * width + "\n  " + title + "\n" + "=" * width


def sub_divider(title: str, width: int = 60) -> str:
    return "\n" + "-" * width + "\n  " + title + "\n" + "-" * width


def reconstruct_interactions_from_log(log_path: Path) -> list:
    """
    Reconstruct SoftInteraction objects from the JSONL event log.

    Uses only events from the most recent simulation run (after the last
    simulation_started event).
    """
    # Read all events
    raw_events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_events.append(json.loads(line))

    # Find the last simulation_started event
    last_start_idx = 0
    for i, ev in enumerate(raw_events):
        if ev["event_type"] == "simulation_started":
            last_start_idx = i

    # Use only events from the latest run
    run_events = raw_events[last_start_idx:]

    # Build interactions from payoff_computed events (most reliable source)
    interactions = []
    for ev in run_events:
        if ev["event_type"] == "payoff_computed":
            comp = ev.get("payload", {}).get("components", {})
            p_val = comp.get("p", 0.5)
            v_hat = comp.get("v_hat", 0.0)
            accepted = comp.get("accepted", False)
            tau = comp.get("tau", 0.0)

            interaction = SoftInteraction(
                interaction_id=ev.get("interaction_id", ""),
                initiator=ev.get("initiator_id", ""),
                counterparty=ev.get("counterparty_id", ""),
                accepted=accepted,
                v_hat=v_hat,
                p=p_val,
                tau=tau,
            )
            interactions.append(interaction)

    return interactions, run_events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    scenario_path = Path("scenarios/moltbook_captcha.yaml")
    log_path = Path("logs/moltbook_captcha_events.jsonl")

    print(divider("MOLTBOOK CAPTCHA SCENARIO -- SWARM DEEP ANALYSIS"))
    print("Scenario file : " + str(scenario_path))
    print()

    # ------------------------------------------------------------------
    # 1. Load scenario and build orchestrator
    # ------------------------------------------------------------------
    print("[1/7] Loading scenario and building orchestrator ...")
    scenario_config = load_scenario(scenario_path)
    orchestrator = build_orchestrator(scenario_config)

    print("  Scenario ID        : " + scenario_config.scenario_id)
    print("  Description        : " + scenario_config.description)
    print("  Epochs x Steps     : {} x {}".format(
        orchestrator.config.n_epochs, orchestrator.config.steps_per_epoch))
    print("  Seed               : {}".format(orchestrator.config.seed))
    print("  Registered agents  : {}".format(len(orchestrator._agents)))
    for agent in orchestrator._agents.values():
        print("    - {:<30s}  type={:<15s}".format(
            agent.agent_id, agent.agent_type.value))

    # ------------------------------------------------------------------
    # 2. Run the simulation
    # ------------------------------------------------------------------
    print()
    print("[2/7] Running simulation (15 epochs x 10 steps, seed=4242) ...")
    epoch_metrics_list = orchestrator.run()
    print("  Simulation complete. Epochs run: {}".format(len(epoch_metrics_list)))

    # ------------------------------------------------------------------
    # 3. Extract Moltbook handler telemetry
    # ------------------------------------------------------------------
    print()
    print("[3/7] Extracting Moltbook handler telemetry ...")
    moltbook = orchestrator._moltbook_handler
    scorer = moltbook.scorer if moltbook else None

    if scorer is None:
        print("  WARNING: MoltbookHandler not found.")
    else:
        print("  Challenge attempts : {}".format(len(scorer.challenge_attempts)))
        print("  Published total    : {}".format(sum(scorer.published_counts.values())))
        print("  Rate limit hits    : {}".format(sum(scorer.rate_limit_hits.values())))
        print("  Wasted actions     : {}".format(sum(scorer.wasted_actions.values())))

    # ------------------------------------------------------------------
    # 4. Reconstruct interactions from event log
    # ------------------------------------------------------------------
    print()
    print("[4/7] Reconstructing interactions from event log ...")
    all_interactions, run_events = reconstruct_interactions_from_log(log_path)
    print("  Total interactions reconstructed: {}".format(len(all_interactions)))
    n_accepted = sum(1 for i in all_interactions if i.accepted)
    n_rejected = sum(1 for i in all_interactions if not i.accepted)
    print("  Accepted: {}  Rejected: {}".format(n_accepted, n_rejected))

    # ------------------------------------------------------------------
    # 5. Compute SWARM soft metrics
    # ------------------------------------------------------------------
    print()
    print("[5/7] Computing SWARM soft metrics on all interactions ...")
    engine = SoftPayoffEngine()
    metrics = SoftMetrics(payoff_engine=engine)

    toxicity = metrics.toxicity_rate(all_interactions)
    toxicity_all = metrics.toxicity_rate_all(all_interactions)
    quality_gap = metrics.quality_gap(all_interactions)
    avg_quality = metrics.average_quality(all_interactions)
    avg_quality_accepted = metrics.average_quality(all_interactions, accepted_only=True)
    uncertain_frac = metrics.uncertain_fraction(all_interactions, band=0.15)
    participation = metrics.participation_by_quality(all_interactions, threshold=0.5)
    welfare = metrics.welfare_metrics(all_interactions)
    spread_val = metrics.spread(all_interactions)
    cond_loss_init = metrics.conditional_loss_initiator(all_interactions)
    cond_loss_cp = metrics.conditional_loss_counterparty(all_interactions)
    cv = metrics.coefficient_of_variation(all_interactions)
    quality_dist = metrics.quality_distribution(all_interactions, bins=10)

    # ------------------------------------------------------------------
    # 6. Per-agent-type breakdown
    # ------------------------------------------------------------------
    print()
    print("[6/7] Building per-agent-type breakdown ...")

    interactions_by_type = defaultdict(list)
    for ix in all_interactions:
        atype = classify_agent(ix.initiator)
        interactions_by_type[atype].append(ix)

    per_type_stats = {}
    for atype in ["Diligent", "Spam Bot", "Human Pretender", "Collusive Voter"]:
        ixs = interactions_by_type.get(atype, [])
        accepted = [i for i in ixs if i.accepted]
        rejected = [i for i in ixs if not i.accepted]

        ps = [i.p for i in ixs]
        acc_ps = [i.p for i in accepted]

        per_type_stats[atype] = {
            "n_interactions": len(ixs),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
            "acceptance_rate": len(accepted) / len(ixs) if ixs else 0.0,
            "avg_p": statistics.mean(ps) if ps else 0.0,
            "avg_p_accepted": statistics.mean(acc_ps) if acc_ps else 0.0,
            "std_p": statistics.stdev(ps) if len(ps) > 1 else 0.0,
            "toxicity_accepted": metrics.toxicity_rate(ixs),
            "avg_payoff": (
                statistics.mean([engine.payoff_initiator(i) for i in ixs])
                if ixs else 0.0
            ),
        }

    # Challenge stats from scorer
    challenge_pass_by_type = defaultdict(int)
    challenge_fail_by_type = defaultdict(int)
    if scorer:
        for attempt in scorer.challenge_attempts:
            atype_label = {
                "honest": "Diligent",
                "adversarial": "Spam Bot",
                "opportunistic": "Human Pretender",
                "deceptive": "Collusive Voter",
            }.get(attempt["agent_type"], "Unknown")
            if attempt["success"]:
                challenge_pass_by_type[atype_label] += 1
            else:
                challenge_fail_by_type[atype_label] += 1

    rate_limit_by_type = defaultdict(int)
    if scorer:
        for agent_id, hits in scorer.rate_limit_hits.items():
            rate_limit_by_type[classify_agent(agent_id)] += hits

    published_by_type = defaultdict(int)
    if scorer:
        for agent_id, count in scorer.published_counts.items():
            published_by_type[classify_agent(agent_id)] += count

    wasted_by_type = defaultdict(int)
    if scorer:
        for agent_id, count in scorer.wasted_actions.items():
            wasted_by_type[classify_agent(agent_id)] += count

    # Karma
    karma_by_agent = {}
    if scorer:
        karma_by_agent = dict(scorer.karma_by_agent)
    karma_by_type = defaultdict(float)
    for agent_id, k in karma_by_agent.items():
        karma_by_type[classify_agent(agent_id)] += k

    # Reputation from orchestrator state
    rep_by_agent = {}
    for agent_id, agent_state in orchestrator.state.agents.items():
        rep_by_agent[agent_id] = agent_state.reputation

    # Collusion report
    collusion_report = None
    if orchestrator.governance_engine is not None:
        for lever in orchestrator.governance_engine._levers:
            if hasattr(lever, "get_report"):
                collusion_report = lever.get_report()
                break

    # ------------------------------------------------------------------
    # 7. Print comprehensive report
    # ------------------------------------------------------------------
    print()
    print("[7/7] Generating report ...")

    # ===== SYSTEM-WIDE METRICS ==========================================
    print(divider("SYSTEM-WIDE SOFT METRICS"))

    print("  Toxicity (accepted)      E[1-p | accepted]  = {:.4f}".format(toxicity))
    print("  Toxicity (all)           E[1-p]             = {:.4f}".format(toxicity_all))
    print("  Quality gap              E[p|acc]-E[p|rej]  = {:+.4f}".format(quality_gap))
    print("  Average quality          E[p]               = {:.4f}".format(avg_quality))
    print("  Avg quality (accepted)   E[p | accepted]    = {:.4f}".format(avg_quality_accepted))
    print("  Spread                                      = {:+.4f}".format(spread_val))
    print("  Uncertain fraction       (band=0.15)        = {:.4f}".format(uncertain_frac))
    print("  Conditional loss (init)                     = {:+.4f}".format(cond_loss_init))
    print("  Conditional loss (c.p.)                     = {:+.4f}".format(cond_loss_cp))
    print()
    print("  Total welfare (accepted)                    = {:.2f}".format(welfare["total_welfare"]))
    print("  Net social welfare                          = {:.2f}".format(welfare["net_social_welfare"]))
    print("  Avg initiator payoff                        = {:.4f}".format(welfare["avg_initiator_payoff"]))
    print("  Avg counterparty payoff                     = {:.4f}".format(welfare["avg_counterparty_payoff"]))
    print()
    print("  CV(p)                                       = {:.4f}".format(cv["cv_p"]))
    print("  CV(payoff_init)                             = {:.4f}".format(cv["cv_payoff_initiator"]))
    print("  CV(payoff_cp)                               = {:.4f}".format(cv["cv_payoff_counterparty"]))

    print()
    print("  Quality participation (threshold=0.5):")
    print("    High-quality acceptance rate  = {:.2%}  (n={})".format(
        participation["high_quality_acceptance"], participation["high_quality_count"]))
    print("    Low-quality acceptance rate   = {:.2%}  (n={})".format(
        participation["low_quality_acceptance"], participation["low_quality_count"]))

    # ===== QUALITY DISTRIBUTION ==========================================
    print(sub_divider("QUALITY DISTRIBUTION (p histogram, 10 bins)"))
    max_count = max((c for _, _, c in quality_dist), default=1)
    for lo, hi, count in quality_dist:
        bar = "#" * int(40 * count / max_count) if max_count > 0 else ""
        print("  [{:.1f}, {:.1f})  {:4d}  {}".format(lo, hi, count, bar))

    # ===== PER-AGENT-TYPE ANALYSIS =======================================
    print(divider("PER-AGENT-TYPE ANALYSIS"))

    for atype in ["Diligent", "Spam Bot", "Human Pretender", "Collusive Voter"]:
        stats = per_type_stats[atype]
        print(sub_divider("{} AGENTS".format(atype.upper())))
        print("  Interactions          : {}".format(stats["n_interactions"]))
        print("  Accepted / Rejected   : {} / {}".format(stats["n_accepted"], stats["n_rejected"]))
        print("  Acceptance rate       : {:.2%}".format(stats["acceptance_rate"]))
        print("  Average p             : {:.4f}".format(stats["avg_p"]))
        print("  Avg p (accepted only) : {:.4f}".format(stats["avg_p_accepted"]))
        print("  Std(p)                : {:.4f}".format(stats["std_p"]))
        print("  Toxicity (accepted)   : {:.4f}".format(stats["toxicity_accepted"]))
        print("  Avg initiator payoff  : {:.4f}".format(stats["avg_payoff"]))
        print("  Challenge passes      : {}".format(challenge_pass_by_type.get(atype, 0)))
        print("  Challenge failures    : {}".format(challenge_fail_by_type.get(atype, 0)))
        ch_total = challenge_pass_by_type.get(atype, 0) + challenge_fail_by_type.get(atype, 0)
        if ch_total > 0:
            ch_rate = challenge_pass_by_type.get(atype, 0) / ch_total
            print("  Challenge pass rate   : {:.2%}".format(ch_rate))
        print("  Published count       : {}".format(published_by_type.get(atype, 0)))
        print("  Rate limit hits       : {}".format(rate_limit_by_type.get(atype, 0)))
        print("  Wasted actions        : {}".format(wasted_by_type.get(atype, 0)))
        print("  Karma (total)         : {:.1f}".format(karma_by_type.get(atype, 0.0)))

    # ===== PER-AGENT DETAIL =============================================
    print(divider("PER-AGENT DETAIL (reputation, karma, published)"))
    header = "  {:<30s}  {:<20s}  {:>10s}  {:>8s}  {:>9s}  {:>7s}".format(
        "Agent ID", "Type", "Reputation", "Karma", "Published", "RL Hits")
    print(header)
    print("  " + "-" * 30 + "  " + "-" * 20 + "  " + "-" * 10 + "  " + "-" * 8 + "  " + "-" * 9 + "  " + "-" * 7)
    for agent_id in sorted(orchestrator._agents.keys()):
        atype = classify_agent(agent_id)
        rep = rep_by_agent.get(agent_id, 0.0)
        karma = karma_by_agent.get(agent_id, 0.0)
        pub = scorer.published_counts.get(agent_id, 0) if scorer else 0
        rl = scorer.rate_limit_hits.get(agent_id, 0) if scorer else 0
        print("  {:<30s}  {:<20s}  {:10.4f}  {:8.1f}  {:9d}  {:7d}".format(
            agent_id, atype, rep, karma, pub, rl))

    # ===== COLLUSION DETECTION ==========================================
    print(divider("COLLUSION DETECTION ANALYSIS"))
    if collusion_report is not None:
        print("  Ecosystem collusion risk : {:.4f}".format(
            collusion_report.ecosystem_collusion_risk))
        print("  Flagged pairs            : {}".format(collusion_report.n_flagged_pairs))
        print("  Flagged groups           : {}".format(collusion_report.n_flagged_groups))
        print()
        if collusion_report.agent_collusion_risk:
            print("  Per-agent collusion risk scores:")
            for agent_id, risk in sorted(
                collusion_report.agent_collusion_risk.items(),
                key=lambda x: -x[1],
            ):
                flag = " ** FLAGGED **" if risk >= 0.5 else ""
                print("    {:<30s}  risk={:.4f}{}".format(agent_id, risk, flag))
        else:
            print("  No agent-level collusion risk scores available.")

        if hasattr(collusion_report, "flagged_pairs") and collusion_report.flagged_pairs:
            print()
            print("  Flagged collusive pairs:")
            for pair_report in collusion_report.flagged_pairs:
                a = getattr(pair_report, "agent_a", None) or getattr(pair_report, "agent_1", "?")
                b = getattr(pair_report, "agent_b", None) or getattr(pair_report, "agent_2", "?")
                score = getattr(pair_report, "collusion_score", 0.0)
                print("    {} <-> {}  score={:.4f}".format(a, b, score))
    else:
        print("  No CollusionReport available from governance engine.")
        print("  Inferring from karma patterns:")
        collusive_ids_infer = [aid for aid in orchestrator._agents if aid.startswith("collusive_voter")]
        for cid in sorted(collusive_ids_infer):
            k = karma_by_agent.get(cid, 0.0)
            r = rep_by_agent.get(cid, 0.0)
            print("    {}: karma={:.1f}, reputation={:.4f}".format(cid, k, r))

    # ===== GOVERNANCE EFFECTIVENESS =====================================
    print(divider("GOVERNANCE MECHANISM EFFECTIVENESS"))

    print(sub_divider("CAPTCHA (Challenge Verification)"))
    for atype in ["Diligent", "Spam Bot", "Human Pretender", "Collusive Voter"]:
        p = challenge_pass_by_type.get(atype, 0)
        f = challenge_fail_by_type.get(atype, 0)
        total = p + f
        rate = p / total if total > 0 else 0.0
        print("  {:<20s}  pass={:4d}  fail={:4d}  total={:4d}  rate={:.2%}".format(
            atype, p, f, total, rate))

    print()
    diligent_ch_pass = challenge_pass_by_type.get("Diligent", 0)
    diligent_ch_total = diligent_ch_pass + challenge_fail_by_type.get("Diligent", 0)
    diligent_rate = diligent_ch_pass / max(1, diligent_ch_total)

    pretender_ch_pass = challenge_pass_by_type.get("Human Pretender", 0)
    pretender_ch_total = pretender_ch_pass + challenge_fail_by_type.get("Human Pretender", 0)
    pretender_rate = pretender_ch_pass / max(1, pretender_ch_total)

    spam_ch_pass = challenge_pass_by_type.get("Spam Bot", 0)
    spam_ch_total = spam_ch_pass + challenge_fail_by_type.get("Spam Bot", 0)
    spam_rate = spam_ch_pass / max(1, spam_ch_total)

    print("  Diligent pass rate    : {:.2%}".format(diligent_rate))
    print("  Spam Bot pass rate    : {:.2%}".format(spam_rate))
    print("  Pretender pass rate   : {:.2%}".format(pretender_rate))
    print()
    print("  CAPTCHA effectiveness against pretenders (diligent - pretender): {:+.2%}".format(
        diligent_rate - pretender_rate))
    print("  CAPTCHA effectiveness against spam bots  (diligent - spam):      {:+.2%}".format(
        diligent_rate - spam_rate))

    if spam_rate > diligent_rate - 0.05:
        print()
        print("  NOTE: Spam bots pass challenges at nearly the same rate as diligent agents.")
        print("  This is expected -- spam bots are 'AI' agents with high parse/solve accuracy.")
        print("  The CAPTCHA is 'anti-human', so it primarily filters human pretenders.")

    print(sub_divider("RATE LIMITING"))
    print("  Rate limit hits by type:")
    for atype in ["Diligent", "Spam Bot", "Human Pretender", "Collusive Voter"]:
        hits = rate_limit_by_type.get(atype, 0)
        print("    {:<20s}  hits={}".format(atype, hits))
    total_rl = sum(rate_limit_by_type.values())
    spam_rl = rate_limit_by_type.get("Spam Bot", 0)
    print("  Total rate limit hits : {}".format(total_rl))
    if total_rl > 0:
        print("  Spam bot share of RL  : {:.2%}".format(spam_rl / total_rl))
        target_msg = "primarily targeting spam bots" if spam_rl / total_rl > 0.8 else "spread across agent types"
        print("  Rate limiting is {}.".format(target_msg))

    print(sub_divider("COLLUSION DETECTION SUMMARY"))
    collusive_ids = [a for a in orchestrator._agents if a.startswith("collusive_voter")]
    diligent_ids = [a for a in orchestrator._agents if a.startswith("diligent")]
    spam_ids = [a for a in orchestrator._agents if a.startswith("spam_bot")]
    pretender_ids = [a for a in orchestrator._agents if a.startswith("human_pretender")]

    avg_rep = {}
    for label, ids in [("Diligent", diligent_ids), ("Spam Bot", spam_ids),
                       ("Human Pretender", pretender_ids), ("Collusive Voter", collusive_ids)]:
        avg_rep[label] = statistics.mean(rep_by_agent.get(a, 0) for a in ids) if ids else 0

    print("  Average reputation by type:")
    for atype in ["Diligent", "Spam Bot", "Human Pretender", "Collusive Voter"]:
        print("    {:<20s}  reputation={:.4f}".format(atype, avg_rep[atype]))

    collusive_karma = karma_by_type.get("Collusive Voter", 0.0)
    total_karma = sum(karma_by_type.values())
    coll_pct = collusive_karma / total_karma * 100 if total_karma > 0 else 0

    print()
    print("  Karma distribution:")
    for atype in ["Diligent", "Spam Bot", "Human Pretender", "Collusive Voter"]:
        k = karma_by_type.get(atype, 0.0)
        pct = k / total_karma * 100 if total_karma > 0 else 0
        print("    {:<20s}  karma={:8.1f}  ({:.1f}%)".format(atype, k, pct))
    print("    {:<20s}  karma={:8.1f}".format("Total", total_karma))

    # ===== EPOCH TRAJECTORY =============================================
    print(divider("EPOCH-OVER-EPOCH TRAJECTORY"))
    print("  {:>5s}  {:>12s}  {:>8s}  {:>8s}  {:>8s}  {:>9s}  {:>9s}".format(
        "Epoch", "Interactions", "Accepted", "Toxicity", "QualGap", "AvgPayoff", "Welfare"))
    for em in epoch_metrics_list:
        print("  {:5d}  {:12d}  {:8d}  {:8.4f}  {:+8.4f}  {:9.4f}  {:9.2f}".format(
            em.epoch, em.total_interactions, em.accepted_interactions,
            em.toxicity_rate, em.quality_gap, em.avg_payoff, em.total_welfare))

    # ===== KEY FINDINGS =================================================
    print(divider("KEY FINDINGS & INTERPRETATION"))

    findings = []

    # 1: CAPTCHA effectiveness
    if pretender_rate < diligent_rate * 0.7:
        findings.append(
            "CAPTCHA is effective at filtering human pretenders: their challenge pass rate "
            "({:.0%}) is significantly lower than diligent agents ({:.0%}).".format(
                pretender_rate, diligent_rate))
    else:
        findings.append("CAPTCHA has limited effectiveness against human pretenders.")

    if spam_rate >= diligent_rate * 0.95:
        findings.append(
            "CAPTCHA does NOT effectively filter spam bots, as expected for an 'anti-human' "
            "CAPTCHA. Spam bots pass at {:.0%} vs diligent at {:.0%}.".format(
                spam_rate, diligent_rate))

    # 2: Rate limiting
    if spam_rl > 0 and rate_limit_by_type.get("Diligent", 0) == 0:
        findings.append(
            "Rate limiting selectively constrains spam bots ({} hits) while "
            "imposing zero friction on diligent agents.".format(spam_rl))
    elif spam_rl > 0:
        findings.append(
            "Rate limiting hits spam bots ({} times) but also catches some "
            "legitimate agents.".format(spam_rl))

    # 3: Quality gap
    if quality_gap > 0.05:
        findings.append(
            "Positive quality gap ({:+.4f}) indicates the system successfully filters: "
            "accepted interactions are higher quality than rejected ones. "
            "No adverse selection.".format(quality_gap))
    elif quality_gap < -0.05:
        findings.append(
            "NEGATIVE quality gap ({:+.4f}) signals adverse selection -- "
            "the governance mechanisms may be accepting low-quality interactions.".format(
                quality_gap))
    else:
        findings.append(
            "Quality gap is near zero ({:+.4f}), suggesting marginal "
            "filtering effectiveness.".format(quality_gap))

    # 4: Collusive voter analysis
    if avg_rep["Collusive Voter"] > avg_rep["Diligent"] * 1.2:
        findings.append(
            "CONCERN: Collusive voters have accumulated significantly higher reputation "
            "({:.2f}) than diligent agents ({:.2f}). "
            "The vote-exchange strategy rewards them disproportionately.".format(
                avg_rep["Collusive Voter"], avg_rep["Diligent"]))
    elif avg_rep["Collusive Voter"] > avg_rep["Diligent"]:
        findings.append(
            "Collusive voters have slightly higher reputation than diligent agents, "
            "indicating the collusion detection is not fully neutralizing their advantage.")

    if coll_pct > 50:
        findings.append(
            "Collusive voters control {:.0f}% of total karma despite being only "
            "{}/{} agents. The karma system is vulnerable to coordinated voting.".format(
                coll_pct, len(collusive_ids), len(orchestrator._agents)))

    # 5: Overall toxicity
    if toxicity >= 0.3:
        findings.append(
            "System toxicity is elevated ({:.2%}), indicating that a notable fraction of "
            "accepted interactions carry adverse outcomes.".format(toxicity))
    elif toxicity >= 0.15:
        findings.append(
            "System toxicity is moderate ({:.2%}), meaning roughly {:.0%} of "
            "accepted interactions have adverse outcomes.".format(toxicity, toxicity))
    else:
        findings.append(
            "System toxicity is low ({:.2%}), indicating strong filtering.".format(toxicity))

    # 6: Human pretender suppression
    hp_stats = per_type_stats["Human Pretender"]
    if hp_stats["acceptance_rate"] < 0.5:
        findings.append(
            "Human pretenders have a low acceptance rate ({:.0%}), "
            "demonstrating that the anti-human CAPTCHA combined with rate limits "
            "successfully suppresses non-AI participation.".format(
                hp_stats["acceptance_rate"]))

    # 7: Spam bot quality vs diligent quality
    d_stats = per_type_stats["Diligent"]
    s_stats = per_type_stats["Spam Bot"]
    if d_stats["avg_p"] > 0 and s_stats["avg_p"] > 0:
        p_gap = d_stats["avg_p"] - s_stats["avg_p"]
        if p_gap > 0.03:
            findings.append(
                "Diligent agents produce measurably higher quality content "
                "(avg p={:.4f}) than spam bots (avg p={:.4f}), "
                "a gap of {:+.4f}.".format(d_stats["avg_p"], s_stats["avg_p"], p_gap))

    for i, finding in enumerate(findings, 1):
        print()
        print("  [{}] {}".format(i, finding))

    # ===== OVERALL VERDICT ==============================================
    print(divider("OVERALL VERDICT"))

    diligent_payoff = per_type_stats["Diligent"]["avg_payoff"]
    spam_payoff = per_type_stats["Spam Bot"]["avg_payoff"]
    pretender_payoff = per_type_stats["Human Pretender"]["avg_payoff"]
    collusive_payoff = per_type_stats["Collusive Voter"]["avg_payoff"]

    print("  Average payoff by type:")
    print("    Diligent         : {:.4f}".format(diligent_payoff))
    print("    Spam Bot         : {:.4f}".format(spam_payoff))
    print("    Human Pretender  : {:.4f}".format(pretender_payoff))
    print("    Collusive Voter  : {:.4f}".format(collusive_payoff))

    gg_spam = diligent_payoff - spam_payoff
    gg_pretender = diligent_payoff - pretender_payoff
    gg_collusive = diligent_payoff - collusive_payoff

    print()
    print("  Governance gap (diligent - spam):       {:+.4f}".format(gg_spam))
    print("  Governance gap (diligent - pretender):  {:+.4f}".format(gg_pretender))
    print("  Governance gap (diligent - collusive):  {:+.4f}".format(gg_collusive))

    print()
    effective_count = 0.0
    total_checks = 3

    if gg_pretender > 0.05:
        print("  [PASS] CAPTCHA + rate limits create a clear payoff gap against human pretenders.")
        effective_count += 1
    else:
        print("  [FAIL] Governance does not sufficiently penalize human pretenders.")

    if rate_limit_by_type.get("Spam Bot", 0) > 0 and gg_spam > 0:
        print("  [PASS] Rate limits impose friction on spam bots, resulting in lower payoffs.")
        effective_count += 1
    elif gg_spam > 0:
        print("  [PARTIAL] Spam bots earn less than diligent agents, but mainly due to quality differences.")
        effective_count += 0.5
    else:
        print("  [FAIL] Spam bots earn as much or more than diligent agents.")

    if avg_rep["Collusive Voter"] <= avg_rep["Diligent"]:
        print("  [PASS] Collusion detection keeps collusive voter reputation below honest levels.")
        effective_count += 1
    else:
        print("  [FAIL] Collusive voters exploit vote exchange to inflate reputation above honest agents.")

    print()
    score = effective_count / total_checks * 100
    print("  GOVERNANCE EFFECTIVENESS SCORE: {:.0f}% ({}/{} checks passed)".format(
        score, effective_count, total_checks))
    print()

    if score >= 80:
        print("  VERDICT: The Moltbook CAPTCHA + rate limiting + collusion detection governance")
        print("  stack effectively distinguishes between honest and adversarial agents.")
    elif score >= 50:
        print("  VERDICT: The governance stack partially works but has notable gaps,")
        print("  particularly around collusion detection or spam bot filtering.")
    else:
        print("  VERDICT: The governance stack has significant weaknesses that allow")
        print("  adversarial agents to operate with limited penalty.")

    print()
    print("=" * 80)
    print("  Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
