"use client";

import React, { useState, useMemo } from "react";
import { useSimulation } from "@/state/use-simulation";
import type { SwarmEvent, SwarmEventType } from "@/data/types";

const EVENT_COLORS: Record<SwarmEventType, string> = {
  simulation_started: "#7D8590",
  simulation_ended: "#7D8590",
  epoch_completed: "#7D8590",
  agent_created: "#2F80ED",
  contract_signing: "#BB6BD9",
  interaction_proposed: "#3ECFB4",
  interaction_completed: "#3ECFB4",
  governance_cost_applied: "#F2994A",
  reputation_updated: "#6FCF97",
  payoff_computed: "#E6EDF3",
  contract_metrics: "#BB6BD9",
};

const EVENT_LABELS: Partial<Record<SwarmEventType, string>> = {
  interaction_proposed: "PROPOSE",
  interaction_completed: "COMPLETE",
  reputation_updated: "REP",
  governance_cost_applied: "GOV",
  payoff_computed: "PAYOFF",
  agent_created: "AGENT",
  contract_signing: "CONTRACT",
  epoch_completed: "EPOCH",
};

// ─── Governance Receipt Detection ──────────────────────────────────

interface GovernanceReceipt {
  type: "circuit_breaker" | "quarantine" | "collusion" | "freeze" | "high_tax";
  label: string;
  detail: string;
  color: string;
}

function detectReceipt(evt: SwarmEvent): GovernanceReceipt | null {
  const p = evt.payload;

  // Circuit breaker fired
  if (evt.event_type === "governance_cost_applied" && p.circuit_breaker_triggered) {
    return {
      type: "circuit_breaker",
      label: "CIRCUIT BREAKER",
      detail: `Triggered at toxicity ${((p.toxicity_rate as number) ?? 0).toFixed(2)} \u2014 interactions halted`,
      color: "#EB5757",
    };
  }

  // Agent quarantined
  if (evt.event_type === "reputation_updated" && p.quarantined) {
    return {
      type: "quarantine",
      label: "QUARANTINE",
      detail: `${evt.agent_id} quarantined (reputation collapsed)`,
      color: "#F2994A",
    };
  }

  // Agent frozen
  if (evt.event_type === "reputation_updated" && p.frozen) {
    return {
      type: "freeze",
      label: "FROZEN",
      detail: `${evt.agent_id} frozen by governance`,
      color: "#56CCF2",
    };
  }

  // Collusion detected
  if (evt.event_type === "contract_metrics" && (p.collusion_detected || p.flagged_pairs)) {
    const nPairs = (p.flagged_pairs as number) ?? (p.n_flagged_pairs as number) ?? 0;
    return {
      type: "collusion",
      label: "COLLUSION DETECTED",
      detail: `${nPairs} suspicious pair${nPairs !== 1 ? "s" : ""} flagged`,
      color: "#EB5757",
    };
  }

  // High governance cost (tax > 0.5)
  if (evt.event_type === "governance_cost_applied") {
    const cost = (p.cost_a as number) ?? (p.cost_b as number) ?? (p.cost as number);
    if (cost != null && cost > 0.5) {
      return {
        type: "high_tax",
        label: "HEAVY TAX",
        detail: `${evt.agent_id} taxed ${cost.toFixed(2)}`,
        color: "#F2994A",
      };
    }
  }

  return null;
}

// ─── Receipt Card ──────────────────────────────────────────────────

function ReceiptCard({ receipt }: { receipt: GovernanceReceipt }) {
  return (
    <div
      className="mx-1 my-1.5 px-3 py-2 rounded-lg border text-xs"
      style={{
        borderColor: receipt.color + "44",
        backgroundColor: receipt.color + "11",
      }}
    >
      <div className="flex items-center gap-2 mb-0.5">
        <span
          className="px-1.5 py-0.5 rounded font-bold text-[10px] tracking-wider"
          style={{ backgroundColor: receipt.color + "33", color: receipt.color }}
        >
          {receipt.label}
        </span>
        <span className="text-muted/60 text-[10px]">Governance Receipt</span>
      </div>
      <div className="text-muted font-mono mt-1">{receipt.detail}</div>
    </div>
  );
}

// ─── Payload Formatting ────────────────────────────────────────────

function formatPayload(evt: SwarmEvent): string {
  switch (evt.event_type) {
    case "interaction_proposed":
      return `${evt.initiator_id} -> ${evt.counterparty_id} (${evt.payload.interaction_type})`;
    case "interaction_completed":
      return `${evt.interaction_id?.slice(0, 8)}... ${evt.payload.accepted ? "accepted" : "rejected"}`;
    case "reputation_updated": {
      const delta = evt.payload.delta as number;
      const sign = delta >= 0 ? "+" : "";
      return `${evt.agent_id} ${sign}${delta.toFixed(3)}`;
    }
    case "governance_cost_applied": {
      const govCost = (evt.payload.cost_a as number) ?? (evt.payload.cost_b as number) ?? (evt.payload.cost as number);
      return `${evt.agent_id} cost=${govCost?.toFixed(2) ?? "?"}`;
    }
    case "payoff_computed":
      return `${evt.initiator_id} -> ${evt.counterparty_id}`;
    default:
      return evt.agent_id ?? "";
  }
}

export function EventFeed() {
  const { data, eventIndex, currentEpoch, currentStep, stepPlayback } = useSimulation();
  const [visible, setVisible] = useState(false);

  // Get events for current position
  const events = useMemo((): SwarmEvent[] => {
    if (!eventIndex || !stepPlayback) return [];
    return eventIndex.eventsAt(currentEpoch, currentStep);
  }, [eventIndex, stepPlayback, currentEpoch, currentStep]);

  // Separate governance receipts from normal events
  const { receipts, normalEvents } = useMemo(() => {
    const r: GovernanceReceipt[] = [];
    const n: SwarmEvent[] = [];
    for (const evt of events) {
      const receipt = detectReceipt(evt);
      if (receipt) {
        r.push(receipt);
      } else {
        n.push(evt);
      }
    }
    return { receipts: r, normalEvents: n };
  }, [events]);

  // Don't render toggle if no event data
  if (!data?.rawEvents || data.rawEvents.length === 0) return null;

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setVisible((v) => !v)}
        className={`relative z-[10001] px-2.5 py-1.5 text-xs rounded transition-colors ${
          visible ? "bg-secondary text-bg font-bold" : "bg-btn hover:bg-btn-hover text-muted"
        }`}
      >
        Events{receipts.length > 0 && (
          <span className="ml-1 px-1 py-0.5 rounded-full bg-red-500/80 text-white text-[9px] font-bold">
            {receipts.length}
          </span>
        )}
      </button>

      {/* Feed panel */}
      {visible && (
        <div className="absolute top-9 right-0 z-[10001] w-80 max-w-[calc(100vw-1rem)] max-h-[50vh] bg-panel/95 border border-border rounded-lg overflow-hidden flex flex-col backdrop-blur-sm">
          <div className="px-3 py-2 border-b border-border text-xs text-muted font-mono">
            Epoch {currentEpoch} / Step {currentStep} \u2014 {events.length} events
          </div>

          {/* Governance receipts (pinned at top) */}
          {receipts.length > 0 && (
            <div className="border-b border-border">
              {receipts.map((r, i) => (
                <ReceiptCard key={`receipt-${i}`} receipt={r} />
              ))}
            </div>
          )}

          <div className="overflow-y-auto flex-1 p-1">
            {normalEvents.length === 0 ? (
              <div className="text-xs text-muted/50 text-center py-4">No events at this step</div>
            ) : (
              normalEvents.map((evt, i) => (
                <div key={evt.event_id || i} className="flex items-start gap-2 px-2 py-1 hover:bg-btn/50 rounded text-xs">
                  <span
                    className="shrink-0 px-1 py-0.5 rounded font-mono text-[10px] leading-tight"
                    style={{ backgroundColor: EVENT_COLORS[evt.event_type] + "22", color: EVENT_COLORS[evt.event_type] }}
                  >
                    {EVENT_LABELS[evt.event_type] ?? evt.event_type.toUpperCase()}
                  </span>
                  <span className="text-muted truncate font-mono">{formatPayload(evt)}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </>
  );
}
