"use client";

import React, { useMemo } from "react";
import { useSimulation } from "@/state/use-simulation";
import { useGame } from "@/state/game-context";
import { AGENT_COLORS, AGENT_LABELS } from "@/engine/constants";
import { formatNumber, formatPercent } from "@/utils/format";

export function InfoPanel() {
  const { selectedAgent, agents, setSelected, agentSnapshots, currentEpoch } =
    useSimulation();
  const { state: gameState, toggleFreeze, toggleQuarantine, removeAgent, spawnAgent } = useGame();

  const agent = agents.find((a) => a.id === selectedAgent);

  // All hooks must be called unconditionally (React hooks rules)
  const history = useMemo(() => {
    if (!agentSnapshots?.length || !agent) return [];
    return agentSnapshots
      .filter((s) => s.agent_id === agent.id)
      .sort((a, b) => a.epoch - b.epoch);
  }, [agentSnapshots, agent]);

  const repHistory = useMemo(() => history.map((h) => h.reputation), [history]);
  const payoffHistory = useMemo(
    () => history.map((h) => h.total_payoff),
    [history],
  );
  const pHistory = useMemo(
    () => history.map((h) => h.avg_p_initiated),
    [history],
  );

  if (!agent) return null;

  const colors = AGENT_COLORS[agent.agentType];

  const totalActivity =
    agent.interactionsInitiated + agent.interactionsReceived;
  const initRatio = totalActivity > 0
    ? agent.interactionsInitiated / totalActivity
    : 0.5;

  return (
    <div className="absolute z-30 bg-panel border border-border rounded-lg shadow-lg overflow-y-auto top-2 left-2 right-2 max-h-[45vh] md:top-4 md:left-4 md:right-auto md:w-72 md:max-h-none md:overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: colors.secondary }}
          />
          <span className="font-semibold text-sm">{agent.name}</span>
        </div>
        <button
          onClick={() => setSelected(null)}
          className="text-muted hover:text-text transition-colors text-lg leading-none"
        >
          x
        </button>
      </div>

      {/* Type badge */}
      <div className="px-4 py-2 border-b border-border flex items-center gap-2">
        <span
          className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
          style={{
            backgroundColor: colors.primary,
            color: colors.accent,
          }}
        >
          {AGENT_LABELS[agent.agentType]}
        </span>
        <span className="text-xs text-muted">{agent.agentType}</span>
        {agent.isFrozen && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-900/50 text-blue-300">
            FROZEN
          </span>
        )}
        {agent.isQuarantined && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-900/50 text-red-300">
            QUARANTINED
          </span>
        )}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-px bg-border">
        <Stat
          label="Reputation"
          value={formatNumber(agent.reputation, 3)}
          sparkline={repHistory}
          color={colors.secondary}
        />
        <Stat
          label="Resources"
          value={formatNumber(agent.resources, 1)}
        />
        <Stat
          label="Total Payoff"
          value={formatNumber(agent.totalPayoff, 2)}
          sparkline={payoffHistory}
        />
        <Stat
          label="Avg P"
          value={formatPercent(agent.avgP)}
          sparkline={pHistory}
          color={agent.avgP > 0.6 ? "#6FCF97" : agent.avgP > 0.3 ? "#F2994A" : "#EB5757"}
        />
        <Stat
          label="Initiated"
          value={String(agent.interactionsInitiated)}
        />
        <Stat
          label="Received"
          value={String(agent.interactionsReceived)}
        />
      </div>

      {/* Activity breakdown bar */}
      {totalActivity > 0 && (
        <div className="px-4 py-2 border-t border-border">
          <div className="text-[10px] text-muted uppercase tracking-wider mb-1">
            Activity Balance
          </div>
          <div className="flex h-2 rounded-full overflow-hidden bg-black/30">
            <div
              className="transition-all duration-300"
              style={{
                width: `${initRatio * 100}%`,
                backgroundColor: colors.secondary,
              }}
            />
            <div
              className="transition-all duration-300"
              style={{
                width: `${(1 - initRatio) * 100}%`,
                backgroundColor: colors.primary,
              }}
            />
          </div>
          <div className="flex justify-between text-[9px] text-muted mt-0.5">
            <span>Initiated {formatPercent(initRatio)}</span>
            <span>Received {formatPercent(1 - initRatio)}</span>
          </div>
        </div>
      )}

      {/* Agent actions (live mode only).
          The viz agent may be briefly stale vs engine state; action handlers
          in game-context guard against missing agents with no-op returns. */}
      {gameState.isLive && (
        <div className="px-4 py-2 border-t border-border flex gap-1.5 flex-wrap">
          <button
            onClick={() => toggleFreeze(agent.id)}
            className={`text-[10px] px-2 py-1 rounded transition-colors ${
              agent.isFrozen
                ? "bg-blue-900/50 text-blue-300 hover:bg-blue-900/70"
                : "bg-btn hover:bg-btn-hover text-muted"
            }`}
          >
            {agent.isFrozen ? "Unfreeze" : "Freeze"}
          </button>
          <button
            onClick={() => toggleQuarantine(agent.id)}
            className={`text-[10px] px-2 py-1 rounded transition-colors ${
              agent.isQuarantined
                ? "bg-red-900/50 text-red-300 hover:bg-red-900/70"
                : "bg-btn hover:bg-btn-hover text-muted"
            }`}
          >
            {agent.isQuarantined ? "Release" : "Quarantine"}
          </button>
          <button
            onClick={() => spawnAgent(agent.agentType)}
            className="text-[10px] px-2 py-1 rounded bg-btn hover:bg-btn-hover text-muted transition-colors"
          >
            Clone
          </button>
          <button
            onClick={() => { removeAgent(agent.id); setSelected(null); }}
            className="text-[10px] px-2 py-1 rounded bg-red-950/50 hover:bg-red-900/50 text-red-400 transition-colors"
          >
            Remove
          </button>
        </div>
      )}

      {/* Epoch indicator */}
      <div className="px-4 py-1.5 border-t border-border text-[10px] text-muted text-center">
        Epoch {currentEpoch} | Power: {(agent.scale * 100).toFixed(0)}%
      </div>
    </div>
  );
}

function MiniSparkline({
  data,
  color = "#3ECFB4",
  width = 48,
  height = 16,
}: {
  data: number[];
  color?: string;
  width?: number;
  height?: number;
}) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 2) - 1;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} className="inline-block ml-1 align-middle">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function Stat({
  label,
  value,
  color,
  sparkline,
}: {
  label: string;
  value: string;
  color?: string;
  sparkline?: number[];
}) {
  return (
    <div className="bg-panel px-3 py-2">
      <div className="text-[10px] text-muted uppercase tracking-wider">
        {label}
      </div>
      <div className="flex items-center justify-between mt-0.5">
        <span
          className="text-sm font-mono"
          style={color ? { color } : undefined}
        >
          {value}
        </span>
        {sparkline && sparkline.length > 1 && (
          <MiniSparkline data={sparkline} color={color || "#3ECFB4"} />
        )}
      </div>
    </div>
  );
}
