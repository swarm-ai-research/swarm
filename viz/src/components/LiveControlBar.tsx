"use client";

import React, { useState, useCallback } from "react";
import { useGame } from "@/state/game-context";
import { AGENT_PROFILES, AGENT_TYPES } from "@/engine/sim/agents";
import { SHOCK_TEMPLATES } from "@/engine/sim/shocks";
import type { AgentType } from "@/data/types";
import { describeWinCondition, describeLoseCondition } from "@/engine/sim/campaign";
import { DEFAULT_CONFIG } from "@/engine/sim/types";

/** Fallback governance when engine is not yet initialized */
const DEFAULT_GOVERNANCE = DEFAULT_CONFIG.governance;

const TICK_RATES = [
  { label: "0.5x", value: 400 },
  { label: "1x", value: 200 },
  { label: "2x", value: 100 },
  { label: "4x", value: 50 },
  { label: "10x", value: 20 },
];

export function LiveControlBar() {
  const {
    state,
    togglePause,
    setTickRate,
    spawnAgent,
    injectShock,
    updateGovernance,
    stopGame,
    saveGame,
    computeRedTeamScore,
    dispatch,
  } = useGame();

  const [showSpawnMenu, setShowSpawnMenu] = useState(false);
  const [showShockMenu, setShowShockMenu] = useState(false);
  const [showGovPanel, setShowGovPanel] = useState(false);

  // Read current governance from engine (falls back to defaults before engine init)
  const engine = useGame().engineRef.current;
  const governance = engine?.governance ?? DEFAULT_GOVERNANCE;
  const epoch = engine?.epoch ?? 0;
  const step = engine?.step ?? 0;
  const agentCount = engine?.agents.length ?? 0;

  const handleSpawn = useCallback((type: AgentType) => {
    spawnAgent(type);
    setShowSpawnMenu(false);
  }, [spawnAgent]);

  const handleSave = useCallback(() => {
    const json = saveGame();
    if (!json) return;
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `swarm-save-epoch${epoch}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [saveGame, epoch]);

  const redTeamScore = state.mode === "redteam" ? computeRedTeamScore() : 0;

  return (
    <div className="absolute bottom-0 left-0 right-0 bg-panel border-t border-border z-[10000] pb-[env(safe-area-inset-bottom)]">
      {/* Campaign objective bar */}
      {state.mode === "campaign" && state.currentLevel && state.levelStatus === "playing" && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 sm:px-4 py-1.5 border-b border-border bg-blue-950/30 text-xs">
          <span className="text-blue-300 font-bold">{state.currentLevel.name}</span>
          <span className="text-muted">Win: {describeWinCondition(state.currentLevel.winCondition)}</span>
          <span className="text-muted">|</span>
          <span className="text-red-400">Lose: {describeLoseCondition(state.currentLevel.loseCondition)}</span>
          {state.currentLevel.hints[state.hintIndex] && (
            <button
              onClick={() => dispatch({ type: "NEXT_HINT" })}
              className="ml-auto text-yellow-400 hover:text-yellow-300 transition-colors"
              title={state.currentLevel.hints[state.hintIndex]}
            >
              Hint ({state.hintIndex + 1}/{state.currentLevel.hints.length})
            </button>
          )}
        </div>
      )}

      {/* Red team score bar */}
      {state.mode === "redteam" && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 sm:px-4 py-1.5 border-b border-border bg-red-950/30 text-xs">
          <span className="text-red-300 font-bold">RED TEAM</span>
          <span className="text-muted">Score:</span>
          <span className="font-mono text-red-400">{redTeamScore.toFixed(1)}</span>
          <span className="text-muted">|</span>
          <span className="text-muted">Best:</span>
          <span className="font-mono text-yellow-400">{state.redTeamBestScore.toFixed(1)}</span>
          <span className="ml-auto text-muted">Cause maximum system damage!</span>
        </div>
      )}

      {/* Main controls */}
      <div className="min-h-14 flex flex-wrap items-center gap-x-2 gap-y-1.5 px-3 py-2 sm:px-4 md:h-14 md:flex-nowrap md:py-0">
        {/* Play/Pause */}
        <button
          onClick={togglePause}
          className="w-9 h-9 flex items-center justify-center rounded bg-btn hover:bg-btn-hover transition-colors"
          title={state.isPaused ? "Resume" : "Pause"}
          aria-label={state.isPaused ? "Resume simulation" : "Pause simulation"}
        >
          {state.isPaused ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <path d="M3 1.5v11l9-5.5z" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <rect x="2" y="1" width="4" height="12" rx="1" />
              <rect x="8" y="1" width="4" height="12" rx="1" />
            </svg>
          )}
        </button>

        {/* Epoch/Step info */}
        <span className="text-xs text-muted font-mono w-28 text-center">
          E{epoch} S{step}/{engine?.config.stepsPerEpoch ?? 0}
        </span>

        {/* Speed selector */}
        <div className="flex items-center gap-1">
          {TICK_RATES.map((r) => (
            <button
              key={r.value}
              onClick={() => setTickRate(r.value)}
              className={`px-1.5 py-1 text-[10px] rounded transition-colors ${
                state.tickRate === r.value
                  ? "bg-accent text-bg font-bold"
                  : "bg-btn hover:bg-btn-hover text-muted"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>

        <div className="w-px h-8 bg-border mx-1" />

        {/* Spawn Agent */}
        <div className="relative">
          <button
            onClick={() => { setShowSpawnMenu(!showSpawnMenu); setShowShockMenu(false); setShowGovPanel(false); }}
            className="px-2 py-1.5 text-xs rounded bg-btn hover:bg-btn-hover transition-colors flex items-center gap-1"
          >
            <span className="text-accent">+</span> Spawn
          </button>
          {showSpawnMenu && (
            <div className="absolute bottom-full mb-1 left-0 bg-panel border border-border rounded-lg shadow-lg py-1 w-44 z-50">
              {AGENT_TYPES.filter((t) => t !== "rlm" && t !== "crewai").map((type) => (
                <button
                  key={type}
                  onClick={() => handleSpawn(type)}
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-btn-hover transition-colors flex justify-between"
                >
                  <span>{type.charAt(0).toUpperCase() + type.slice(1)}</span>
                  <span className="text-muted">{AGENT_PROFILES[type].disposition > 0 ? "+" : ""}{AGENT_PROFILES[type].disposition.toFixed(1)}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Inject Shock */}
        <div className="relative">
          <button
            onClick={() => { setShowShockMenu(!showShockMenu); setShowSpawnMenu(false); setShowGovPanel(false); }}
            className="px-2 py-1.5 text-xs rounded bg-btn hover:bg-btn-hover transition-colors flex items-center gap-1"
          >
            <span className="text-yellow-400">!</span> Shock
          </button>
          {showShockMenu && (
            <div className="absolute bottom-full mb-1 left-0 bg-panel border border-border rounded-lg shadow-lg py-1 w-56 z-50">
              {SHOCK_TEMPLATES.map((tpl) => (
                <button
                  key={tpl.id}
                  onClick={() => { injectShock(tpl.shock); setShowShockMenu(false); }}
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-btn-hover transition-colors"
                  title={tpl.description}
                >
                  <span className="mr-1.5">{tpl.icon}</span>
                  <span style={{ color: tpl.color }}>{tpl.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Governance toggle */}
        <button
          onClick={() => { setShowGovPanel(!showGovPanel); setShowSpawnMenu(false); setShowShockMenu(false); }}
          className={`px-2 py-1.5 text-xs rounded transition-colors flex items-center gap-1 ${
            showGovPanel ? "bg-secondary text-bg font-bold" : "bg-btn hover:bg-btn-hover"
          }`}
        >
          Gov
        </button>

        <div className="flex-1" />

        {/* Agent count */}
        <span className="text-xs text-muted font-mono">{agentCount} agents</span>

        {/* Save */}
        <button
          onClick={handleSave}
          className="px-2 py-1.5 text-xs rounded bg-btn hover:bg-btn-hover transition-colors"
          title="Save game"
        >
          Save
        </button>

        {/* Stop */}
        <button
          onClick={stopGame}
          className="px-2 py-1.5 text-xs rounded bg-red-900/50 hover:bg-red-900/70 text-red-300 transition-colors"
        >
          End
        </button>
      </div>

      {/* Governance panel (slides up) */}
      {showGovPanel && (
        <div className="border-t border-border px-4 py-3 bg-panel">
          <div className="grid grid-cols-2 gap-4 text-xs sm:grid-cols-4">
            <div>
              <label className="text-muted block mb-1">Tax Rate</label>
              <input
                type="range"
                min={0}
                max={0.5}
                step={0.01}
                value={governance.taxRate}
                onChange={(e) => updateGovernance({ taxRate: parseFloat(e.target.value) })}
                className="w-full h-1 accent-accent"
              />
              <span className="font-mono text-muted">{governance.taxRate.toFixed(2)}</span>
            </div>
            <div>
              <label className="text-muted block mb-1">Rep. Decay</label>
              <input
                type="range"
                min={0.8}
                max={1.0}
                step={0.01}
                value={governance.reputationDecay}
                onChange={(e) => updateGovernance({ reputationDecay: parseFloat(e.target.value) })}
                className="w-full h-1 accent-accent"
              />
              <span className="font-mono text-muted">{governance.reputationDecay.toFixed(2)}</span>
            </div>
            <div>
              <label className="text-muted block mb-1">Circuit Breaker</label>
              <button
                onClick={() => updateGovernance({ circuitBreakerEnabled: !governance.circuitBreakerEnabled })}
                className={`px-2 py-0.5 rounded text-xs ${governance.circuitBreakerEnabled ? "bg-accent text-bg" : "bg-btn"}`}
              >
                {governance.circuitBreakerEnabled ? "ON" : "OFF"}
              </button>
            </div>
            <div>
              <label className="text-muted block mb-1">CB Threshold</label>
              <input
                type="range"
                min={0.1}
                max={0.6}
                step={0.05}
                value={governance.circuitBreakerThreshold}
                onChange={(e) => updateGovernance({ circuitBreakerThreshold: parseFloat(e.target.value) })}
                className="w-full h-1 accent-accent"
              />
              <span className="font-mono text-muted">{governance.circuitBreakerThreshold.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
