"use client";

import React, { useState, useCallback } from "react";
import { usePlayback } from "@/state/use-playback";
import { useSimulation } from "@/state/use-simulation";
import { getStoredConfig } from "@/engine/sim/config-store";
import { runSimulation } from "@/engine/sim/orchestrator";
import type { GovernanceIntervention as Intervention } from "@/engine/sim/orchestrator";
import { recordToLeaderboard } from "./Leaderboard";

export function GovernanceIntervention() {
  const { currentEpoch, maxEpoch, pause } = usePlayback();
  const { data, loadData } = useSimulation();
  const [open, setOpen] = useState(false);
  const [taxRate, setTaxRate] = useState(0.05);
  const [repDecay, setRepDecay] = useState(0.95);
  const [cbEnabled, setCbEnabled] = useState(false);

  const storedConfig = getStoredConfig();

  const handleOpen = useCallback(() => {
    pause();
    // Initialize sliders from stored config
    if (storedConfig) {
      setTaxRate(storedConfig.governance.taxRate);
      setRepDecay(storedConfig.governance.reputationDecay);
      setCbEnabled(storedConfig.governance.circuitBreakerEnabled);
    }
    setOpen(true);
  }, [pause, storedConfig]);

  const handleApply = useCallback(() => {
    if (!storedConfig || !data) return;

    const intervention: Intervention = {
      epoch: currentEpoch,
      governance: {
        taxRate,
        reputationDecay: repDecay,
        circuitBreakerEnabled: cbEnabled,
        circuitBreakerThreshold: storedConfig.governance.circuitBreakerThreshold,
      },
    };

    const newData = runSimulation(storedConfig, undefined, [intervention]);
    recordToLeaderboard(newData);
    loadData(newData);
    setOpen(false);
  }, [storedConfig, data, currentEpoch, taxRate, repDecay, cbEnabled, loadData]);

  // Only show if we have a stored config (sim-generated data)
  if (!storedConfig || !data) return null;

  return (
    <>
      <button
        onClick={handleOpen}
        className="w-9 h-9 flex items-center justify-center rounded bg-btn hover:bg-btn-hover transition-colors"
        title={`Intervene at epoch ${currentEpoch}`}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 1v3M7 10v3M1 7h3M10 7h3" />
          <circle cx="7" cy="7" r="2.5" />
        </svg>
      </button>

      {open && (
        <div className="absolute bottom-full mb-1.5 left-2 right-2 w-auto bg-panel border border-border rounded-lg shadow-2xl p-4 z-[10001] md:left-auto md:right-4 md:w-64">
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-bold">Intervene at Epoch {currentEpoch}/{maxEpoch}</span>
            <button onClick={() => setOpen(false)} className="text-muted hover:text-text text-xs">✕</button>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-1">
              <span className="text-xs text-muted w-16 shrink-0">Tax</span>
              <input
                type="range" min={0} max={0.5} step={0.01}
                value={taxRate}
                onChange={(e) => setTaxRate(parseFloat(e.target.value))}
                className="flex-1 h-1 accent-accent"
              />
              <span className="text-xs font-mono w-8">{taxRate.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-xs text-muted w-16 shrink-0">Rep Decay</span>
              <input
                type="range" min={0.8} max={1.0} step={0.01}
                value={repDecay}
                onChange={(e) => setRepDecay(parseFloat(e.target.value))}
                className="flex-1 h-1 accent-accent"
              />
              <span className="text-xs font-mono w-8">{repDecay.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-xs text-muted w-16 shrink-0">CB</span>
              <button
                onClick={() => setCbEnabled(!cbEnabled)}
                className={`px-2 py-0.5 rounded text-xs font-bold ${cbEnabled ? "bg-accent text-bg" : "bg-btn text-muted"}`}
              >
                {cbEnabled ? "ON" : "OFF"}
              </button>
            </div>
          </div>

          <button
            onClick={handleApply}
            className="w-full mt-3 py-1.5 rounded font-bold text-xs bg-accent text-bg hover:opacity-90 transition-opacity"
          >
            Apply Intervention
          </button>
          <p className="text-xs text-muted mt-1 text-center">
            Re-runs sim with new governance from epoch {currentEpoch}
          </p>
        </div>
      )}
    </>
  );
}
