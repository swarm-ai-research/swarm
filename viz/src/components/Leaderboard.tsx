"use client";

import React, { useMemo, useState } from "react";
import { useSimulation } from "@/state/use-simulation";
import { useIsMobile } from "@/state/use-media-query";

const STORAGE_KEY = "swarm-leaderboard";
const MAX_ENTRIES = 10;

export interface LeaderboardEntry {
  id: string;
  date: string;
  nAgents: number;
  nEpochs: number;
  seed: number | null;
  finalToxicity: number;
  finalGini: number;
  finalWelfare: number;
  finalAvgP: number;
}

function loadEntries(): LeaderboardEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveEntries(entries: LeaderboardEntry[]): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
}

/** Record current simulation to leaderboard */
export function recordToLeaderboard(data: import("@/data/types").SimulationData): void {
  const last = data.epoch_snapshots[data.epoch_snapshots.length - 1];
  if (!last) return;

  const entry: LeaderboardEntry = {
    id: data.simulation_id,
    date: new Date().toISOString().slice(0, 10),
    nAgents: data.n_agents,
    nEpochs: data.n_epochs,
    seed: data.seed,
    finalToxicity: last.toxicity_rate,
    finalGini: last.gini_coefficient,
    finalWelfare: last.total_welfare,
    finalAvgP: last.avg_p,
  };

  const existing = loadEntries().filter((e) => e.id !== data.simulation_id);
  // Sort by composite score: high welfare + high avgP + low toxicity + low gini
  const scored = [...existing, entry].map((e) => ({
    ...e,
    score: e.finalWelfare * 0.3 + e.finalAvgP * 30 - e.finalToxicity * 20 - e.finalGini * 10,
  }));
  scored.sort((a, b) => b.score - a.score);
  saveEntries(scored.map(({ score: _, ...rest }) => rest));
}

export function Leaderboard() {
  const { data } = useSimulation();
  const isMobile = useIsMobile();
  const [open, setOpen] = useState<boolean | null>(null);

  const entries = useMemo(() => loadEntries(), [data]); // eslint-disable-line react-hooks/exhaustive-deps -- reload when data changes

  if (entries.length === 0) return null;

  // null = user hasn't chosen; default open on desktop, collapsed on phones
  const isOpen = open ?? !isMobile;
  return (
    <div className="absolute z-20 bg-panel border border-border rounded-lg shadow-lg text-xs overflow-y-auto top-9 left-2 w-44 max-h-40 md:top-4 md:left-4 md:w-64 md:max-h-72">
      <button
        onClick={() => setOpen(!isOpen)}
        aria-expanded={isOpen}
        className="w-full px-3 py-2 border-b border-border font-bold flex items-center justify-between text-left"
      >
        Leaderboard
        <span className="text-muted" aria-hidden>{isOpen ? "\u25BE" : "\u25B8"}</span>
      </button>
      {isOpen && (
        <div className="divide-y divide-border">
        {entries.map((e, i) => (
          <div
            key={e.id}
            className={`px-3 py-1.5 flex items-center gap-2 ${data?.simulation_id === e.id ? "bg-accent/10" : ""}`}
          >
            <span className="text-muted w-4 font-mono">{i + 1}</span>
            <div className="flex-1 min-w-0">
              <div className="flex justify-between">
                <span className="text-muted truncate">{e.nAgents}a/{e.nEpochs}e</span>
                <span className="text-muted">{e.date}</span>
              </div>
              <div className="flex gap-2 mt-0.5">
                <span style={{ color: e.finalToxicity < 0.15 ? "#6FCF97" : e.finalToxicity < 0.3 ? "#F2994A" : "#EB5757" }}>
                  T:{(e.finalToxicity * 100).toFixed(0)}%
                </span>
                <span className="text-accent">P:{e.finalAvgP.toFixed(2)}</span>
                <span style={{ color: e.finalGini < 0.3 ? "#6FCF97" : "#F2994A" }}>
                  G:{e.finalGini.toFixed(2)}
                </span>
                <span className="text-info">W:{e.finalWelfare.toFixed(0)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      )}
    </div>
  );
}
