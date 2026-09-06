"use client";

import React, { useState, useCallback, useMemo } from "react";
import type { ScenarioConfig } from "@/engine/sim/types";
import { DEFAULT_CONFIG } from "@/engine/sim/types";
import { AGENT_PROFILES } from "@/engine/sim/agents";
import { useSimWorker } from "@/state/use-sim-worker";
import type { SimulationData } from "@/data/types";
import { DEMO_PRESETS, type DemoPreset } from "@/data/presets";

// ─── Shared UI Components ──────────────────────────────────────────

interface Props {
  onComplete: (data: SimulationData) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted w-28 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 h-1 accent-accent"
      />
      <span className="text-xs font-mono w-12 text-right">{format ? format(value) : value}</span>
    </div>
  );
}

function Section({
  title,
  collapsible,
  children,
}: {
  title: string;
  collapsible?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(!collapsible);
  return (
    <div className="border-t border-border pt-3 mt-3 first:border-0 first:pt-0 first:mt-0">
      <button
        onClick={collapsible ? () => setOpen(!open) : undefined}
        className={`text-xs font-bold uppercase tracking-wider mb-2 flex items-center gap-1 ${
          collapsible ? "cursor-pointer hover:text-accent" : "cursor-default"
        }`}
      >
        {collapsible && <span className="text-muted">{open ? "\u25BC" : "\u25B6"}</span>}
        {title}
      </button>
      {open && <div className="space-y-2">{children}</div>}
    </div>
  );
}

/** Mini sparkline for post-run summary */
function MiniSparkline({ data, color, label, value }: { data: number[]; color: string; label: string; value: string }) {
  if (data.length < 2) return null;
  const w = 80, h = 20;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * (h - 2) - 1}`).join(" ");
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted w-16 shrink-0">{label}</span>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="shrink-0">
        <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" opacity="0.7" />
      </svg>
      <span className="text-xs font-mono" style={{ color }}>{value}</span>
    </div>
  );
}

// ─── Main Panel ────────────────────────────────────────────────────

export function SimConfigPanel({ onComplete }: Props) {
  const [config, setConfig] = useState<ScenarioConfig>({ ...DEFAULT_CONFIG });
  const { status, progress, error, result, run } = useSimWorker();
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [hoveredAgentType, setHoveredAgentType] = useState<string | null>(null);
  const [showSummary, setShowSummary] = useState(false);

  const applyPreset = useCallback((preset: DemoPreset) => {
    setConfig({
      ...DEFAULT_CONFIG,
      ...preset.config,
      governance: { ...DEFAULT_CONFIG.governance, ...preset.config.governance },
      payoff: { ...DEFAULT_CONFIG.payoff, ...preset.config.payoff },
    });
    setActivePreset(preset.name);
    setShowSummary(false);
  }, []);

  const updateAgentCount = useCallback((typeIdx: number, count: number) => {
    setActivePreset(null);
    setShowSummary(false);
    setConfig((prev) => {
      const agents = [...prev.agents];
      agents[typeIdx] = { ...agents[typeIdx], count };
      return { ...prev, agents };
    });
  }, []);

  const updateGov = useCallback(<K extends keyof ScenarioConfig["governance"]>(key: K, val: ScenarioConfig["governance"][K]) => {
    setActivePreset(null);
    setShowSummary(false);
    setConfig((prev) => ({ ...prev, governance: { ...prev.governance, [key]: val } }));
  }, []);

  const updatePayoff = useCallback(<K extends keyof ScenarioConfig["payoff"]>(key: K, val: number) => {
    setActivePreset(null);
    setShowSummary(false);
    setConfig((prev) => ({ ...prev, payoff: { ...prev.payoff, [key]: val } }));
  }, []);

  const handleRun = useCallback(async () => {
    const totalAgents = config.agents.reduce((s, a) => s + a.count, 0);
    if (totalAgents < 2) {
      alert("Need at least 2 agents to run a simulation.");
      return;
    }
    setShowSummary(false);
    const data = await run(config);
    if (data) setShowSummary(true);
  }, [config, run]);

  const handleViewSim = useCallback(() => {
    if (result) onComplete(result);
  }, [result, onComplete]);

  const isRunning = status === "running";

  // Post-run summary data
  const summaryData = useMemo(() => {
    if (!result) return null;
    const epochs = result.epoch_snapshots;
    const last = epochs[epochs.length - 1];
    return {
      toxicity: epochs.map((e) => e.toxicity_rate),
      gini: epochs.map((e) => e.gini_coefficient),
      avgP: epochs.map((e) => e.avg_p),
      welfare: epochs.map((e) => e.total_welfare),
      finalToxicity: last.toxicity_rate,
      finalGini: last.gini_coefficient,
      finalAvgP: last.avg_p,
      finalWelfare: last.total_welfare,
    };
  }, [result]);

  return (
    <div className="space-y-1">
      {/* Presets */}
      <Section title="Presets">
        <div className="grid grid-cols-2 gap-1.5">
          {DEMO_PRESETS.map((p) => (
            <button
              key={p.slug}
              onClick={() => applyPreset(p)}
              title={p.description}
              className={`text-left px-2 py-1.5 rounded text-xs transition-colors ${
                activePreset === p.name
                  ? "bg-accent text-bg font-bold"
                  : "bg-btn hover:bg-btn-hover"
              }`}
            >
              <span className="mr-1">{p.badge}</span>{p.name}
            </button>
          ))}
        </div>
        {activePreset && (
          <p className="text-xs text-muted italic mt-1">
            {DEMO_PRESETS.find((p) => p.name === activePreset)?.description}
          </p>
        )}
      </Section>

      {/* Agents */}
      <Section title="Agents">
        {config.agents.map((ag, i) => (
          <div
            key={ag.type}
            onMouseEnter={() => setHoveredAgentType(ag.type)}
            onMouseLeave={() => setHoveredAgentType(null)}
            onFocus={() => setHoveredAgentType(ag.type)}
            onBlur={() => setHoveredAgentType(null)}
            tabIndex={0}
          >
            <Slider
              label={ag.type.charAt(0).toUpperCase() + ag.type.slice(1)}
              value={ag.count}
              min={0}
              max={10}
              step={1}
              onChange={(v) => updateAgentCount(i, v)}
            />
          </div>
        ))}
        <p className="text-xs text-muted">
          Total: {config.agents.reduce((s, a) => s + a.count, 0)} agents
        </p>
        {/* Agent profile tooltip */}
        {hoveredAgentType && AGENT_PROFILES[hoveredAgentType as keyof typeof AGENT_PROFILES] && (() => {
          const p = AGENT_PROFILES[hoveredAgentType as keyof typeof AGENT_PROFILES];
          return (
            <div className="bg-bg border border-border rounded px-2.5 py-2 text-xs space-y-0.5 mt-1">
              <div className="font-bold text-text mb-1">{hoveredAgentType.charAt(0).toUpperCase() + hoveredAgentType.slice(1)} Profile</div>
              <div className="flex justify-between"><span className="text-muted">Progress</span><span className="font-mono">{p.progressMean > 0 ? "+" : ""}{p.progressMean.toFixed(1)} \u00B1{p.progressStd.toFixed(2)}</span></div>
              <div className="flex justify-between"><span className="text-muted">Rework risk</span><span className="font-mono">{(p.reworkProb * 100).toFixed(0)}%</span></div>
              <div className="flex justify-between"><span className="text-muted">Rejection risk</span><span className="font-mono">{(p.rejectionProb * 100).toFixed(0)}%</span></div>
              <div className="flex justify-between"><span className="text-muted">Engagement</span><span className="font-mono">{p.engagementMean > 0 ? "+" : ""}{p.engagementMean.toFixed(1)}</span></div>
              <div className="flex justify-between"><span className="text-muted">Disposition</span><span className="font-mono" style={{ color: p.disposition > 0.3 ? "#6FCF97" : p.disposition < 0 ? "#EB5757" : "#F2994A" }}>{p.disposition > 0 ? "+" : ""}{p.disposition.toFixed(1)}</span></div>
            </div>
          );
        })()}
      </Section>

      {/* Governance */}
      <Section title="Governance">
        <Slider label="Tax rate" value={config.governance.taxRate} min={0} max={0.5} step={0.01} onChange={(v) => updateGov("taxRate", v)} format={(v) => v.toFixed(2)} />
        <Slider label="Rep. decay" value={config.governance.reputationDecay} min={0.8} max={1.0} step={0.01} onChange={(v) => updateGov("reputationDecay", v)} format={(v) => v.toFixed(2)} />
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted w-28 shrink-0">Circuit breaker</span>
          <button
            onClick={() => updateGov("circuitBreakerEnabled", !config.governance.circuitBreakerEnabled)}
            className={`px-2 py-0.5 rounded text-xs ${config.governance.circuitBreakerEnabled ? "bg-accent text-bg" : "bg-btn"}`}
          >
            {config.governance.circuitBreakerEnabled ? "ON" : "OFF"}
          </button>
          {config.governance.circuitBreakerEnabled && (
            <span className="text-xs text-muted">@ {config.governance.circuitBreakerThreshold}</span>
          )}
        </div>
      </Section>

      {/* Payoff (collapsible) */}
      <Section title="Payoff Parameters" collapsible>
        <Slider label="s_plus" value={config.payoff.s_plus} min={0} max={5} step={0.1} onChange={(v) => updatePayoff("s_plus", v)} format={(v) => v.toFixed(1)} />
        <Slider label="s_minus" value={config.payoff.s_minus} min={0} max={5} step={0.1} onChange={(v) => updatePayoff("s_minus", v)} format={(v) => v.toFixed(1)} />
        <Slider label="h (harm)" value={config.payoff.h} min={0} max={5} step={0.1} onChange={(v) => updatePayoff("h", v)} format={(v) => v.toFixed(1)} />
        <Slider label="theta" value={config.payoff.theta} min={0} max={1} step={0.05} onChange={(v) => updatePayoff("theta", v)} format={(v) => v.toFixed(2)} />
        <Slider label="rho_a" value={config.payoff.rho_a} min={0} max={1} step={0.05} onChange={(v) => updatePayoff("rho_a", v)} format={(v) => v.toFixed(2)} />
        <Slider label="rho_b" value={config.payoff.rho_b} min={0} max={1} step={0.05} onChange={(v) => updatePayoff("rho_b", v)} format={(v) => v.toFixed(2)} />
      </Section>

      {/* Simulation (collapsible) */}
      <Section title="Simulation" collapsible>
        <Slider label="Epochs" value={config.epochs} min={5} max={100} step={5} onChange={(v) => setConfig((p) => ({ ...p, epochs: v }))} />
        <Slider label="Steps/epoch" value={config.stepsPerEpoch} min={5} max={50} step={5} onChange={(v) => setConfig((p) => ({ ...p, stepsPerEpoch: v }))} />
        <Slider label="Seed" value={config.seed} min={1} max={9999} step={1} onChange={(v) => setConfig((p) => ({ ...p, seed: v }))} />
      </Section>

      {/* Run button + progress */}
      <div className="pt-3 mt-3 border-t border-border">
        {showSummary && summaryData ? (
          <>
            {/* Post-run summary */}
            <div className="space-y-1.5 mb-3">
              <MiniSparkline data={summaryData.toxicity} color={summaryData.finalToxicity > 0.3 ? "#EB5757" : "#6FCF97"} label="Toxicity" value={(summaryData.finalToxicity * 100).toFixed(0) + "%"} />
              <MiniSparkline data={summaryData.avgP} color="#3ECFB4" label="Avg P" value={summaryData.finalAvgP.toFixed(3)} />
              <MiniSparkline data={summaryData.gini} color={summaryData.finalGini > 0.4 ? "#EB5757" : "#F2994A"} label="Gini" value={summaryData.finalGini.toFixed(3)} />
              <MiniSparkline data={summaryData.welfare} color="#5698FF" label="Welfare" value={summaryData.finalWelfare.toFixed(1)} />
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleViewSim}
                className="flex-1 py-2 rounded font-bold text-sm bg-accent text-bg hover:opacity-90 transition-opacity"
              >
                View in City
              </button>
              <button
                onClick={handleRun}
                className="px-3 py-2 rounded font-bold text-sm bg-btn hover:bg-btn-hover transition-colors"
              >
                Re-run
              </button>
            </div>
          </>
        ) : (
          <>
            <button
              onClick={handleRun}
              disabled={isRunning}
              className="w-full py-2 rounded font-bold text-sm bg-accent text-bg hover:opacity-90 disabled:opacity-50 transition-opacity"
            >
              {isRunning ? "Running..." : "Run Simulation"}
            </button>
            {isRunning && (
              <div className="mt-2 h-1.5 bg-btn rounded overflow-hidden">
                <div
                  className="h-full bg-accent transition-all duration-200"
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
            )}
          </>
        )}
        {error && <p className="text-xs text-red-400 mt-2">{error}</p>}
      </div>
    </div>
  );
}
