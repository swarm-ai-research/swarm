"use client";

import React, { useMemo } from "react";
import { useSimulation } from "@/state/use-simulation";
import { formatNumber, formatPercent } from "@/utils/format";
import { pToHealthColor } from "@/utils/color";

export function MetricsOverlay() {
  const { data, currentEpochSnap, overlays, currentEpoch } = useSimulation();

  // useMemo must be called unconditionally (React hooks rules)
  const sparklineData = useMemo(() => {
    if (!data) return { toxicity: [], avgP: [], gini: [], welfare: [] };
    const epochs = data.epoch_snapshots.slice(0, currentEpoch + 1);
    return {
      toxicity: epochs.map((e) => e.toxicity_rate),
      avgP: epochs.map((e) => e.avg_p),
      gini: epochs.map((e) => e.gini_coefficient),
      welfare: epochs.map((e) => e.total_welfare),
    };
  }, [data, currentEpoch]);

  if (!overlays.metricsHud || !currentEpochSnap || !data) return null;

  const e = currentEpochSnap;

  return (
    <div className="absolute z-20 bg-panel border border-border rounded-lg shadow-lg text-xs top-9 right-2 w-44 md:top-4 md:right-4 md:w-56">
      <div className="px-3 py-2 border-b border-border">
        <span className="text-muted">Epoch {e.epoch}</span>
        <span className="text-muted float-right">
          {e.n_agents} agents
        </span>
      </div>

      <div className="p-3 space-y-2.5">
        <MetricRow
          label="Toxicity"
          value={formatPercent(e.toxicity_rate)}
          color={e.toxicity_rate > 0.3 ? "#EB5757" : e.toxicity_rate > 0.15 ? "#F2994A" : "#6FCF97"}
          sparkline={sparklineData.toxicity}
        />
        <MetricRow
          label="Avg P"
          value={formatNumber(e.avg_p, 3)}
          color={pToHealthColor(e.avg_p)}
          sparkline={sparklineData.avgP}
        />
        <MetricRow
          label="Quality Gap"
          value={formatNumber(e.quality_gap, 3)}
          color={e.quality_gap < 0 ? "#EB5757" : "#6FCF97"}
        />
        <MetricRow
          label="Gini"
          value={formatNumber(e.gini_coefficient, 3)}
          color={e.gini_coefficient > 0.4 ? "#EB5757" : "#F2994A"}
          sparkline={sparklineData.gini}
        />
        <MetricRow
          label="Total Welfare"
          value={formatNumber(e.total_welfare, 1)}
          sparkline={sparklineData.welfare}
        />

        <div className="border-t border-border pt-2 mt-2">
          <div className="flex justify-between text-muted">
            <span>Interactions</span>
            <span className="font-mono">{e.total_interactions}</span>
          </div>
          <div className="flex justify-between text-muted">
            <span>Accepted</span>
            <span className="font-mono text-accent">
              {e.accepted_interactions}
            </span>
          </div>
          <div className="flex justify-between text-muted">
            <span>Rejected</span>
            <span className="font-mono text-alert">
              {e.rejected_interactions}
            </span>
          </div>
        </div>

        {(e.ecosystem_threat_level ?? 0) > 0 && (
          <div className="border-t border-border pt-2">
            <div className="flex justify-between text-muted">
              <span>Threat Level</span>
              <span
                className="font-mono"
                style={{
                  color:
                    (e.ecosystem_threat_level ?? 0) > 0.5
                      ? "#EB5757"
                      : "#F2994A",
                }}
              >
                {formatPercent(e.ecosystem_threat_level ?? 0)}
              </span>
            </div>
          </div>
        )}

        {e.n_frozen > 0 && (
          <div className="flex justify-between text-muted">
            <span>Frozen</span>
            <span className="font-mono" style={{ color: "#A8CFF5" }}>
              {e.n_frozen}
            </span>
          </div>
        )}
        {e.n_quarantined > 0 && (
          <div className="flex justify-between text-muted">
            <span>Quarantined</span>
            <span className="font-mono text-alert">{e.n_quarantined}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricRow({
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
    <div>
      <div className="flex justify-between items-center">
        <span className="text-muted">{label}</span>
        <span className="font-mono" style={color ? { color } : undefined}>
          {value}
        </span>
      </div>
      {sparkline && sparkline.length > 1 && (
        <Sparkline data={sparkline} color={color ?? "#3ECFB4"} />
      )}
    </div>
  );
}

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const width = 200;
  const height = 16;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 2) - 1;
    return `${x},${y}`;
  });

  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="mt-0.5"
    >
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.6"
      />
    </svg>
  );
}
