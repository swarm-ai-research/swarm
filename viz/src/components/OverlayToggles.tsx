"use client";

import React from "react";
import { useSimulation } from "@/state/use-simulation";
import type { OverlayState } from "@/engine/types";

const TOGGLE_ITEMS: { key: keyof OverlayState; label: string }[] = [
  { key: "interactions", label: "Arcs" },
  { key: "metricsHud", label: "HUD" },
  { key: "particles", label: "FX" },
  { key: "minimap", label: "Map" },
  { key: "collusionLines", label: "Collusion" },
  { key: "threatZones", label: "Threats" },
  { key: "digitalRain", label: "Rain" },
  { key: "tierraStrip", label: "Memory" },
  { key: "networkWeb", label: "Network" },
];

export function OverlayToggles() {
  const { overlays, toggleOverlay, data } = useSimulation();

  if (!data) return null;

  return (
    <div className="absolute z-20 flex flex-wrap gap-1 bottom-[8.75rem] left-2 right-[7.75rem] md:bottom-24 md:left-4 md:right-auto md:max-w-xs">
      {TOGGLE_ITEMS.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => toggleOverlay(key)}
          className={`px-3 py-1.5 text-xs rounded-full border transition-colors md:px-2.5 md:py-1 md:text-[10px] ${
            overlays[key]
              ? "bg-accent/20 border-accent/50 text-accent"
              : "bg-btn border-border text-muted hover:text-text"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
