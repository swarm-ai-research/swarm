"use client";

import React, { useState, useCallback } from "react";
import { SimulationProvider } from "@/state/simulation-context";
import { GameProvider, useGame } from "@/state/game-context";
import { useSimulation } from "@/state/use-simulation";
import { IsometricCanvas } from "./IsometricCanvas";
import { TimelineControls } from "./TimelineControls";
import { LiveControlBar } from "./LiveControlBar";
import { InfoPanel } from "./InfoPanel";
import { MetricsOverlay } from "./MetricsOverlay";
import { Minimap } from "./Minimap";
import { DataLoader } from "./DataLoader";
import { OverlayToggles } from "./OverlayToggles";
import { SplashScreen } from "./SplashScreen";
import { NarrativeOverlay } from "./NarrativeOverlay";
import { ToastContainer } from "./Toast";
import { LevelEndOverlay } from "./CampaignPanel";
import { LiveTickDriver } from "./LiveTickDriver";
import dynamic from "next/dynamic";
const Leaderboard = dynamic(() => import("./Leaderboard").then((m) => m.Leaderboard), { ssr: false });
import { EventFeed } from "./EventFeed";
import { useShareUrl } from "@/state/use-url-state";
import { DEFAULT_CONFIG } from "@/engine/sim/types";

// ─── Share Button ──────────────────────────────────────────────────

function ShareButton() {
  const { data } = useSimulation();
  const [copied, setCopied] = useState(false);
  const buildShareUrl = useShareUrl();

  const handleShare = useCallback(() => {
    if (!data || typeof window === "undefined") return;

    const config = {
      ...DEFAULT_CONFIG,
      seed: data.seed ?? DEFAULT_CONFIG.seed,
      epochs: data.n_epochs ?? DEFAULT_CONFIG.epochs,
      stepsPerEpoch: data.steps_per_epoch ?? DEFAULT_CONFIG.stepsPerEpoch,
    };
    const url = buildShareUrl(config) + "&autorun=1";

    navigator.clipboard.writeText(url).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [data, buildShareUrl]);

  if (!data) return null;

  return (
    <button
      onClick={handleShare}
      className="relative z-[10001] px-2.5 py-1.5 text-xs rounded bg-btn hover:bg-btn-hover text-muted hover:text-text transition-colors"
      title="Copy shareable URL"
    >
      {copied ? "\u2713 Copied!" : "Share"}
    </button>
  );
}

// ─── Main Viewer ───────────────────────────────────────────────────

function SimulationViewerInner() {
  const { state: gameState } = useGame();
  const { data } = useSimulation();

  return (
    <div className="relative w-screen h-[100dvh] overflow-hidden bg-bg">
      {/* Live tick driver (invisible, just runs the hook) */}
      {gameState.isLive && <LiveTickDriver />}

      {/* Canvas layer */}
      <div className="absolute inset-0">
        <IsometricCanvas />
      </div>

      {/* UI overlays */}
      <DataLoader />
      <InfoPanel />
      <MetricsOverlay />
      <NarrativeOverlay />
      <Minimap />
      <Leaderboard />
      <OverlayToggles />
      {/* Top-right chip cluster: Share + Events toggle (flex, so badge growth pushes Share left) */}
      <div className="absolute top-2 right-2 z-[10001] flex items-center gap-1.5">
        <ShareButton />
        <EventFeed />
      </div>
      <ToastContainer />
      <LevelEndOverlay />

      {/* Show LiveControlBar in live mode, TimelineControls in replay mode.
          Replay bar stays hidden until data exists (no empty "Epoch 0/0" bar behind the start menu). */}
      {gameState.isLive ? <LiveControlBar /> : data ? <TimelineControls /> : null}

      <SplashScreen />
    </div>
  );
}

export function SimulationViewer() {
  return (
    <SimulationProvider>
      <GameProvider>
        <SimulationViewerInner />
      </GameProvider>
    </SimulationProvider>
  );
}
