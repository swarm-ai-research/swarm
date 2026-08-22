"use client";

import React, { useCallback } from "react";
import { usePlayback } from "@/state/use-playback";
import { useSimulation } from "@/state/use-simulation";
import { GovernanceIntervention } from "./GovernanceIntervention";

const SPEEDS = [0.5, 1, 2, 4];

function exportSimData(data: import("@/data/types").SimulationData) {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${data.simulation_id || "history"}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function TimelineControls() {
  const {
    play, pause, setEpoch, setStep, setSpeed,
    stepForward, stepBack, toggleStepPlayback,
    playing, speed, currentEpoch, currentStep, maxStepInEpoch, stepPlayback, maxEpoch,
  } = usePlayback();
  const { data } = useSimulation();

  const handleExport = useCallback(() => {
    if (data) exportSimData(data);
  }, [data]);

  return (
    <div className="absolute bottom-0 left-0 right-0 bg-panel border-t border-border flex flex-col z-[10000] pb-[env(safe-area-inset-bottom)]">
      {/* Step scrubber row — only visible in step playback mode */}
      {stepPlayback && (
        <div className="flex items-center gap-2 px-4 pt-2 pb-1">
          {/* Step back */}
          <button
            onClick={stepBack}
            className="w-7 h-7 flex items-center justify-center rounded bg-btn hover:bg-btn-hover transition-colors text-xs"
            title="Step back"
            aria-label="Step back"
          >
            <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
              <rect x="0" y="1" width="2" height="8" />
              <path d="M9 1v8L3 5z" />
            </svg>
          </button>

          {/* Step label */}
          <span className="text-xs text-muted w-28 text-center font-mono whitespace-nowrap">
            Step {currentStep}/{maxStepInEpoch}
          </span>

          {/* Step scrubber */}
          <input
            type="range"
            min={0}
            max={maxStepInEpoch}
            value={currentStep}
            onChange={(e) => setStep(Number(e.target.value))}
            aria-label="Step scrubber"
            className="flex-1 h-1 appearance-none bg-border/60 rounded-full cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2.5
              [&::-webkit-slider-thumb]:h-2.5 [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-secondary [&::-webkit-slider-thumb]:cursor-pointer
              [&::-moz-range-thumb]:w-2.5 [&::-moz-range-thumb]:h-2.5
              [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-secondary
              [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer"
          />

          {/* Step forward */}
          <button
            onClick={stepForward}
            className="w-7 h-7 flex items-center justify-center rounded bg-btn hover:bg-btn-hover transition-colors text-xs"
            title="Step forward"
            aria-label="Step forward"
          >
            <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
              <path d="M1 1v8l6-4z" />
              <rect x="8" y="1" width="2" height="8" />
            </svg>
          </button>
        </div>
      )}

      {/* Main controls row */}
      <div className="min-h-14 flex flex-wrap items-center gap-x-2 gap-y-1.5 px-3 py-2 sm:px-4 md:h-14 md:flex-nowrap md:gap-3 md:py-0">
        {/* Play/Pause */}
        <button
          onClick={() => (playing ? pause() : play())}
          className="w-9 h-9 flex items-center justify-center rounded bg-btn hover:bg-btn-hover transition-colors"
          title={playing ? "Pause" : "Play"}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <rect x="2" y="1" width="4" height="12" rx="1" />
              <rect x="8" y="1" width="4" height="12" rx="1" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <path d="M3 1.5v11l9-5.5z" />
            </svg>
          )}
        </button>

        {/* Epoch label */}
        <span className="text-xs text-muted w-20 text-center font-mono">
          Epoch {currentEpoch}/{maxEpoch}
        </span>

        {/* Epoch scrubber */}
        <input
          type="range"
          min={0}
          max={maxEpoch}
          value={currentEpoch}
          onChange={(e) => setEpoch(Number(e.target.value))}
          aria-label="Epoch scrubber"
          className="flex-1 min-w-28 h-1.5 appearance-none bg-border rounded-full cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:cursor-pointer
            [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-accent
            [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer"
        />

        {/* Speed selector */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-muted mr-1 hidden sm:inline">Speed</span>
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-2.5 py-1.5 text-xs rounded transition-colors md:px-2 md:py-1 ${
                speed === s
                  ? "bg-accent text-bg font-bold"
                  : "bg-btn hover:bg-btn-hover text-muted"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Step playback toggle — only show when event index is available */}
        {data?.rawEvents && data.rawEvents.length > 0 && (
          <button
            onClick={toggleStepPlayback}
            className={`px-2.5 py-1.5 text-xs rounded transition-colors md:px-2 md:py-1 ${
              stepPlayback
                ? "bg-secondary text-bg font-bold"
                : "bg-btn hover:bg-btn-hover text-muted"
            }`}
            title={stepPlayback ? "Switch to epoch-level playback" : "Switch to step-level playback"}
          >
            {stepPlayback ? "Step" : "Epoch"}
          </button>
        )}

        {/* Governance intervention */}
        {data && <GovernanceIntervention />}

        {/* Export button */}
        {data && (
          <button
            onClick={handleExport}
            className="w-9 h-9 flex items-center justify-center rounded bg-btn hover:bg-btn-hover transition-colors"
            title="Export as history.json"
            aria-label="Export as history.json"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M7 1v8M4 6l3 3 3-3M2 11v1.5h10V11" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
