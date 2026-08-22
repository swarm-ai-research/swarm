"use client";

import React, { useRef, useEffect, useCallback } from "react";
import { useSimulation } from "@/state/use-simulation";
import { useCamera } from "@/state/use-camera";
import { render } from "@/engine/renderer";
import { screenToWorld } from "@/engine/camera";
import { screenToGrid } from "@/engine/isometric";
import { getCharacterBounds } from "@/engine/entities/agent-character";

export function IsometricCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const {
    agents,
    arcs,
    viewport,
    hoveredAgent,
    selectedAgent,
    currentEpochSnap,
    environment,
    overlays,
    particles,
    gridSize,
    setHovered,
    setSelected,
    codeTrailSystem,
    digitalRainRef,
    recompileStateRef,
  } = useSimulation();
  const { handlePan, handleZoom, handleZoomAt, resetCamera, resize } = useCamera();

  // Unified pointer tracking: 1 finger/mouse pans, 2 fingers pinch-zoom, tap selects.
  const activePointers = useRef(new Map<number, { x: number; y: number }>());
  const lastPan = useRef({ x: 0, y: 0 });
  const pinchState = useRef<{ startDist: number; startZoom: number } | null>(null);
  const pressStart = useRef<{ x: number; y: number } | null>(null);
  const lastTap = useRef<{ time: number; x: number; y: number } | null>(null);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        canvas.width = width * devicePixelRatio;
        canvas.height = height * devicePixelRatio;
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";
        resize(width, height);
      }
    });
    obs.observe(parent);
    return () => obs.disconnect();
  }, [resize]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.scale(devicePixelRatio, devicePixelRatio);

    render(ctx, {
      agents,
      arcs,
      viewport,
      hoveredAgent,
      selectedAgent,
      epoch: currentEpochSnap,
      environment,
      overlays,
      particles,
      gridSize,
      digitalRain: digitalRainRef.current,
      codeTrails: codeTrailSystem.current.particles,
      recompileState: recompileStateRef.current,
    });

    ctx.restore();
  });

  // Hit testing for hover/click
  const hitTest = useCallback(
    (clientX: number, clientY: number): string | null => {
      const world = screenToWorld(viewport, clientX, clientY);
      // Check agents in reverse depth order (front to back)
      const sorted = [...agents].sort(
        (a, b) => b.gridX + b.gridY - (a.gridX + a.gridY),
      );
      for (const agent of sorted) {
        const bounds = getCharacterBounds(agent);
        if (
          world.x >= bounds.minX &&
          world.x <= bounds.maxX &&
          world.y >= bounds.minY &&
          world.y <= bounds.maxY
        ) {
          return agent.id;
        }
      }
      return null;
    },
    [agents, viewport],
  );

  const pointerPos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    return rect ? { x: e.clientX - rect.left, y: e.clientY - rect.top } : null;
  };

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      // Capture can throw if the pointer is already released (e.g. synthetic events)
      try {
        canvasRef.current?.setPointerCapture(e.pointerId);
      } catch {
        /* pointer already gone; tracking map still works */
      }
      const p = pointerPos(e);
      if (!p) return;
      activePointers.current.set(e.pointerId, p);
      if (activePointers.current.size === 1) {
        lastPan.current = p;
        pressStart.current = p;
      } else if (activePointers.current.size === 2) {
        // Second finger down: switch from pan to pinch, anchored at current zoom
        const [a, b] = [...activePointers.current.values()];
        pinchState.current = {
          startDist: Math.hypot(a.x - b.x, a.y - b.y),
          startZoom: viewport.zoom,
        };
        pressStart.current = null;
      }
    },
    [viewport.zoom],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const p = pointerPos(e);
      if (!p) return;

      if (!activePointers.current.has(e.pointerId)) {
        // Hover preview (mouse only, nothing pressed)
        if (e.pointerType === "mouse" && e.buttons === 0) setHovered(hitTest(p.x, p.y));
        return;
      }
      activePointers.current.set(e.pointerId, p);

      if (activePointers.current.size >= 2 && pinchState.current) {
        const [a, b] = [...activePointers.current.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchState.current.startDist > 0 && dist > 0) {
          handleZoomAt(
            (pinchState.current.startZoom * dist) / pinchState.current.startDist,
            (a.x + b.x) / 2,
            (a.y + b.y) / 2,
          );
        }
      } else {
        handlePan(p.x - lastPan.current.x, p.y - lastPan.current.y);
        lastPan.current = p;
      }
    },
    [handlePan, handleZoomAt, hitTest, setHovered],
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const hadPointer = activePointers.current.delete(e.pointerId);
      try {
        canvasRef.current?.releasePointerCapture(e.pointerId);
      } catch {
        /* nothing captured */
      }

      if (activePointers.current.size < 2) pinchState.current = null;
      if (activePointers.current.size === 1) {
        // Pinch ended with one finger still down: resume panning from it
        const [p] = [...activePointers.current.values()];
        lastPan.current = p;
      }

      // Tap-to-select: single pointer released with minimal movement.
      // Double-tap resets the camera (iOS Safari never fires dblclick for touches).
      const start = pressStart.current;
      pressStart.current = null;
      if (hadPointer && start && activePointers.current.size === 0) {
        const end = pointerPos(e);
        if (end && Math.hypot(end.x - start.x, end.y - start.y) < 6) {
          const now = performance.now();
          const prev = lastTap.current;
          lastTap.current = { time: now, x: end.x, y: end.y };
          if (
            e.pointerType !== "mouse" &&
            prev &&
            now - prev.time < 350 &&
            Math.hypot(end.x - prev.x, end.y - prev.y) < 30
          ) {
            lastTap.current = null;
            resetCamera();
            return;
          }
          setSelected(hitTest(end.x, end.y));
        }
      }
    },
    [hitTest, resetCamera, setSelected],
  );

  const handlePointerCancel = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      activePointers.current.delete(e.pointerId);
      try {
        canvasRef.current?.releasePointerCapture(e.pointerId);
      } catch {
        /* nothing captured */
      }
      pinchState.current = null;
      pressStart.current = null;
    },
    [],
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      handleZoom(e.deltaY, e.clientX - rect.left, e.clientY - rect.top);
    },
    [handleZoom],
  );

  const handleDoubleClick = useCallback(() => {
    resetCamera();
  }, [resetCamera]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 cursor-grab touch-none select-none active:cursor-grabbing"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerCancel}
      onPointerLeave={() => {
        if (activePointers.current.size === 0) setHovered(null);
      }}
      onWheel={handleWheel}
      onDoubleClick={handleDoubleClick}
      onContextMenu={(e) => e.preventDefault()}
    />
  );
}
