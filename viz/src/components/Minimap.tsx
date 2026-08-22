"use client";

import React, { useRef, useEffect, useMemo } from "react";
import { useSimulation } from "@/state/use-simulation";
import { gridToScreen } from "@/engine/isometric";
import { useCamera } from "@/state/use-camera";
import { useIsMobile } from "@/state/use-media-query";
import { AGENT_COLORS, COLORS } from "@/engine/constants";
import { rgba } from "@/utils/color";

const MINIMAP_SIZE_DESKTOP = 140;
const MINIMAP_SIZE_MOBILE = 96;

export function Minimap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { agents, overlays, viewport } = useSimulation();
  const { handlePan } = useCamera();
  const isMobile = useIsMobile();
  const MINIMAP_SIZE = isMobile ? MINIMAP_SIZE_MOBILE : MINIMAP_SIZE_DESKTOP;

  // Shared world→minimap mapping used by both drawing and click-to-navigate
  const mapping = useMemo(() => {
    let minX = 0, minY = 0, maxX = 100, maxY = 100;
    for (const agent of agents) {
      const s = gridToScreen(agent.gridX, agent.gridY);
      minX = Math.min(minX, s.x - 40);
      minY = Math.min(minY, s.y - 100);
      maxX = Math.max(maxX, s.x + 40);
      maxY = Math.max(maxY, s.y + 40);
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const scale = Math.min(MINIMAP_SIZE / rangeX, MINIMAP_SIZE / rangeY) * 0.85;
    return {
      minX, minY, scale,
      offX: (MINIMAP_SIZE - rangeX * scale) / 2,
      offY: (MINIMAP_SIZE - rangeY * scale) / 2,
    };
  }, [agents, MINIMAP_SIZE]);

  const handleMapClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const worldX = (mx - mapping.offX) / mapping.scale + mapping.minX;
    const worldY = (my - mapping.offY) / mapping.scale + mapping.minY;
    // Pan the camera so the tapped world point sits at viewport center
    handlePan(
      viewport.width / 2 - (worldX * viewport.zoom + viewport.x),
      viewport.height / 2 - (worldY * viewport.zoom + viewport.y),
    );
  };

  useEffect(() => {
    if (!overlays.minimap) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = devicePixelRatio;
    canvas.width = MINIMAP_SIZE * dpr;
    canvas.height = MINIMAP_SIZE * dpr;
    ctx.scale(dpr, dpr);
    const { minX, minY, scale, offX, offY } = mapping;

    // Background
    ctx.fillStyle = rgba(COLORS.bg, 0.9);
    ctx.fillRect(0, 0, MINIMAP_SIZE, MINIMAP_SIZE);

    // Agents as dots
    for (const agent of agents) {
      const s = gridToScreen(agent.gridX, agent.gridY);
      const mx = (s.x - minX) * scale + offX;
      const my = (s.y - minY) * scale + offY;
      ctx.fillStyle = AGENT_COLORS[agent.agentType].secondary;
      ctx.beginPath();
      ctx.arc(mx, my, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Viewport rectangle
    const vpLeft = (-viewport.x / viewport.zoom - minX) * scale + offX;
    const vpTop = (-viewport.y / viewport.zoom - minY) * scale + offY;
    const vpWidth = (viewport.width / viewport.zoom) * scale;
    const vpHeight = (viewport.height / viewport.zoom) * scale;
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 1;
    ctx.strokeRect(vpLeft, vpTop, vpWidth, vpHeight);

    // Border
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, MINIMAP_SIZE, MINIMAP_SIZE);
  }, [agents, overlays.minimap, viewport, isMobile, mapping]);

  if (!overlays.minimap) return null;

  return (
    <canvas
      ref={canvasRef}
      onClick={handleMapClick}
      aria-label="Minimap: click to center the view on a point"
      className={`absolute rounded border border-border z-20 cursor-pointer ${
        isMobile ? "bottom-[13.5rem] right-2" : "bottom-16 right-4"
      }`}
      style={{ width: MINIMAP_SIZE, height: MINIMAP_SIZE }}
    />
  );
}
