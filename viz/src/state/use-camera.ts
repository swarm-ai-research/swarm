"use client";

import { useContext, useCallback } from "react";
import { SimContext } from "./simulation-context";
import * as cam from "@/engine/camera";
import { gridToScreen } from "@/engine/isometric";

import type { SimContextValue } from "./simulation-context";
type Viewport = SimContextValue["state"]["viewport"];

export function useCamera() {
  const { state, dispatch, agentPositions } = useContext(SimContext);

  const setViewport = useCallback(
    (viewport: Viewport) => dispatch({ type: "SET_VIEWPORT", viewport }),
    [dispatch],
  );

  const handlePan = useCallback(
    (dx: number, dy: number) => {
      setViewport(cam.pan(state.viewport, dx, dy));
    },
    [state.viewport, setViewport],
  );

  const handleZoomAt = useCallback(
    (targetZoom: number, focusX: number, focusY: number) => {
      setViewport(cam.zoomTo(state.viewport, targetZoom, focusX, focusY));
    },
    [state.viewport, setViewport],
  );

  const handleZoom = useCallback(
    (delta: number, focusX: number, focusY: number) => {
      setViewport(cam.zoom(state.viewport, delta, focusX, focusY));
    },
    [state.viewport, setViewport],
  );

  const resetCamera = useCallback(() => {
    if (agentPositions.current.size === 0) {
      setViewport(cam.centerOn(state.viewport, 0, 0));
      return;
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const pos of agentPositions.current.values()) {
      const screen = gridToScreen(pos.gridX, pos.gridY);
      minX = Math.min(minX, screen.x);
      minY = Math.min(minY, screen.y - 100);
      maxX = Math.max(maxX, screen.x);
      maxY = Math.max(maxY, screen.y + 20);
    }

    setViewport(cam.fitBounds(state.viewport, minX, minY, maxX, maxY));
  }, [state.viewport, setViewport, agentPositions]);

  // Stable resize callback - only dispatches RESIZE action, no viewport dependency
  const resize = useCallback(
    (width: number, height: number) => {
      dispatch({ type: "RESIZE", width, height });
    },
    [dispatch],
  );

  return {
    viewport: state.viewport,
    handlePan,
    handleZoom,
    handleZoomAt,
    resetCamera,
    resize,
  };
}
