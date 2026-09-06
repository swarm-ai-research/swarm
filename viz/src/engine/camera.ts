import { MIN_ZOOM, MAX_ZOOM } from "./constants";
import { clamp } from "@/utils/math";
import type { Viewport, Point } from "./types";

export function createViewport(width: number, height: number): Viewport {
  return { x: 0, y: 0, width, height, zoom: 1 };
}

export function pan(vp: Viewport, dx: number, dy: number): Viewport {
  return { ...vp, x: vp.x + dx, y: vp.y + dy };
}

export function zoom(vp: Viewport, delta: number, focusX: number, focusY: number): Viewport {
  const newZoom = clamp(vp.zoom * (1 - delta * 0.001), MIN_ZOOM, MAX_ZOOM);
  const ratio = newZoom / vp.zoom;
  return {
    ...vp,
    zoom: newZoom,
    x: focusX - (focusX - vp.x) * ratio,
    y: focusY - (focusY - vp.y) * ratio,
  };
}

/** Zoom to an absolute zoom level around a focus point (used by pinch gestures). */
export function zoomTo(vp: Viewport, targetZoom: number, focusX: number, focusY: number): Viewport {
  const newZoom = clamp(targetZoom, MIN_ZOOM, MAX_ZOOM);
  const ratio = newZoom / vp.zoom;
  return {
    ...vp,
    zoom: newZoom,
    x: focusX - (focusX - vp.x) * ratio,
    y: focusY - (focusY - vp.y) * ratio,
  };
}

/** Convert screen mouse coordinates to world coordinates */
export function screenToWorld(vp: Viewport, screenX: number, screenY: number): Point {
  return {
    x: (screenX - vp.x) / vp.zoom,
    y: (screenY - vp.y) / vp.zoom,
  };
}

/** Convert world coordinates to screen coordinates */
export function worldToScreen(vp: Viewport, worldX: number, worldY: number): Point {
  return {
    x: worldX * vp.zoom + vp.x,
    y: worldY * vp.zoom + vp.y,
  };
}

/** Center viewport on a point */
export function centerOn(vp: Viewport, worldX: number, worldY: number): Viewport {
  return {
    ...vp,
    x: vp.width / 2 - worldX * vp.zoom,
    y: vp.height / 2 - worldY * vp.zoom,
  };
}

/** Fit all agents in view */
export function fitBounds(
  vp: Viewport,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  padding = 80,
): Viewport {
  const contentW = maxX - minX + padding * 2;
  const contentH = maxY - minY + padding * 2;
  const zoomX = vp.width / contentW;
  const zoomY = vp.height / contentH;
  const newZoom = clamp(Math.min(zoomX, zoomY), MIN_ZOOM, MAX_ZOOM);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return {
    ...vp,
    zoom: newZoom,
    x: vp.width / 2 - cx * newZoom,
    y: vp.height / 2 - cy * newZoom,
  };
}
