"use client";

import { useSyncExternalStore } from "react";

/**
 * SSR-safe media query subscription. Returns false on the server and during
 * hydration, then flips to the real value after mount.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = (onChange: () => void) => {
    const mql = window.matchMedia(query);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  };

  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/** Viewports below the md breakpoint get the compact single-column overlay layout. */
export function useIsMobile(): boolean {
  return useMediaQuery("(max-width: 767px)");
}
