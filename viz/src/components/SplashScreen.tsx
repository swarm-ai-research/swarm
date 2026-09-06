"use client";

import React, { useState, useEffect, useCallback } from "react";

export function SplashScreen() {
  const [phase, setPhase] = useState<"enter" | "exit" | "done">("enter");

  const skip = useCallback(() => {
    if (phase !== "done") setPhase("exit");
  }, [phase]);

  useEffect(() => {
    // After enter + glow hold (2.5s), start exit
    const timer = setTimeout(() => {
      setPhase((p) => (p === "enter" ? "exit" : p));
    }, 2500);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (phase === "exit") {
      const timer = setTimeout(() => setPhase("done"), 700);
      return () => clearTimeout(timer);
    }
  }, [phase]);

  if (phase === "done") return null;

  return (
    <div
      className={`absolute inset-0 z-50 flex flex-col items-center justify-center bg-bg cursor-pointer ${
        phase === "exit" ? "animate-[splash-exit_0.7s_ease-in_forwards]" : ""
      }`}
      onClick={skip}
    >
      <img
        src={`${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/splash.webp`}
        alt="SWARM"
        className={`w-44 h-44 object-contain sm:w-64 sm:h-64 ${
          phase === "enter"
            ? "animate-[splash-enter_0.8s_ease-out_forwards,splash-glow_1.7s_ease-in-out_0.8s_infinite]"
            : ""
        }`}
      />
      <p className="text-muted text-sm mt-4 animate-[splash-enter_1.2s_ease-out_forwards] opacity-0" style={{ animationDelay: "0.6s" }}>
        Governance benchmark suite with receipts
      </p>
      <p className="text-muted/40 text-xs mt-6 animate-[splash-enter_1.4s_ease-out_forwards] opacity-0" style={{ animationDelay: "1.2s" }}>
        click or tap to continue
      </p>
    </div>
  );
}
