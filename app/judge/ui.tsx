"use client";

import type { ReactNode } from "react";

// Small shared pieces for the Judge Lab. Colors are the validated dataviz steps for the
// dark card surface (#111113): accent #3987e5, emphasis #d95926 — see judge-lab.tsx.

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cx("rounded-2xl border border-white/[0.08] bg-[var(--jl-surface)]", className)}>
      {children}
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex min-w-[1.4rem] h-[1.4rem] items-center justify-center rounded-md border border-white/10 bg-white/[0.04] px-1.5 font-sans text-[11px] text-zinc-400">
      {children}
    </kbd>
  );
}

export function Segmented<T extends string>({
  value, options, onChange, label,
}: {
  value: T;
  options: { value: T; label: string; hint?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="inline-flex rounded-full border border-white/[0.08] bg-white/[0.03] p-1">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-full px-4 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40",
            value === o.value ? "bg-white text-black font-medium" : "text-zinc-400 hover:text-white",
          )}
        >
          {o.label}
          {o.hint && <span className={cx("ml-1.5 tabular-nums", value === o.value ? "text-black/50" : "text-zinc-600")}>{o.hint}</span>}
        </button>
      ))}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cx("inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white", className)}
    />
  );
}

export function PlayIcon({ playing, className }: { playing: boolean; className?: string }) {
  return playing ? (
    <svg viewBox="0 0 16 16" className={className} aria-hidden>
      <rect x="3.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" />
      <rect x="9.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" className={className} aria-hidden>
      <path d="M4.5 2.9v10.2a.9.9 0 0 0 1.36.77l8.1-5.1a.9.9 0 0 0 0-1.54l-8.1-5.1A.9.9 0 0 0 4.5 2.9Z" fill="currentColor" />
    </svg>
  );
}

/** 95% Wilson interval for k agreements out of n — honest at small n, unlike ±2·SE. */
export function wilson(k: number, n: number): [number, number] | null {
  if (!n) return null;
  const z = 1.96;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

export const pct = (x: number) => `${Math.round(x * 100)}%`;

export function formatTime(s: number) {
  if (!Number.isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

export function DownloadIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={className} aria-hidden>
      <path d="M8 2.5v8m0 0l-3.25-3.25M8 10.5l3.25-3.25M3 13.5h10" fill="none" stroke="currentColor"
        strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/** Download name from the prompt's first words — never says which model made the take. */
export function songFileName(prompt: string, label: string) {
  const words = prompt.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().split(" ").filter(Boolean).slice(0, 6);
  return `${words.join("-") || "song"}-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.mp3`;
}

export function EyeOffIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={className} aria-hidden>
      <path d="M2.5 8s2-4 5.5-4 5.5 4 5.5 4-2 4-5.5 4S2.5 8 2.5 8Z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <circle cx="8" cy="8" r="1.8" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M3 13L13 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}
