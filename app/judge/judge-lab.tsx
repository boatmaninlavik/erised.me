"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import type { SetSummary } from "@/lib/judge-lab-types";
import { BlindTest } from "./blind-test";
import { ResultsView } from "./results";
import { Segmented, Spinner } from "./ui";

// Model colors on the #111113 surface: the model being tested is blue (#3987e5), the one it's
// compared against is light gray (#a1a1aa) — they differ in hue and lightness, so colorblind-safe.
const THEME = {
  "--jl-surface": "#111113",
  "--jl-accent": "#3987e5",
  "--jl-older": "#a1a1aa",
} as React.CSSProperties;

type View = "test" | "results";

export function JudgeLab() {
  const [sets, setSets] = useState<SetSummary[] | null>(null);
  const [setId, setSetId] = useState("");
  const [view, setView] = useState<View>("test");
  const [unlocked, setUnlocked] = useState<Record<string, boolean>>({});
  const [error, setError] = useState(false);

  const refresh = useCallback(() => {
    fetch("/api/judge-lab/sets")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((s: SetSummary[]) => {
        setSets(s);
        setSetId((cur) => cur || s[0]?.id || "");
      })
      .catch(() => setError(true));
  }, []);
  useEffect(refresh, [refresh]);

  const set = sets?.find((s) => s.id === setId) ?? null;
  const finished = !!set && set.rated >= set.n;
  const canSeeResults = finished || !!unlocked[setId];

  const unlockEarly = () => {
    if (!set) return;
    const left = set.n - set.rated;
    if (window.confirm(`${left} pair${left === 1 ? "" : "s"} still unrated. Knowing which take is which can sway the rest of your picks. Show results anyway?`)) {
      setUnlocked((u) => ({ ...u, [setId]: true }));
      setView("results");
    }
  };

  return (
    <div style={THEME} className="min-h-screen">
      <header className="border-b border-white/[0.06]">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <Link href="/" className="text-xl font-semibold tracking-tighter text-white">Erised</Link>
            <span className="text-zinc-700">/</span>
            <span className="text-sm text-zinc-300">Blind A/B</span>
          </div>
          {set && (
            <span className="text-sm tabular-nums text-zinc-500">
              <span className="text-white">{set.rated}</span> of {set.n} rated
            </span>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 pb-36 pt-10">
        {error ? (
          <p className="text-sm text-red-400">
            Couldn&apos;t reach the API. Check that .env.local has JUDGE_LAB=1, GOOGLE_APPLICATION_CREDENTIALS,
            B2_KEY_ID and B2_APP_KEY, then restart <code className="text-zinc-300">npm run dev</code>.
          </p>
        ) : !sets ? (
          <div className="flex items-center gap-3 py-16 text-zinc-500"><Spinner /> Loading from your bucket…</div>
        ) : !set ? (
          <p className="text-sm text-zinc-400">No blind A/B sets published yet.</p>
        ) : (
          <>
            <div className="flex flex-wrap items-end justify-between gap-6">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight">{set.title}</h1>
                <p className="mt-3 max-w-2xl text-zinc-400">{set.blurb}</p>
              </div>
              <Segmented<View>
                label="View"
                value={view}
                onChange={(v) => (v === "results" && !canSeeResults ? unlockEarly() : setView(v))}
                options={[
                  { value: "test", label: "Listen" },
                  { value: "results", label: canSeeResults ? "Results" : "Results (locked)" },
                ]}
              />
            </div>

            {sets.length > 1 && (
              <div className="mt-6">
                <Segmented<string>
                  label="Pair set"
                  value={setId}
                  onChange={(id) => { setSetId(id); setView("test"); }}
                  options={sets.map((s) => ({ value: s.id, label: s.title, hint: `${s.rated}/${s.n}` }))}
                />
              </div>
            )}

            <div className="mt-8">
              {view === "results" && canSeeResults ? (
                <ResultsView setId={set.id} />
              ) : (
                <BlindTest key={set.id} set={set} onVoted={refresh} onSeeResults={() => setView("results")} />
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
