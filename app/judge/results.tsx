"use client";

import { useEffect, useState } from "react";
import type { Results, Side } from "@/lib/judge-lab-types";
import { audioUrl, MiniPlayer, type PlayRequest } from "./audio";
import { Card, DownloadIcon, PlayIcon, Spinner, cx, pct, songFileName, wilson } from "./ui";

// Newer model (listed last in the pair set) is blue, the one it's compared against light gray.
const COLORS = ["var(--jl-older)", "var(--jl-accent)"];

const shortPrompt = (p: string) => {
  const text = p.replace(/\s+/g, " ").trim() || "Untitled prompt";
  return text.length <= 70 ? text : `${text.slice(0, 70).replace(/[,; ]+\S*$/, "")}…`;
};

export function ResultsView({ setId }: { setId: string }) {
  const [res, setRes] = useState<Results | null>(null);
  const [error, setError] = useState(false);
  const [request, setRequest] = useState<PlayRequest>(null);
  const [nowPlaying, setNowPlaying] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    fetch(`/api/judge-lab/results?id=${encodeURIComponent(setId)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d: Results) => live && setRes(d))
      .catch(() => live && setError(true));
    return () => { live = false; };
  }, [setId]);

  if (error) return <p className="text-sm text-red-400">Couldn&apos;t load the results.</p>;
  if (!res) return <div className="flex items-center gap-3 py-16 text-zinc-500"><Spinner /> Unblinding…</div>;

  const ids = Object.keys(res.models);
  const color = Object.fromEntries(ids.map((m, i) => [m, COLORS[Math.min(i, COLORS.length - 1)]]));
  const newer = ids[ids.length - 1];
  const decided = ids.reduce((n, m) => n + (res.wins[m] ?? 0), 0);
  const ci = wilson(res.wins[newer] ?? 0, decided);
  const play = (p: Results["pairs"][number], side: Side) => setRequest((r) => ({
    n: (r?.n ?? 0) + 1,
    track: { id: `${p.uid}:${side}`, title: res.models[p.model[side]], sub: shortPrompt(p.prompt), audio: p.audio[side] },
  }));

  return (
    <div className="space-y-8">
      <Card className="p-6">
        {decided ? (
          <>
            <p className="text-sm text-zinc-400">On the {decided} pairs where you had a preference</p>
            <p className="mt-2 text-3xl font-semibold tracking-tight">
              <span style={{ color: color[newer] }}>{res.models[newer]}</span> won {res.wins[newer] ?? 0} of {decided}
              <span className="text-zinc-500"> · {pct((res.wins[newer] ?? 0) / decided)}</span>
            </p>
            {ci && (
              <p className="mt-2 text-sm text-zinc-400">
                Likely range {pct(ci[0])}–{pct(ci[1])}. {ci[0] > 0.5 ? "That's clearly better than a coin flip."
                  : ci[1] < 0.5 ? "That's clearly worse than a coin flip." : "That range still includes 50%, so it could be a coin flip — more pairs would settle it."}
              </p>
            )}
          </>
        ) : (
          <p className="text-zinc-400">No clear picks yet.</p>
        )}

        <div className="mt-6 flex h-3 overflow-hidden rounded-full bg-zinc-800" role="img"
          aria-label={ids.map((m) => `${res.models[m]} ${res.wins[m] ?? 0}`).join(", ") + `, can't tell ${res.ties}`}>
          {[...ids].reverse().map((m) => (
            <div key={m} style={{ width: `${((res.wins[m] ?? 0) / Math.max(1, res.total)) * 100}%`, background: color[m] }} />
          ))}
          <div className="bg-zinc-600" style={{ width: `${(res.ties / Math.max(1, res.total)) * 100}%` }} />
        </div>
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-sm">
          {[...ids].reverse().map((m) => (
            <span key={m} className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: color[m] }} />
              <span className="text-zinc-300">{res.models[m]}</span>
              <span className="tabular-nums text-white">{res.wins[m] ?? 0}</span>
            </span>
          ))}
          <span className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-zinc-600" />
            <span className="text-zinc-300">Can&apos;t tell</span>
            <span className="tabular-nums text-white">{res.ties}</span>
          </span>
          {res.rated < res.total && <span className="text-zinc-500">{res.total - res.rated} not rated yet</span>}
        </div>
      </Card>

      <div>
        <h2 className="text-lg font-medium tracking-tight">Pair by pair</h2>
        <p className="mt-1 text-sm text-zinc-400">Replay either take, now labeled.</p>
        <ul className="mt-4 divide-y divide-white/[0.06] rounded-2xl border border-white/[0.06]">
          {res.pairs.map((p) => {
            const pickedModel = p.choice && p.choice !== "tie" ? p.model[p.choice] : null;
            return (
              <li key={p.uid} className="flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-4">
                <p className="min-w-0 flex-1 basis-64 truncate text-sm text-zinc-300" title={p.prompt}>{shortPrompt(p.prompt)}</p>
                <span className="w-56 shrink-0 text-sm">
                  {pickedModel ? (
                    <><span className="text-zinc-500">You picked </span><span style={{ color: color[pickedModel] }}>{res.models[pickedModel]}</span></>
                  ) : (
                    <span className="text-zinc-500">{p.choice === "tie" ? "Couldn't tell" : "Not rated"}</span>
                  )}
                </span>
                <div className="flex shrink-0 gap-2">
                  {(["a", "b"] as const)
                    .slice()
                    .sort((x, y) => ids.indexOf(p.model[x]) - ids.indexOf(p.model[y]))
                    .map((side) => {
                      const id = `${p.uid}:${side}`;
                      const on = nowPlaying === id;
                      const name = res.models[p.model[side]];
                      return (
                        <div key={side} className="flex items-center gap-1">
                          <button onClick={() => play(p, side)}
                            className={cx("flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition-colors",
                              on ? "border-white/40 bg-white/[0.08] text-white" : "border-white/10 text-zinc-300 hover:bg-white/[0.05]")}>
                            <PlayIcon playing={false} className="h-3 w-3" />
                            <span style={{ color: color[p.model[side]] }}>{name}</span>
                          </button>
                          <a href={audioUrl(p.audio[side])} download={songFileName(p.prompt, name)}
                            aria-label={`Download ${name}`} title={`Download ${name}`}
                            className="inline-flex h-7 w-7 items-center justify-center rounded-md text-zinc-500 transition-colors hover:bg-white/[0.06] hover:text-white">
                            <DownloadIcon className="h-3.5 w-3.5" />
                          </a>
                        </div>
                      );
                    })}
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      <MiniPlayer request={request} onTrack={setNowPlaying} />
    </div>
  );
}
