"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { BlindPair, BlindSet, Choice, HistoryItem, Revealed, SetSummary, Side } from "@/lib/judge-lab-types";
import { audioUrl, MiniPlayer, PlayButton, Scrubber, usePlayer, type Player, type PlayRequest } from "./audio";
import { Card, DownloadIcon, EyeOffIcon, Kbd, PlayIcon, Spinner, cx, formatTime, songFileName } from "./ui";

const other = (s: Side): Side => (s === "a" ? "b" : "a");
const coin = (): Side => (Math.random() < 0.5 ? "a" : "b");

function shuffled<T>(xs: T[]): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function BlindTest({ set, onVoted, onSeeResults }: {
  set: SetSummary; onVoted: () => void; onSeeResults: () => void;
}) {
  const [data, setData] = useState<BlindSet | null>(null);
  const [queue, setQueue] = useState<string[]>([]);
  const [left, setLeft] = useState<Side>(coin);
  const [saved, setSaved] = useState<Choice | null>(null);
  const [revising, setRevising] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [replay, setReplay] = useState<PlayRequest>(null);
  const [replaying, setReplaying] = useState<string | null>(null);
  // Reveals are fetched once and cached; hiding just stops showing them, so re-revealing is instant.
  const revealCache = useRef<Record<string, Revealed>>({});
  const [shown, setShown] = useState<Set<string>>(new Set());
  const reveal = useCallback(async (uid: string) => {
    if (!revealCache.current[uid]) {
      const r = await fetch(`/api/judge-lab/reveal?set=${encodeURIComponent(set.id)}&uid=${encodeURIComponent(uid)}`);
      if (!r.ok) return;
      revealCache.current[uid] = await r.json();
    }
    setShown((s) => new Set(s).add(uid));
  }, [set.id]);
  const hide = useCallback((uid: string) => setShown((s) => {
    const n = new Set(s);
    n.delete(uid);
    return n;
  }), []);
  const revealed = (uid: string) => (shown.has(uid) ? revealCache.current[uid] : undefined);
  const active = useRef<"L" | "R">("L");

  useEffect(() => {
    let live = true;
    fetch(`/api/judge-lab/set?id=${encodeURIComponent(set.id)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d: BlindSet) => {
        if (!live) return;
        const done = new Set(d.rated);
        setData(d);
        setHistory(d.history);
        setQueue(shuffled(d.pairs.filter((p) => !done.has(p.uid)).map((p) => p.uid)));
      })
      .catch(() => live && setError("Couldn't load this set."));
    return () => { live = false; };
  }, [set.id]);

  const pair = useMemo(() => data?.pairs.find((p) => p.uid === queue[0]) ?? null, [data, queue]);
  const right = other(left);
  const L = usePlayer(pair ? audioUrl(pair.audio[left]) : null);
  const R = usePlayer(pair ? audioUrl(pair.audio[right]) : null);

  const toggle = (which: "L" | "R") => {
    active.current = which;
    (which === "L" ? R : L).pause();
    (which === "L" ? L : R).toggle();
  };

  const vote = async (display: "A" | "B" | "tie") => {
    if (!pair || saved || saving) return;
    const choice: Choice = display === "tie" ? "tie" : display === "A" ? left : right;
    L.pause(); R.pause();
    setSaving(true); setError(null);
    try {
      const res = await fetch("/api/judge-lab/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          set: set.id, uid: pair.uid, choice, left, revised: revising,
          listened_ms: { [left]: L.heardMs(), [right]: R.heardMs() },
        }),
      });
      if (!res.ok) throw new Error(String(res.status));
      setSaved(choice);
      const item: HistoryItem = { uid: pair.uid, choice, left, ts: new Date().toISOString() };
      setHistory((h) => [item, ...h.filter((x) => x.uid !== pair.uid)]);
      onVoted();
    } catch {
      setError("Couldn't save that vote — is the dev server still running?");
    } finally {
      setSaving(false);
    }
  };

  const next = useCallback(() => {
    setSaved(null); setRevising(false); setLeft(coin());
    setQueue((q) => q.slice(1));
  }, []);

  const skip = () => {
    L.pause(); R.pause();
    setSaved(null); setRevising(false); setLeft(coin());
    setQueue((q) => (q.length > 1 ? [...q.slice(1), q[0]] : q));
  };

  // One window listener for the whole session; it reads the latest render's handlers.
  const keys = useRef({ L, R, toggle, vote, next, saved });
  keys.current = { L, R, toggle, vote, next, saved };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const { L, R, toggle, vote, next, saved } = keys.current;
      const cur = active.current === "L" ? L : R;
      const k = e.key.toLowerCase();
      if (k === "1") toggle("L");
      else if (k === "2") toggle("R");
      else if (k === " ") { e.preventDefault(); toggle(active.current); }
      else if (k === "arrowleft") { e.preventDefault(); cur.seek(cur.time - 5); }
      else if (k === "arrowright") { e.preventDefault(); cur.seek(cur.time + 5); }
      else if (!saved && k === "a") vote("A");
      else if (!saved && k === "b") vote("B");
      else if (!saved && k === "t") vote("tie");
      else if (saved && k === "enter") next();
      else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (error && !data) return <Empty>{error}</Empty>;
  if (!data) return <div className="flex items-center gap-3 py-16 text-zinc-500"><Spinner /> Loading pairs…</div>;

  return (
    <div className="space-y-6">
      <div className="h-1 overflow-hidden rounded-full bg-zinc-800">
        <div className="h-full rounded-full bg-white transition-all duration-500" style={{ width: `${(set.rated / set.n) * 100}%` }} />
      </div>

      {!pair ? (
        <Empty>
          <p>You&apos;ve rated all {set.n} pairs.</p>
          <button onClick={onSeeResults}
            className="mt-5 rounded-xl bg-white px-5 py-3 text-sm font-semibold tracking-tight text-black transition-colors hover:bg-zinc-200">
            See which take was which
          </button>
        </Empty>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2">
            <Take label="A" hotkey="1" player={L} onToggle={() => toggle("L")} picked={pickedDisplay(saved, left) === "A"}
              download={{ href: audioUrl(pair.audio[left]), name: songFileName(pair.prompt, "take A") }} />
            <Take label="B" hotkey="2" player={R} onToggle={() => toggle("R")} picked={pickedDisplay(saved, left) === "B"}
              download={{ href: audioUrl(pair.audio[right]), name: songFileName(pair.prompt, "take B") }} />
          </div>

          {saved ? (
            <Card className="flex flex-wrap items-center justify-between gap-4 p-5">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <p className="font-medium">
                  {saved === "tie" ? "Saved: you couldn't tell them apart." : `Saved: you picked ${pickedDisplay(saved, left)}.`}
                </p>
                <PickReveal choice={saved} left={left} info={revealed(pair.uid)}
                  onReveal={() => reveal(pair.uid)} onHide={() => hide(pair.uid)} />
              </div>
              <div className="flex items-center gap-4">
                <button onClick={() => { setSaved(null); setRevising(true); }} className="text-xs text-zinc-500 transition-colors hover:text-white">
                  Misclicked? Change pick
                </button>
                <button onClick={next} autoFocus
                  className="flex items-center gap-2 rounded-xl bg-white px-5 py-3 text-sm font-semibold tracking-tight text-black transition-colors hover:bg-zinc-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/60 focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--jl-surface)]">
                  Next pair <span className="text-black/40">↵</span>
                </button>
              </div>
            </Card>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <VoteButton onClick={() => vote("A")} hotkey="A" disabled={saving}>A is better</VoteButton>
                <VoteButton onClick={() => vote("tie")} hotkey="T" disabled={saving} quiet>Can&apos;t tell</VoteButton>
                <VoteButton onClick={() => vote("B")} hotkey="B" disabled={saving}>B is better</VoteButton>
              </div>
              <div className="flex items-center justify-between text-xs text-zinc-500">
                <span>{listenHint(L, R)}</span>
                <button onClick={skip} className="transition-colors hover:text-white">Skip this pair</button>
              </div>
            </div>
          )}
          {error && <p className="text-sm text-red-400">{error}</p>}

          {(pair.prompt || pair.lyrics) && (
            <details className="group rounded-2xl border border-white/[0.06] px-5 py-4 open:bg-white/[0.02]">
              <summary className="cursor-pointer list-none text-sm text-zinc-400 transition-colors hover:text-white">
                <span className="mr-2 inline-block transition-transform group-open:rotate-90">›</span>
                What both takes were asked for
              </summary>
              {pair.prompt && <p className="mt-3 text-sm text-zinc-300">{pair.prompt}</p>}
              {pair.lyrics && (
                <pre className="mt-3 max-h-64 overflow-y-auto whitespace-pre-wrap font-sans text-sm leading-relaxed text-zinc-500">{pair.lyrics}</pre>
              )}
            </details>
          )}
        </>
      )}

      {history.length > 0 && (
        <RatedPairs
          history={history}
          pairs={data.pairs}
          nowPlaying={replaying}
          revealed={revealed}
          shownCount={history.filter((h) => shown.has(h.uid)).length}
          onReveal={reveal}
          onHide={hide}
          onHideAll={() => setShown(new Set())}
          onPlay={(p, letter, side) => {
            L.pause(); R.pause();
            setReplay((r) => ({ n: (r?.n ?? 0) + 1, track: { id: `${p.uid}:${side}`, title: `Take ${letter}`, sub: p.prompt, audio: p.audio[side] } }));
          }}
        />
      )}
      <MiniPlayer request={replay} onTrack={setReplaying} />

      <div className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-zinc-500">
        <span><Kbd>1</Kbd> <Kbd>2</Kbd> play A / B</span>
        <span><Kbd>Space</Kbd> play / pause</span>
        <span><Kbd>←</Kbd> <Kbd>→</Kbd> 5 s</span>
        <span><Kbd>A</Kbd> <Kbd>B</Kbd> <Kbd>T</Kbd> vote</span>
        <span><Kbd>↵</Kbd> next pair</span>
      </div>
    </div>
  );
}

const pickedDisplay = (choice: Choice | null, left: Side) => (!choice || choice === "tie" ? null : choice === left ? "A" : "B");

function listenHint(L: Player, R: Player) {
  const a = L.heardMs() > 2000, b = R.heardMs() > 2000;
  if (!a && !b) return "Listen to both, then pick.";
  if (!a) return "You haven't heard A yet.";
  if (!b) return "You haven't heard B yet.";
  return "Pick when you're ready.";
}

function Take({ label, hotkey, player, onToggle, picked, download }: {
  label: string; hotkey: string; player: Player; onToggle: () => void; picked: boolean;
  download: { href: string; name: string };
}) {
  return (
    <Card className={cx("p-5 transition-shadow", player.playing && "ring-1 ring-white/25", picked && "ring-1 ring-white/60")}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium tracking-tight">Take {label}{picked && <span className="ml-2 text-zinc-400">· your pick</span>}</span>
        <div className="flex items-center gap-2">
          <DownloadLink {...download} label={`Download take ${label}`} />
          <Kbd>{hotkey}</Kbd>
        </div>
      </div>
      <div className="mt-5 flex items-center gap-4">
        <PlayButton player={{ ...player, toggle: onToggle }} label={`take ${label}`} />
        <div className="min-w-0 flex-1">
          <Scrubber player={player} />
          <div className="mt-1 flex justify-between text-xs tabular-nums text-zinc-500">
            <span>{formatTime(player.time)}</span>
            <span>{player.duration ? formatTime(player.duration) : "–:––"}</span>
          </div>
        </div>
      </div>
      {player.error && <p className="mt-3 text-xs text-red-400">Couldn&apos;t load this take — skip the pair.</p>}
    </Card>
  );
}

function VoteButton({ children, onClick, hotkey, disabled, quiet }: {
  children: React.ReactNode; onClick: () => void; hotkey: string; disabled?: boolean; quiet?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cx(
        "flex items-center justify-center gap-2 rounded-xl border py-3.5 text-sm font-medium transition-colors disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40",
        quiet ? "border-white/[0.06] text-zinc-400 hover:text-white hover:bg-white/[0.04]"
              : "border-white/15 bg-white/[0.04] text-white hover:bg-white/[0.09]",
      )}
    >
      {children}
      <span className="hidden sm:inline"><Kbd>{hotkey}</Kbd></span>
    </button>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <Card className="px-6 py-14 text-center text-sm text-zinc-400">{children}</Card>;
}

function DownloadLink({ href, name, label, text }: { href: string; name: string; label: string; text?: string }) {
  return (
    <a href={href} download={name} aria-label={label} title={label}
      className={cx("inline-flex items-center gap-1.5 rounded-md text-zinc-500 transition-colors hover:bg-white/[0.06] hover:text-white",
        text ? "px-2 py-1 text-xs" : "h-[1.4rem] w-[1.4rem] justify-center")}>
      <DownloadIcon className="h-3.5 w-3.5" />
      {text}
    </a>
  );
}

const shortPrompt = (p: string) => {
  const text = p.replace(/\s+/g, " ").trim() || "Untitled prompt";
  return text.length <= 70 ? text : `${text.slice(0, 70).replace(/[,; ]+\S*$/, "")}…`;
};

/** Every pair you've already rated, in the A/B layout you heard it — replay or download either take. */
function RatedPairs({ history, pairs, nowPlaying, revealed, shownCount, onReveal, onHide, onHideAll, onPlay }: {
  history: HistoryItem[];
  pairs: BlindPair[];
  nowPlaying: string | null;
  revealed: (uid: string) => Revealed | undefined;
  shownCount: number;
  onReveal: (uid: string) => void;
  onHide: (uid: string) => void;
  onHideAll: () => void;
  onPlay: (p: BlindPair, letter: "A" | "B", side: Side) => void;
}) {
  const byUid = new Map(pairs.map((p) => [p.uid, p]));
  return (
    <details className="group rounded-2xl border border-white/[0.06] px-5 py-4 open:bg-white/[0.02]">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-sm text-zinc-400">
        <span className="transition-colors hover:text-white">
          <span className="mr-2 inline-block transition-transform group-open:rotate-90">›</span>
          Pairs you&apos;ve rated ({history.length}) — replay or download a take
        </span>
        {shownCount > 0 && (
          <button
            onClick={(e) => { e.preventDefault(); onHideAll(); }}
            className="hidden items-center gap-1.5 text-xs text-zinc-500 transition-colors hover:text-white group-open:inline-flex"
          >
            <EyeOffIcon className="h-3.5 w-3.5" /> Hide all models
          </button>
        )}
      </summary>
      <ul className="mt-3 divide-y divide-white/[0.06]">
        {history.map((h) => {
          const p = byUid.get(h.uid);
          if (!p) return null;
          const takes = [["A", h.left], ["B", other(h.left)]] as const;
          return (
            <li key={h.uid} className="flex flex-wrap items-center gap-x-5 gap-y-2 py-3">
              <p className="min-w-0 flex-1 basis-64 truncate text-sm text-zinc-300" title={p.prompt}>{shortPrompt(p.prompt)}</p>
              <div className="flex w-80 shrink-0 items-center gap-3 text-xs">
                <span className="whitespace-nowrap text-zinc-500">{h.choice === "tie" ? "Couldn't tell" : `You picked ${h.choice === h.left ? "A" : "B"}`}</span>
                <PickReveal choice={h.choice} left={h.left} info={revealed(h.uid)}
                  onReveal={() => onReveal(h.uid)} onHide={() => onHide(h.uid)} small />
              </div>
              <div className="flex shrink-0 items-center gap-3">
                {takes.map(([letter, side]) => {
                  const on = nowPlaying === `${p.uid}:${side}`;
                  return (
                    <div key={letter} className="flex items-center gap-1">
                      <button onClick={() => onPlay(p, letter, side)}
                        className={cx("flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition-colors",
                          on ? "border-white/40 bg-white/[0.08] text-white" : "border-white/10 text-zinc-300 hover:bg-white/[0.05]")}>
                        <PlayIcon playing={false} className="h-3 w-3" /> {letter}
                      </button>
                      <DownloadLink href={audioUrl(p.audio[side])} name={songFileName(p.prompt, `take ${letter}`)} label={`Download take ${letter}`} />
                    </div>
                  );
                })}
              </div>
            </li>
          );
        })}
      </ul>
    </details>
  );
}

/** "Fine-tuned Shao (sft_v1)" -> "Fine-tuned", "Original Shao" -> "Original" (full name stays in the tooltip). */
const shortName = (name: string) => name.replace(/\s*\([^)]*\)\s*$/, "").replace(/\s*\bShao\b\s*/g, " ").trim() || name;

/** Dot + model name: the model being tested is blue, the one it's compared against light gray. */
function ModelTag({ m, short }: { m: Revealed[Side]; short?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span className="h-2 w-2 rounded-full" style={{ background: m.newer ? "var(--jl-accent)" : "var(--jl-older)" }} />
      <span className="text-zinc-200">{short ? shortName(m.name) : m.name}</span>
    </span>
  );
}

/** "Reveal model" button that turns into the model behind your pick (both models for "can't tell").
 *  Click the revealed label to hide it again. */
function PickReveal({ choice, left, info, onReveal, onHide, small }: {
  choice: Choice; left: Side; info?: Revealed; onReveal: () => void; onHide: () => void; small?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  if (info) {
    const full = choice === "tie"
      ? `A: ${info[left].name} · B: ${info[other(left)].name}`
      : info[choice].name;
    return (
      <button
        onClick={onHide}
        title={`${full} — click to hide`}
        className={cx("group/tag inline-flex items-center gap-2 rounded-md px-1.5 py-0.5 -mx-1.5 transition-colors hover:bg-white/[0.05]",
          small ? "text-xs" : "text-sm")}
      >
        {choice === "tie" ? (
          <>
            <span className="text-zinc-500">A</span><ModelTag m={info[left]} short={small} />
            <span className="text-zinc-500">B</span><ModelTag m={info[other(left)]} short={small} />
          </>
        ) : (
          <ModelTag m={info[choice]} short={small} />
        )}
        <EyeOffIcon className="h-3.5 w-3.5 text-zinc-500 opacity-0 transition-opacity group-hover/tag:opacity-100" />
      </button>
    );
  }
  return (
    <button
      onClick={async () => { setBusy(true); await onReveal(); setBusy(false); }}
      disabled={busy}
      className={cx("rounded-md border border-white/10 text-zinc-400 transition-colors hover:border-white/25 hover:text-white disabled:opacity-50",
        small ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs")}
    >
      {busy ? "…" : "Reveal model"}
    </button>
  );
}
