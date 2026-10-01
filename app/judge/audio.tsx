"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cx, formatTime, PlayIcon, Spinner } from "./ui";

export interface Player {
  playing: boolean;
  loading: boolean;
  error: boolean;
  time: number;
  duration: number;
  /** total milliseconds actually heard since the source was set */
  heardMs: () => number;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (seconds: number) => void;
}

/** One <audio> element per hook, streamed from /api/judge-lab/audio. */
export function usePlayer(src: string | null): Player {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const heard = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const a = new Audio();
    a.preload = "metadata";
    audioRef.current = a;
    const on = (ev: string, fn: () => void) => a.addEventListener(ev, fn);
    on("loadedmetadata", () => setDuration(a.duration || 0));
    on("durationchange", () => setDuration(a.duration || 0));
    on("play", () => setPlaying(true));
    on("pause", () => setPlaying(false));
    on("ended", () => setPlaying(false));
    on("waiting", () => setLoading(true));
    on("playing", () => setLoading(false));
    on("canplay", () => setLoading(false));
    on("seeked", () => setTime(a.currentTime));
    on("error", () => { setError(true); setLoading(false); setPlaying(false); });
    return () => {
      a.pause();
      a.removeAttribute("src");
      a.load();
    };
  }, []);

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    a.pause();
    heard.current = 0;
    setTime(0); setDuration(0); setError(false); setPlaying(false); setLoading(false);
    if (src) {
      a.src = src;
      a.load();
    } else {
      a.removeAttribute("src");
    }
  }, [src]);

  // Smooth progress + listened-time accounting while playing.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const a = audioRef.current;
      if (a && !a.paused && !a.seeking) heard.current += now - last;
      last = now;
      if (a) setTime(a.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const play = useCallback(() => {
    const a = audioRef.current;
    if (!a?.src) return;
    setLoading(a.readyState < 3);
    a.play().catch(() => setLoading(false));
  }, []);
  const pause = useCallback(() => audioRef.current?.pause(), []);
  const toggle = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    if (a.paused) play();
    else a.pause();
  }, [play]);
  const seek = useCallback((seconds: number) => {
    const a = audioRef.current;
    if (!a || !Number.isFinite(a.duration)) return;
    a.currentTime = Math.min(Math.max(0, seconds), Math.max(0, a.duration - 0.05));
    setTime(a.currentTime);
  }, []);
  const heardMs = useCallback(() => heard.current, []);

  return { playing, loading, error, time, duration, heardMs, play, pause, toggle, seek };
}

export function Scrubber({ player, className }: { player: Player; className?: string }) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const frac = player.duration ? Math.min(1, player.time / player.duration) : 0;

  const seekTo = (clientX: number) => {
    const el = trackRef.current;
    if (!el || !player.duration) return;
    const r = el.getBoundingClientRect();
    player.seek(((clientX - r.left) / r.width) * player.duration);
  };

  return (
    <div
      ref={trackRef}
      role="slider"
      tabIndex={-1}
      aria-label="Position"
      aria-valuemin={0}
      aria-valuemax={Math.round(player.duration)}
      aria-valuenow={Math.round(player.time)}
      aria-valuetext={`${formatTime(player.time)} of ${formatTime(player.duration)}`}
      className={cx("group relative h-4 cursor-pointer touch-none select-none", className)}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        setDragging(true);
        seekTo(e.clientX);
      }}
      onPointerMove={(e) => dragging && seekTo(e.clientX)}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
    >
      <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 overflow-hidden rounded-full bg-zinc-800">
        <div className="h-full rounded-full bg-white" style={{ width: `${frac * 100}%` }} />
      </div>
      <div
        className={cx(
          "absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow transition-opacity",
          dragging ? "opacity-100" : "opacity-0 group-hover:opacity-100",
        )}
        style={{ left: `${frac * 100}%` }}
      />
    </div>
  );
}

export function PlayButton({
  player, size = "lg", label,
}: { player: Player; size?: "lg" | "sm"; label: string }) {
  const big = size === "lg";
  return (
    <button
      onClick={player.toggle}
      disabled={player.error}
      aria-label={`${player.playing ? "Pause" : "Play"} ${label}`}
      className={cx(
        "flex shrink-0 items-center justify-center rounded-full bg-white text-black transition-colors hover:bg-zinc-200 disabled:bg-zinc-800 disabled:text-zinc-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-offset-2 focus-visible:ring-offset-black",
        big ? "h-12 w-12" : "h-8 w-8",
      )}
    >
      {player.loading && player.playing ? (
        <Spinner className={cx("border-black/20 border-t-black", big ? "h-5 w-5" : "h-3.5 w-3.5")} />
      ) : (
        <PlayIcon playing={player.playing} className={big ? "h-5 w-5" : "h-3.5 w-3.5"} />
      )}
    </button>
  );
}

export const audioUrl = (key: string) => `/api/judge-lab/audio?k=${encodeURIComponent(key)}`;

export interface Track {
  id: string;
  title: string;
  sub?: string;
  chip?: string;
  audio: string; // opaque key
}
export type PlayRequest = { track: Track; n: number } | null;

/** Floating player at the bottom of the page. A new request plays that track; repeating it toggles. */
export function MiniPlayer({ request, onTrack }: { request: PlayRequest; onTrack?: (id: string | null) => void }) {
  const [track, setTrack] = useState<Track | null>(null);
  const player = usePlayer(track ? audioUrl(track.audio) : null);

  useEffect(() => {
    if (!request) return;
    if (track && request.track.id === track.id) player.toggle();
    else setTrack(request.track);
  }, [request]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    onTrack?.(track?.id ?? null);
    if (track) player.play();
  }, [track]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!track) return null;
  return (
    <div className="fixed inset-x-0 bottom-5 z-20 mx-auto w-[min(44rem,calc(100%-2rem))]">
      <div className="flex items-center gap-4 rounded-2xl border border-white/10 bg-zinc-900/95 px-4 py-3 shadow-2xl backdrop-blur">
        <PlayButton player={player} size="sm" label={track.title} />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-3">
            <p className="truncate text-sm text-white">
              {track.title}{track.sub && <span className="text-zinc-500"> · {track.sub}</span>}
            </p>
            <p className="shrink-0 text-xs tabular-nums text-zinc-500">{formatTime(player.time)} / {formatTime(player.duration)}</p>
          </div>
          <Scrubber player={player} className="mt-1" />
        </div>
        {track.chip && <span className="hidden shrink-0 rounded-full bg-white/[0.06] px-2 py-0.5 text-[11px] text-zinc-300 sm:inline">{track.chip}</span>}
        <button onClick={() => { player.pause(); setTrack(null); }} aria-label="Close player"
          className="shrink-0 rounded-full p-1.5 text-zinc-500 transition-colors hover:bg-white/[0.06] hover:text-white">
          <svg viewBox="0 0 16 16" className="h-3.5 w-3.5"><path d="M4 4l8 8m0-8l-8 8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
        </button>
      </div>
    </div>
  );
}
