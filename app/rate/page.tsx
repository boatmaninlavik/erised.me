"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { supabase } from "@/lib/supabase";

interface Pair {
  pair_id: string;
  prompt: string;
  a_audio: string;
  b_audio: string;
  tags_a: string;
  tags_b: string;
}

interface ServerStatus {
  pending: number;
  ready: number;
  generating: boolean;
  rated: number;
}

export default function RatePage() {
  const [backendUrl, setBackendUrl] = useState<string | null>(null);
  const [gpuStatus, setGpuStatus] = useState<"loading" | "online" | "offline">("loading");
  const [prompt, setPrompt] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [maxSec, setMaxSec] = useState(60);
  const [count, setCount] = useState(3);
  const [queueing, setQueueing] = useState(false);
  const [currentPair, setCurrentPair] = useState<Pair | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus>({ pending: 0, ready: 0, generating: false, rated: 0 });
  const [voted, setVoted] = useState<"a" | "b" | null>(null);
  const [pairState, setPairState] = useState<"none" | "loading" | "ready" | "generating">("none");

  const loadBackendUrl = useCallback(async () => {
    const { data } = await supabase
      .from("erised_config")
      .select("value")
      .eq("key", "backend_url")
      .single();

    const url = data?.value?.trim();
    if (!url) {
      setBackendUrl(null);
      setGpuStatus("offline");
      return;
    }

    // Verify the server is actually reachable
    try {
      const resp = await fetch(`${url}/health`, { signal: AbortSignal.timeout(5000) });
      if (resp.ok) {
        setBackendUrl(url);
        setGpuStatus("online");
      } else {
        setBackendUrl(null);
        setGpuStatus("offline");
      }
    } catch {
      setBackendUrl(null);
      setGpuStatus("offline");
    }
  }, []);

  useEffect(() => {
    loadBackendUrl();
    const interval = setInterval(loadBackendUrl, 30000);
    return () => clearInterval(interval);
  }, [loadBackendUrl]);

  useEffect(() => {
    if (!backendUrl) return;
    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`${backendUrl}/api/status`);
        const data = await resp.json();
        setServerStatus(data);
        if (!currentPair && data.ready > 0) fetchNext();
      } catch {}
    }, 2000);
    return () => clearInterval(interval);
  }, [backendUrl, currentPair]);

  async function fetchNext() {
    if (!backendUrl) return;
    setPairState("loading");
    try {
      const resp = await fetch(`${backendUrl}/api/next`);
      const data = await resp.json();
      if (data.status === "ready") {
        setCurrentPair(data.pair);
        setVoted(null);
        setPairState("ready");
      } else if (data.status === "generating") {
        setPairState("generating");
      } else {
        setPairState("none");
      }
    } catch {
      setPairState("none");
    }
  }

  async function queuePairs() {
    if (!backendUrl || !prompt.trim() || !lyrics.trim()) return;
    setQueueing(true);
    try {
      await fetch(`${backendUrl}/api/queue`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, count }),
      });
      if (!currentPair) fetchNext();
    } finally {
      setQueueing(false);
    }
  }

  async function vote(choice: "a" | "b") {
    if (!backendUrl || !currentPair || voted) return;
    setVoted(choice);
    try {
      const resp = await fetch(`${backendUrl}/api/rate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pair_id: currentPair.pair_id, choice }),
      });
      const data = await resp.json();
      setServerStatus((s) => ({ ...s, rated: data.count }));
    } catch {}
    setTimeout(() => {
      setCurrentPair(null);
      fetchNext();
    }, 400);
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <nav className="px-6 py-5 flex items-center justify-between border-b border-zinc-900">
        <Link href="/" className="text-xl font-semibold tracking-tighter text-white">
          Erised
        </Link>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            gpuStatus === "online" ? "bg-green-500" :
            gpuStatus === "offline" ? "bg-zinc-600" : "bg-yellow-500 animate-pulse"
          }`} />
          <span className="text-xs text-zinc-500">
            {gpuStatus === "online" ? "GPU online" :
             gpuStatus === "offline" ? "GPU offline" : "Connecting..."}
          </span>
        </div>
      </nav>

      {gpuStatus === "loading" && (
        <div className="flex items-center justify-center min-h-[60vh]">
          <p className="text-zinc-500 text-sm animate-pulse">Connecting to GPU...</p>
        </div>
      )}

      {gpuStatus === "offline" && (
        <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4 px-6 text-center">
          <p className="text-2xl font-semibold tracking-tight">GPU is offline</p>
          <p className="text-zinc-500 text-sm max-w-sm">
            The generation server isn&apos;t running right now. Check back later.
          </p>
          <button
            onClick={loadBackendUrl}
            className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors mt-4"
          >
            Retry connection
          </button>
        </div>
      )}

      {gpuStatus === "online" && (
        <div className="max-w-2xl mx-auto px-6 py-10 space-y-8">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Rate Songs</h2>
            <p className="text-zinc-500 text-sm mt-1">
              Help Erised learn your taste. Rated: {serverStatus.rated}
            </p>
          </div>

          <div className="flex items-center gap-6 text-xs text-zinc-500">
            <span>Queued: <span className="text-white">{serverStatus.pending}</span></span>
            <span>Ready: <span className="text-white">{serverStatus.ready}</span></span>
            {serverStatus.generating && (
              <span className="text-zinc-400 animate-pulse">Generating...</span>
            )}
          </div>

          {/* Queue form */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 space-y-4">
            <h3 className="text-sm font-medium text-zinc-300">Queue new pairs</h3>
            <div>
              <label className="block text-xs text-zinc-500 mb-2 uppercase tracking-wide">Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={2}
                placeholder="e.g. UK Drill, aggressive, 808 bass"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 text-white text-sm placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 resize-none"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-2 uppercase tracking-wide">Lyrics</label>
              <textarea
                value={lyrics}
                onChange={(e) => setLyrics(e.target.value)}
                rows={6}
                placeholder={"[Verse 1]\nYour lyrics..."}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 text-white text-sm placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 resize-none font-mono"
              />
            </div>
            <div className="flex gap-4">
              <div className="flex-1">
                <label className="block text-xs text-zinc-500 mb-2 uppercase tracking-wide">
                  Max length — {maxSec}s
                </label>
                <input
                  type="range" min={10} max={240} step={5} value={maxSec}
                  onChange={(e) => setMaxSec(Number(e.target.value))}
                  className="w-full accent-white"
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-500 mb-2 uppercase tracking-wide">Pairs</label>
                <select
                  value={count}
                  onChange={(e) => setCount(Number(e.target.value))}
                  className="bg-zinc-800 border border-zinc-700 rounded-xl px-3 py-2 text-white text-sm focus:outline-none"
                >
                  {[1, 2, 3, 5, 10].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </div>
            <button
              onClick={queuePairs}
              disabled={queueing || !prompt.trim() || !lyrics.trim()}
              className="w-full py-3 bg-white text-black font-semibold rounded-xl text-sm tracking-tight hover:bg-zinc-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {queueing ? "Queuing..." : "Queue for Generation"}
            </button>
          </div>

          {/* Rating */}
          {pairState === "ready" && currentPair ? (
            <div className="space-y-4">
              <p className="text-xs text-zinc-500 text-center">Which sounds better?</p>
              <p className="text-xs text-zinc-600 text-center font-mono">{currentPair.prompt}</p>
              {(["a", "b"] as const).map((side) => (
                <div
                  key={side}
                  className={`bg-zinc-900 border rounded-2xl p-5 space-y-3 transition-colors ${
                    voted === side ? "border-white" : "border-zinc-800"
                  }`}
                >
                  <h3 className="text-sm font-medium text-zinc-300">Option {side.toUpperCase()}</h3>
                  <audio
                    controls
                    src={`${backendUrl}/audio/${currentPair[`${side}_audio` as "a_audio" | "b_audio"]}`}
                    className="w-full"
                  />
                  <p className="text-xs text-zinc-600 font-mono">
                    {currentPair[`tags_${side}` as "tags_a" | "tags_b"]}
                  </p>
                  <button
                    onClick={() => vote(side)}
                    disabled={!!voted}
                    className={`w-full py-3 rounded-xl text-sm font-medium transition-colors ${
                      voted === side
                        ? "bg-white text-black"
                        : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
                    }`}
                  >
                    {side.toUpperCase()} is better
                  </button>
                </div>
              ))}
            </div>
          ) : pairState === "generating" ? (
            <div className="text-center py-12 text-zinc-500 animate-pulse text-sm">
              Generating next pair...
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
