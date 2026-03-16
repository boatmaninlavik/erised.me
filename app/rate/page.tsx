"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import { supabase } from "@/lib/supabase";

interface Pair {
  pair_id: string;
  prompt: string;
  lyrics?: string;
  a_audio: string;
  b_audio?: string;
  tags_a: string;
  tags_b?: string;
  a_ready?: boolean;
  b_ready?: boolean;
}

interface ServerStatus {
  pending: number;
  ready: number;
  generating: boolean;
  rated: number;
}

export default function RatePage() {
  const [backendUrl, setBackendUrl] = useState<string | null>(null);
  const [gpuStatus, setGpuStatus] = useState<"loading" | "online" | "offline" | "starting">("loading");
  const [prompt, setPrompt] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [maxSec, setMaxSec] = useState(60);
  const [count, setCount] = useState(3);
  const [queueing, setQueueing] = useState(false);
  const [currentPair, setCurrentPair] = useState<Pair | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus>({ pending: 0, ready: 0, generating: false, rated: 0 });
  const [voted, setVoted] = useState<"a" | "b" | null>(null);
  const [pairState, setPairState] = useState<"none" | "loading" | "ready" | "partial" | "generating">("none");
  const [randomizingPrompt, setRandomizingPrompt] = useState(false);
  const [randomizingLyrics, setRandomizingLyrics] = useState(false);
  const [randomError, setRandomError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const pollingPairId = useRef<string | null>(null);

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

    // Quick check first (5s) — if it responds, GPU is already warm
    try {
      const resp = await fetch(`${url}/health`, { signal: AbortSignal.timeout(5000) });
      if (resp.ok) {
        setBackendUrl(url);
        setGpuStatus("online");
        return;
      }
    } catch {
      // Timeout or error — GPU is likely cold-starting on Modal
    }

    // Cold start: show "starting" and wait longer (up to 120s)
    setGpuStatus("starting");
    try {
      const resp = await fetch(`${url}/health`, { signal: AbortSignal.timeout(120000) });
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
        if (!currentPair && !pollingPairId.current && data.ready > 0) fetchNext();
      } catch {}
    }, 2000);
    return () => clearInterval(interval);
  }, [backendUrl, currentPair]);

  // Poll for song B when we have a partial pair
  useEffect(() => {
    if (!backendUrl || pairState !== "partial" || !pollingPairId.current) return;
    const pid = pollingPairId.current;
    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`${backendUrl}/api/pair/${pid}`);
        const data = await resp.json();
        if (data.status === "ready" && data.pair.b_ready) {
          setCurrentPair(data.pair);
          setPairState("ready");
          pollingPairId.current = null;
        }
      } catch {}
    }, 2000);
    return () => clearInterval(interval);
  }, [backendUrl, pairState]);

  async function fetchNext() {
    if (!backendUrl) return;
    setPairState("loading");
    try {
      const resp = await fetch(`${backendUrl}/api/next`);
      const data = await resp.json();
      if (data.status === "ready") {
        setCurrentPair(data.pair);
        setVoted(null);
        setSaved(false);
        setPairState("ready");
        pollingPairId.current = null;
      } else if (data.status === "partial") {
        setCurrentPair(data.pair);
        setVoted(null);
        setSaved(false);
        setPairState("partial");
        pollingPairId.current = data.pair.pair_id;
      } else if (data.status === "generating") {
        setPairState("generating");
      } else {
        setPairState("none");
      }
    } catch {
      setPairState("none");
    }
  }

  async function randomize(type: "prompt" | "lyrics") {
    const setLoading = type === "prompt" ? setRandomizingPrompt : setRandomizingLyrics;
    const setValue = type === "prompt" ? setPrompt : setLyrics;
    setLoading(true);
    setRandomError(null);
    try {
      const resp = await fetch("/api/generate-random", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, context: type === "lyrics" ? prompt : undefined }),
      });
      const data = await resp.json();
      if (data.text) setValue(data.text);
      else if (data.error) setRandomError(data.error);
    } catch {
      setRandomError("Failed to generate random " + type);
    } finally {
      setLoading(false);
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
    if (!backendUrl || !currentPair || voted || pairState !== "ready") return;
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
    // Don't auto-advance — let user save winner first if they want
  }

  function goToNext() {
    setCurrentPair(null);
    pollingPairId.current = null;
    setSaved(false);
    setVoted(null);
    fetchNext();
  }

  async function saveWinnerToLibrary() {
    if (!backendUrl || !currentPair || !voted) return;
    setSaving(true);
    try {
      const winnerAudio = voted === "a" ? currentPair.a_audio : currentPair.b_audio!;
      const winnerTags = voted === "a" ? currentPair.tags_a : currentPair.tags_b!;

      const audioResp = await fetch(`${backendUrl}/audio/${winnerAudio}`);
      const blob = await audioResp.blob();
      const ext = winnerAudio.split(".").pop() || "wav";
      const filename = `${Date.now()}_rate_winner.${ext}`;

      const { data: uploadData, error: uploadErr } = await supabase.storage
        .from("dpo-songs")
        .upload(filename, blob, { contentType: blob.type || "audio/wav", upsert: false });

      if (uploadErr) throw uploadErr;

      const { data: urlData } = supabase.storage.from("dpo-songs").getPublicUrl(uploadData.path);

      const { error: insertErr } = await supabase.from("dpo-songs").insert({
        title: "Untitled",
        prompt: currentPair.prompt,
        lyrics: currentPair.lyrics || "",
        tags: winnerTags,
        audio_url: urlData.publicUrl,
        num_frames: null,
        model: "rate-winner",
      });

      if (insertErr) throw insertErr;
      setSaved(true);
    } catch (e: unknown) {
      console.error("Save failed:", e);
    } finally {
      setSaving(false);
    }
  }

  const bReady = currentPair?.b_ready !== false;

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
             gpuStatus === "offline" ? "GPU offline" :
             gpuStatus === "starting" ? "GPU starting..." : "Connecting..."}
          </span>
        </div>
      </nav>

      {(gpuStatus === "loading" || gpuStatus === "starting") && (
        <div className="flex items-center justify-center min-h-[60vh]">
          <p className="text-zinc-500 text-sm animate-pulse">
            {gpuStatus === "starting" ? "Starting GPU — this takes about a minute..." : "Connecting to GPU..."}
          </p>
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
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-zinc-500 uppercase tracking-wide">Prompt</label>
                <button
                  onClick={() => randomize("prompt")}
                  disabled={randomizingPrompt}
                  className="text-xs text-zinc-500 hover:text-white transition-colors disabled:opacity-40"
                >
                  {randomizingPrompt ? "Generating..." : "get random prompt"}
                </button>
              </div>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={2}
                placeholder="e.g. UK Drill, aggressive, 808 bass"
                className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 text-white text-sm placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 resize-none"
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-zinc-500 uppercase tracking-wide">Lyrics</label>
                <button
                  onClick={() => randomize("lyrics")}
                  disabled={randomizingLyrics}
                  className="text-xs text-zinc-500 hover:text-white transition-colors disabled:opacity-40"
                >
                  {randomizingLyrics ? "Generating..." : "get random lyrics"}
                </button>
              </div>
              <textarea
                value={lyrics}
                onChange={(e) => setLyrics(e.target.value)}
                rows={6}
                placeholder={"[Verse 1]\nYour lyrics..."}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 text-white text-sm placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 resize-none font-mono"
              />
            </div>
            {randomError && (
              <p className="text-red-400 text-xs">{randomError}</p>
            )}
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
          {(pairState === "ready" || pairState === "partial") && currentPair ? (
            <div className="space-y-4">
              <p className="text-xs text-zinc-500 text-center">
                {voted ? "Voted!" : bReady ? "Which sounds better?" : "Listen to A while B generates..."}
              </p>
              <p className="text-xs text-zinc-600 text-center font-mono break-words overflow-wrap-anywhere">{currentPair.prompt}</p>

              {/* Option A */}
              <div
                className={`bg-zinc-900 border rounded-2xl p-5 space-y-3 transition-colors ${
                  voted === "a" ? "border-white" : "border-zinc-800"
                }`}
              >
                <h3 className="text-sm font-medium text-zinc-300">Option A</h3>
                <audio
                  controls
                  src={`${backendUrl}/audio/${currentPair.a_audio}`}
                  className="w-full"
                />
                <p className="text-xs text-zinc-600 font-mono break-words overflow-wrap-anywhere">
                  {currentPair.tags_a}
                </p>
                <button
                  onClick={() => vote("a")}
                  disabled={!!voted || !bReady}
                  className={`w-full py-3 rounded-xl text-sm font-medium transition-colors ${
                    voted === "a"
                      ? "bg-white text-black"
                      : !bReady
                        ? "bg-zinc-800/50 text-zinc-600 cursor-not-allowed"
                        : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
                  }`}
                >
                  {!bReady ? "Waiting for B..." : voted === "a" ? "Winner" : "A is better"}
                </button>
              </div>

              {/* Option B */}
              <div
                className={`bg-zinc-900 border rounded-2xl p-5 space-y-3 transition-colors ${
                  !bReady ? "border-zinc-800/50 opacity-60" :
                  voted === "b" ? "border-white" : "border-zinc-800"
                }`}
              >
                <h3 className="text-sm font-medium text-zinc-300">Option B</h3>
                {bReady ? (
                  <>
                    <audio
                      controls
                      src={`${backendUrl}/audio/${currentPair.b_audio}`}
                      className="w-full"
                    />
                    <p className="text-xs text-zinc-600 font-mono break-words overflow-wrap-anywhere">
                      {currentPair.tags_b}
                    </p>
                    <button
                      onClick={() => vote("b")}
                      disabled={!!voted}
                      className={`w-full py-3 rounded-xl text-sm font-medium transition-colors ${
                        voted === "b"
                          ? "bg-white text-black"
                          : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
                      }`}
                    >
                      {voted === "b" ? "Winner" : "B is better"}
                    </button>
                  </>
                ) : (
                  <div className="py-8 text-center">
                    <div className="inline-block w-5 h-5 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin mb-3" />
                    <p className="text-sm text-zinc-500 animate-pulse">Generating song B...</p>
                  </div>
                )}
              </div>

              {/* Post-vote actions: save winner + next */}
              {voted && (
                <div className="flex gap-3 pt-2">
                  <button
                    onClick={saveWinnerToLibrary}
                    disabled={saving || saved}
                    className={`flex-1 py-3 rounded-xl text-sm font-medium transition-colors ${
                      saved
                        ? "bg-green-900/30 text-green-400 border border-green-800"
                        : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 border border-zinc-700"
                    }`}
                  >
                    {saved ? "Saved to Library" : saving ? "Saving..." : "Save Winner to Library"}
                  </button>
                  <button
                    onClick={goToNext}
                    className="flex-1 py-3 bg-white text-black font-semibold rounded-xl text-sm tracking-tight hover:bg-zinc-100 transition-colors"
                  >
                    Next Pair
                  </button>
                </div>
              )}
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
