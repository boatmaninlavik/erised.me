"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { supabase } from "@/lib/supabase";
import { useAuth } from "@/lib/auth-context";
import { Navbar } from "@/components/navbar";

type JobStatus = "idle" | "pending" | "running" | "done" | "error";

interface GenerationResult {
  audio_file: string;
  tags: string;
  num_frames: number;
  elapsed: number;
  model: string;
}

interface StreamState {
  status: JobStatus;
  result: GenerationResult | null;
  progress: { current_frame: number; total_frames: number } | null;
  partialAudio: string | null;
  partialVersion: number;
  chunkFiles: string[] | null;
}

const EMPTY_STREAM: StreamState = {
  status: "idle",
  result: null,
  progress: null,
  partialAudio: null,
  partialVersion: 0,
  chunkFiles: null,
};

interface Pair {
  pair_id: string;
  prompt: string;
  lyrics?: string;
  a_job: string;
  b_job: string;
  a_audio: string | null;
  b_audio: string | null;
  tags_a: string | null;
  tags_b: string | null;
  a_status?: string;
  b_status?: string;
}

interface ServerStatus {
  pending: number;
  ready: number;
  generating: boolean;
  rated: number;
}

async function fetchRetry(url: string, options?: RequestInit, maxRetries = 30): Promise<Response> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const resp = await fetch(url, options);
      if (resp.status === 502 || resp.status === 504) {
        await new Promise((r) => setTimeout(r, 3000));
        continue;
      }
      return resp;
    } catch {
      await new Promise((r) => setTimeout(r, 3000));
    }
  }
  throw new Error("Server unreachable after retries");
}

function streamJob(
  backendUrl: string,
  jobId: string,
  onUpdate: (state: Partial<StreamState>) => void,
): () => void {
  let cancelled = false;

  const es = new EventSource(`${backendUrl}/api/job/${jobId}/stream`);
  let sseWorking = false;

  es.addEventListener("progress", (e) => {
    sseWorking = true;
    const data = JSON.parse((e as MessageEvent).data);
    onUpdate({ status: "running", progress: data });
  });

  es.addEventListener("chunk", (e) => {
    sseWorking = true;
    const data = JSON.parse((e as MessageEvent).data);
    onUpdate({
      status: "running",
      partialAudio: data.audio_file,
      partialVersion: data.version,
      chunkFiles: data.chunk_files || null,
    });
  });

  es.addEventListener("done", (e) => {
    sseWorking = true;
    const data = JSON.parse((e as MessageEvent).data);
    onUpdate({ status: "done", result: data });
    es.close();
  });

  es.addEventListener("error", () => {
    es.close();
    if (!sseWorking) {
      fallbackPoll();
      return;
    }
    fallbackPoll();
  });

  async function fallbackPoll() {
    while (!cancelled) {
      await new Promise((r) => setTimeout(r, 2000));
      if (cancelled) break;
      try {
        const resp = await fetchRetry(`${backendUrl}/api/job/${jobId}`);
        const data = await resp.json();
        if (data.status === "done") {
          onUpdate({ status: "done", result: data.result });
          return;
        }
        if (data.status === "error") {
          onUpdate({ status: "error" });
          return;
        }
        const update: Partial<StreamState> = { status: "running" };
        if (data.progress) update.progress = data.progress;
        if (data.partial_audio_file) update.partialAudio = data.partial_audio_file;
        if (data.partial_version != null) update.partialVersion = data.partial_version;
        onUpdate(update);
      } catch {
        // retry
      }
    }
  }

  return () => {
    cancelled = true;
    es.close();
  };
}

const SEAN_EMAIL = "zsean@berkeley.edu";

function SongCard({
  backendUrl,
  status,
  result,
  progress,
  partialAudio,
  partialVersion,
  chunkFiles,
  label,
}: {
  backendUrl: string;
  status: JobStatus;
  result: GenerationResult | null;
  progress: { current_frame: number; total_frames: number } | null;
  partialAudio: string | null;
  partialVersion: number;
  chunkFiles: string[] | null;
  label?: string;
}) {
  const audioRefA = useRef<HTMLAudioElement>(null);
  const audioRefB = useRef<HTMLAudioElement>(null);
  const finalAudioRef = useRef<HTMLAudioElement>(null);

  const chunksRef = useRef<{ duration: number; startOffset: number }[]>([]);
  const activeIndexRef = useRef(-1);
  const loadedChunksRef = useRef(0);
  const loadedVersionRef = useRef(0);
  const blobUrlsRef = useRef<string[]>([]);

  const [totalDuration, setTotalDuration] = useState(0);
  const [displayTime, setDisplayTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasAudio, setHasAudio] = useState(false);
  const [showFinalPlayer, setShowFinalPlayer] = useState(false);

  const getEl = useCallback((idx: number) => {
    return idx % 2 === 0 ? audioRefA.current : audioRefB.current;
  }, []);

  useEffect(() => {
    if (status === "pending" || status === "idle") {
      chunksRef.current = [];
      activeIndexRef.current = -1;
      loadedChunksRef.current = 0;
      loadedVersionRef.current = 0;
      setTotalDuration(0);
      setDisplayTime(0);
      setIsPlaying(false);
      setHasAudio(false);
      setShowFinalPlayer(false);
      blobUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      blobUrlsRef.current = [];
      [audioRefA.current, audioRefB.current].forEach((el) => {
        if (el) { el.pause(); el.removeAttribute("src"); el.load(); }
      });
    }
  }, [status]);

  useEffect(() => () => {
    blobUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  useEffect(() => {
    if (result || !partialAudio || partialVersion <= 0) return;
    if (partialVersion <= loadedVersionRef.current) return;
    loadedVersionRef.current = partialVersion;

    if (!hasAudio) {
      const url = `${backendUrl}/audio/${partialAudio}?v=${partialVersion}`;
      const el = audioRefA.current;
      if (!el) return;
      const onReady = () => {
        el.removeEventListener("canplay", onReady);
        if (!isFinite(el.duration)) return;
        chunksRef.current = [{ duration: el.duration, startOffset: 0 }];
        activeIndexRef.current = 0;
        loadedChunksRef.current = 1;
        setTotalDuration(el.duration);
        setHasAudio(true);
        el.play().catch(() => {});
        setIsPlaying(true);
      };
      el.addEventListener("canplay", onReady);
      el.src = url;
      el.load();
      return;
    }

    const newChunkFile = chunkFiles && chunkFiles.length > 0
      ? chunkFiles[chunkFiles.length - 1]
      : null;
    if (!newChunkFile) return;

    const chunkUrl = `${backendUrl}/audio/${newChunkFile}`;
    const nextIdx = loadedChunksRef.current;

    fetch(chunkUrl)
      .then((r) => r.blob())
      .then((blob) => {
        const blobUrl = URL.createObjectURL(blob);
        blobUrlsRef.current.push(blobUrl);
        const el = getEl(nextIdx);
        if (!el) return;
        const onReady = () => {
          el.removeEventListener("canplaythrough", onReady);
          if (!isFinite(el.duration)) return;
          const prevTotal = chunksRef.current.reduce((s, c) => s + c.duration, 0);
          chunksRef.current.push({ duration: el.duration, startOffset: prevTotal });
          loadedChunksRef.current = nextIdx + 1;
          setTotalDuration(prevTotal + el.duration);
        };
        el.addEventListener("canplaythrough", onReady);
        el.src = blobUrl;
        el.load();
      })
      .catch(() => {});
  }, [partialAudio, partialVersion, chunkFiles, result, backendUrl, hasAudio, getEl]);

  useEffect(() => {
    if (!isPlaying || activeIndexRef.current < 0) return;
    const tick = () => {
      const idx = activeIndexRef.current;
      const el = getEl(idx);
      if (!el || !isFinite(el.currentTime)) return;
      const chunk = chunksRef.current[idx];
      if (chunk) setDisplayTime(chunk.startOffset + el.currentTime);

      const remaining = el.duration - el.currentTime;
      const nextIdx = idx + 1;
      if (nextIdx < loadedChunksRef.current && isFinite(remaining) && remaining < 0.5) {
        const nextEl = getEl(nextIdx);
        if (nextEl) {
          nextEl.currentTime = 0;
          nextEl.play().then(() => {
            el.pause();
            activeIndexRef.current = nextIdx;
          }).catch(() => {});
        }
      }
    };
    const id = setInterval(tick, 50);
    return () => clearInterval(id);
  }, [isPlaying, getEl]);

  const savedPosRef = useRef(0);
  useEffect(() => {
    if (!result) return;
    const idx = activeIndexRef.current;
    const el = idx >= 0 ? getEl(idx) : null;
    let pos = 0;
    if (el && isFinite(el.currentTime) && !el.paused) {
      const chunk = chunksRef.current[idx];
      pos = chunk ? chunk.startOffset + el.currentTime : 0;
    }
    savedPosRef.current = pos;
    [audioRefA.current, audioRefB.current].forEach((a) => { if (a) a.pause(); });
    setIsPlaying(false);
    setShowFinalPlayer(true);
  }, [result, getEl]);

  useEffect(() => {
    if (!showFinalPlayer || !result) return;
    const fin = finalAudioRef.current;
    if (!fin) return;
    if (fin.src && fin.src.includes(result.audio_file)) return;
    const pos = savedPosRef.current;
    const onReady = () => {
      fin.removeEventListener("canplay", onReady);
      if (pos > 0 && isFinite(fin.duration) && pos < fin.duration) fin.currentTime = pos;
      fin.play().catch(() => {});
    };
    fin.addEventListener("canplay", onReady);
    fin.src = `${backendUrl}/audio/${result.audio_file}`;
    fin.load();
  }, [showFinalPlayer, result, backendUrl]);

  const togglePlay = useCallback(() => {
    const el = getEl(activeIndexRef.current);
    if (!el) return;
    if (el.paused) { el.play().catch(() => {}); setIsPlaying(true); }
    else { el.pause(); setIsPlaying(false); }
  }, [getEl]);

  const formatTime = (t: number) => {
    if (!isFinite(t) || t < 0) t = 0;
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const isLoading = status === "pending" || status === "running";
  const progressPct = progress?.total_frames
    ? Math.round((progress.current_frame / progress.total_frames) * 100)
    : 0;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        {label && <h3 className="font-medium text-sm tracking-tight text-zinc-300">{label}</h3>}
        {isLoading && !hasAudio && (
          <span className="text-xs text-zinc-500 animate-pulse">
            Composing{progressPct > 0 ? ` (${progressPct}%)` : "..."}
          </span>
        )}
        {isLoading && hasAudio && (
          <span className="text-xs text-zinc-500 animate-pulse">
            Streaming{progressPct > 0 ? ` (${progressPct}%)` : "..."}
          </span>
        )}
        {result && (
          <span className="text-xs text-zinc-500">{result.elapsed}s · {result.num_frames} frames</span>
        )}
      </div>

      {isLoading && !hasAudio && progressPct > 0 && (
        <div className="bg-zinc-800 rounded-full h-1.5 overflow-hidden">
          <div
            className="bg-zinc-600 rounded-full h-full transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      )}

      <audio ref={audioRefA} preload="auto" style={{ display: "none" }} />
      <audio ref={audioRefB} preload="auto" style={{ display: "none" }} />

      {hasAudio && !showFinalPlayer && (
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            className="w-8 h-8 flex-shrink-0 flex items-center justify-center rounded-full bg-white text-black hover:bg-zinc-200 transition-colors"
          >
            {isPlaying ? (
              <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
                <rect x="0" y="0" width="3" height="12" />
                <rect x="7" y="0" width="3" height="12" />
              </svg>
            ) : (
              <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
                <polygon points="0,0 10,6 0,12" />
              </svg>
            )}
          </button>
          <div className="flex-1 bg-zinc-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-white rounded-full h-full transition-all duration-100"
              style={{ width: totalDuration > 0 ? `${Math.min(100, (displayTime / totalDuration) * 100)}%` : "0%" }}
            />
          </div>
          <span className="text-xs text-zinc-400 tabular-nums whitespace-nowrap">
            {formatTime(displayTime)} / {formatTime(totalDuration)}
          </span>
        </div>
      )}

      {showFinalPlayer && (
        <audio ref={finalAudioRef} controls className="w-full" />
      )}

      {isLoading && hasAudio && progressPct > 0 && (
        <div className="bg-zinc-800 rounded-full h-1 overflow-hidden">
          <div
            className="bg-zinc-600 rounded-full h-full transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      )}

      {status === "error" && (
        <p className="text-xs text-red-400">Generation failed</p>
      )}

      {result && (
        <p className="text-xs text-zinc-600 font-mono break-all">{result.tags}</p>
      )}
    </div>
  );
}

export default function RatePage() {
  const { user, guestId } = useAuth();
  const isSean = user?.email === SEAN_EMAIL;
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
  const [pairState, setPairState] = useState<"none" | "loading" | "ready" | "generating">("none");
  const [mode, setMode] = useState<"orig_vs_dpo" | "orig_vs_orig">("orig_vs_dpo");
  const [randomizingPrompt, setRandomizingPrompt] = useState(false);
  const [randomizingLyrics, setRandomizingLyrics] = useState(false);
  const [randomError, setRandomError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const [aState, setAState] = useState<StreamState>(EMPTY_STREAM);
  const [bState, setBState] = useState<StreamState>(EMPTY_STREAM);
  const aCleanupRef = useRef<(() => void) | null>(null);
  const bCleanupRef = useRef<(() => void) | null>(null);
  const fetchingNextRef = useRef(false);

  const loadBackendUrl = useCallback(async () => {
    const RUNPOD_URL = process.env.NEXT_PUBLIC_RUNPOD_URL;
    const MODAL_URL = "https://boatmaninlavik--erised-gpu-serve.modal.run";

    if (RUNPOD_URL) {
      try {
        const resp = await fetch(`${RUNPOD_URL}/health`, { signal: AbortSignal.timeout(3000) });
        if (resp.ok) {
          setBackendUrl(RUNPOD_URL);
          setGpuStatus("online");
          return;
        }
      } catch {}
    }

    try {
      const resp = await fetch(`${MODAL_URL}/health`, { signal: AbortSignal.timeout(5000) });
      if (resp.ok) {
        setBackendUrl(MODAL_URL);
        setGpuStatus("online");
        return;
      }
    } catch {}

    setGpuStatus("starting");
    try {
      const resp = await fetch(`${MODAL_URL}/health`, { signal: AbortSignal.timeout(120000) });
      if (resp.ok) {
        setBackendUrl(MODAL_URL);
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

  const stopStreams = useCallback(() => {
    if (aCleanupRef.current) { aCleanupRef.current(); aCleanupRef.current = null; }
    if (bCleanupRef.current) { bCleanupRef.current(); bCleanupRef.current = null; }
  }, []);

  const fetchNext = useCallback(async () => {
    if (!backendUrl || fetchingNextRef.current) return;
    fetchingNextRef.current = true;
    setPairState("loading");
    try {
      const resp = await fetch(`${backendUrl}/api/next`);
      const data = await resp.json();
      if (data.status === "ready") {
        const pair: Pair = data.pair;
        stopStreams();
        setAState({ ...EMPTY_STREAM, status: "pending" });
        setBState({ ...EMPTY_STREAM, status: "pending" });
        setCurrentPair(pair);
        setVoted(null);
        setSaved(false);
        setPairState("ready");
        aCleanupRef.current = streamJob(backendUrl, pair.a_job, (u) =>
          setAState((s) => ({ ...s, ...u }))
        );
        bCleanupRef.current = streamJob(backendUrl, pair.b_job, (u) =>
          setBState((s) => ({ ...s, ...u }))
        );
      } else if (data.status === "generating") {
        setPairState("generating");
      } else {
        setPairState("none");
      }
    } catch {
      setPairState("none");
    } finally {
      fetchingNextRef.current = false;
    }
  }, [backendUrl, stopStreams]);

  useEffect(() => {
    if (!backendUrl) return;
    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`${backendUrl}/api/status`);
        const data = await resp.json();
        setServerStatus(data);
        if (!currentPair && !fetchingNextRef.current && data.ready > 0) fetchNext();
      } catch {}
    }, 2000);
    return () => clearInterval(interval);
  }, [backendUrl, currentPair, fetchNext]);

  useEffect(() => () => { stopStreams(); }, [stopStreams]);

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
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, count, mode, user_email: user?.email || null }),
      });
      if (!currentPair) fetchNext();
    } finally {
      setQueueing(false);
    }
  }

  const bothDone = aState.status === "done" && bState.status === "done";

  async function vote(choice: "a" | "b") {
    if (!backendUrl || !currentPair || voted || !bothDone) return;
    setVoted(choice);
    try {
      const resp = await fetch(`${backendUrl}/api/rate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pair_id: currentPair.pair_id, choice, user_email: user?.email || null }),
      });
      const data = await resp.json();
      setServerStatus((s) => ({ ...s, rated: data.count ?? s.rated + 1 }));
    } catch {}
  }

  function goToNext() {
    stopStreams();
    setCurrentPair(null);
    setAState(EMPTY_STREAM);
    setBState(EMPTY_STREAM);
    setSaved(false);
    setVoted(null);
    fetchNext();
  }

  async function skipPair() {
    if (!backendUrl || !currentPair) return;
    const pid = currentPair.pair_id;
    try {
      await fetch(`${backendUrl}/api/pair/${pid}`, { method: "DELETE" });
    } catch {}
    goToNext();
  }

  async function saveWinnerToLibrary() {
    if (!currentPair || !voted) return;
    const winnerResult = voted === "a" ? aState.result : bState.result;
    if (!winnerResult) return;
    setSaving(true);
    try {
      const winnerAudio = winnerResult.audio_file;
      const winnerTags = winnerResult.tags;

      const audioResp = await fetch(`/api/proxy-audio?file=${encodeURIComponent(winnerAudio)}`);
      if (!audioResp.ok) {
        const errData = await audioResp.json().catch(() => ({ error: "Failed to fetch audio" }));
        throw new Error(errData.error || "Failed to fetch audio");
      }
      const blob = await audioResp.blob();
      const ext = winnerAudio.split(".").pop() || "wav";
      const filename = `${Date.now()}_rate_winner.${ext}`;

      const { data: uploadData, error: uploadErr } = await supabase.storage
        .from("dpo-songs")
        .upload(filename, blob, { contentType: blob.type || "audio/wav", upsert: false });

      if (uploadErr) throw uploadErr;

      const { data: urlData } = supabase.storage.from("dpo-songs").getPublicUrl(uploadData.path);

      const baseRow = {
        title: "Untitled",
        prompt: currentPair.prompt,
        lyrics: currentPair.lyrics || "",
        tags: winnerTags,
        audio_url: urlData.publicUrl,
        num_frames: null,
        model: "rate-winner",
      };

      const { error: insertErr } = await supabase.from("dpo-songs").insert({
        ...baseRow,
        guest_id: guestId,
        user_id: user?.id || null,
      });

      if (insertErr) {
        const { error: retryErr } = await supabase.from("dpo-songs").insert(baseRow);
        if (retryErr) throw retryErr;
      }
      setSaved(true);
    } catch (e: unknown) {
      console.error("Save failed:", e);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar>
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
      </Navbar>

      {(gpuStatus === "loading" || gpuStatus === "starting") && (
        <div className="flex items-center justify-center min-h-[60vh]">
          <p className="text-zinc-500 text-sm animate-pulse">
            {gpuStatus === "starting" ? "Setting up the GPUs needed in the background, this may take up to 1 minute..." : "Connecting to GPU..."}
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

          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 space-y-4">
            <h3 className="text-sm font-medium text-zinc-300">Queue new pairs</h3>
            {isSean && (
              <div>
                <label className="block text-xs text-zinc-500 mb-2 uppercase tracking-wide">Compare</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setMode("orig_vs_dpo")}
                    className={`flex-1 py-2 rounded-xl text-xs font-medium border transition-colors ${
                      mode === "orig_vs_dpo"
                        ? "bg-white text-black border-white"
                        : "bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-700"
                    }`}
                  >
                    Original vs DPO
                  </button>
                  <button
                    type="button"
                    onClick={() => setMode("orig_vs_orig")}
                    className={`flex-1 py-2 rounded-xl text-xs font-medium border transition-colors ${
                      mode === "orig_vs_orig"
                        ? "bg-white text-black border-white"
                        : "bg-zinc-800 text-zinc-300 border-zinc-700 hover:bg-zinc-700"
                    }`}
                  >
                    Original vs Original
                  </button>
                </div>
              </div>
            )}
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

          {pairState === "ready" && currentPair ? (
            <div className="space-y-4">
              <p className="text-xs text-zinc-500 text-center">
                {voted ? "Voted!" : bothDone ? "Which sounds better?" : "Listen as they generate..."}
              </p>
              <p className="text-xs text-zinc-600 text-center font-mono break-words overflow-wrap-anywhere">{currentPair.prompt}</p>

              <div
                className={`bg-zinc-900 border rounded-2xl p-5 space-y-3 transition-colors ${
                  voted === "a" ? "border-white" : "border-zinc-800"
                }`}
              >
                <SongCard
                  backendUrl={backendUrl!}
                  status={aState.status}
                  result={aState.result}
                  progress={aState.progress}
                  partialAudio={aState.partialAudio}
                  partialVersion={aState.partialVersion}
                  chunkFiles={aState.chunkFiles}
                  label="Option A"
                />
                <button
                  onClick={() => vote("a")}
                  disabled={!!voted || !bothDone}
                  className={`w-full py-3 rounded-xl text-sm font-medium transition-colors ${
                    voted === "a"
                      ? "bg-white text-black"
                      : !bothDone
                        ? "bg-zinc-800/50 text-zinc-600 cursor-not-allowed"
                        : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
                  }`}
                >
                  {!bothDone ? "Waiting for both to finish..." : voted === "a" ? "Winner" : "A is better"}
                </button>
              </div>

              <div
                className={`bg-zinc-900 border rounded-2xl p-5 space-y-3 transition-colors ${
                  voted === "b" ? "border-white" : "border-zinc-800"
                }`}
              >
                <SongCard
                  backendUrl={backendUrl!}
                  status={bState.status}
                  result={bState.result}
                  progress={bState.progress}
                  partialAudio={bState.partialAudio}
                  partialVersion={bState.partialVersion}
                  chunkFiles={bState.chunkFiles}
                  label="Option B"
                />
                <button
                  onClick={() => vote("b")}
                  disabled={!!voted || !bothDone}
                  className={`w-full py-3 rounded-xl text-sm font-medium transition-colors ${
                    voted === "b"
                      ? "bg-white text-black"
                      : !bothDone
                        ? "bg-zinc-800/50 text-zinc-600 cursor-not-allowed"
                        : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
                  }`}
                >
                  {!bothDone ? "Waiting for both to finish..." : voted === "b" ? "Winner" : "B is better"}
                </button>
              </div>

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

              {!voted && (
                <div className="pt-1">
                  <button
                    onClick={skipPair}
                    className="w-full py-2 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                  >
                    Skip this pair (audio broken or unplayable)
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
