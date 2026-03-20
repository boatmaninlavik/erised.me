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

/**
 * Connect to the SSE stream for a job. Falls back to polling if SSE fails.
 * Returns a cleanup function.
 */
function streamJob(
  backendUrl: string,
  jobId: string,
  onUpdate: (state: Partial<StreamState>) => void,
): () => void {
  let cancelled = false;

  // Try SSE first
  const es = new EventSource(`${backendUrl}/api/job/${jobId}/stream`);
  let sseWorking = false;

  es.addEventListener("progress", (e) => {
    sseWorking = true;
    const data = JSON.parse(e.data);
    onUpdate({ status: "running", progress: data });
  });

  es.addEventListener("chunk", (e) => {
    sseWorking = true;
    const data = JSON.parse(e.data);
    onUpdate({
      status: "running",
      partialAudio: data.audio_file,
      partialVersion: data.version,
      chunkFiles: data.chunk_files || null,
    });
  });

  es.addEventListener("done", (e) => {
    sseWorking = true;
    const data = JSON.parse(e.data);
    onUpdate({ status: "done", result: data });
    es.close();
  });

  es.addEventListener("error", (e) => {
    es.close();
    if (!sseWorking) {
      // SSE never worked — fall back to polling
      fallbackPoll();
      return;
    }
    // SSE was working but connection dropped (e.g. cloudflared timeout
    // during a long decode). Fall back to polling so we still catch the
    // done event instead of getting stuck at "Streaming 99%".
    fallbackPoll();
  });

  // Fallback polling (for Modal which doesn't have the SSE endpoint)
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
        if (data.chunk_paths) update.chunkFiles = data.chunk_paths;
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

/**
 * Song player with seamless chunk transitions using Web Audio API.
 *
 * Each individual chunk WAV is fetched, decoded as an AudioBuffer, and
 * scheduled to play exactly when the previous chunk ends — sample-accurate
 * gapless playback with zero source-swapping.
 *
 * totalDuration grows as soon as each chunk is decoded, so the progress bar
 * extends (e.g. 0:15/0:24 → 0:15/0:44) BEFORE the current chunk finishes.
 *
 * If all buffered audio plays out before the next chunk arrives, the
 * AudioContext is suspended (paused) and automatically resumed when new
 * audio is decoded — no gap, no reset to 0:00.
 *
 * After generation completes, switches to native <audio controls> for
 * scrubbing/replay.
 */
function SongCard({
  backendUrl,
  status,
  result,
  progress,
  partialAudio,
  partialVersion,
  chunkFiles,
  label,
  onSave,
  saving,
  saved,
}: {
  backendUrl: string;
  status: JobStatus;
  result: GenerationResult | null;
  progress: { current_frame: number; total_frames: number } | null;
  partialAudio: string | null;
  partialVersion: number;
  chunkFiles: string[] | null;
  label?: string;
  onSave?: () => void;
  saving?: boolean;
  saved?: boolean;
}) {
  // Web Audio API refs
  const ctxRef = useRef<AudioContext | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const nextTimeRef = useRef(0);       // When the next buffer should start (ctx time)
  const startCtxTimeRef = useRef(0);   // ctx.currentTime when playback first started
  const totalDurRef = useRef(0);       // Total buffered audio duration
  const loadedIdxRef = useRef(0);      // How many chunk files we've loaded
  const startedRef = useRef(false);    // Whether playback has started
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const queueRef = useRef<Promise<void>>(Promise.resolve());
  const statusRef = useRef(status);
  statusRef.current = status;

  const finalAudioRef = useRef<HTMLAudioElement>(null);

  const [totalDuration, setTotalDuration] = useState(0);
  const [displayTime, setDisplayTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasAudio, setHasAudio] = useState(false);
  const [showFinalPlayer, setShowFinalPlayer] = useState(false);

  // Reset on new generation
  useEffect(() => {
    if (status !== "pending") return;
    if (ctxRef.current) { ctxRef.current.close().catch(() => {}); ctxRef.current = null; }
    gainRef.current = null;
    nextTimeRef.current = 0;
    startCtxTimeRef.current = 0;
    totalDurRef.current = 0;
    loadedIdxRef.current = 0;
    startedRef.current = false;
    queueRef.current = Promise.resolve();
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setTotalDuration(0);
    setDisplayTime(0);
    setIsPlaying(false);
    setHasAudio(false);
    setShowFinalPlayer(false);
  }, [status]);

  // Cleanup on unmount
  useEffect(() => () => {
    if (ctxRef.current) ctxRef.current.close().catch(() => {});
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  // Start display-time tracking interval
  const startTimer = useCallback(() => {
    if (timerRef.current) return;
    timerRef.current = setInterval(() => {
      const ctx = ctxRef.current;
      if (!ctx) return;
      const elapsed = ctx.currentTime - startCtxTimeRef.current;
      setDisplayTime(Math.min(elapsed, totalDurRef.current));
      // Auto-suspend when all buffered audio has played but stream is still going
      if (
        elapsed >= totalDurRef.current - 0.05 &&
        ctx.state === "running" &&
        statusRef.current === "running"
      ) {
        ctx.suspend();
      }
    }, 50);
  }, []);

  // Process new chunk files via Web Audio API — gapless scheduling
  useEffect(() => {
    if (result) return;

    // Determine new files to load
    let filesToLoad: string[] = [];

    if (chunkFiles && chunkFiles.length > loadedIdxRef.current) {
      filesToLoad = chunkFiles.slice(loadedIdxRef.current);
      loadedIdxRef.current = chunkFiles.length;
    } else if (loadedIdxRef.current === 0 && partialAudio && partialVersion > 0) {
      // Fallback: no individual chunk files, use cumulative file for first chunk
      filesToLoad = [partialAudio];
      loadedIdxRef.current = 1;
    }

    if (filesToLoad.length === 0) return;

    // Init audio context
    if (!ctxRef.current) {
      const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      ctxRef.current = new AC();
      gainRef.current = ctxRef.current.createGain();
      gainRef.current.connect(ctxRef.current.destination);
    }
    const ctx = ctxRef.current;
    const gain = gainRef.current!;

    // Queue chunk loading sequentially to preserve ordering
    for (const file of filesToLoad) {
      queueRef.current = queueRef.current.then(async () => {
        try {
          const resp = await fetch(`${backendUrl}/audio/${file}`);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          const ab = await resp.arrayBuffer();
          const buf = await ctx.decodeAudioData(ab);

          totalDurRef.current += buf.duration;
          setTotalDuration(totalDurRef.current);
          setHasAudio(true);

          // First chunk: kick off playback
          if (!startedRef.current) {
            startedRef.current = true;
            if (ctx.state === "suspended") await ctx.resume();
            startCtxTimeRef.current = ctx.currentTime;
            nextTimeRef.current = ctx.currentTime;
            setIsPlaying(true);
            startTimer();
          } else if (ctx.state === "suspended") {
            // Was auto-suspended due to buffer underrun — resume
            await ctx.resume();
            setIsPlaying(true);
          }

          // Schedule buffer for gapless playback
          const src = ctx.createBufferSource();
          src.buffer = buf;
          src.connect(gain);
          const t = Math.max(nextTimeRef.current, ctx.currentTime);
          src.start(t);
          nextTimeRef.current = t + buf.duration;
        } catch (err) {
          console.error("Chunk load/decode error:", err);
        }
      });
    }
  }, [partialAudio, partialVersion, chunkFiles, result, backendUrl, startTimer]);

  // Switch to final player when generation completes
  useEffect(() => {
    if (!result) return;

    // Get current playback position
    let pos = 0;
    if (ctxRef.current && startedRef.current) {
      pos = ctxRef.current.currentTime - startCtxTimeRef.current;
      pos = Math.min(pos, totalDurRef.current);
    }

    // Tear down Web Audio
    if (ctxRef.current) { ctxRef.current.close().catch(() => {}); ctxRef.current = null; }
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setIsPlaying(false);
    setShowFinalPlayer(true);

    // Load final file with native controls
    const fin = finalAudioRef.current;
    if (!fin) return;
    const onReady = () => {
      fin.removeEventListener("canplay", onReady);
      if (pos > 0 && pos < fin.duration) fin.currentTime = pos;
      fin.play().catch(() => {});
    };
    fin.addEventListener("canplay", onReady);
    fin.src = `${backendUrl}/audio/${result.audio_file}`;
    fin.load();
  }, [result, backendUrl]);

  const togglePlay = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (ctx.state === "running") {
      ctx.suspend();
      setIsPlaying(false);
    } else {
      ctx.resume();
      setIsPlaying(true);
    }
  }, []);

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

  if (status === "idle") return null;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        {label && <h3 className="font-medium text-sm tracking-tight">{label}</h3>}
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

      {/* Custom streaming player — Web Audio API gapless playback */}
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

      {/* Native player after generation completes */}
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
        <>
          <p className="text-xs text-zinc-600 font-mono break-all">{result.tags}</p>
          {onSave && (
            <button
              onClick={onSave}
              disabled={saving || saved}
              className="text-xs text-zinc-400 hover:text-white transition-colors disabled:opacity-40"
            >
              {saved ? "Saved" : saving ? "Saving..." : "Save to My Library"}
            </button>
          )}
        </>
      )}
    </div>
  );
}

export default function GeneratePage() {
  const { user, guestId } = useAuth();
  const isSean = user?.email === SEAN_EMAIL;
  const [backendUrl, setBackendUrl] = useState<string | null>(null);
  const [gpuStatus, setGpuStatus] = useState<"loading" | "online" | "offline" | "starting">("loading");
  const [prompt, setPrompt] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [maxSec, setMaxSec] = useState(60);
  const [dpoScale, setDpoScale] = useState(3.0);
  const [tab, setTab] = useState<"ab" | "single">("ab");
  const [selectedModel, setSelectedModel] = useState<"dpo" | "original">("dpo");

  const [origState, setOrigState] = useState<StreamState>({
    status: "idle", result: null, progress: null, partialAudio: null, partialVersion: 0, chunkFiles: null,
  });
  const [dpoState, setDpoState] = useState<StreamState>({
    status: "idle", result: null, progress: null, partialAudio: null, partialVersion: 0, chunkFiles: null,
  });
  const [singleState, setSingleState] = useState<StreamState>({
    status: "idle", result: null, progress: null, partialAudio: null, partialVersion: 0, chunkFiles: null,
  });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const [savedModels, setSavedModels] = useState<Set<string>>(new Set());
  const [randomizingPrompt, setRandomizingPrompt] = useState(false);
  const [randomizingLyrics, setRandomizingLyrics] = useState(false);
  const [songTitle, setSongTitle] = useState("Untitled");

  // Keep cleanup refs for SSE connections
  const cleanupRef = useRef<(() => void)[]>([]);

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

  // Cleanup SSE on unmount
  useEffect(() => () => {
    cleanupRef.current.forEach((fn) => fn());
  }, []);

  async function generateBoth() {
    if (!backendUrl || !prompt.trim() || !lyrics.trim()) return;
    // Cancel any existing streams
    cleanupRef.current.forEach((fn) => fn());
    cleanupRef.current = [];

    setError(null);
    setSavedModels(new Set());
    const blank: StreamState = { status: "pending", result: null, progress: null, partialAudio: null, partialVersion: 0, chunkFiles: null };
    setOrigState(blank);
    setDpoState({ ...blank, status: "idle" });

    try {
      // Generate Original first
      const origResp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: "original", user_email: user?.email || null }),
      });
      const origJob = await origResp.json();
      setOrigState((s) => ({ ...s, status: "running" }));

      // Wait for original via SSE, then start DPO
      await new Promise<void>((resolve, reject) => {
        const cleanup = streamJob(backendUrl, origJob.job_id, (update) => {
          setOrigState((s) => ({ ...s, ...update }));
          if (update.status === "done") resolve();
          if (update.status === "error") reject(new Error("Generation failed"));
        });
        cleanupRef.current.push(cleanup);
      });

      // Now generate DPO
      setDpoState({ ...blank, status: "pending" });
      const dpoResp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: isSean ? "dpo" : "original", dpo_scale: dpoScale, user_email: user?.email || null }),
      });
      const dpoJob = await dpoResp.json();
      setDpoState((s) => ({ ...s, status: "running" }));

      await new Promise<void>((resolve, reject) => {
        const cleanup = streamJob(backendUrl, dpoJob.job_id, (update) => {
          setDpoState((s) => ({ ...s, ...update }));
          if (update.status === "done") resolve();
          if (update.status === "error") reject(new Error("Generation failed"));
        });
        cleanupRef.current.push(cleanup);
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Generation failed");
    }
  }

  async function generateSingle() {
    if (!backendUrl || !prompt.trim() || !lyrics.trim()) return;
    cleanupRef.current.forEach((fn) => fn());
    cleanupRef.current = [];

    setError(null);
    setSavedModels(new Set());
    const blank: StreamState = { status: "pending", result: null, progress: null, partialAudio: null, partialVersion: 0, chunkFiles: null };
    setSingleState(blank);

    try {
      const effectiveModel = isSean ? selectedModel : "original";
      const resp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: effectiveModel, dpo_scale: dpoScale, user_email: user?.email || null }),
      });
      const job = await resp.json();
      setSingleState((s) => ({ ...s, status: "running" }));

      await new Promise<void>((resolve, reject) => {
        const cleanup = streamJob(backendUrl, job.job_id, (update) => {
          setSingleState((s) => ({ ...s, ...update }));
          if (update.status === "done") resolve();
          if (update.status === "error") reject(new Error("Generation failed"));
        });
        cleanupRef.current.push(cleanup);
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setSingleState((s) => ({ ...s, status: "error" }));
    }
  }

  async function saveToLibrary(result: GenerationResult) {
    setSaving(result.model);
    try {
      const audioResp = await fetch(`/api/proxy-audio?file=${encodeURIComponent(result.audio_file)}`);
      if (!audioResp.ok) {
        const errData = await audioResp.json().catch(() => ({ error: "Failed to fetch audio" }));
        throw new Error(errData.error || "Failed to fetch audio");
      }
      const blob = await audioResp.blob();
      const ext = result.audio_file.split(".").pop() || "wav";
      const filename = `${Date.now()}_${result.model}.${ext}`;

      const { data: uploadData, error: uploadErr } = await supabase.storage
        .from("dpo-songs")
        .upload(filename, blob, { contentType: blob.type || "audio/wav", upsert: false });

      if (uploadErr) throw uploadErr;

      const { data: urlData } = supabase.storage.from("dpo-songs").getPublicUrl(uploadData.path);

      const baseRow = {
        title: songTitle,
        prompt,
        lyrics,
        tags: result.tags,
        audio_url: urlData.publicUrl,
        num_frames: result.num_frames,
        model: result.model,
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

      setSavedModels((prev) => new Set(prev).add(result.model));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(null);
    }
  }

  async function randomize(type: "prompt" | "lyrics") {
    const setLoading = type === "prompt" ? setRandomizingPrompt : setRandomizingLyrics;
    const setValue = type === "prompt" ? setPrompt : setLyrics;
    setLoading(true);
    try {
      const resp = await fetch("/api/generate-random", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, context: type === "lyrics" ? prompt : undefined }),
      });
      const data = await resp.json();
      if (data.text) {
        setValue(data.text);
        if (type === "lyrics" && data.title) {
          setSongTitle(data.title);
        }
      } else if (data.error) setError(data.error);
    } catch {
      setError("Failed to generate random " + type);
    } finally {
      setLoading(false);
    }
  }

  const effectiveTab = isSean ? tab : "single";

  const isGenerating =
    origState.status === "pending" || origState.status === "running" ||
    dpoState.status === "pending" || dpoState.status === "running" ||
    singleState.status === "pending" || singleState.status === "running";

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
            <h2 className="text-2xl font-semibold tracking-tight">Generate</h2>
            <p className="text-zinc-500 text-sm mt-1">
              {isSean ? "Compare original vs DPO Guided model output." : "Generate music with Erised."}
            </p>
          </div>

          {isSean ? (
            <div className="flex gap-1 bg-zinc-900 rounded-xl p-1">
              {(["ab", "single"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={`flex-1 py-2 text-sm font-medium rounded-lg transition-colors ${
                    tab === t ? "bg-zinc-700 text-white" : "text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {t === "ab" ? "A/B Compare" : "Single Generate"}
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-zinc-600">
              Rate songs to start training your personalized model.
            </p>
          )}

          <div className="space-y-4">
            {tab === "single" && isSean && (
              <div className="flex gap-2">
                {(["original", "dpo"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setSelectedModel(m)}
                    className={`flex-1 py-2.5 rounded-xl text-sm font-medium border transition-colors ${
                      selectedModel === m
                        ? "border-white text-white bg-zinc-800"
                        : "border-zinc-800 text-zinc-500 hover:border-zinc-600"
                    }`}
                  >
                    {m === "dpo" ? "DPO Guided" : "Original"}
                  </button>
                ))}
              </div>
            )}

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-zinc-400 font-medium tracking-wide uppercase">
                  Musical Prompt
                </label>
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
                rows={3}
                placeholder="e.g. emotional pop ballad with piano and strings"
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-white text-sm placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 resize-none"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-zinc-400 font-medium tracking-wide uppercase">
                  Lyrics
                </label>
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
                rows={8}
                placeholder={"[Verse 1]\nYour lyrics here...\n\n[Chorus]\nYour chorus here..."}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-white text-sm placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600 resize-none font-mono"
              />
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-2 font-medium tracking-wide uppercase">
                Max length — {maxSec}s
              </label>
              <input
                type="range"
                min={10}
                max={240}
                step={5}
                value={maxSec}
                onChange={(e) => setMaxSec(Number(e.target.value))}
                className="w-full accent-white"
              />
            </div>

            {isSean && (
              <div>
                <label className="block text-xs text-zinc-400 mb-2 font-medium tracking-wide uppercase">
                  DPO Taste Influence — {dpoScale.toFixed(1)}
                </label>
                <p className="text-xs text-zinc-600 mb-2">
                  How much should the DPO model be influenced by your taste? 0 = no influence, 3 = max.
                </p>
                <input
                  type="range"
                  min={0}
                  max={3}
                  step={0.1}
                  value={dpoScale}
                  onChange={(e) => setDpoScale(Number(e.target.value))}
                  className="w-full accent-white"
                />
              </div>
            )}

            <button
              onClick={effectiveTab === "ab" ? generateBoth : generateSingle}
              disabled={isGenerating || !prompt.trim() || !lyrics.trim()}
              className="w-full py-4 bg-white text-black font-semibold rounded-xl text-sm tracking-tight hover:bg-zinc-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isGenerating
                ? "Generating..."
                : !isSean
                ? "Generate"
                : tab === "ab"
                ? "Generate Both (Original + DPO Guided)"
                : `Generate with ${selectedModel === "dpo" ? "DPO Guided" : "Original"} model`}
            </button>
          </div>

          {error && (
            <p className="text-red-400 text-sm bg-red-950/30 border border-red-900 rounded-xl px-4 py-3">
              {error}
            </p>
          )}

          {/* A/B Results */}
          {effectiveTab === "ab" && (origState.status !== "idle" || dpoState.status !== "idle") && backendUrl && (
            <div className="space-y-4">
              <SongCard
                backendUrl={backendUrl}
                status={origState.status}
                result={origState.result}
                progress={origState.progress}
                partialAudio={origState.partialAudio}
                partialVersion={origState.partialVersion}
                chunkFiles={origState.chunkFiles}
                label="Original Model"
                onSave={origState.result ? () => saveToLibrary(origState.result!) : undefined}
                saving={saving === "original"}
                saved={savedModels.has("original")}
              />
              <SongCard
                backendUrl={backendUrl}
                status={dpoState.status}
                result={dpoState.result}
                progress={dpoState.progress}
                partialAudio={dpoState.partialAudio}
                partialVersion={dpoState.partialVersion}
                chunkFiles={dpoState.chunkFiles}
                label="DPO Guided"
                onSave={dpoState.result ? () => saveToLibrary(dpoState.result!) : undefined}
                saving={saving === "dpo"}
                saved={savedModels.has("dpo")}
              />
            </div>
          )}

          {/* Single Result */}
          {effectiveTab === "single" && singleState.status !== "idle" && backendUrl && (
            <SongCard
              backendUrl={backendUrl}
              status={singleState.status}
              result={singleState.result}
              progress={singleState.progress}
              partialAudio={singleState.partialAudio}
              partialVersion={singleState.partialVersion}
              chunkFiles={singleState.chunkFiles}
              onSave={singleState.result ? () => saveToLibrary(singleState.result!) : undefined}
              saving={saving === selectedModel}
              saved={savedModels.has(selectedModel)}
            />
          )}
        </div>
      )}
    </div>
  );
}
