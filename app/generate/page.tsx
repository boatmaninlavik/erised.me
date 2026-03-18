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

interface Progress {
  current_frame: number;
  total_frames: number;
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

async function pollJob(
  backendUrl: string,
  jobId: string,
  onProgress?: (progress: Progress, partialAudio: string | null, partialVersion: number | null) => void,
): Promise<GenerationResult> {
  while (true) {
    await new Promise((r) => setTimeout(r, 2000));
    const resp = await fetchRetry(`${backendUrl}/api/job/${jobId}`);
    const data = await resp.json();
    if (data.status === "done") return data.result;
    if (data.status === "error") throw new Error(data.error);
    if (onProgress && data.progress) {
      onProgress(data.progress, data.partial_audio_file || null, data.partial_version ?? null);
    }
  }
}

const SEAN_EMAIL = "zsean@berkeley.edu";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Single song player card — streams audio chunks seamlessly via Web Audio API.
 *
 * Instead of downloading the entire growing WAV file and switching <audio> src
 * (which causes gaps), this fetches individual chunk files and schedules them
 * with sample-accurate timing using AudioBufferSourceNode.start(exactTime).
 * Once generation completes, switches to a standard <audio> element for full
 * controls (seeking, scrubbing).
 */
function SongCard({
  backendUrl,
  status,
  result,
  progress,
  partialAudio,
  partialVersion,
  label,
  onSave,
  saving,
  saved,
}: {
  backendUrl: string;
  status: JobStatus;
  result: GenerationResult | null;
  progress: Progress | null;
  partialAudio: string | null;
  partialVersion: number;
  label?: string;
  onSave?: () => void;
  saving?: boolean;
  saved?: boolean;
}) {
  // ── Web Audio API streaming state ──
  const audioCtxRef = useRef<AudioContext | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const scheduledRef = useRef<AudioBufferSourceNode[]>([]);
  const nextTimeRef = useRef(0);
  const loadedChunksRef = useRef(0);
  const totalBufferedRef = useRef(0);
  const playbackStartRef = useRef(0);
  const streamStartedRef = useRef(false);
  const loadingRef = useRef(false);
  const chunksFailedRef = useRef(false); // true = backend doesn't support chunks, use legacy

  const [streamStarted, setStreamStarted] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [bufferedDuration, setBufferedDuration] = useState(0);

  // ── Legacy fallback: full-file <audio> with src switching ──
  const [legacyMode, setLegacyMode] = useState(false);
  const [legacySrc, setLegacySrc] = useState<string | null>(null);
  const legacyVersionRef = useRef(0);
  const legacySeekRef = useRef(0);

  // ── Final audio (after generation completes) ──
  const audioRef = useRef<HTMLAudioElement>(null);
  const [useFinalPlayer, setUseFinalPlayer] = useState(false);
  const finalSeekRef = useRef(0);

  // Reset all streaming state when a new generation starts
  useEffect(() => {
    if (status === "pending") {
      scheduledRef.current.forEach((s) => { try { s.stop(); } catch { /* already stopped */ } });
      audioCtxRef.current?.close().catch(() => {});
      audioCtxRef.current = null;
      gainRef.current = null;
      scheduledRef.current = [];
      nextTimeRef.current = 0;
      loadedChunksRef.current = 0;
      totalBufferedRef.current = 0;
      playbackStartRef.current = 0;
      streamStartedRef.current = false;
      loadingRef.current = false;
      chunksFailedRef.current = false;
      setStreamStarted(false);
      setIsPlaying(false);
      setCurrentTime(0);
      setBufferedDuration(0);
      setLegacyMode(false);
      setLegacySrc(null);
      legacyVersionRef.current = 0;
      legacySeekRef.current = 0;
      setUseFinalPlayer(false);
      finalSeekRef.current = 0;
    }
  }, [status]);

  // Load a single chunk WAV, decode it, and schedule for gapless playback
  const loadChunk = useCallback(async (url: string) => {
    let ctx = audioCtxRef.current;
    if (!ctx) {
      ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const gain = ctx.createGain();
      gain.connect(ctx.destination);
      gainRef.current = gain;
    }
    if (ctx.state === "suspended") await ctx.resume();

    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Chunk fetch failed: ${resp.status}`);
    const buf = await resp.arrayBuffer();
    const audio = await ctx.decodeAudioData(buf);

    const src = ctx.createBufferSource();
    src.buffer = audio;
    src.connect(gainRef.current!);

    // Schedule to start exactly when the previous chunk ends
    const startAt = Math.max(nextTimeRef.current, ctx.currentTime);
    src.start(startAt);
    nextTimeRef.current = startAt + audio.duration;
    totalBufferedRef.current += audio.duration;
    scheduledRef.current.push(src);

    if (!streamStartedRef.current) {
      playbackStartRef.current = startAt;
      streamStartedRef.current = true;
      setStreamStarted(true);
      setIsPlaying(true);
    }

    setBufferedDuration(totalBufferedRef.current);
  }, []);

  // Fetch new chunk files when partialVersion increases
  // Falls back to legacy (full-file) mode if chunk0 returns 404
  useEffect(() => {
    if (!partialAudio || partialVersion <= 0 || useFinalPlayer || legacyMode) return;
    if (chunksFailedRef.current) return;
    if (loadedChunksRef.current >= partialVersion) return;

    const load = async () => {
      if (loadingRef.current) return;
      loadingRef.current = true;
      try {
        const base = partialAudio.replace(/\.[^.]+$/, "");
        const ext = partialAudio.split(".").pop() || "wav";
        while (loadedChunksRef.current < partialVersion) {
          const i = loadedChunksRef.current;
          try {
            await loadChunk(`${backendUrl}/audio/${base}_chunk${i}.${ext}`);
            loadedChunksRef.current = i + 1;
          } catch {
            if (i === 0) {
              // chunk0 failed — backend doesn't support chunks, use legacy mode
              chunksFailedRef.current = true;
              // Clean up any AudioContext we created
              audioCtxRef.current?.close().catch(() => {});
              audioCtxRef.current = null;
              setLegacyMode(true);
            }
            break;
          }
        }
      } finally {
        loadingRef.current = false;
      }
    };
    load();
  }, [partialAudio, partialVersion, backendUrl, loadChunk, useFinalPlayer, legacyMode]);

  // ── Legacy mode: load the full partial audio file ──
  useEffect(() => {
    if (!legacyMode || !partialAudio || useFinalPlayer) return;
    if (partialVersion <= legacyVersionRef.current) return;
    const newSrc = `${backendUrl}/audio/${partialAudio}?v=${partialVersion}`;
    // Preload in browser cache
    fetch(newSrc).catch(() => {});
    const el = audioRef.current;
    if (!el) {
      // First load
      legacyVersionRef.current = partialVersion;
      setLegacySrc(newSrc);
      return;
    }
    // Switch when near end of current audio or paused/ended
    const shouldSwitch = el.paused || el.ended ||
      (isFinite(el.duration) && el.duration - el.currentTime < 2);
    if (shouldSwitch) {
      legacySeekRef.current = el.currentTime || 0;
      legacyVersionRef.current = partialVersion;
      el.pause();
      setLegacySrc(newSrc);
    }
  }, [legacyMode, partialAudio, partialVersion, backendUrl, useFinalPlayer]);

  // Legacy mode: poll for switch opportunity
  useEffect(() => {
    if (!legacyMode || useFinalPlayer) return;
    if (status !== "running" && status !== "pending") return;
    const iv = setInterval(() => {
      const el = audioRef.current;
      if (!el || !partialAudio) return;
      if (partialVersion <= legacyVersionRef.current) return;
      const shouldSwitch = el.paused || el.ended ||
        (isFinite(el.duration) && el.duration - el.currentTime < 2);
      if (shouldSwitch) {
        legacySeekRef.current = el.currentTime || 0;
        legacyVersionRef.current = partialVersion;
        el.pause();
        setLegacySrc(`${backendUrl}/audio/${partialAudio}?v=${partialVersion}`);
      }
    }, 500);
    return () => clearInterval(iv);
  }, [legacyMode, status, partialVersion, partialAudio, backendUrl, useFinalPlayer]);

  // Update playback time for UI
  useEffect(() => {
    if (!streamStarted || useFinalPlayer) return;
    const iv = setInterval(() => {
      const ctx = audioCtxRef.current;
      if (ctx && ctx.state === "running") {
        setCurrentTime(ctx.currentTime - playbackStartRef.current);
      }
    }, 200);
    return () => clearInterval(iv);
  }, [streamStarted, useFinalPlayer]);

  // Pause / resume via AudioContext.suspend/resume
  const togglePlay = useCallback(async () => {
    const ctx = audioCtxRef.current;
    if (!ctx) return;
    if (ctx.state === "running") {
      await ctx.suspend();
      setIsPlaying(false);
    } else {
      await ctx.resume();
      setIsPlaying(true);
    }
  }, []);

  // Switch to regular <audio> element when generation completes (for seeking/scrubbing)
  useEffect(() => {
    if (!result || useFinalPlayer) return;
    const ctx = audioCtxRef.current;
    const pos = ctx ? ctx.currentTime - playbackStartRef.current : 0;

    scheduledRef.current.forEach((s) => { try { s.stop(); } catch { /* already stopped */ } });
    if (ctx) ctx.close().catch(() => {});
    audioCtxRef.current = null;

    finalSeekRef.current = legacyMode ? 0 : pos;
    setUseFinalPlayer(true);
  }, [result, useFinalPlayer, legacyMode]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      scheduledRef.current.forEach((s) => { try { s.stop(); } catch { /* noop */ } });
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  // ── Render ──
  const isLoading = status === "pending" || status === "running";
  const progressPct = progress && progress.total_frames > 0
    ? Math.round((progress.current_frame / progress.total_frames) * 100)
    : 0;

  const isComposing = isLoading && !partialAudio;
  const isStreaming = isLoading && !!partialAudio;

  if (status === "idle") return null;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 space-y-3">
      {label && (
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-sm tracking-tight">{label}</h3>
          {isComposing && (
            <span className="text-xs text-zinc-500 animate-pulse">
              Composing{progressPct > 0 ? ` (${progressPct}%)` : "..."}
            </span>
          )}
          {isStreaming && (
            <span className="text-xs text-zinc-500 animate-pulse">
              Streaming{progressPct > 0 ? ` (${progressPct}%)` : "..."}
            </span>
          )}
        </div>
      )}
      {!label && isComposing && (
        <span className="text-xs text-zinc-500 animate-pulse">
          Composing{progressPct > 0 ? ` (${progressPct}%)` : "..."}
        </span>
      )}
      {!label && isStreaming && (
        <span className="text-xs text-zinc-500 animate-pulse">
          Streaming{progressPct > 0 ? ` (${progressPct}%)` : "..."}
        </span>
      )}

      {/* Streaming player — Web Audio API with gapless scheduling */}
      {!useFinalPlayer && !legacyMode && streamStarted && (
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            className="w-8 h-8 flex items-center justify-center rounded-full bg-zinc-800 text-white hover:bg-zinc-700 transition-colors text-xs shrink-0"
          >
            {isPlaying ? "\u23F8" : "\u25B6"}
          </button>
          <div className="flex-1 bg-zinc-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-white rounded-full h-full transition-all duration-200"
              style={{
                width: `${bufferedDuration > 0 ? Math.min(100, (currentTime / bufferedDuration) * 100) : 0}%`,
              }}
            />
          </div>
          <span className="text-xs text-zinc-500 tabular-nums min-w-[5rem] text-right">
            {formatTime(currentTime)} / {formatTime(bufferedDuration)}
          </span>
        </div>
      )}

      {/* Legacy player — full-file <audio> when backend doesn't support chunks */}
      {!useFinalPlayer && legacyMode && legacySrc && (
        <audio
          ref={audioRef}
          controls
          src={legacySrc}
          className="w-full"
          onCanPlay={() => {
            const el = audioRef.current;
            if (!el) return;
            if (legacySeekRef.current > 0) {
              el.currentTime = legacySeekRef.current;
              legacySeekRef.current = 0;
            }
            el.play().catch(() => {});
          }}
        />
      )}

      {/* Final player — standard <audio> controls after generation completes */}
      {useFinalPlayer && result && (
        <audio
          ref={!legacyMode ? audioRef : undefined}
          controls
          src={`${backendUrl}/audio/${result.audio_file}`}
          className="w-full"
          onCanPlay={(e) => {
            const el = e.currentTarget;
            if (finalSeekRef.current > 0) {
              el.currentTime = finalSeekRef.current;
              finalSeekRef.current = 0;
            }
            el.play().catch(() => {});
          }}
        />
      )}

      {status === "error" && (
        <p className="text-xs text-red-400">Generation failed</p>
      )}

      {/* Tags + save — only when done */}
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

  const [origStatus, setOrigStatus] = useState<JobStatus>("idle");
  const [dpoStatus, setDpoStatus] = useState<JobStatus>("idle");
  const [singleStatus, setSingleStatus] = useState<JobStatus>("idle");
  const [origResult, setOrigResult] = useState<GenerationResult | null>(null);
  const [dpoResult, setDpoResult] = useState<GenerationResult | null>(null);
  const [singleResult, setSingleResult] = useState<GenerationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const [savedModels, setSavedModels] = useState<Set<string>>(new Set());
  const [randomizingPrompt, setRandomizingPrompt] = useState(false);
  const [randomizingLyrics, setRandomizingLyrics] = useState(false);
  const [songTitle, setSongTitle] = useState("Untitled");

  // Streaming state
  const [origProgress, setOrigProgress] = useState<Progress | null>(null);
  const [dpoProgress, setDpoProgress] = useState<Progress | null>(null);
  const [singleProgress, setSingleProgress] = useState<Progress | null>(null);
  const [origPartialAudio, setOrigPartialAudio] = useState<string | null>(null);
  const [dpoPartialAudio, setDpoPartialAudio] = useState<string | null>(null);
  const [singlePartialAudio, setSinglePartialAudio] = useState<string | null>(null);
  const [origPartialVersion, setOrigPartialVersion] = useState(0);
  const [dpoPartialVersion, setDpoPartialVersion] = useState(0);
  const [singlePartialVersion, setSinglePartialVersion] = useState(0);

  const loadBackendUrl = useCallback(async () => {
    // Try RunPod first (fast 3s check), fall back to Modal
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

    // Fall back to Modal
    try {
      const resp = await fetch(`${MODAL_URL}/health`, { signal: AbortSignal.timeout(5000) });
      if (resp.ok) {
        setBackendUrl(MODAL_URL);
        setGpuStatus("online");
        return;
      }
    } catch {}

    // Modal cold-starting
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

  async function generateBoth() {
    if (!backendUrl || !prompt.trim() || !lyrics.trim()) return;
    setError(null);
    setOrigResult(null);
    setDpoResult(null);
    setSavedModels(new Set());
    setOrigStatus("pending");
    setDpoStatus("idle");
    setOrigProgress(null);
    setDpoProgress(null);
    setOrigPartialAudio(null);
    setDpoPartialAudio(null);
    setOrigPartialVersion(0);
    setDpoPartialVersion(0);

    try {
      // ── Generate Original first ──
      const origResp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: "original", user_email: user?.email || null }),
      });
      const origJob = await origResp.json();
      setOrigStatus("running");

      const origResult = await pollJob(backendUrl, origJob.job_id, (prog, partial, ver) => {
        setOrigProgress(prog);
        if (partial) setOrigPartialAudio(partial);
        if (ver !== null) setOrigPartialVersion(ver);
      });
      setOrigResult(origResult);
      setOrigStatus("done");

      // ── Then generate DPO ──
      setDpoStatus("pending");
      const dpoResp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: isSean ? "dpo" : "original", dpo_scale: dpoScale, user_email: user?.email || null }),
      });
      const dpoJob = await dpoResp.json();
      setDpoStatus("running");

      const dpoResult = await pollJob(backendUrl, dpoJob.job_id, (prog, partial, ver) => {
        setDpoProgress(prog);
        if (partial) setDpoPartialAudio(partial);
        if (ver !== null) setDpoPartialVersion(ver);
      });
      setDpoResult(dpoResult);
      setDpoStatus("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Generation failed");
    }
  }

  async function generateSingle() {
    if (!backendUrl || !prompt.trim() || !lyrics.trim()) return;
    setError(null);
    setSingleResult(null);
    setSavedModels(new Set());
    setSingleStatus("pending");
    setSingleProgress(null);
    setSinglePartialAudio(null);
    setSinglePartialVersion(0);

    try {
      const effectiveModel = isSean ? selectedModel : "original";
      const resp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: effectiveModel, dpo_scale: dpoScale, user_email: user?.email || null }),
      });
      const job = await resp.json();
      setSingleStatus("running");
      const result = await pollJob(backendUrl, job.job_id, (prog, partial, ver) => {
        setSingleProgress(prog);
        if (partial) setSinglePartialAudio(partial);
        if (ver !== null) setSinglePartialVersion(ver);
      });
      setSingleResult(result);
      setSingleStatus("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setSingleStatus("error");
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
    origStatus === "pending" || origStatus === "running" ||
    dpoStatus === "pending" || dpoStatus === "running" ||
    singleStatus === "pending" || singleStatus === "running";

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
          {effectiveTab === "ab" && (origStatus !== "idle" || dpoStatus !== "idle") && backendUrl && (
            <div className="space-y-4">
              <SongCard
                backendUrl={backendUrl}
                status={origStatus}
                result={origResult}
                progress={origProgress}
                partialAudio={origPartialAudio}
                partialVersion={origPartialVersion}
                label="Original Model"
                onSave={origResult ? () => saveToLibrary(origResult) : undefined}
                saving={saving === "original"}
                saved={savedModels.has("original")}
              />
              <SongCard
                backendUrl={backendUrl}
                status={dpoStatus}
                result={dpoResult}
                progress={dpoProgress}
                partialAudio={dpoPartialAudio}
                partialVersion={dpoPartialVersion}
                label="DPO Guided"
                onSave={dpoResult ? () => saveToLibrary(dpoResult) : undefined}
                saving={saving === "dpo"}
                saved={savedModels.has("dpo")}
              />
            </div>
          )}

          {/* Single Result */}
          {effectiveTab === "single" && singleStatus !== "idle" && backendUrl && (
            <SongCard
              backendUrl={backendUrl}
              status={singleStatus}
              result={singleResult}
              progress={singleProgress}
              partialAudio={singlePartialAudio}
              partialVersion={singlePartialVersion}
              onSave={singleResult ? () => saveToLibrary(singleResult) : undefined}
              saving={saving === selectedModel}
              saved={savedModels.has(selectedModel)}
            />
          )}
        </div>
      )}
    </div>
  );
}
