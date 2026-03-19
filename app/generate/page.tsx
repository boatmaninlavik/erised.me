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
 * Single song player card.
 * During generation: plays individual chunk WAVs via Web Audio API for gapless
 * streaming (AudioContext must be created in a click handler to satisfy browser
 * autoplay policy). Falls back to <audio> src-switching if chunks unavailable.
 * After generation: standard <audio> element for seeking/scrubbing.
 */
function SongCard({
  backendUrl,
  status,
  result,
  progress,
  partialAudio,
  partialVersion,
  audioCtx,
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
  audioCtx: AudioContext | null;
  label?: string;
  onSave?: () => void;
  saving?: boolean;
  saved?: boolean;
}) {
  // ── Web Audio gapless streaming ──
  const streamRef = useRef({
    active: false,
    nextTime: 0,
    startTime: 0,
    loadedChunks: 0,
    totalDuration: 0,
    loading: false,
    sources: [] as AudioBufferSourceNode[],
  });
  const [webAudioFailed, setWebAudioFailed] = useState(false);
  const [streamPlaying, setStreamPlaying] = useState(false);
  const [streamTime, setStreamTime] = useState(0);
  const [streamBuffered, setStreamBuffered] = useState(0);

  // ── Fallback <audio> (when chunks unavailable) ──
  const audioRef = useRef<HTMLAudioElement>(null);
  const fallbackVersionRef = useRef(0);
  const fallbackSeekRef = useRef(0);
  const [fallbackSrc, setFallbackSrc] = useState<string | null>(null);

  // ── Final <audio> (after generation complete) ──
  const [finalSrc, setFinalSrc] = useState<string | null>(null);
  const finalSeekRef = useRef(0);

  const canWebAudio = !!audioCtx && !webAudioFailed;

  // Reset everything on new generation
  useEffect(() => {
    if (status !== "pending") return;
    streamRef.current.sources.forEach((s) => { try { s.stop(); } catch { /* already stopped */ } });
    streamRef.current = {
      active: false, nextTime: 0, startTime: 0,
      loadedChunks: 0, totalDuration: 0, loading: false, sources: [],
    };
    setWebAudioFailed(false);
    setStreamPlaying(false);
    setStreamTime(0);
    setStreamBuffered(0);
    setFallbackSrc(null);
    setFinalSrc(null);
    fallbackVersionRef.current = 0;
    fallbackSeekRef.current = 0;
  }, [status]);

  // ── Web Audio: fetch and schedule chunk files gaplessly ──
  useEffect(() => {
    const s = streamRef.current;
    if (!canWebAudio || !partialAudio || partialVersion <= 0 || finalSrc) return;
    if (s.loading || s.loadedChunks >= partialVersion) return;
    s.loading = true;

    const dotIdx = partialAudio.lastIndexOf(".");
    const base = dotIdx >= 0 ? partialAudio.substring(0, dotIdx) : partialAudio;
    const target = partialVersion;

    (async () => {
      try {
        for (let i = s.loadedChunks; i < target; i++) {
          const resp = await fetch(`${backendUrl}/audio/${base}_chunk${i}.wav`);
          if (!resp.ok) {
            if (i === 0) { setWebAudioFailed(true); return; }
            break;
          }
          const audioBuf = await audioCtx!.decodeAudioData(await resp.arrayBuffer());
          const source = audioCtx!.createBufferSource();
          source.buffer = audioBuf;
          source.connect(audioCtx!.destination);

          if (!s.active) {
            await audioCtx!.resume();
            const t0 = audioCtx!.currentTime + 0.05;
            source.start(t0);
            s.startTime = t0;
            s.nextTime = t0 + audioBuf.duration;
            s.active = true;
            setStreamPlaying(true);
          } else {
            source.start(s.nextTime);
            s.nextTime += audioBuf.duration;
          }

          s.sources.push(source);
          s.totalDuration += audioBuf.duration;
          s.loadedChunks = i + 1;
          setStreamBuffered(s.totalDuration);
        }
      } catch (e) {
        console.error("Web Audio chunk error:", e);
        setWebAudioFailed(true);
      } finally {
        s.loading = false;
      }
    })();
  }, [canWebAudio, partialAudio, partialVersion, finalSrc, audioCtx, backendUrl]);

  // ── Fallback: set initial <audio> src ──
  useEffect(() => {
    if (canWebAudio || !partialAudio || partialVersion <= 0 || finalSrc) return;
    if (partialVersion <= fallbackVersionRef.current) return;
    const newSrc = `${backendUrl}/audio/${partialAudio}?v=${partialVersion}`;
    if (!fallbackSrc) {
      fallbackVersionRef.current = partialVersion;
      setFallbackSrc(newSrc);
    }
  }, [canWebAudio, partialAudio, partialVersion, backendUrl, finalSrc, fallbackSrc]);

  // Fallback: poll for safe switch moment
  useEffect(() => {
    if (canWebAudio || finalSrc) return;
    if (status !== "running") return;
    const iv = setInterval(() => {
      const el = audioRef.current;
      if (!el || !partialAudio) return;
      if (partialVersion <= fallbackVersionRef.current) return;
      const shouldSwitch = el.paused || el.ended ||
        (isFinite(el.duration) && el.duration - el.currentTime < 2);
      if (shouldSwitch) {
        fallbackSeekRef.current = el.currentTime || 0;
        fallbackVersionRef.current = partialVersion;
        el.pause();
        el.removeAttribute("src");
        el.load();
        setFallbackSrc(`${backendUrl}/audio/${partialAudio}?v=${partialVersion}`);
      }
    }, 500);
    return () => clearInterval(iv);
  }, [canWebAudio, status, partialVersion, partialAudio, backendUrl, finalSrc]);

  // ── Switch to final <audio> when generation complete ──
  useEffect(() => {
    if (!result || finalSrc) return;
    const s = streamRef.current;
    let seekPos = 0;
    if (s.active && audioCtx) {
      seekPos = Math.max(0, audioCtx.currentTime - s.startTime);
      s.sources.forEach((src) => { try { src.stop(); } catch { /* noop */ } });
      s.sources = [];
      s.active = false;
    } else if (audioRef.current) {
      seekPos = audioRef.current.currentTime || 0;
      audioRef.current.pause();
    }
    finalSeekRef.current = seekPos;
    setStreamPlaying(false);
    setFinalSrc(`${backendUrl}/audio/${result.audio_file}`);
  }, [result, finalSrc, audioCtx, backendUrl]);

  // ── Update time display for Web Audio streaming ──
  useEffect(() => {
    if (!streamPlaying || !audioCtx) return;
    const iv = setInterval(() => {
      if (audioCtx.state === "running") {
        setStreamTime(Math.max(0, audioCtx.currentTime - streamRef.current.startTime));
      }
    }, 250);
    return () => clearInterval(iv);
  }, [streamPlaying, audioCtx]);

  // Cleanup on unmount
  useEffect(() => () => {
    streamRef.current.sources.forEach((s) => { try { s.stop(); } catch { /* noop */ } });
  }, []);

  const isLoading = status === "pending" || status === "running";
  const progressPct = progress?.total_frames
    ? Math.round((progress.current_frame / progress.total_frames) * 100)
    : 0;
  const isComposing = isLoading && !partialAudio && !streamPlaying && !fallbackSrc;
  const isStreaming = isLoading && (streamPlaying || !!fallbackSrc || !!partialAudio);

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

      {/* Web Audio: gapless chunk streaming with progress bar */}
      {streamPlaying && !finalSrc && (
        <div className="flex items-center gap-3">
          <div className="flex-1 bg-zinc-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-white rounded-full h-full transition-all duration-200"
              style={{ width: `${streamBuffered > 0 ? Math.min(100, (streamTime / streamBuffered) * 100) : 0}%` }}
            />
          </div>
          <span className="text-xs text-zinc-500 tabular-nums min-w-[5rem] text-right">
            {formatTime(streamTime)} / {formatTime(streamBuffered)}
          </span>
        </div>
      )}

      {/* Fallback: standard <audio> during streaming (when chunks unavailable) */}
      {!streamPlaying && fallbackSrc && !finalSrc && (
        <audio
          ref={audioRef}
          controls
          src={fallbackSrc}
          className="w-full"
          onCanPlay={() => {
            const el = audioRef.current;
            if (!el) return;
            if (fallbackSeekRef.current > 0) {
              el.currentTime = fallbackSeekRef.current;
              fallbackSeekRef.current = 0;
            }
            el.play().catch(() => {});
          }}
        />
      )}

      {/* Final <audio> after generation complete — full controls for seeking */}
      {finalSrc && (
        <audio
          ref={audioRef}
          controls
          src={finalSrc}
          className="w-full"
          onCanPlay={() => {
            const el = audioRef.current;
            if (!el) return;
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

  // AudioContext for Web Audio gapless streaming — MUST be created in a click
  // handler to satisfy browser autoplay policy. Shared across SongCards.
  const audioCtxRef = useRef<AudioContext | null>(null);

  /** Create or resume AudioContext — call synchronously inside a click handler. */
  function ensureAudioContext(): AudioContext {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      audioCtxRef.current = new AudioContext({ sampleRate: 48000 });
    }
    // resume() in gesture context ensures it won't be blocked
    audioCtxRef.current.resume();
    return audioCtxRef.current;
  }

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
    // Create AudioContext NOW — inside the click handler (user gesture)
    ensureAudioContext();

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
    // Create AudioContext NOW — inside the click handler (user gesture)
    ensureAudioContext();

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
                audioCtx={audioCtxRef.current}
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
                audioCtx={audioCtxRef.current}
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
              audioCtx={audioCtxRef.current}
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
