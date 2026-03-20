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
 * Simple song player. During generation, plays the partial combined audio
 * file (which the server updates after each chunk). After generation,
 * switches to the final file. No Web Audio scheduling — just a plain
 * <audio> element that won't produce overlapping/stacking artifacts.
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
  progress: { current_frame: number; total_frames: number } | null;
  partialAudio: string | null;
  partialVersion: number;
  label?: string;
  onSave?: () => void;
  saving?: boolean;
  saved?: boolean;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const loadedVersionRef = useRef(0);
  const seekOnSwitch = useRef(0);
  const pendingUrlRef = useRef<string | null>(null);
  const blobUrlsRef = useRef<string[]>([]);
  const fetchControllerRef = useRef<AbortController | null>(null);

  // Reset on new generation
  useEffect(() => {
    if (status === "pending") {
      setAudioSrc(null);
      loadedVersionRef.current = 0;
      seekOnSwitch.current = 0;
      pendingUrlRef.current = null;
      if (fetchControllerRef.current) fetchControllerRef.current.abort();
      blobUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      blobUrlsRef.current = [];
    }
  }, [status]);

  // Cleanup blob URLs on unmount
  useEffect(() => () => {
    blobUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  // When partial audio becomes available or updates
  useEffect(() => {
    if (result || !partialAudio || partialVersion <= 0) return;
    if (partialVersion <= loadedVersionRef.current) return;

    const networkUrl = `${backendUrl}/audio/${partialAudio}?v=${partialVersion}`;
    loadedVersionRef.current = partialVersion;

    // First chunk — no audio loaded yet, play immediately
    if (!audioSrc) {
      setAudioSrc(networkUrl);
      return;
    }

    // If the audio has already ended or is paused,
    // switch immediately with network URL.
    const el = audioRef.current;
    if (el && (el.ended || el.paused)) {
      seekOnSwitch.current = el.currentTime;
      setAudioSrc(networkUrl);
      return;
    }

    // Audio is still playing. Pre-download the longer cumulative file
    // into browser memory NOW so that when we switch at remaining < 2s,
    // the browser loads from memory (12ms) instead of network (2000ms).
    // Set network URL as fallback in case the fetch doesn't finish in time.
    pendingUrlRef.current = networkUrl;

    if (fetchControllerRef.current) fetchControllerRef.current.abort();
    const controller = new AbortController();
    fetchControllerRef.current = controller;

    fetch(networkUrl, { signal: controller.signal })
      .then((r) => r.blob())
      .then((blob) => {
        if (controller.signal.aborted) return;
        const blobUrl = URL.createObjectURL(blob);
        blobUrlsRef.current.push(blobUrl);
        // Upgrade pending URL from network to blob — if the switch
        // hasn't happened yet, it will now use the in-memory blob.
        if (pendingUrlRef.current === networkUrl) {
          pendingUrlRef.current = blobUrl;
        }
      })
      .catch(() => {}); // network URL remains as fallback
  }, [partialAudio, partialVersion, result, backendUrl, audioSrc]);

  // When generation completes, switch to final audio
  useEffect(() => {
    if (!result) return;
    const el = audioRef.current;
    if (el && !el.paused && isFinite(el.currentTime)) {
      seekOnSwitch.current = el.currentTime;
    }
    setAudioSrc(`${backendUrl}/audio/${result.audio_file}`);
  }, [result, backendUrl]);

  const isLoading = status === "pending" || status === "running";
  const progressPct = progress?.total_frames
    ? Math.round((progress.current_frame / progress.total_frames) * 100)
    : 0;
  const hasAudio = !!audioSrc;

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

      {/* Progress bar while composing (before first audio) */}
      {isLoading && !hasAudio && progressPct > 0 && (
        <div className="bg-zinc-800 rounded-full h-1.5 overflow-hidden">
          <div
            className="bg-zinc-600 rounded-full h-full transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      )}

      {/* Audio player */}
      {audioSrc && (
        <audio
          ref={audioRef}
          controls
          src={audioSrc}
          className="w-full"
          onCanPlay={() => {
            const el = audioRef.current;
            if (!el) return;
            if (seekOnSwitch.current > 0 && seekOnSwitch.current < el.duration) {
              el.currentTime = seekOnSwitch.current;
              seekOnSwitch.current = 0;
            }
            el.play().catch(() => {});
          }}
          onTimeUpdate={() => {
            const el = audioRef.current;
            if (!el || !pendingUrlRef.current) return;
            const remaining = el.duration - el.currentTime;
            // When within 2s of the end, switch to the longer partial file
            if (isFinite(remaining) && remaining < 2) {
              seekOnSwitch.current = el.currentTime;
              setAudioSrc(pendingUrlRef.current);
              pendingUrlRef.current = null;
            }
          }}
          onEnded={() => {
            // If audio ends and there's a pending longer version, switch to it
            const el = audioRef.current;
            if (pendingUrlRef.current && el) {
              seekOnSwitch.current = el.currentTime;
              setAudioSrc(pendingUrlRef.current);
              pendingUrlRef.current = null;
            }
          }}
        />
      )}

      {/* Streaming indicator while audio is playing */}
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
    status: "idle", result: null, progress: null, partialAudio: null, partialVersion: 0,
  });
  const [dpoState, setDpoState] = useState<StreamState>({
    status: "idle", result: null, progress: null, partialAudio: null, partialVersion: 0,
  });
  const [singleState, setSingleState] = useState<StreamState>({
    status: "idle", result: null, progress: null, partialAudio: null, partialVersion: 0,
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
    const blank: StreamState = { status: "pending", result: null, progress: null, partialAudio: null, partialVersion: 0 };
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
    const blank: StreamState = { status: "pending", result: null, progress: null, partialAudio: null, partialVersion: 0 };
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
