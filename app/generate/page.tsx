"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { supabase } from "@/lib/supabase";

type JobStatus = "idle" | "pending" | "running" | "done" | "error";

interface GenerationResult {
  audio_file: string;
  tags: string;
  num_frames: number;
  elapsed: number;
  model: string;
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

async function pollJob(backendUrl: string, jobId: string): Promise<GenerationResult> {
  while (true) {
    await new Promise((r) => setTimeout(r, 3000));
    const resp = await fetchRetry(`${backendUrl}/api/job/${jobId}`);
    const data = await resp.json();
    if (data.status === "done") return data.result;
    if (data.status === "error") throw new Error(data.error);
  }
}

export default function GeneratePage() {
  const [backendUrl, setBackendUrl] = useState<string | null>(null);
  const [gpuStatus, setGpuStatus] = useState<"loading" | "online" | "offline">("loading");
  const [prompt, setPrompt] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [maxSec, setMaxSec] = useState(60);
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

  async function generateBoth() {
    if (!backendUrl || !prompt.trim() || !lyrics.trim()) return;
    setError(null);
    setOrigResult(null);
    setDpoResult(null);
    setSavedModels(new Set());
    setOrigStatus("pending");
    setDpoStatus("pending");

    try {
      const origResp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: "original" }),
      });
      const origJob = await origResp.json();
      setOrigStatus("running");

      const dpoResp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: "dpo" }),
      });
      const dpoJob = await dpoResp.json();
      setDpoStatus("running");

      // Show each result as soon as it's ready (original first so user can listen while DPO generates)
      const origPromise = pollJob(backendUrl, origJob.job_id).then((res) => {
        setOrigResult(res);
        setOrigStatus("done");
      }).catch((e) => {
        setOrigStatus("error");
        throw e;
      });

      const dpoPromise = pollJob(backendUrl, dpoJob.job_id).then((res) => {
        setDpoResult(res);
        setDpoStatus("done");
      }).catch((e) => {
        setDpoStatus("error");
        throw e;
      });

      await Promise.all([origPromise, dpoPromise]);
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

    try {
      const resp = await fetchRetry(`${backendUrl}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: selectedModel }),
      });
      const job = await resp.json();
      setSingleStatus("running");
      const result = await pollJob(backendUrl, job.job_id);
      setSingleResult(result);
      setSingleStatus("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setSingleStatus("error");
    }
  }

  async function saveToLibrary(result: GenerationResult) {
    if (!backendUrl) return;
    setSaving(result.model);
    try {
      const audioResp = await fetch(`${backendUrl}/audio/${result.audio_file}`);
      const blob = await audioResp.blob();
      const ext = result.audio_file.split(".").pop() || "wav";
      const filename = `${Date.now()}_${result.model}.${ext}`;

      const { data: uploadData, error: uploadErr } = await supabase.storage
        .from("dpo-songs")
        .upload(filename, blob, { contentType: blob.type || "audio/wav", upsert: false });

      if (uploadErr) throw uploadErr;

      const { data: urlData } = supabase.storage.from("dpo-songs").getPublicUrl(uploadData.path);

      const { error: insertErr } = await supabase.from("dpo-songs").insert({
        title: "Untitled",
        prompt,
        lyrics,
        tags: result.tags,
        audio_url: urlData.publicUrl,
        num_frames: result.num_frames,
        model: result.model,
      });

      if (insertErr) throw insertErr;

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
      if (data.text) setValue(data.text);
      else if (data.error) setError(data.error);
    } catch {
      setError("Failed to generate random " + type);
    } finally {
      setLoading(false);
    }
  }

  const isGenerating =
    origStatus === "pending" || origStatus === "running" ||
    dpoStatus === "pending" || dpoStatus === "running" ||
    singleStatus === "pending" || singleStatus === "running";

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
            <h2 className="text-2xl font-semibold tracking-tight">Generate</h2>
            <p className="text-zinc-500 text-sm mt-1">Compare original vs DPO-tuned model output.</p>
          </div>

          {/* Tabs */}
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

          {/* Form */}
          <div className="space-y-4">
            {tab === "single" && (
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
                    {m === "dpo" ? "DPO-tuned" : "Original"}
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

            <button
              onClick={tab === "ab" ? generateBoth : generateSingle}
              disabled={isGenerating || !prompt.trim() || !lyrics.trim()}
              className="w-full py-4 bg-white text-black font-semibold rounded-xl text-sm tracking-tight hover:bg-zinc-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isGenerating
                ? "Generating..."
                : tab === "ab"
                ? "Generate Both (Original + DPO)"
                : `Generate with ${selectedModel === "dpo" ? "DPO-tuned" : "Original"} model`}
            </button>
          </div>

          {error && (
            <p className="text-red-400 text-sm bg-red-950/30 border border-red-900 rounded-xl px-4 py-3">
              {error}
            </p>
          )}

          {/* A/B Results */}
          {tab === "ab" && (origStatus !== "idle" || dpoStatus !== "idle") && (
            <div className="space-y-4">
              {(["original", "dpo"] as const).map((model) => {
                const result = model === "original" ? origResult : dpoResult;
                const status = model === "original" ? origStatus : dpoStatus;
                return (
                  <div key={model} className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="font-medium text-sm tracking-tight">
                        {model === "dpo" ? "DPO-Tuned Model" : "Original Model"}
                      </h3>
                      {(status === "running" || status === "pending") && (
                        <span className="text-xs text-zinc-500 animate-pulse">Generating...</span>
                      )}
                      {result && (
                        <span className="text-xs text-zinc-500">{result.elapsed}s · {result.num_frames} frames</span>
                      )}
                    </div>
                    {result && backendUrl ? (
                      <>
                        <audio controls src={`${backendUrl}/audio/${result.audio_file}`} className="w-full" />
                        <p className="text-xs text-zinc-600 font-mono">{result.tags}</p>
                        <button
                          onClick={() => saveToLibrary(result)}
                          disabled={saving !== null || savedModels.has(model)}
                          className="text-xs text-zinc-400 hover:text-white transition-colors disabled:opacity-40"
                        >
                          {savedModels.has(model) ? "Saved" : saving === model ? "Saving..." : "Save to My Library"}
                        </button>
                      </>
                    ) : status === "error" ? (
                      <p className="text-xs text-red-400">Generation failed</p>
                    ) : (
                      <div className="h-10 bg-zinc-800 rounded-lg animate-pulse" />
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Single Result */}
          {tab === "single" && singleStatus !== "idle" && (
            <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-sm tracking-tight">
                  {selectedModel === "dpo" ? "DPO-Tuned Model" : "Original Model"}
                </h3>
                {(singleStatus === "running" || singleStatus === "pending") && (
                  <span className="text-xs text-zinc-500 animate-pulse">Generating...</span>
                )}
              </div>
              {singleResult && backendUrl ? (
                <>
                  <audio controls src={`${backendUrl}/audio/${singleResult.audio_file}`} className="w-full" />
                  <p className="text-xs text-zinc-600 font-mono">{singleResult.tags}</p>
                  <button
                    onClick={() => saveToLibrary(singleResult)}
                    disabled={saving !== null || savedModels.has(selectedModel)}
                    className="text-xs text-zinc-400 hover:text-white transition-colors disabled:opacity-40"
                  >
                    {savedModels.has(selectedModel) ? "Saved" : saving === selectedModel ? "Saving..." : "Save to My Library"}
                  </button>
                </>
              ) : singleStatus === "error" ? (
                <p className="text-xs text-red-400">Generation failed</p>
              ) : (
                <div className="h-10 bg-zinc-800 rounded-lg animate-pulse" />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
