"use client";

import { useState, useEffect, useRef } from "react";
import { supabase } from "@/lib/supabase";
import { useAuth } from "@/lib/auth-context";
import { Navbar } from "@/components/navbar";
import { GEN_URL } from "@/lib/backend";

type JobStatus = "idle" | "pending" | "running" | "done" | "error";

interface GenerationResult {
  audio_file: string;
  tags: string;
  num_frames: number;
  seconds_audio?: number;
  elapsed: number;
  model: string;
}

interface JobState {
  status: JobStatus;
  stage: string | null;
  progress: number; // 0..1
  result: GenerationResult | null;
}

const IDLE: JobState = { status: "idle", stage: null, progress: 0, result: null };

const STAGE_LABEL: Record<string, string> = {
  queued: "Waking up the GPU",
  composing: "Composing",
  "adding detail": "Adding detail",
  "rendering audio": "Rendering audio",
};

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

/** Poll a job until it finishes. Returns a cleanup function. */
function pollJob(jobId: string, onUpdate: (s: Partial<JobState>) => void): () => void {
  let cancelled = false;
  (async () => {
    while (!cancelled) {
      try {
        const resp = await fetchRetry(`${GEN_URL}/api/job/${jobId}`);
        if (resp.ok) {
          const data = await resp.json();
          const p = data.progress;
          const progress = p?.total_frames ? p.current_frame / p.total_frames : 0;
          if (data.status === "done") {
            onUpdate({ status: "done", stage: null, progress: 1, result: data.result });
            return;
          }
          if (data.status === "error") {
            onUpdate({ status: "error", stage: null });
            return;
          }
          onUpdate({ status: data.status === "running" ? "running" : "pending", stage: data.stage ?? null, progress });
        }
      } catch {
        // keep polling
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
  })();
  return () => { cancelled = true; };
}

function SongCard({ job, onSave, saving, saved }: {
  job: JobState;
  onSave?: () => void;
  saving?: boolean;
  saved?: boolean;
}) {
  const isLoading = job.status === "pending" || job.status === "running";
  const pct = Math.round(job.progress * 100);
  if (job.status === "idle") return null;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        {isLoading && (
          <span className="text-xs text-zinc-500 animate-pulse">
            {STAGE_LABEL[job.stage ?? "queued"] ?? "Composing"}{pct > 0 ? ` (${pct}%)` : "..."}
          </span>
        )}
        {job.result && (
          <span className="text-xs text-zinc-500">
            {job.result.seconds_audio ? `${Math.round(job.result.seconds_audio)}s song · ` : ""}made in {Math.round(job.result.elapsed)}s
          </span>
        )}
      </div>

      {isLoading && (
        <div className="bg-zinc-800 rounded-full h-1.5 overflow-hidden">
          <div className="bg-zinc-600 rounded-full h-full transition-all duration-500" style={{ width: `${pct}%` }} />
        </div>
      )}

      {job.result && (
        <audio controls autoPlay src={`${GEN_URL}/audio/${job.result.audio_file}`} className="w-full" />
      )}

      {job.status === "error" && <p className="text-xs text-red-400">Generation failed</p>}

      {job.result && (
        <>
          <p className="text-xs text-zinc-600 font-mono break-all">{job.result.tags}</p>
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
  const [prompt, setPrompt] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [minutes, setMinutes] = useState(2);
  const [job, setJob] = useState<JobState>(IDLE);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [randomizingPrompt, setRandomizingPrompt] = useState(false);
  const [randomizingLyrics, setRandomizingLyrics] = useState(false);
  const [songTitle, setSongTitle] = useState("Untitled");

  const cleanupRef = useRef<(() => void) | null>(null);
  useEffect(() => () => cleanupRef.current?.(), []);

  async function generate() {
    if (!prompt.trim() || !lyrics.trim()) return;
    cleanupRef.current?.();
    setError(null);
    setSaved(false);
    setJob({ ...IDLE, status: "pending", stage: "queued" });

    try {
      const resp = await fetchRetry(`${GEN_URL}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, lyrics, max_sec: minutes * 60, user_email: user?.email || null }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || !data.job_id) throw new Error(data.detail || "Generation failed");
      cleanupRef.current = pollJob(data.job_id, (update) => setJob((s) => ({ ...s, ...update })));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setJob((s) => ({ ...s, status: "error" }));
    }
  }

  async function saveToLibrary(result: GenerationResult) {
    setSaving(true);
    try {
      const audioResp = await fetch(`/api/proxy-audio?file=${encodeURIComponent(result.audio_file)}`);
      if (!audioResp.ok) {
        const errData = await audioResp.json().catch(() => ({ error: "Failed to fetch audio" }));
        throw new Error(errData.error || "Failed to fetch audio");
      }
      const blob = await audioResp.blob();
      const ext = result.audio_file.split(".").pop() || "mp3";
      const filename = `${Date.now()}_${result.model}.${ext}`;

      const { data: uploadData, error: uploadErr } = await supabase.storage
        .from("dpo-songs")
        .upload(filename, blob, { contentType: blob.type || "audio/mpeg", upsert: false });

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

      setSaved(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
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

  const isGenerating = job.status === "pending" || job.status === "running";

  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-xs text-zinc-500">GPU online</span>
        </div>
      </Navbar>

      <div className="max-w-2xl mx-auto px-6 py-10 space-y-8">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Generate</h2>
          <p className="text-zinc-500 text-sm mt-1">Generate music with Erised.</p>
        </div>

        <div className="space-y-4">
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
              Length — about {minutes} min
            </label>
            <input
              type="range"
              min={1}
              max={4}
              step={1}
              value={minutes}
              onChange={(e) => setMinutes(Number(e.target.value))}
              className="w-full accent-white"
            />
            <p className="text-xs text-zinc-600 mt-2">
              Takes about as long as the song itself, plus up to a minute if the GPU is waking up.
            </p>
          </div>

          <button
            onClick={generate}
            disabled={isGenerating || !prompt.trim() || !lyrics.trim()}
            className="w-full py-4 bg-white text-black font-semibold rounded-xl text-sm tracking-tight hover:bg-zinc-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isGenerating ? "Generating..." : "Generate"}
          </button>
        </div>

        {error && (
          <p className="text-red-400 text-sm bg-red-950/30 border border-red-900 rounded-xl px-4 py-3">
            {error}
          </p>
        )}

        <SongCard
          job={job}
          onSave={job.result ? () => saveToLibrary(job.result!) : undefined}
          saving={saving}
          saved={saved}
        />

        <p className="text-[11px] text-zinc-700">
          Songs are made by Shao (CC BY-NC 4.0), fine-tuned by Erised.
        </p>
      </div>
    </div>
  );
}
