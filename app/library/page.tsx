"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { supabase, type DpoSong } from "@/lib/supabase";

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function LibraryPage() {
  const [songs, setSongs] = useState<DpoSong[]>([]);
  const [loading, setLoading] = useState(true);
  const [playingId, setPlayingId] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const { data, error } = await supabase
        .from("dpo_songs")
        .select("*")
        .eq("model", "dpo")
        .order("created_at", { ascending: false });

      if (!error && data) setSongs(data as DpoSong[]);
      setLoading(false);
    }
    load();
  }, []);

  return (
    <div className="min-h-screen bg-black text-white">
      <nav className="px-6 py-5 flex items-center justify-between border-b border-zinc-900">
        <Link href="/" className="text-xl font-semibold tracking-tighter text-white">
          Erised
        </Link>
        <Link href="/generate" className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
          Generate →
        </Link>
      </nav>

      <div className="max-w-2xl mx-auto px-6 py-10 space-y-8">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">My Library</h2>
          <p className="text-zinc-500 text-sm mt-1">DPO-tuned songs you&apos;ve saved.</p>
        </div>

        {loading ? (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="bg-zinc-900 rounded-2xl p-5 animate-pulse h-28" />
            ))}
          </div>
        ) : songs.length === 0 ? (
          <div className="text-center py-20 text-zinc-600">
            <p className="text-lg">No songs yet.</p>
            <Link
              href="/generate"
              className="mt-4 inline-block text-sm text-zinc-400 hover:text-white transition-colors"
            >
              Generate one →
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {songs.map((song) => (
              <div
                key={song.id}
                className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 space-y-3"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">
                      {song.prompt}
                    </p>
                    <p className="text-xs text-zinc-500 mt-1">{formatDate(song.created_at)}</p>
                    {song.tags && (
                      <p className="text-xs text-zinc-600 font-mono mt-1 truncate">{song.tags}</p>
                    )}
                  </div>
                </div>
                <audio
                  controls
                  src={song.audio_url}
                  className="w-full"
                  onPlay={() => setPlayingId(song.id)}
                  onPause={() => setPlayingId(null)}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
