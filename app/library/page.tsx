"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { supabase, type DpoSong } from "@/lib/supabase";
import { useAuth } from "@/lib/auth-context";
import { Navbar } from "@/components/navbar";

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function LibraryPage() {
  const { user, guestId, loading: authLoading } = useAuth();
  const [songs, setSongs] = useState<DpoSong[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const editRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (authLoading) return;

    async function load() {
      // Use separate queries to avoid PostgREST .or() edge cases
      const seen = new Set<string>();
      const results: DpoSong[] = [];

      function addUnique(data: DpoSong[] | null) {
        for (const s of data || []) {
          if (!seen.has(s.id)) { seen.add(s.id); results.push(s); }
        }
      }

      if (user) {
        const { data } = await supabase
          .from("dpo-songs").select("*").eq("user_id", user.id);
        addUnique(data as DpoSong[] | null);
      }

      if (guestId) {
        // Guest songs (not yet migrated to an account)
        const q = supabase.from("dpo-songs").select("*").eq("guest_id", guestId);
        const { data } = user ? await q.is("user_id", null) : await q;
        addUnique(data as DpoSong[] | null);
      }

      // Legacy songs (pre-system, no identity at all)
      const { data: legacy, error: legacyErr } = await supabase
        .from("dpo-songs").select("*").is("user_id", null).is("guest_id", null);

      if (legacyErr) {
        // Columns likely don't exist yet — fall back to loading all songs
        const { data: all } = await supabase
          .from("dpo-songs").select("*").order("created_at", { ascending: false });
        setSongs((all || []) as DpoSong[]);
        setLoading(false);
        return;
      }

      addUnique(legacy as DpoSong[] | null);
      results.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      setSongs(results);
      setLoading(false);
    }
    load();
  }, [authLoading, user, guestId]);

  useEffect(() => {
    if (editingId && editRef.current) {
      editRef.current.focus();
      editRef.current.select();
    }
  }, [editingId]);

  async function renameTitle(id: string) {
    const trimmed = editTitle.trim() || "Untitled";
    const { error } = await supabase
      .from("dpo-songs")
      .update({ title: trimmed })
      .eq("id", id);

    if (!error) {
      setSongs((prev) =>
        prev.map((s) => (s.id === id ? { ...s, title: trimmed } : s))
      );
    }
    setEditingId(null);
  }

  function startEditing(song: DpoSong) {
    setEditingId(song.id);
    setEditTitle(song.title || "Untitled");
  }

  async function deleteSong(id: string) {
    const song = songs.find((s) => s.id === id);
    if (!song) return;

    // Delete from Supabase storage (extract filename from URL)
    const urlParts = song.audio_url.split("/");
    const storagePath = urlParts[urlParts.length - 1];
    if (storagePath) {
      await supabase.storage.from("dpo-songs").remove([storagePath]);
    }

    // Delete from database
    const { error } = await supabase.from("dpo-songs").delete().eq("id", id);
    if (!error) {
      setSongs((prev) => prev.filter((s) => s.id !== id));
    }
    setDeletingId(null);
  }

  const expanded = expandedId ? songs.find((s) => s.id === expandedId) : null;

  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar>
        <Link href="/generate" className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
          Generate →
        </Link>
      </Navbar>

      <div className="max-w-2xl mx-auto px-6 py-10 space-y-8">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">My Library</h2>
          <p className="text-zinc-500 text-sm mt-1">Songs you&apos;ve saved.</p>
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
                    <div className="flex items-center gap-2">
                      {editingId === song.id ? (
                        <input
                          ref={editRef}
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onBlur={() => renameTitle(song.id)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") renameTitle(song.id);
                            if (e.key === "Escape") setEditingId(null);
                          }}
                          className="bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1 text-sm font-medium text-white focus:outline-none focus:border-zinc-500 w-full"
                        />
                      ) : (
                        <button
                          onClick={() => startEditing(song)}
                          className="text-sm font-medium text-white truncate hover:text-zinc-300 transition-colors text-left"
                          title="Click to rename"
                        >
                          {song.title || "Untitled"}
                        </button>
                      )}
                      <span className={`text-[10px] font-medium tracking-wide uppercase px-1.5 py-0.5 rounded shrink-0 ${
                        song.model === "rate-winner"
                          ? "bg-emerald-950 text-emerald-400"
                          : song.model === "dpo"
                            ? "bg-zinc-800 text-zinc-400"
                            : "bg-zinc-800 text-zinc-500"
                      }`}>
                        {song.model === "rate-winner" ? "Rated" : song.model === "dpo" ? "DPO" : "Original"}
                      </span>
                    </div>
                    <p className="text-xs text-zinc-500 mt-1">{formatDate(song.created_at)}</p>
                  </div>
                  <div className="flex items-center gap-3 shrink-0 pt-0.5">
                    <button
                      onClick={() => setExpandedId(expandedId === song.id ? null : song.id)}
                      className="text-xs text-zinc-500 hover:text-white transition-colors"
                    >
                      {expandedId === song.id ? "Close" : "Details"}
                    </button>
                    {deletingId === song.id ? (
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => deleteSong(song.id)}
                          className="text-xs text-red-400 hover:text-red-300 transition-colors"
                        >
                          Confirm
                        </button>
                        <button
                          onClick={() => setDeletingId(null)}
                          className="text-xs text-zinc-500 hover:text-white transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setDeletingId(song.id)}
                        className="text-xs text-zinc-600 hover:text-red-400 transition-colors"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>

                <audio
                  controls
                  src={song.audio_url}
                  className="w-full"
                />

                {expandedId === song.id && (
                  <div className="space-y-3 pt-2 border-t border-zinc-800">
                    {song.prompt && (
                      <div>
                        <p className="text-[10px] text-zinc-500 font-medium tracking-wide uppercase mb-1">Prompt</p>
                        <p className="text-xs text-zinc-400">{song.prompt}</p>
                      </div>
                    )}
                    {song.tags && (
                      <div>
                        <p className="text-[10px] text-zinc-500 font-medium tracking-wide uppercase mb-1">Tags</p>
                        <p className="text-xs text-zinc-600 font-mono break-all">{song.tags}</p>
                      </div>
                    )}
                    {song.lyrics && (
                      <div>
                        <p className="text-[10px] text-zinc-500 font-medium tracking-wide uppercase mb-1">Lyrics</p>
                        <pre className="text-xs text-zinc-400 font-mono whitespace-pre-wrap leading-relaxed bg-zinc-950 rounded-xl p-4 max-h-64 overflow-y-auto">
                          {song.lyrics}
                        </pre>
                      </div>
                    )}
                    {song.num_frames && (
                      <p className="text-[10px] text-zinc-600">{song.num_frames} frames</p>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
