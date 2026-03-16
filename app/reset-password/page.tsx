"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function ResetPasswordPage() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    // Supabase puts the recovery tokens in the URL hash.
    // The client library picks them up automatically via onAuthStateChange.
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event) => {
      if (event === "PASSWORD_RECOVERY") {
        setReady(true);
      }
    });

    // Also check if we already have a session (user clicked link and tokens were processed)
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) setReady(true);
    });

    return () => subscription.unsubscribe();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");

    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    setLoading(true);
    const { error } = await supabase.auth.updateUser({ password });

    if (error) {
      setError(error.message);
      setLoading(false);
    } else {
      setMessage("Password reset successfully. Redirecting...");
      setTimeout(() => router.push("/"), 2000);
    }
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-black relative overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-white/[0.02] rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-sm w-full z-10 px-6">
        <h1 className="text-3xl font-semibold tracking-tighter text-white text-center mb-2">
          Reset Password
        </h1>

        {message ? (
          <p className="text-green-400 text-sm text-center mt-4">{message}</p>
        ) : !ready ? (
          <div className="text-center space-y-4 mt-6">
            <p className="text-zinc-400 text-sm">
              Loading reset session...
            </p>
            <p className="text-zinc-600 text-xs">
              If this takes too long, the link may be invalid or expired.
            </p>
            <button
              onClick={() => router.push("/")}
              className="text-white/60 hover:text-white text-sm transition-colors"
            >
              Back to home
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-2 mt-6">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="new password"
              required
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-white/30 transition-colors text-sm"
            />
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="confirm new password"
              required
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-white/30 transition-colors text-sm"
            />
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-white text-black font-semibold rounded-sm hover:bg-white/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
              {loading ? "..." : "Reset Password"}
            </button>
            {error && (
              <p className="text-red-400 text-xs text-center">{error}</p>
            )}
          </form>
        )}
      </div>
    </main>
  );
}
