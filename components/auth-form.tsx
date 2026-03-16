"use client";

import { useState } from "react";
import { supabase } from "@/lib/supabase";

type Mode = "sign-in" | "sign-up" | "forgot-password" | "reset-sent";

export function AuthForm({ onSuccess }: { onSuccess?: () => void }) {
  const [mode, setMode] = useState<Mode>("sign-in");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function switchMode(m: Mode) {
    setMode(m);
    setError("");
  }

  async function handleSignIn() {
    setLoading(true);
    setError("");
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      setError(error.message);
      setLoading(false);
    } else {
      onSuccess?.();
    }
  }

  async function handleSignUp() {
    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    setLoading(true);
    setError("");
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: { data: { name } },
    });
    if (error) {
      setError(error.message);
      setLoading(false);
    } else if (data.session) {
      // Email confirmation disabled — logged in immediately
      onSuccess?.();
    } else {
      // Email confirmation enabled
      setError("Check your email to confirm your account.");
      setLoading(false);
    }
  }

  async function handleForgotPassword() {
    if (!email.trim()) {
      setError("Enter your email address");
      return;
    }
    setLoading(true);
    setError("");
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    if (error) {
      setError(error.message);
      setLoading(false);
    } else {
      setMode("reset-sent");
      setLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (mode === "sign-in") handleSignIn();
    else if (mode === "sign-up") handleSignUp();
    else if (mode === "forgot-password") handleForgotPassword();
  }

  if (mode === "reset-sent") {
    return (
      <div className="text-center space-y-4">
        <p className="text-sm text-zinc-300">
          Check your email for a password reset link.
        </p>
        <button
          onClick={() => switchMode("sign-in")}
          className="text-sm text-zinc-500 hover:text-white transition-colors"
        >
          Back to sign in
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      {mode === "sign-up" && (
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="name"
          className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-white/30 transition-colors text-sm"
        />
      )}

      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="email"
        required
        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-white/30 transition-colors text-sm"
      />

      {mode !== "forgot-password" && (
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="password"
          required
          className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-white/30 transition-colors text-sm"
        />
      )}

      {mode === "sign-up" && (
        <input
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          placeholder="confirm password"
          required
          className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-sm text-white placeholder:text-zinc-500 focus:outline-none focus:border-white/30 transition-colors text-sm"
        />
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 bg-white text-black font-semibold rounded-sm hover:bg-white/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
      >
        {loading
          ? "..."
          : mode === "sign-in"
            ? "Sign In"
            : mode === "sign-up"
              ? "Create Account"
              : "Send Reset Link"}
      </button>

      {error && <p className="text-red-400 text-xs text-center">{error}</p>}

      <div className="text-center pt-2 space-y-1">
        {mode === "sign-in" && (
          <>
            <button
              type="button"
              onClick={() => switchMode("forgot-password")}
              className="text-xs text-zinc-500 hover:text-white transition-colors block w-full"
            >
              forgot password?
            </button>
            <button
              type="button"
              onClick={() => switchMode("sign-up")}
              className="text-xs text-zinc-500 hover:text-white transition-colors block w-full"
            >
              create an account
            </button>
          </>
        )}
        {mode === "sign-up" && (
          <button
            type="button"
            onClick={() => switchMode("sign-in")}
            className="text-xs text-zinc-500 hover:text-white transition-colors"
          >
            already have an account? sign in
          </button>
        )}
        {mode === "forgot-password" && (
          <button
            type="button"
            onClick={() => switchMode("sign-in")}
            className="text-xs text-zinc-500 hover:text-white transition-colors"
          >
            back to sign in
          </button>
        )}
      </div>
    </form>
  );
}
