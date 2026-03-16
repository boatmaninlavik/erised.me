"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { AuthModal } from "./auth-modal";

export function HomeAuthButton() {
  const { user, signOut } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);

  return (
    <>
      <div className="fixed top-6 right-6 z-10">
        {user ? (
          <div className="flex items-center gap-3">
            <span className="text-xs text-zinc-600 hidden sm:inline">
              {user.email}
            </span>
            <button
              onClick={signOut}
              className="text-xs text-zinc-500 hover:text-white transition-colors"
            >
              Sign Out
            </button>
          </div>
        ) : (
          <button
            onClick={() => setAuthOpen(true)}
            className="text-xs text-zinc-400 hover:text-white transition-colors"
          >
            Sign In
          </button>
        )}
      </div>
      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} />
    </>
  );
}
