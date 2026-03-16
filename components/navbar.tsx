"use client";

import { useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { AuthModal } from "./auth-modal";

export function Navbar({ children }: { children?: React.ReactNode }) {
  const { user, signOut } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);

  return (
    <>
      <nav className="px-6 py-5 flex items-center justify-between border-b border-zinc-900">
        <Link
          href="/"
          className="text-xl font-semibold tracking-tighter text-white"
        >
          Erised
        </Link>
        <div className="flex items-center gap-4">
          {children}
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
      </nav>
      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} />
    </>
  );
}
