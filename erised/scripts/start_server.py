#!/usr/bin/env python3
"""
One-command server launcher for Erised.

Starts the FastAPI server + ngrok, then publishes the public URL to Supabase
so users at erised-me.vercel.app can connect without pasting any links.

Usage:
    # Start compare server (original vs DPO A/B):
    python -m erised.scripts.start_server

    # Start rating server (collect DPO preferences):
    python -m erised.scripts.start_server --mode rate

    # Use a different DPO checkpoint:
    python -m erised.scripts.start_server --dpo-path /workspace/dpo_checkpoints_v6/dpo_best
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Public Supabase credentials (anon key — safe to hardcode)
SUPABASE_URL = "https://zllpoyoumsfnwgfecryh.supabase.co"
SUPABASE_KEY = "sb_publishable_ysZ14tc1WvwJNnSEGSb2XA_gjsykNf5"

_repo_root = str(Path(__file__).resolve().parent.parent.parent)


def get_ngrok_url(max_wait: int = 45) -> str | None:
    """Poll local ngrok API until HTTPS tunnel URL appears."""
    import urllib.request
    import json

    for i in range(max_wait):
        try:
            with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2) as r:
                data = json.loads(r.read())
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    return tunnel["public_url"]
        except Exception:
            pass
        time.sleep(1)
        if i % 5 == 4:
            print(f"  waiting for ngrok... ({i+1}s)")
    return None


def _supabase_patch(payload: dict) -> None:
    import urllib.request
    import json

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/erised_config?key=eq.backend_url",
        data=body,
        method="PATCH",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    urllib.request.urlopen(req, timeout=10)


def publish_url(url: str) -> None:
    _supabase_patch({"value": url})
    print(f"✓ Published backend URL to Supabase: {url}")


def clear_url() -> None:
    try:
        _supabase_patch({"value": ""})
        print("✓ Backend URL cleared — site will show GPU offline.")
    except Exception as e:
        print(f"  Warning: could not clear URL from Supabase: {e}")


def main():
    parser = argparse.ArgumentParser(description="Start Erised GPU server for all users")
    parser.add_argument(
        "--mode", choices=["compare", "rate"], default="compare",
        help="compare = A/B compare original vs DPO; rate = collect preferences for DPO training",
    )
    parser.add_argument(
        "--dpo-path", type=str, default="/workspace/dpo_checkpoints_v6/dpo_best",
        help="Path to DPO checkpoint (only used in compare mode)",
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # ── Start FastAPI server ─────────────────────────────────────────
    if args.mode == "compare":
        server_cmd = [
            sys.executable, "-m", "erised.scripts.compare_local",
            "--dpo-path", args.dpo_path,
            "--port", str(args.port),
        ]
        print(f"Starting compare server (original vs DPO)...")
        print(f"  DPO path: {args.dpo_path}")
    else:
        server_cmd = [
            sys.executable, "-m", "erised.scripts.rate_local",
            "--port", str(args.port),
        ]
        print("Starting rating server (preference collection)...")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    server_proc = subprocess.Popen(server_cmd, cwd=_repo_root, env=env)
    print(f"  Server PID: {server_proc.pid} — loading model, this takes ~30s...")
    time.sleep(5)  # give server time to start binding

    # ── Start ngrok ──────────────────────────────────────────────────
    print(f"\nStarting ngrok on port {args.port}...")
    ngrok_proc = subprocess.Popen(
        ["ngrok", "http", str(args.port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url = get_ngrok_url(max_wait=45)
    if not url:
        print("\nERROR: Could not get ngrok URL.")
        print("Make sure ngrok is installed and authenticated:")
        print("  ngrok config add-authtoken <your-token>")
        server_proc.terminate()
        ngrok_proc.terminate()
        sys.exit(1)

    # ── Publish to Supabase ──────────────────────────────────────────
    try:
        publish_url(url)
    except Exception as e:
        print(f"\nERROR: Could not publish URL to Supabase: {e}")
        server_proc.terminate()
        ngrok_proc.terminate()
        sys.exit(1)

    mode_label = "Compare (A/B)" if args.mode == "compare" else "Rate (preference collection)"
    print(f"\n{'='*55}")
    print(f"  Erised GPU server is LIVE")
    print(f"  Mode:    {mode_label}")
    print(f"  Users:   https://erised-me.vercel.app")
    print(f"  Direct:  {url}")
    print(f"{'='*55}")
    print(f"\nPress Ctrl+C to stop and take the server offline.\n")

    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server_proc.terminate()
        ngrok_proc.terminate()
        clear_url()


if __name__ == "__main__":
    main()
