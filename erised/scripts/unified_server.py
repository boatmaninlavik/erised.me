#!/usr/bin/env python3
"""
Unified FastAPI server that runs BOTH the compare (original vs DPO) and
rate (preference collection) features on available GPUs.

With 2+ GPUs:
  GPU 0: Compare pipeline — swaps weights between original and DPO checkpoint
  GPU 1: Rate pipeline — generates pairs for RLHF preference collection

With 1 GPU:
  Loads only one pipeline at a time on cuda:0, based on which mode the user
  picks in the UI. Automatically unloads the other pipeline when switching.

All endpoints served on a single port so both /generate and /rate pages
on erised.me work with one backend_url.

Usage:
    python3 -u -m erised.scripts.unified_server \
        --dpo-path /workspace/dpo_checkpoints_v6/dpo_best \
        --port 7860
"""

import argparse
import collections
import glob
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("erised.unified")


def load_safetensors_sharded(model, model_path, device="cuda"):
    from safetensors.torch import load_file
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*", "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f, device=str(device)))
    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded %d tensors from %d file(s) in %s", len(state_dict), len(safetensor_files), model_path)


def main():
    parser = argparse.ArgumentParser(description="Unified Erised server (compare + rate)")
    parser.add_argument("--original-path", type=str, default=None)
    parser.add_argument("--dpo-path", type=str, default="/workspace/dpo_checkpoints_v6/dpo_best")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    import gc
    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    from erised.config import ErisedConfig
    from erised.pipeline import ErisedPipeline
    from erised.dpo.data import PreferenceStore

    num_gpus = torch.cuda.device_count()
    single_gpu = num_gpus < 2
    logger.info("Detected %d GPU(s) — running in %s mode",
                num_gpus, "single-GPU" if single_gpu else "multi-GPU")

    base_config = ErisedConfig.from_env()
    original_model_path = args.original_path or base_config.model_path

    # ── Active pipeline state (single-GPU lazy-swap) ─────────────────────
    # In single-GPU mode only one pipeline is loaded at a time.
    # active_mode is "compare", "rate", or None.
    active_mode = None
    mode_lock = threading.Lock()

    compare_pipeline = None
    compare_device = None
    rate_pipeline = None

    def _unload_pipeline(name):
        """Free a pipeline and reclaim GPU memory."""
        nonlocal compare_pipeline, rate_pipeline
        if name == "compare" and compare_pipeline is not None:
            logger.info("[single-gpu] Unloading compare pipeline...")
            del compare_pipeline
            compare_pipeline = None
        elif name == "rate" and rate_pipeline is not None:
            logger.info("[single-gpu] Unloading rate pipeline...")
            del rate_pipeline
            rate_pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

    def _load_compare():
        nonlocal compare_pipeline, compare_device
        if compare_pipeline is not None:
            return
        logger.info("Loading COMPARE pipeline on cuda:0...")
        cfg = ErisedConfig.from_env()
        cfg.mula_device = "cuda:0"
        cfg.codec_device = "cuda:0"
        cfg.lazy_load = False
        compare_pipeline = ErisedPipeline(cfg)
        compare_device = next(compare_pipeline.pipe.mula.parameters()).device
        logger.info("Compare pipeline loaded on %s", compare_device)

    def _load_rate():
        nonlocal rate_pipeline
        if rate_pipeline is not None:
            return
        device = "cuda:0" if single_gpu else "cuda:1"
        logger.info("Loading RATE pipeline on %s...", device)
        cfg = ErisedConfig.from_env()
        cfg.mula_device = device
        cfg.codec_device = device
        cfg.lazy_load = False
        rate_pipeline = ErisedPipeline(cfg)
        logger.info("Rate pipeline loaded on %s", device)

    def ensure_mode(mode):
        """Switch to the requested mode, lazy-loading/unloading as needed (single-GPU only)."""
        nonlocal active_mode
        if not single_gpu:
            return  # both always loaded
        with mode_lock:
            if active_mode == mode:
                return
            # Unload the other pipeline
            if mode == "compare":
                _unload_pipeline("rate")
                _load_compare()
            elif mode == "rate":
                _unload_pipeline("compare")
                _load_rate()
            active_mode = mode
            logger.info("[single-gpu] Switched to %s mode", mode)

    if single_gpu:
        # Don't load anything yet — wait for first request
        logger.info("Single-GPU mode: pipelines will load on first request")
    else:
        # ── Multi-GPU: load both pipelines upfront ───────────────────────
        logger.info("=" * 60)
        logger.info("Loading COMPARE pipeline on cuda:0...")
        logger.info("=" * 60)
        _load_compare()

        logger.info("=" * 60)
        logger.info("Loading RATE pipeline on cuda:1...")
        logger.info("=" * 60)
        _load_rate()

    pref_store = PreferenceStore(base_config.dpo_db_path)
    logger.info("Preference DB: %s (%d pairs)", base_config.dpo_db_path, pref_store.count())

    # ── Compare state (GPU 0) ────────────────────────────────────────────
    compare_lock = threading.Lock()
    compare_current_weights = {"path": original_model_path}
    compare_stats = {"pending": 0, "completed": 0, "generating": False}
    compare_jobs = {}
    compare_queue = queue.Queue()

    def compare_swap_weights(target_path):
        if compare_current_weights["path"] == target_path:
            return
        logger.info("[compare] Swapping weights -> %s", target_path)
        load_safetensors_sharded(compare_pipeline.pipe.mula, target_path, device=compare_device)
        compare_current_weights["path"] = target_path
        torch.cuda.empty_cache()

    def compare_worker():
        """Background thread for compare jobs."""
        while True:
            job_id, prompt, lyrics, max_sec, model_name = compare_queue.get()
            compare_jobs[job_id]["status"] = "running"
            compare_stats["generating"] = True

            try:
                ensure_mode("compare")
                target_path = args.dpo_path if model_name == "dpo" else original_model_path
                with compare_lock:
                    compare_swap_weights(target_path)
                    t0 = time.time()
                    result = compare_pipeline.generate(
                        prompt=prompt,
                        lyrics=lyrics,
                        max_audio_length_ms=int(max_sec * 1000),
                    )
                    elapsed = time.time() - t0
                    logger.info("[compare/%s] Generated %s in %.1fs (%d frames)",
                                model_name, result.audio_path, elapsed, result.num_frames)

                compare_jobs[job_id]["status"] = "done"
                compare_jobs[job_id]["result"] = {
                    "audio_file": os.path.basename(result.audio_path),
                    "tags": result.tags_used,
                    "num_frames": result.num_frames,
                    "elapsed": round(elapsed, 1),
                    "model": model_name,
                }
                compare_stats["completed"] += 1

            except Exception as e:
                logger.exception("Compare generation error for job %s", job_id)
                compare_jobs[job_id]["status"] = "error"
                compare_jobs[job_id]["error"] = str(e)

            finally:
                compare_stats["pending"] = max(0, compare_stats["pending"] - 1)
                compare_stats["generating"] = compare_queue.qsize() > 0
                compare_queue.task_done()

    compare_thread = threading.Thread(target=compare_worker, daemon=True)
    compare_thread.start()
    logger.info("[compare] Worker thread started on GPU 0")

    # ── Rate state (GPU 1) ──────────────────────────────────────────────
    rate_gen_queue = queue.Queue()
    rate_ready_pairs = collections.deque()
    rate_gen_state = {"generating": False}
    rate_served_pairs = {}

    def rate_worker():
        """Background thread for rate pair generation."""
        while True:
            try:
                job = rate_gen_queue.get()
                rate_gen_state["generating"] = True
                prompt, lyrics, max_sec = job["prompt"], job["lyrics"], job["max_sec"]

                ensure_mode("rate")
                logger.info("[rate] Generating pair for: %s", prompt[:60])
                t0 = time.time()

                result_a = rate_pipeline.generate(
                    prompt=prompt, lyrics=lyrics,
                    max_audio_length_ms=int(max_sec * 1000),
                    temperature=0.8, cfg_scale=1.5,
                )
                logger.info("[rate/A] Generated %s in %.1fs", result_a.audio_path, time.time() - t0)

                t1 = time.time()
                result_b = rate_pipeline.generate(
                    prompt=prompt, lyrics=lyrics,
                    max_audio_length_ms=int(max_sec * 1000),
                    temperature=1.2, cfg_scale=0.8,
                )
                logger.info("[rate/B] Generated %s in %.1fs", result_b.audio_path, time.time() - t1)

                pair_id = f"{result_a.generation_id}_{result_b.generation_id}"
                pair = {
                    "pair_id": pair_id,
                    "prompt": prompt,
                    "lyrics": lyrics,
                    "a_id": result_a.generation_id,
                    "b_id": result_b.generation_id,
                    "a_audio": os.path.basename(result_a.audio_path),
                    "b_audio": os.path.basename(result_b.audio_path),
                    "a_tokens_path": result_a.tokens_path,
                    "b_tokens_path": result_b.tokens_path,
                    "tags_a": result_a.tags_used,
                    "tags_b": result_b.tags_used,
                }
                rate_ready_pairs.append(pair)
                logger.info("[rate] Pair ready: %s (total ready: %d)", pair_id, len(rate_ready_pairs))

            except Exception:
                logger.exception("[rate] Error in generation worker")
            finally:
                rate_gen_queue.task_done()
                rate_gen_state["generating"] = rate_gen_queue.qsize() > 0

    rate_thread = threading.Thread(target=rate_worker, daemon=True)
    rate_thread.start()
    logger.info("[rate] Worker thread started on GPU 1")

    # ── FastAPI app ─────────────────────────────────────────────────────
    app = FastAPI(title="Erised Unified Server")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # ── Shared: audio serving ───────────────────────────────────────────

    @app.get("/audio/{filename}")
    def serve_audio(filename: str):
        path = os.path.join(base_config.output_dir, filename)
        if not os.path.isfile(path):
            raise HTTPException(404, "Audio file not found")
        media = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(path, media_type=media)

    @app.get("/health")
    def health():
        if single_gpu:
            return {"status": "ok", "mode": "single-gpu", "active": active_mode, "gpu": "cuda:0"}
        return {"status": "ok", "compare_gpu": "cuda:0", "rate_gpu": "cuda:1"}

    # ── Compare endpoints (used by /generate page) ──────────────────────

    class SubmitRequest(BaseModel):
        prompt: str
        lyrics: str
        max_sec: int = 60
        model: str = "dpo"

    @app.post("/api/submit")
    def submit(req: SubmitRequest):
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")
        if req.model not in ("original", "dpo"):
            raise HTTPException(400, "Model must be 'original' or 'dpo'")

        job_id = str(uuid.uuid4())[:8]
        compare_jobs[job_id] = {"status": "pending"}
        compare_stats["pending"] += 1
        compare_queue.put((job_id, req.prompt, req.lyrics, req.max_sec, req.model))
        logger.info("[compare] Job %s submitted: model=%s, max_sec=%d", job_id, req.model, req.max_sec)
        return {"job_id": job_id}

    @app.get("/api/job/{job_id}")
    def get_job(job_id: str):
        if job_id not in compare_jobs:
            raise HTTPException(404, "Unknown job_id")
        return compare_jobs[job_id]

    @app.get("/api/info")
    def info():
        return {
            "original_path": original_model_path,
            "dpo_path": args.dpo_path,
            "mode": "single-gpu (lazy swap)" if single_gpu else "unified (compare on GPU 0, rate on GPU 1)",
        }

    # ── Rate endpoints (used by /rate page) ─────────────────────────────

    class QueueRequest(BaseModel):
        prompt: str
        lyrics: str
        max_sec: int = 60
        count: int = 3

    class RateRequest(BaseModel):
        pair_id: str
        choice: str

    @app.post("/api/queue")
    def queue_pairs(req: QueueRequest):
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")
        for _ in range(req.count):
            rate_gen_queue.put({
                "prompt": req.prompt,
                "lyrics": req.lyrics,
                "max_sec": req.max_sec,
            })
        logger.info("[rate] Queued %d pairs (queue size: %d)", req.count, rate_gen_queue.qsize())
        return {"queued": req.count, "total_pending": rate_gen_queue.qsize()}

    @app.get("/api/next")
    def next_pair():
        if rate_ready_pairs:
            pair = rate_ready_pairs.popleft()
            rate_served_pairs[pair["pair_id"]] = pair
            return {"status": "ready", "pair": pair}
        elif rate_gen_queue.qsize() > 0 or rate_gen_state["generating"]:
            return {"status": "generating"}
        else:
            return {"status": "empty"}

    @app.post("/api/rate")
    def rate(req: RateRequest):
        if req.choice not in ("a", "b"):
            raise HTTPException(400, "Choice must be 'a' or 'b'")

        pair = rate_served_pairs.get(req.pair_id)
        if not pair:
            raise HTTPException(400, f"Unknown pair_id: {req.pair_id}")

        if req.choice == "a":
            winner_id, loser_id = pair["a_id"], pair["b_id"]
            winner_tokens, loser_tokens = pair["a_tokens_path"], pair["b_tokens_path"]
        else:
            winner_id, loser_id = pair["b_id"], pair["a_id"]
            winner_tokens, loser_tokens = pair["b_tokens_path"], pair["a_tokens_path"]

        pref_store.add_preference(
            pair_id=req.pair_id,
            prompt=pair["prompt"],
            lyrics=pair["lyrics"],
            winner_id=winner_id,
            loser_id=loser_id,
            winner_tokens_path=winner_tokens,
            loser_tokens_path=loser_tokens,
        )
        count = pref_store.count()
        logger.info("[rate] Rated pair %s — winner: %s (total: %d)", req.pair_id, winner_id, count)

        del rate_served_pairs[req.pair_id]
        return {"count": count}

    @app.post("/api/undo")
    def undo():
        all_prefs = pref_store.get_all()
        if all_prefs:
            pref_store.delete_preference(all_prefs[-1].pair_id)
        return {"count": pref_store.count()}

    @app.get("/api/status")
    def status():
        return {
            "pending": rate_gen_queue.qsize(),
            "ready": len(rate_ready_pairs),
            "generating": rate_gen_state["generating"],
            "rated": pref_store.count(),
        }

    # ── Launch ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  UNIFIED SERVER READY")
    if single_gpu:
        logger.info("  Single-GPU mode: pipeline loads on first request")
        logger.info("  Switching between compare/rate unloads the other")
    else:
        logger.info("  Compare (original vs DPO): GPU 0  [/api/submit, /api/job]")
        logger.info("  Rate (preference collection): GPU 1  [/api/queue, /api/next, /api/rate]")
    logger.info("  Port: %d", args.port)
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
