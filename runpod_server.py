"""
Standalone GPU server for Erised — runs on a single A100 (RunPod).

Uses pause-decode-resume for streaming: pauses generation at checkpoints,
decodes audio on the same GPU, resumes. First audio in ~18s.

Run:
    python runpod_server.py

The API matches modal_app.py so the frontend works with either backend.
Set GPU_URL in the frontend to point here instead of Modal.
"""

import os
import json
import logging
import queue
import threading
import time
import uuid

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("erised.server")

# ── Paths ────────────────────────────────────────────────────────────
model_path = os.environ.get("ERISED_MODEL_PATH", "/workspace/heartlib/ckpt")
dpo_path = os.environ.get("ERISED_DPO_PATH", "/workspace/dpo_checkpoints_v11/dpo_best")
output_dir = os.environ.get("ERISED_OUTPUT_DIR", "/workspace/erised_data/outputs")
jobs_dir = os.environ.get("ERISED_JOBS_DIR", "/workspace/erised_data/jobs")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(jobs_dir, exist_ok=True)

os.environ.setdefault("ERISED_MODEL_PATH", model_path)
os.environ.setdefault("ERISED_OUTPUT_DIR", output_dir)

# ── Load pipeline ────────────────────────────────────────────────────
from erised.config import ErisedConfig
from erised.pipeline import ErisedPipeline

config = ErisedConfig.from_env()
config.model_path = model_path
config.output_dir = output_dir
config.lazy_load = False

logger.info("Loading pipeline from %s ...", model_path)
pipeline = ErisedPipeline(config)
logger.info("Pipeline loaded.")

logger.info("Initializing DPO Guided from %s ...", dpo_path)
pipeline.init_guided(dpo_checkpoint_path=dpo_path)
logger.info("DPO Guided ready.")

# ── Job management ───────────────────────────────────────────────────
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
gen_lock = threading.Lock()
gen_stats = {"pending": 0, "completed": 0, "generating": False}
job_queue: queue.Queue = queue.Queue()

DPO_USER_EMAIL = os.environ.get("ERISED_DPO_USER_EMAIL", "zsean@berkeley.edu")


def _job_path(job_id: str) -> str:
    return os.path.join(jobs_dir, f"{job_id}.json")


def _save_job(job_id: str):
    data = jobs.get(job_id)
    if not data:
        return
    try:
        tmp = _job_path(job_id) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, _job_path(job_id))
    except OSError:
        pass


# ── Generation worker ────────────────────────────────────────────────
def generation_worker():
    while True:
        job_id, prompt, lyrics, max_sec, model_name, dpo_scale = job_queue.get()
        with jobs_lock:
            jobs[job_id]["status"] = "running"
            _save_job(job_id)
        gen_stats["generating"] = True

        audio_filename = f"{job_id}.wav"
        last_save = [0.0]

        def on_progress(current_frame, total_frames, partial_audio_file=None, partial_version=None,
                        chunk_paths=None):
            with jobs_lock:
                jobs[job_id]["progress"] = {
                    "current_frame": current_frame,
                    "total_frames": total_frames,
                }
                if partial_audio_file:
                    jobs[job_id]["partial_audio_file"] = partial_audio_file
                if partial_version is not None:
                    jobs[job_id]["partial_version"] = partial_version
                if chunk_paths is not None:
                    jobs[job_id]["chunk_paths"] = chunk_paths
                now = time.time()
                if now - last_save[0] > 2:
                    _save_job(job_id)
                    last_save[0] = now

        try:
            with gen_lock:
                t0 = time.time()
                if model_name == "dpo":
                    gen_result = pipeline.generate_guided(
                        prompt=prompt, lyrics=lyrics,
                        max_audio_length_ms=int(max_sec * 1000),
                        dpo_scale=dpo_scale,
                        on_progress=on_progress,
                        streaming_decode=True,
                        streaming_first_chunk=540,
                        streaming_lean_gc=True,
                    )
                else:
                    gen_result = pipeline.generate(
                        prompt=prompt, lyrics=lyrics,
                        max_audio_length_ms=int(max_sec * 1000),
                        on_progress=on_progress,
                        streaming_decode=True,
                        streaming_first_chunk=540,
                        streaming_lean_gc=True,
                    )
                elapsed = time.time() - t0
                logger.info("[%s] Generated + decoded in %.1fs", model_name, elapsed)

            actual_audio = os.path.basename(gen_result.audio_path)
            with jobs_lock:
                # Keep partial_audio_file/version so the SSE can still
                # send the last chunk event before the done event.
                # The done event carries the final audio_file anyway.
                jobs[job_id]["status"] = "done"
                jobs[job_id]["result"] = {
                    "audio_file": actual_audio,
                    "tags": gen_result.tags_used,
                    "num_frames": gen_result.num_frames,
                    "elapsed": round(elapsed, 1),
                    "model": model_name,
                }
                _save_job(job_id)
            gen_stats["completed"] += 1
        except Exception as e:
            logger.exception("Generation error for job %s", job_id)
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
                _save_job(job_id)
        finally:
            gen_stats["pending"] = max(0, gen_stats["pending"] - 1)
            gen_stats["generating"] = job_queue.qsize() > 0
            job_queue.task_done()


worker = threading.Thread(target=generation_worker, daemon=True)
worker.start()

# ── FastAPI app ──────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Erised GPU (RunPod)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class SubmitRequest(BaseModel):
    prompt: str
    lyrics: str
    max_sec: int = 60
    model: str = "dpo"
    dpo_scale: float = 3.0
    user_email: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": "dpo_guided"}


@app.post("/api/submit")
def submit(req: SubmitRequest):
    if not req.prompt.strip() or not req.lyrics.strip():
        raise HTTPException(400, "Prompt and lyrics required")
    if req.model not in ("original", "dpo"):
        raise HTTPException(400, "Model must be 'original' or 'dpo'")

    effective_model = req.model
    if req.model == "dpo" and req.user_email != DPO_USER_EMAIL:
        effective_model = "original"

    job_id = str(uuid.uuid4())[:8]
    with jobs_lock:
        jobs[job_id] = {"status": "pending"}
        _save_job(job_id)
    gen_stats["pending"] += 1
    job_queue.put((job_id, req.prompt, req.lyrics, req.max_sec, effective_model, req.dpo_scale))
    logger.info("Job %s: model=%s, user=%s", job_id, effective_model, req.user_email)
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        if job_id in jobs:
            return jobs[job_id]
    raise HTTPException(404, "Unknown job_id")


@app.get("/api/job/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE endpoint — pushes events as soon as chunks are ready."""
    import asyncio
    from starlette.responses import StreamingResponse

    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Unknown job_id")

    async def event_generator():
        last_version = 0
        last_frame = 0
        heartbeat_counter = 0
        while True:
            with jobs_lock:
                job = jobs.get(job_id, {})
            status = job.get("status", "pending")
            progress = job.get("progress", {})
            cur_frame = progress.get("current_frame", 0)
            total = progress.get("total_frames", 0)
            partial = job.get("partial_audio_file")
            version = job.get("partial_version", 0)

            sent_event = False

            # Send progress update when frames advance
            if cur_frame > last_frame:
                last_frame = cur_frame
                yield f"event: progress\ndata: {json.dumps({'current_frame': cur_frame, 'total_frames': total})}\n\n"
                sent_event = True

            # Send chunk_ready when new chunks are decoded
            if version > last_version and partial:
                last_version = version
                chunk_data = {'audio_file': partial, 'version': version}
                # Include individual chunk file paths so the client can
                # download just the new chunk (~7MB) instead of the full
                # cumulative file (~16MB) — halves download time.
                cp = job.get("chunk_paths")
                if cp:
                    chunk_data["chunk_files"] = cp
                yield f"event: chunk\ndata: {json.dumps(chunk_data)}\n\n"
                sent_event = True

            if status == "done":
                result = job.get("result", {})
                yield f"event: done\ndata: {json.dumps(result)}\n\n"
                return
            if status == "error":
                yield f"event: error\ndata: {json.dumps({'error': job.get('error', 'unknown')})}\n\n"
                return

            # Heartbeat every ~2s to keep the connection alive through
            # cloudflared tunnel during long decode pauses.
            heartbeat_counter += 1
            if not sent_event and heartbeat_counter >= 20:
                yield f"event: heartbeat\ndata: {{}}\n\n"
                heartbeat_counter = 0

            await asyncio.sleep(0.1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    path = os.path.join(output_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Audio file not found")
    media = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
    return FileResponse(path, media_type=media)


@app.get("/api/status")
def status():
    with rate_lock:
        ready_n = len(rate_ready)
    return {
        "pending": gen_stats["pending"],
        "completed": gen_stats["completed"],
        "generating": gen_stats["generating"],
        "ready": ready_n,
        "rated": rate_stats["rated"],
    }


# ── Rate / A-B pair rating ──────────────────────────────────────────
import random
import sqlite3
import subprocess

DPO_DB_PATH = os.environ.get(
    "ERISED_DPO_DB", "/workspace/heartlib/heartlib/dpo_preferences.db"
)
GCS_PREFS_PREFIX = os.environ.get(
    "ERISED_GCS_PREFS_PREFIX", "gcs:erised-dpo/preferences"
)
GCS_KEY = os.environ.get("ERISED_GCS_KEY", "/workspace/key.json")
GCS_PROJECT = os.environ.get("ERISED_GCS_PROJECT", "gen-lang-client-0191019282")

rate_pending: dict[str, dict] = {}
rate_ready: list[dict] = []
rate_stats = {"rated": 0}
rate_lock = threading.Lock()

try:
    _c = sqlite3.connect(DPO_DB_PATH)
    rate_stats["rated"] = _c.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
    _c.close()
    logger.info("Loaded rated count from %s: %d", DPO_DB_PATH, rate_stats["rated"])
except Exception as _e:
    logger.warning("Could not read preference count from %s: %s", DPO_DB_PATH, _e)


class QueueRequest(BaseModel):
    prompt: str
    lyrics: str
    max_sec: int = 60
    count: int = 1
    mode: str = "orig_vs_dpo"  # or "orig_vs_orig"
    user_email: str | None = None


class RateVote(BaseModel):
    pair_id: str
    choice: str
    user_email: str | None = None


def _resolve_ready_pair(pair_id: str, info: dict):
    """Return a display-ready pair dict if both jobs are done, else None."""
    with jobs_lock:
        a = jobs.get(info["a_job"], {})
        b = jobs.get(info["b_job"], {})
    if a.get("status") != "done" or b.get("status") != "done":
        return None
    # Randomize A/B slot assignment so the rater can't bias on model position.
    if random.random() < 0.5:
        slot_a, slot_b = a["result"], b["result"]
        m_a, m_b = info["a_model"], info["b_model"]
    else:
        slot_a, slot_b = b["result"], a["result"]
        m_a, m_b = info["b_model"], info["a_model"]
    return {
        "pair_id": pair_id,
        "prompt": info["prompt"],
        "lyrics": info["lyrics"],
        "a_audio": slot_a["audio_file"],
        "b_audio": slot_b["audio_file"],
        "tags_a": slot_a["tags"],
        "tags_b": slot_b["tags"],
        "a_model": m_a,
        "b_model": m_b,
        "mode": info["mode"],
    }


@app.post("/api/queue")
def queue_pairs(req: QueueRequest):
    if not req.prompt.strip() or not req.lyrics.strip():
        raise HTTPException(400, "Prompt and lyrics required")
    if req.mode not in ("orig_vs_dpo", "orig_vs_orig"):
        raise HTTPException(400, "mode must be orig_vs_dpo or orig_vs_orig")

    mode = req.mode
    # DPO model is gated to the authorized user; downgrade silently for guests
    if mode == "orig_vs_dpo" and req.user_email != DPO_USER_EMAIL:
        mode = "orig_vs_orig"

    if mode == "orig_vs_dpo":
        a_model, b_model = "original", "dpo"
    else:
        a_model, b_model = "original", "original"

    n = max(1, min(req.count, 10))
    for _ in range(n):
        pair_id = str(uuid.uuid4())[:8]
        a_job = str(uuid.uuid4())[:8]
        b_job = str(uuid.uuid4())[:8]
        with jobs_lock:
            jobs[a_job] = {"status": "pending"}
            jobs[b_job] = {"status": "pending"}
            _save_job(a_job)
            _save_job(b_job)
        gen_stats["pending"] += 2

        with rate_lock:
            rate_pending[pair_id] = {
                "prompt": req.prompt,
                "lyrics": req.lyrics,
                "a_job": a_job,
                "b_job": b_job,
                "a_model": a_model,
                "b_model": b_model,
                "user_email": req.user_email,
                "mode": mode,
            }

        job_queue.put((a_job, req.prompt, req.lyrics, req.max_sec, a_model, 3.0))
        job_queue.put((b_job, req.prompt, req.lyrics, req.max_sec, b_model, 3.0))

    return {"queued": n, "mode": mode}


@app.get("/api/next")
def next_pair():
    with rate_lock:
        for pid, info in list(rate_pending.items()):
            pair = _resolve_ready_pair(pid, info)
            if pair is not None:
                rate_ready.append(pair)
                del rate_pending[pid]
        if rate_ready:
            return {"status": "ready", "pair": rate_ready[0]}
        if rate_pending or gen_stats["generating"]:
            return {"status": "generating"}
        return {"status": "empty"}


@app.get("/api/pair/{pair_id}")
def get_pair(pair_id: str):
    with rate_lock:
        for p in rate_ready:
            if p["pair_id"] == pair_id:
                return {
                    "status": "ready",
                    "pair": {**p, "a_ready": True, "b_ready": True},
                }
        info = rate_pending.get(pair_id)
    if info is None:
        raise HTTPException(404, "Unknown pair_id")
    pair = _resolve_ready_pair(pair_id, info)
    if pair is not None:
        with rate_lock:
            rate_pending.pop(pair_id, None)
            rate_ready.append(pair)
        return {
            "status": "ready",
            "pair": {**pair, "a_ready": True, "b_ready": True},
        }
    return {"status": "generating"}


def _save_pref_sqlite(row: dict):
    conn = sqlite3.connect(DPO_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO preferences (
                pair_id, prompt, lyrics, winner_id, loser_id,
                winner_tokens_path, loser_tokens_path, rater_id, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["pair_id"],
                row["prompt"],
                row["lyrics"],
                row["winner_id"],
                row["loser_id"],
                row["winner_tokens_path"],
                row["loser_tokens_path"],
                row["rater_id"],
                row["timestamp"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _save_pref_gcs(row: dict):
    """Best-effort upload of preference row + token files to GCS."""
    env = {
        **os.environ,
        "RCLONE_CONFIG_GCS_TYPE": "google cloud storage",
        "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_FILE": GCS_KEY,
        "RCLONE_CONFIG_GCS_PROJECT_NUMBER": GCS_PROJECT,
        "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY": "true",
    }
    tmp = f"/tmp/pref_{row['pair_id']}.json"
    try:
        with open(tmp, "w") as f:
            json.dump(row, f)
        subprocess.run(
            ["rclone", "copyto", tmp, f"{GCS_PREFS_PREFIX}/{row['pair_id']}.json"],
            env=env, timeout=30, check=False,
        )
    except Exception as e:
        logger.warning("GCS upload of pref row failed: %s", e)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    for p in (row["winner_tokens_path"], row["loser_tokens_path"]):
        if not os.path.isfile(p):
            logger.warning("Token file missing for GCS upload: %s", p)
            continue
        name = os.path.basename(p)
        try:
            subprocess.run(
                ["rclone", "copyto", p, f"{GCS_PREFS_PREFIX}/tokens/{name}"],
                env=env, timeout=180, check=False,
            )
        except Exception as e:
            logger.warning("GCS upload of token file %s failed: %s", name, e)


@app.post("/api/rate")
def rate_pair(req: RateVote):
    if req.choice not in ("a", "b"):
        raise HTTPException(400, "Choice must be 'a' or 'b'")
    with rate_lock:
        pair = None
        for i, p in enumerate(rate_ready):
            if p["pair_id"] == req.pair_id:
                pair = rate_ready.pop(i)
                break
    if pair is None:
        raise HTTPException(404, "Pair not found")

    if req.choice == "a":
        winner_audio, loser_audio = pair["a_audio"], pair["b_audio"]
        winner_model, loser_model = pair["a_model"], pair["b_model"]
    else:
        winner_audio, loser_audio = pair["b_audio"], pair["a_audio"]
        winner_model, loser_model = pair["b_model"], pair["a_model"]

    winner_id = winner_audio.rsplit(".", 1)[0]
    loser_id = loser_audio.rsplit(".", 1)[0]
    winner_tokens = os.path.join(output_dir, f"{winner_id}_tokens.pt")
    loser_tokens = os.path.join(output_dir, f"{loser_id}_tokens.pt")

    row = {
        "pair_id": f"{winner_id}_{loser_id}",
        "prompt": pair["prompt"],
        "lyrics": pair["lyrics"],
        "winner_id": winner_id,
        "loser_id": loser_id,
        "winner_tokens_path": winner_tokens,
        "loser_tokens_path": loser_tokens,
        "rater_id": req.user_email or "anonymous",
        "timestamp": time.time(),
        "mode": pair.get("mode"),
        "winner_model": winner_model,
        "loser_model": loser_model,
    }

    try:
        _save_pref_sqlite(row)
    except Exception as e:
        logger.exception("SQLite write failed for pair %s", req.pair_id)
        raise HTTPException(500, f"SQLite write failed: {e}")

    threading.Thread(target=_save_pref_gcs, args=(row,), daemon=True).start()

    rate_stats["rated"] += 1
    logger.info(
        "Rating: pair=%s winner=%s(%s) loser=%s(%s) mode=%s user=%s",
        req.pair_id, winner_id, winner_model, loser_id, loser_model,
        pair.get("mode"), req.user_email,
    )
    return {"status": "recorded", "count": rate_stats["rated"]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
