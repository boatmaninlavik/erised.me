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
        job_id, prompt, lyrics, max_sec, model_name, dpo_scale, temperature = job_queue.get()
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
                        temperature=temperature,
                        on_progress=on_progress,
                        streaming_decode=True,
                        streaming_first_chunk=540,
                        streaming_lean_gc=True,
                    )
                else:
                    gen_result = pipeline.generate(
                        prompt=prompt, lyrics=lyrics,
                        max_audio_length_ms=int(max_sec * 1000),
                        temperature=temperature,
                        on_progress=on_progress,
                        streaming_decode=True,
                        streaming_first_chunk=540,
                        streaming_lean_gc=True,
                    )
                elapsed = time.time() - t0
                logger.info("[%s temp=%.2f] Generated + decoded in %.1fs", model_name, temperature, elapsed)

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
    job_queue.put((job_id, req.prompt, req.lyrics, req.max_sec, effective_model, req.dpo_scale, 1.0))
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
        ready_n = len(rate_pairs)
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

rate_pairs: dict[str, dict] = {}  # pair_id -> info, insertion-ordered FIFO queue
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


def _pair_view(pair_id: str, info: dict) -> dict:
    """Render a pair for /api/next and /api/pair responses.

    Each slot exposes the underlying job_id so the frontend can SSE-stream
    chunks via /api/job/{id}/stream just like the /generate page does. Audio
    filenames + tags are filled in lazily once each job's result lands, so the
    pair becomes visible in the UI immediately and chunks arrive over SSE.
    """
    with jobs_lock:
        a_job_state = jobs.get(info["slot_a_job"], {})
        b_job_state = jobs.get(info["slot_b_job"], {})
    a_result = a_job_state.get("result") or {}
    b_result = b_job_state.get("result") or {}
    return {
        "pair_id": pair_id,
        "prompt": info["prompt"],
        "lyrics": info["lyrics"],
        "mode": info["mode"],
        "a_job": info["slot_a_job"],
        "b_job": info["slot_b_job"],
        "a_audio": a_result.get("audio_file"),
        "b_audio": b_result.get("audio_file"),
        "tags_a": a_result.get("tags"),
        "tags_b": b_result.get("tags"),
        "a_model": info["slot_a_model"],
        "b_model": info["slot_b_model"],
        "a_status": a_job_state.get("status", "pending"),
        "b_status": b_job_state.get("status", "pending"),
    }


# Per-slot temperature spread for orig_vs_orig: one slot generates "safer"
# (lower temp), the other "more adventurous" (higher temp). Picked tight enough
# that both slots stay coherent — wide spreads (>0.4) push the high-temp slot
# into nonsense and the low-temp slot into monotone, which makes ratings less
# about taste and more about "which one is broken".
ORIG_TEMP_LOW = 0.85
ORIG_TEMP_HIGH = 1.15


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
        gen_a_model, gen_b_model = "original", "dpo"
        # In orig_vs_dpo the variable being rated IS the model, so keep
        # temperature fixed across both slots to avoid confounding signal.
        gen_a_temp, gen_b_temp = 1.0, 1.0
    else:
        gen_a_model, gen_b_model = "original", "original"
        # In orig_vs_orig both slots are the same model — vary temperature
        # so the user is rating a meaningful "creative vs safer" contrast
        # instead of two near-identical seed rolls.
        gen_a_temp, gen_b_temp = ORIG_TEMP_LOW, ORIG_TEMP_HIGH

    n = max(1, min(req.count, 10))
    for _ in range(n):
        pair_id = str(uuid.uuid4())[:8]
        gen_a_job = str(uuid.uuid4())[:8]
        gen_b_job = str(uuid.uuid4())[:8]
        with jobs_lock:
            jobs[gen_a_job] = {"status": "pending"}
            jobs[gen_b_job] = {"status": "pending"}
            _save_job(gen_a_job)
            _save_job(gen_b_job)
        gen_stats["pending"] += 2

        # Randomize which generator job is shown as slot A vs slot B at queue
        # time so we don't have to wait until both are done to assign slots.
        if random.random() < 0.5:
            slot_a_job, slot_b_job = gen_a_job, gen_b_job
            slot_a_model, slot_b_model = gen_a_model, gen_b_model
            slot_a_temp, slot_b_temp = gen_a_temp, gen_b_temp
        else:
            slot_a_job, slot_b_job = gen_b_job, gen_a_job
            slot_a_model, slot_b_model = gen_b_model, gen_a_model
            slot_a_temp, slot_b_temp = gen_b_temp, gen_a_temp

        with rate_lock:
            rate_pairs[pair_id] = {
                "prompt": req.prompt,
                "lyrics": req.lyrics,
                "slot_a_job": slot_a_job,
                "slot_b_job": slot_b_job,
                "slot_a_model": slot_a_model,
                "slot_b_model": slot_b_model,
                "slot_a_temp": slot_a_temp,
                "slot_b_temp": slot_b_temp,
                "user_email": req.user_email,
                "mode": mode,
            }

        job_queue.put((gen_a_job, req.prompt, req.lyrics, req.max_sec, gen_a_model, 3.0, gen_a_temp))
        job_queue.put((gen_b_job, req.prompt, req.lyrics, req.max_sec, gen_b_model, 3.0, gen_b_temp))

    return {"queued": n, "mode": mode}


@app.get("/api/next")
def next_pair():
    """Return the front of the rate queue. The pair is shown in the UI as
    soon as it's been queued; per-song chunks arrive over /api/job/{id}/stream."""
    with rate_lock:
        if rate_pairs:
            pid, info = next(iter(rate_pairs.items()))
            return {"status": "ready", "pair": _pair_view(pid, info)}
        if gen_stats["generating"]:
            return {"status": "generating"}
        return {"status": "empty"}


@app.get("/api/pair/{pair_id}")
def get_pair(pair_id: str):
    with rate_lock:
        info = rate_pairs.get(pair_id)
    if info is None:
        raise HTTPException(404, "Unknown pair_id")
    return {"status": "ready", "pair": _pair_view(pair_id, info)}


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
        info = rate_pairs.get(req.pair_id)
    if info is None:
        raise HTTPException(404, "Pair not found")

    if req.choice == "a":
        winner_job = info["slot_a_job"]
        loser_job = info["slot_b_job"]
        winner_model = info["slot_a_model"]
        loser_model = info["slot_b_model"]
        winner_temp = info.get("slot_a_temp", 1.0)
        loser_temp = info.get("slot_b_temp", 1.0)
    else:
        winner_job = info["slot_b_job"]
        loser_job = info["slot_a_job"]
        winner_model = info["slot_b_model"]
        loser_model = info["slot_a_model"]
        winner_temp = info.get("slot_b_temp", 1.0)
        loser_temp = info.get("slot_a_temp", 1.0)

    with jobs_lock:
        winner_state = jobs.get(winner_job, {})
        loser_state = jobs.get(loser_job, {})
    winner_result = winner_state.get("result")
    loser_result = loser_state.get("result")
    if not winner_result or not loser_result:
        raise HTTPException(409, "Both songs must finish generating before rating")

    winner_audio = winner_result["audio_file"]
    loser_audio = loser_result["audio_file"]
    winner_id = winner_audio.rsplit(".", 1)[0]
    loser_id = loser_audio.rsplit(".", 1)[0]
    winner_tokens = os.path.join(output_dir, f"{winner_id}_tokens.pt")
    loser_tokens = os.path.join(output_dir, f"{loser_id}_tokens.pt")

    row = {
        "pair_id": f"{winner_id}_{loser_id}",
        "prompt": info["prompt"],
        "lyrics": info["lyrics"],
        "winner_id": winner_id,
        "loser_id": loser_id,
        "winner_tokens_path": winner_tokens,
        "loser_tokens_path": loser_tokens,
        "rater_id": req.user_email or "anonymous",
        "timestamp": time.time(),
        "mode": info.get("mode"),
        "winner_model": winner_model,
        "loser_model": loser_model,
        "winner_temp": winner_temp,
        "loser_temp": loser_temp,
    }

    try:
        _save_pref_sqlite(row)
    except Exception as e:
        logger.exception("SQLite write failed for pair %s", req.pair_id)
        raise HTTPException(500, f"SQLite write failed: {e}")

    threading.Thread(target=_save_pref_gcs, args=(row,), daemon=True).start()

    with rate_lock:
        rate_pairs.pop(req.pair_id, None)
    rate_stats["rated"] += 1
    logger.info(
        "Rating: pair=%s winner=%s(%s,t=%.2f) loser=%s(%s,t=%.2f) mode=%s user=%s",
        req.pair_id, winner_id, winner_model, winner_temp,
        loser_id, loser_model, loser_temp,
        info.get("mode"), req.user_email,
    )
    return {"status": "recorded", "count": rate_stats["rated"]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
