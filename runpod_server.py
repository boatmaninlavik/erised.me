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
model_path = os.environ.get("ERISED_MODEL_PATH", "/data/ckpt")
dpo_path = os.environ.get("ERISED_DPO_PATH", "/data/dpo_checkpoints_v11/dpo_best")
output_dir = os.environ.get("ERISED_OUTPUT_DIR", "/data/outputs")
jobs_dir = os.environ.get("ERISED_JOBS_DIR", "/data/jobs")

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

        def on_progress(current_frame, total_frames, partial_audio_file=None, partial_version=None):
            with jobs_lock:
                jobs[job_id]["progress"] = {
                    "current_frame": current_frame,
                    "total_frames": total_frames,
                }
                if partial_audio_file:
                    jobs[job_id]["partial_audio_file"] = partial_audio_file
                if partial_version is not None:
                    jobs[job_id]["partial_version"] = partial_version
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
                    )
                else:
                    gen_result = pipeline.generate(
                        prompt=prompt, lyrics=lyrics,
                        max_audio_length_ms=int(max_sec * 1000),
                        on_progress=on_progress,
                        streaming_decode=True,
                    )
                elapsed = time.time() - t0
                logger.info("[%s] Generated + decoded in %.1fs", model_name, elapsed)

            with jobs_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["result"] = {
                    "audio_file": audio_filename,
                    "tags": gen_result.tags_used,
                    "num_frames": gen_result.num_frames,
                    "elapsed": round(elapsed, 1),
                    "model": model_name,
                }
                jobs[job_id].pop("partial_audio_file", None)
                jobs[job_id].pop("partial_version", None)
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


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    path = os.path.join(output_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Audio file not found")
    media = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
    return FileResponse(path, media_type=media)


@app.get("/api/status")
def status():
    return {
        "pending": gen_stats["pending"],
        "completed": gen_stats["completed"],
        "generating": gen_stats["generating"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
