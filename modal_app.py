"""
Modal serverless deployment for Erised GPU backend.

Deploys the generation server as a Modal web endpoint with:
- Auto cold-start: container boots when users visit erised.me (~45s)
- Auto shutdown: container stops after 5 min idle (no charges)
- Permanent URL: stored once in Supabase, never changes

Deploy:
    modal deploy modal_app.py
"""

import os
import modal

app = modal.App("erised-gpu")

# Volume for model weights + DPO checkpoints + generated outputs
erised_vol = modal.Volume.from_name("erised-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        # heartlib dependencies (pinned from pyproject.toml)
        "numpy==2.0.2",
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "torchtune==0.4.0",
        "torchao==0.9.0",
        "torchvision==0.19.1",
        "tqdm==4.67.1",
        "transformers==4.57.0",
        "tokenizers==0.22.1",
        "einops==0.8.1",
        "accelerate==1.12.0",
        "bitsandbytes==0.49.0",
        "vector-quantize-pytorch==1.27.15",
        "modelscope==1.33.0",
        "soundfile",
        "safetensors",
        # erised dependencies
        "fastapi",
        "uvicorn[standard]",
        "pydantic>=2.0",
        "openai>=1.0",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # Copy heartlib source into the image
    .add_local_dir("/workspace/heartlib/src/heartlib", remote_path="/root/heartlib_pkg/heartlib")
    # Copy erised package into the image
    .add_local_dir("erised", remote_path="/root/erised")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": erised_vol},
    timeout=600,
    scaledown_window=300,  # 5 min idle → shut down
    secrets=[modal.Secret.from_name("erised-secrets")],
)
@modal.asgi_app()
def serve():
    """Create and return the FastAPI app."""
    import sys
    import logging
    import queue
    import threading
    import time
    import uuid

    # Make heartlib and erised importable
    sys.path.insert(0, "/root/heartlib_pkg")
    sys.path.insert(0, "/root")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("erised.modal")

    from erised.config import ErisedConfig
    from erised.pipeline import ErisedPipeline

    # Paths inside the Modal volume
    model_path = os.environ.get("ERISED_MODEL_PATH", "/data/ckpt")
    dpo_path = os.environ.get("ERISED_DPO_PATH", "/data/dpo_checkpoints_v11/dpo_best")
    output_dir = "/data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    os.environ.setdefault("ERISED_MODEL_PATH", model_path)
    os.environ.setdefault("ERISED_OUTPUT_DIR", output_dir)

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

    # ── Job queue (same as compare_local.py) ──────────────────────────
    gen_lock = threading.Lock()
    gen_stats = {"pending": 0, "completed": 0, "generating": False}
    jobs = {}
    job_queue = queue.Queue()

    DPO_USER_EMAIL = os.environ.get("ERISED_DPO_USER_EMAIL", "zsean@berkeley.edu")

    def generation_worker():
        while True:
            job_id, prompt, lyrics, max_sec, model_name, dpo_scale = job_queue.get()
            jobs[job_id]["status"] = "running"
            gen_stats["generating"] = True

            def on_progress(current_frame, total_frames, partial_audio_file=None):
                jobs[job_id]["progress"] = {
                    "current_frame": current_frame,
                    "total_frames": total_frames,
                }
                if partial_audio_file:
                    jobs[job_id]["partial_audio_file"] = partial_audio_file

            try:
                with gen_lock:
                    t0 = time.time()
                    if model_name == "dpo":
                        result = pipeline.generate_guided(
                            prompt=prompt, lyrics=lyrics,
                            max_audio_length_ms=int(max_sec * 1000),
                            dpo_scale=dpo_scale,
                            on_progress=on_progress,
                        )
                    else:
                        result = pipeline.generate(
                            prompt=prompt, lyrics=lyrics,
                            max_audio_length_ms=int(max_sec * 1000),
                            on_progress=on_progress,
                        )
                    elapsed = time.time() - t0
                    logger.info("[%s] Generated %s in %.1fs (%d frames)",
                                model_name, result.audio_path, elapsed, result.num_frames)

                jobs[job_id]["status"] = "done"
                jobs[job_id]["result"] = {
                    "audio_file": os.path.basename(result.audio_path),
                    "tags": result.tags_used,
                    "num_frames": result.num_frames,
                    "elapsed": round(elapsed, 1),
                    "model": model_name,
                }
                gen_stats["completed"] += 1
            except Exception as e:
                logger.exception("Generation error for job %s", job_id)
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
            finally:
                gen_stats["pending"] = max(0, gen_stats["pending"] - 1)
                gen_stats["generating"] = job_queue.qsize() > 0
                job_queue.task_done()

    worker = threading.Thread(target=generation_worker, daemon=True)
    worker.start()

    # ── FastAPI app ───────────────────────────────────────────────────
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    fapi = FastAPI(title="Erised GPU")
    fapi.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class SubmitRequest(BaseModel):
        prompt: str
        lyrics: str
        max_sec: int = 60
        model: str = "dpo"
        dpo_scale: float = 3.0
        user_email: str | None = None

    @fapi.get("/health")
    def health():
        return {"status": "ok", "model": "dpo_guided"}

    @fapi.post("/api/submit")
    def submit(req: SubmitRequest):
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")
        if req.model not in ("original", "dpo"):
            raise HTTPException(400, "Model must be 'original' or 'dpo'")

        effective_model = req.model
        if req.model == "dpo" and req.user_email != DPO_USER_EMAIL:
            effective_model = "original"

        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {"status": "pending"}
        gen_stats["pending"] += 1
        job_queue.put((job_id, req.prompt, req.lyrics, req.max_sec, effective_model, req.dpo_scale))
        logger.info("Job %s: model=%s, user=%s", job_id, effective_model, req.user_email)
        return {"job_id": job_id}

    @fapi.get("/api/job/{job_id}")
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Unknown job_id")
        return jobs[job_id]

    @fapi.get("/api/status")
    def status():
        return {
            "pending": gen_stats["pending"],
            "completed": gen_stats["completed"],
            "generating": gen_stats["generating"],
            "ready": len(rate_ready_pairs),
            "rated": rate_stats["rated"],
        }

    @fapi.get("/audio/{filename}")
    def serve_audio(filename: str):
        path = os.path.join(output_dir, filename)
        if not os.path.isfile(path):
            raise HTTPException(404, "Audio file not found")
        media = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(path, media_type=media)

    # ── Rating endpoints ──────────────────────────────────────────────
    rate_ready_pairs = []
    rate_pending_pairs = {}
    rate_stats = {"rated": 0}

    class QueueRequest(BaseModel):
        prompt: str
        lyrics: str
        max_sec: int = 60
        count: int = 1
        user_email: str | None = None

    class RateVote(BaseModel):
        pair_id: str
        choice: str
        user_email: str | None = None

    @fapi.post("/api/queue")
    def queue_pairs(req: QueueRequest):
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")

        use_dpo = req.user_email == DPO_USER_EMAIL
        second_model = "dpo" if use_dpo else "original"

        for _ in range(min(req.count, 10)):
            pair_id = str(uuid.uuid4())[:8]
            orig_job_id = str(uuid.uuid4())[:8]
            dpo_job_id = str(uuid.uuid4())[:8]

            jobs[orig_job_id] = {"status": "pending"}
            jobs[dpo_job_id] = {"status": "pending"}
            gen_stats["pending"] += 2

            rate_pending_pairs[pair_id] = {
                "prompt": req.prompt,
                "orig_job": orig_job_id,
                "dpo_job": dpo_job_id,
                "orig_result": None,
                "dpo_result": None,
                "user_email": req.user_email,
            }

            job_queue.put((orig_job_id, req.prompt, req.lyrics, req.max_sec, "original", 0.0))
            job_queue.put((dpo_job_id, req.prompt, req.lyrics, req.max_sec, second_model, 3.0))

        return {"queued": min(req.count, 10)}

    @fapi.get("/api/next")
    def next_pair():
        import random
        for pair_id, info in list(rate_pending_pairs.items()):
            orig = jobs.get(info["orig_job"], {})
            dpo = jobs.get(info["dpo_job"], {})
            if orig.get("status") == "done" and dpo.get("status") == "done":
                if random.random() < 0.5:
                    a_result, b_result = orig["result"], dpo["result"]
                    a_model, b_model = "original", "dpo"
                else:
                    a_result, b_result = dpo["result"], orig["result"]
                    a_model, b_model = "dpo", "original"
                rate_ready_pairs.append({
                    "pair_id": pair_id,
                    "prompt": info["prompt"],
                    "a_audio": a_result["audio_file"],
                    "b_audio": b_result["audio_file"],
                    "tags_a": a_result["tags"],
                    "tags_b": b_result["tags"],
                    "a_model": a_model,
                    "b_model": b_model,
                })
                del rate_pending_pairs[pair_id]

        if rate_ready_pairs:
            return {"status": "ready", "pair": rate_ready_pairs[0]}
        elif rate_pending_pairs or gen_stats["generating"]:
            return {"status": "generating"}
        return {"status": "empty"}

    @fapi.post("/api/rate")
    def rate_pair(req: RateVote):
        if req.choice not in ("a", "b"):
            raise HTTPException(400, "Choice must be 'a' or 'b'")
        pair = None
        for i, p in enumerate(rate_ready_pairs):
            if p["pair_id"] == req.pair_id:
                pair = rate_ready_pairs.pop(i)
                break
        if pair is None:
            raise HTTPException(404, "Pair not found")
        winner_model = pair["a_model"] if req.choice == "a" else pair["b_model"]
        rate_stats["rated"] += 1
        logger.info("Rating: pair=%s choice=%s winner=%s user=%s", req.pair_id, req.choice, winner_model, req.user_email)
        return {"status": "recorded", "count": rate_stats["rated"]}

    return fapi
