"""
Modal serverless deployment for Erised GPU backend.

Deploys the generation server as a Modal web endpoint with:
- A100-80GB: token generation (mula model + DPO)
- L4: codec decode (streaming audio while generating, pre-spawned at job start)
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


# ── L4 Codec Worker ──────────────────────────────────────────────────
# Separate GPU for streaming codec decode while A100 generates tokens.
# Pre-spawned at job start so cold-start overlaps with token generation.
# Polls for new frames on the shared volume and decodes incrementally.

@app.cls(
    image=image,
    gpu="L4",
    volumes={"/data": erised_vol},
    timeout=600,
    scaledown_window=300,
)
class CodecWorker:
    @modal.enter()
    def load_codec(self):
        import sys
        import logging
        import torch

        sys.path.insert(0, "/root/heartlib_pkg")
        sys.path.insert(0, "/root")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        self.logger = logging.getLogger("erised.codec_worker")

        model_path = os.environ.get("ERISED_MODEL_PATH", "/data/ckpt")
        self.logger.info("Loading codec from %s ...", model_path)

        from heartlib import HeartMuLaGenPipeline

        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device={"mula": torch.device("cpu"), "codec": torch.device("cuda:0")},
            dtype={"mula": torch.bfloat16, "codec": torch.float32},
            version="3B",
            lazy_load=False,
        )
        self.codec = pipe.codec
        self.logger.info(
            "Codec loaded. VRAM: %.2f GB",
            torch.cuda.memory_allocated() / 1024**3,
        )

    @modal.method()
    def decode_job(self, job_id: str, audio_filename: str):
        """Poll for frames on volume and decode incrementally."""
        import json
        import time
        import torch
        from erised.streaming import StreamingDecoder

        save_path = os.path.join("/data/outputs", audio_filename)
        signal_path = os.path.join("/data/frames", f"{job_id}.json")
        frames_path = os.path.join("/data/frames", f"{job_id}.pt")
        decode_status_path = os.path.join("/data/frames", f"{job_id}_decoded.json")

        os.makedirs("/data/outputs", exist_ok=True)

        decoder = StreamingDecoder(self.codec, save_path)
        last_num_frames = 0

        self.logger.info("Starting decode loop for job %s → %s", job_id, audio_filename)

        while True:
            # See latest writes from A100
            erised_vol.reload()

            if not os.path.isfile(signal_path):
                time.sleep(1)
                continue

            with open(signal_path) as f:
                signal = json.load(f)

            current_num_frames = signal.get("num_frames", 0)
            is_final = signal.get("is_final", False)

            # Only decode if we have new frames
            if current_num_frames > last_num_frames or is_final:
                frames = torch.load(frames_path, map_location="cuda:0", weights_only=True)
                new_chunks = decoder.decode_available(frames)
                last_num_frames = current_num_frames

                if new_chunks > 0:
                    self.logger.info(
                        "Job %s: decoded %d new chunks (total: %d, frames: %d)",
                        job_id, new_chunks, decoder.chunks_decoded, current_num_frames,
                    )
                    # Write decode status for A100 to read
                    with open(decode_status_path, "w") as f:
                        json.dump({
                            "chunks": decoder.chunks_decoded,
                            "audio_file": audio_filename,
                            "done": is_final,
                        }, f)
                    erised_vol.commit()

            if is_final:
                # One final decode pass to catch any remaining
                frames = torch.load(frames_path, map_location="cuda:0", weights_only=True)
                extra = decoder.decode_available(frames)
                if extra > 0:
                    self.logger.info("Job %s: final pass decoded %d more chunks", job_id, extra)
                # Write final status
                with open(decode_status_path, "w") as f:
                    json.dump({
                        "chunks": decoder.chunks_decoded,
                        "audio_file": audio_filename,
                        "done": True,
                    }, f)
                erised_vol.commit()
                break

            time.sleep(2)

        self.logger.info(
            "Decode complete for job %s: %d chunks → %s",
            job_id, decoder.chunks_decoded, save_path,
        )


# ── A100 Generation Server ───────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": erised_vol},
    timeout=600,
    scaledown_window=300,  # 5 min idle → shut down
    secrets=[modal.Secret.from_name("erised-secrets")],
)
@modal.concurrent(max_inputs=1000)  # keep all requests on ONE container
@modal.asgi_app()
def serve():
    """Create and return the FastAPI app."""
    import sys
    import json
    import logging
    import queue
    import threading
    import time
    import uuid

    import torch

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
    jobs_dir = "/data/jobs"
    frames_dir = "/data/frames"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

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

    # ── Job persistence (survives deploys & container routing) ─────
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    def _job_path(job_id: str) -> str:
        return os.path.join(jobs_dir, f"{job_id}.json")

    def _save_job(job_id: str):
        """Write job state to volume (atomic via temp file)."""
        data = jobs.get(job_id)
        if data is None:
            return
        tmp = _job_path(job_id) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, _job_path(job_id))

    def _load_job(job_id: str) -> dict | None:
        """Load job from volume if not in memory."""
        path = _job_path(job_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            jobs[job_id] = data
            return data
        except (json.JSONDecodeError, OSError):
            return None

    # Load any existing jobs from a previous container
    for fname in os.listdir(jobs_dir):
        if fname.endswith(".json") and not fname.endswith(".tmp"):
            jid = fname[:-5]
            _load_job(jid)
    if jobs:
        logger.info("Loaded %d existing jobs from volume", len(jobs))

    # ── Job queue ──────────────────────────────────────────────────
    gen_lock = threading.Lock()
    gen_stats = {"pending": 0, "completed": 0, "generating": False}
    job_queue: queue.Queue = queue.Queue()

    DPO_USER_EMAIL = os.environ.get("ERISED_DPO_USER_EMAIL", "zsean@berkeley.edu")

    # Handle to the L4 codec worker
    codec_worker = CodecWorker()

    def generation_worker():
        while True:
            job_id, prompt, lyrics, max_sec, model_name, dpo_scale = job_queue.get()
            with jobs_lock:
                jobs[job_id]["status"] = "running"
                _save_job(job_id)
            gen_stats["generating"] = True

            last_save_time = 0.0
            t4_audio_filename = [f"{job_id}.wav"]
            decode_status_path = os.path.join(frames_dir, f"{job_id}_decoded.json")
            # Track L4 decode progress for streaming audio to frontend
            t4_decode_info = [None]  # Updated by background checker thread

            # Pre-spawn L4 codec worker NOW so it cold-starts during token
            # generation (~50-90s) instead of blocking after first checkpoint.
            logger.info("Pre-spawning L4 codec worker for job %s", job_id)
            codec_worker.decode_job.spawn(job_id, t4_audio_filename[0])

            def on_progress(current_frame, total_frames, partial_audio_file=None, partial_version=None):
                nonlocal last_save_time
                with jobs_lock:
                    jobs[job_id]["progress"] = {
                        "current_frame": current_frame,
                        "total_frames": total_frames,
                    }
                    if partial_audio_file:
                        jobs[job_id]["partial_audio_file"] = partial_audio_file
                    if partial_version is not None:
                        jobs[job_id]["partial_version"] = partial_version

                    # Check T4 decode progress (non-blocking, uses cached info)
                    info = t4_decode_info[0]
                    if info and info.get("chunks", 0) > 0:
                        jobs[job_id]["partial_audio_file"] = info["audio_file"]
                        jobs[job_id]["partial_version"] = info["chunks"]

                    now = time.time()
                    if partial_audio_file or (now - last_save_time > 5):
                        _save_job(job_id)
                        last_save_time = now

            def _t4_decode_checker():
                """Background thread: polls volume for L4 decode status."""
                while True:
                    try:
                        erised_vol.reload()
                        if os.path.isfile(decode_status_path):
                            with open(decode_status_path) as f:
                                t4_decode_info[0] = json.load(f)
                            if t4_decode_info[0].get("done"):
                                break
                    except Exception:
                        pass
                    time.sleep(3)

            checker_thread = threading.Thread(target=_t4_decode_checker, daemon=True)
            checker_thread.start()

            sync_lock = threading.Lock()  # Serialize background syncs

            def on_frames_checkpoint(frames_tensor, is_final=False):
                """Save frames to volume and signal L4 codec worker."""
                cpu_frames = frames_tensor.cpu()

                def _sync():
                    with sync_lock:  # Ensure sequential commits
                        frames_path = os.path.join(frames_dir, f"{job_id}.pt")
                        signal_path = os.path.join(frames_dir, f"{job_id}.json")
                        torch.save(cpu_frames, frames_path)
                        with open(signal_path, "w") as f:
                            json.dump({
                                "num_frames": cpu_frames.shape[-1],
                                "is_final": is_final,
                            }, f)
                        erised_vol.commit()
                        logger.info(
                            "Checkpoint for job %s: %d frames, final=%s",
                            job_id, cpu_frames.shape[-1], is_final,
                        )

                if is_final:
                    # Final checkpoint: BLOCKING — must commit before we
                    # wait for T4, otherwise T4 never sees final frames.
                    _sync()
                else:
                    # Intermediate checkpoints: non-blocking
                    threading.Thread(target=_sync, daemon=True).start()

            try:
                with gen_lock:
                    t0 = time.time()
                    if model_name == "dpo":
                        result = pipeline.generate_guided(
                            prompt=prompt, lyrics=lyrics,
                            max_audio_length_ms=int(max_sec * 1000),
                            dpo_scale=dpo_scale,
                            on_progress=on_progress,
                            on_frames_checkpoint=on_frames_checkpoint,
                        )
                    else:
                        result = pipeline.generate(
                            prompt=prompt, lyrics=lyrics,
                            max_audio_length_ms=int(max_sec * 1000),
                            on_progress=on_progress,
                            on_frames_checkpoint=on_frames_checkpoint,
                        )
                    elapsed = time.time() - t0
                    logger.info("[%s] Generated %s in %.1fs (%d frames)",
                                model_name, result.audio_path, elapsed, result.num_frames)

                # Wait for L4 to finish decoding the final audio
                t4_done = False
                logger.info("Waiting for L4 codec decode to finish for job %s ...", job_id)
                for attempt in range(180):  # Max 6 min
                    erised_vol.reload()
                    if os.path.isfile(decode_status_path):
                        with open(decode_status_path) as f:
                            decode_info = json.load(f)
                        if decode_info.get("done"):
                            logger.info("L4 decode complete for job %s (%d chunks)",
                                        job_id, decode_info.get("chunks", 0))
                            t4_done = True
                            break
                    time.sleep(2)
                else:
                    logger.warning("L4 decode timed out for job %s", job_id)

                if t4_done:
                    final_audio = t4_audio_filename[0]
                else:
                    # Fallback: decode locally on A100 (safe, generation done)
                    logger.info("Falling back to local decode for job %s", job_id)
                    from erised.streaming import streaming_detokenize
                    fallback_frames = torch.load(result.tokens_path, map_location="cuda:0", weights_only=True)
                    fallback_path = os.path.join(output_dir, f"{job_id}_fallback.wav")
                    streaming_detokenize(
                        pipeline.pipe.codec, fallback_frames, fallback_path,
                    )
                    final_audio = os.path.basename(fallback_path)

                with jobs_lock:
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["result"] = {
                        "audio_file": final_audio,
                        "tags": result.tags_used,
                        "num_frames": result.num_frames,
                        "elapsed": round(elapsed, 1),
                        "model": model_name,
                    }
                    # Clear partial fields now that we have final result
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
                # Clean up frames files
                for suffix in [".pt", ".json", "_decoded.json"]:
                    path = os.path.join(frames_dir, f"{job_id}{suffix}")
                    try:
                        os.remove(path)
                    except OSError:
                        pass

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
        import uuid as _uuid
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")
        if req.model not in ("original", "dpo"):
            raise HTTPException(400, "Model must be 'original' or 'dpo'")

        effective_model = req.model
        if req.model == "dpo" and req.user_email != DPO_USER_EMAIL:
            effective_model = "original"

        job_id = str(_uuid.uuid4())[:8]
        with jobs_lock:
            jobs[job_id] = {"status": "pending"}
            _save_job(job_id)
        gen_stats["pending"] += 1
        job_queue.put((job_id, req.prompt, req.lyrics, req.max_sec, effective_model, req.dpo_scale))
        logger.info("Job %s: model=%s, user=%s", job_id, effective_model, req.user_email)
        return {"job_id": job_id}

    @fapi.get("/api/job/{job_id}")
    def get_job(job_id: str):
        with jobs_lock:
            if job_id in jobs:
                return jobs[job_id]
        # Try loading from volume (handles multi-container / post-deploy)
        data = _load_job(job_id)
        if data is not None:
            return data
        raise HTTPException(404, "Unknown job_id")

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
            # L4 may have written the file — reload volume and retry
            erised_vol.reload()
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
        import uuid as _uuid
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")

        use_dpo = req.user_email == DPO_USER_EMAIL
        second_model = "dpo" if use_dpo else "original"

        for _ in range(min(req.count, 10)):
            pair_id = str(_uuid.uuid4())[:8]
            orig_job_id = str(_uuid.uuid4())[:8]
            dpo_job_id = str(_uuid.uuid4())[:8]

            with jobs_lock:
                jobs[orig_job_id] = {"status": "pending"}
                jobs[dpo_job_id] = {"status": "pending"}
                _save_job(orig_job_id)
                _save_job(dpo_job_id)
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
