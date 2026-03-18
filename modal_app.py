"""
Modal serverless deployment for Erised GPU backend.

Deploys the generation server as a Modal web endpoint with:
- A100-80GB: token generation (mula model + DPO)
- A100 codec worker: streaming decode via RPC during generation
  (frames sent at checkpoints, decoded in parallel on separate GPU)
- Auto cold-start: container boots when users visit erised.me (~45s)
- Auto shutdown: container stops after 5 min idle (no charges)
- Permanent URL: stored once in Supabase, never changes

Streaming flow:
  1. A100 generates frames, sends checkpoints every ~80 frames via RPC
  2. Codec worker decodes each checkpoint incrementally (StreamingDecoder)
  3. After 2+ chunks decoded (~20s audio), frontend starts playback
  4. Generation continues → codec decodes more chunks → seamless playback
  5. If codec worker fails, fallback to local A100 decode after generation

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


# ── A100 Codec Worker ────────────────────────────────────────────────
# Separate GPU for streaming codec decode while A100 generates tokens.
# Receives frames via RPC (no volume polling) and decodes incrementally.
# Pre-warmed at serve() startup so cold-start overlaps with model load.

@app.cls(
    image=image,
    gpu="A100",
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
        self.decoders = {}  # job_id → StreamingDecoder (persists across RPC calls)
        self.logger.info(
            "Codec loaded. VRAM: %.2f GB",
            torch.cuda.memory_allocated() / 1024**3,
        )

    @modal.method()
    def warmup(self):
        """Trigger cold-start so container is ready when first job arrives."""
        import torch
        self.logger.info(
            "Warmup ping — VRAM: %.2f GB",
            torch.cuda.memory_allocated() / 1024**3,
        )
        return True

    @modal.method()
    def decode_chunk(self, frames_cpu, job_id: str, audio_filename: str, is_final: bool) -> dict:
        """Decode frames received via RPC. StreamingDecoder state persists per job."""
        import torch
        from erised.streaming import StreamingDecoder

        frames = frames_cpu.to("cuda:0")

        if job_id not in self.decoders:
            save_path = os.path.join("/data/outputs", audio_filename)
            os.makedirs("/data/outputs", exist_ok=True)
            self.decoders[job_id] = StreamingDecoder(self.codec, save_path, duration=12)

        decoder = self.decoders[job_id]
        new_chunks = decoder.decode_available(frames)

        if new_chunks > 0 or is_final:
            try:
                erised_vol.commit()
            except Exception:
                pass

        total = decoder.chunks_decoded
        self.logger.info(
            "Job %s: +%d chunks (total: %d, frames: %d, final: %s)",
            job_id, new_chunks, total, frames.shape[-1], is_final,
        )

        if is_final:
            self.decoders.pop(job_id, None)

        return {"chunks_decoded": total, "new_chunks": new_chunks}


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

    # Reload volume first so FUSE mount is synced, then create dirs
    erised_vol.reload()
    for d in (output_dir, jobs_dir, frames_dir):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            logger.warning("makedirs %s: %s (continuing)", d, e)

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

    # ── Pre-warm codec worker (cold-start overlaps with model load) ──
    codec_worker = CodecWorker()

    def _warmup_codec():
        try:
            codec_worker.warmup.remote()
            logger.info("Codec worker pre-warmed")
        except Exception as e:
            logger.warning("Codec worker warmup failed: %s", e)

    warmup_thread = threading.Thread(target=_warmup_codec, daemon=True)
    warmup_thread.start()

    # ── Job persistence (survives deploys & container routing) ─────
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    def _job_path(job_id: str) -> str:
        return os.path.join(jobs_dir, f"{job_id}.json")

    def _save_job(job_id: str):
        """Write job state to volume (best-effort, jobs also live in memory)."""
        data = jobs.get(job_id)
        if data is None:
            return
        try:
            os.makedirs(jobs_dir, exist_ok=True)
        except OSError:
            pass
        try:
            tmp = _job_path(job_id) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, _job_path(job_id))
        except OSError as e:
            logger.warning("Failed to persist job %s to disk: %s", job_id, e)

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
    try:
        for fname in os.listdir(jobs_dir):
            if fname.endswith(".json") and not fname.endswith(".tmp"):
                jid = fname[:-5]
                _load_job(jid)
        if jobs:
            logger.info("Loaded %d existing jobs from volume", len(jobs))
    except OSError as e:
        logger.warning("Could not load existing jobs: %s", e)

    # ── Job queue ──────────────────────────────────────────────────
    gen_lock = threading.Lock()
    gen_stats = {"pending": 0, "completed": 0, "generating": False}
    job_queue: queue.Queue = queue.Queue()

    DPO_USER_EMAIL = os.environ.get("ERISED_DPO_USER_EMAIL", "zsean@berkeley.edu")

    def generation_worker():
        while True:
            job_id, prompt, lyrics, max_sec, model_name, dpo_scale = job_queue.get()
            with jobs_lock:
                jobs[job_id]["status"] = "running"
                _save_job(job_id)
            gen_stats["generating"] = True

            last_save_time = [0.0]
            audio_filename = f"{job_id}.wav"
            audio_path = os.path.join(output_dir, audio_filename)

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
                    if now - last_save_time[0] > 5:
                        _save_job(job_id)
                        last_save_time[0] = now

            # ── Codec worker streaming: send frames via RPC during gen ──
            checkpoint_q: queue.Queue = queue.Queue()
            codec_failed = [False]
            chunks_decoded = [0]

            def codec_sender():
                """Background thread: sends frame checkpoints to codec worker."""
                while True:
                    item = checkpoint_q.get()
                    if item is None:  # poison pill
                        break
                    frames_cpu, is_final = item
                    try:
                        res = codec_worker.decode_chunk.remote(
                            frames_cpu, job_id, audio_filename, is_final,
                        )
                        chunks_decoded[0] = res["chunks_decoded"]
                        # Expose audio to frontend after 2+ chunks (~20s buffer)
                        # or on final decode (short songs)
                        if chunks_decoded[0] >= 2 or is_final:
                            with jobs_lock:
                                jobs[job_id]["partial_audio_file"] = audio_filename
                                jobs[job_id]["partial_version"] = chunks_decoded[0]
                                _save_job(job_id)
                            logger.info("Job %s: codec chunks=%d, exposed to frontend",
                                        job_id, chunks_decoded[0])
                    except Exception as e:
                        logger.warning("Codec worker failed for job %s: %s", job_id, e)
                        codec_failed[0] = True
                        break

            sender = threading.Thread(target=codec_sender, daemon=True)
            sender.start()

            def on_frames_checkpoint(frames_tensor, is_final):
                if not codec_failed[0]:
                    checkpoint_q.put((frames_tensor.cpu().clone(), is_final))

            try:
                # ── Phase 1: Generate frames, streaming to codec worker ──
                with gen_lock:
                    t0 = time.time()
                    if model_name == "dpo":
                        gen_result = pipeline.generate_guided(
                            prompt=prompt, lyrics=lyrics,
                            max_audio_length_ms=int(max_sec * 1000),
                            dpo_scale=dpo_scale,
                            on_progress=on_progress,
                            on_frames_checkpoint=on_frames_checkpoint,
                        )
                    else:
                        gen_result = pipeline.generate(
                            prompt=prompt, lyrics=lyrics,
                            max_audio_length_ms=int(max_sec * 1000),
                            on_progress=on_progress,
                            on_frames_checkpoint=on_frames_checkpoint,
                        )
                    gen_elapsed = time.time() - t0
                    logger.info("[%s] Generated %d frames in %.1fs",
                                model_name, gen_result.num_frames, gen_elapsed)

                # Wait for codec sender to finish all queued items
                checkpoint_q.put(None)  # poison pill
                sender.join(timeout=180)

                elapsed = time.time() - t0

                if not codec_failed[0] and chunks_decoded[0] > 0:
                    # ── Success: codec worker decoded everything ──
                    logger.info("Job %s: codec streaming complete (%d chunks, %.1fs)",
                                job_id, chunks_decoded[0], elapsed)
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
                else:
                    # ── Fallback: local A100 decode (codec worker failed) ──
                    logger.warning("Job %s: codec worker unavailable, local decode fallback",
                                   job_id)
                    from erised.guided_generate import _reset_model_caches
                    _reset_model_caches(pipeline.pipe.mula)
                    if hasattr(pipeline, 'guider') and pipeline.guider:
                        _reset_model_caches(pipeline.guider.dpo_model)
                    torch.cuda.empty_cache()

                    with jobs_lock:
                        jobs[job_id]["status"] = "decoding"
                        _save_job(job_id)

                    frames = torch.load(gen_result.tokens_path, map_location="cuda:0",
                                        weights_only=True)
                    from erised.streaming import StreamingDecoder
                    decoder = StreamingDecoder(pipeline.pipe.codec, audio_path)

                    def on_decode_chunk(chunk_idx, total_chunks):
                        try:
                            erised_vol.commit()
                        except Exception:
                            pass
                        with jobs_lock:
                            jobs[job_id]["partial_audio_file"] = audio_filename
                            jobs[job_id]["partial_version"] = chunk_idx
                            _save_job(job_id)
                        logger.info("Job %s: local decode chunk %d/%d",
                                    job_id, chunk_idx, total_chunks)

                    decoder.decode_available(frames, on_chunk_ready=on_decode_chunk)
                    try:
                        erised_vol.commit()
                    except Exception:
                        pass

                    elapsed = time.time() - t0
                    logger.info("Job %s: local decode complete in %.1fs", job_id, elapsed)

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
        # Reload volume to see codec worker's latest writes.
        # May fail if codec worker has open files — serve from cache.
        try:
            erised_vol.reload()
        except Exception:
            pass
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
