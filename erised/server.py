"""
FastAPI server exposing the Erised music generation pipeline.

Endpoints:
  POST /generate          — single song from prompt + lyrics
  POST /generate-pair     — two songs for A/B preference rating
  POST /rate              — submit a preference (winner/loser)
  GET  /outputs/{filename} — serve generated audio files
  GET  /health            — health check
"""

import os
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import ErisedConfig
from .pipeline import ErisedPipeline
from .dpo.data import PreferenceStore
from .jobs import job_manager, JobStatus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

_pipeline: Optional[ErisedPipeline] = None
_pref_store: Optional[PreferenceStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _pref_store
    config = ErisedConfig.from_env()
    logger.info("Initializing Erised pipeline...")
    _pipeline = ErisedPipeline(config)
    _pref_store = PreferenceStore(config.dpo_db_path)

    # DPO Guided: if a DPO checkpoint path is set, initialize the guider
    # so the /generate-guided endpoint is available
    dpo_path = os.environ.get("ERISED_DPO_PATH")
    if dpo_path and os.path.isdir(dpo_path):
        n_layers = int(os.environ.get("ERISED_DPO_LAYERS", "2"))
        logger.info("Initializing DPO Guided from %s (%d layers)...", dpo_path, n_layers)
        _pipeline.init_guided(dpo_checkpoint_path=dpo_path, n_dpo_layers=n_layers)
        logger.info("DPO Guided ready.")

    logger.info("Erised ready on %s:%d", config.host, config.port)
    yield
    logger.info("Shutting down Erised.")


app = FastAPI(title="Erised Music Generation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ──────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    lyrics: str
    max_audio_length_ms: Optional[int] = None
    temperature: Optional[float] = None
    topk: Optional[int] = None
    cfg_scale: Optional[float] = None
    dpo_scale: Optional[float] = None  # DPO Guided: set to enable logit-level guidance


class GenerateResponse(BaseModel):
    generation_id: str
    audio_url: str
    tags_used: str
    num_frames: int


class PairResponse(BaseModel):
    pair_id: str
    a: GenerateResponse
    b: GenerateResponse


class RateRequest(BaseModel):
    pair_id: str
    winner_id: str
    loser_id: str
    prompt: str
    lyrics: str
    rater_id: Optional[str] = None


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/rate-ui")
async def rate_ui():
    """Serve the RLHF rating interface."""
    html_path = Path(__file__).parent / "static" / "rate.html"
    if not html_path.exists():
        raise HTTPException(404, "Rating UI not found")
    return HTMLResponse(html_path.read_text())


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    result = _pipeline.generate(
        prompt=req.prompt,
        lyrics=req.lyrics,
        max_audio_length_ms=req.max_audio_length_ms,
        temperature=req.temperature,
        topk=req.topk,
        cfg_scale=req.cfg_scale,
    )

    filename = os.path.basename(result.audio_path)
    return GenerateResponse(
        generation_id=result.generation_id,
        audio_url=f"/outputs/{filename}",
        tags_used=result.tags_used,
        num_frames=result.num_frames,
    )


# ── DPO Guided endpoint ────────────────────────────────────────────
# Uses logit-level DPO guidance instead of merged weights.
# The original model drives generation (stays internally consistent),
# while the DPO model's learned preferences steer token selection:
#   logits_final = logits_orig + scale * (logits_dpo - logits_orig)
@app.post("/generate-guided", response_model=GenerateResponse)
async def generate_guided(req: GenerateRequest):
    """Generate a song using DPO Guided inference.

    Requires the pipeline to have been initialized with init_guided().
    The dpo_scale parameter controls guidance strength (default 1.0).
    """
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    if not hasattr(_pipeline, "guider") or _pipeline.guider is None:
        raise HTTPException(503, "DPO Guided not initialized. Set ERISED_DPO_PATH env var.")

    result = _pipeline.generate_guided(
        prompt=req.prompt,
        lyrics=req.lyrics,
        max_audio_length_ms=req.max_audio_length_ms,
        temperature=req.temperature,
        topk=req.topk,
        cfg_scale=req.cfg_scale,
        dpo_scale=req.dpo_scale or 1.0,
    )

    filename = os.path.basename(result.audio_path)
    return GenerateResponse(
        generation_id=result.generation_id,
        audio_url=f"/outputs/{filename}",
        tags_used=result.tags_used,
        num_frames=result.num_frames,
    )


@app.post("/generate-pair", response_model=PairResponse)
async def generate_pair(req: GenerateRequest):
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    a, b = _pipeline.generate_pair(
        prompt=req.prompt,
        lyrics=req.lyrics,
        max_audio_length_ms=req.max_audio_length_ms,
        temperature=req.temperature,
        topk=req.topk,
        cfg_scale=req.cfg_scale,
    )

    pair_id = f"{a.generation_id}_{b.generation_id}"

    return PairResponse(
        pair_id=pair_id,
        a=GenerateResponse(
            generation_id=a.generation_id,
            audio_url=f"/outputs/{os.path.basename(a.audio_path)}",
            tags_used=a.tags_used,
            num_frames=a.num_frames,
        ),
        b=GenerateResponse(
            generation_id=b.generation_id,
            audio_url=f"/outputs/{os.path.basename(b.audio_path)}",
            tags_used=b.tags_used,
            num_frames=b.num_frames,
        ),
    )


class StartJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[PairResponse] = None
    error: Optional[str] = None


_generation_lock = threading.Lock()


@app.post("/start-pair", response_model=StartJobResponse)
async def start_pair_generation(req: GenerateRequest):
    """Start async pair generation. Returns immediately with job_id.
    Only one generation runs at a time (GPU lock) to avoid meta-tensor errors."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    
    job_id = job_manager.create_job(req.prompt, req.lyrics)
    
    def run_generation():
        with _generation_lock:
            try:
                base_temp = req.temperature or _pipeline.config.temperature
                base_cfg = req.cfg_scale or _pipeline.config.cfg_scale
                
                # A: conservative (lower temp, higher cfg for more prompt adherence)
                job_manager.update_job(job_id, status=JobStatus.GENERATING_A, progress=10)
                a = _pipeline.generate(
                    prompt=req.prompt,
                    lyrics=req.lyrics,
                    max_audio_length_ms=req.max_audio_length_ms,
                    temperature=base_temp * 0.8,
                    topk=req.topk,
                    cfg_scale=base_cfg * 1.2,
                )
                job_manager.set_result_a(job_id, {
                    "generation_id": a.generation_id,
                    "audio_url": f"/outputs/{os.path.basename(a.audio_path)}",
                    "tags_used": a.tags_used,
                    "num_frames": a.num_frames,
                })
                
                # B: creative (higher temp, lower cfg for more variation)
                b = _pipeline.generate(
                    prompt=req.prompt,
                    lyrics=req.lyrics,
                    max_audio_length_ms=req.max_audio_length_ms,
                    temperature=base_temp * 1.2,
                    topk=req.topk,
                    cfg_scale=base_cfg * 0.8,
                )
                job_manager.set_result_b(job_id, {
                    "generation_id": b.generation_id,
                    "audio_url": f"/outputs/{os.path.basename(b.audio_path)}",
                    "tags_used": b.tags_used,
                    "num_frames": b.num_frames,
                })
            except Exception as e:
                job_manager.set_failed(job_id, str(e))
    
    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()
    
    return StartJobResponse(job_id=job_id, status="pending")


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll this endpoint to check generation progress."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    
    result = None
    if job.status == JobStatus.COMPLETE and job.result_a and job.result_b:
        pair_id = f"{job.result_a['generation_id']}_{job.result_b['generation_id']}"
        result = PairResponse(
            pair_id=pair_id,
            a=GenerateResponse(**job.result_a),
            b=GenerateResponse(**job.result_b),
        )
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        result=result,
        error=job.error,
    )


@app.post("/rate")
async def rate(req: RateRequest):
    if _pref_store is None:
        raise HTTPException(503, "Preference store not initialized")

    output_dir = _pipeline.config.output_dir

    winner_tokens = os.path.join(output_dir, f"{req.winner_id}_tokens.pt")
    loser_tokens = os.path.join(output_dir, f"{req.loser_id}_tokens.pt")

    if not os.path.exists(winner_tokens) or not os.path.exists(loser_tokens):
        raise HTTPException(404, "Token files not found for the given generation IDs")

    _pref_store.add_preference(
        pair_id=req.pair_id,
        prompt=req.prompt,
        lyrics=req.lyrics,
        winner_id=req.winner_id,
        loser_id=req.loser_id,
        winner_tokens_path=winner_tokens,
        loser_tokens_path=loser_tokens,
        rater_id=req.rater_id or "anonymous",
    )

    count = _pref_store.count()
    return {"status": "recorded", "total_preferences": count}


@app.get("/outputs/{filename}")
async def serve_output(filename: str):
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    path = os.path.join(_pipeline.config.output_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    media = "audio/mpeg" if filename.endswith(".mp3") else "application/octet-stream"
    return FileResponse(path, media_type=media)


@app.get("/preferences/count")
async def pref_count():
    if _pref_store is None:
        raise HTTPException(503, "Preference store not initialized")
    return {"count": _pref_store.count()}


@app.get("/stats")
async def stats():
    """Get overall stats for the UI."""
    count = _pref_store.count() if _pref_store else 0
    return {"count": count}


@app.delete("/preferences/{pair_id}")
async def delete_preference(pair_id: str):
    """Remove a preference (e.g. phantom rating)."""
    if _pref_store is None:
        raise HTTPException(503, "Preference store not initialized")
    if not _pref_store.delete_preference(pair_id):
        raise HTTPException(404, f"Preference {pair_id} not found")
    return {"status": "deleted", "pair_id": pair_id, "total_preferences": _pref_store.count()}


@app.get("/preferences/debug")
async def preferences_debug(limit: int = 20):
    """List preferences with prompt/lyrics snippets to identify phantom ratings."""
    if _pref_store is None:
        raise HTTPException(503, "Preference store not initialized")
    rows = _pref_store.get_all()
    recent = rows[-limit:] if len(rows) > limit else rows
    from datetime import datetime

    def trunc(s: str, n: int = 60) -> str:
        return (s[:n] + "...") if s and len(s) > n else (s or "")

    return {
        "total": len(rows),
        "recent": [
            {
                "pair_id": r.pair_id,
                "winner_id": r.winner_id,
                "loser_id": r.loser_id,
                "timestamp": datetime.fromtimestamp(r.timestamp).isoformat() if r.timestamp else None,
                "prompt": trunc(r.prompt),
                "lyrics": trunc(r.lyrics),
            }
            for r in recent
        ],
    }


@app.post("/backup")
async def create_backup():
    """Create a backup zip of all preference data."""
    from .backup import export_backup
    # Use the same paths the server is actually using
    db_path = _pref_store.db_path if _pref_store else None
    output_dir = _pipeline.config.output_dir if _pipeline else None
    zip_path = export_backup(db_path=db_path, output_dir=output_dir)
    if zip_path is None:
        raise HTTPException(400, "No data to backup yet")
    return {
        "status": "created",
        "path": str(zip_path),
        "download_url": f"/backups/{zip_path.name}"
    }


@app.get("/backups/{filename}")
async def download_backup(filename: str):
    """Download a backup zip file."""
    backup_path = Path("/workspace/erised_backups") / filename
    if not backup_path.exists() or not filename.endswith(".zip"):
        raise HTTPException(404, "Backup not found")
    return FileResponse(backup_path, media_type="application/zip", filename=filename)


def main():
    """Entry point: `python -m erised.server`"""
    import uvicorn
    config = ErisedConfig.from_env()
    uvicorn.run(
        "erised.server:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
