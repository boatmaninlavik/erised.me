"""
erised.me song generator: fine-tuned Shao (stage-1 SFT on Suno songs, run sft_v1) on Modal.

  serve        (CPU web app)  POST /api/submit, GET /api/job/{id}, GET /audio/{file}, GET /health
                              — the same contract app/generate/page.tsx and /api/proxy-audio use.
  ShaoWorker   (H100)         one song at a time; each song is a spawned call, so it finishes even if
                              the browser tab closes. Job status + the mp3 live on the volume.
  setup        (CPU, once)    copies the weights onto the volume.

Generation is exactly the recipe of the blind-tested songs (shao_dpo/eval_sft.py::generate):
dpo_common.build_prompt_text (the prompt format we fine-tuned on — empty language slot), backbone
temperature 1.0 / top-k 50 up to (minutes + 0.8) minutes of tokens or the end marker, super-res
top-k 10 on the prompt clipped to 2048 tokens, codec decode in 1920-frame chunks with 480 overlap.

Spend guard (HARD RULE: never past $30 in a workspace): every container start and every song adds
its estimated H100 cost to a Modal Dict; once the month's estimate reaches SPEND_CAP_USD,
/api/submit refuses new songs until the next month.

Deploy (workspace erised9):
    MODAL_PROFILE=erised9 modal run modal_shao.py::setup      (once: weights -> volume)
    MODAL_PROFILE=erised9 modal deploy modal_shao.py
"""
import datetime
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
RUN = "sft_v1"
MODEL_NAME = f"shao-{RUN}"
B2_BUCKET, B2_ENDPOINT = "erised-sft", "https://s3.us-west-004.backblazeb2.com"
MPS_REPO, SHAO_REPO = "Vinpolar/Khala-MusicGeneration-v1.0-MPS", "liujiafeng/Shao-MusicGeneration-v1.0"
SR, FRAME, EOS = 44100, 2048, 128001
MIN_MINUTES, MAX_MINUTES = 1, 4

H100_USD_PER_S = 3.95 / 3600
IDLE_S_PER_SONG = 60          # = scaledown_window: the GPU stays up this long after the last song
SPEND_CAP_USD = 25.0          # leaves room under the $30 workspace credit for the web app + rounding

app = modal.App("erised-shao")
vol = modal.Volume.from_name("erised-shao", create_if_missing=True)
spend = modal.Dict.from_name("erised-shao-spend", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", "numpy<2", "omegaconf", "einops", "lightning", "boto3", "huggingface_hub",
                 "safetensors", "transformers==4.44.2", "soundfile")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(HERE / "khala_runtime", remote_path="/root/khala")
    .add_local_file(HERE / "shao_dpo" / "dpo_common.py", "/root/dpo_common.py")
)
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi", "uvicorn[standard]", "pydantic>=2.0")

W = "/data/weights"
JOBS, OUT = "/data/jobs", "/data/outputs"


def _month():
    return datetime.datetime.utcnow().strftime("%Y-%m")


def _charge(usd: float):
    """Add an estimated cost to this month's running total (best-effort, never fails a song)."""
    try:
        key = _month()
        spend[key] = round(spend.get(key, 0.0) + usd, 4)
    except Exception as e:
        print("spend tracking failed:", e, flush=True)


def _write_job(job_id: str, data: dict):
    os.makedirs(JOBS, exist_ok=True)
    tmp = f"{JOBS}/{job_id}.json.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, f"{JOBS}/{job_id}.json")
    try:
        vol.commit()
    except Exception:
        pass


# ------------------------------------------------------------------ one-time weights
@app.function(image=gpu_image, cpu=4, memory=32768, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("b2-key")], timeout=3600)
def setup():
    import shutil
    import boto3
    import torch
    from huggingface_hub import hf_hub_download
    Path(W).mkdir(parents=True, exist_ok=True)
    backbone = f"{W}/backbone_{RUN}.safetensors"
    if not Path(backbone).exists():
        boto3.client("s3", endpoint_url=B2_ENDPOINT).download_file(B2_BUCKET, f"sft_runs/{RUN}/final.safetensors", backbone)
    for f in ("backbone_megatron_args.json", "khala_superres.safetensors", "superres_megatron_args.json"):
        if not Path(f"{W}/{f}").exists():
            shutil.copyfile(os.path.realpath(hf_hub_download(MPS_REPO, f, cache_dir="/tmp/hf")), f"{W}/{f}")
    if not Path(f"{W}/codec_generator.pt").exists():
        sd = torch.load(hf_hub_download(SHAO_REPO, "dac_rvq_2490000.ckpt", cache_dir="/tmp/hf"), map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        torch.save({k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}, f"{W}/codec_generator.pt")
    vol.commit()
    return {f.name: round(f.stat().st_size / 1e9, 2) for f in Path(W).iterdir()}


# ------------------------------------------------------------------ the GPU worker
@app.cls(image=gpu_image, gpu="H100", memory=65536, volumes={"/data": vol}, timeout=1800,
         scaledown_window=IDLE_S_PER_SONG, max_containers=1)
class ShaoWorker:
    @modal.enter()
    def load(self):
        import sys
        import torch
        from omegaconf import OmegaConf
        sys.path[:0] = ["/root", "/root/khala", "/root/khala/models/Decoder"]
        from transformers import AutoTokenizer
        from core.khala_runtime import load_vanilla_model
        from dac_rvq import DacRVQ
        t0 = time.time()
        vol.reload()
        self.tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
        self.bb = load_vanilla_model("backbone", f"{W}/backbone_{RUN}.safetensors", f"{W}/backbone_megatron_args.json", "cuda", torch.bfloat16)
        self.sr = load_vanilla_model("superres", f"{W}/khala_superres.safetensors", f"{W}/superres_megatron_args.json", "cuda", torch.bfloat16)
        dac = DacRVQ(OmegaConf.load("/root/khala/models/Decoder/dac_rvq_1024_64_golden.yaml"))
        dac.load_state_dict(torch.load(f"{W}/codec_generator.pt"))
        self.dac = dac.eval().cuda()
        os.makedirs(OUT, exist_ok=True)
        load_s = time.time() - t0
        _charge(load_s * H100_USD_PER_S)
        print(f"fine-tuned Shao ({RUN}) loaded in {load_s:.0f}s", flush=True)

    def _backbone(self, prompt_ids, num_tokens, on_step):
        """core.khala_runtime.sample_backbone (temperature 1.0, top-k 50), line for line, plus a progress hook."""
        import torch
        from core.khala_model import KhalaKVCache
        from core.khala_runtime import _empty_cache, _select_token
        model = self.bb
        V = model.config.vocab_size
        cache = KhalaKVCache(model.config.num_layers)
        with torch.inference_mode():
            h = model.forward_hidden_states(torch.tensor([list(prompt_ids)], dtype=torch.long, device="cuda"), causal=True, kv_cache=cache)
            logits = model.lm_head(h[:, -1])[0, :V].float()
            out = []
            for step_i in range(int(num_tokens)):
                nxt = _select_token(logits, 1.0, 50)
                if nxt == EOS:
                    break
                out.append(nxt)
                h = model.forward_hidden_states(torch.tensor([[nxt]], dtype=torch.long, device="cuda"), causal=True, kv_cache=cache)
                logits = model.lm_head(h[:, -1])[0, :V].float()
                if (step_i & 63) == 63:
                    _empty_cache("cuda")
                if (step_i & 511) == 511:
                    on_step(len(out))
        return out

    @modal.method()
    def generate(self, job_id: str, p: dict):
        import subprocess
        import numpy as np
        import soundfile as sf
        import torch
        import dpo_common as dc
        from core.khala_runtime import generate_superres_projection

        t_start = time.time()
        state = {"status": "running", "stage": "composing", "submitted": p.get("submitted"),
                 "progress": {"current_frame": 0, "total_frames": 1000}}
        _write_job(job_id, state)

        def progress(frac, stage):
            state.update(stage=stage, progress={"current_frame": int(1000 * min(frac, 0.999)), "total_frames": 1000})
            _write_job(job_id, state)

        try:
            minutes = int(p["minutes"])
            text = dc.build_prompt_text(p["style"], p["lyrics"], not p["lyrics"].strip(), minutes)
            ids = self.tok.encode(text, add_special_tokens=False)
            if len(ids) > 4096:                                   # Shao's own limit: keep the final marker
                ids = ids[:4095] + [ids[-1]]
            max_tokens = min(round(dc.TOKENS_PER_MINUTE * (minutes + 0.8)), dc.CONTEXT_LEN - len(ids))

            # 0-80%: composing the coarse layers, 80-95%: adding detail (super-res), 95-100%: rendering audio
            out_ids = self._backbone(ids, max_tokens, lambda n: progress(0.8 * n / max_tokens, "composing"))
            arr = np.array(out_ids, dtype=np.int64)
            arr = arr[: len(arr) // 2 * 2].reshape(-1, 2)
            codes2 = np.clip(np.stack([arr[:, 0] - 128256, arr[:, 1] - 129280], 1), 0, 1023)
            t_bb = time.time() - t_start

            progress(0.8, "adding detail")
            text_ids = np.array(ids[:2047] + [ids[-1]] if len(ids) > 2048 else ids, dtype=np.int64)
            audio_ids = dc.audio_ids_from_codes(codes2).numpy().reshape(-1, 2)
            L = len(text_ids) + len(audio_ids)
            tokens = torch.full((L, 2), -1, dtype=torch.long)
            tokens[:len(text_ids), 0] = torch.from_numpy(text_ids)
            tokens[len(text_ids):] = torch.from_numpy(audio_ids)
            with torch.inference_mode():
                full = generate_superres_projection(
                    self.sr, tokens[None].cuda(), torch.zeros(1, 1, 1, L, dtype=torch.bool).cuda(),
                    (tokens[:, -1] != -1).float()[None].cuda(), torch.arange(L)[None].cuda(),
                    len(text_ids), len(audio_ids), 10)
                codes64 = full[:, 0, :].cpu().numpy().T.astype(np.int64) - (128256 + np.arange(64) * 1024)[None, :]
                codes = torch.from_numpy(np.ascontiguousarray(codes64.T))[:, None].cuda()
                progress(0.95, "rendering audio")
                wave, chunk, overlap = None, 1920, 480
                for start in range(0, codes.shape[2], chunk - overlap):
                    piece = self.dac.decode(codes[..., start:start + chunk]).cpu()
                    if wave is None:
                        wave = piece
                    else:
                        n = min(round(piece.shape[2] * overlap / codes[..., start:start + chunk].shape[2]), wave.shape[2], piece.shape[2])
                        fade = torch.linspace(1.0, 0.0, n).view(1, 1, -1)
                        wave = torch.cat([wave[..., :-n], wave[..., -n:] * fade + piece[..., :n] * (1 - fade), piece[..., n:]], 2)
                    if start + chunk >= codes.shape[2]:
                        break
            wav, mp3 = f"/tmp/{job_id}.wav", f"{OUT}/{job_id}.mp3"
            sf.write(wav, wave[0].numpy().T, SR, subtype="FLOAT")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-b:a", "320k", mp3], check=True)
            os.remove(wav)

            elapsed = round(time.time() - t_start, 1)
            state.update(status="done", stage="done", progress={"current_frame": 1000, "total_frames": 1000}, result={
                "audio_file": f"{job_id}.mp3", "tags": p["style"], "num_frames": int(len(codes2)),
                "seconds_audio": round(len(codes2) * FRAME / SR, 1), "elapsed": elapsed, "model": MODEL_NAME,
                "stopped_at_end_marker": len(out_ids) < max_tokens})
            _write_job(job_id, state)
            print(f"[{job_id}] {minutes} min requested -> {len(codes2) * FRAME / SR:.0f}s song in {elapsed:.0f}s "
                  f"(backbone {t_bb:.0f}s)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            state.update(status="error", error=str(e)[:300])
            _write_job(job_id, state)
        finally:
            _charge((time.time() - t_start + IDLE_S_PER_SONG) * H100_USD_PER_S)


# ------------------------------------------------------------------ the web app
@app.function(image=web_image, volumes={"/data": vol}, scaledown_window=60)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    import uuid
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field

    fapi = FastAPI(title="Erised · Shao")
    fapi.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    worker = ShaoWorker()

    class SubmitRequest(BaseModel):
        prompt: str = Field(max_length=1000)
        lyrics: str = Field(max_length=6000)
        max_sec: int = 120            # the page's length control; rounded to whole minutes (Shao's length hint)
        user_email: str | None = None

    @fapi.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME}

    @fapi.post("/api/submit")
    def submit(req: SubmitRequest):
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")
        if spend.get(_month(), 0.0) >= SPEND_CAP_USD:
            raise HTTPException(429, "Erised has reached its song limit for this month. Please come back next month.")
        minutes = max(MIN_MINUTES, min(MAX_MINUTES, round(req.max_sec / 60)))
        job_id = uuid.uuid4().hex[:12]
        _write_job(job_id, {"status": "pending", "stage": "queued", "submitted": time.time()})
        worker.generate.spawn(job_id, {"style": req.prompt.strip(), "lyrics": req.lyrics, "minutes": minutes,
                                       "submitted": time.time()})
        print(f"job {job_id}: {minutes} min, user={req.user_email}", flush=True)
        return {"job_id": job_id}

    @fapi.get("/api/job/{job_id}")
    def get_job(job_id: str):
        try:
            vol.reload()
        except Exception:
            pass
        path = f"{JOBS}/{os.path.basename(job_id)}.json"
        if not os.path.isfile(path):
            raise HTTPException(404, "Unknown job_id")
        return json.load(open(path))

    @fapi.get("/audio/{filename}")
    def audio(filename: str):
        path = f"{OUT}/{os.path.basename(filename)}"
        if not os.path.isfile(path):
            try:
                vol.reload()
            except Exception:
                pass
        if not os.path.isfile(path):
            raise HTTPException(404, "Audio file not found")
        return FileResponse(path, media_type="audio/mpeg")

    return fapi
