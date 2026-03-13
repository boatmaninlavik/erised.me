#!/usr/bin/env python3
"""
FastAPI-based A/B comparison UI for DPO Guided vs original Erised.

DPO Guided approach: instead of merging DPO weights (which causes noise
accumulation), we apply DPO preferences at the logit level:
    logits_final = logits_orig + scale * (logits_dpo - logits_orig)

The bottom 26 shared backbone layers run once; only the top 2 DPO-trained
layers are branched, giving ~7% overhead instead of 2x slowdown.

Uses a job queue + polling pattern so ngrok doesn't timeout on long generations.
Submit a job -> returns job_id instantly -> poll /api/job/{id} until done.

Usage (on RunPod):
    python3 -u -m erised.scripts.compare_local \
        --dpo-path /workspace/dpo_checkpoints_v8/dpo_best

    # Then in another terminal: ngrok http 7860
"""

import argparse
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
logger = logging.getLogger("erised.compare_local")


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


# ── HTML UI (inline) ─────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erised - Compare Models</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0a0a0a; color: #e5e5e5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 40px 60px;
}
@media (min-width: 1200px) { body { padding: 40px 10%; } }
@media (max-width: 768px) { body { padding: 20px; } }

h1 { text-align: center; color: #a855f7; font-size: 2.5rem; font-weight: 700; margin-bottom: 8px; }
.subtitle { text-align: center; color: #888; margin-bottom: 30px; }

.card {
    background: #151515; padding: 24px; border-radius: 12px; margin-bottom: 24px;
}
label {
    display: block; color: #7c3aed; font-weight: 600; font-size: 1.05rem;
    margin-bottom: 6px; letter-spacing: 0.02em;
}
textarea, input[type=text] {
    width: 100%; background: #3d3845; border: 1px solid #4a4555;
    color: #e5e5e5; padding: 14px; border-radius: 8px; font-size: 15px;
    font-weight: 500; font-family: inherit; line-height: 1.5; resize: vertical;
}
textarea:focus, input:focus { outline: none; border-color: #a855f7; }

input[type=range] { accent-color: #38bdf8; width: 100%; }
.slider-wrap { position: relative; }
.slider-val {
    font-size: 12px; font-weight: 600; color: #38bdf8;
    position: absolute; right: 0; top: -20px;
}

.btn {
    width: 100%; min-height: 56px; padding: 18px; font-size: 18px; font-weight: 600;
    border-radius: 8px; cursor: pointer; border: 1px solid #e2e8f0;
    background: #f8fafc; color: #7c3aed; display: block; text-align: center;
    transition: background 0.15s, border-color 0.15s;
}
.btn:hover { background: #f1f5f9; border-color: #c4b5fd; }
.btn:disabled { background: #334155; color: #94a3b8; border-color: #475569; cursor: not-allowed; }

.audio-card {
    background: #151515; padding: 24px; border-radius: 12px; margin-bottom: 16px;
    border: 2px solid #333; transition: border-color 0.2s;
}
.audio-card:hover { border-color: #555; }
.audio-card h3 { color: #a855f7; margin: 0 0 12px 0; }
.audio-card h3 span { font-size: 12px; color: #888; font-weight: normal; }
.audio-card audio { width: 100%; margin: 8px 0; border-radius: 8px; }
.tags { font-size: 12px; color: #64748b; font-family: monospace; margin-top: 4px; }

.speed-btns { display: flex; gap: 6px; margin: 8px 0; }
.speed-btn {
    padding: 4px 10px; font-size: 12px; border-radius: 4px; cursor: pointer;
    border: 1px solid #444; background: #222; color: #ccc;
    transition: all 0.15s;
}
.speed-btn:hover { background: #333; }
.speed-btn.active { background: #7c3aed; color: #fff; border-color: #7c3aed; }

/* Tabs */
.tabs { display: flex; border-bottom: 1px solid #333; margin-bottom: 24px; }
.tab {
    padding: 12px 24px; cursor: pointer; color: #888; font-weight: 500;
    border: none; background: transparent; font-size: 16px;
    border-bottom: 2px solid transparent; transition: all 0.2s;
}
.tab:hover { color: #ccc; }
.tab.active { color: #a855f7; border-bottom-color: #a855f7; }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Status */
.status-bar {
    display: flex; justify-content: center; gap: 24px; padding: 12px;
    background: #151515; border-radius: 8px; margin-bottom: 24px;
    font-size: 14px; color: #888;
}
.status-bar .val { color: #a855f7; font-weight: 600; }
.status-bar .generating { color: #38bdf8; }

/* Progress spinner */
.spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid #38bdf8; border-top-color: transparent;
    border-radius: 50%; animation: spin 0.8s linear infinite;
    vertical-align: middle; margin-right: 8px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Model selector radio */
.model-select { display: flex; gap: 16px; margin-bottom: 16px; }
.model-opt {
    flex: 1; padding: 14px 20px; border-radius: 8px; cursor: pointer;
    border: 2px solid #333; background: #1a1a1a; text-align: center;
    font-weight: 600; color: #888; transition: all 0.2s;
}
.model-opt:hover { border-color: #555; }
.model-opt.selected { border-color: #a855f7; color: #a855f7; background: #1a1525; }

#results-section { display: none; }
#single-result { display: none; }

.paths-info {
    text-align: center; color: #666; font-size: 12px; margin-top: 24px;
    padding-top: 16px; border-top: 1px solid #222;
}
.paths-info code {
    color: #a855f7; background: #1a1a1a; padding: 2px 6px; border-radius: 4px;
}
</style>
</head>
<body>

<h1>Erised</h1>
<p class="subtitle">Compare <b>original</b> vs <b>DPO Guided</b> generations. You can leave the page while generating.</p>

<!-- Status bar -->
<div class="status-bar">
    <div>Queue: <span class="val" id="st-queue">0</span></div>
    <div id="st-gen-wrap" style="display:none">
        <span class="generating"><span class="spinner"></span>Generating...</span>
    </div>
    <div>Completed: <span class="val" id="st-completed">0</span></div>
</div>

<!-- Tabs -->
<div class="tabs">
    <button class="tab active" data-tab="ab-compare" onclick="switchTab('ab-compare')">A/B Compare</button>
    <button class="tab" data-tab="single" onclick="switchTab('single')">Single Generate</button>
</div>

<!-- A/B Compare Tab -->
<div class="tab-content active" id="tab-ab-compare">
    <div class="card">
        <div style="margin-bottom: 16px;">
            <label for="prompt-ab">Musical Prompt</label>
            <textarea id="prompt-ab" rows="3"
                placeholder="e.g., emotional pop ballad with piano and strings"></textarea>
        </div>
        <div style="margin-bottom: 16px;">
            <label for="lyrics-ab">Lyrics</label>
            <textarea id="lyrics-ab" rows="8"
                placeholder="[Verse 1]&#10;Your lyrics here...&#10;&#10;[Chorus]&#10;Your chorus here..."></textarea>
        </div>
        <div class="slider-wrap" style="margin-bottom: 16px;">
            <label for="max-sec-ab">Max length (seconds)</label>
            <span class="slider-val" id="slider-val-ab">60s</span>
            <input type="range" id="max-sec-ab" min="10" max="240" step="5" value="60">
        </div>
        <button class="btn" id="gen-both-btn" onclick="generateBoth()">Generate Both (Original + DPO)</button>
    </div>

    <div id="results-section">
        <div class="audio-card" id="card-orig">
            <h3>Original Model</h3>
            <audio id="audio-orig" controls preload="auto"></audio>
            <div class="speed-btns" data-for="audio-orig">
                <button class="speed-btn" data-speed="0.5">0.5x</button>
                <button class="speed-btn active" data-speed="1">1x</button>
                <button class="speed-btn" data-speed="1.25">1.25x</button>
                <button class="speed-btn" data-speed="1.5">1.5x</button>
                <button class="speed-btn" data-speed="2">2x</button>
            </div>
            <div class="tags" id="tags-orig"></div>
        </div>

        <div class="audio-card" id="card-dpo">
            <h3>DPO Guided</h3>
            <audio id="audio-dpo" controls preload="auto"></audio>
            <div class="speed-btns" data-for="audio-dpo">
                <button class="speed-btn" data-speed="0.5">0.5x</button>
                <button class="speed-btn active" data-speed="1">1x</button>
                <button class="speed-btn" data-speed="1.25">1.25x</button>
                <button class="speed-btn" data-speed="1.5">1.5x</button>
                <button class="speed-btn" data-speed="2">2x</button>
            </div>
            <div class="tags" id="tags-dpo"></div>
        </div>
    </div>

    <div id="ab-progress" style="display:none; text-align:center; padding: 30px; color: #888;">
        <span class="spinner"></span> <span id="ab-progress-text">Generating original model output...</span>
    </div>
</div>

<!-- Single Generate Tab -->
<div class="tab-content" id="tab-single">
    <div class="card">
        <label>Model</label>
        <div class="model-select">
            <div class="model-opt" data-model="original" onclick="selectModel('original')">Original</div>
            <div class="model-opt selected" data-model="dpo" onclick="selectModel('dpo')">DPO Guided</div>
        </div>
        <div style="margin-bottom: 16px;">
            <label for="prompt-single">Musical Prompt</label>
            <textarea id="prompt-single" rows="3"
                placeholder="Describe the music..."></textarea>
        </div>
        <div style="margin-bottom: 16px;">
            <label for="lyrics-single">Lyrics</label>
            <textarea id="lyrics-single" rows="8"
                placeholder="Lyrics..."></textarea>
        </div>
        <div class="slider-wrap" style="margin-bottom: 16px;">
            <label for="max-sec-single">Max length (seconds)</label>
            <span class="slider-val" id="slider-val-single">60s</span>
            <input type="range" id="max-sec-single" min="10" max="240" step="5" value="60">
        </div>
        <button class="btn" id="gen-single-btn" onclick="generateSingle()">Generate</button>
    </div>

    <div id="single-result">
        <div class="audio-card">
            <h3 id="single-model-label">DPO Guided</h3>
            <audio id="audio-single" controls preload="auto"></audio>
            <div class="speed-btns" data-for="audio-single">
                <button class="speed-btn" data-speed="0.5">0.5x</button>
                <button class="speed-btn active" data-speed="1">1x</button>
                <button class="speed-btn" data-speed="1.25">1.25x</button>
                <button class="speed-btn" data-speed="1.5">1.5x</button>
                <button class="speed-btn" data-speed="2">2x</button>
            </div>
            <div class="tags" id="tags-single"></div>
        </div>
    </div>

    <div id="single-progress" style="display:none; text-align:center; padding: 30px; color: #888;">
        <span class="spinner"></span> <span id="single-progress-text">Generating...</span>
    </div>
</div>

<div class="paths-info" id="paths-info"></div>

<script>
let selectedModel = 'dpo';

// ── Tabs ──
function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === 'tab-' + tabId));
}

// ── Model selector ──
function selectModel(model) {
    selectedModel = model;
    document.querySelectorAll('.model-opt').forEach(o =>
        o.classList.toggle('selected', o.dataset.model === model));
}

// ── Sliders ──
document.querySelectorAll('input[type=range]').forEach(slider => {
    const valEl = document.getElementById(slider.id.replace('max-sec', 'slider-val'));
    if (valEl) {
        slider.addEventListener('input', () => { valEl.textContent = slider.value + 's'; });
    }
});

// ── Speed buttons ──
document.querySelectorAll('.speed-btns').forEach(group => {
    group.querySelectorAll('.speed-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const audioId = group.dataset.for;
            const audio = document.getElementById(audioId);
            audio.playbackRate = parseFloat(btn.dataset.speed);
            group.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
});

function resetSpeed(audioId) {
    const audio = document.getElementById(audioId);
    audio.playbackRate = 1;
    const group = document.querySelector('.speed-btns[data-for="' + audioId + '"]');
    if (group) {
        group.querySelectorAll('.speed-btn').forEach(b =>
            b.classList.toggle('active', b.dataset.speed === '1'));
    }
}

// ── Resilient fetch with retries (handles 502s from ngrok during GPU work) ──
async function fetchRetry(url, options, maxRetries = 30) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            const resp = await fetch(url, options);
            if (resp.status === 502 || resp.status === 504) {
                console.warn('Server busy (', resp.status, '), retry', i + 1);
                await new Promise(r => setTimeout(r, 3000));
                continue;
            }
            return resp;
        } catch (e) {
            console.warn('Fetch error, retry', i + 1, ':', e.message);
            await new Promise(r => setTimeout(r, 3000));
        }
    }
    throw new Error('Server unreachable after ' + maxRetries + ' retries');
}

// ── Poll a job until done ──
async function pollJob(jobId) {
    while (true) {
        await new Promise(r => setTimeout(r, 3000));
        try {
            const resp = await fetchRetry('/api/job/' + jobId);
            const data = await resp.json();
            if (data.status === 'done') return data.result;
            if (data.status === 'error') throw new Error(data.error);
            // status === 'pending' or 'running' — keep polling
        } catch (e) {
            if (e.message.includes('Server unreachable')) throw e;
            console.warn('Poll error, retrying:', e);
        }
    }
}

// ── A/B Compare ──
async function generateBoth() {
    const prompt = document.getElementById('prompt-ab').value.trim();
    const lyrics = document.getElementById('lyrics-ab').value.trim();
    if (!prompt || !lyrics) { alert('Please fill in both prompt and lyrics.'); return; }

    const maxSec = parseInt(document.getElementById('max-sec-ab').value);
    const btn = document.getElementById('gen-both-btn');
    btn.disabled = true;

    document.getElementById('results-section').style.display = 'none';
    document.getElementById('ab-progress').style.display = 'block';
    document.getElementById('ab-progress-text').textContent = 'Submitting original model job...';

    try {
        // Submit original job (with retry)
        const origSubmit = await fetchRetry('/api/submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: 'original' }),
        });
        const origJob = await origSubmit.json();
        document.getElementById('ab-progress-text').textContent = 'Generating original model output...';
        const origResult = await pollJob(origJob.job_id);

        // Show original result immediately
        document.getElementById('results-section').style.display = 'block';
        const audioOrig = document.getElementById('audio-orig');
        audioOrig.src = '/audio/' + origResult.audio_file;
        audioOrig.load();
        document.getElementById('tags-orig').textContent = 'Tags: ' + origResult.tags;
        resetSpeed('audio-orig');

        // Submit DPO job (with retry — server may be busy finishing up)
        document.getElementById('ab-progress-text').textContent = 'Submitting DPO model job...';
        const dpoSubmit = await fetchRetry('/api/submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: 'dpo' }),
        });
        const dpoJob = await dpoSubmit.json();
        document.getElementById('ab-progress-text').textContent = 'Generating DPO Guided model output...';
        const dpoResult = await pollJob(dpoJob.job_id);

        const audioDpo = document.getElementById('audio-dpo');
        audioDpo.src = '/audio/' + dpoResult.audio_file;
        audioDpo.load();
        document.getElementById('tags-dpo').textContent = 'Tags: ' + dpoResult.tags;
        resetSpeed('audio-dpo');

        document.getElementById('ab-progress').style.display = 'none';

    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.disabled = false;
        document.getElementById('ab-progress').style.display = 'none';
    }
}

// ── Single Generate ──
async function generateSingle() {
    const prompt = document.getElementById('prompt-single').value.trim();
    const lyrics = document.getElementById('lyrics-single').value.trim();
    if (!prompt || !lyrics) { alert('Please fill in both prompt and lyrics.'); return; }

    const maxSec = parseInt(document.getElementById('max-sec-single').value);
    const btn = document.getElementById('gen-single-btn');
    btn.disabled = true;

    document.getElementById('single-result').style.display = 'none';
    document.getElementById('single-progress').style.display = 'block';
    document.getElementById('single-progress-text').textContent =
        'Generating with ' + (selectedModel === 'dpo' ? 'DPO Guided' : 'original') + ' model...';

    try {
        const submitResp = await fetchRetry('/api/submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: selectedModel }),
        });
        const job = await submitResp.json();
        const result = await pollJob(job.job_id);

        document.getElementById('single-result').style.display = 'block';
        document.getElementById('single-model-label').textContent =
            (selectedModel === 'dpo' ? 'DPO Guided' : 'Original') + ' Model';
        const audio = document.getElementById('audio-single');
        audio.src = '/audio/' + result.audio_file;
        audio.load();
        document.getElementById('tags-single').textContent =
            'Tags: ' + result.tags + ' | Frames: ' + result.num_frames;
        resetSpeed('audio-single');

    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.disabled = false;
        document.getElementById('single-progress').style.display = 'none';
    }
}

// ── Status polling ──
async function pollStatus() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        document.getElementById('st-queue').textContent = data.pending;
        document.getElementById('st-completed').textContent = data.completed;
        document.getElementById('st-gen-wrap').style.display = data.generating ? '' : 'none';
    } catch (e) {}
}
setInterval(pollStatus, 2000);
pollStatus();

// ── Load paths info ──
fetch('/api/info').then(r => r.json()).then(data => {
    document.getElementById('paths-info').innerHTML =
        '<b>Original:</b> <code>' + data.original_path + '</code> &nbsp; ' +
        '<b>DPO Guided:</b> <code>' + data.dpo_path + '</code>';
});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="FastAPI A/B comparison UI for Erised DPO")
    parser.add_argument("--original-path", type=str, default=None)
    parser.add_argument("--dpo-path", type=str, default="/workspace/dpo_checkpoints_v8/dpo_best")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-length", type=int, default=60)
    args = parser.parse_args()

    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel
    import uvicorn

    from erised.config import ErisedConfig
    from erised.pipeline import ErisedPipeline

    config = ErisedConfig.from_env()
    original_model_path = args.original_path or config.model_path
    config.lazy_load = False

    logger.info("Loading pipeline from %s ...", original_model_path)
    pipeline = ErisedPipeline(config)
    device = next(pipeline.pipe.mula.parameters()).device
    logger.info("Pipeline loaded on %s", device)

    gen_lock = threading.Lock()
    gen_stats = {"pending": 0, "completed": 0, "generating": False}

    # ── DPO Guided: instead of swapping weights, we initialize the DPOGuider
    # which holds only the top 2 DPO-trained layers alongside the original model.
    # The bottom 26 shared layers run once; the top 2 branch for orig vs DPO logits.
    # logits_final = logits_orig + scale * (logits_dpo - logits_orig)
    logger.info("Initializing DPO Guided system from %s ...", args.dpo_path)
    pipeline.init_guided(dpo_checkpoint_path=args.dpo_path, n_dpo_layers=2)
    logger.info("DPO Guided system ready.")

    # ── Old weight-swapping approach (kept for reference) ──────────────
    # current_weights = {"path": original_model_path}
    #
    # def swap_weights(target_path):
    #     if current_weights["path"] == target_path:
    #         return
    #     logger.info("Swapping weights -> %s", target_path)
    #     load_safetensors_sharded(pipeline.pipe.mula, target_path, device=device)
    #     current_weights["path"] = target_path
    #     torch.cuda.empty_cache()
    # ── End old weight-swapping approach ───────────────────────────────

    # Job store: job_id -> {"status": "pending"|"running"|"done"|"error", "result": ..., "error": ...}
    jobs = {}
    job_queue = queue.Queue()

    def generation_worker():
        """Background thread that processes generation jobs one at a time.

        DPO Guided approach: for "original" model, we generate normally.
        For "dpo" model, we use the DPOGuider which applies DPO preferences
        at the logit level without swapping any weights.
        """
        while True:
            job_id, prompt, lyrics, max_sec, model_name = job_queue.get()
            jobs[job_id]["status"] = "running"
            gen_stats["generating"] = True

            try:
                with gen_lock:
                    t0 = time.time()

                    if model_name == "dpo":
                        # DPO Guided: use logit-level guidance (no weight swap needed)
                        result = pipeline.generate_guided(
                            prompt=prompt,
                            lyrics=lyrics,
                            max_audio_length_ms=int(max_sec * 1000),
                            dpo_scale=1.0,
                        )
                    else:
                        # Original model: standard generation
                        result = pipeline.generate(
                            prompt=prompt,
                            lyrics=lyrics,
                            max_audio_length_ms=int(max_sec * 1000),
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
    app = FastAPI(title="Erised Compare")

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class SubmitRequest(BaseModel):
        prompt: str
        lyrics: str
        max_sec: int = 60
        model: str = "dpo"  # "original" or "dpo"

    # NOTE: All endpoints are sync `def` (not `async def`) so FastAPI runs them
    # in a thread pool. This prevents GIL contention with the generation thread
    # from blocking responses — the main cause of 502s through ngrok.

    @app.get("/health")
    def health():
        """Health check endpoint for the Vercel frontend to detect GPU status."""
        return {"status": "ok", "model": "dpo_guided"}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/audio/{filename}")
    def serve_audio(filename: str):
        path = os.path.join(config.output_dir, filename)
        if not os.path.isfile(path):
            raise HTTPException(404, "Audio file not found")
        media = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(path, media_type=media)

    @app.post("/api/submit")
    def submit(req: SubmitRequest):
        """Submit a generation job. Returns immediately with a job_id to poll."""
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")
        if req.model not in ("original", "dpo"):
            raise HTTPException(400, "Model must be 'original' or 'dpo'")

        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {"status": "pending"}
        gen_stats["pending"] += 1
        job_queue.put((job_id, req.prompt, req.lyrics, req.max_sec, req.model))
        logger.info("Job %s submitted: model=%s, max_sec=%d", job_id, req.model, req.max_sec)
        return {"job_id": job_id}

    @app.get("/api/job/{job_id}")
    def get_job(job_id: str):
        """Poll job status. Returns status + result when done."""
        if job_id not in jobs:
            raise HTTPException(404, "Unknown job_id")
        return jobs[job_id]

    @app.get("/api/status")
    def status():
        return {
            "pending": gen_stats["pending"],
            "completed": gen_stats["completed"],
            "generating": gen_stats["generating"],
        }

    @app.get("/api/info")
    def info():
        return {
            "original_path": original_model_path,
            "dpo_path": args.dpo_path,
        }

    logger.info("Starting server on port %d", args.port)
    logger.info("Original: %s", original_model_path)
    logger.info("DPO Guided: %s", args.dpo_path)
    logger.info("Access at http://0.0.0.0:%d or via ngrok", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
