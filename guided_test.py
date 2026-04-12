#!/usr/bin/env python3
"""
DPO Logit Guidance: Use the original model for clean generation,
guided by DPO model's logit differences.

Instead of baking DPO into the weights (which causes noise accumulation),
we keep the original model driving generation and apply the DPO preference
signal at the logit level:

    logits_final = logits_orig + scale * (logits_dpo - logits_orig)

The original model stays internally consistent (no noise), while the DPO
model's learned preferences steer token selection.
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
from copy import deepcopy
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent / "heartlib")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("erised.guided")

import torch
import torch.nn.functional as F
from tqdm import tqdm


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


def sample_topk(logits, topk, temperature):
    """Sample from logits with top-k filtering and temperature."""
    logits = logits / temperature
    filter_value = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = F.log_softmax(scores_processed, dim=-1)
    probs = F.softmax(scores_processed, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def _index_causal_mask(mask, pos):
    """Index into a causal mask with position indices."""
    return mask[None, None, pos.squeeze()]


def reset_model_caches(model):
    """Fully reset KV caches on a HeartMuLa model."""
    for model_part in (model.backbone, model.decoder):
        try:
            model_part.reset_caches()
        except (RuntimeError, AttributeError):
            pass
        for layer in model_part.layers:
            attn = getattr(layer, "attn", None)
            if attn is not None and getattr(attn, "kv_cache", None) is not None:
                attn.kv_cache = None
                attn.cache_enabled = False


def generate_frame_logits(model, tokens, tokens_mask, input_pos, cfg_scale,
                          continuous_segments=None, starts=None):
    """
    Run a single frame through the model and return raw logits
    (c0_logits and backbone last hidden state) WITHOUT sampling.
    """
    from heartlib.heartmula.modeling_heartmula import _index_causal_mask as _icm

    b, s, _ = tokens.size()
    curr_backbone_mask = _icm(model.backbone_causal_mask, input_pos)

    uncond_mask = None
    if cfg_scale > 1.0 and b > 1:
        actual_B = b // 2
        uncond_mask = torch.cat([
            torch.zeros(actual_B, dtype=torch.bool, device=tokens.device),
            torch.ones(actual_B, dtype=torch.bool, device=tokens.device),
        ])

    embeds = model._embed_tokens(tokens, uncond_mask=uncond_mask)
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2, dtype=embeds.dtype)

    if continuous_segments is not None:
        continuous_segments = model.muq_linear(continuous_segments)
        if uncond_mask is not None:
            uncond_embed = model.unconditional_text_embedding(
                torch.zeros(1, device=tokens.device, dtype=torch.long)
            )
            mask_expanded = uncond_mask.view(b, 1).expand_as(continuous_segments)
            continuous_segments = torch.where(mask_expanded, uncond_embed, continuous_segments)
        batch_indices = torch.arange(h.shape[0], device=h.device)
        h[batch_indices, starts] = continuous_segments

    h = model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
    last_h = h[:, -1, :]
    c0_logits = model.codebook0_head(last_h)

    return c0_logits, last_h


def generate_decoder_logits(model, last_h, c0_sample, cfg_scale, b, embeds_dtype):
    """
    Run the decoder for codebooks 1-7 and return logits at each step WITHOUT sampling.
    Returns list of (ci_logits, codebook_index) tuples.
    """
    from heartlib.heartmula.modeling_heartmula import _index_causal_mask as _icm

    model.decoder.reset_caches()
    c0_embed = model._embed_audio(0, c0_sample)
    curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
    curr_pos = (
        torch.arange(0, curr_h.size(1), device=curr_h.device)
        .unsqueeze(0).repeat(curr_h.size(0), 1)
    )
    curr_h = curr_h.to(embeds_dtype)

    decoder_logits_list = []
    for i in range(1, model.config.audio_num_codebooks):
        curr_decoder_mask = _icm(model.decoder_causal_mask, curr_pos)
        decoder_h = model.decoder(
            model.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
        )
        ci_logits = torch.mm(decoder_h[:, -1, :], model.audio_head[i - 1])
        decoder_logits_list.append(ci_logits)
        # We need to sample here to feed the next step — caller will provide the sample
        # Actually, we return logits and let the caller handle sampling + feeding
        # But the decoder is sequential... we need samples to continue.
        # So we return after each codebook and let caller drive.
        break  # Return one at a time

    return decoder_logits_list


def guided_generate_frame(orig_model, dpo_model, tokens, tokens_mask, input_pos,
                          temperature, topk, cfg_scale, dpo_scale,
                          continuous_segments=None, starts=None):
    """
    Generate one frame using logit guidance from two models.

    For codebook 0: combine logits from both models with guidance.
    For codebooks 1-7: use original model's decoder (these don't compound
    across frames, so guidance is less critical here).
    """
    from heartlib.heartmula.modeling_heartmula import _index_causal_mask as _icm

    b = tokens.size(0)
    embeds_dtype = next(orig_model.parameters()).dtype

    # === Codebook 0: guided logits ===
    # Get logits from both models
    orig_c0_logits, orig_last_h = generate_frame_logits(
        orig_model, tokens, tokens_mask, input_pos, cfg_scale,
        continuous_segments, starts
    )
    dpo_c0_logits, dpo_last_h = generate_frame_logits(
        dpo_model, tokens, tokens_mask, input_pos, cfg_scale,
        continuous_segments, starts
    )

    # Apply CFG first (if applicable), then DPO guidance
    if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
        actual_B = b // 2
        # CFG on original
        orig_cond = orig_c0_logits[:actual_B, :]
        orig_uncond = orig_c0_logits[actual_B:, :]
        orig_guided = orig_uncond + (orig_cond - orig_uncond) * cfg_scale
        # CFG on DPO
        dpo_cond = dpo_c0_logits[:actual_B, :]
        dpo_uncond = dpo_c0_logits[actual_B:, :]
        dpo_guided = dpo_uncond + (dpo_cond - dpo_uncond) * cfg_scale
        # DPO logit guidance
        final_logits = orig_guided + dpo_scale * (dpo_guided - orig_guided)
        c0_sample = sample_topk(final_logits, topk, temperature)
        c0_sample = c0_sample.repeat(2, 1)
    else:
        # DPO logit guidance (no CFG)
        final_logits = orig_c0_logits + dpo_scale * (dpo_c0_logits - orig_c0_logits)
        c0_sample = sample_topk(final_logits, topk, temperature)

    # === Codebooks 1-7: use original model's decoder (no cross-frame compounding) ===
    # Feed the guided c0_sample to original model's decoder
    orig_model.decoder.reset_caches()
    c0_embed_orig = orig_model._embed_audio(0, c0_sample)
    curr_h = torch.cat([orig_last_h.unsqueeze(1), c0_embed_orig], dim=1)
    curr_sample = c0_sample.clone()
    curr_pos = (
        torch.arange(0, curr_h.size(1), device=curr_h.device)
        .unsqueeze(0).repeat(curr_h.size(0), 1)
    )
    curr_h = curr_h.to(embeds_dtype)

    for i in range(1, orig_model.config.audio_num_codebooks):
        curr_decoder_mask = _icm(orig_model.decoder_causal_mask, curr_pos)
        decoder_h = orig_model.decoder(
            orig_model.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
        )
        ci_logits = torch.mm(decoder_h[:, -1, :], orig_model.audio_head[i - 1])

        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            cond_ci = ci_logits[:actual_B, :]
            uncond_ci = ci_logits[actual_B:, :]
            guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale
            ci_sample = sample_topk(guided_ci, topk, temperature)
            ci_sample = ci_sample.repeat(2, 1)
        else:
            ci_sample = sample_topk(ci_logits, topk, temperature)

        ci_embed = orig_model._embed_audio(i, ci_sample)
        curr_h = ci_embed
        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
        curr_pos = curr_pos[:, -1:] + 1

    # Also update the DPO model's decoder caches to stay in sync
    # (not strictly needed since we don't use DPO decoder, but keeps state consistent)
    # Actually, we need to feed the same c0_sample through the DPO model's backbone
    # KV cache is already updated from generate_frame_logits, so next frame will work.

    return curr_sample


def guided_forward(orig_model, dpo_model, pipe, model_inputs, max_audio_length_ms,
                   temperature, topk, cfg_scale, dpo_scale):
    """
    Full guided generation loop — replaces HeartMuLaGenPipeline._forward().
    """
    prompt_tokens = model_inputs["tokens"].to(next(orig_model.parameters()).device)
    prompt_tokens_mask = model_inputs["tokens_mask"].to(prompt_tokens.device)
    continuous_segment = model_inputs["muq_embed"].to(prompt_tokens.device)
    starts = model_inputs["muq_idx"]
    prompt_pos = model_inputs["pos"].to(prompt_tokens.device)
    frames = []

    bs_size = 2 if cfg_scale != 1.0 else 1

    # Setup caches for both models
    reset_model_caches(orig_model)
    reset_model_caches(dpo_model)
    torch.cuda.empty_cache()
    orig_model.setup_caches(bs_size)
    dpo_model.setup_caches(bs_size)

    device_type = prompt_tokens.device.type
    dtype = next(orig_model.parameters()).dtype

    # Initial frame (prompt processing)
    with torch.autocast(device_type=device_type, dtype=dtype):
        curr_token = guided_generate_frame(
            orig_model, dpo_model,
            tokens=prompt_tokens,
            tokens_mask=prompt_tokens_mask,
            input_pos=prompt_pos,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
            dpo_scale=dpo_scale,
            continuous_segments=continuous_segment,
            starts=starts,
        )
    frames.append(curr_token[0:1,])

    # Pad helper
    parallel_number = 8 + 1
    empty_id = pipe.config.empty_id

    def _pad_audio_token(token):
        padded = torch.ones((token.shape[0], parallel_number), device=token.device, dtype=torch.long) * empty_id
        padded[:, :-1] = token
        padded = padded.unsqueeze(1)
        mask = torch.ones_like(padded, device=token.device, dtype=torch.bool)
        mask[..., -1] = False
        return padded, mask

    max_audio_frames = max_audio_length_ms // 80

    # Autoregressive loop
    for i in tqdm(range(max_audio_frames), desc="Guided generation"):
        curr_token_padded, curr_token_mask = _pad_audio_token(curr_token)
        with torch.autocast(device_type=device_type, dtype=dtype):
            curr_token = guided_generate_frame(
                orig_model, dpo_model,
                tokens=curr_token_padded,
                tokens_mask=curr_token_mask,
                input_pos=prompt_pos[..., -1:] + i + 1,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                dpo_scale=dpo_scale,
            )
        if torch.any(curr_token[0:1, :] >= pipe.config.audio_eos_id):
            break
        frames.append(curr_token[0:1,])

    frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
    return {"frames": frames}


# ── HTML UI ──────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erised - DPO Guidance Test</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0a0a0a; color: #e5e5e5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 40px 60px;
}
h1 { text-align: center; color: #a855f7; font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; }
.subtitle { text-align: center; color: #888; margin-bottom: 30px; font-size: 14px; }
.card { background: #151515; padding: 24px; border-radius: 12px; margin-bottom: 24px; }
label { display: block; color: #7c3aed; font-weight: 600; font-size: 1.05rem; margin-bottom: 6px; }
textarea, input[type=text] {
    width: 100%; background: #3d3845; border: 1px solid #4a4555;
    color: #e5e5e5; padding: 14px; border-radius: 8px; font-size: 15px;
    font-family: inherit; line-height: 1.5; resize: vertical;
}
textarea:focus, input:focus { outline: none; border-color: #a855f7; }
input[type=range] { accent-color: #38bdf8; width: 100%; }
.slider-wrap { position: relative; margin-bottom: 16px; }
.slider-val { font-size: 12px; font-weight: 600; color: #38bdf8; position: absolute; right: 0; top: -20px; }
.btn {
    width: 100%; min-height: 56px; padding: 18px; font-size: 18px; font-weight: 600;
    border-radius: 8px; cursor: pointer; border: 1px solid #e2e8f0;
    background: #f8fafc; color: #7c3aed; display: block; text-align: center;
}
.btn:hover { background: #f1f5f9; }
.btn:disabled { background: #334155; color: #94a3b8; border-color: #475569; cursor: not-allowed; }
.audio-card {
    background: #151515; padding: 24px; border-radius: 12px; margin-bottom: 16px;
    border: 2px solid #333;
}
.audio-card h3 { color: #a855f7; margin: 0 0 12px 0; }
.audio-card audio { width: 100%; margin: 8px 0; }
.tags { font-size: 12px; color: #64748b; font-family: monospace; margin-top: 4px; }
.spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid #38bdf8; border-top-color: transparent;
    border-radius: 50%; animation: spin 0.8s linear infinite;
    vertical-align: middle; margin-right: 8px;
}
@keyframes spin { to { transform: rotate(360deg); } }
#results { display: none; }
</style>
</head>
<body>

<h1>Erised - Guided DPO</h1>
<p class="subtitle">Original model generates cleanly. DPO model guides preferences at the logit level. No noise accumulation.</p>

<div class="card">
    <div style="margin-bottom: 16px;">
        <label for="prompt">Musical Prompt</label>
        <textarea id="prompt" rows="3" placeholder="e.g., emotional pop ballad with piano and strings"></textarea>
    </div>
    <div style="margin-bottom: 16px;">
        <label for="lyrics">Lyrics</label>
        <textarea id="lyrics" rows="8" placeholder="[Verse 1]&#10;Your lyrics here...&#10;&#10;[Chorus]&#10;Your chorus here..."></textarea>
    </div>
    <div class="slider-wrap">
        <label for="max-sec">Max length (seconds)</label>
        <span class="slider-val" id="slider-val-sec">60s</span>
        <input type="range" id="max-sec" min="10" max="240" step="5" value="60">
    </div>
    <div class="slider-wrap">
        <label for="dpo-scale">DPO Guidance Scale</label>
        <span class="slider-val" id="slider-val-dpo">1.0</span>
        <input type="range" id="dpo-scale" min="0" max="1.0" step="0.05" value="0.4">
    </div>
    <button class="btn" id="gen-btn" onclick="generate()">Generate (Original + Guided)</button>
</div>

<div id="progress" style="display:none; text-align:center; padding: 30px; color: #888;">
    <span class="spinner"></span> <span id="progress-text">Generating...</span>
</div>

<div id="results">
    <div class="audio-card">
        <h3>Original Model (baseline)</h3>
        <audio id="audio-orig" controls preload="auto"></audio>
        <div class="tags" id="tags-orig"></div>
    </div>
    <div class="audio-card">
        <h3>DPO Guided (original + DPO logit guidance)</h3>
        <audio id="audio-guided" controls preload="auto"></audio>
        <div class="tags" id="tags-guided"></div>
    </div>
</div>

<script>
document.querySelectorAll('input[type=range]').forEach(slider => {
    const valEl = document.getElementById(slider.id === 'max-sec' ? 'slider-val-sec' : 'slider-val-dpo');
    if (valEl) slider.addEventListener('input', () => {
        valEl.textContent = slider.id === 'max-sec' ? slider.value + 's' : slider.value;
    });
});

async function fetchRetry(url, options, maxRetries = 60) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            const resp = await fetch(url, options);
            if (resp.status === 502 || resp.status === 504) {
                await new Promise(r => setTimeout(r, 3000)); continue;
            }
            return resp;
        } catch (e) { await new Promise(r => setTimeout(r, 3000)); }
    }
    throw new Error('Server unreachable');
}

async function pollJob(jobId) {
    while (true) {
        await new Promise(r => setTimeout(r, 3000));
        try {
            const resp = await fetchRetry('/api/job/' + jobId);
            const data = await resp.json();
            if (data.status === 'done') return data.result;
            if (data.status === 'error') throw new Error(data.error);
        } catch (e) {
            if (e.message.includes('unreachable')) throw e;
        }
    }
}

async function generate() {
    const prompt = document.getElementById('prompt').value.trim();
    const lyrics = document.getElementById('lyrics').value.trim();
    if (!prompt || !lyrics) { alert('Fill in both fields.'); return; }

    const maxSec = parseInt(document.getElementById('max-sec').value);
    const dpoScale = parseFloat(document.getElementById('dpo-scale').value);
    const btn = document.getElementById('gen-btn');
    btn.disabled = true;
    document.getElementById('results').style.display = 'none';
    document.getElementById('progress').style.display = 'block';

    try {
        // Generate original
        document.getElementById('progress-text').textContent = 'Generating original...';
        let resp = await fetchRetry('/api/submit', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: 'original' }),
        });
        let job = await resp.json();
        const origResult = await pollJob(job.job_id);

        document.getElementById('results').style.display = 'block';
        document.getElementById('audio-orig').src = '/audio/' + origResult.audio_file;
        document.getElementById('audio-orig').load();
        document.getElementById('tags-orig').textContent = 'Tags: ' + origResult.tags;

        // Generate guided
        document.getElementById('progress-text').textContent = 'Generating DPO guided (scale=' + dpoScale + ')...';
        resp = await fetchRetry('/api/submit', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt, lyrics, max_sec: maxSec, model: 'guided', dpo_scale: dpoScale }),
        });
        job = await resp.json();
        const guidedResult = await pollJob(job.job_id);

        document.getElementById('audio-guided').src = '/audio/' + guidedResult.audio_file;
        document.getElementById('audio-guided').load();
        document.getElementById('tags-guided').textContent = 'Tags: ' + guidedResult.tags + ' | DPO scale: ' + dpoScale;

        document.getElementById('progress').style.display = 'none';
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.disabled = false;
        document.getElementById('progress').style.display = 'none';
    }
}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="DPO Logit Guidance Test")
    parser.add_argument("--dpo-path", type=str, default="/workspace/dpo_checkpoints_v8/dpo_best")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--dpo-scale", type=float, default=0.4)
    args = parser.parse_args()

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    from erised.config import ErisedConfig
    from erised.pipeline import ErisedPipeline

    # Load pipeline (original model)
    config = ErisedConfig.from_env()
    config.lazy_load = False
    logger.info("Loading original pipeline...")
    pipeline = ErisedPipeline(config)
    orig_model = pipeline.pipe.mula
    device = next(orig_model.parameters()).device
    dtype = next(orig_model.parameters()).dtype
    logger.info("Original model loaded on %s", device)

    # Create DPO model (deep copy + load DPO weights)
    logger.info("Creating DPO model copy...")
    dpo_model = deepcopy(orig_model)
    dpo_model.to(device)
    logger.info("Loading DPO weights from %s", args.dpo_path)
    load_safetensors_sharded(dpo_model, args.dpo_path, device=device)
    dpo_model.eval()
    logger.info("DPO model loaded. VRAM: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    gen_lock = threading.Lock()
    gen_stats = {"pending": 0, "completed": 0, "generating": False}
    jobs = {}
    job_queue = queue.Queue()

    def generate_original(prompt, lyrics, max_sec, cfg_scale):
        """Generate with original model only."""
        result = pipeline.generate(
            prompt=prompt, lyrics=lyrics,
            max_audio_length_ms=int(max_sec * 1000),
            cfg_scale=cfg_scale,
        )
        return result

    def generate_guided(prompt, lyrics, max_sec, cfg_scale, dpo_scale):
        """Generate with DPO logit guidance."""
        tags = pipeline.tag_converter.convert(prompt)
        gen_id = uuid.uuid4().hex[:12]
        audio_path = os.path.join(config.output_dir, f"{gen_id}.mp3")

        model_inputs = pipeline.pipe.preprocess(
            {"tags": tags, "lyrics": lyrics},
            cfg_scale=cfg_scale,
        )

        # Reset caches for both models
        reset_model_caches(orig_model)
        reset_model_caches(dpo_model)
        torch.cuda.empty_cache()

        with torch.no_grad():
            model_outputs = guided_forward(
                orig_model, dpo_model, pipeline.pipe,
                model_inputs,
                max_audio_length_ms=int(max_sec * 1000),
                temperature=config.temperature,
                topk=config.topk,
                cfg_scale=cfg_scale,
                dpo_scale=dpo_scale,
            )

        frames = model_outputs["frames"]
        with torch.no_grad():
            pipeline.pipe.postprocess(model_outputs, save_path=audio_path)

        logger.info("Guided audio saved to %s (%d frames, dpo_scale=%.1f)",
                     audio_path, frames.shape[-1], dpo_scale)

        from erised.pipeline import GenerationResult
        return GenerationResult(
            generation_id=gen_id,
            audio_path=audio_path,
            tokens_path="",
            prompt=prompt,
            lyrics=lyrics,
            tags_used=tags,
            num_frames=frames.shape[-1],
        )

    def worker():
        while True:
            job_id, prompt, lyrics, max_sec, model_name, cfg_scale, dpo_scale = job_queue.get()
            jobs[job_id]["status"] = "running"
            gen_stats["generating"] = True
            try:
                with gen_lock:
                    t0 = time.time()
                    if model_name == "original":
                        result = generate_original(prompt, lyrics, max_sec, cfg_scale)
                    else:
                        result = generate_guided(prompt, lyrics, max_sec, cfg_scale, dpo_scale)
                    elapsed = time.time() - t0
                    logger.info("[%s] Generated in %.1fs (%d frames)", model_name, elapsed, result.num_frames)

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

    threading.Thread(target=worker, daemon=True).start()

    app = FastAPI(title="Erised Guided DPO")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class SubmitRequest(BaseModel):
        prompt: str
        lyrics: str
        max_sec: int = 60
        model: str = "guided"
        cfg_scale: float | None = None
        dpo_scale: float | None = None

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True, "mode": "guided-dpo-v9"}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/audio/{filename}")
    def serve_audio(filename: str):
        path = os.path.join(config.output_dir, filename)
        if not os.path.isfile(path):
            raise HTTPException(404, "Not found")
        media = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(path, media_type=media)

    @app.post("/api/submit")
    def submit(req: SubmitRequest):
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")
        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {"status": "pending"}
        gen_stats["pending"] += 1
        cfg = req.cfg_scale if req.cfg_scale is not None else 1.5
        dpo_s = req.dpo_scale if req.dpo_scale is not None else args.dpo_scale
        # Accept "dpo" as alias for "guided" (erised.me frontend sends "dpo")
        model_name = "guided" if req.model == "dpo" else req.model
        job_queue.put((job_id, req.prompt, req.lyrics, req.max_sec, model_name, cfg, dpo_s))
        return {"job_id": job_id}

    @app.get("/api/job/{job_id}")
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Unknown job")
        return jobs[job_id]

    @app.get("/api/status")
    def status():
        return gen_stats

    logger.info("Starting guided server on port %d", args.port)
    logger.info("Original: %s", config.model_path)
    logger.info("DPO: %s", args.dpo_path)
    logger.info("Default DPO scale: %.1f", args.dpo_scale)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
