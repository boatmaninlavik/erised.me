#!/usr/bin/env python3
"""
Generate matched songs from 3 models for A/B/C listening test:
  1. BASE      - untrained HeartMuLa
  2. V12       - current DPO approach (logsigmoid, beta=0.1, local=0.5, 6 ep)
  3. EXP_IPO   - proposed approach  (IPO, beta=0.05, local=0.0, 2 ep)

Uses the SAME prompt + SAME seed for each model so differences come
from training only, not sampling variance.

Outputs:
  /workspace/compare_output/<prompt_id>_base.mp3
  /workspace/compare_output/<prompt_id>_v12.mp3
  /workspace/compare_output/<prompt_id>_exp_ipo.mp3

Time budget: 2 prompts × 3 models × ~90s/song + ~3 model loads × 60s
  = ~12 min generation + ~3 min loading = ~15 min total
"""
import os, sys, time, glob, logging, gc
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent / "heartlib")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("gen_compare")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = "/workspace/compare_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Two real prompts from the preference DB ────────────────────────
# Chosen because they span very different genres (pop-punk vs phonk)
# so any melody-learning effect should be easier to hear.
PROMPTS = [
    {
        "id": "popPunk",
        "prompt": (
            "High-energy 1960s bubblegum pop-punk, 160 BPM, bright upbeat indie pop, "
            "Features driving dry drums with heavy tambourine on every beat, "
            "bouncy melodic bassline, staccato Juno-60 synth stabs. Catchy hook."
        ),
        "lyrics": (
            "[Verse 1]\nThe air is thick\nLike static on a radio out of tune\n"
            "I was waiting for a signal in a crowded room\n"
            "[Chorus]\nTurn the dial one more time\nLet it ring inside my mind"
        ),
    },
    {
        "id": "mandopop",
        "prompt": (
            "2000s-era Mandopop power ballad. Melancholic, dramatic, heartfelt. "
            "Prominent clean grand piano melody opens the track, simple foundational "
            "bassline, emotional vocal lead."
        ),
        "lyrics": (
            "[Verse 1]\nYou leave a coffee cup here by the sink\n"
            "It's still half full, it gives me time to think\n"
            "[Chorus]\nAbout the histories you wrote on me\nAnd all the pages we won't see"
        ),
    },
]

SEED = 42  # fixed so all models face the same sampling randomness

import torch
from safetensors.torch import load_file

def load_safetensors_sharded(model, model_path):
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors in {model_path}")
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f, device="cuda"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded %d tensors from %s (missing=%d, unexpected=%d)",
                len(state_dict), model_path, len(missing), len(unexpected))

def load_pipeline():
    from erised.config import ErisedConfig
    from erised.pipeline import ErisedPipeline
    config = ErisedConfig.from_env()
    config.lazy_load = False
    return ErisedPipeline(config)

def gen_for_model(pipeline, model_tag, prompts):
    for p in prompts:
        out_path = os.path.join(OUT_DIR, f"{p['id']}_{model_tag}.mp3")
        if os.path.exists(out_path):
            logger.info("[%s] %s already exists, skipping", model_tag, out_path)
            continue
        logger.info("[%s] generating %s ...", model_tag, p["id"])
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        t0 = time.time()
        result = pipeline.generate(
            prompt=p["prompt"],
            lyrics=p["lyrics"],
            max_audio_length_ms=30_000,  # 30s is enough to hear melody character
        )
        # Copy the generated file to our comparison dir under the tagged name
        import shutil
        shutil.copy(result.audio_path, out_path)
        logger.info("[%s] %s done in %.1fs  → %s",
                    model_tag, p["id"], time.time() - t0, out_path)

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# ── 1. BASE MODEL ────────────────────────────────────────────────────
logger.info("═" * 60)
logger.info("STAGE 1/3: BASE MODEL")
logger.info("═" * 60)
pipeline = load_pipeline()
gen_for_model(pipeline, "base", PROMPTS)

# ── 2. V12 (load checkpoint into the SAME pipeline) ──────────────────
logger.info("═" * 60)
logger.info("STAGE 2/3: V12 (DPO logsigmoid, beta=0.1, local=0.5, 6 ep)")
logger.info("═" * 60)
v12_ckpt = "/workspace/dpo_checkpoints_v12/dpo_best"
if os.path.isdir(v12_ckpt) and glob.glob(os.path.join(v12_ckpt, "*.safetensors")):
    load_safetensors_sharded(pipeline.pipe.mula, v12_ckpt)
    gen_for_model(pipeline, "v12", PROMPTS)
else:
    logger.warning("No v12 best checkpoint found at %s — skipping v12 stage", v12_ckpt)

# ── 3. EXP_IPO ───────────────────────────────────────────────────────
logger.info("═" * 60)
logger.info("STAGE 3/3: EXP_IPO (IPO, beta=0.05, local=0.0, 2 ep)")
logger.info("═" * 60)
exp_ckpt = "/workspace/dpo_checkpoints_exp_ipo/dpo_best"
if os.path.isdir(exp_ckpt) and glob.glob(os.path.join(exp_ckpt, "*.safetensors")):
    load_safetensors_sharded(pipeline.pipe.mula, exp_ckpt)
    gen_for_model(pipeline, "exp_ipo", PROMPTS)
else:
    logger.error("No exp_ipo checkpoint found at %s — did training finish?", exp_ckpt)

logger.info("═" * 60)
logger.info("DONE. Listen to files in %s :", OUT_DIR)
for p in PROMPTS:
    for tag in ["base", "v12", "exp_ipo"]:
        fp = os.path.join(OUT_DIR, f"{p['id']}_{tag}.mp3")
        if os.path.exists(fp):
            logger.info("  %s", fp)
