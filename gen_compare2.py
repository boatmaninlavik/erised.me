#!/usr/bin/env python3
"""
Round 2 of A/B/C generation: 2 more prompts × 3 models.

Same seed (42), same model loading order (base → v12 → exp_ipo).
New prompts chosen for genre diversity vs gen_compare.py:
  - ABBA-style Euro-pop anthem (upbeat, melodic, big chorus)
  - Drift phonk (aggressive, dark, rhythm-driven, electronic)
"""
import os, sys, time, glob, logging, gc, shutil
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
logger = logging.getLogger("gen_compare2")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = "/workspace/compare_output"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = [
    {
        "id": "abba",
        "prompt": (
            "A high-energy, cathartic Euro-pop anthem in the style of late-70s ABBA. "
            "Built around a driving, rhythmic piano playing staccato octave chords and "
            "a melodic, funky bassline. Bright, layered vocal harmonies on a soaring chorus."
        ),
        "lyrics": (
            "[Verse 1]\nThe taxi glass is streaked with rain\nEach drop distorts a window pane\n"
            "[Chorus]\nDance until the morning comes\nSinging out our broken songs"
        ),
    },
    {
        "id": "phonk",
        "prompt": (
            "An aggressive, dark drift phonk track. The beat is driven by a heavily distorted "
            "and compressed 808 kick, a relentless TR-808 cowbell melody, and fast, skittering "
            "hi-hats. Tempo is fast, around 150 BPM. Menacing and rhythmic."
        ),
        "lyrics": (
            "[Verse 1]\nThe wire's cut, the engine's cold\nA story that you left untold\n"
            "[Chorus]\nReached the edge and never showed\nNothing left along this road"
        ),
    },
]

SEED = 42

import torch
from safetensors.torch import load_file

def load_safetensors_sharded(model, model_path):
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f, device="cuda"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded %d tensors from %s (missing=%d)", len(state_dict), model_path, len(missing))

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
            max_audio_length_ms=30_000,
        )
        shutil.copy(result.audio_path, out_path)
        logger.info("[%s] %s done in %.1fs  → %s",
                    model_tag, p["id"], time.time() - t0, out_path)

# 1. BASE
logger.info("═" * 60)
logger.info("STAGE 1/3: BASE MODEL")
logger.info("═" * 60)
pipeline = load_pipeline()
gen_for_model(pipeline, "base", PROMPTS)

# 2. V12
logger.info("═" * 60)
logger.info("STAGE 2/3: V12")
logger.info("═" * 60)
load_safetensors_sharded(pipeline.pipe.mula, "/workspace/dpo_checkpoints_v12/dpo_best")
gen_for_model(pipeline, "v12", PROMPTS)

# 3. EXP_IPO
logger.info("═" * 60)
logger.info("STAGE 3/3: EXP_IPO")
logger.info("═" * 60)
load_safetensors_sharded(pipeline.pipe.mula, "/workspace/dpo_checkpoints_exp_ipo/dpo_best")
gen_for_model(pipeline, "exp_ipo", PROMPTS)

logger.info("═" * 60)
logger.info("DONE. New files in %s :", OUT_DIR)
for p in PROMPTS:
    for tag in ["base", "v12", "exp_ipo"]:
        logger.info("  %s_%s.mp3", p["id"], tag)
