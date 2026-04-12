#!/usr/bin/env python3
"""
Round 3 of A/B/C generation: 2 more prompts × 3 models.

Same seed (42), same model loading order (base → v12 → exp_ipo).
New prompts chosen for genre diversity vs gen_compare.py & gen_compare2.py:
  - Neo-soul / R&B ballad (warm, lush, syncopated groove, live feel)
  - Bluegrass / country (acoustic, fast-picked, melody-forward, banjo-driven)

Prior rounds covered: popPunk, mandopop (R1) and abba, phonk (R2).
"""
import os, sys, time, glob, logging, shutil
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
logger = logging.getLogger("gen_compare3")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT_DIR = "/workspace/compare_output"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = [
    {
        "id": "neosoul",
        "prompt": (
            "A warm, late-night neo-soul ballad. Smooth Rhodes electric piano playing rich "
            "extended chords, a deep round fingerstyle electric bass with subtle slides, "
            "soft brushed drums with a laid-back swung groove around 75 BPM. Smoky, "
            "intimate, emotional vocal lead."
        ),
        "lyrics": (
            "[Verse 1]\nLate light pooling on the floor\nYour jacket hanging by the door\n"
            "[Chorus]\nStay a little longer love\nThe night is not yet done with us"
        ),
    },
    {
        "id": "bluegrass",
        "prompt": (
            "An upbeat traditional bluegrass tune around 130 BPM. Fast-picked acoustic guitar "
            "and a bright rolling five-string banjo trading the melody, supported by an "
            "upright bass walking the root and fifth, and a high lonesome fiddle on the chorus. "
            "Bright, joyful, and relentlessly forward-moving."
        ),
        "lyrics": (
            "[Verse 1]\nDown the holler past the pine\nWater running clear and fine\n"
            "[Chorus]\nCarry me home where the river bends\nBack to where the long road ends"
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
