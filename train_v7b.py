#!/usr/bin/env python3
"""
v7b training: same LR as v6 (5e-6) + local_loss_weight=0.5 to fix decoder noise.
Starts from v6's best weights. Epochs=4, checkpoint every 80 steps.
"""
import os, sys, time, logging
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent / "erised_repo")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("train_v7b")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_DPO_DB"] = "/workspace/heartlib/heartlib/dpo_preferences.db"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import glob as _glob
from safetensors.torch import load_file as _load_safetensors
from erised.config import ErisedConfig
from erised.pipeline import ErisedPipeline
from erised.dpo.data import PreferenceStore
from erised.dpo.trainer import DPOTrainer

start = time.time()

# Load pipeline
config = ErisedConfig.from_env()
config.lazy_load = False
pipeline = ErisedPipeline(config)

# Load v6 best weights
logger.info("Loading v6 best weights...")
v6_path = "/workspace/dpo_checkpoints_v6/dpo_best"
device = next(pipeline.pipe.mula.parameters()).device
state_dict = {}
for f in sorted(_glob.glob(os.path.join(v6_path, "*.safetensors"))):
    state_dict.update(_load_safetensors(f, device=str(device)))
missing, unexpected = pipeline.pipe.mula.load_state_dict(state_dict, strict=False)
logger.info("v6 weights loaded: %d tensors, %d missing, %d unexpected",
            len(state_dict), len(missing), len(unexpected))

# Train
pref_store = PreferenceStore(os.environ["ERISED_DPO_DB"])
trainer = DPOTrainer(
    pipeline=pipeline.pipe,
    preference_store=pref_store,
    beta=0.1,
    learning_rate=5e-6,       # Same as v6
    epochs=4,                 # 4 epochs
    grad_accumulation_steps=4,
    checkpoint_dir="/workspace/dpo_checkpoints_v7b",
    global_loss_weight=1.0,
    local_loss_weight=0.5,    # THE FIX: train decoder too
    warmup_steps=10,
    use_amp=True,
)
trainer.save_every_n_steps = 80  # checkpoint every 80 steps

trainer.train()

logger.info("DONE in %.1f minutes", (time.time() - start) / 60)
logger.info("Checkpoints: /workspace/dpo_checkpoints_v7b")
