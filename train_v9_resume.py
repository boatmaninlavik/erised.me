#!/usr/bin/env python3
"""Resume v9 training from epoch 6 checkpoint to finish epochs 7-8."""
import os, sys, time, logging
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
logger = logging.getLogger("train_v9_resume")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_DPO_DB"] = "/workspace/heartlib/heartlib/dpo_preferences.db"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

# Patch trainer BEFORE any erised imports
trainer_path = os.path.join(_repo_root, "erised", "dpo", "trainer.py")
with open(trainer_path, 'r') as f:
    original_source = f.read()

patched_source = original_source.replace("min(8, n_layers)", "min(2, n_layers)")
if patched_source != original_source:
    with open(trainer_path, 'w') as f:
        f.write(patched_source)
    logger.info("Patched trainer: n_trainable = min(2, n_layers)")

from erised.config import ErisedConfig
from erised.pipeline import ErisedPipeline
from erised.dpo.data import PreferenceStore
from erised.dpo.trainer import DPOTrainer

start = time.time()

config = ErisedConfig.from_env()
config.lazy_load = False
pipeline = ErisedPipeline(config)
logger.info("Base model loaded.")

pref_store = PreferenceStore(os.environ["ERISED_DPO_DB"])
logger.info("Preference count: %d", pref_store.count())

trainer = DPOTrainer(
    pipeline=pipeline.pipe,
    preference_store=pref_store,
    beta=0.1,
    learning_rate=7e-6,
    epochs=8,
    grad_accumulation_steps=4,
    checkpoint_dir="/workspace/dpo_checkpoints_v9",
    global_loss_weight=1.0,
    local_loss_weight=0.0,
    warmup_steps=10,
    use_amp=True,
)
trainer.save_every_n_steps = 80

try:
    trainer.train(resume_from="/workspace/dpo_checkpoints_v9/dpo_epoch_6")
finally:
    with open(trainer_path, 'w') as f:
        f.write(original_source)
    logger.info("Restored original trainer.py")

logger.info("DONE in %.1f minutes", (time.time() - start) / 60)
