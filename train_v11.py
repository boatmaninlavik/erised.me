#!/usr/bin/env python3
"""
v11 training: Same setup as v8 (top-2 backbone layers, lr=5e-6, 6 epochs)
but trained on all 200 preferences instead of 158.

Key differences from v10:
  - NO resuming. Run start-to-finish in tmux to avoid interruption.
  - v10's dpo_best was corrupted by a resume bug (avg_loss divided by
    total pairs instead of only the steps that actually ran after resume,
    giving a fake 0.1386 loss that triggered a bad best-model save).

Lessons learned:
  - Never resume mid-epoch — the avg_loss calculation is broken on resume.
  - Run in tmux so closing laptop doesn't kill training.
  - Keep it simple: same hyperparams that worked for v8.
"""
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
logger = logging.getLogger("train_v11")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_DPO_DB"] = "/workspace/heartlib/heartlib/dpo_preferences.db"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

# ── Patch trainer: top-2 layers instead of top-8 ─────────────────
trainer_path = os.path.join(_repo_root, "erised", "dpo", "trainer.py")
trainer_bak = trainer_path + ".bak"

with open(trainer_path, 'r') as f:
    original_source = f.read()

# Write backup BEFORE patching
with open(trainer_bak, 'w') as f:
    f.write(original_source)
logger.info("Saved trainer.py backup to %s", trainer_bak)

patched_source = original_source.replace("min(8, n_layers)", "min(2, n_layers)")
if patched_source != original_source:
    with open(trainer_path, 'w') as f:
        f.write(patched_source)
    logger.info("Patched trainer: n_trainable = min(2, n_layers) [was min(8, n_layers)]")
else:
    logger.warning("Could not find 'min(8, n_layers)' to patch!")

# Import AFTER patching (erised/__init__.py triggers pipeline imports)
from erised.config import ErisedConfig
from erised.pipeline import ErisedPipeline
from erised.dpo.data import PreferenceStore
from erised.dpo.trainer import DPOTrainer

start = time.time()

# Load pipeline
config = ErisedConfig.from_env()
config.lazy_load = False
pipeline = ErisedPipeline(config)
logger.info("Base model loaded.")

# Train — exact same hyperparams as v8
pref_store = PreferenceStore(os.environ["ERISED_DPO_DB"])
logger.info("Preference count: %d", pref_store.count())

trainer = DPOTrainer(
    pipeline=pipeline.pipe,
    preference_store=pref_store,
    beta=0.1,
    learning_rate=5e-6,
    epochs=6,
    grad_accumulation_steps=4,
    checkpoint_dir="/workspace/dpo_checkpoints_v11",
    global_loss_weight=1.0,
    local_loss_weight=0.5,
    warmup_steps=10,
    use_amp=True,
)
trainer.save_every_n_steps = 80

try:
    trainer.train()
finally:
    # ALWAYS restore the original trainer.py
    try:
        with open(trainer_path, 'w') as f:
            f.write(original_source)
        logger.info("Restored original trainer.py (n_trainable = min(8, n_layers))")
    except Exception as e:
        logger.warning("Failed to restore trainer.py from memory: %s", e)
        logger.warning("Restore manually from: %s", trainer_bak)

logger.info("DONE in %.1f minutes", (time.time() - start) / 60)
logger.info("Checkpoints: /workspace/dpo_checkpoints_v11")
