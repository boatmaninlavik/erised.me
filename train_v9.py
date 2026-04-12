#!/usr/bin/env python3
"""
v9 training: Top-2 backbone layers (same as v8), but more aggressive.

Changes from v8:
  - 200 preferences (up from 158) → better signal, less overfitting risk
  - lr=7e-6 (up from 5e-6) → faster convergence
  - 8 epochs (up from 6) → push loss lower
  - Using guided DPO at inference (original model generates, DPO steers logits),
    so we can tolerate more divergence from base without noise accumulation.

Starts from original base model (not any prior DPO checkpoint).
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
logger = logging.getLogger("train_v9")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_DPO_DB"] = "/workspace/heartlib/heartlib/dpo_preferences.db"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from erised.config import ErisedConfig
from erised.pipeline import ErisedPipeline
from erised.dpo.data import PreferenceStore
from erised.dpo import trainer as trainer_module

# ── Monkey-patch: change n_trainable from 8 to 2 ─────────────────
import types

trainer_path = os.path.join(_repo_root, "erised", "dpo", "trainer.py")
with open(trainer_path, 'r') as f:
    original_source = f.read()

# Patch: change "min(8, n_layers)" to "min(2, n_layers)"
patched_source = original_source.replace("min(8, n_layers)", "min(2, n_layers)")

if patched_source == original_source:
    logger.warning("Could not find 'min(8, n_layers)' to patch! Training with original 8 layers.")
else:
    with open(trainer_path, 'w') as f:
        f.write(patched_source)
    logger.info("Patched trainer: n_trainable = min(2, n_layers) [was min(8, n_layers)]")

    # Reload the module
    import importlib
    importlib.reload(trainer_module)

# Now import the patched trainer
from erised.dpo.trainer import DPOTrainer

start = time.time()

# Load pipeline
config = ErisedConfig.from_env()
config.lazy_load = False
pipeline = ErisedPipeline(config)
logger.info("Base model loaded.")

# Train
pref_store = PreferenceStore(os.environ["ERISED_DPO_DB"])
logger.info("Preference count: %d", pref_store.count())

trainer = DPOTrainer(
    pipeline=pipeline.pipe,
    preference_store=pref_store,
    beta=0.1,
    learning_rate=7e-6,       # Up from 5e-6 — guided DPO tolerates more divergence
    epochs=8,                 # Up from 6 — more data supports more epochs
    grad_accumulation_steps=4,
    checkpoint_dir="/workspace/dpo_checkpoints_v9",
    global_loss_weight=1.0,
    local_loss_weight=0.0,    # No decoder training (same as v8)
    warmup_steps=10,
    use_amp=True,
)
trainer.save_every_n_steps = 80

try:
    trainer.train()
finally:
    # ALWAYS restore the original trainer.py
    with open(trainer_path, 'w') as f:
        f.write(original_source)
    logger.info("Restored original trainer.py (n_trainable = min(8, n_layers))")

logger.info("DONE in %.1f minutes", (time.time() - start) / 60)
logger.info("Checkpoints: /workspace/dpo_checkpoints_v9")
