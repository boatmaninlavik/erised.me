#!/usr/bin/env python3
"""
v8 training: Top-2 backbone layers only (instead of top-8).

Root cause of noise: Training 8/28 backbone layers creates too much mismatch
between frozen lower layers and tuned upper layers. During autoregressive generation,
errors compound over time → progressive noise buildup.

Fix: Train only top 2 layers + output heads = minimal mismatch, but still enough
for gradients to flow. This is ~7% of params instead of ~26%.

Starts from original base model (not v6).
Uses the standard DPOTrainer with a monkey-patched n_trainable.
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
logger = logging.getLogger("train_v8")

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
_original_code = trainer_module.DPOTrainer.train.__code__

# Instead of patching bytecode, we patch the source at the module level
# by replacing the hardcoded "min(8, n_layers)" in the train method.
# We do this by modifying the trainer's train() to intercept after freeze.
import types

_orig_train = trainer_module.DPOTrainer.train

def _patched_train(self):
    """Wrapper that patches n_trainable to 2 before training starts."""
    # Temporarily patch the min() call by replacing the source attribute
    # Actually, easier: just override the freeze logic by calling original train
    # but intercepting the model setup. Let's do it differently:
    # We'll modify the source of the trainer module directly.
    pass

# Simpler approach: just edit the trainer.py temporarily
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
trainer = DPOTrainer(
    pipeline=pipeline.pipe,
    preference_store=pref_store,
    beta=0.1,
    learning_rate=5e-6,       # Same as v6
    epochs=6,                 # More epochs to compensate for fewer params
    grad_accumulation_steps=4,
    checkpoint_dir="/workspace/dpo_checkpoints_v8",
    global_loss_weight=1.0,
    local_loss_weight=0.0,    # No decoder training
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
logger.info("Checkpoints: /workspace/dpo_checkpoints_v8")
