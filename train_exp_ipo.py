#!/usr/bin/env python3
"""
EXPERIMENT: IPO loss + beta=0.05 + local=0 + 2 epochs
========================================================

Purpose: Test whether the "recommended ML improvements" discussed in
DPO_NOTES would actually produce an audibly more melody-aligned model.

Changes vs v12:
  1. Loss: DPO logsigmoid  ->  IPO squared loss
     (bounded target, less prone to overfit on 200 pairs)
  2. Beta: 0.1  ->  0.05
     (lower beta in IPO = higher target delta = more aggressive push)
  3. local_loss_weight: 0.5  ->  0.0
     (melody-only: all gradient signal into codebook-0 / structure)
  4. Epochs: 6  ->  2
     (IPO's stronger gradient signal means fewer steps needed,
      and grad clip at norm 1.0 bounds update size regardless)

Unchanged:
  - lr = 5e-6 (NOT raised — user correctly identified that higher LR
    is the noise-accumulation failure mode; we get "more learning"
    through lower beta + better loss, not higher LR)
  - Top 2 backbone layers trainable
  - grad_accumulation_steps = 4
  - warmup_steps = 10

Output:
  /workspace/dpo_checkpoints_exp_ipo/dpo_best
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
logger = logging.getLogger("train_exp_ipo")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_DPO_DB"] = "/workspace/heartlib/heartlib/dpo_preferences.db"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

# ── Patch trainer: (1) top-2 layers, (2) IPO loss ───────────────────
trainer_path = os.path.join(_repo_root, "erised", "dpo", "trainer.py")
trainer_bak = trainer_path + ".exp_ipo.bak"

with open(trainer_path, 'r') as f:
    original_source = f.read()

with open(trainer_bak, 'w') as f:
    f.write(original_source)
logger.info("Saved trainer.py backup to %s", trainer_bak)

patched = original_source

# Patch 1: fewer trainable backbone layers (same as v11/v12)
new = patched.replace("min(8, n_layers)", "min(2, n_layers)")
if new == patched:
    logger.warning("Could not patch n_trainable")
patched = new

# Patch 2: IPO loss (key experimental change)
# Replace:  loss = -F.logsigmoid(self.beta * delta).mean()
# With:     loss = ((delta - 1.0 / (2.0 * self.beta)) ** 2).mean()
# IPO target: delta should converge to 1/(2*beta). At beta=0.05, target=10.
# Unlike DPO, IPO has a bounded objective — it stops pushing once the
# target margin is reached, which prevents runaway drift on small data.
old_loss = "loss = -F.logsigmoid(self.beta * delta).mean()"
new_loss = "loss = ((delta - 1.0 / (2.0 * self.beta)) ** 2).mean()  # IPO experimental"
new = patched.replace(old_loss, new_loss)
if new == patched:
    logger.error("Could not patch loss line! Aborting.")
    sys.exit(1)
patched = new

with open(trainer_path, 'w') as f:
    f.write(patched)
logger.info("Patched trainer: top-2 layers AND IPO loss enabled")

# Import AFTER patching
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
    beta=0.05,                     # ← HALVED from v12
    learning_rate=5e-6,            # unchanged
    epochs=2,                      # ← 6 → 2 (IPO needs fewer steps)
    grad_accumulation_steps=4,
    checkpoint_dir="/workspace/dpo_checkpoints_exp_ipo",
    global_loss_weight=1.0,
    local_loss_weight=0.0,         # ← MELODY ONLY
    warmup_steps=10,
    use_amp=True,
)
trainer.save_every_n_steps = 9999  # don't save mid-epoch, just save best

try:
    trainer.train()
finally:
    try:
        with open(trainer_path, 'w') as f:
            f.write(original_source)
        logger.info("Restored original trainer.py")
    except Exception as e:
        logger.warning("Failed to restore trainer.py: %s", e)
        logger.warning("Restore manually from: %s", trainer_bak)

logger.info("DONE in %.1f minutes", (time.time() - start) / 60)
logger.info("Checkpoint: /workspace/dpo_checkpoints_exp_ipo/dpo_best")
