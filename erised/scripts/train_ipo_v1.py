#!/usr/bin/env python3
"""
IPO v1 — Full IPO training run
================================

Extends the successful exp_ipo experiment to a full 6-epoch training run.

IPO (Identity Preference Optimization) — Azar et al. 2023, arxiv 2310.12036
  loss = (delta - 1/(2*beta))^2
Bounded objective: gradient → 0 once delta reaches target (1/(2β) = 10),
preventing the cumulative noise / runaway drift problem from DPO.

Changes vs exp_ipo (2-epoch test):
  - epochs: 2 → 6 (safe because IPO self-regulates; 3x more gradient steps
    to let deltas climb closer to the target of 10)
  - save_every_n_steps: 9999 → 80 (track progression across epochs)
  - Everything else identical to what showed potential in exp_ipo

Recipe (unchanged from exp_ipo):
  - Loss: IPO squared loss
  - beta: 0.05 (IPO target = 1/(2*0.05) = 10)
  - local_loss_weight: 0.0 (melody-only, all signal into codebook-0)
  - lr: 5e-6, top 2 backbone layers, grad_accumulation=4, warmup=10
  - grad clip 1.0 (norm)

Output:
  /workspace/dpo_checkpoints_ipo_v1/dpo_best
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
logger = logging.getLogger("train_ipo_v1")

os.environ["ERISED_MODEL_PATH"] = "/workspace/heartlib/ckpt"
os.environ["ERISED_DPO_DB"] = "/workspace/heartlib/heartlib/dpo_preferences.db"
os.environ["ERISED_OUTPUT_DIR"] = "/workspace/heartlib/outputs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

# ── Patch trainer: (1) top-2 layers, (2) IPO loss ───────────────────
trainer_path = os.path.join(_repo_root, "erised", "dpo", "trainer.py")
trainer_bak = trainer_path + ".ipo_v1.bak"

with open(trainer_path, 'r') as f:
    original_source = f.read()

with open(trainer_bak, 'w') as f:
    f.write(original_source)
logger.info("Saved trainer.py backup to %s", trainer_bak)

patched = original_source

# Patch 1: fewer trainable backbone layers (top 2 of 28)
new = patched.replace("min(8, n_layers)", "min(2, n_layers)")
if new == patched:
    logger.warning("Could not patch n_trainable — may already be patched")
patched = new

# Patch 2: IPO loss (the key change)
# Replace:  loss = -F.logsigmoid(self.beta * delta).mean()
# With:     loss = ((delta - 1.0 / (2.0 * self.beta)) ** 2).mean()
# IPO target: delta should converge to 1/(2*beta) = 10. Unlike DPO,
# IPO has a bounded objective — gradient → 0 at the target, preventing
# runaway drift on small datasets.
old_loss = "loss = -F.logsigmoid(self.beta * delta).mean()"
new_loss = "loss = ((delta - 1.0 / (2.0 * self.beta)) ** 2).mean()  # IPO v1"
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
    beta=0.05,                     # IPO target = 1/(2*0.05) = 10
    learning_rate=5e-6,
    epochs=6,                      # Full run (3x exp_ipo's 2 epochs)
    grad_accumulation_steps=4,
    checkpoint_dir="/workspace/dpo_checkpoints_ipo_v1",
    global_loss_weight=1.0,
    local_loss_weight=0.0,         # Melody only — what showed potential
    warmup_steps=10,
    use_amp=True,
)
trainer.save_every_n_steps = 80    # Track progression across epochs

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

elapsed = (time.time() - start) / 60
logger.info("=" * 60)
logger.info("IPO v1 COMPLETE in %.1f minutes", elapsed)
logger.info("Checkpoint: /workspace/dpo_checkpoints_ipo_v1/dpo_best")
logger.info("=" * 60)
