#!/usr/bin/env python3
"""
End-to-end DPO training script for HeartMuLa on RunPod.

Usage:
    # Quick test (1 epoch):
    python erised/scripts/train_dpo.py --epochs 1

    # Full overnight training:
    nohup python erised/scripts/train_dpo.py \
        --epochs 10 --lr 5e-7 --beta 0.1 \
        --checkpoint-dir /workspace/dpo_checkpoints \
        > /workspace/dpo_training.log 2>&1 &

    # Watch the log:
    tail -f /workspace/dpo_training.log
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the repo root is on the Python path so `erised` is importable
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("erised.train_dpo")


def find_and_restore_backup(db_path: str, output_dir: str) -> bool:
    """Find the most recent backup and restore it if no local DB exists."""
    if Path(db_path).exists():
        import sqlite3
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
        conn.close()
        if count > 0:
            logger.info("Found existing preference DB at %s with %d pairs", db_path, count)
            return True

    # Look for backups
    backup_dirs = [
        Path("/workspace/erised_backups"),
        Path("/workspace/heartlib/erised_backups"),
        Path(os.path.dirname(os.path.abspath(__file__))).parent.parent / "erised_backups",
    ]

    backup_zip = None
    backup_folder = None

    for bdir in backup_dirs:
        if not bdir.exists():
            continue
        # Check for unzipped backup folders first
        folders = sorted(bdir.glob("erised_backup_*"), reverse=True)
        for f in folders:
            if f.is_dir() and (f / "preferences.db").exists():
                backup_folder = f
                break
        # Check zip files
        zips = sorted(bdir.glob("erised_backup_*.zip"), reverse=True)
        if zips:
            backup_zip = zips[0]
        if backup_folder or backup_zip:
            break

    if backup_folder:
        logger.info("Found backup folder: %s", backup_folder)
        from erised.backup import import_restore
        # The folder itself is the extracted backup structure
        # We need to restore from it manually
        import shutil
        import sqlite3

        src_db = backup_folder / "preferences.db"
        tokens_src = backup_folder / "tokens"
        audio_src = backup_folder / "audio"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        shutil.copy(src_db, db_path)

        # Copy token files
        copied = 0
        if tokens_src.exists():
            for f in tokens_src.glob("*_tokens.pt"):
                shutil.copy(f, os.path.join(output_dir, f.name))
                copied += 1
        logger.info("Copied %d token files to %s", copied, output_dir)

        # Rewrite token paths in DB
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, winner_id, loser_id FROM preferences").fetchall()
        out_dir_abs = os.path.abspath(output_dir)
        for row_id, winner_id, loser_id in rows:
            conn.execute(
                "UPDATE preferences SET winner_tokens_path = ?, loser_tokens_path = ? WHERE id = ?",
                (
                    os.path.join(out_dir_abs, f"{winner_id}_tokens.pt"),
                    os.path.join(out_dir_abs, f"{loser_id}_tokens.pt"),
                    row_id,
                ),
            )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
        conn.close()
        logger.info("Restored %d preferences from backup folder", count)
        return True

    if backup_zip:
        logger.info("Found backup zip: %s", backup_zip)
        from erised.backup import import_restore
        return import_restore(str(backup_zip), db_path=db_path, output_dir=output_dir)

    logger.error("No preference data found. Cannot train.")
    return False


def verify_data_integrity(db_path: str) -> int:
    """Check that all token files referenced in the DB exist."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT pair_id, winner_tokens_path, loser_tokens_path FROM preferences"
    ).fetchall()
    conn.close()

    missing = 0
    for pair_id, w_path, l_path in rows:
        if not Path(w_path).exists():
            logger.warning("Missing winner tokens for pair %s: %s", pair_id, w_path)
            missing += 1
        if not Path(l_path).exists():
            logger.warning("Missing loser tokens for pair %s: %s", pair_id, l_path)
            missing += 1

    if missing > 0:
        logger.error("%d token files are missing! Training may fail.", missing)
    else:
        logger.info("All token files verified (%d pairs)", len(rows))

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="DPO training for HeartMuLa")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (default: from ERISED_MODEL_PATH env)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to preferences DB (default: from config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory with token/audio files (default: from config)")
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/dpo_checkpoints",
                        help="Where to save DPO checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--global-weight", type=float, default=1.0,
                        help="Loss weight for global (backbone) component")
    parser.add_argument("--local-weight", type=float, default=0.5,
                        help="Loss weight for local (decoder) component")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint dir (default: auto-detect from checkpoint-dir/latest)")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Load model weights from this checkpoint dir WITHOUT restoring optimizer state (fresh training start)")
    args = parser.parse_args()

    start_time = time.time()

    # Load config
    from erised.config import ErisedConfig
    config = ErisedConfig.from_env()
    if args.model_path:
        config.model_path = args.model_path
    config.lazy_load = False  # Need model resident for training

    db_path = args.db_path or config.dpo_db_path
    output_dir = args.output_dir or config.output_dir

    # Step 1: Ensure preference data is available
    logger.info("=" * 60)
    logger.info("STEP 1: Checking preference data")
    logger.info("=" * 60)
    if not find_and_restore_backup(db_path, output_dir):
        sys.exit(1)

    n_pairs = verify_data_integrity(db_path)
    if n_pairs == 0:
        logger.error("No valid preference pairs. Exiting.")
        sys.exit(1)

    # Step 2: Load the model
    logger.info("=" * 60)
    logger.info("STEP 2: Loading HeartMuLa pipeline")
    logger.info("=" * 60)
    import torch

    from erised.pipeline import ErisedPipeline
    pipeline = ErisedPipeline(config)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info("VRAM after model load: %.2f GB", allocated)

    # Optionally load weights from a prior checkpoint without restoring optimizer state.
    # Use this to continue from an earlier DPO checkpoint with new hyperparameters (e.g. lower LR).
    if args.init_weights:
        import glob as _glob
        from safetensors.torch import load_file as _load_safetensors
        logger.info("Loading init weights from: %s", args.init_weights)
        sf_files = sorted(_glob.glob(os.path.join(args.init_weights, "*.safetensors")))
        if not sf_files:
            logger.error("No .safetensors files found in %s", args.init_weights)
            sys.exit(1)
        _device = next(pipeline.pipe.mula.parameters()).device
        _state_dict = {}
        for f in sf_files:
            _state_dict.update(_load_safetensors(f, device=str(_device)))
        missing, unexpected = pipeline.pipe.mula.load_state_dict(_state_dict, strict=False)
        logger.info("Init weights loaded: %d tensors, %d missing keys, %d unexpected keys",
                    len(_state_dict), len(missing), len(unexpected))

    # Step 3: Run DPO training
    logger.info("=" * 60)
    logger.info("STEP 3: Starting DPO training")
    logger.info("=" * 60)

    from erised.dpo.data import PreferenceStore
    from erised.dpo.trainer import DPOTrainer

    pref_store = PreferenceStore(db_path)

    trainer = DPOTrainer(
        pipeline=pipeline.pipe,
        preference_store=pref_store,
        beta=args.beta,
        learning_rate=args.lr,
        epochs=args.epochs,
        grad_accumulation_steps=args.grad_accum,
        checkpoint_dir=args.checkpoint_dir,
        global_loss_weight=args.global_weight,
        local_loss_weight=args.local_weight,
        warmup_steps=args.warmup_steps,
        use_amp=not args.no_amp,
    )
    trainer.train(resume_from=args.resume)

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TOTAL TIME: %.1f minutes", total_time / 60)
    logger.info("Checkpoints at: %s", args.checkpoint_dir)
    logger.info("Best model at: %s/dpo_best", args.checkpoint_dir)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Generate a song with the DPO-tuned model:")
    logger.info("     python -m erised generate --model-path %s/dpo_best --prompt '...' --lyrics '...'",
                args.checkpoint_dir)
    logger.info("  2. Compare with original model and collect more preferences")
    logger.info("  3. Run another DPO round for iterative improvement")


if __name__ == "__main__":
    main()
