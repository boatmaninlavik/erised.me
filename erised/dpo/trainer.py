"""
DPO (Direct Preference Optimization) trainer for HeartMuLa.

Implements the DPO loss from the HeartMuLa paper (Section 3.3.2, Eq. 11–13):

    L_DPO = -E[ log σ( β · Δ ) ]

where Δ = (log p_θ(A_w|C) - log p_ref(A_w|C)) - (log p_θ(A_l|C) - log p_ref(A_l|C))

The log-probability is decomposed into global (layer-0) and local (layers 1–7)
components per the hierarchical architecture.
"""

import os
import copy
import json
import random
import logging
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from heartlib.heartmula.modeling_heartmula import HeartMuLa
from heartlib.pipelines.music_generation import HeartMuLaGenPipeline

from .data import PreferenceStore
from .forward import compute_sequence_log_probs, build_training_sequence

logger = logging.getLogger(__name__)


class DPOTrainer:
    def __init__(
        self,
        pipeline: HeartMuLaGenPipeline,
        preference_store: PreferenceStore,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        epochs: int = 10,
        grad_accumulation_steps: int = 4,
        checkpoint_dir: str = "./dpo_checkpoints",
        global_loss_weight: float = 1.0,
        local_loss_weight: float = 0.5,
        warmup_steps: int = 10,
        use_amp: bool = True,
    ):
        self.pipeline = pipeline
        self.pref_store = preference_store
        self.beta = beta
        self.lr = learning_rate
        self.epochs = epochs
        self.grad_accumulation_steps = grad_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.global_weight = global_loss_weight
        self.local_weight = local_loss_weight
        self.warmup_steps = warmup_steps
        self.use_amp = use_amp
        self.save_every_n_steps = 50  # save mid-epoch every N steps
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _save_training_state(self, optimizer, scheduler, scaler, epoch, step, global_step, best_loss, path):
        """Save full training state for resumption."""
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "best_loss": best_loss,
        }
        torch.save(state, os.path.join(path, "training_state.pt"))

    def _load_training_state(self, optimizer, scheduler, scaler, path):
        """Load training state if a checkpoint exists. Returns (epoch, step, global_step, best_loss) or None."""
        state_path = os.path.join(path, "training_state.pt")
        if not os.path.exists(state_path):
            return None
        state = torch.load(state_path, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        return state["epoch"], state["step"], state["global_step"], state["best_loss"]

    def train(self, resume_from: Optional[str] = None):
        """Run the full DPO training loop."""

        preferences = self.pref_store.get_all()
        if not preferences:
            logger.error("No preference data found. Generate pairs and collect ratings first.")
            return

        logger.info("=" * 60)
        logger.info("DPO TRAINING")
        logger.info("=" * 60)
        logger.info("Preference pairs: %d", len(preferences))
        logger.info("Epochs: %d", self.epochs)
        logger.info("Learning rate: %s", self.lr)
        logger.info("Beta: %s", self.beta)
        logger.info("Grad accumulation steps: %d", self.grad_accumulation_steps)
        logger.info("Global loss weight: %s", self.global_weight)
        logger.info("Local loss weight: %s", self.local_weight)
        logger.info("AMP (mixed precision): %s", self.use_amp)
        logger.info("=" * 60)

        policy_model: HeartMuLa = self.pipeline.mula
        device = next(policy_model.parameters()).device
        dtype = next(policy_model.parameters()).dtype

        # Move codec off GPU — not needed during DPO training
        if hasattr(self.pipeline, 'codec') and self.pipeline.codec is not None:
            self.pipeline.codec.cpu()
            torch.cuda.empty_cache()
            logger.info("Moved codec to CPU to free VRAM")

        # Freeze most of the model — only train top 8 backbone layers + output heads
        # DPO is a subtle alignment, no need to update all 28 layers
        for p in policy_model.parameters():
            p.requires_grad = False
        # Unfreeze top 8 backbone layers
        n_layers = len(policy_model.backbone.layers)
        n_trainable = min(8, n_layers)
        for layer in policy_model.backbone.layers[n_layers - n_trainable:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Unfreeze output heads and projection
        if hasattr(policy_model, 'codebook0_head'):
            for p in policy_model.codebook0_head.parameters():
                p.requires_grad = True
        if hasattr(policy_model, 'audio_head'):
            policy_model.audio_head.requires_grad = True
        if hasattr(policy_model, 'projection'):
            for p in policy_model.projection.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy_model.parameters())
        logger.info("Trainable params: %d / %d (%.1f%%)", trainable, total, 100 * trainable / total)

        # Freeze a reference copy of the model — keep on CPU to save VRAM
        logger.info("Creating reference model (deepcopy, CPU-offloaded)...")
        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model = ref_model.cpu()
        logger.info("Reference model created on CPU. Policy device: %s, Dtype: %s", device, dtype)

        # Log VRAM usage
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            logger.info("VRAM after ref model (CPU): %.2f GB allocated, %.2f GB reserved", allocated, reserved)

        policy_model.train()

        # Enable gradient checkpointing on backbone layers to save VRAM
        # Trades ~2x compute for ~60% less activation memory
        from torch.utils.checkpoint import checkpoint as ckpt_fn
        if hasattr(policy_model, 'backbone') and hasattr(policy_model.backbone, 'layers'):
            for layer in policy_model.backbone.layers:
                orig_fwd = layer.forward
                def _make_ckpt_fwd(fn):
                    def _fwd(*args, **kwargs):
                        return ckpt_fn(fn, *args, use_reentrant=False, **kwargs)
                    return _fwd
                layer.forward = _make_ckpt_fwd(orig_fwd)
            logger.info("Gradient checkpointing enabled on backbone (%d layers)", len(policy_model.backbone.layers))

        # Reset KV caches if present (not needed for teacher-forced training)
        try:
            if hasattr(policy_model, 'reset_caches'):
                policy_model.reset_caches()
        except RuntimeError:
            pass  # Caches were never set up — that's fine for training
        try:
            if hasattr(ref_model, 'reset_caches'):
                ref_model.reset_caches()
        except RuntimeError:
            pass

        # fp32 master weights on CPU — bf16 precision (~0.8% relative) swallows
        # tiny DPO gradient updates entirely.  We keep the model in bf16 for
        # fast forward/backward, accumulate into fp32 copies on CPU, then copy
        # the updated values back.  Bonus: optimizer states on CPU frees ~8 GB VRAM.
        _trainable_bf16 = [p for p in policy_model.parameters() if p.requires_grad]
        _fp32_masters = [torch.nn.Parameter(p.data.float().cpu().clone()) for p in _trainable_bf16]
        optimizer = AdamW(_fp32_masters, lr=self.lr, weight_decay=0.01)
        logger.info("fp32 master weights on CPU (%d tensors, %.1f M params)",
                     len(_fp32_masters), sum(p.numel() for p in _fp32_masters) / 1e6)

        # Cosine annealing with warmup
        total_steps = len(preferences) * self.epochs
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(self.warmup_steps, 1)
            progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
            return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # GradScaler only works with float16, not bfloat16 (which has full dynamic range)
        use_scaler = self.use_amp and device.type == "cuda" and dtype != torch.bfloat16
        scaler = GradScaler("cuda", enabled=use_scaler)

        def _optimizer_step():
            """Sync bf16 GPU grads → fp32 CPU, step optimizer, copy fp32 → bf16 model."""
            for bp, fp in zip(_trainable_bf16, _fp32_masters):
                fp.grad = bp.grad.float().cpu() if bp.grad is not None else None
            torch.nn.utils.clip_grad_norm_(_fp32_masters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Copy updated fp32 weights back to bf16 model on GPU
            for bp, fp in zip(_trainable_bf16, _fp32_masters):
                bp.data.copy_(fp.data.to(bp.device, dtype=bp.dtype))
                bp.grad = None

        global_step = 0
        best_loss = float("inf")
        start_epoch = 0
        start_step = 0
        training_start = time.time()

        # Try to resume from checkpoint
        resume_dir = resume_from or os.path.join(self.checkpoint_dir, "latest")
        if os.path.exists(os.path.join(resume_dir, "training_state.pt")):
            logger.info("Resuming from checkpoint: %s", resume_dir)
            # Sharded safetensors checkpoints must be reassembled from every
            # shard listed in the index file, not a single arbitrary file.
            index_path = os.path.join(resume_dir, "model.safetensors.index.json")
            if os.path.exists(index_path):
                from safetensors.torch import load_file
                import json as _json
                with open(index_path) as _f:
                    _index = _json.load(_f)
                _shard_files = sorted(set(_index["weight_map"].values()))
                state_dict = {}
                for _shard in _shard_files:
                    state_dict.update(load_file(os.path.join(resume_dir, _shard), device=str(device)))
                logger.info("Loaded %d tensors from %d safetensors shards", len(state_dict), len(_shard_files))
                policy_model.load_state_dict(state_dict, strict=False)
            else:
                ckpt_files = [f for f in os.listdir(resume_dir) if f.endswith(('.bin', '.safetensors', '.pt')) and f != 'training_state.pt']
                if ckpt_files:
                    ckpt_file = os.path.join(resume_dir, ckpt_files[0])
                    if ckpt_file.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(ckpt_file, device=str(device))
                    else:
                        state_dict = torch.load(ckpt_file, weights_only=True, map_location=device)
                    policy_model.load_state_dict(state_dict, strict=False)
            # Refresh fp32 masters from the loaded policy — otherwise the next
            # _optimizer_step copies the pre-resume (base) master weights back
            # onto the bf16 model and silently wipes out the checkpoint.
            for _bp, _fp in zip(_trainable_bf16, _fp32_masters):
                _fp.data.copy_(_bp.data.float().cpu())
            resumed = self._load_training_state(optimizer, scheduler, scaler, resume_dir)
            if resumed:
                start_epoch, start_step, global_step, best_loss = resumed
                logger.info("Resumed at epoch %d, step %d, global_step %d, best_loss %.4f",
                           start_epoch + 1, start_step, global_step, best_loss)

        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()
            total_loss = 0.0
            epoch_losses = []
            optimizer.zero_grad()
            for p in _trainable_bf16:
                p.grad = None

            # Shuffle preferences each epoch (use epoch as seed for reproducibility on resume)
            epoch_prefs = list(preferences)
            rng = random.Random(epoch)
            rng.shuffle(epoch_prefs)

            # Skip already-completed steps when resuming mid-epoch
            resume_step = start_step if epoch == start_epoch else 0

            for step, pref in enumerate(epoch_prefs):
                if step < resume_step:
                    continue
                step_start = time.time()

                # Load token tensors
                winner_frames = torch.load(pref.winner_tokens_path, weights_only=True)
                loser_frames = torch.load(pref.loser_tokens_path, weights_only=True)
                # frames: [K, num_audio_frames] — truncate long sequences to cap
                # attention matrix memory (quadratic in sequence length)
                MAX_FRAMES = 1500
                winner_frames = winner_frames[:, :MAX_FRAMES]
                loser_frames = loser_frames[:, :MAX_FRAMES]

                tags_from_prompt = pref.prompt
                lyrics = pref.lyrics

                # Build full training sequences
                w_tokens, w_mask = build_training_sequence(
                    self.pipeline, tags_from_prompt, lyrics, winner_frames.to(device),
                )
                l_tokens, l_mask = build_training_sequence(
                    self.pipeline, tags_from_prompt, lyrics, loser_frames.to(device),
                )

                w_tokens = w_tokens.to(device)
                w_mask = w_mask.to(device)
                l_tokens = l_tokens.to(device)
                l_mask = l_mask.to(device)

                with autocast("cuda", enabled=self.use_amp and device.type == "cuda"):
                    # Reference model: move to GPU, compute, move back to CPU
                    ref_model.to(device)
                    with torch.no_grad():
                        w_global_ref, w_local_ref = compute_sequence_log_probs(ref_model, w_tokens, w_mask)
                        l_global_ref, l_local_ref = compute_sequence_log_probs(ref_model, l_tokens, l_mask)
                    ref_model.cpu()
                    torch.cuda.empty_cache()

                    # Policy model log-probs (with gradient checkpointing, full VRAM available)
                    w_global, w_local = compute_sequence_log_probs(policy_model, w_tokens, w_mask)
                    l_global, l_local = compute_sequence_log_probs(policy_model, l_tokens, l_mask)

                    # DPO loss (decomposed into global + local per paper Eq. 13)
                    delta_global = (w_global - w_global_ref) - (l_global - l_global_ref)
                    delta_local = (w_local - w_local_ref) - (l_local - l_local_ref)

                    delta = (
                        self.global_weight * delta_global
                        + self.local_weight * delta_local
                    )

                    loss = -F.logsigmoid(self.beta * delta).mean()
                    loss_scaled = loss / self.grad_accumulation_steps

                scaler.scale(loss_scaled).backward()

                step_loss = loss.item()
                total_loss += step_loss
                epoch_losses.append(step_loss)

                # Compute reward margin for logging
                with torch.no_grad():
                    reward_margin = (self.beta * delta).item() if delta.dim() == 0 else (self.beta * delta).mean().item()
                    accuracy = 1.0 if reward_margin > 0 else 0.0

                step_time = time.time() - step_start

                # Log every step
                logger.info(
                    "Epoch %d/%d | Step %d/%d | Loss: %.6f | Δ_global: %.6f | Δ_local: %.6f | "
                    "Reward margin: %.6f | Acc: %.0f%% | LR: %.2e | Time: %.1fs",
                    epoch + 1, self.epochs, step + 1, len(epoch_prefs),
                    step_loss, delta_global.item(), delta_local.item(),
                    reward_margin, accuracy * 100,
                    scheduler.get_last_lr()[0],
                    step_time,
                )

                if (step + 1) % self.grad_accumulation_steps == 0:
                    _optimizer_step()
                    global_step += 1

                # Mid-epoch checkpoint
                if (step + 1) % self.save_every_n_steps == 0:
                    latest_path = os.path.join(self.checkpoint_dir, "latest")
                    os.makedirs(latest_path, exist_ok=True)
                    policy_model.save_pretrained(latest_path)
                    self._save_training_state(optimizer, scheduler, scaler, epoch, step + 1, global_step, best_loss, latest_path)
                    logger.info("Mid-epoch checkpoint saved (epoch %d, step %d/%d)", epoch + 1, step + 1, len(epoch_prefs))

            # Handle remaining accumulated gradients
            if len(epoch_prefs) % self.grad_accumulation_steps != 0:
                _optimizer_step()
                global_step += 1

            avg_loss = total_loss / len(epoch_prefs)
            epoch_time = time.time() - epoch_start

            logger.info("=" * 60)
            logger.info(
                "EPOCH %d/%d COMPLETE | Avg loss: %.4f | Time: %.1fs",
                epoch + 1, self.epochs, avg_loss, epoch_time,
            )

            # Save epoch checkpoint
            ckpt_path = os.path.join(self.checkpoint_dir, f"dpo_epoch_{epoch + 1}")
            os.makedirs(ckpt_path, exist_ok=True)
            policy_model.save_pretrained(ckpt_path)
            self._save_training_state(optimizer, scheduler, scaler, epoch + 1, 0, global_step, best_loss, ckpt_path)
            logger.info("Checkpoint saved to %s", ckpt_path)

            # Also save as "latest" for easy resume
            latest_path = os.path.join(self.checkpoint_dir, "latest")
            os.makedirs(latest_path, exist_ok=True)
            policy_model.save_pretrained(latest_path)
            self._save_training_state(optimizer, scheduler, scaler, epoch + 1, 0, global_step, best_loss, latest_path)

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(self.checkpoint_dir, "dpo_best")
                policy_model.save_pretrained(best_path)
                logger.info("New best model saved to %s (loss=%.4f)", best_path, best_loss)

            logger.info("=" * 60)

        total_time = time.time() - training_start
        logger.info("DPO training complete in %.1f minutes.", total_time / 60)
        logger.info("Best loss: %.4f", best_loss)
        logger.info("Checkpoints saved to: %s", self.checkpoint_dir)

        del ref_model
        torch.cuda.empty_cache()
