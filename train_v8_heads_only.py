#!/usr/bin/env python3
"""
v8 training: Heads-only DPO (NO backbone layers trained).

Root cause of noise: DPO trains top 8/28 backbone layers, creating a mismatch
between frozen lower layers and tuned upper layers. During autoregressive generation,
this mismatch compounds over time → progressive noise buildup.

Fix: Train ONLY output heads (codebook0_head, audio_head, projection).
The backbone remains 100% original = zero internal inconsistency.
Output heads only steer which tokens get higher probability.

Starts from original base model (not v6).
"""
import os, sys, time, logging, copy, math, random
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
import torch.nn.functional as F
from torch.cuda.amp import autocast
from safetensors.torch import save_file as save_safetensors
from torch.utils.checkpoint import checkpoint as ckpt_fn

from erised.config import ErisedConfig
from erised.pipeline import ErisedPipeline
from erised.dpo.data import PreferenceStore
from erised.dpo.forward import compute_sequence_log_probs, build_training_sequence

# ── Hyperparameters ──────────────────────────────────────────────
BETA = 0.1
LR = 1e-5               # Higher LR since fewer params
EPOCHS = 6              # More epochs since heads-only learns slower
GRAD_ACCUM = 4
WARMUP_STEPS = 10
CHECKPOINT_DIR = "/workspace/dpo_checkpoints_v8"
SAVE_EVERY_N_STEPS = 80
MAX_FRAMES = 1500

start = time.time()

# ── Load pipeline ────────────────────────────────────────────────
config = ErisedConfig.from_env()
config.lazy_load = False
pipeline = ErisedPipeline(config)
pipe = pipeline.pipe  # HeartMuLaGenPipeline
policy_model = pipe.mula
device = next(policy_model.parameters()).device
dtype = next(policy_model.parameters()).dtype
logger.info("Base model loaded on %s", device)

# ── Move codec off GPU ──────────────────────────────────────────
if hasattr(pipe, 'codec') and pipe.codec is not None:
    pipe.codec.cpu()
    torch.cuda.empty_cache()
    logger.info("Moved codec to CPU")

# ── Freeze everything, unfreeze only output heads ────────────────
for p in policy_model.parameters():
    p.requires_grad = False

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
logger.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)
logger.info("BACKBONE LAYERS TRAINED: 0 — heads only!")

# ── Reference model (CPU-offloaded) ─────────────────────────────
logger.info("Creating reference model (deepcopy → CPU)...")
ref_model = copy.deepcopy(policy_model)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
ref_model = ref_model.cpu()
logger.info("Reference model on CPU")

# ── Gradient checkpointing ──────────────────────────────────────
if hasattr(policy_model, 'backbone') and hasattr(policy_model.backbone, 'layers'):
    for layer in policy_model.backbone.layers:
        orig_fwd = layer.forward
        def _make_ckpt_fwd(fn):
            def _fwd(*args, **kwargs):
                return ckpt_fn(fn, *args, use_reentrant=False, **kwargs)
            return _fwd
        layer.forward = _make_ckpt_fwd(orig_fwd)
    logger.info("Gradient checkpointing enabled on backbone (%d layers)", len(policy_model.backbone.layers))

# ── Reset KV caches ─────────────────────────────────────────────
try:
    if hasattr(policy_model, 'reset_caches'):
        policy_model.reset_caches()
except RuntimeError:
    pass
try:
    for model_part in (policy_model.backbone, getattr(policy_model, 'decoder', None)):
        if model_part is None:
            continue
        for layer in model_part.layers:
            attn = getattr(layer, 'attn', None)
            if attn and getattr(attn, 'kv_cache', None) is not None:
                attn.kv_cache = None
                attn.cache_enabled = False
except Exception:
    pass

# ── Load preferences ────────────────────────────────────────────
pref_store = PreferenceStore(os.environ["ERISED_DPO_DB"])
preferences = pref_store.get_all()
logger.info("Loaded %d preference pairs", len(preferences))
if not preferences:
    logger.error("No preferences found! Exiting.")
    sys.exit(1)

# ── Optimizer + scheduler ────────────────────────────────────────
policy_model.train()
trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

effective_steps = (len(preferences) * EPOCHS) // GRAD_ACCUM
def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(effective_steps - WARMUP_STEPS, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda', enabled=True)

# ── Checkpoint helper ────────────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_ckpt(name):
    ckpt_path = os.path.join(CHECKPOINT_DIR, name)
    os.makedirs(ckpt_path, exist_ok=True)
    state_dict = {k: v.cpu().clone() for k, v in policy_model.state_dict().items()}
    save_safetensors(state_dict, os.path.join(ckpt_path, "model.safetensors"))
    logger.info("Checkpoint saved to %s", ckpt_path)

# ── Training loop ────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("V8 HEADS-ONLY DPO TRAINING")
logger.info("Preferences: %d | Epochs: %d | LR: %s | Beta: %s", len(preferences), EPOCHS, LR, BETA)
logger.info("=" * 60)

global_step = 0
best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    epoch_losses = []
    optimizer.zero_grad()

    epoch_prefs = list(preferences)
    random.Random(epoch).shuffle(epoch_prefs)

    for step, pref in enumerate(epoch_prefs, 1):
        step_start = time.time()

        try:
            # Load token tensors from disk
            winner_frames = torch.load(pref.winner_tokens_path, weights_only=True)
            loser_frames = torch.load(pref.loser_tokens_path, weights_only=True)
            winner_frames = winner_frames[:, :MAX_FRAMES]
            loser_frames = loser_frames[:, :MAX_FRAMES]

            # Build training sequences (matches original trainer exactly)
            w_tokens, w_mask = build_training_sequence(
                pipe, pref.prompt, pref.lyrics, winner_frames.to(device),
            )
            l_tokens, l_mask = build_training_sequence(
                pipe, pref.prompt, pref.lyrics, loser_frames.to(device),
            )
            w_tokens, w_mask = w_tokens.to(device), w_mask.to(device)
            l_tokens, l_mask = l_tokens.to(device), l_mask.to(device)

        except Exception as e:
            logger.warning("Skip pair %s: %s", pref.pair_id, e)
            continue

        with torch.amp.autocast('cuda', enabled=True):
            # Reference model → GPU, compute, → CPU
            ref_model.to(device)
            with torch.no_grad():
                w_global_ref, w_local_ref = compute_sequence_log_probs(ref_model, w_tokens, w_mask)
                l_global_ref, l_local_ref = compute_sequence_log_probs(ref_model, l_tokens, l_mask)
            ref_model.cpu()
            torch.cuda.empty_cache()

            # Policy model (with grads on heads)
            w_global, w_local = compute_sequence_log_probs(policy_model, w_tokens, w_mask)
            l_global, l_local = compute_sequence_log_probs(policy_model, l_tokens, l_mask)

            # DPO loss — global only (no local/decoder loss)
            delta_global = (w_global - w_global_ref) - (l_global - l_global_ref)
            loss = -F.logsigmoid(BETA * delta_global).mean()
            loss_scaled = loss / GRAD_ACCUM

        scaler.scale(loss_scaled).backward()

        actual_loss = loss.item()
        epoch_losses.append(actual_loss)
        reward_margin = (BETA * delta_global).mean().item()
        acc = 100 if delta_global.mean().item() > 0 else 0

        if step % GRAD_ACCUM == 0 or step == len(epoch_prefs):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        step_time = time.time() - step_start
        logger.info(
            "Epoch %d/%d | Step %d/%d | Loss: %.6f | Δ_global: %.6f | "
            "Reward margin: %.6f | Acc: %d%% | LR: %.2e | Time: %.1fs",
            epoch, EPOCHS, step, len(epoch_prefs),
            actual_loss, delta_global.mean().item(),
            reward_margin, acc, optimizer.param_groups[0]['lr'], step_time
        )

        if step % SAVE_EVERY_N_STEPS == 0:
            save_ckpt(f"mid_epoch_{epoch}_step_{step}")
            logger.info("Mid-epoch checkpoint saved (epoch %d, step %d/%d)", epoch, step, len(epoch_prefs))

    # End of epoch
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
    logger.info("=" * 60)
    logger.info("EPOCH %d/%d COMPLETE | Avg loss: %.4f | Time: %.1fs",
                epoch, EPOCHS, avg_loss, time.time() - epoch_start)

    save_ckpt(f"dpo_epoch_{epoch}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        save_ckpt("dpo_best")
        logger.info("New best model saved to %s/dpo_best (loss=%.4f)", CHECKPOINT_DIR, best_loss)

    logger.info("=" * 60)

logger.info("=" * 60)
logger.info("DPO training complete in %.1f minutes.", (time.time() - start) / 60)
logger.info("Best loss: %.4f", best_loss)
logger.info("Checkpoints saved to: %s", CHECKPOINT_DIR)
