"""
DPO Logit Guidance with shared backbone optimization.

    logits_final = logits_orig + scale * (logits_dpo - logits_orig)

v11 DPO only trains backbone layers 26-27 (of 28) + output heads.
Layers 0-25 are identical between orig and DPO, so we run them once
and branch only at layer 26. This gives ~1.8x speedup over running
both full backbones independently.

Requires A100-80GB to fit both models in VRAM.
"""

import glob
import logging
import os
from copy import deepcopy
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .streaming import streaming_detokenize

logger = logging.getLogger(__name__)

# Number of backbone layers shared between orig and DPO models during
# guided inference.  v11 DPO only trains the top 2 layers (26-27 of 28),
# so layers 0-25 produce identical outputs and need only run once.
N_SHARED_LAYERS = 26


def _sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    logits = logits / temperature
    filter_value = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores = logits.masked_fill(indices_to_remove, filter_value)
    scores = F.log_softmax(scores, dim=-1)
    probs = F.softmax(scores, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def _index_causal_mask(mask, input_pos):
    return mask[input_pos, :]


def _reset_model_caches(model):
    for part in (model.backbone, model.decoder):
        try:
            part.reset_caches()
        except (RuntimeError, AttributeError):
            pass
        for layer in part.layers:
            attn = getattr(layer, "attn", None)
            if attn is not None and getattr(attn, "kv_cache", None) is not None:
                attn.kv_cache = None
                attn.cache_enabled = False


def _generate_frame_logits(model, tokens, tokens_mask, input_pos, cfg_scale,
                           continuous_segments=None, starts=None):
    """
    Run one frame through a model's backbone and return (c0_logits, last_h).
    Identical to v8's generate_frame_logits.
    """
    b, s, _ = tokens.size()
    backbone_mask = _index_causal_mask(model.backbone_causal_mask, input_pos)

    uncond_mask = None
    if cfg_scale > 1.0 and b > 1:
        actual_B = b // 2
        uncond_mask = torch.cat([
            torch.zeros(actual_B, dtype=torch.bool, device=tokens.device),
            torch.ones(actual_B, dtype=torch.bool, device=tokens.device),
        ])

    embeds = model._embed_tokens(tokens, uncond_mask=uncond_mask)
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2, dtype=embeds.dtype)

    if continuous_segments is not None:
        continuous_segments = model.muq_linear(continuous_segments)
        if uncond_mask is not None:
            uncond_embed = model.unconditional_text_embedding(
                torch.zeros(1, device=tokens.device, dtype=torch.long)
            )
            mask_expanded = uncond_mask.view(b, 1).expand_as(continuous_segments)
            continuous_segments = torch.where(mask_expanded, uncond_embed, continuous_segments)
        batch_indices = torch.arange(h.shape[0], device=h.device)
        h[batch_indices, starts] = continuous_segments

    h = model.backbone(h, input_pos=input_pos, mask=backbone_mask)
    last_h = h[:, -1, :]
    c0_logits = model.codebook0_head(last_h)
    return c0_logits, last_h


def _sync_kv_caches(src_layers, dst_layers, n_layers):
    """Copy KV cache state from src layers 0..n-1 to dst layers 0..n-1."""
    for i in range(n_layers):
        src_cache = src_layers[i].attn.kv_cache
        dst_cache = dst_layers[i].attn.kv_cache
        if src_cache is not None and dst_cache is not None:
            dst_cache.k_cache.copy_(src_cache.k_cache)
            dst_cache.v_cache.copy_(src_cache.v_cache)
            dst_cache.cache_pos.copy_(src_cache.cache_pos)


def _guided_generate_frame(orig_model, dpo_model, tokens, tokens_mask, input_pos,
                           temperature, topk, cfg_scale, dpo_scale,
                           continuous_segments=None, starts=None):
    """
    Generate one frame using logit guidance with shared backbone.

    Layers 0-25 are identical between orig and DPO (v11 only trains 26-27),
    so we run them once on the orig model and sync KV caches to the DPO model.
    Only layers 26-27 + output heads run on both models (~1.8x faster).
    """
    b = tokens.size(0)
    embeds_dtype = next(orig_model.parameters()).dtype

    # ── Prepare embeddings (identical weights, run once) ──
    b_size, s, _ = tokens.size()
    backbone_mask = _index_causal_mask(orig_model.backbone_causal_mask, input_pos)

    uncond_mask = None
    if cfg_scale > 1.0 and b > 1:
        actual_B = b // 2
        uncond_mask = torch.cat([
            torch.zeros(actual_B, dtype=torch.bool, device=tokens.device),
            torch.ones(actual_B, dtype=torch.bool, device=tokens.device),
        ])

    embeds = orig_model._embed_tokens(tokens, uncond_mask=uncond_mask)
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2, dtype=embeds.dtype)

    if continuous_segments is not None:
        continuous_segments = orig_model.muq_linear(continuous_segments)
        if uncond_mask is not None:
            uncond_embed = orig_model.unconditional_text_embedding(
                torch.zeros(1, device=tokens.device, dtype=torch.long)
            )
            mask_expanded = uncond_mask.view(b, 1).expand_as(continuous_segments)
            continuous_segments = torch.where(mask_expanded, uncond_embed, continuous_segments)
        batch_indices = torch.arange(h.shape[0], device=h.device)
        h[batch_indices, starts] = continuous_segments

    # ── Shared layers 0-25: run once on orig model ──
    for layer in orig_model.backbone.layers[:N_SHARED_LAYERS]:
        h = layer(h, input_pos=input_pos, mask=backbone_mask)

    # Sync KV caches for shared layers to DPO model
    _sync_kv_caches(orig_model.backbone.layers, dpo_model.backbone.layers, N_SHARED_LAYERS)

    # ── Branched layers 26-27: run on both models ──
    h_orig = h.clone()
    for layer in orig_model.backbone.layers[N_SHARED_LAYERS:]:
        h_orig = layer(h_orig, input_pos=input_pos, mask=backbone_mask)
    h_orig = orig_model.backbone.norm(h_orig)

    h_dpo = h
    for layer in dpo_model.backbone.layers[N_SHARED_LAYERS:]:
        h_dpo = layer(h_dpo, input_pos=input_pos, mask=backbone_mask)
    h_dpo = dpo_model.backbone.norm(h_dpo)

    # ── Get logits from both heads ──
    orig_last_h = h_orig[:, -1, :]
    dpo_last_h = h_dpo[:, -1, :]
    orig_c0_logits = orig_model.codebook0_head(orig_last_h)
    dpo_c0_logits = dpo_model.codebook0_head(dpo_last_h)

    # ── Combine logits with CFG + DPO guidance ──
    if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
        actual_B = b // 2
        orig_cond = orig_c0_logits[:actual_B]
        orig_uncond = orig_c0_logits[actual_B:]
        orig_guided = orig_uncond + (orig_cond - orig_uncond) * cfg_scale

        dpo_cond = dpo_c0_logits[:actual_B]
        dpo_uncond = dpo_c0_logits[actual_B:]
        dpo_guided = dpo_uncond + (dpo_cond - dpo_uncond) * cfg_scale

        final_logits = orig_guided + dpo_scale * (dpo_guided - orig_guided)
        c0_sample = _sample_topk(final_logits, topk, temperature)
        c0_sample = c0_sample.repeat(2, 1)
    else:
        final_logits = orig_c0_logits + dpo_scale * (dpo_c0_logits - orig_c0_logits)
        c0_sample = _sample_topk(final_logits, topk, temperature)

    # ── Decoder: codebooks 1-7 (original model only, same as v8) ──
    orig_model.decoder.reset_caches()
    c0_embed = orig_model._embed_audio(0, c0_sample)
    curr_h = torch.cat([orig_last_h.unsqueeze(1), c0_embed], dim=1)
    curr_sample = c0_sample.clone()
    curr_pos = (
        torch.arange(0, curr_h.size(1), device=curr_h.device)
        .unsqueeze(0).repeat(curr_h.size(0), 1)
    )
    curr_h = curr_h.to(embeds_dtype)

    for i in range(1, orig_model.config.audio_num_codebooks):
        curr_decoder_mask = _index_causal_mask(orig_model.decoder_causal_mask, curr_pos)
        decoder_h = orig_model.decoder(
            orig_model.projection(curr_h),
            input_pos=curr_pos,
            mask=curr_decoder_mask,
        )
        ci_logits = torch.mm(decoder_h[:, -1, :], orig_model.audio_head[i - 1])

        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            cond_ci = ci_logits[:actual_B]
            uncond_ci = ci_logits[actual_B:]
            guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale
            ci_sample = _sample_topk(guided_ci, topk, temperature)
            ci_sample = ci_sample.repeat(2, 1)
        else:
            ci_sample = _sample_topk(ci_logits, topk, temperature)

        ci_embed = orig_model._embed_audio(i, ci_sample)
        curr_h = ci_embed
        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
        curr_pos = curr_pos[:, -1:] + 1

    return curr_sample


class DPOGuider:
    """
    Exact v8 approach: two full model copies, each with independent KV caches.
    Requires A100-80GB.
    """

    def __init__(self, orig_model, dpo_checkpoint_path: str):
        self.orig_model = orig_model
        self.device = next(orig_model.parameters()).device
        self.dtype = next(orig_model.parameters()).dtype

        # Full deep copy of the model (same as v8)
        logger.info("Creating full DPO model copy (deepcopy)...")
        self.dpo_model = deepcopy(orig_model)
        self.dpo_model.to(self.device)

        # Load ALL DPO checkpoint weights (same as v8)
        logger.info("Loading DPO weights from %s ...", dpo_checkpoint_path)
        from safetensors.torch import load_file
        files = sorted(glob.glob(os.path.join(dpo_checkpoint_path, "*.safetensors")))
        if not files:
            files = sorted(glob.glob(os.path.join(dpo_checkpoint_path, "*", "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors in {dpo_checkpoint_path}")

        full_state = {}
        for f in files:
            full_state.update(load_file(f, device=str(self.device)))
        self.dpo_model.load_state_dict(full_state, strict=False)
        logger.info("Loaded %d DPO tensors", len(full_state))
        del full_state

        self.dpo_model.eval()
        logger.info(
            "DPOGuider ready (full copy). VRAM: %.2f GB",
            torch.cuda.memory_allocated(self.device) / 1024**3,
        )

    def generate(
        self,
        pipeline,
        tags: str,
        lyrics: str,
        save_path: str,
        *,
        max_audio_length_ms: int = 60_000,
        temperature: float = 1.0,
        topk: int = 50,
        cfg_scale: float = 1.5,
        dpo_scale: float = 1.0,
        on_progress: Optional[Callable] = None,
        on_frames_checkpoint: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Full guided generation loop — identical to v8's guided_forward."""
        pipe = pipeline.pipe
        orig_model = self.orig_model
        dpo_model = self.dpo_model
        device = self.device

        model_inputs = pipe.preprocess(
            {"tags": tags, "lyrics": lyrics},
            cfg_scale=cfg_scale,
        )

        prompt_tokens = model_inputs["tokens"].to(device)
        prompt_tokens_mask = model_inputs["tokens_mask"].to(device)
        continuous_segment = model_inputs["muq_embed"].to(device)
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"].to(device)
        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1

        # Reset and setup caches for BOTH models (same as v8)
        _reset_model_caches(orig_model)
        _reset_model_caches(dpo_model)
        torch.cuda.empty_cache()

        orig_model.setup_caches(bs_size)
        dpo_model.setup_caches(bs_size)

        parallel_number = 8 + 1
        empty_id = pipe.config.empty_id

        def _pad(token):
            padded = (
                torch.ones((token.shape[0], parallel_number), device=device, dtype=torch.long)
                * empty_id
            )
            padded[:, :-1] = token
            padded = padded.unsqueeze(1)
            mask = torch.ones_like(padded, dtype=torch.bool)
            mask[..., -1] = False
            return padded, mask

        max_audio_frames = max_audio_length_ms // 80

        # Frame checkpoint thresholds for streaming decode on separate GPU.
        # First chunk at 125 frames (~10s audio) — codec pads to 372 internally.
        # This gets first audio to user in ~35s (shared backbone ~7.5fps + decode).
        _FIRST_CHUNK = 125
        _HOP = int(29.76 * 12.5) // 93 * 80  # 320 (codec's natural hop)
        next_checkpoint = _FIRST_CHUNK if on_frames_checkpoint else None

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=self.dtype):
            # Initial frame
            curr_token = _guided_generate_frame(
                orig_model, dpo_model,
                prompt_tokens, prompt_tokens_mask, prompt_pos,
                temperature, topk, cfg_scale, dpo_scale,
                continuous_segment, starts,
            )
            frames.append(curr_token[0:1,])

            # Autoregressive loop — NO codec decode here (causes SIGABRT)
            for i in tqdm(range(max_audio_frames), desc="Guided generation"):
                curr_padded, curr_mask = _pad(curr_token)
                curr_token = _guided_generate_frame(
                    orig_model, dpo_model,
                    curr_padded, curr_mask,
                    prompt_pos[..., -1:] + i + 1,
                    temperature, topk, cfg_scale, dpo_scale,
                )
                if torch.any(curr_token[0:1, :] >= pipe.config.audio_eos_id):
                    break
                frames.append(curr_token[0:1,])

                if len(frames) % 10 == 0 and on_progress:
                    on_progress(len(frames), max_audio_frames, None, None)

                # Send frames to external codec worker at checkpoint thresholds
                if next_checkpoint and len(frames) >= next_checkpoint:
                    frames_tensor_cp = torch.stack(frames).permute(1, 2, 0).squeeze(0)
                    on_frames_checkpoint(frames_tensor_cp, is_final=False)
                    next_checkpoint += _HOP

        # Stack frames and decode to audio
        frames_tensor = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        num_gen_frames = len(frames)

        if on_frames_checkpoint:
            # External decoder handles all audio — send final frames
            on_frames_checkpoint(frames_tensor, is_final=True)
            logger.info("Guided frames sent to external decoder (%d frames)", frames_tensor.shape[-1])
        else:
            # No external decoder — decode locally (original path)
            def _on_chunk(chunk_idx, total_chunks):
                if on_progress:
                    on_progress(
                        num_gen_frames, max_audio_frames,
                        os.path.basename(save_path), chunk_idx,
                    )

            streaming_detokenize(
                pipe.codec, frames_tensor, save_path,
                on_chunk_ready=_on_chunk,
            )
            logger.info("Guided audio saved to %s (%d frames)", save_path, frames_tensor.shape[-1])

        return frames_tensor
