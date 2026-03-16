"""
DPO Logit Guidance with Shared-Layer Optimization.

Instead of baking DPO preferences into the model weights (which causes noise
accumulation during autoregressive generation), we apply DPO preferences at
the logit level:

    logits_final = logits_orig + scale * (logits_dpo - logits_orig)

The original model drives generation (staying internally consistent = no noise),
while the DPO model's learned preferences steer token selection.

Shared-Layer Optimization:
    Since DPO only trained the top N backbone layers (e.g., 2 out of 28),
    the bottom layers are identical between original and DPO models. We run
    those shared layers ONCE, then branch for the top layers only. This reduces
    overhead from ~100% to ~7% (for 2/28 layers).

Usage:
    from erised.guided_generate import DPOGuider

    guider = DPOGuider(
        orig_model=pipeline.pipe.mula,
        dpo_checkpoint_path="/path/to/dpo_best",
        n_dpo_layers=2,  # how many top backbone layers were DPO-trained
    )
    frames = guider.generate(pipeline, tags, lyrics, dpo_scale=1.0, **kwargs)
"""

import glob
import logging
import os
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    """Sample from logits with top-k filtering and temperature scaling."""
    logits = logits / temperature
    filter_value = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores = logits.masked_fill(indices_to_remove, filter_value)
    scores = F.log_softmax(scores, dim=-1)
    probs = F.softmax(scores, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def _load_safetensors_sharded(model, model_path, device="cuda"):
    """Load sharded safetensors checkpoint into a model."""
    from safetensors.torch import load_file
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        files = sorted(glob.glob(os.path.join(model_path, "*", "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors in {model_path}")
    state_dict = {}
    for f in files:
        state_dict.update(load_file(f, device=str(device)))
    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded %d tensors from %s", len(state_dict), model_path)


def _reset_model_caches(model):
    """Fully reset KV caches on a HeartMuLa model's backbone and decoder."""
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


class DPOGuider:
    """
    Applies DPO preferences at inference time via logit-level guidance,
    using a shared-layer optimization to avoid redundant computation.

    Only the top N backbone layers (the ones actually trained by DPO) and the
    codebook0_head are duplicated. The rest of the model is shared.
    """

    def __init__(
        self,
        orig_model,
        dpo_checkpoint_path: str,
        n_dpo_layers: int = 2,
    ):
        """
        Args:
            orig_model: The original HeartMuLa model (already loaded).
            dpo_checkpoint_path: Path to the DPO checkpoint directory.
            n_dpo_layers: Number of top backbone layers that were DPO-trained.
        """
        self.orig_model = orig_model
        self.n_dpo_layers = n_dpo_layers
        self.device = next(orig_model.parameters()).device
        self.dtype = next(orig_model.parameters()).dtype

        n_layers = len(orig_model.backbone.layers)
        self.n_shared = n_layers - n_dpo_layers
        logger.info(
            "DPOGuider: %d total layers, %d shared, %d DPO-branched",
            n_layers, self.n_shared, n_dpo_layers,
        )

        # Deep copy only the top N layers and codebook0_head from the original,
        # then overwrite with DPO weights directly (no full model copy needed).
        self.dpo_top_layers = deepcopy(
            orig_model.backbone.layers[self.n_shared:]
        )
        self.dpo_codebook0_head = deepcopy(orig_model.codebook0_head)
        # Also copy the backbone norm (applied after all layers)
        self.dpo_backbone_norm = deepcopy(orig_model.backbone.norm)

        # Load DPO checkpoint state dict to CPU, then extract only the weights
        # we need for the top layers, codebook0_head, and backbone norm.
        # This avoids creating a full model copy which would OOM on A100-40GB.
        logger.info("Loading DPO weights from %s ...", dpo_checkpoint_path)
        from safetensors.torch import load_file
        files = sorted(glob.glob(os.path.join(dpo_checkpoint_path, "*.safetensors")))
        if not files:
            files = sorted(glob.glob(os.path.join(dpo_checkpoint_path, "*", "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors in {dpo_checkpoint_path}")

        full_state = {}
        for f in files:
            full_state.update(load_file(f, device="cpu"))
        logger.info("Loaded %d tensors from checkpoint", len(full_state))

        # Map checkpoint keys to our small copies.
        # Checkpoint keys look like: backbone.layers.26.xxx, backbone.norm.xxx, codebook0_head.xxx
        # For top layers, we need backbone.layers[n_shared:] remapped to indices 0..n_dpo_layers-1
        top_layer_state = {}
        for key, val in full_state.items():
            if key.startswith("backbone.layers."):
                parts = key.split(".", 3)  # ['backbone', 'layers', '<idx>', '<rest>']
                layer_idx = int(parts[2])
                if layer_idx >= self.n_shared:
                    new_idx = layer_idx - self.n_shared
                    new_key = f"{new_idx}.{parts[3]}"
                    top_layer_state[new_key] = val

        norm_state = {}
        for key, val in full_state.items():
            if key.startswith("backbone.norm."):
                new_key = key[len("backbone.norm."):]
                norm_state[new_key] = val

        head_state = {}
        for key, val in full_state.items():
            if key.startswith("codebook0_head."):
                new_key = key[len("codebook0_head."):]
                head_state[new_key] = val

        if top_layer_state:
            self.dpo_top_layers.load_state_dict(top_layer_state, strict=False)
            logger.info("Loaded %d tensors into DPO top layers", len(top_layer_state))
        if norm_state:
            self.dpo_backbone_norm.load_state_dict(norm_state, strict=False)
            logger.info("Loaded %d tensors into DPO backbone norm", len(norm_state))
        if head_state:
            self.dpo_codebook0_head.load_state_dict(head_state, strict=False)
            logger.info("Loaded %d tensors into DPO codebook0 head", len(head_state))

        del full_state, top_layer_state, norm_state, head_state

        # Move to device
        self.dpo_top_layers.to(self.device)
        self.dpo_codebook0_head.to(self.device)
        self.dpo_backbone_norm.to(self.device)

        logger.info(
            "DPOGuider ready. VRAM: %.2f GB",
            torch.cuda.memory_allocated(self.device) / 1024**3,
        )

    def _setup_dpo_caches(self, bs_size: int):
        """Set up KV caches for the DPO branch layers.

        Must pass encoder_max_seq_len and decoder_max_seq_len to match
        how torchtune's TransformerDecoder.setup_caches() calls each layer.
        """
        for layer in self.dpo_top_layers:
            attn = getattr(layer, "attn", None)
            if attn is not None:
                if getattr(attn, "kv_cache", None) is not None:
                    attn.kv_cache = None
                    attn.cache_enabled = False

        # Get the cache seq len from the backbone (set during model.setup_caches)
        backbone = self.orig_model.backbone
        max_seq = getattr(backbone, "decoder_max_cache_seq_len",
                          getattr(backbone, "max_seq_len", 4096))

        with torch.device(self.device):
            for layer in self.dpo_top_layers:
                layer.setup_caches(
                    bs_size, self.dtype,
                    encoder_max_seq_len=max_seq,
                    decoder_max_seq_len=max_seq,
                )

    def _reset_dpo_caches(self):
        """Reset KV caches on DPO branch layers."""
        for layer in self.dpo_top_layers:
            attn = getattr(layer, "attn", None)
            if attn is not None:
                if getattr(attn, "kv_cache", None) is not None:
                    attn.kv_cache = None
                    attn.cache_enabled = False

    def _guided_generate_frame(
        self,
        tokens, tokens_mask, input_pos,
        temperature, topk, cfg_scale, dpo_scale,
        continuous_segments=None, starts=None,
    ):
        """
        Generate one audio frame using DPO logit guidance with shared-layer optimization.

        1. Embed tokens (shared)
        2. Run shared backbone layers 0..N-1 (shared, once)
        3. Branch: run top layers with original weights -> orig logits
        4. Branch: run top layers with DPO weights -> dpo logits
        5. Combine: final = orig + scale * (dpo - orig)
        6. Sample from combined logits
        7. Run decoder for codebooks 1-7 (original model, no cross-frame compounding)
        """
        model = self.orig_model
        b, s, _ = tokens.size()

        # ── Causal mask ──
        backbone_mask = model.backbone_causal_mask[input_pos, :]

        # ── Unconditional mask for CFG ──
        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            uncond_mask = torch.cat([
                torch.zeros(actual_B, dtype=torch.bool, device=tokens.device),
                torch.ones(actual_B, dtype=torch.bool, device=tokens.device),
            ])

        # ── Embed tokens (shared — embeddings are not DPO-trained) ──
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
                continuous_segments = torch.where(
                    mask_expanded, uncond_embed, continuous_segments
                )
            batch_indices = torch.arange(h.shape[0], device=h.device)
            h[batch_indices, starts] = continuous_segments

        # ── Shared backbone layers (run once) ──
        for layer in model.backbone.layers[:self.n_shared]:
            h = layer(h, mask=backbone_mask, input_pos=input_pos)

        # Save shared hidden state for DPO branch
        shared_h = h.clone()

        # ── Original branch: top layers + norm + head ──
        orig_h = h
        for layer in model.backbone.layers[self.n_shared:]:
            orig_h = layer(orig_h, mask=backbone_mask, input_pos=input_pos)
        orig_h = model.backbone.norm(orig_h)
        orig_last_h = orig_h[:, -1, :]
        orig_c0_logits = model.codebook0_head(orig_last_h)

        # ── DPO branch: top layers + norm + head ──
        dpo_h = shared_h
        for layer in self.dpo_top_layers:
            dpo_h = layer(dpo_h, mask=backbone_mask, input_pos=input_pos)
        dpo_h = self.dpo_backbone_norm(dpo_h)
        dpo_last_h = dpo_h[:, -1, :]
        dpo_c0_logits = self.dpo_codebook0_head(dpo_last_h)

        # ── Combine logits: CFG first, then DPO guidance ──
        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            # Apply CFG to both branches
            orig_guided = (
                orig_c0_logits[actual_B:]
                + (orig_c0_logits[:actual_B] - orig_c0_logits[actual_B:]) * cfg_scale
            )
            dpo_guided = (
                dpo_c0_logits[actual_B:]
                + (dpo_c0_logits[:actual_B] - dpo_c0_logits[actual_B:]) * cfg_scale
            )
            # DPO logit guidance
            final_logits = orig_guided + dpo_scale * (dpo_guided - orig_guided)
            c0_sample = _sample_topk(final_logits, topk, temperature)
            c0_sample = c0_sample.repeat(2, 1)
        else:
            final_logits = orig_c0_logits + dpo_scale * (dpo_c0_logits - orig_c0_logits)
            c0_sample = _sample_topk(final_logits, topk, temperature)

        # ── Decoder: codebooks 1-7 (original model, no cross-frame issue) ──
        model.decoder.reset_caches()
        c0_embed = model._embed_audio(0, c0_sample)
        curr_h = torch.cat([orig_last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0).repeat(curr_h.size(0), 1)
        )
        curr_h = curr_h.to(embeds.dtype)

        for i in range(1, model.config.audio_num_codebooks):
            curr_decoder_mask = model.decoder_causal_mask[curr_pos, :]
            decoder_h = model.decoder(
                model.projection(curr_h),
                input_pos=curr_pos,
                mask=curr_decoder_mask,
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], model.audio_head[i - 1])

            if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
                actual_B = b // 2
                cond_ci = ci_logits[:actual_B]
                uncond_ci = ci_logits[actual_B:]
                guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale
                ci_sample = _sample_topk(guided_ci, topk, temperature)
                ci_sample = ci_sample.repeat(2, 1)
            else:
                ci_sample = _sample_topk(ci_logits, topk, temperature)

            ci_embed = model._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

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
    ) -> torch.Tensor:
        """
        Full guided generation loop. Returns token frames tensor.

        Args:
            pipeline: The ErisedPipeline instance (for preprocessing/postprocessing).
            tags: Structured music tags string.
            lyrics: Song lyrics.
            save_path: Where to save the output audio file.
            max_audio_length_ms: Maximum audio duration in milliseconds.
            temperature: Sampling temperature.
            topk: Top-k filtering parameter.
            cfg_scale: Classifier-free guidance scale.
            dpo_scale: DPO guidance scale (0 = no DPO, 1 = full DPO, >1 = amplified).
        """
        pipe = pipeline.pipe
        model = self.orig_model
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

        # Reset and setup caches
        _reset_model_caches(model)
        self._reset_dpo_caches()
        torch.cuda.empty_cache()

        model.setup_caches(bs_size)
        self._setup_dpo_caches(bs_size)

        # Initial frame (processes the full prompt)
        with torch.autocast(device_type=device.type, dtype=self.dtype):
            curr_token = self._guided_generate_frame(
                prompt_tokens, prompt_tokens_mask, prompt_pos,
                temperature, topk, cfg_scale, dpo_scale,
                continuous_segment, starts,
            )
        frames.append(curr_token[0:1,])

        # Padding helper
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

        # Autoregressive loop
        for i in tqdm(range(max_audio_frames), desc="Guided generation"):
            curr_padded, curr_mask = _pad(curr_token)
            with torch.autocast(device_type=device.type, dtype=self.dtype):
                curr_token = self._guided_generate_frame(
                    curr_padded, curr_mask,
                    prompt_pos[..., -1:] + i + 1,
                    temperature, topk, cfg_scale, dpo_scale,
                )
            if torch.any(curr_token[0:1, :] >= pipe.config.audio_eos_id):
                break
            frames.append(curr_token[0:1,])

        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)

        # Decode to audio
        with torch.no_grad():
            pipe.postprocess({"frames": frames}, save_path=save_path)

        logger.info("Guided audio saved to %s (%d frames)", save_path, frames.shape[-1])
        return frames
