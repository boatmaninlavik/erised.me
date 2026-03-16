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
    the original model's backbone normally (proven code path), use a forward
    hook to capture the hidden state after the shared layers, then branch
    only for the DPO top layers. This avoids duplicating the full model.

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


def _index_causal_mask(mask, input_pos):
    """Index into a causal mask with position indices (same as model's own method)."""
    return mask[input_pos, :]


class DPOGuider:
    """
    Applies DPO preferences at inference time via logit-level guidance,
    using a shared-layer optimization to avoid redundant computation.

    Only the top N backbone layers (the ones actually trained by DPO) and the
    codebook0_head are duplicated. The rest of the model is shared.

    Uses the model's own backbone() forward pass (proven working) with a
    forward hook to capture shared hidden state, rather than manually calling
    individual layers.
    """

    def __init__(
        self,
        orig_model,
        dpo_checkpoint_path: str,
        n_dpo_layers: int = 2,
    ):
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
        self.dpo_backbone_norm = deepcopy(orig_model.backbone.norm)

        # Load DPO checkpoint state dict to CPU, then extract only the weights
        # we need. This avoids creating a full model copy which would OOM.
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
        # For top layers: backbone.layers[n_shared:] remapped to indices 0..n_dpo_layers-1
        top_layer_state = {}
        for key, val in full_state.items():
            if key.startswith("backbone.layers."):
                parts = key.split(".", 3)
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

        self.dpo_top_layers.to(self.device)
        self.dpo_codebook0_head.to(self.device)
        self.dpo_backbone_norm.to(self.device)

        logger.info(
            "DPOGuider ready. VRAM: %.2f GB",
            torch.cuda.memory_allocated(self.device) / 1024**3,
        )

    def _setup_dpo_caches(self, bs_size: int):
        """Set up KV caches for the DPO branch layers."""
        for layer in self.dpo_top_layers:
            attn = getattr(layer, "attn", None)
            if attn is not None:
                if getattr(attn, "kv_cache", None) is not None:
                    attn.kv_cache = None
                    attn.cache_enabled = False

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

    def _get_dpo_logits(self, shared_h, backbone_mask, input_pos):
        """Run shared hidden state through DPO top layers to get DPO logits."""
        dpo_h = shared_h
        for layer in self.dpo_top_layers:
            dpo_h = layer(dpo_h, mask=backbone_mask, input_pos=input_pos)
        dpo_h = self.dpo_backbone_norm(dpo_h)
        dpo_last_h = dpo_h[:, -1, :]
        return self.dpo_codebook0_head(dpo_last_h)

    def _guided_generate_frame(
        self,
        tokens, tokens_mask, input_pos,
        temperature, topk, cfg_scale, dpo_scale,
        continuous_segments=None, starts=None,
    ):
        """
        Generate one audio frame using DPO logit guidance.

        Uses the model's own backbone forward pass (proven code) and captures
        the shared hidden state via a forward hook for the DPO branch.
        """
        model = self.orig_model
        b, s, _ = tokens.size()

        # ── Causal mask (same as model.generate_frame) ──
        backbone_mask = _index_causal_mask(model.backbone_causal_mask, input_pos)

        # ── Unconditional mask for CFG ──
        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            uncond_mask = torch.cat([
                torch.zeros(actual_B, dtype=torch.bool, device=tokens.device),
                torch.ones(actual_B, dtype=torch.bool, device=tokens.device),
            ])

        # ── Embed tokens (same as model.generate_frame) ──
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

        # ── Run original backbone (proven code path) with hook to capture shared state ──
        captured = {}

        def _capture_hook(module, inp, out):
            # Capture the OUTPUT of the last shared layer (input to DPO branch)
            captured["shared_h"] = out.clone()

        hook = model.backbone.layers[self.n_shared - 1].register_forward_hook(_capture_hook)
        try:
            # This calls TransformerDecoder.forward() — same path as model.generate_frame()
            orig_backbone_out = model.backbone(h, input_pos=input_pos, mask=backbone_mask)
        finally:
            hook.remove()

        # ── Original logits (from proven backbone forward) ──
        orig_last_h = orig_backbone_out[:, -1, :]
        orig_c0_logits = model.codebook0_head(orig_last_h)

        # ── DPO logits (from shared hidden state through DPO top layers) ──
        dpo_c0_logits = self._get_dpo_logits(captured["shared_h"], backbone_mask, input_pos)

        # ── Combine logits: CFG first, then DPO guidance ──
        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            orig_guided = (
                orig_c0_logits[actual_B:]
                + (orig_c0_logits[:actual_B] - orig_c0_logits[actual_B:]) * cfg_scale
            )
            dpo_guided = (
                dpo_c0_logits[actual_B:]
                + (dpo_c0_logits[:actual_B] - dpo_c0_logits[actual_B:]) * cfg_scale
            )
            final_logits = orig_guided + dpo_scale * (dpo_guided - orig_guided)
            c0_sample = _sample_topk(final_logits, topk, temperature)
            c0_sample = c0_sample.repeat(2, 1)
        else:
            final_logits = orig_c0_logits + dpo_scale * (dpo_c0_logits - orig_c0_logits)
            c0_sample = _sample_topk(final_logits, topk, temperature)

        # ── Decoder: codebooks 1-7 (original model, same as generate_frame) ──
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
            curr_decoder_mask = _index_causal_mask(model.decoder_causal_mask, curr_pos)
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
        """Full guided generation loop. Returns token frames tensor."""
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

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=self.dtype):
            # Initial frame (processes the full prompt)
            curr_token = self._guided_generate_frame(
                prompt_tokens, prompt_tokens_mask, prompt_pos,
                temperature, topk, cfg_scale, dpo_scale,
                continuous_segment, starts,
            )
            frames.append(curr_token[0:1,])

            # Autoregressive loop
            for i in tqdm(range(max_audio_frames), desc="Guided generation"):
                curr_padded, curr_mask = _pad(curr_token)
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
