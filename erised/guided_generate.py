"""
DPO Logit Guidance — same approach as guided_test.py (v8) but memory-efficient.

v8 used two full model copies and called model.backbone() on each.
That OOMs on A100-40GB. Instead, we keep only the DPO-trained layers
as separate copies and swap them into the original model for the DPO
forward pass, then swap back. Same code path, same results, ~7% VRAM overhead.

    logits_final = logits_orig + scale * (logits_dpo - logits_orig)
"""

import glob
import logging
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
    This is the exact same logic as guided_test.py's generate_frame_logits (v8).
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


class DPOGuider:
    """
    Same as v8's two-model guided generation, but memory-efficient.

    Instead of deepcopy-ing the entire model (~19GB), we only copy the
    DPO-trained layers (~1.3GB for 2 layers + norm + head). For the DPO
    forward pass, we temporarily swap these into the original model's
    backbone, call model.backbone() (same proven code path as v8), then
    swap the original layers back. Each set of layers carries its own
    KV caches, so both branches maintain independent cache histories.
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

        # Deep copy only the DPO-trained parts
        self.dpo_top_layers = [
            deepcopy(orig_model.backbone.layers[self.n_shared + i])
            for i in range(n_dpo_layers)
        ]
        self.dpo_backbone_norm = deepcopy(orig_model.backbone.norm)
        self.dpo_codebook0_head = deepcopy(orig_model.codebook0_head)

        # Load DPO weights from checkpoint (to CPU first to avoid OOM)
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

        # Extract top layer weights (remap indices)
        for i in range(n_dpo_layers):
            src_idx = self.n_shared + i
            prefix = f"backbone.layers.{src_idx}."
            layer_state = {
                k[len(prefix):]: v
                for k, v in full_state.items()
                if k.startswith(prefix)
            }
            if layer_state:
                self.dpo_top_layers[i].load_state_dict(layer_state, strict=False)
                logger.info("Loaded %d tensors into DPO layer %d (backbone.layers.%d)",
                            len(layer_state), i, src_idx)

        # Extract norm weights
        norm_state = {
            k[len("backbone.norm."):]: v
            for k, v in full_state.items()
            if k.startswith("backbone.norm.")
        }
        if norm_state:
            self.dpo_backbone_norm.load_state_dict(norm_state, strict=False)

        # Extract codebook0_head weights
        head_state = {
            k[len("codebook0_head."):]: v
            for k, v in full_state.items()
            if k.startswith("codebook0_head.")
        }
        if head_state:
            self.dpo_codebook0_head.load_state_dict(head_state, strict=False)

        del full_state

        # Move to device
        for layer in self.dpo_top_layers:
            layer.to(self.device)
        self.dpo_backbone_norm.to(self.device)
        self.dpo_codebook0_head.to(self.device)

        logger.info(
            "DPOGuider ready. VRAM: %.2f GB",
            torch.cuda.memory_allocated(self.device) / 1024**3,
        )

    def _setup_dpo_caches(self, bs_size: int):
        """Set up KV caches for the DPO layers (mirrors model.setup_caches)."""
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
        for layer in self.dpo_top_layers:
            attn = getattr(layer, "attn", None)
            if attn is not None:
                if getattr(attn, "kv_cache", None) is not None:
                    attn.kv_cache = None
                    attn.cache_enabled = False

    def _swap_in_dpo(self):
        """Swap DPO layers into the model. Returns saved originals."""
        model = self.orig_model
        saved = {
            "layers": [],
            "norm": model.backbone.norm,
            "head": model.codebook0_head,
        }
        for i in range(self.n_dpo_layers):
            idx = self.n_shared + i
            saved["layers"].append(model.backbone.layers[idx])
            model.backbone.layers[idx] = self.dpo_top_layers[i]
        model.backbone.norm = self.dpo_backbone_norm
        model.codebook0_head = self.dpo_codebook0_head
        return saved

    def _swap_back(self, saved):
        """Restore original layers."""
        model = self.orig_model
        for i in range(self.n_dpo_layers):
            idx = self.n_shared + i
            model.backbone.layers[idx] = saved["layers"][i]
        model.backbone.norm = saved["norm"]
        model.codebook0_head = saved["head"]

    def _guided_generate_frame(
        self,
        tokens, tokens_mask, input_pos,
        temperature, topk, cfg_scale, dpo_scale,
        continuous_segments=None, starts=None,
    ):
        """
        Generate one frame — same as v8's guided_generate_frame.

        1. Get original logits via model.backbone() (original layers)
        2. Swap in DPO layers, get DPO logits via model.backbone() (same code path)
        3. Swap back original layers
        4. Combine logits, sample, run decoder
        """
        model = self.orig_model
        b = tokens.size(0)
        embeds_dtype = self.dtype

        # === Original forward (proven code path) ===
        orig_c0_logits, orig_last_h = _generate_frame_logits(
            model, tokens, tokens_mask, input_pos, cfg_scale,
            continuous_segments, starts,
        )

        # === DPO forward (swap layers, same code path) ===
        saved = self._swap_in_dpo()
        try:
            dpo_c0_logits, _ = _generate_frame_logits(
                model, tokens, tokens_mask, input_pos, cfg_scale,
                continuous_segments, starts,
            )
        finally:
            self._swap_back(saved)

        # === Combine logits (same as v8) ===
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

        # === Decoder: codebooks 1-7 (original model, same as v8) ===
        model.decoder.reset_caches()
        c0_embed = model._embed_audio(0, c0_sample)
        curr_h = torch.cat([orig_last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0).repeat(curr_h.size(0), 1)
        )
        curr_h = curr_h.to(embeds_dtype)

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
        """Full guided generation loop (same structure as v8's guided_forward)."""
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

        # Reset and setup caches (original model + DPO layers)
        _reset_model_caches(model)
        self._reset_dpo_caches()
        torch.cuda.empty_cache()

        model.setup_caches(bs_size)
        self._setup_dpo_caches(bs_size)

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
            # Initial frame
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

        with torch.no_grad():
            pipe.postprocess({"frames": frames}, save_path=save_path)

        logger.info("Guided audio saved to %s (%d frames)", save_path, frames.shape[-1])
        return frames
