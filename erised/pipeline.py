"""
ErisedPipeline — wraps HeartMuLaGenPipeline to:
  1. Accept rich musical prompts instead of raw tags
  2. Capture generated token sequences (needed for DPO training)
  3. Generate comparison pairs for RLHF preference collection
"""

import os
import uuid
import json
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torchaudio

from heartlib import HeartMuLaGenPipeline
from .config import ErisedConfig
from .prompt_to_tags import PromptToTags
from .guided_generate import DPOGuider

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    generation_id: str
    audio_path: str
    tokens_path: str
    prompt: str
    lyrics: str
    tags_used: str
    num_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


class ErisedPipeline:
    def __init__(self, config: ErisedConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config.dpo_db_path), exist_ok=True)

        self.tag_converter = PromptToTags(
            use_llm=config.use_llm_for_tags,
            api_key=config.llm_api_key,
            model=config.llm_model,
            base_url=config.llm_base_url,
        )

        device_map = {
            "mula": torch.device(config.mula_device),
            "codec": torch.device(config.codec_device),
        }
        dtype_map = {
            "mula": _parse_dtype(config.mula_dtype),
            "codec": _parse_dtype(config.codec_dtype),
        }

        logger.info("Loading HeartMuLa pipeline (version=%s)...", config.version)
        self.pipe = HeartMuLaGenPipeline.from_pretrained(
            config.model_path,
            device=device_map,
            dtype=dtype_map,
            version=config.version,
            lazy_load=config.lazy_load,
        )
        logger.info("HeartMuLa pipeline loaded.")

    def generate(
        self,
        prompt: str,
        lyrics: str,
        *,
        max_audio_length_ms: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> GenerationResult:
        """Generate a song from a musical prompt + lyrics. Returns audio path + saved tokens."""

        tags = self.tag_converter.convert(prompt)
        logger.info("Prompt → tags: %s", tags)

        gen_id = uuid.uuid4().hex[:12]
        audio_path = os.path.join(self.config.output_dir, f"{gen_id}.mp3")
        tokens_path = os.path.join(self.config.output_dir, f"{gen_id}_tokens.pt")

        kwargs = {
            "max_audio_length_ms": max_audio_length_ms or self.config.max_audio_length_ms,
            "temperature": temperature or self.config.temperature,
            "topk": topk or self.config.topk,
            "cfg_scale": cfg_scale or self.config.cfg_scale,
        }

        frames = self._generate_and_capture(tags, lyrics, audio_path, **kwargs)

        torch.save(frames.cpu(), tokens_path)
        logger.info("Saved tokens (%d frames) to %s", frames.shape[-1], tokens_path)

        return GenerationResult(
            generation_id=gen_id,
            audio_path=audio_path,
            tokens_path=tokens_path,
            prompt=prompt,
            lyrics=lyrics,
            tags_used=tags,
            num_frames=frames.shape[-1],
        )

    def generate_pair(
        self,
        prompt: str,
        lyrics: str,
        **kwargs,
    ) -> Tuple[GenerationResult, GenerationResult]:
        """
        Generate two songs from the same prompt for A/B preference comparison.
        
        To get meaningful contrast:
        - A uses lower temperature (more conservative/predictable)
        - B uses higher temperature (more creative/varied)
        - Both use the same target duration
        """
        base_temp = kwargs.pop("temperature", None) or self.config.temperature
        
        # A: conservative (lower temp, higher cfg for more prompt adherence)
        a = self.generate(
            prompt, lyrics,
            temperature=base_temp * 0.8,
            cfg_scale=kwargs.get("cfg_scale", self.config.cfg_scale) * 1.2,
            **kwargs
        )
        
        # B: creative (higher temp, lower cfg for more variation)
        b = self.generate(
            prompt, lyrics,
            temperature=base_temp * 1.2,
            cfg_scale=kwargs.get("cfg_scale", self.config.cfg_scale) * 0.8,
            **kwargs
        )
        
        return a, b

    def _generate_and_capture(
        self,
        tags: str,
        lyrics: str,
        save_path: str,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the HeartMuLa pipeline and intercept the token frames
        before they are decoded to audio (we need them for DPO later).
        """
        cfg_scale = kwargs.get("cfg_scale", self.config.cfg_scale)

        model_inputs = self.pipe.preprocess(
            {"tags": tags, "lyrics": lyrics},
            cfg_scale=cfg_scale,
        )

        # Force-reset KV caches before each generation to avoid shape mismatch
        # when generating back-to-back (e.g. A/B pairs for DPO rating).
        # torchtune's reset_cache() only zeros tensors but doesn't clear the
        # kv_cache reference, so setup_caches() thinks caches are "already setup"
        # and skips re-initialization. We must set kv_cache=None on every
        # attention layer so setup_caches() can create fresh caches.
        for model_part in (self.pipe.mula.backbone, self.pipe.mula.decoder):
            try:
                model_part.reset_caches()
            except (RuntimeError, AttributeError):
                pass
            for layer in model_part.layers:
                attn = getattr(layer, "attn", None)
                if attn is not None and getattr(attn, "kv_cache", None) is not None:
                    attn.kv_cache = None
                    attn.cache_enabled = False
        torch.cuda.empty_cache()

        with torch.no_grad():
            model_outputs = self.pipe._forward(model_inputs, **kwargs)

        frames = model_outputs["frames"]  # [8, num_audio_frames]

        with torch.no_grad():
            self.pipe.postprocess(model_outputs, save_path=save_path)

        logger.info("Audio saved to %s", save_path)
        return frames

    def get_model(self):
        """Access the underlying HeartMuLa model (for DPO training)."""
        return self.pipe.mula

    # ── DPO Guided Generation ─────────────────────────────────────────
    # Instead of merging DPO weights into the model (which causes noise
    # accumulation during autoregressive generation), we apply DPO
    # preferences at the logit level:
    #
    #   logits_final = logits_orig + scale * (logits_dpo - logits_orig)
    #
    # The original model drives generation (stays internally consistent),
    # while the DPO model's learned preferences steer token selection.
    # Only the top N backbone layers (trained by DPO) are branched;
    # the bottom shared layers run once for ~7% overhead instead of 2x.

    def init_guided(
        self,
        dpo_checkpoint_path: str,
        n_dpo_layers: int = 2,
    ):
        """
        Initialize the DPO Guided system. Call this once after pipeline init.

        Creates a DPOGuider that holds only the top N DPO-trained layers,
        sharing the bottom layers with the original model to minimize VRAM.

        Args:
            dpo_checkpoint_path: Path to the DPO checkpoint (e.g. dpo_best).
            n_dpo_layers: Number of top backbone layers that were DPO-trained.
        """
        self.guider = DPOGuider(
            orig_model=self.pipe.mula,
            dpo_checkpoint_path=dpo_checkpoint_path,
            n_dpo_layers=n_dpo_layers,
        )
        logger.info("DPO Guided system initialized from %s", dpo_checkpoint_path)

    def generate_guided(
        self,
        prompt: str,
        lyrics: str,
        *,
        max_audio_length_ms: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        dpo_scale: float = 1.0,
    ) -> GenerationResult:
        """
        Generate a song using DPO Guided inference.

        Instead of running a merged DPO model, this uses the original model
        for internal consistency and applies DPO preferences at the logit
        level. The dpo_scale parameter controls guidance strength:
          - 0.0 = pure original model (no DPO influence)
          - 1.0 = full DPO guidance (default)
          - >1.0 = amplified DPO preferences

        Requires init_guided() to have been called first.
        """
        if not hasattr(self, "guider") or self.guider is None:
            raise RuntimeError(
                "DPO Guided not initialized. Call pipeline.init_guided(path) first."
            )

        tags = self.tag_converter.convert(prompt)
        logger.info("Prompt → tags: %s", tags)

        gen_id = uuid.uuid4().hex[:12]
        audio_path = os.path.join(self.config.output_dir, f"{gen_id}.mp3")
        tokens_path = os.path.join(self.config.output_dir, f"{gen_id}_tokens.pt")

        frames = self.guider.generate(
            pipeline=self,
            tags=tags,
            lyrics=lyrics,
            save_path=audio_path,
            max_audio_length_ms=max_audio_length_ms or self.config.max_audio_length_ms,
            temperature=temperature or self.config.temperature,
            topk=topk or self.config.topk,
            cfg_scale=cfg_scale or self.config.cfg_scale,
            dpo_scale=dpo_scale,
        )

        torch.save(frames.cpu(), tokens_path)
        logger.info("DPO Guided: saved tokens (%d frames) to %s", frames.shape[-1], tokens_path)

        return GenerationResult(
            generation_id=gen_id,
            audio_path=audio_path,
            tokens_path=tokens_path,
            prompt=prompt,
            lyrics=lyrics,
            tags_used=tags,
            num_frames=frames.shape[-1],
        )

    def get_codec(self):
        """Access the underlying HeartCodec (for audio decode during eval)."""
        return self.pipe.codec


def _parse_dtype(s: str) -> torch.dtype:
    return {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }[s.lower()]
