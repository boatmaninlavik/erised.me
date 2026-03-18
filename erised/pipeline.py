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
from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torchaudio

from heartlib import HeartMuLaGenPipeline
from .config import ErisedConfig
from .prompt_to_tags import PromptToTags
from .guided_generate import DPOGuider
from .streaming import streaming_detokenize

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
        on_progress: Optional[Callable] = None,
        on_frames_checkpoint: Optional[Callable] = None,
    ) -> GenerationResult:
        """Generate a song from a musical prompt + lyrics. Returns audio path + saved tokens."""

        tags = self.tag_converter.convert(prompt)
        logger.info("Prompt → tags: %s", tags)

        gen_id = uuid.uuid4().hex[:12]
        audio_path = os.path.join(self.config.output_dir, f"{gen_id}.wav")
        tokens_path = os.path.join(self.config.output_dir, f"{gen_id}_tokens.pt")

        kwargs = {
            "max_audio_length_ms": max_audio_length_ms or self.config.max_audio_length_ms,
            "temperature": temperature or self.config.temperature,
            "topk": topk or self.config.topk,
            "cfg_scale": cfg_scale or self.config.cfg_scale,
        }

        frames = self._generate_and_capture(tags, lyrics, audio_path, on_progress=on_progress,
                                              on_frames_checkpoint=on_frames_checkpoint, **kwargs)

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
        on_progress: Optional[Callable] = None,
        on_frames_checkpoint: Optional[Callable] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the HeartMuLa generation loop with streaming support.

        Uses an explicit autoregressive loop (matching heartlib's _forward)
        so we can inject partial audio decoding and progress callbacks.
        """
        cfg_scale = kwargs.get("cfg_scale", self.config.cfg_scale)
        temperature = kwargs.get("temperature", self.config.temperature)
        topk = kwargs.get("topk", self.config.topk)
        max_audio_length_ms = kwargs.get("max_audio_length_ms", self.config.max_audio_length_ms)

        model_inputs = self.pipe.preprocess(
            {"tags": tags, "lyrics": lyrics},
            cfg_scale=cfg_scale,
        )

        # Force-reset KV caches before each generation to avoid shape mismatch
        # when generating back-to-back (e.g. A/B pairs for DPO rating).
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

        device = self.pipe.mula_device
        dtype = self.pipe.mula_dtype
        mula = self.pipe.mula

        prompt_tokens = model_inputs["tokens"].to(device)
        prompt_tokens_mask = model_inputs["tokens_mask"].to(device)
        continuous_segment = model_inputs["muq_embed"].to(device)
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"].to(device)
        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        mula.setup_caches(bs_size)

        # Initial frame (processes the full prompt)
        with torch.autocast(device_type=device.type, dtype=dtype):
            curr_token = mula.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
            )
        frames.append(curr_token[0:1,])

        # Padding helper (matches heartlib's _pad_audio_token)
        parallel_number = 8 + 1
        empty_id = self.pipe.config.empty_id

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
        # Codec uses 12s chunks (min_samples=150, hop=80). Send frames early
        # and often so codec worker can decode new chunks every ~80 frames.
        _FIRST_CHUNK = 100
        _HOP = 80
        next_checkpoint = _FIRST_CHUNK if on_frames_checkpoint else None

        # Autoregressive generation loop — NO codec decode here.
        # Running codec (flow_matching + scalar_model) while mula has
        # active KV caches causes SIGABRT / GPU state corruption.
        for i in range(max_audio_frames):
            curr_padded, curr_mask = _pad(curr_token)
            with torch.autocast(device_type=device.type, dtype=dtype):
                curr_token = mula.generate_frame(
                    tokens=curr_padded,
                    tokens_mask=curr_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                )
            if torch.any(curr_token[0:1, :] >= self.pipe.config.audio_eos_id):
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
            logger.info("Frames sent to external decoder (%d frames)", frames_tensor.shape[-1])
        else:
            # No external decoder — decode locally (original path)
            def _on_chunk(chunk_idx, total_chunks):
                if on_progress:
                    on_progress(
                        num_gen_frames, max_audio_frames,
                        os.path.basename(save_path), chunk_idx,
                    )

            streaming_detokenize(
                self.pipe.codec, frames_tensor, save_path,
                on_chunk_ready=_on_chunk,
            )
            logger.info("Audio saved to %s (%d frames)", save_path, frames_tensor.shape[-1])

        return frames_tensor

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
        **kwargs,
    ):
        """
        Initialize the DPO Guided system. Call this once after pipeline init.

        Creates a full deep copy of the model and loads DPO weights into it
        (same proven approach as v8). Requires A100-80GB.
        """
        self.guider = DPOGuider(
            orig_model=self.pipe.mula,
            dpo_checkpoint_path=dpo_checkpoint_path,
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
        on_progress: Optional[Callable] = None,
        on_frames_checkpoint: Optional[Callable] = None,
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
        audio_path = os.path.join(self.config.output_dir, f"{gen_id}.wav")
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
            on_progress=on_progress,
            on_frames_checkpoint=on_frames_checkpoint,
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
