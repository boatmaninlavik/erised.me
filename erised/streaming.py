"""
Progressive streaming codec decode.

Mirrors HeartCodec.detokenize exactly, but interleaves the flow-matching
and scalar-model decode phases so that partial audio is available after
each ~30-second chunk.  Total wall-clock time is identical to the
original all-at-once decode — no extra overhead.
"""

import logging
import math
import os
from typing import Callable, Optional

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class StreamingDecoder:
    """
    Incremental codec decoder that decodes new chunks without re-decoding
    previous ones.  Create once before the autoregressive loop, then call
    decode_available() whenever new frames are ready.
    """

    def __init__(
        self,
        codec,
        save_path: str,
        duration: float = 29.76,
        num_steps: int = 3,
        guidance_scale: float = 1.25,
    ):
        self.codec = codec
        self.save_path = save_path
        self.duration = duration
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale

        self.device = next(codec.parameters()).device
        self.dtype = next(codec.parameters()).dtype

        # Code-frame-space geometry
        self.min_samples = int(duration * 12.5)          # 372
        self.hop_samples = self.min_samples // 93 * 80   # 320
        self.ovlp_samples = self.min_samples - self.hop_samples  # 52
        self.ovlp_frames = self.ovlp_samples * 2

        # Audio-sample-space geometry — derive from code-frame overlap ratio
        # so it stays correct for ANY chunk duration (the old // 93 * 80
        # formula only worked when min_samples was an exact multiple of 93).
        self.audio_min_samples = int(duration * codec.sample_rate)
        self.audio_ovlp_samples = int(
            self.audio_min_samples * self.ovlp_samples / self.min_samples
        )
        self.audio_hop_samples = self.audio_min_samples - self.audio_ovlp_samples

        self.latent_length = int(duration * 25)
        self.first_latent = torch.randn(
            1, self.latent_length, 256, dtype=self.dtype,
        ).to(self.device)
        self.first_latent_length = 0

        # State preserved between calls
        self.latent_list: list[torch.Tensor] = []
        self.output: Optional[torch.Tensor] = None
        self.chunks_decoded = 0
        self.chunk_paths: list[str] = []

    def next_chunk_at(self) -> int:
        """Number of frames needed before the next chunk can be decoded."""
        if self.chunks_decoded == 0:
            return self.min_samples
        return self.min_samples + self.chunks_decoded * self.hop_samples

    @torch.inference_mode()
    def decode_available(
        self,
        frames_tensor: torch.Tensor,
        on_chunk_ready: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Decode any NEW chunks available from frames_tensor (num_codebooks, num_frames).
        Skips already-decoded chunks.  Returns the number of new chunks decoded.
        """
        codes = frames_tensor.unsqueeze(0).to(self.device)
        codes_len_orig = codes.shape[-1]
        target_len = int(codes_len_orig / 12.5 * self.codec.sample_rate)

        # Pad codes to full-chunk boundaries.
        # Match heartlib's strategy: double the code sequence (wraps real
        # musical content around) instead of repeating the last frame
        # which creates a flatline that flow matching turns into noise.
        if codes_len_orig < self.min_samples:
            while codes.shape[-1] < self.min_samples:
                codes = torch.cat([codes, codes], -1)
            codes = codes[:, :, :self.min_samples]
        codes_len = codes.shape[-1]
        if (codes_len - self.ovlp_samples) % self.hop_samples > 0:
            len_codes = (
                math.ceil((codes_len - self.ovlp_samples) / float(self.hop_samples))
                * self.hop_samples + self.ovlp_samples
            )
            while codes.shape[-1] < len_codes:
                codes = torch.cat([codes, codes], -1)
            codes = codes[:, :, :len_codes]

        chunk_starts = list(
            range(0, codes.shape[-1] - self.hop_samples + 1, self.hop_samples)
        )
        total_chunks = len(chunk_starts)
        new_decoded = 0

        for idx in range(self.chunks_decoded, total_chunks):
            sinx = chunk_starts[idx]

            # Skip the last chunk if >40% of its frames are padding —
            # repeated tokens cause flow matching to produce noise.
            real_frames = min(codes_len_orig - sinx, self.min_samples)
            if real_frames < 0:
                real_frames = 0
            pad_ratio = 1.0 - real_frames / self.min_samples
            if idx == total_chunks - 1 and pad_ratio > 0.4:
                logger.info(
                    "Skipping chunk %d/%d (%.0f%% padding)",
                    idx + 1, total_chunks, pad_ratio * 100,
                )
                continue

            # ── flow matching ──
            codes_input = [codes[:, :, sinx : sinx + self.min_samples]]
            if sinx == 0 or self.ovlp_frames == 0:
                incontext_length = self.first_latent_length
                latents = self.codec.flow_matching.inference_codes(
                    codes_input, self.first_latent, self.latent_length,
                    incontext_length,
                    guidance_scale=self.guidance_scale,
                    num_steps=self.num_steps,
                    disable_progress=True, scenario="other_seg",
                )
            else:
                true_latent = self.latent_list[-1][:, -self.ovlp_frames:, :]
                len_add = self.latent_length - true_latent.shape[1]
                incontext_length = true_latent.shape[1]
                true_latent = torch.cat([
                    true_latent,
                    torch.randn(
                        true_latent.shape[0], len_add, true_latent.shape[-1],
                        dtype=self.dtype,
                    ).to(self.device),
                ], 1)
                latents = self.codec.flow_matching.inference_codes(
                    codes_input, true_latent, self.latent_length,
                    incontext_length,
                    guidance_scale=self.guidance_scale,
                    num_steps=self.num_steps,
                    disable_progress=True, scenario="other_seg",
                )
            self.latent_list.append(latents)

            # ── scalar decode ──
            latent = latents
            if idx == 0:
                latent = latent[:, self.first_latent_length:, :]

            bsz, t, f = latent.shape
            latent = latent.reshape(bsz, t, 2, f // 2).permute(0, 2, 1, 3)
            latent = latent.reshape(bsz * 2, t, f // 2)
            cur_output = (
                self.codec.scalar_model.decode(latent.transpose(1, 2))
                .squeeze(0).squeeze(1)
            )
            cur_output = cur_output[:, :self.audio_min_samples].detach().cpu()
            if cur_output.dim() == 3:
                cur_output = cur_output[0]
            # Normalize if peaks exceed 1.0 — the codec's energy escalates
            # on longer songs.  Scaling (not clamping) preserves the waveform
            # shape so lyrics and melody stay intact instead of turning to
            # square-wave distortion.
            peak = cur_output.abs().max()
            if peak > 1.0:
                cur_output = cur_output / peak * 0.95

            # ── append new audio only (discard overlap to avoid repeated lyrics) ──
            if self.output is None:
                self.output = cur_output
            else:
                # The overlap region decodes the same lyrics twice (different
                # noise → slightly different audio). Discard it entirely and
                # just append the genuinely new audio.
                self.output = torch.cat([
                    self.output,
                    cur_output[:, self.audio_ovlp_samples:],
                ], -1)

            # ── save individual chunk for incremental streaming ──
            chunk_audio = cur_output if idx == 0 else cur_output[:, self.audio_ovlp_samples:]
            chunk_base, chunk_ext = os.path.splitext(self.save_path)
            chunk_path = f"{chunk_base}_chunk{idx}{chunk_ext}"
            # Use .tmp.wav (not .tmp) so torchaudio's soundfile backend
            # recognises the format from the extension.
            temp_chunk_path = f"{chunk_base}_chunk{idx}.tmp{chunk_ext}"
            torchaudio.save(temp_chunk_path, chunk_audio.to(torch.float32), 48000)
            os.replace(temp_chunk_path, chunk_path)
            self.chunk_paths.append(os.path.basename(chunk_path))

            # ── save partial audio ──
            partial = self.output[:, :min(self.output.shape[-1], target_len)]
            temp_path = f"{chunk_base}.tmp{chunk_ext}"
            torchaudio.save(temp_path, partial.to(torch.float32), 48000)
            os.replace(temp_path, self.save_path)

            self.chunks_decoded = idx + 1
            new_decoded += 1
            logger.info(
                "Streaming chunk %d/%d saved (%d samples, %.1fs)",
                self.chunks_decoded, total_chunks,
                partial.shape[-1], partial.shape[-1] / 48000,
            )

            if on_chunk_ready:
                on_chunk_ready(self.chunks_decoded, total_chunks)

        return new_decoded


@torch.inference_mode()
def streaming_detokenize(
    codec,
    codes: torch.Tensor,
    save_path: str,
    on_chunk_ready: Optional[Callable[[int, int], None]] = None,
    duration: float = 29.76,
    num_steps: int = 10,
    guidance_scale: float = 1.25,
) -> torch.Tensor:
    """
    Progressive codec decode that saves partial audio after each chunk.
    Convenience wrapper around StreamingDecoder for one-shot decode.
    """
    decoder = StreamingDecoder(
        codec, save_path,
        duration=duration, num_steps=num_steps, guidance_scale=guidance_scale,
    )
    decoder.decode_available(codes, on_chunk_ready=on_chunk_ready)
    assert decoder.output is not None
    target_len = int(codes.shape[-1] / 12.5 * codec.sample_rate)
    return decoder.output[:, :target_len]
