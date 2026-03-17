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

    Args:
        codec: HeartCodec model instance.
        codes: Token tensor of shape (num_codebooks, num_frames).
        save_path: Where to write the (progressively growing) WAV file.
        on_chunk_ready: Called as on_chunk_ready(chunk_idx, total_chunks)
                        after each chunk is decoded and saved to disk.
        duration / num_steps / guidance_scale: Same meaning as in
                        HeartCodec.detokenize.

    Returns:
        Final audio tensor of shape (channels, samples).
    """
    device = next(codec.parameters()).device
    dtype = next(codec.parameters()).dtype

    codes = codes.unsqueeze(0).to(device)
    first_latent = torch.randn(
        codes.shape[0], int(duration * 25), 256, dtype=dtype
    ).to(device)
    first_latent_length = 0
    first_latent_codes_length = 0

    # --- code-frame-space parameters ---
    min_samples = int(duration * 12.5)
    hop_samples = min_samples // 93 * 80
    ovlp_samples = min_samples - hop_samples
    ovlp_frames = ovlp_samples * 2          # latent space runs at 2x code rate

    codes_len_orig = codes.shape[-1]
    target_len = int(
        (codes_len_orig - first_latent_codes_length) / 12.5 * codec.sample_rate
    )

    # Pad codes to full-chunk boundaries (same logic as HeartCodec.detokenize)
    if codes_len_orig < min_samples:
        while codes.shape[-1] < min_samples:
            codes = torch.cat([codes, codes], -1)
        codes = codes[:, :, 0:min_samples]
    codes_len = codes.shape[-1]
    if (codes_len - ovlp_frames) % hop_samples > 0:
        len_codes = (
            math.ceil((codes_len - ovlp_samples) / float(hop_samples))
            * hop_samples + ovlp_samples
        )
        while codes.shape[-1] < len_codes:
            codes = torch.cat([codes, codes], -1)
        codes = codes[:, :, 0:len_codes]

    latent_length = int(duration * 25)

    # --- audio-sample-space parameters (for scalar decode phase) ---
    audio_min_samples = int(duration * codec.sample_rate)
    audio_hop_samples = audio_min_samples // 93 * 80
    audio_ovlp_samples = audio_min_samples - audio_hop_samples

    # --- interleaved chunk processing ---
    latent_list: list[torch.Tensor] = []
    output: Optional[torch.Tensor] = None
    chunk_starts = list(
        range(0, codes.shape[-1] - hop_samples + 1, hop_samples)
    )
    num_chunks = len(chunk_starts)

    for chunk_idx, sinx in enumerate(chunk_starts):
        # ── flow matching for this chunk ──
        codes_input = [codes[:, :, sinx : sinx + min_samples]]
        if sinx == 0 or ovlp_frames == 0:
            incontext_length = first_latent_length
            latents = codec.flow_matching.inference_codes(
                codes_input, first_latent, latent_length, incontext_length,
                guidance_scale=guidance_scale, num_steps=num_steps,
                disable_progress=True, scenario="other_seg",
            )
        else:
            true_latent = latent_list[-1][:, -ovlp_frames:, :]
            len_add = latent_length - true_latent.shape[1]
            incontext_length = true_latent.shape[1]
            true_latent = torch.cat([
                true_latent,
                torch.randn(
                    true_latent.shape[0], len_add, true_latent.shape[-1],
                    dtype=dtype,
                ).to(device),
            ], 1)
            latents = codec.flow_matching.inference_codes(
                codes_input, true_latent, latent_length, incontext_length,
                guidance_scale=guidance_scale, num_steps=num_steps,
                disable_progress=True, scenario="other_seg",
            )
        latent_list.append(latents)

        # ── scalar decode for this chunk ──
        latent = latents
        if chunk_idx == 0:
            latent = latent[:, first_latent_length:, :]

        bsz, t, f = latent.shape
        latent = latent.reshape(bsz, t, 2, f // 2).permute(0, 2, 1, 3)
        latent = latent.reshape(bsz * 2, t, f // 2)
        cur_output = (
            codec.scalar_model.decode(latent.transpose(1, 2))
            .squeeze(0).squeeze(1)
        )
        cur_output = cur_output[:, 0:audio_min_samples].detach().cpu()
        if cur_output.dim() == 3:
            cur_output = cur_output[0]

        if output is None:
            output = cur_output
        else:
            if audio_ovlp_samples == 0:
                output = torch.cat([output, cur_output], -1)
            else:
                ov_win = torch.from_numpy(
                    np.linspace(0, 1, audio_ovlp_samples)[None, :]
                )
                ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                output[:, -audio_ovlp_samples:] = (
                    output[:, -audio_ovlp_samples:]
                    * ov_win[:, -audio_ovlp_samples:]
                    + cur_output[:, 0:audio_ovlp_samples]
                    * ov_win[:, 0:audio_ovlp_samples]
                )
                output = torch.cat(
                    [output, cur_output[:, audio_ovlp_samples:]], -1
                )

        # ── save partial audio (atomic via temp file) ──
        partial = output[:, 0:min(output.shape[-1], target_len)]
        temp_path = save_path + ".tmp"
        torchaudio.save(temp_path, partial.to(torch.float32), 48000, format="wav")
        os.replace(temp_path, save_path)
        logger.info(
            "Streaming chunk %d/%d saved (%d samples, %.1fs)",
            chunk_idx + 1, num_chunks, partial.shape[-1],
            partial.shape[-1] / 48000,
        )

        if on_chunk_ready:
            on_chunk_ready(chunk_idx + 1, num_chunks)

    assert output is not None
    return output[:, 0:target_len]
