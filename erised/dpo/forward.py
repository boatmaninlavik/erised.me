"""
Training-mode forward pass for HeartMuLa.

During inference, HeartMuLa generates one frame at a time with KV caches.
For DPO training, we need to compute log p(A | C) over an entire sequence
using teacher forcing. This module implements that.

Architecture recap (from the paper, Section 3.1):
  - Global backbone (3B) predicts codebook-0 tokens given the full context
  - Local decoder (300M) predicts codebooks 1–7 given backbone hidden state
    + preceding codebook embeddings within each frame
  - Loss = weighted CE across all codebook layers (Eq. 8–10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from heartlib.heartmula.modeling_heartmula import HeartMuLa, _create_causal_mask, _index_causal_mask


def compute_sequence_log_probs(
    model: HeartMuLa,
    tokens: torch.Tensor,
    tokens_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher-forced forward pass through HeartMuLa.

    Args:
        model: HeartMuLa model (should NOT have caches enabled)
        tokens: [B, L, 9] — full sequence (text prompt + audio frames).
                Last dim: 8 audio codebooks + 1 text channel.
        tokens_mask: [B, L, 9] bool — which channels are active at each position

    Returns:
        global_log_probs: [B] — mean log p(a_{l,0} | h_{<l}) over audio frames
        local_log_probs:  [B] — mean log p(a_{l,k} | ...) for k=1..K-1 over audio frames
                          (averaged over both frames and codebooks so global/local are on same scale)
    """
    B, L, C = tokens.shape
    K = model.config.audio_num_codebooks  # 8
    device = tokens.device

    text_len = _find_text_length(tokens, tokens_mask, K)

    # ── Backbone forward (full sequence, teacher forcing) ───────────
    embeds = model._embed_tokens(tokens, uncond_mask=None)  # [B, L, K+1, D]
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2, dtype=embeds.dtype)  # [B, L, D]

    causal_mask = _create_causal_mask(L, device)
    input_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    backbone_mask = _index_causal_mask(causal_mask, input_pos)  # [B, L, L]

    backbone_out = model.backbone(h, input_pos=input_pos, mask=backbone_mask)
    # backbone_out: [B, L, D]

    # ── Layer-0 log-probs (global) ──────────────────────────────────
    # At position l-1, backbone predicts a_{l,0}
    # So logits from positions text_len .. L-2 predict audio frames 1 .. L-1-text_len
    audio_start = text_len
    num_audio = L - audio_start

    # Cast to match head dtype (backbone may output float32 under autocast while head is bfloat16)
    head_dtype = model.codebook0_head.weight.dtype
    c0_logits = model.codebook0_head(backbone_out[:, audio_start - 1 : L - 1, :].to(head_dtype))
    # c0_logits: [B, num_audio_frames, vocab]

    c0_targets = tokens[:, audio_start:L, 0]  # [B, num_audio_frames]
    c0_log_probs = F.log_softmax(c0_logits, dim=-1)
    c0_token_log_probs = c0_log_probs.gather(2, c0_targets.unsqueeze(-1)).squeeze(-1).float()
    # [B, num_audio_frames]

    # Average over audio frames so the magnitude doesn't scale with song length.
    # Without this, a 60s song would have ~3x the gradient of a 20s song,
    # and the DPO sigmoid would saturate for long sequences.
    global_log_probs = c0_token_log_probs.sum(dim=1) / max(1, num_audio)  # [B]

    # ── Layers 1–7 log-probs (local decoder) ────────────────────────
    if num_audio <= 0:
        return global_log_probs, torch.zeros(B, device=device)

    # Collect backbone hidden states at audio positions
    backbone_h = backbone_out[:, audio_start:L, :]  # [B, num_audio, D]

    # Collect true audio tokens for codebooks 0..K-2 at audio positions
    audio_tokens = tokens[:, audio_start:L, :K]  # [B, num_audio, K]

    # Build decoder input for all frames in parallel:
    #   For frame l, input = [backbone_h_l, embed(a_l,0), embed(a_l,1), ..., embed(a_l,K-2)]
    #   Length = K (backbone_h + K-1 audio codebook embeddings)
    decoder_inputs = []
    decoder_inputs.append(backbone_h)  # position 0: backbone hidden

    for k in range(K - 1):
        ck_embed = model._embed_audio(k, audio_tokens[:, :, k])  # [B, num_audio, D]
        decoder_inputs.append(ck_embed)

    # Stack: [B, num_audio, K, D]
    decoder_seq = torch.stack(decoder_inputs, dim=2)

    # Reshape to batch frames: [B * num_audio, K, D]
    D = decoder_seq.shape[-1]
    decoder_seq_flat = decoder_seq.reshape(B * num_audio, K, D)

    # Project through the model's projection layer (match dtype)
    proj_dtype = next(model.projection.parameters()).dtype
    decoder_seq_flat = model.projection(decoder_seq_flat.to(proj_dtype))

    # Decoder causal mask and positions (shared across chunks)
    dec_mask_raw = _create_causal_mask(K, device)

    # Process decoder in chunks to avoid OOM (each chunk = CHUNK_SIZE frames)
    DECODER_CHUNK = 128
    local_log_probs_total = torch.zeros(B, device=device)
    total_frames = B * num_audio

    for chunk_start in range(0, total_frames, DECODER_CHUNK):
        chunk_end = min(chunk_start + DECODER_CHUNK, total_frames)
        chunk_size = chunk_end - chunk_start

        chunk_input = decoder_seq_flat[chunk_start:chunk_end]  # [chunk, K, D]
        dec_pos = torch.arange(K, device=device).unsqueeze(0).expand(chunk_size, -1)
        dec_mask = _index_causal_mask(dec_mask_raw, dec_pos)

        chunk_out = model.decoder(chunk_input, input_pos=dec_pos, mask=dec_mask)
        # [chunk, K, D_dec]

        for k in range(1, K):
            dec_h_k = chunk_out[:, k, :]  # [chunk, D_dec]
            ah = model.audio_head[k - 1]
            ck_logits = torch.mm(dec_h_k.to(ah.dtype), ah)  # [chunk, vocab]
            ck_log_probs = F.log_softmax(ck_logits, dim=-1)

            ck_targets = audio_tokens[:, :, k].reshape(total_frames)[chunk_start:chunk_end]
            ck_token_log_probs = ck_log_probs.gather(1, ck_targets.unsqueeze(-1)).squeeze(-1).float()
            # [chunk] — accumulate directly into the per-batch sum
            # Map chunk indices back to batch dimension
            batch_indices = torch.arange(chunk_start, chunk_end, device=device) // num_audio
            local_log_probs_total.scatter_add_(0, batch_indices, ck_token_log_probs)

    # Average over both frames AND the 7 codebooks.
    # Raw local sums over 7 codebooks vs global's 1, so without this
    # normalization, local would dominate global by ~7x regardless of weights.
    local_log_probs_total = local_log_probs_total / max(1, num_audio * (K - 1))

    return global_log_probs, local_log_probs_total


def _find_text_length(tokens: torch.Tensor, tokens_mask: torch.Tensor, K: int) -> int:
    """
    Find where audio frames start in the sequence.
    Text positions have tokens_mask[:, l, -1] = True and audio channels inactive.
    Audio positions have tokens_mask[:, l, :-1] = True and text channel inactive.
    We look for the first position where any audio codebook is active.
    """
    audio_active = tokens_mask[0, :, :K].any(dim=-1)  # [L]
    active_positions = torch.where(audio_active)[0]
    if len(active_positions) == 0:
        return tokens.shape[1]
    return active_positions[0].item()


def build_training_sequence(
    pipe,
    tags: str,
    lyrics: str,
    audio_frames: torch.Tensor,
    cfg_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct the full [text + audio] token sequence for teacher-forced training.

    Args:
        pipe: HeartMuLaGenPipeline instance (used for text tokenization)
        tags: comma-separated tags string
        lyrics: lyrics string
        audio_frames: [K, num_audio_frames] tensor of audio codebook tokens
        cfg_scale: set to 1.0 for training (no classifier-free guidance duplication)

    Returns:
        tokens: [1, L, 9] — full sequence
        tokens_mask: [1, L, 9] — active channel mask
    """
    model_inputs = pipe.preprocess({"tags": tags, "lyrics": lyrics}, cfg_scale=cfg_scale)

    prompt_tokens = model_inputs["tokens"]  # [1, prompt_len, 9]
    prompt_mask = model_inputs["tokens_mask"]  # [1, prompt_len, 9]

    K, num_audio = audio_frames.shape
    parallel = prompt_tokens.shape[-1]  # 9

    # Build audio token rows: codebooks in channels 0..7, text channel (8) is empty
    audio_tokens = torch.zeros(1, num_audio, parallel, dtype=torch.long)
    audio_tokens[0, :, :K] = audio_frames.T  # [num_audio, K]

    audio_mask = torch.zeros(1, num_audio, parallel, dtype=torch.bool)
    audio_mask[0, :, :K] = True

    tokens = torch.cat([prompt_tokens, audio_tokens], dim=1)
    mask = torch.cat([prompt_mask, audio_mask], dim=1)

    return tokens, mask
