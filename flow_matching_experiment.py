"""
Experiment: Does flow matching change the melody?

Take one set of tokens, decode them 5 times with different random seeds.
Compare the outputs by extracting pitch (F0) contours and measuring
how much they differ.

If melody is fully determined by tokens: pitch contours should be nearly identical.
If flow matching influences melody: pitch contours will vary across runs.
"""

import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, "/workspace/heartlib")

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("experiment")

# Load the codec model
logger.info("Loading HeartCodec...")
from heartlib.heartcodec.modeling_heartcodec import HeartCodec

codec = HeartCodec.from_pretrained("/workspace/heartlib/ckpt/HeartCodec-oss")
codec = codec.to("cuda").to(torch.bfloat16)
codec.eval()
logger.info("HeartCodec loaded.")

# Load tokens
tokens = torch.load("/workspace/erised_data/outputs/00101758ca9b_tokens.pt", weights_only=True)
logger.info("Tokens shape: %s", tokens.shape)

# Use only first 375 frames (~30 seconds) to keep it manageable
tokens = tokens[:, :375].to("cuda")

# Decode 5 times with different random seeds
NUM_RUNS = 5
waveforms = []

for i in range(NUM_RUNS):
    torch.manual_seed(i * 12345)
    torch.cuda.manual_seed(i * 12345)
    logger.info("Decoding run %d/%d (seed=%d)...", i + 1, NUM_RUNS, i * 12345)

    wav = codec.detokenize(
        tokens,
        duration=29.76,
        num_steps=10,
        disable_progress=True,
        guidance_scale=1.25,
    )
    waveforms.append(wav.float().cpu().numpy())
    logger.info("  Output shape: %s", wav.shape)

# Save waveforms for manual listening if desired
import soundfile as sf
for i, wav in enumerate(waveforms):
    # wav shape: [channels, samples] — take first channel or mono
    audio = wav[0] if wav.ndim == 2 else wav
    sf.write(f"/workspace/flow_test_run{i}.wav", audio, 48000)
    logger.info("Saved /workspace/flow_test_run%d.wav", i)

# ── Analysis: compare waveforms ──────────────────────────────

logger.info("\n" + "=" * 60)
logger.info("WAVEFORM COMPARISON")
logger.info("=" * 60)

# 1. Direct waveform correlation
ref = waveforms[0][0] if waveforms[0].ndim == 2 else waveforms[0]
for i in range(1, NUM_RUNS):
    other = waveforms[i][0] if waveforms[i].ndim == 2 else waveforms[i]
    min_len = min(len(ref), len(other))
    corr = np.corrcoef(ref[:min_len], other[:min_len])[0, 1]
    diff = np.abs(ref[:min_len] - other[:min_len])
    logger.info("Run 0 vs Run %d: correlation=%.6f, mean_abs_diff=%.6f, max_abs_diff=%.6f",
                i, corr, diff.mean(), diff.max())

# 2. Spectral comparison (compare magnitude spectrograms)
logger.info("\n--- Spectral Comparison ---")
from scipy import signal as sig

def get_spectrogram(audio, sr=48000):
    f, t, Sxx = sig.spectrogram(audio, fs=sr, nperseg=2048, noverlap=1536)
    return f, t, np.log1p(Sxx)

ref_f, ref_t, ref_spec = get_spectrogram(ref)
for i in range(1, NUM_RUNS):
    other = waveforms[i][0] if waveforms[i].ndim == 2 else waveforms[i]
    _, _, other_spec = get_spectrogram(other)
    min_t = min(ref_spec.shape[1], other_spec.shape[1])
    spec_corr = np.corrcoef(ref_spec[:, :min_t].flatten(), other_spec[:, :min_t].flatten())[0, 1]
    spec_diff = np.abs(ref_spec[:, :min_t] - other_spec[:, :min_t]).mean()
    logger.info("Run 0 vs Run %d: spectral_correlation=%.6f, mean_spectral_diff=%.6f",
                i, spec_corr, spec_diff)

# 3. Pitch (F0) comparison using autocorrelation-based method
logger.info("\n--- Pitch (F0) Comparison ---")

def simple_pitch_track(audio, sr=48000, frame_len=2048, hop=512):
    """Simple autocorrelation-based pitch tracker."""
    pitches = []
    for start in range(0, len(audio) - frame_len, hop):
        frame = audio[start:start + frame_len]
        # Autocorrelation
        frame = frame - frame.mean()
        if np.abs(frame).max() < 1e-6:
            pitches.append(0)
            continue
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        # Find first peak after minimum (skip lag 0)
        # Look for pitch between 50 Hz and 2000 Hz
        min_lag = sr // 2000  # ~24
        max_lag = sr // 50    # ~960
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        segment = corr[min_lag:max_lag]
        if len(segment) == 0 or segment.max() < 0.1 * corr[0]:
            pitches.append(0)
            continue
        peak_idx = segment.argmax() + min_lag
        pitches.append(sr / peak_idx)
    return np.array(pitches)

ref_pitch = simple_pitch_track(ref)
logger.info("Reference pitch track: %d frames, mean F0=%.1f Hz (of voiced frames)",
            len(ref_pitch), ref_pitch[ref_pitch > 0].mean() if (ref_pitch > 0).any() else 0)

for i in range(1, NUM_RUNS):
    other = waveforms[i][0] if waveforms[i].ndim == 2 else waveforms[i]
    other_pitch = simple_pitch_track(other)
    min_frames = min(len(ref_pitch), len(other_pitch))

    # Compare only voiced frames (both have pitch > 0)
    both_voiced = (ref_pitch[:min_frames] > 0) & (other_pitch[:min_frames] > 0)
    if both_voiced.sum() > 10:
        voiced_ref = ref_pitch[:min_frames][both_voiced]
        voiced_other = other_pitch[:min_frames][both_voiced]
        pitch_corr = np.corrcoef(voiced_ref, voiced_other)[0, 1]
        pitch_diff_hz = np.abs(voiced_ref - voiced_other).mean()
        pitch_diff_cents = 1200 * np.abs(np.log2(voiced_other / np.clip(voiced_ref, 1, None))).mean()
        logger.info("Run 0 vs Run %d: pitch_correlation=%.6f, mean_pitch_diff=%.1f Hz (%.1f cents), voiced_frames=%d/%d",
                    i, pitch_corr, pitch_diff_hz, pitch_diff_cents, both_voiced.sum(), min_frames)
    else:
        logger.info("Run 0 vs Run %d: too few voiced frames to compare (%d)", i, both_voiced.sum())

logger.info("\n" + "=" * 60)
logger.info("INTERPRETATION:")
logger.info("  - Pitch correlation > 0.99 = melody is identical (flow matching doesn't affect it)")
logger.info("  - Pitch correlation 0.8-0.99 = melody mostly same but some drift")
logger.info("  - Pitch correlation < 0.8 = flow matching significantly changes melody")
logger.info("  - Pitch diff < 20 cents = imperceptible pitch difference")
logger.info("  - Pitch diff 20-50 cents = noticeable but minor")
logger.info("  - Pitch diff > 50 cents = clearly different melody")
logger.info("=" * 60)
