# rm_v8_crossmix — GOLDEN deployment RM

Built 2026-05-19 from session of 2026-05-18 work.

## Performance (5-fold track-OOD CV on 836 hand-labeled songs)

| Metric | Value |
|---|---|
| BG (Good vs Bad pair accuracy, GOOD={T3,T4,T5}, BAD={T0,T1}) | **70.72%** |
| g5 (T5 vs T0 ranking accuracy) | **86.76%** |
| g5_within-artist | **88.68%** |
| g5_cross-artist | **86.48%** |
| ov (all-pair accuracy) | ~63.7% |

## Architecture

```
audio (full song, up to 360s)
  ├──> MuQ-base.hidden_states[3], mean-pool over 30s windows (avg ~7 windows = 3.5 min)
  │     ├──> L2-normalize (1024-dim)
  │     └──> MLPHead_L3 (1024 → 64 → 1, LayerNorm + Dropout 0.7 + GELU)
  │           └──> K=10 seed ensemble (averaged)
  │                 └──> s_L3
  │
  └──> First 90s of audio
        └──> MuQ-base.hidden_states[12], mean-pool over 30s windows (3 windows)
              ├──> L2-normalize (1024-dim)
              └──> MLPHead_L12 (1024 → 64 → 1)
                    └──> K=10 seed ensemble
                          └──> s_L12

final_score = 0.5 * z(s_L3) + 0.5 * z(s_L12)
  where z uses (mean, std) of each head's predictions on the 836-song training set
```

## Why this design wins (vs prior approaches)

| Approach | Best BG | Best g5 | Why beaten |
|---|---|---|---|
| MuQ-MuLan top + LoRA-warm-b4 (prior best, Yesterday) | 64.86 | 82.11 | Used MuLan's text-contrastive projection (loses ~5pp). Used 30s window only. |
| MuQ-base L12 alone (single 30s) | 66.45 | 80.77 | Single 30s window misses song variation. |
| MuQ-base L12 mw3 (90s of audio) | 68.74 | 83.50 | Doesn't use L3's acoustic complement. |
| **rm_v8 cross-mix (L3 fullsong + L12 90s)** | **70.72** | **86.76** | Audio-length asymmetry: acoustic layers benefit from more audio; semantic layers get diluted past 90s. |

## Critical design insights (learned today)

1. **Use MuQ-base raw layers, NOT MuQ-MuLan's `audio_to_latents`** — the MuLan text-contrastive projection actively destroys ~5pp of taste-relevant signal in service of text-genre alignment. Always work with `MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")` and access `hidden_states[layer_ix]` directly.

2. **Layer choice matters more than encoder choice** — at the time of confusion, we tried MuQ-base, MusicFM, MERT, vocal-only, fusion across encoders. None of those moved the needle. Just picking the right LAYER inside MuQ-base (L3 for acoustic, L12 for semantic) gave +5pp.

3. **Audio-length asymmetry per layer** — L3 (acoustic) wants the full song (avg ~3.5 min); L12 (semantic) wants only the first 90s. More audio for L12 ADDS noise (vibe is established quickly). Inverse cross (L3 short + L12 long) does NOT win → directional, not generic fusion lift.

4. **Score-level fusion > feature-level fusion** — concatenating features (early fusion) at 1024+1024=2048d gave only marginal lift. Score-level fusion (z-norm + weighted sum) lets each head fully exploit its preferred modality.

5. **Cold-start v6 recipe (PL-K6 + 300ep + K=10) was already optimal head training** — LoRA fine-tuning of the encoder added zero over a well-trained cold head. Don't bother with LoRA unless you're working with a totally different encoder.

## Inference for new audio

```python
import torch
from muq import MuQ
import librosa

ckpt = torch.load('rm_v8_crossmix_GOLDEN.pt', weights_only=False)
m = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").cuda().eval()
for p in m.parameters(): p.requires_grad = False

# Score one song:
sr = 24000; win = 30 * sr
# L3 over full song (up to 360s)
wav_full, _ = librosa.load(audio_path, sr=sr, mono=True, duration=360)
# L12 over first 90s
wav_90, _ = librosa.load(audio_path, sr=sr, mono=True, duration=90)
# ... (extract via MuQ, run heads, z-norm against ckpt['zn_L3']/ckpt['zn_L12'], fuse 50/50)
```

Full scoring code: `modal_batch/modal_rm_v8_crossmix.py::score_songs`

## Validation on user's real songs (2026-05-19)

| Song | Fused | Percentile | User-confirmed ranking |
|---|---|---|---|
| October | +0.48 | 77.8% | (highest of 4) |
| What I See in Her | +0.21 | 62.9% | |
| Wind Shadow | +0.20 | 62.8% | preferred over Can u see me ✓ |
| Can u see me | +0.05 | 51.7% | least-preferred ✓ |

User confirmed: Wind Shadow > Can u see me ordering matches gut.

## Known weaknesses

1. **30% T5 miss rate** (39/129 of user's masterpieces ranked below T2 median)
2. **Jay Chou polarity inversion** — model learned Jay Chou direction but mis-applies within his catalog (3/4 of his T0s rated above-median)
3. **Ballad bias** — model underrates quieter emotional T5s (Billie Eilish ballads, Taylor Swift deep cuts)
4. **L3 needs full song** — short input songs (e.g., 60-90s versions) get less reliable L3 estimates

## Storage locations

- `gs://erised-dpo/models/rm_v8_crossmix_GOLDEN.pt` (canonical backup)
- `erised-data:/ckpt/rm_v8_crossmix.pt` (Modal volume — used by scoring code)
- `~/Desktop/real_song/rm_checkpoints/rm_v8_crossmix_GOLDEN.pt` (local copy)
