# erised reward model

The reward model that scores songs against personal melodic taste. Used to rank
generations from the song generator, and as the value signal for preference
optimization (DPO/IPO).

## Current SOTA: `rm_v8_crossmix`

5-fold track-OOD CV on 836 hand-labeled songs:

| Metric | Value |
|---|---|
| BG (Good T3-T5 vs Bad T0-T1 pair accuracy) | **70.72%** |
| g5 (T5 vs T0 ranking accuracy) | **86.76%** |
| g5 within-artist | **88.68%** |
| g5 cross-artist | **86.48%** |

**Architecture:** frozen MuQ-base (`OpenMuQ/MuQ-large-msd-iter`) at two complementary
layers — L3 (acoustic) over the full song + L12 (semantic) over the first 90s — each
fed to an independent K=10 MLP head ensemble, then score-fused 50/50 after z-norm.

See [`RECIPE.md`](./RECIPE.md) for the full architecture, training config, and
the design history (why L3/L12, why audio-length asymmetry, what got beaten).

## Files

- [`modal_rm_v8.py`](./modal_rm_v8.py) — deployment-ready Modal app. Two entrypoints:
  - `build_and_save` — trains K=10 heads on all 836 labels, writes checkpoint
  - `score_songs` — scores arbitrary audio files against the trained model
- [`RECIPE.md`](./RECIPE.md) — architecture spec + per-design-decision rationale
- Trained weights live at `gs://erised-dpo/models/rm_v8_crossmix_GOLDEN.pt` (not in repo)
- The 836-song label file (`meta_ts_6tier.json`) is personal taste data kept out
  of this public repo; placed alongside the Modal script before image build
