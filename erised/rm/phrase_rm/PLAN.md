# phrase_rm — detailed plan

## Stage 1 — Phrase extraction (`extract_phrases.py`)

**Input:** 24 kHz mono vocal stem (Demucs-separated from full-mix audio).

**Pitch tracking:**
- `librosa.pyin` with `fmin=C2 (≈65 Hz)`, `fmax=C6 (≈1046 Hz)`, `frame_length=2048`,
  `hop_length=256` — gives ~10 ms hop resolution at 24 kHz.
- Returns per-frame `f0` (Hz, `NaN` if unvoiced) and a voicing-confidence.

**Note grouping:**
- Convert f0 → semitones (`12 * log2(f0/440) + 69` = MIDI pitch).
- Smooth with 30 ms median filter to kill octave-jump artefacts.
- A *note* = maximal run of consecutive voiced frames whose pitches all fall
  inside a ±0.5-semitone band of their running median, with duration ≥ 60 ms.
- Each note stored as `{pitch: float, start_s, end_s, duration_s, mean_amp}`.

**Phrase grouping:**
- A *phrase* = consecutive notes separated by gaps < `phrase_gap_thresh` ms
  (default 250 ms, but adaptive: use the 80th-percentile inter-note gap per
  song so dense-pop and ballad phrases get cut differently).
- Minimum phrase length: 3 notes (singletons are usually ad-libs / errors).

**Output per song (JSON):**
```json
{
  "id": "W043",
  "n_phrases": 47,
  "phrases": [
    {"phrase_idx": 0, "start_s": 12.4, "end_s": 14.7, "n_notes": 5,
     "notes": [{"pitch": 67.0, "start_s": 12.4, "duration_s": 0.4, "amp": 0.31}, ...]},
    ...
  ]
}
```

## Stage 2 — Phrase features (`phrase_features.py`)

Per phrase, ~30-50 numbers:

- **Contour:** start_pitch, end_pitch, range, mean, std, direction
  (+1 rising / -1 falling / 0 flat), n_inflection_points, arch_score
  (correlation with parabola).
- **Rhythm:** n_notes, total_duration_s, note_density (notes / sec),
  mean_note_dur, std_note_dur, n_long_notes (≥ 1 s), proportion of held tones.
- **Intervals:** list of consecutive interval sizes in semitones — emit
  mean_abs_interval, max_interval, n_steps (|interval| ≤ 2), n_leaps (≥ 4),
  ascending_ratio.
- **Position context:** phrase_position_norm (idx / n_phrases),
  is_after_long_gap (boolean — likely section boundary),
  silence_before_s, silence_after_s.

All z-scored over the dataset.

## Stage 3 — Phrase-pair / song-structure features

For each consecutive pair (phrase_i, phrase_i+1):
- pitch_center_shift (semitones)
- range_ratio (i+1 range / i range)
- contour_cosine_sim
- rhythm_density_ratio
- whether phrase_i+1 starts higher (ascending vs descending transition)
- inter-phrase gap_s

Song-global:
- **Repetition score:** for each phrase, find its max-cosine-similarity
  twin elsewhere in the song; histogram of similarities. A hooky song will
  have phrases that match strongly later.
- **Climax score:** max(pitch_range × duration) over all phrases, and where
  in the song it sits (early/middle/late).

## Stage 4 — Hierarchical Multiple-Instance Learning head

```
Phrase features [n_phrases, d_phrase=~50]
  ↓ phrase encoder (Linear → GELU → Linear) [n_phrases, h=128]
Phrase-pair features [n_phrases-1, d_pair=~10]
  ↓ pair encoder [n_phrases-1, h_pair=64]
  ↓ concat to neighbouring phrase embeddings
  ↓ attention pool over phrases (learned query vector) [h_total]
  ↓ Linear → scalar score
```

Training:
- Same v6 hyper-recipe: AdamW lr=3.33e-4, wd=1e-2, 300 epochs, K=10 seed
  ensemble, 5-fold track-OOD CV.
- Loss: **PL-K6** (same listwise as rm_v8) — sample one song per tier,
  the model ranks them.
- Crucially, **attention weights = interpretability**: at eval time we can
  ask "which phrases of this T5 song drove its high score?" and listen.

## Stage 5 — Fusion eval vs rm_v8

Fuse `s_phrase` with rm_v8's `s_L3` + `s_L12`. Weight sweep over (w_L3,
w_L12, w_phrase). Compare best BG to canonical 70.72%.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pitch tracking noisy on dense pop / harmonies | Use voicing-confidence threshold; report extraction quality per song |
| Phrase boundaries miss-segmented | Try multiple gap thresholds; inspect a sample with user; consider beat-aware segmentation |
| MIL on 836 too sparse | Start with simpler aggregation (mean over phrase scores) before attention; only escalate if signal exists |
| Novel infrastructure | Build incrementally; each stage has a sanity check before next |

## Data needs

- **MVP:** zero new labels. Uses existing `meta_ts_6tier.json` (836 song-level tiers).
- **Refinement (optional):** ~50-100 hand-labelled phrase-pair preferences for sharper supervision.

## Cost estimate (Modal)

- Stage 1 extraction (CPU, 847 songs × pyin): ~$0.50
- Stages 2-3 featurisation: local, free
- Stage 4 training: ~$0.30 (CPU is fine for small MLP)
- **Total MVP: ~$1**
