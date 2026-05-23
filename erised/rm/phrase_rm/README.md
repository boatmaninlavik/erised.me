# phrase_rm — melodic-phrase reward model (WIP)

A second-pass reward model that scores songs by analysing **melodic phrases**
rather than averaged audio embeddings. Designed to capture the parts of taste
that the pooled-MuQ rm_v8 SOTA (BG=70.72%, see `../`) demonstrably misses.

## Motivation

Three previous melody-feature experiments came back null vs SOTA:

- **per-frame GRU** over MuQ embeddings → +1pp over own baseline, didn't beat SOTA
- **pitch-class bigram motif** features → 0.0 weight in best fusion
- **CQT reconstruction encoder fine-tuning** → no transfer to taste

What those have in common: they all treat melody as a *bag* or *flat sequence*
of pitch-level info. Real melody has **phrase structure** — notes group into
phrases separated by pauses, phrases relate to each other across a song
("the hook comes back three times, slightly different each time"), and the
*relationships between phrases* are where musical greatness actually lives.

## Approach

Four-stage pipeline:

1. **Phrase extraction** — vocal stem → pitch track → notes → phrases
   (segmented at pauses ≥ ~250ms). See `extract_phrases.py`.
2. **Phrase featurisation** — small handmade feature vector per phrase
   (contour, rhythm, intervals, position). See `phrase_features.py` (TBD).
3. **Phrase-pair relationships** — delta features between consecutive
   phrases; repetition / climax detection. (TBD)
4. **Hierarchical RM with Multiple-Instance Learning (MIL)** — song = bag of
   phrases; learn per-phrase scoring + attention pool → song score, trained
   on song-level tier labels via PL-K6 (same loss as rm_v8). (TBD)

## Status

- [x] Subfolder scaffolded
- [ ] Stage 1: phrase extraction code
- [ ] Stage 1: extracted phrase database for the 836 labeled songs
- [ ] Stage 2: phrase features
- [ ] Stage 3: phrase-pair features
- [ ] Stage 4: hierarchical MIL training
- [ ] Fusion eval against rm_v8 SOTA

## Data note

The MVP needs **no new labels** — uses existing song-level tier labels via MIL.
Hand-labeled phrase-pair preferences could refine the model later but are not
required to test the core hypothesis.

See [PLAN.md](./PLAN.md) for the full design with risks/mitigations.
