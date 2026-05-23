# erised/melody_graph — graph-based melody representation

Tools for representing a song's melody as a **graph** (notes + phrases as nodes; intervals, recurrence, phrase repetition as edges) instead of as a flat sequence of notes.

## Why graphs

A song's melody has structure at multiple levels (note → interval → phrase → song). A flat note sequence captures local detail well but hides structural patterns: which phrases repeat, which motifs recur, which intervals are unusual relative to their phrase. The graph representation makes these structural patterns explicit edges and features, so they can be measured directly (and eventually used as input to a graph neural network).

## Files

- **`build_graph_v1.py`** — original PoC. Builds a heterogeneous graph with note-nodes + phrase-nodes and 5 edge types. Renders a per-song PNG (notes colored by pitch, sized by duration; phrases as bounding boxes; consecutive note edges as gray lines; phrase-repetition arcs at the top). Dumps notes/phrases/edges feature CSVs per song. Tested on Cruel Summer (T5) and But Daddy I Love Him (T0).
- **`build_graph_v2.py`** — adds music-theory-aware features WITHOUT changing graph structure:
  - **`interval_quality_category`** on note-consecutive edges: 0=unison, 1=step (1–2 semitones), 2=consonant_leap (3,4,5,7,12), 3=dissonant_leap (6,8–11), 4=octave+
  - **`interval_direction`**: explicit −1/0/+1 (separated from sign of semitones for cleaner GNN attention)
  - **`interval_magnitude_within_phrase`**: how unusual this interval is relative to other intervals in the same phrase (z-scored)
  - **`pitch_deviation_from_phrase_mean`** on note nodes: MIDI − phrase mean MIDI
- **`zoom_guide.py`** — visualization helper. Renders one phrase from a song in detail: every note as a labeled bar (pitch name + duration), every interval explicitly arrowed and labeled in semitones. For reading the graph at note-level granularity instead of song-level overview.

## What the v2 features reveal (Cruel Summer T5 vs But Daddy T0)

| Feature | T5 | T0 | Reading |
|---|---|---|---|
| pct_unison | 3.0% | 12.8% | T0 has 4× more notes that don't change pitch — repetitive |
| pct_dissonant_leap | 10.1% | 3.9% | T5 takes 2.6× more dissonant-interval risks |
| pct_octave_plus | 2.1% | 0.0% | T5 has dramatic octave jumps, T0 has none |
| mean abs pitch deviation from phrase mean | 2.70 | 1.93 | T5 notes range further from phrase center |
| mean phrase pitch range | 9.94 | 8.14 | T5 phrases span wider pitch range |

T5 melodies are dynamic and risk-taking; T0 melodies are static and narrow. These differences are explicit graph features rather than something a human has to manually scan a note table to notice.

## Where it fits in the broader RM stack

The v8 reward model (`erised/rm/modal_rm_v8.py`) is a frozen-MuQ + MLP head pipeline that scores audio holistically. It achieves BG=70.72% on hand-labeled songs but is a black box about WHY a song is good or bad.

A GNN trained on these melody graphs could:
1. Add a melody-pattern signal complementary to v8's holistic audio judgment
2. Provide *interpretable* scoring — pointing at specific phrases / motifs / intervals as the reason for a score
3. Surface aggregate "winning patterns" vs "losing patterns" across the 836-song corpus

Whether this lifts the 70.72% BG ceiling is an open empirical question — six prior melody-specific RM attempts have failed for various reasons (mostly supervision-starvation at the phrase level). Graphs' stronger relational inductive bias is the leading argument for why it might work where transformers haven't.

## How to run

```bash
# Build graphs for Cruel Summer + But Daddy (writes to /tmp/graph_poc/output_v2/)
python build_graph_v2.py

# View detail of one phrase from one song
python zoom_guide.py
```

Both scripts read note + phrase data extracted by `phrase_demo/melody_v2/` (CREPE pitch tracker + Whisper phrase boundaries). That extraction pipeline isn't in this repo; only the graph layer on top of it.

## What's NOT in this directory

- The CREPE / Whisper extraction pipeline that produces note + phrase tables
- Any trained GNN model (proof-of-concept only — no model trained yet)
- Cross-song aggregate analysis (planned next step: extend v2 to all 836 songs, dump feature CSV, group by tier)
