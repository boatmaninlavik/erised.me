# Representing Songs as Graphs

*Proposal for a melody-pattern reward model that complements rm_v8's holistic audio judgment.*

## The problem

Our SOTA reward model (`rm_v8_crossmix`, BG=70.72%) is a black box on raw audio. It judges songs holistically: feeds the whole audio through a frozen MuQ encoder, mean-pools, runs a small MLP, outputs a taste score. **It works but it's stuck.** Six attempts to add melody-specific signals to it (per-frame GRU, motif bigrams, autoencoder LoRA, vocal-only MuQ, phrase-transformer, symbolic-melody transformer) all got weight ≈0 in fusion. The obvious failure modes have been ruled out — pooling layer, encoder fine-tuning, missing pitch info.

The leading diagnosis for the 30%-BG gap is **supervision starvation**: 836 hand-labeled songs aren't enough to teach a flexible neural net to discriminate at the phrase level. The last symbolic-melody-transformer attempt's pool attention collapsed to uniform — the model never learned which phrases mattered, because 836 song-level labels are too sparse a signal for that.

## The hypothesis: a structured musical representation might extract more per label

What we want a representation to capture about a song:
- **Each note** (pitch, duration)
- **Each phrase** (lyric-line group of notes — "fever dream high in the quiet of the night" is one phrase)
- **The interval between consecutive notes** (the melodic motion)
- **The contrast between notes within a phrase** (does the singer leap dramatically or stay close?)
- **The contrast between phrases** (does the chorus soar above the verse? does any phrase repeat?)
- **The duration dynamics** (sustained vs short, rhythm changes between phrases)

A flat note table (`idx | start | dur | MIDI | interval`) captures local detail but hides structural relationships. A **graph** makes the relationships explicit edges and lets us measure them directly.

## What the graph looks like

A song becomes a **heterogeneous graph** — two kinds of nodes:

- **Note nodes** — one per detected note. Features: pitch (MIDI), duration (seconds), octave, position within its phrase, deviation from phrase mean pitch.
- **Phrase nodes** — one per lyric line. Features: number of notes, mean pitch, pitch range, contour direction, total duration, position in song.

And **five kinds of edges** capturing the relationships:

| Edge | Connects | Captures |
|---|---|---|
| consecutive | note → next note | interval (semitones), duration ratio, time gap |
| recurrence | note → same-pitch-class note elsewhere | motif/hook repetition (non-local) |
| membership | note → its phrase | which phrase a note belongs to |
| phrase-consecutive | phrase → next phrase | pitch shift between phrases, range ratio, duration ratio |
| phrase-repetition | phrase → similar earlier phrase | verse/chorus/hook structure |

### v2 adds music-theory-aware features

The v1 graph captures raw musical primitives. The v2 version (in [`build_graph_v2.py`](build_graph_v2.py)) adds:

- **`interval_quality_category`** on note-consecutive edges: 0=unison, 1=step (1–2 semitones), 2=consonant_leap (3,4,5,7,12), 3=dissonant_leap (6,8–11), 4=octave+
- **`interval_direction`**: explicit −1/0/+1 (separated from sign of semitones for cleaner GNN attention)
- **`interval_magnitude_within_phrase`**: how unusual this interval is relative to other intervals in the same phrase (z-scored)
- **`pitch_deviation_from_phrase_mean`** on note nodes: MIDI − phrase mean MIDI

This explicit categorization makes patterns like "lots of dissonant leaps" or "all unison" measurable directly, not buried in a long sequence of integers.

## What the graph actually surfaces — Cruel Summer (T5) vs But Daddy I Love Him (T0)

Two Taylor Swift songs, opposite tiers. v2 graph features:

| Feature | Cruel Summer (T5) | But Daddy (T0) | Reading |
|---|---|---|---|
| % unison intervals | **3.0%** | 12.8% | T0 has 4× more notes that don't move — repetitive |
| % dissonant leaps | **10.1%** | 3.9% | T5 takes 2.6× more interval risks |
| % octave-plus jumps | **2.1%** | 0.0% | T5 has dramatic octave leaps; T0 has none |
| Pitch deviation from phrase mean | **2.70** | 1.93 | T5 notes range further from phrase center |
| Phrase pitch range | **9.94 st** | 8.14 st | T5 phrases span wider pitches |

T5 takes risks (dissonant leaps, octave jumps, wandering within phrase). T0 is monotonous (4× more unisons, narrower range). **These differences are explicit columns in a CSV, not something you scan a 500-note table for.**

The same data in the flat note-table format would require manually scanning hundreds of rows and computing these statistics by eye. The graph + v2 features make them just-look-at-the-CSV obvious.

## How this could complement v8 (not replace it)

v8 is an audio-features judge. It hears the whole song's production and timbre. The graph would be a melody-pattern judge — it operates on the symbolic structure alone (note pitches, durations, lyric phrases), independent of production quality.

The two judgments are about different things and could provide independent signals:
- A song that scores well on BOTH = good melodic structure AND good production
- A song that scores well on v8 but poorly on the graph (or vice versa) = the disagreement tells us which dimension is the source of the prediction

The concrete fusion plan:

```
reward = w₁·z(s_L3) + w₂·z(s_L12) + w₃·z(s_graph)
```

…with weights swept. If the graph head adds even +1-2pp BG on top of v8, that's a real lift on a metric that's been stuck.

## The "winning pattern vs losing pattern" angle

Beyond a single score, the graph representation supports **interpretable pattern extraction**:
- Cluster the 836 songs' graph-feature vectors and find the "shape" of T5 songs vs T0 songs in feature space
- Identify **winner motifs** — recurring 3-note or 4-note patterns that appear disproportionately in T5 songs
- Identify **loser motifs** — patterns that mark T0 songs (e.g., long runs of unisons)
- For any new song, surface "this phrase looks 80% like winner-motif-3 (the soaring chorus type)"

This is the kind of explainability v8 alone cannot provide. v8 says "this song is 0.7." The graph could say "the chorus uses chord-drone-style sustains [winner trait #3], the verse follows a stepwise-descent pattern [winner trait #7], and the bridge has an octave leap [winner trait #1]." That's what makes the approach interpretable to non-ML musicians and useful for both the RM AND for cofounder-facing tools.

## What we DON'T know yet (honest)

- Whether a GNN trained on these graphs actually learns useful features. We have a working visualization and feature extraction pipeline; we have NOT trained a GNN yet.
- Whether the +graph_head fusion lifts v8's BG ceiling, or sits at weight 0 like the six prior melody attempts.
- Whether the structural patterns we see in Cruel Summer vs But Daddy generalize across 836 songs or are anecdotal.
- Whether the "winner motifs" actually exist in a clusterable way or whether T5 songs differ from each other more than from T0 songs.

## Next steps (concrete)

1. **Extend graph extraction to all 836 songs** (free, ~half-day of local compute). Dump per-song feature CSV. Group by tier. See which features actually separate at the population level — not just our 2-song case study.
2. **Train a small GNN** (~$5 on Modal). Heterogeneous graph, 2-3 message-passing layers, ~50k params. Predict tier from graph. Cross-validate.
3. **Fuse with v8** via the existing z-norm + weighted-sum recipe. Sweep the weight of the graph head. Report new BG and g5.
4. **If +graph adds lift**, build the interpretability layer: phrase-level winner-motif typing.
5. **If +graph doesn't add lift**, the graph rep is still useful for the cofounder-facing explainability tooling (showing WHY a song is judged a certain way).

## How to run the current proof-of-concept

```bash
cd erised/melody_graph

# Build the v2 graphs for Cruel Summer + But Daddy and produce CSVs + summary
python build_graph_v2.py

# Render a single phrase from a song with every note + interval labeled
python zoom_guide.py
```

Outputs land in `/tmp/graph_poc/output_v2/` (the scripts hardcode paths to the existing note + phrase tables from the CREPE + Whisper extraction pipeline, which lives outside this repo).

## Where this fits in the overall plan

The graph-melody RM is one of three open levers for breaking past v8's BG=70.72% ceiling:
1. **More hand-labels** (historically reliable, ~+2.25pp last time we added 234 labels)
2. **Lyrics signal** (untouched)
3. **Graph-based melody RM** (this proposal)

None are guaranteed, all are worth trying. The graph approach has the additional benefit of producing *interpretable* output — patterns we can show to non-ML musicians and to founders for product decisions.
