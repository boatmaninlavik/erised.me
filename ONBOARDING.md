# Erised RM + GRPO — Cofounder Onboarding

*Last updated 2026-05-23.*

## Goal in one sentence

Train a reward model that predicts **Sean's taste** in music, then use reinforcement learning to bend our HeartMuLa music generator toward outputs Sean would prefer.

## Where we honestly are right now

- **Best RM (for ranking real music): `rm_v8_crossmix_GOLDEN`.** BG=70.72%, g5=86.76%, g5_c=86.48% on 836-song hand-labeled CV. Solid as a *taste ranker*. Recipe in [`erised/rm/RECIPE.md`](erised/rm/RECIPE.md).
- **GRPO has failed 5 times in a row.** Every nonzero-learning-rate run ended with the policy producing pure noise, confirmed by Sean's listening on each.
- **Only lr=0 produces music.** A diagnostic with lr=0 (no actual weight updates) preserved music quality, confirming the bug is in the gradient/optimizer step, not in rollout pipeline or model loading.
- **An auxiliary RM (v9) was built but caused a different failure** (collapsed in-distribution resolution among HeartMuLa outputs). v9 is shelved.

## The current diagnosis (end of day 2026-05-23)

The pattern across 5 GRPO runs:

| Run | Reward formula | Reward Δ | Audio verdict |
|---|---|---|---|
| v8 fused lr=2e-6 | fused on v8 | +0.29 | noise |
| v8 fused lr=5e-6 | fused on v8 | +0.38 | noise |
| v9 fused | fused on v9 | −2.0 | noise |
| v9 pess_min | min of two heads on v9 | −0.03 | noise |
| Stage 1 mcd_uwo | mcd_uwo on v8 + drone guard | −0.39 | noise |
| **lr=0 diagnostic** | fused on v8 | (no update) | **music** |

**Every nonzero-lr run produces noise regardless of reward formula or KL strength.** This is much more consistent with a training-side gradient bug than reward hacking — reward hacking would cause different reward functions to push the policy toward different garbage modes; we're getting the SAME garbage from all of them.

### Two suspect bugs identified (both unverified)

1. **K3 KL estimator overflows on long sequences.** [`modal_grpo_v3_par.py:277-278`](file:///Users/thuscodedzara/modal_batch/modal_grpo_v3_par.py#L277-L278) computes `exp(sum_log_ratio)` over 6000 tokens (750 frames × 8 codebooks). For moderate policy divergence, `log_ratio` can reach 60+, and `exp(60) ≈ 10^26`. Gradient clipping bounds the magnitude but the gradient *direction* gets dominated by KL → oscillates wildly → corrupts weights. The fix: per-token K3 then average.
2. **Full-model fine-tuning of 3.3B parameters with lr=5e-6 is unusually aggressive.** Standard LLM RLHF practice uses lr=1e-7 to 5e-7 OR uses LoRA. We're at 10–50× standard lr AND touching every parameter.

A combined-fix experiment (lr=5e-7 + KL fix in per-token form) is running detached on workspace `erised3` as of evening 2026-05-23.

## Part 1 — The Reward Model

### What it does

Input: a song. Output: one scalar = predicted Sean-taste score on a learned scale.

### Architecture (frozen MuQ + two small probes)

```
audio (full song)
  ├──> MuQ-large L3, 30s-window mean-pool → MLPHead → s_L3
  └──> MuQ-large L12 (first 90s), 30s-window mean-pool → MLPHead → s_L12
final = 0.5·z(s_L3) + 0.5·z(s_L12)
```

- MuQ encoder is frozen (310M params from [arxiv:2501.01108](https://arxiv.org/abs/2501.01108)).
- Two layers probed: L3 (acoustic, full song) + L12 (semantic, first 90s only). Asymmetric temporal extent is part of the SOTA recipe.
- 10-seed ensemble per layer (averaged) for stability.
- See [`erised/rm/RECIPE.md`](erised/rm/RECIPE.md) for the full training recipe.

### Performance (on 836 hand-labeled songs)

| Metric | Score | Meaning |
|---|---|---|
| BG | 70.72% | T0–T1 vs T3–T5 accuracy on held-out real songs |
| g5 | 86.76% | top-5 ranking quality, cross-validated |
| g5_c | 86.48% | same but artist-out-of-distribution (real-world proxy) |

These are stable; six attempts to add melody-specific signals have all failed (got weight ≈0 in fusion). The 30% gap to perfect is the open problem.

### v8's OOD blind spot (relevant for GRPO)

v8 was trained only on real music. When asked to score *non-music* inputs (synthetic sine drones, chord drones), it gives them masterpiece-level scores. Stress test:

| input | z_L12 (v8) | what it means |
|---|---|---|
| Real Cruel Summer (T5) | +0.18 | correctly above mean |
| chord drone (3 sines stacked) | **+1.26** | scores HIGHER than any real T5 (!) |
| sine 440Hz | +0.36 | rewarded |
| white noise | −0.25 | weakly rejected |

This is the OOD vulnerability earlier GRPO runs exploited (the v8 hack at +0.29, +0.38) before we knew about the gradient bug.

### v9 attempt — shelved

Augmented v8 training set with 200 hard-negative non-music examples. Fixed the OOD stress test (all formulas pass) BUT collapsed in-distribution resolution — every HeartMuLa output scored in a tight ~−0.5 band, killing the gradient signal. Trade-off didn't work.

## Part 2 — GRPO (Reinforcement Learning on the Generator)

### Implementation

DeepSeek-style GRPO with μ=1 (one gradient step per rollout batch, mathematically REINFORCE + group baseline + KL anchor). Algorithm:

```
For each prompt: sample G=8 rollouts from current policy
Score each rollout with the RM, get reward r_i
Advantage A_i = (r_i − mean(group)) / (std(group) + ε)
Loss = −A_i · log_prob(rollout_i) + β · KL(policy || frozen_base)
Backprop, gradient-clip to norm 1.0, AdamW step, save policy_ep+1.pt
```

Parallel architecture: 7 H100 containers fan-out via `Modal .starmap()` for rollouts, single H100 for the gradient step. **Each individual run uses ~10 concurrent H100s** (7 train + 3 eval). Two runs in parallel = ~20 concurrent H100s — this is why a single workspace can burn fast.

### Failed runs (5 to date)

See diagnosis table above. The reward-hacking-vs-gradient-bug question was resolved in favor of "gradient bug" once lr=0 was confirmed to preserve music while every nonzero-lr run produces noise regardless of reward.

### What's running now (evening 2026-05-23)

A combined-fix run on **erised3**: lr=5e-7 (10× smaller than failing 5e-6) + K3 KL per-token fix + β_KL=0.20 + 2 epochs + G=8 + 60s clips. Expected ~$10-12 of erised3's $30 free credit. Verdict pending Sean's listening.

## The bottleneck (independent of GRPO)

The RM's BG=70.72% on real songs has been stuck. We've ruled out:
- Pooling layer (attn vs mean tied)
- Encoder fine-tuning via reconstruction (3 autoencoder LoRA attempts, all flat or negative)
- Pitch information missing (overfit probe proved pitch IS in MuQ frames, corr 0.998)
- Six melody-specific RM heads (all got weight ≈0 in fusion; the latest's attention pool collapsed to uniform = supervision starvation)

What's left to try:
- **More hand-labels** — historically the most reliable lever (+602 → +836 gave +2.25pp g5_c)
- **Lyrics signal** — untouched
- **Production-quality features** (vocal effort, mix dynamics) — untouched
- **Graph-based melody RM** — proof-of-concept built today in [`erised/melody_graph/`](erised/melody_graph/); cross-song analysis pending

See [`erised/melody_graph/REPRESENTING_SONGS.md`](erised/melody_graph/REPRESENTING_SONGS.md) for the graph-RM proposal in detail.

## Modal infrastructure

- **Modal workspaces** (each has $30 free credit; Sean is on Starter plan):
  - `sean-46797`: $55.27 spent ($25.27 over — DO NOT use)
  - `erised1`: exhausted
  - `erised2`: ~$20-25 spent, ~$5-10 left
  - `erised3`: ~$12 spent (combined-fix run in flight)
  - `boatmaninlavik`: depleted, retired
- **GCS** (`gs://erised-dpo/`): source of truth for labels, audio, RM checkpoints, GRPO run artifacts. Service-account key at `~/Desktop/migration/modal-grpo-key.json` (sensitive, never commit).
- **Code repo** (this one): RM training code + recipe + graph PoC. GRPO orchestrator (`modal_grpo_v3_par.py`) lives in `~/modal_batch/` privately — it produced broken policies and is under active debugging, not pushed.

## How our RM differs from the MuQ paper ([arxiv:2501.01108](https://arxiv.org/abs/2501.01108))

What's the same: the MuQ encoder itself (frozen, unmodified), the frozen-encoder + probe paradigm, time-mean pooling.

What's different (our additions):
- Probe head: 2-layer MLP with heavy dropout (vs paper's single linear) — for our 836-label small-data regime
- Layer choice: **L3 + L12 fused**, not single last layer — independently +5pp
- Asymmetric temporal extent: L3 full song, L12 first 90s — directional, inverse doesn't win
- 10-seed ensemble per head, +1pp stability
- Score-level fusion via z-norm + weighted sum, not feature-level

The encoder is theirs; everything above it is ours. SOTA comes from compounding these small deltas.

## What you can do day-one

1. **Listen to a few before/after MP3s from the GRPO runs** in `gs://erised-dpo/grpo_runs/` — confirms the noise verdict yourself.
2. **Read [`erised/rm/RECIPE.md`](erised/rm/RECIPE.md)** for the full RM training recipe.
3. **Read [`erised/melody_graph/REPRESENTING_SONGS.md`](erised/melody_graph/REPRESENTING_SONGS.md)** for the graph-based melody-RM proposal — the most interesting next-direction work.
4. **Run a single RM scoring**: `python erised/rm/modal_rm_v8.py score --audio path/to/song.wav` (requires Modal account + GCS access).

## Honest one-paragraph summary

We have a strong real-music ranking RM (v8, BG=70.72%) and a generator (HeartMuLa) that produces music. Combining them via GRPO has failed five times in a row — every nonzero-learning-rate run produces noise, while lr=0 preserves music. The leading hypothesis is a gradient-side numerical bug (KL math overflow + lr too aggressive for full-model fine-tune); a combined-fix experiment is running tonight. Independently, we're investigating whether a graph-based representation of song melodies can surface taste-relevant patterns the audio RM misses (proof-of-concept already shows T5 vs T0 songs differ in interval-quality distribution and within-phrase pitch deviation in a measurable, interpretable way). The 30% gap to perfect taste prediction is the long-term lever; getting a single GRPO run to produce music is the immediate one.
