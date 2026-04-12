# DPO Training Notes for HeartMuLa

## Architecture Summary
- **HeartMuLa** (3B LLM) = the music generation brain. Outputs discrete audio tokens (8 codebooks x L frames). This is what DPO trains.
- **HeartCodec** (flow matching decoder) = converts tokens to audio waveforms. NOT trained by DPO. Experiment confirmed melody is fully determined by tokens (mel spectrogram correlation 0.96 across different random seeds).

## Training Versions

### v11 (baseline)
- `local_loss_weight=0.0` (only trained on codebook-0, ignored codebooks 1-7)
- `learning_rate=5e-6` (50x higher than paper's 1e-7)
- `epochs=6`, `beta=0.1`, `grad_accumulation_steps=4`
- Top 2 backbone layers only (patched from default 8)
- 200 manual preference pairs
- Results: loss 0.6931 -> 0.6813, accuracy 50% -> 82%, reward margins tiny (0.02)
- Verdict: learned something but very weak signal

### v12 (local loss enabled)
- `local_loss_weight=0.5` (trains on ALL 8 codebooks — both structure and acoustic detail)
- Everything else same as v11
- Status: training started 2026-04-12, running
- Log: /workspace/dpo_train_v12.log
- Checkpoints: /workspace/dpo_checkpoints_v12/

## Known Issues

### Cumulative noise problem
When training with MORE than 2 backbone layers (e.g., 8), the DPO'd model generates music that starts clean but becomes increasingly noisy over time (fine at 0:02, noisy by 0:40). This is caused by the model drifting too far from the base model's long-sequence coherence. The combination of too-high learning rate (5e-6) + too many trainable layers is the likely cause. If we lower LR to 1e-7, we may be able to safely train more layers.

### Weak learning signal
- 200 preference pairs is very few (paper used thousands)
- Reward margins are ~100x smaller than typical DPO
- Loss decrease of 0.01 is barely above random
- 82-88% accuracy is on TRAINING data (memorization), not generalization

## Paper vs Our Setup

| Aspect | Paper | Our Setup |
|---|---|---|
| Learning rate | 1e-7 | 5e-6 (50x too high) |
| Preference pairs | Thousands (automated) | 200 (manual) |
| GPUs | 8x A100 | 1x A100 |
| Trainable layers | Not specified | Top 2 of 28 |
| Local loss weight | Implicitly included | v11: 0.0, v12: 0.5 |
| Epochs | 3 | 6 |

## Next Steps for 20% Improvement Target

### Priority 1: More preference data (biggest impact)
The paper generated 4 songs per prompt, scored them with automated metrics (MUQ similarity, PER, AudioBox/SongEval), and picked best vs worst. We should do the same:
- Generate 4 songs per prompt for ~500 prompts = 2000 candidate songs
- Score automatically
- Build 500+ high-quality preference pairs with clear margins
- This requires GPU time for generation + evaluation models

### Priority 2: Lower learning rate to 1e-7
- Paper's value, prevents overshooting and the noise accumulation problem
- May also allow safely training 8 backbone layers instead of 2

### Priority 3: Local loss (done in v12)
- Already fixed: local_loss_weight=0.5

### Honest Limitations
- Even with all fixes, DPO on 200 manual pairs may not produce 20% measurable improvement
- More data is the single biggest bottleneck
- The cumulative noise problem may resurface if we increase trainable layers even with lower LR — need to test carefully
- "20% improvement in preference alignment" is hard to measure without a proper A/B evaluation framework (generate N songs from base vs DPO model, blind rate them)
- DPO doesn't change the flow matching decoder, so audio reconstruction quality stays the same
- With only 1 GPU, generating thousands of songs for automated preference data takes time (~5-10 seconds per song)

## Key Files
- Training scripts: /workspace/train_v11.py, /workspace/train_v12.py
- DPO trainer: /workspace/heartlib/erised/dpo/trainer.py
- DPO forward pass: /workspace/heartlib/erised/dpo/forward.py
- Preference DB: /workspace/heartlib/heartlib/dpo_preferences.db (200 pairs)
- Token files: /workspace/heartlib/heartlib/outputs/*_tokens.pt
- Model checkpoints: /workspace/heartlib/ckpt/
- HeartMuLa paper: /workspace/HeartMuLa.pdf
- Flow matching experiment: /workspace/flow_matching_experiment.py (proved tokens determine melody)
- Flow matching test outputs: /workspace/flow_test_run0-4.wav

## Google Cloud Storage
- Bucket: erised-dpo
- Path: gs://erised-dpo/workspace/
- Service account: runpod-upload@gen-lang-client-0191019282.iam.gserviceaccount.com
- Key file: DO NOT upload key.json to version control
- Storage cost: ~$2.40/month for 114GB (covered by $300 free credits)
