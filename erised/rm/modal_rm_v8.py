"""rm_v8_crossmix — deployment-ready RM using the cross-mix recipe.

Architecture:
  audio  ->  MuQ-base L3  (extracted over FULL song, 30s windows averaged)  ->  head_L3   ->  s_L3
        ->  MuQ-base L12 (extracted over FIRST 90s,   3 x 30s avg)         ->  head_L12  ->  s_L12

  final_score = 0.5 * z(s_L3) + 0.5 * z(s_L12)
  where z(.) uses the reference mean/std computed on the 836-song training set.

Two entrypoints:
  build_and_save: train K=10 heads on ALL 836 songs (no held-out), save checkpoint to /data/ckpt/rm_v8_crossmix.pt
  score_songs:    given list of song basenames in /data/score_input/, output fused score per song
"""
import modal, pathlib

app = modal.App("erised-rm-v8")
vol = modal.Volume.from_name("erised-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "numpy==2.0.2",
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "librosa==0.10.2",
        "soundfile",
        "muq==0.1.0",
        "transformers==4.57.0",
        "huggingface_hub",
    )
    .add_local_file("./meta_ts_6tier.json", remote_path="/root/ts_6tier.json")
)


def _build_mlp_head(d_in=1024, hidden=64, dropout=0.7):
    import torch.nn as nn
    class MLPHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(
                nn.LayerNorm(d_in), nn.Dropout(dropout),
                nn.Linear(d_in, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )
        def forward(self, x): return self.head(x).squeeze(-1)
    return MLPHead()


def _extract_layer_features(layer_ix, audio_tensor_per_song, m, win_samples):
    """audio_tensor_per_song: list of [T] tensors (variable length). Returns [N, d_out] L2-normed."""
    import torch, torch.nn.functional as F
    feats_out = []
    for wav in audio_tensor_per_song:
        L = wav.shape[0]
        n_full = max(1, L // win_samples)
        if L < win_samples:
            pad = torch.zeros(win_samples, dtype=wav.dtype, device=wav.device)
            pad[:L] = wav; wav = pad; n_full = 1
        chunks = torch.stack([wav[w*win_samples:(w+1)*win_samples] for w in range(n_full)], dim=0)
        with torch.no_grad():
            out = m(chunks, output_hidden_states=True)
            pooled = out.hidden_states[layer_ix].mean(dim=1).mean(dim=0)
            feats_out.append(F.normalize(pooled, dim=-1))
    return torch.stack(feats_out, dim=0)


@app.function(image=image, gpu="A100", volumes={"/data": vol}, timeout=21600)
def build_and_save(k_ens: int = 10, epochs: int = 300, batch: int = 256,
                   lr_head: float = 3.33e-4, wd: float = 1e-2,
                   l3_max_song_sec: int = 360, l12_seconds: int = 90):
    """Train K=10 head_L3 + K=10 head_L12 on ALL 836 songs (no held-out).
    Save heads + z-norm stats (computed from CV predictions we already have).
    """
    import json, pathlib, time
    import numpy as np
    import torch, torch.nn as nn, torch.nn.functional as F
    import librosa
    from muq import MuQ

    META_P = pathlib.Path("/root/ts_6tier.json")
    AUDIO_FULL = pathlib.Path("/data/audio_full")
    AUDIO_CLIP = pathlib.Path("/data/audio")
    CKPT_DIR = pathlib.Path("/data/ckpt"); CKPT_DIR.mkdir(exist_ok=True)
    GOOD = {3,4,5}; BAD = {0,1}; N_TIERS = 6
    sr = 24000
    win_samples = 30 * sr

    entries = json.loads(META_P.read_text())
    by_id = {e["id"]: e for e in entries}
    # Common ids: have both full song and 90s clip
    have = sorted({p.stem for p in AUDIO_FULL.glob("*.m4a")} &
                  {p.stem for p in AUDIO_CLIP.glob("*.m4a")} & set(by_id))
    n = len(have)
    y_list = [by_id[s]["tier_rank"] for s in have]
    y = torch.tensor(y_list, dtype=torch.long)
    print(f"[rm_v8 build] training on n={n} songs (have BOTH full song and 90s clip)")

    print("[rm_v8 build] loading MuQ-base ...")
    m = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").cuda().eval()
    for p in m.parameters(): p.requires_grad = False

    # --- Extract L3 features over FULL SONG (up to l3_max_song_sec) ---
    print("[rm_v8 build] extracting L3 features over full songs ...")
    feats_L3 = torch.zeros(n, 1024, dtype=torch.float32, device="cuda")
    t0 = time.time()
    for i, sid in enumerate(have):
        wav, _ = librosa.load(str(AUDIO_FULL / f"{sid}.m4a"), sr=sr, mono=True, duration=l3_max_song_sec)
        wav_t = torch.from_numpy(np.ascontiguousarray(wav)).cuda()
        L = wav_t.shape[0]; n_full = max(1, L // win_samples)
        if L < win_samples:
            pad = torch.zeros(win_samples, device=wav_t.device); pad[:L] = wav_t; wav_t = pad; n_full = 1
        chunks = torch.stack([wav_t[w*win_samples:(w+1)*win_samples] for w in range(n_full)], dim=0)
        with torch.no_grad():
            out = m(chunks, output_hidden_states=True)
            pooled = out.hidden_states[3].mean(dim=1).mean(dim=0)
            feats_L3[i] = F.normalize(pooled, dim=-1)
        if (i+1) % 100 == 0:
            print(f"  L3 {i+1}/{n}  elapsed={(time.time()-t0)/60:.1f}min")
    print(f"[rm_v8 build] L3 features done in {(time.time()-t0)/60:.1f}min")

    # --- Extract L12 features over FIRST 90s (3 x 30s averaged) ---
    print("[rm_v8 build] extracting L12 features over first 90s ...")
    feats_L12 = torch.zeros(n, 1024, dtype=torch.float32, device="cuda")
    t0 = time.time()
    for i, sid in enumerate(have):
        wav, _ = librosa.load(str(AUDIO_CLIP / f"{sid}.m4a"), sr=sr, mono=True, duration=l12_seconds)
        wav_t = torch.from_numpy(np.ascontiguousarray(wav)).cuda()
        L = wav_t.shape[0]; n_full = max(1, L // win_samples)
        if L < win_samples:
            pad = torch.zeros(win_samples, device=wav_t.device); pad[:L] = wav_t; wav_t = pad; n_full = 1
        chunks = torch.stack([wav_t[w*win_samples:(w+1)*win_samples] for w in range(n_full)], dim=0)
        with torch.no_grad():
            out = m(chunks, output_hidden_states=True)
            pooled = out.hidden_states[12].mean(dim=1).mean(dim=0)
            feats_L12[i] = F.normalize(pooled, dim=-1)
        if (i+1) % 100 == 0:
            print(f"  L12 {i+1}/{n}  elapsed={(time.time()-t0)/60:.1f}min")
    print(f"[rm_v8 build] L12 features done in {(time.time()-t0)/60:.1f}min")

    del m
    torch.cuda.empty_cache()

    # --- Train K=10 heads per layer on ALL data, no CV ---
    def by_tier(y_arr):
        return {t: (y_arr == t).nonzero(as_tuple=True)[0]
                for t in range(N_TIERS) if (y_arr == t).any()}

    def train_one(feats, seed):
        torch.manual_seed(seed)
        m2 = _build_mlp_head(d_in=1024).cuda()
        opt = torch.optim.AdamW(m2.parameters(), lr=lr_head, weight_decay=wd)
        bt = by_tier(y); tiers_desc = sorted(bt.keys(), reverse=True); K = len(tiers_desc)
        bt_cu = {t: idx.cuda() for t, idx in bt.items()}
        g = torch.Generator().manual_seed(seed); m2.train()
        for ep in range(epochs):
            opt.zero_grad()
            sampled = [bt_cu[t][torch.randint(0, len(bt_cu[t]), (batch,), generator=g).cuda()] for t in tiers_desc]
            flat = torch.stack(sampled, dim=1).flatten()
            scores = m2(feats[flat]).view(batch, K)
            pl = 0.0
            for k in range(K - 1):
                pl = pl + (-scores[:, k] + torch.logsumexp(scores[:, k:], dim=1)).mean()
            (pl / (K - 1)).backward(); opt.step()
        m2.eval(); return m2

    print(f"[rm_v8 build] training K={k_ens} heads on L3 features ...")
    heads_L3 = [train_one(feats_L3, seed=s) for s in range(k_ens)]
    print(f"[rm_v8 build] training K={k_ens} heads on L12 features ...")
    heads_L12 = [train_one(feats_L12, seed=s) for s in range(k_ens)]

    # --- Compute reference scores (avg across K heads) for z-norm stats ---
    with torch.no_grad():
        ref_L3 = torch.stack([h(feats_L3) for h in heads_L3], dim=0).mean(dim=0).cpu().numpy()
        ref_L12 = torch.stack([h(feats_L12) for h in heads_L12], dim=0).mean(dim=0).cpu().numpy()
    zn_L3 = {"mean": float(ref_L3.mean()), "std": float(ref_L3.std())}
    zn_L12 = {"mean": float(ref_L12.mean()), "std": float(ref_L12.std())}
    print(f"[rm_v8 build] reference z-norm  L3: mean={zn_L3['mean']:.3f} std={zn_L3['std']:.3f}")
    print(f"[rm_v8 build] reference z-norm L12: mean={zn_L12['mean']:.3f} std={zn_L12['std']:.3f}")

    ckpt = {
        "recipe": "rm_v8_crossmix",
        "version": 1,
        "description": "Cross-mix: L3 from full song + L12 from first 90s, score-fused @ w_L3=0.5",
        "config": dict(epochs=epochs, k_ens=k_ens, batch=batch, lr_head=lr_head, wd=wd,
                       l3_max_song_sec=l3_max_song_sec, l12_seconds=l12_seconds,
                       win_sec=30, sr=sr, mlp_hidden=64, mlp_dropout=0.7,
                       encoder="OpenMuQ/MuQ-large-msd-iter",
                       layer_l3=3, layer_l12=12,
                       fusion="0.5 * z(s_L3) + 0.5 * z(s_L12)"),
        "heads_L3": [h.state_dict() for h in heads_L3],
        "heads_L12": [h.state_dict() for h in heads_L12],
        "zn_L3": zn_L3, "zn_L12": zn_L12,
        "n_train": n, "ids_train": have,
        "ref_scores_L3": ref_L3.tolist(),
        "ref_scores_L12": ref_L12.tolist(),
    }
    torch.save(ckpt, "/data/ckpt/rm_v8_crossmix.pt")
    vol.commit()
    print(f"[rm_v8 build] saved /data/ckpt/rm_v8_crossmix.pt")


@app.function(image=image, gpu="A100", volumes={"/data": vol}, timeout=3600)
def score_songs(song_basenames: list[str], score_input_dir: str = "/data/score_input"):
    """Score songs in /data/score_input/{basename} using saved rm_v8_crossmix.
    Returns a dict: basename -> {raw_L3, raw_L12, z_L3, z_L12, fused, percentile_among_train}.
    """
    import json, pathlib
    import numpy as np
    import torch, torch.nn.functional as F
    import librosa
    from muq import MuQ

    ckpt = torch.load("/data/ckpt/rm_v8_crossmix.pt", weights_only=False, map_location="cuda")
    cfg = ckpt["config"]
    sr = cfg["sr"]; win_samples = cfg["win_sec"] * sr
    l3_dur = cfg["l3_max_song_sec"]; l12_dur = cfg["l12_seconds"]

    # Rebuild heads
    heads_L3 = []
    for sd in ckpt["heads_L3"]:
        h = _build_mlp_head(d_in=1024, hidden=cfg["mlp_hidden"], dropout=cfg["mlp_dropout"]).cuda()
        h.load_state_dict(sd); h.eval(); heads_L3.append(h)
    heads_L12 = []
    for sd in ckpt["heads_L12"]:
        h = _build_mlp_head(d_in=1024, hidden=cfg["mlp_hidden"], dropout=cfg["mlp_dropout"]).cuda()
        h.load_state_dict(sd); h.eval(); heads_L12.append(h)

    print("[score] loading MuQ-base ...")
    m = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").cuda().eval()
    for p in m.parameters(): p.requires_grad = False

    in_dir = pathlib.Path(score_input_dir)
    results = {}
    for basename in song_basenames:
        # find file (any audio extension)
        cands = list(in_dir.glob(f"{basename}.*"))
        if not cands:
            print(f"  [score] {basename}: FILE NOT FOUND in {in_dir}")
            results[basename] = {"error": "file not found"}; continue
        path = cands[0]
        print(f"[score] processing {basename} -> {path.name}")

        # L3 over full song
        wav, _ = librosa.load(str(path), sr=sr, mono=True, duration=l3_dur)
        wav_t = torch.from_numpy(np.ascontiguousarray(wav)).cuda()
        L = wav_t.shape[0]; n_full = max(1, L // win_samples)
        if L < win_samples:
            pad = torch.zeros(win_samples, device=wav_t.device); pad[:L] = wav_t; wav_t = pad; n_full = 1
        chunks = torch.stack([wav_t[w*win_samples:(w+1)*win_samples] for w in range(n_full)], dim=0)
        with torch.no_grad():
            out = m(chunks, output_hidden_states=True)
            f_L3 = F.normalize(out.hidden_states[3].mean(dim=1).mean(dim=0), dim=-1).unsqueeze(0)
        n_windows_L3 = n_full

        # L12 over first 90s
        wav12, _ = librosa.load(str(path), sr=sr, mono=True, duration=l12_dur)
        wav12_t = torch.from_numpy(np.ascontiguousarray(wav12)).cuda()
        L12 = wav12_t.shape[0]; n_full12 = max(1, L12 // win_samples)
        if L12 < win_samples:
            pad = torch.zeros(win_samples, device=wav12_t.device); pad[:L12] = wav12_t; wav12_t = pad; n_full12 = 1
        chunks12 = torch.stack([wav12_t[w*win_samples:(w+1)*win_samples] for w in range(n_full12)], dim=0)
        with torch.no_grad():
            out12 = m(chunks12, output_hidden_states=True)
            f_L12 = F.normalize(out12.hidden_states[12].mean(dim=1).mean(dim=0), dim=-1).unsqueeze(0)

        # Run heads
        with torch.no_grad():
            s_L3 = float(torch.stack([h(f_L3) for h in heads_L3], dim=0).mean(dim=0).item())
            s_L12 = float(torch.stack([h(f_L12) for h in heads_L12], dim=0).mean(dim=0).item())

        # Z-norm against training distribution
        z_L3 = (s_L3 - ckpt["zn_L3"]["mean"]) / ckpt["zn_L3"]["std"]
        z_L12 = (s_L12 - ckpt["zn_L12"]["mean"]) / ckpt["zn_L12"]["std"]
        fused = 0.5 * z_L3 + 0.5 * z_L12

        # Percentile of fused score against training set's fused scores
        ref_L3 = np.array(ckpt["ref_scores_L3"]); ref_L12 = np.array(ckpt["ref_scores_L12"])
        ref_zL3 = (ref_L3 - ckpt["zn_L3"]["mean"]) / ckpt["zn_L3"]["std"]
        ref_zL12 = (ref_L12 - ckpt["zn_L12"]["mean"]) / ckpt["zn_L12"]["std"]
        ref_fused = 0.5 * ref_zL3 + 0.5 * ref_zL12
        pct = float((ref_fused < fused).mean() * 100)

        # Which training songs have nearest fused score?
        ref_diff = np.abs(ref_fused - fused)
        nearest_idx = np.argsort(ref_diff)[:3]
        nearest = [(ckpt["ids_train"][i], float(ref_fused[i])) for i in nearest_idx]

        results[basename] = {
            "raw_L3": s_L3, "raw_L12": s_L12,
            "z_L3": z_L3, "z_L12": z_L12, "fused": fused,
            "percentile_vs_training": pct,
            "n_windows_L3": int(n_windows_L3), "n_windows_L12": int(n_full12),
            "nearest_training_songs": nearest,
        }
        print(f"  [{basename}]  fused={fused:+.3f}  z_L3={z_L3:+.2f}  z_L12={z_L12:+.2f}  "
              f"percentile={pct:.1f}%  L3_windows={n_windows_L3}")

    # Save results
    out_path = pathlib.Path("/data/results/rm_v8_user_scores.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    vol.commit()
    print(f"[score] saved {out_path}")
    return results


@app.local_entrypoint()
def main(do: str = "info", songs: str = ""):
    if do == "build":
        build_and_save.remote()
    elif do == "score":
        names = [s.strip() for s in songs.split(",") if s.strip()]
        print(f"scoring: {names}")
        res = score_songs.remote(song_basenames=names)
        print("\n=== RESULTS ===")
        for name, r in res.items():
            print(f"{name}: {r}")
    else:
        print("Usage: --do build  |  --do score --songs 'name1,name2,...'")
