"""Evaluate stage-1 SFT (run sft_v1) against original Shao — workspace erised8.

Three checks:
  check A     surprise on ALL 2,315 test songs (artists never trained on), for original Shao and every
              saved copy (after 5,200 / 10,400 / 15,600 songs, and final). Lower = expects Suno songs better.
  forgetting  the same surprise score on 200 of Sean's rated REAL (non-Suno) songs. If fine-tuned Shao gets
              much MORE surprised by normal music, it is losing what it knew.
  check B     original and final Shao each write a song from the same 30 English test prompts (same seed,
              Shao's default sampling). Each song is scored on: lyrics accuracy (Whisper transcript vs lyrics),
              style match (MuQ-MuLan text-audio similarity), musical quality (SongEval, 5 scores 1-5),
              audio quality (AudioBox, 4 scores 1-10). Compared prompt by prompt.

Everything runs in the cloud from one coordinator (laptop can sleep):
  MODAL_PROFILE=erised8 modal run --detach shao_dpo/eval_sft.py
Progress: B2 erised-sft/sft_eval/sft_v1/progress.jsonl ; final numbers: sft_eval/sft_v1/report.json
"""
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
RUN = "sft_v1"
B2_BUCKET, B2_ENDPOINT = "erised-sft", "https://s3.us-west-004.backblazeb2.com"
OUT = f"sft_eval/{RUN}/"
TAGS = ["original", "step_00650", "step_01300", "step_01950", "final"]
MPS_REPO, SHAO_REPO = "Vinpolar/Khala-MusicGeneration-v1.0-MPS", "liujiafeng/Shao-MusicGeneration-v1.0"
SR, FRAME, EOS = 44100, 2048, 128001
N_PROMPTS, N_FORGET = 30, 200

app = modal.App("shao-sft-eval")
vol = modal.Volume.from_name("shao-sft", create_if_missing=True)
b2_secret = modal.Secret.from_name("b2-key")
gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", "numpy<2", "omegaconf", "einops", "lightning", "boto3", "huggingface_hub",
                 "safetensors", "transformers==4.44.2", "soundfile", "google-cloud-storage")
    .add_local_dir(HERE.parent / "khala_runtime", remote_path="/root/khala")
    .add_local_file(HERE / "dpo_common.py", "/root/dpo_common.py")
)
score_image = (  # same pins as modal_songeval_judge.py (muq 0.1.0 breaks on newer transformers)
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.7.0", "torchaudio==2.7.0", "muq==0.1.0", "transformers==4.57.0", "librosa==0.11.0",
                 "hydra-core==1.3.2", "omegaconf", "safetensors", "einops", "boto3", "openai-whisper", "jiwer")
    .run_commands("git clone --depth 1 https://github.com/ASLP-lab/SongEval /root/SongEval",
                  "test $(stat -c %s /root/SongEval/ckpt/model.safetensors) -gt 50000000")
)
audiobox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", "torchaudio==2.4.1", "audiobox_aesthetics", "boto3", "requests")
)


def _b2():
    import boto3
    return boto3.client("s3", endpoint_url=B2_ENDPOINT)


def _put(key, obj):
    _b2().put_object(Bucket=B2_BUCKET, Key=OUT + key, Body=json.dumps(obj, indent=1, ensure_ascii=False).encode())


def _progress(event, **kw):
    """Append one line to the progress log on B2 (read-modify-write; only the coordinator calls this)."""
    b2 = _b2()
    try:
        old = b2.get_object(Bucket=B2_BUCKET, Key=OUT + "progress.jsonl")["Body"].read().decode()
    except Exception:
        old = ""
    row = json.dumps({"time": time.strftime("%H:%M:%S"), "event": event, **kw}, ensure_ascii=False)
    b2.put_object(Bucket=B2_BUCKET, Key=OUT + "progress.jsonl", Body=(old + row + "\n").encode())
    print(row, flush=True)


def _weights_path(tag):
    return "/data/weights/khala_backbone.safetensors" if tag == "original" else f"/data/ckpt/{tag}.safetensors"


# ------------------------------------------------------------------ setup
@app.function(image=gen_image, cpu=4, memory=32768, volumes={"/data": vol}, secrets=[b2_secret], timeout=3600)
def weights():
    import os, shutil
    import torch
    from huggingface_hub import hf_hub_download
    b2 = _b2()
    Path("/data/ckpt").mkdir(parents=True, exist_ok=True)
    for tag in TAGS[1:]:
        if not Path(f"/data/ckpt/{tag}.safetensors").exists():
            b2.download_file(B2_BUCKET, f"sft_runs/{RUN}/{tag}.safetensors", f"/data/ckpt/{tag}.safetensors")
    for f in ("khala_superres.safetensors", "superres_megatron_args.json"):
        if not Path(f"/data/weights/{f}").exists():
            shutil.copyfile(os.path.realpath(hf_hub_download(MPS_REPO, f, cache_dir="/tmp/hf")), f"/data/weights/{f}")
    if not Path("/data/codec/generator.pt").exists():
        sd = torch.load(hf_hub_download(SHAO_REPO, "dac_rvq_2490000.ckpt", cache_dir="/tmp/hf"), map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        Path("/data/codec").mkdir(parents=True, exist_ok=True)
        torch.save({k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}, "/data/codec/generator.pt")
    vol.commit()


@app.function(image=gen_image, cpu=2, memory=16384, volumes={"/data": vol}, secrets=[b2_secret], timeout=1800)
def pick_prompts():
    """30 English vocal test prompts, one per artist; lyrics cut to the first ~50 words (a ~1-minute song)."""
    import random, re, sys
    sys.path[:0] = ["/root", "/root/khala"]
    import dpo_common as dc
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in _b2().get_object(Bucket=B2_BUCKET, Key="shao_tokens/v1/manifest.jsonl")["Body"].read().decode().split("\n") if l.strip()]
    ascii_share = lambda t: sum(c.isascii() for c in t) / max(1, len(t))
    ok = [r for r in rows if r["split"] == "test" and not r.get("instrumental") and (r.get("style_prompt") or "").strip()
          and ascii_share(r.get("lyrics") or "") > 0.97 and ascii_share(r["style_prompt"]) > 0.95
          and len(re.findall(r"[A-Za-z']+", re.sub(r"\[[^\]]*\]", " ", r.get("lyrics") or ""))) >= 60]
    random.Random(0).shuffle(ok)
    tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
    picked, artists = [], set()
    for r in ok:
        if r["user_id"] in artists:
            continue
        lines, words = [], 0
        for line in (r["lyrics"] or "").strip().split("\n"):
            lines.append(line)
            words += len(re.findall(r"[A-Za-z']+", re.sub(r"\[[^\]]*\]", " ", line)))
            if words >= 50:
                break
        lyrics = "\n".join(lines).strip()
        style = r["style_prompt"].strip()
        ids = tok.encode(dc.build_prompt_text(style, lyrics, False, 1), add_special_tokens=False)
        picked.append({"pid": len(picked), "song_id": r["id"], "title": r.get("title"), "handle": r.get("handle"),
                       "style": style, "lyrics": lyrics, "prompt_ids": ids})
        artists.add(r["user_id"])
        if len(picked) == N_PROMPTS:
            break
    Path("/data/eval").mkdir(parents=True, exist_ok=True)
    json.dump(picked, open("/data/eval/check_b_prompts.json", "w"))
    vol.commit()
    _put("check_b_prompts.json", [{k: v for k, v in p.items() if k != "prompt_ids"} for p in picked])
    return len(picked)


def _load_codec():
    import sys
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, "/root/khala/models/Decoder")
    from dac_rvq import DacRVQ
    dac = DacRVQ(OmegaConf.load("/root/khala/models/Decoder/dac_rvq_1024_64_golden.yaml"))
    dac.load_state_dict(torch.load("/data/codec/generator.pt"))
    return dac.eval().cuda()


def _decode_file(path):
    import subprocess
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32).reshape(-1, 2).T.copy()


@app.function(image=gen_image, gpu="L4", memory=32768, volumes={"/data": vol},
              secrets=[b2_secret, modal.Secret.from_name("gcs-key")], timeout=3 * 3600)
def forget_prep():
    """Convert 200 of Sean's rated real songs (non-Suno) to layer-0/1 tokens, same encoding as the SFT data."""
    import os, random, tempfile
    import numpy as np, torch
    from google.cloud import storage
    from google.oauth2 import service_account
    if Path("/data/forget/meta.json").exists():
        return len(json.load(open("/data/forget/meta.json")))
    info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    bk = storage.Client(credentials=service_account.Credentials.from_service_account_info(info), project=info["project_id"]).bucket("erised-dpo")
    songs = json.loads(bk.blob("judge_lab/songs.json").download_as_bytes())
    random.Random(0).shuffle(songs)
    dac = _load_codec()
    codes_all, meta, c0 = [], [], 0
    for s in songs:
        if len(meta) == N_FORGET:
            break
        try:
            with tempfile.NamedTemporaryFile(suffix=Path(s["audio"]).suffix) as f:
                bk.blob(s["audio"][len("gs://erised-dpo/"):]).download_to_filename(f.name)
                wav = _decode_file(f.name)
        except Exception as e:
            print("skip", s["id"], e)
            continue
        T = -(-wav.shape[1] // FRAME)
        wav = np.pad(wav, ((0, 0), (0, T * FRAME - wav.shape[1])))
        core, ctx, parts = 640 * FRAME, 44 * FRAME, []
        with torch.no_grad():
            for st in range(0, T * FRAME, core):
                a, b = max(0, st - ctx), min(T * FRAME, st + core + ctx)
                c = dac.encode(torch.from_numpy(wav[:, a:b])[None].cuda())
                parts.append(c[:2, 0, (st - a) // FRAME:(st - a) // FRAME + (min(st + core, T * FRAME) - st) // FRAME])
        codes = torch.cat(parts, 1).T.cpu().numpy().astype(np.int16)
        codes_all.append(codes)
        meta.append({"id": s["id"], "title": s.get("title"), "artist": s.get("artist"), "tier": s.get("tier"),
                     "frames": int(len(codes)), "c0": c0})
        c0 += len(codes)
    Path("/data/forget").mkdir(parents=True, exist_ok=True)
    np.save("/data/forget/codes.npy", np.concatenate(codes_all))
    json.dump(meta, open("/data/forget/meta.json", "w"))
    vol.commit()
    return len(meta)


# ------------------------------------------------------------------ check A + forgetting
@app.function(image=gen_image, gpu="H100", memory=65536, volumes={"/data": vol}, secrets=[b2_secret], timeout=3 * 3600)
def check_a(tag: str):
    import sys
    import numpy as np, torch
    sys.path[:0] = ["/root", "/root/khala"]
    import dpo_common as dc
    from core.khala_runtime import load_vanilla_model
    from transformers import AutoTokenizer
    vol.reload()
    model = load_vanilla_model("backbone", _weights_path(tag), "/data/weights/backbone_megatron_args.json", "cuda", torch.float32).eval()
    head_w = model.lm_head.weight[:dc.REAL_VOCAB]

    @torch.no_grad()
    def surprise(prompt, codes):
        ids, a0 = dc.build_sequence(prompt, codes)
        ids = ids.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = dc.hidden_states(model, ids[None], use_checkpoint=False)[0]
            lp = torch.cat([dc._chunk_logp(h[a0 - 1:-1][s:s + 1024], head_w, ids[a0:][s:s + 1024]) for s in range(0, len(ids) - a0, 1024)])
        return float(-lp.mean())

    meta = json.load(open("/data/sft_data/meta.json"))
    codes_all = np.load("/data/sft_data/codes.npy", mmap_mode="r")
    prompts = np.load("/data/sft_data/prompts.npy", mmap_mode="r")
    out = {"test": {}, "forget": {}}
    t0 = time.time()
    for m in meta:
        if m["split"] == "test":
            out["test"][m["id"]] = surprise(prompts[m["p0"]:m["p0"] + m["plen"]].tolist(), np.array(codes_all[m["c0"]:m["c0"] + m["frames"]]))
    fmeta = json.load(open("/data/forget/meta.json"))
    fcodes = np.load("/data/forget/codes.npy", mmap_mode="r")
    tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
    for m in fmeta:   # real songs: no style/lyrics text available -> the same empty prompt for every model
        minutes = int(min(10, max(1, round(m["frames"] / (SR / FRAME) / 60))))
        p = tok.encode(dc.build_prompt_text("", "", False, minutes), add_special_tokens=False)
        out["forget"][m["id"]] = surprise(p, np.array(fcodes[m["c0"]:m["c0"] + m["frames"]]))
    out["minutes"] = round((time.time() - t0) / 60, 1)
    _put(f"check_a/{tag}.json", out)
    return tag


# ------------------------------------------------------------------ check B: generation
@app.function(image=gen_image, gpu="H100", memory=65536, volumes={"/data": vol}, secrets=[b2_secret], timeout=3 * 3600)
def generate(tag: str, pids: list):
    import io, subprocess, sys, tempfile
    import numpy as np, soundfile as sf, torch
    sys.path[:0] = ["/root", "/root/khala"]
    import dpo_common as dc
    from core.khala_runtime import load_vanilla_model, sample_backbone, generate_superres_projection
    vol.reload()
    prompts = {p["pid"]: p for p in json.load(open("/data/eval/check_b_prompts.json"))}
    bb = load_vanilla_model("backbone", _weights_path(tag), "/data/weights/backbone_megatron_args.json", "cuda", torch.bfloat16)
    sr_model = load_vanilla_model("superres", "/data/weights/khala_superres.safetensors", "/data/weights/superres_megatron_args.json", "cuda", torch.bfloat16)
    dac = _load_codec()
    b2 = _b2()
    stats = []
    for pid in pids:
        p = prompts[pid]
        seed = 1000 + pid                                     # same seed for both models
        torch.manual_seed(seed)
        t0 = time.time()
        max_tokens = round(dc.TOKENS_PER_MINUTE * (1 + 0.8))  # Shao's own limit for a 1-minute request
        out_ids = sample_backbone(bb, p["prompt_ids"], num_tokens=max_tokens, temperature=1.0, top_k=50, eos_id=EOS)
        ids = np.array(out_ids, dtype=np.int64)
        ids = ids[: len(ids) // 2 * 2].reshape(-1, 2)
        ok = (ids[:, 0] >= 128256) & (ids[:, 0] < 129280) & (ids[:, 1] >= 129280) & (ids[:, 1] < 130304)
        codes2 = np.clip(np.stack([ids[:, 0] - 128256, ids[:, 1] - 129280], 1), 0, 1023)
        # super-res + decoder, exactly as Shao's worker (see overfit_one_song.py)
        torch.manual_seed(seed)
        text = np.array(p["prompt_ids"][:2047] + [p["prompt_ids"][-1]] if len(p["prompt_ids"]) > 2048 else p["prompt_ids"], dtype=np.int64)
        audio_ids = dc.audio_ids_from_codes(codes2).numpy().reshape(-1, 2)
        L = len(text) + len(audio_ids)
        tokens = torch.full((L, 2), -1, dtype=torch.long)
        tokens[:len(text), 0] = torch.from_numpy(text)
        tokens[len(text):] = torch.from_numpy(audio_ids)
        with torch.inference_mode():
            full = generate_superres_projection(sr_model, tokens[None].cuda(), torch.zeros(1, 1, 1, L, dtype=torch.bool).cuda(),
                                                (tokens[:, -1] != -1).float()[None].cuda(), torch.arange(L)[None].cuda(),
                                                len(text), len(audio_ids), 10)
            codes64 = full[:, 0, :].cpu().numpy().T.astype(np.int64) - (128256 + np.arange(64) * 1024)[None, :]
            codes = torch.from_numpy(np.ascontiguousarray(codes64.T))[:, None].cuda()
            wave, chunk, overlap = None, 1920, 480
            for start in range(0, codes.shape[2], chunk - overlap):
                piece = dac.decode(codes[..., start:start + chunk]).cpu()
                if wave is None:
                    wave = piece
                else:
                    n = min(round(piece.shape[2] * overlap / codes[..., start:start + chunk].shape[2]), wave.shape[2], piece.shape[2])
                    fade = torch.linspace(1.0, 0.0, n).view(1, 1, -1)
                    wave = torch.cat([wave[..., :-n], wave[..., -n:] * fade + piece[..., :n] * (1 - fade), piece[..., n:]], 2)
                if start + chunk >= codes.shape[2]:
                    break
        with tempfile.TemporaryDirectory() as d:
            sf.write(f"{d}/a.wav", wave[0].numpy().T, SR, subtype="FLOAT")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/a.wav", "-b:a", "320k", f"{d}/a.mp3"], check=True)
            b2.put_object(Bucket=B2_BUCKET, Key=f"{OUT}check_b/{tag}/{pid:02d}.mp3", Body=open(f"{d}/a.mp3", "rb").read())
        s = {"tag": tag, "pid": pid, "seconds_audio": round(len(codes2) / (SR / FRAME), 1), "stopped_at_end_marker": len(out_ids) < max_tokens,
             "non_audio_tokens": int((~ok).sum()), "generate_seconds": round(time.time() - t0, 1)}
        stats.append(s)
        print(json.dumps(s), flush=True)
    _put(f"check_b/gen_stats_{tag}_{pids[0]:02d}.json", stats)
    return stats


# ------------------------------------------------------------------ check B: scoring
@app.function(image=score_image, gpu="L40S", memory=65536, secrets=[b2_secret], timeout=3 * 3600)
def score_b(prompts: list):
    import re, subprocess, sys, tempfile
    import jiwer, librosa, numpy as np, torch, whisper
    from hydra.utils import instantiate
    from muq import MuQ, MuQMuLan
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    sys.path.insert(0, "/root/SongEval")
    dev = torch.device("cuda")
    asr = whisper.load_model("large-v3", device="cuda")
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(dev).eval()
    cfg = OmegaConf.load("/root/SongEval/config.yaml")
    head = instantiate(cfg.generator).to(dev).eval()
    head.load_state_dict(load_file("/root/SongEval/ckpt/model.safetensors", device="cpu"), strict=False)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(dev).eval()
    norm = lambda t: " ".join(re.findall(r"[a-z']+", re.sub(r"\[[^\]]*\]", " ", t.lower())))
    b2 = _b2()
    out = []
    for p in prompts:
        for tag in ("original", "final"):
            with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                f.write(b2.get_object(Bucket=B2_BUCKET, Key=f"{OUT}check_b/{tag}/{p['pid']:02d}.mp3")["Body"].read()); f.flush()
                text = asr.transcribe(f.name, language="en", temperature=0.0, fp16=True)["text"]
                wav24, _ = librosa.load(f.name, sr=24000)
            ref, hyp = norm(p["lyrics"]), norm(text)
            m = jiwer.process_words(ref, hyp if hyp else "x")
            with torch.no_grad():
                a = torch.tensor(wav24).unsqueeze(0).to(dev)
                style_sim = float(mulan.calc_similarity(mulan(wavs=a), mulan(texts=[p["style"][:300]])).squeeze())
                se = head(muq(a, output_hidden_states=True)["hidden_states"][6]).squeeze(0).tolist()
            row = {"pid": p["pid"], "tag": tag, "lyrics_word_error_rate": round(m.wer, 3),
                   "lyrics_words_sung_correctly": round(m.hits / max(1, len(ref.split())), 3),
                   "style_match": round(style_sim, 4), "transcript": text.strip()[:400],
                   **{k: round(v, 3) for k, v in zip(["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"], se)}}
            out.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "transcript"}), flush=True)
    _put("check_b/scores.json", out)
    return len(out)


@app.function(image=audiobox_image, gpu="L4", memory=32768, secrets=[b2_secret], timeout=3600)
def score_audiobox(pids: list):
    import subprocess, tempfile
    from audiobox_aesthetics.infer import initialize_predictor
    predictor = initialize_predictor()
    b2 = _b2()
    out = []
    for pid in pids:
        for tag in ("original", "final"):
            with tempfile.TemporaryDirectory() as d:
                open(f"{d}/a.mp3", "wb").write(b2.get_object(Bucket=B2_BUCKET, Key=f"{OUT}check_b/{tag}/{pid:02d}.mp3")["Body"].read())
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/a.mp3", f"{d}/a.wav"], check=True)
                r = predictor.forward([{"path": f"{d}/a.wav"}])[0]
            out.append({"pid": pid, "tag": tag, **{k: round(float(v), 3) for k, v in r.items()}})
    _put("check_b/audiobox.json", out)
    return len(out)


# ------------------------------------------------------------------ report
@app.function(image=gen_image, cpu=2, memory=16384, volumes={"/data": vol}, secrets=[b2_secret], timeout=1800)
def report():
    import numpy as np
    b2 = _b2()
    get = lambda k: json.loads(b2.get_object(Bucket=B2_BUCKET, Key=OUT + k)["Body"].read())
    meta = {m["id"]: m for m in json.load(open("/data/sft_data/meta.json"))}
    rng = np.random.default_rng(0)

    def paired(base, new, groups=None, n_boot=2000):
        ids = sorted(set(base) & set(new))
        d = np.array([new[i] - base[i] for i in ids])
        if groups:                                   # resample whole groups (artists) together
            g = [groups[i] for i in ids]
            uniq = sorted(set(g))
            idx = {u: [k for k, x in enumerate(g) if x == u] for u in uniq}
            boots = [d[np.concatenate([idx[uniq[j]] for j in rng.integers(0, len(uniq), len(uniq))])].mean() for _ in range(n_boot)]
        else:
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"n": len(ids), "mean_change": round(float(d.mean()), 4), "range95": [round(float(lo), 4), round(float(hi), 4)],
                "share_lower": round(float((d < 0).mean()), 3)}

    rep = {"run": RUN, "check_a_all_test_songs": {}, "forgetting_real_songs": {}}
    scores = {t: get(f"check_a/{t}.json") for t in TAGS}
    artist = {i: meta[i]["user_id"] for i in scores["original"]["test"]}
    rep["original_mean_surprise"] = {"test": round(float(np.mean(list(scores["original"]["test"].values()))), 4),
                                     "real_songs": round(float(np.mean(list(scores["original"]["forget"].values()))), 4)}
    for t in TAGS[1:]:
        rep["check_a_all_test_songs"][t] = paired(scores["original"]["test"], scores[t]["test"], artist)
        rep["forgetting_real_songs"][t] = paired(scores["original"]["forget"], scores[t]["forget"])
    sb = get("check_b/scores.json")
    ab = get("check_b/audiobox.json")
    rows = {}
    for r in sb + ab:
        rows.setdefault((r["pid"], r["tag"]), {}).update({k: v for k, v in r.items() if isinstance(v, (int, float)) and k != "pid"})
    metrics = sorted({k for v in rows.values() for k in v})
    rep["check_b_final_minus_original"] = {}
    for mname in metrics:
        base = {pid: v[mname] for (pid, tag), v in rows.items() if tag == "original" and mname in v}
        new = {pid: v[mname] for (pid, tag), v in rows.items() if tag == "final" and mname in v}
        res = paired(base, new)
        res["original_avg"] = round(float(np.mean(list(base.values()))), 4)
        res["final_avg"] = round(float(np.mean(list(new.values()))), 4)
        rep["check_b_final_minus_original"][mname] = res
    gen = []
    for k in b2.list_objects_v2(Bucket=B2_BUCKET, Prefix=OUT + "check_b/gen_stats_").get("Contents", []):
        gen += json.loads(b2.get_object(Bucket=B2_BUCKET, Key=k["Key"])["Body"].read())
    for t in ("original", "final"):
        g = [s for s in gen if s["tag"] == t]
        rep[f"generation_{t}"] = {"songs": len(g), "avg_seconds": round(float(np.mean([s["seconds_audio"] for s in g])), 1),
                                  "stopped_at_end_marker": round(float(np.mean([s["stopped_at_end_marker"] for s in g])), 3),
                                  "songs_with_non_audio_tokens": int(sum(s["non_audio_tokens"] > 0 for s in g))}
    _put("report.json", rep)
    return rep


# ------------------------------------------------------------------ blind listening (the /judge page)
LISTEN_LUFS, LISTEN_TP = -16.0, -1.5   # every take played at the same loudness: louder tends to win blind tests


def _loudness(path, extra=""):
    import re, subprocess
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                          f"loudnorm=I={LISTEN_LUFS}:TP={LISTEN_TP}:LRA=11{extra}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True, check=True).stderr
    return json.loads(re.findall(r"\{[^{}]*\}", err)[-1])


@app.function(image=gen_image, cpu=8, memory=8192, secrets=[b2_secret], timeout=1800)
def listen_prep(seed: int = 2026):
    """Loudness-matched copies of every check B song + a blind pair set for the /judge page.

    Two-pass ffmpeg loudnorm in linear mode (one gain per song, no compression) to LISTEN_LUFS; each
    copy is re-measured to confirm. Which model is take a / take b is randomized per pair and stored
    only in the pair set (the page never sends it to the browser until the results are unlocked)."""
    import random, subprocess, tempfile
    from concurrent.futures import ThreadPoolExecutor
    b2 = _b2()
    prompts = json.loads(b2.get_object(Bucket=B2_BUCKET, Key=OUT + "check_b_prompts.json")["Body"].read())

    def one(job):
        pid, tag = job
        with tempfile.TemporaryDirectory() as d:
            src, dst = f"{d}/in.mp3", f"{d}/out.mp3"
            open(src, "wb").write(b2.get_object(Bucket=B2_BUCKET, Key=f"{OUT}check_b/{tag}/{pid:02d}.mp3")["Body"].read())
            m = _loudness(src)
            measured = (f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
                        f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af",
                            f"loudnorm=I={LISTEN_LUFS}:TP={LISTEN_TP}:LRA=11{measured}",
                            "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "192k", dst], check=True)
            after = _loudness(dst)
            b2.upload_file(dst, B2_BUCKET, f"{OUT}listen/{pid:02d}_{tag}.mp3", ExtraArgs={"ContentType": "audio/mpeg"})
        return {"pid": pid, "tag": tag, "lufs_before": float(m["input_i"]), "lufs_after": float(after["input_i"]),
                "true_peak_after": float(after["input_tp"])}

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, [(p["pid"], t) for p in prompts for t in ("original", "final")]))
    for r in rows:
        print(json.dumps(r), flush=True)
    _put("listen/loudness.json", rows)

    rng = random.Random(seed)
    pairs = []
    for p in prompts:
        a, b = rng.sample(["original", "final"], 2)
        pairs.append({"uid": f"p{p['pid']:02d}", "prompt": p["style"], "lyrics": p["lyrics"],
                      "a": f"b2://{B2_BUCKET}/{OUT}listen/{p['pid']:02d}_{a}.mp3",
                      "b": f"b2://{B2_BUCKET}/{OUT}listen/{p['pid']:02d}_{b}.mp3",
                      "model": {"a": a, "b": b}, "prior": None, "prior_label": None})
    _put("listen/pairset.json", {
        "id": f"shao-{RUN}", "kind": "model-ab", "title": "Original Shao vs fine-tuned Shao",
        "blurb": f"Same {len(pairs)} test prompts, one take from each model, loudness-matched to {LISTEN_LUFS:g} LUFS. "
                 "Which take is which stays hidden until you finish.",
        "models": {"original": "Original Shao", "final": f"Fine-tuned Shao ({RUN})"},
        "pairs": pairs})
    return {"pairs": len(pairs), "max_lufs_error": max(abs(r["lufs_after"] - LISTEN_LUFS) for r in rows)}


@app.local_entrypoint()
def listen():
    print(listen_prep.remote())


# ------------------------------------------------------------------ when does the singing start?
demucs_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", "torchaudio==2.4.1", "numpy<2", "librosa", "soundfile", "demucs", "boto3")
)
VOCAL_WIN_S = 0.5
VOCAL_SHARES = (0.1, 0.2, 0.3)   # several cut-offs, so no single hand-picked number decides the answer


@app.function(image=demucs_image, gpu="L4", memory=16384, secrets=[b2_secret], timeout=1800)
def vocal_timeline():
    """Split every check B song into vocals vs everything else (Demucs htdemucs) and record, per 0.5 s,
    what share of the sound energy is the voice. Reports when singing starts and how much of the song
    has singing, at several cut-offs."""
    import io, numpy as np, torch, librosa
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    dem = get_model("htdemucs").to("cuda").eval()          # stems: drums, bass, other, vocals
    b2 = _b2()
    prompts = json.loads(b2.get_object(Bucket=B2_BUCKET, Key=OUT + "check_b_prompts.json")["Body"].read())
    rows = []
    for p in prompts:
        for tag in ("original", "final"):
            raw = b2.get_object(Bucket=B2_BUCKET, Key=f"{OUT}check_b/{tag}/{p['pid']:02d}.mp3")["Body"].read()
            wav, sr = librosa.load(io.BytesIO(raw), sr=dem.samplerate, mono=False)
            wav = np.atleast_2d(wav)
            if wav.shape[0] == 1:
                wav = np.repeat(wav, 2, 0)
            with torch.no_grad():
                src = apply_model(dem, torch.tensor(wav)[None].float(), device="cuda", split=True)[0].cpu().numpy()
            win = int(VOCAL_WIN_S * sr)
            n = src.shape[-1] // win
            e = (src[..., : n * win].reshape(4, 2, n, win) ** 2).sum(axis=(1, 3))     # [stem, window] energy
            vocal, total = e[3], e.sum(0) + 1e-12
            share = vocal / total
            smooth = np.convolve(share, np.ones(2) / 2, mode="same")                  # 1 s smoothing
            row = {"pid": p["pid"], "tag": tag, "seconds": round(src.shape[-1] / sr, 1),
                   "vocal_energy_share": round(float(vocal.sum() / total.sum()), 4),
                   "share_timeline": [round(float(x), 3) for x in share]}
            for c in VOCAL_SHARES:
                hit = np.nonzero(smooth > c)[0]
                row[f"onset_s@{c}"] = round(float(hit[0] * VOCAL_WIN_S), 1) if len(hit) else None
                row[f"sung_fraction@{c}"] = round(float((smooth > c).mean()), 3)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "share_timeline"}), flush=True)
    _put("check_b/vocal_timeline.json", rows)
    return len(rows)


@app.local_entrypoint()
def vocals():
    print(vocal_timeline.remote())


# ------------------------------------------------------------------ coordinator
@app.function(image=gen_image, cpu=1, memory=4096, volumes={"/data": vol}, secrets=[b2_secret], timeout=8 * 3600)
def run_all():
    _progress("start")
    weights.remote()
    n = pick_prompts.remote()
    _progress("setup_done", prompts=n)
    vol.reload()
    prompts = json.load(open("/data/eval/check_b_prompts.json"))
    pids = [p["pid"] for p in prompts]
    gen_calls = [generate.spawn(tag, pids[i:i + 10]) for tag in ("original", "final") for i in range(0, len(pids), 10)]
    nf = forget_prep.remote()
    _progress("real_songs_converted", songs=nf)
    a_calls = [check_a.spawn(t) for t in TAGS]
    for c in gen_calls:
        c.get()
    _progress("check_b_songs_written", songs=2 * len(pids))
    sb = score_b.spawn([{k: v for k, v in p.items() if k != "prompt_ids"} for p in prompts])
    ab = score_audiobox.spawn(pids)
    for c in a_calls:
        _progress("check_a_done", tag=c.get())
    sb.get(); ab.get()
    _progress("check_b_scored")
    rep = report.remote()
    _progress("done", report=OUT + "report.json")
    return rep


@app.function(image=gen_image, cpu=1, memory=4096, volumes={"/data": vol}, secrets=[b2_secret], timeout=3 * 3600)
def finish():
    """Re-run only the audio-quality scoring and the report (everything before them is already in B2)."""
    prompts = json.load(open("/data/eval/check_b_prompts.json"))
    score_audiobox.remote([p["pid"] for p in prompts])
    _progress("check_b_scored")
    rep = report.remote()
    _progress("done", report=OUT + "report.json")
    return rep


@app.local_entrypoint()
def finish_rest():
    call = finish.spawn()
    print(f"finish launched: {call.object_id} (safe to close the laptop)")


@app.local_entrypoint()
def main():
    call = run_all.spawn()
    print(f"evaluation launched: {call.object_id} (safe to close the laptop); progress: B2 {OUT}progress.jsonl")
