"""Overfit test: train Shao's whole backbone on ONE song until it memorizes it, then make it sing it back.

If this fails, something in the pipeline is broken (prompt format, token layout, loss, precision,
generation path) and full SFT would fail too. Everything is logged and every clip is saved.

Clips (all full-length mp3, same song, same prompt):
  1_original_suno                      the real Suno audio
  2_codec_all64_true_layers            the real song through Shao's codec (all 64 real layers) = best Shao's audio path can sound
  3_true_layers01_plus_shao_superres   real layers 0-1 + Shao's super-res inventing layers 2-63 = best a PERFECT backbone can sound
  4_shao_before_training               original Shao, same prompt (normal sampling)
  5a_..._greedy_fullprecision          overfit Shao, no randomness, fp32 weights  <- the test
  5b_..._greedy_bf16                   same, weights rounded to bf16 (how Shao normally runs)
  6_..._sampled_bf16                   overfit Shao, normal sampling
Checks (v2): train to loss < target_loss; update-size/weight-size per step; EOS prediction; save->reload through
Shao's loader gives identical weights and loss; fp32 vs bf16 accuracy; mismatch vs saved token file.

Outputs: B2 erised-sft/overfit_test/<run>/ : train_log.jsonl, report.json, *.mp3
Run (workspace erised4): MODAL_PROFILE=erised4 modal run --detach shao_dpo/overfit_one_song.py --run-name v1
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
B2_BUCKET, B2_ENDPOINT = "erised-sft", "https://s3.us-west-004.backblazeb2.com"
SR, FRAME, EOS = 44100, 2048, 128001          # EOS = backbone end-of-song id (backend_worker.BACKBONE_EOD_ID)
MPS_REPO, SHAO_REPO = "Vinpolar/Khala-MusicGeneration-v1.0-MPS", "liujiafeng/Shao-MusicGeneration-v1.0"

app = modal.App("shao-overfit-test")
vol = modal.Volume.from_name("shao-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", "numpy<2", "omegaconf", "einops", "lightning", "boto3", "huggingface_hub",
                 "safetensors", "transformers==4.44.2", "soundfile")
    .add_local_dir(HERE.parent / "khala_runtime", remote_path="/root/khala")
    .add_local_file(HERE / "dpo_common.py", "/root/dpo_common.py")
)


@app.function(image=image, gpu="H100", memory=65536, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("b2-key")], timeout=3 * 3600)
def run(run_name: str, lr: float = 5e-5, max_steps: int = 400, seed: int = 0, target_loss: float = 5e-5,
        skip_baseline: bool = False):
    import io, math, os, subprocess, sys, tempfile, time
    import boto3, numpy as np, soundfile as sf, torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from omegaconf import OmegaConf
    sys.path[:0] = ["/root", "/root/khala", "/root/khala/models/Decoder"]
    import dpo_common as dc
    from dac_rvq import DacRVQ
    from core.khala_runtime import load_vanilla_model, sample_backbone, generate_superres_projection
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    t_start = time.time()
    b2 = boto3.client("s3", endpoint_url=B2_ENDPOINT)
    out = f"overfit_test/{run_name}/"
    log_lines, report = [], {"run": run_name, "lr": lr, "max_steps": max_steps, "seed": seed, "target_loss": target_loss}

    def log(event, **kw):
        row = {"t": round(time.time() - t_start, 1), "event": event, **kw}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        log_lines.append(row)
        if len(log_lines) % 10 == 0 or event in ("done", "error"):
            b2.put_object(Bucket=B2_BUCKET, Key=out + "train_log.jsonl",
                          Body="\n".join(json.dumps(r, ensure_ascii=False) for r in log_lines).encode())

    def put_mp3(name, wav):
        with tempfile.TemporaryDirectory() as d:
            sf.write(f"{d}/a.wav", wav.T, SR, subtype="FLOAT")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/a.wav", "-b:a", "320k", f"{d}/a.mp3"], check=True)
            b2.put_object(Bucket=B2_BUCKET, Key=f"{out}{name}.mp3", Body=open(f"{d}/a.mp3", "rb").read())
        log("saved_clip", name=name, seconds=round(wav.shape[1] / SR, 1))

    # ---------- pick one song (vocal, real lyrics, 2-3 min) and one held-out song ----------
    get = lambda k: b2.get_object(Bucket=B2_BUCKET, Key=k)["Body"].read()
    pool = [json.loads(l) for l in get("shao_tokens/v1/source_list.jsonl").decode().split("\n") if l.strip()]
    ok = lambda s: (not s.get("instrumental") and len((s.get("lyrics") or "").strip()) > 300
                    and 120 <= (s["audio"].get("duration_seconds") or 0) <= 180 and (s.get("style_prompt") or "").strip())
    song = next(s for s in pool if ok(s))
    other = next(s for s in pool if ok(s) and s["user_id"] != song["user_id"])
    report["song"] = {k: song.get(k) for k in ("id", "title", "handle", "song_url", "style_prompt", "major_model_version")}
    report["song"]["duration_seconds"] = song["audio"]["duration_seconds"]
    report["heldout_song"] = {k: other.get(k) for k in ("id", "title", "handle")}
    log("picked", song=report["song"], heldout=report["heldout_song"])

    # ---------- weights (free from Hugging Face, cached on the volume) ----------
    bb_w = hf_hub_download(MPS_REPO, "khala_backbone.safetensors", cache_dir="/data/hf")
    bb_a = hf_hub_download(MPS_REPO, "backbone_megatron_args.json", cache_dir="/data/hf")
    sr_w = hf_hub_download(MPS_REPO, "khala_superres.safetensors", cache_dir="/data/hf")
    sr_a = hf_hub_download(MPS_REPO, "superres_megatron_args.json", cache_dir="/data/hf")
    codec_ckpt = hf_hub_download(SHAO_REPO, "dac_rvq_2490000.ckpt", cache_dir="/data/hf")
    vol.commit()
    sd = torch.load(codec_ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    dac = DacRVQ(OmegaConf.load("/root/khala/models/Decoder/dac_rvq_1024_64_golden.yaml"))
    dac.load_state_dict({k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")})  # strict, like Shao's own loader
    dac = dac.eval().cuda()
    del sd
    log("weights_loaded")

    # ---------- the real song: audio -> 64-layer codes ----------
    def decode_file(key):
        with tempfile.NamedTemporaryFile(suffix=Path(key).suffix) as f:
            f.write(get(key)); f.flush()
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", f.name, "-f", "f32le", "-ac", "2", "-ar", str(SR), "pipe:1"],
                                 capture_output=True, check=True).stdout
        return np.frombuffer(pcm, dtype=np.float32).reshape(-1, 2).T.copy()

    @torch.no_grad()
    def encode(wav):    # same windowing as tokenize_songs.py
        T = -(-wav.shape[1] // FRAME)
        wav = np.pad(wav, ((0, 0), (0, T * FRAME - wav.shape[1])))
        core, ctx, parts = 640 * FRAME, 44 * FRAME, []
        for s in range(0, T * FRAME, core):
            a, b = max(0, s - ctx), min(T * FRAME, s + core + ctx)
            c = dac.encode(torch.from_numpy(wav[:, a:b])[None].cuda())
            parts.append(c[:, 0, (s - a) // FRAME:(s - a) // FRAME + (min(s + core, T * FRAME) - s) // FRAME])
        return torch.cat(parts, 1).T.cpu().numpy().astype(np.int64)          # [T, 64]

    @torch.no_grad()
    def decode_codes(codes64):   # Shao's own decoder chunking: 1920-frame chunks, 480 overlap, linear crossfade
        codes = torch.from_numpy(np.ascontiguousarray(codes64.T))[:, None].cuda()   # [64, 1, T]
        chunk, overlap = 1920, 480
        wave = None
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
        return wave[0].numpy()

    wav = decode_file(song["audio"]["key"])
    true64 = encode(wav)
    b2_q01 = np.load(io.BytesIO(get(f"shao_tokens/v1/q01/{song['id']}.npy")))
    report["check_encode_matches_saved_tokens"] = bool(np.array_equal(true64[:, :2], b2_q01))
    if len(b2_q01) == len(true64):   # how many frames differ between this GPU's encode and the saved (L4) encode
        diff = true64[:, :2] != b2_q01
        report["encode_vs_saved_mismatch"] = {"q0": round(float(diff[:, 0].mean()), 5), "q1": round(float(diff[:, 1].mean()), 5)}
    log("encoded", frames=int(true64.shape[0]), matches_saved_q01=report["check_encode_matches_saved_tokens"],
        mismatch_vs_saved=report.get("encode_vs_saved_mismatch"))
    if not skip_baseline:
        put_mp3("1_original_suno", wav)
        put_mp3("2_codec_all64_true_layers", decode_codes(true64))

    # ---------- prompt, exactly as Shao's app builds it ----------
    tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
    minutes = int(min(10, max(1, round(song["audio"]["duration_seconds"] / 60))))
    prompt_text = dc.build_prompt_text(song["style_prompt"], song["lyrics"], False, minutes)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    report["prompt"] = {"text": prompt_text, "n_tokens": len(prompt_ids), "duration_token_minutes": minutes}
    true_ids = dc.audio_ids_from_codes(true64[:, :2])                               # q0,q1,q0,q1,...
    seq = torch.cat([torch.tensor(prompt_ids), true_ids, torch.tensor([EOS])]).cuda()
    a0 = len(prompt_ids)
    assert len(seq) <= dc.CONTEXT_LEN, len(seq)
    log("prompt", n_prompt_tokens=len(prompt_ids), n_audio_tokens=int(len(true_ids)), total=int(len(seq)))

    other_codes = np.load(io.BytesIO(get(f"shao_tokens/v1/q01/{other['id']}.npy")))
    other_prompt = tok.encode(dc.build_prompt_text(other["style_prompt"] or "", other["lyrics"] or "", False,
                                                   int(min(10, max(1, round(other["audio"]["duration_seconds"] / 60))))),
                              add_special_tokens=False)
    other_seq = torch.cat([torch.tensor(other_prompt), dc.audio_ids_from_codes(other_codes)]).cuda()[:dc.CONTEXT_LEN]

    # ---------- helpers: scoring with per-layer loss/accuracy, and full generation ----------
    def chunk_stats(h, w, t):
        logits = (h @ w.t()).float()
        return logits.gather(-1, t[:, None]).squeeze(-1) - torch.logsumexp(logits, -1), logits.argmax(-1) == t

    def score(model, ids, start, grad, autocast=True):
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            h = dc.hidden_states(model, ids[None], use_checkpoint=grad)[0]
            pred, tgt, w = h[start - 1:-1], ids[start:], model.lm_head.weight[:dc.REAL_VOCAB]
            lps, hits = [], []
            for s in range(0, len(tgt), 1024):
                args = (pred[s:s + 1024], w, tgt[s:s + 1024])
                lp, hit = checkpoint(chunk_stats, *args, use_reentrant=False) if grad else chunk_stats(*args)
                lps.append(lp); hits.append(hit)
        return torch.cat(lps), torch.cat(hits)

    def summarize(lp, hit, n_audio):
        lp_a, hit_a = lp[:n_audio], hit[:n_audio]
        return {"loss": round(float(-lp.mean()), 4), "loss_q0": round(float(-lp_a[0::2].mean()), 4),
                "loss_q1": round(float(-lp_a[1::2].mean()), 4), "acc_q0": round(float(hit_a[0::2].float().mean()), 4),
                "acc_q1": round(float(hit_a[1::2].float().mean()), 4)}

    def generate(model, temperature, top_k, tag):
        torch.manual_seed(seed)
        t0 = time.time()
        out_ids = sample_backbone(model, prompt_ids, num_tokens=len(true_ids) + 400,
                                  temperature=temperature, top_k=top_k, eos_id=EOS)
        ids = np.array(out_ids, dtype=np.int64)
        ids = ids[: len(ids) // 2 * 2].reshape(-1, 2)
        valid = bool(((ids[:, 0] >= 128256) & (ids[:, 0] < 129280) & (ids[:, 1] >= 129280) & (ids[:, 1] < 130304)).all())
        codes = np.stack([ids[:, 0] - 128256, ids[:, 1] - 129280], 1)
        n = min(len(codes), len(true64))
        same = codes[:n] == true64[:n, :2]
        diverge = np.where(~same.all(1))[0]
        stats = {"frames_generated": int(len(codes)), "frames_true": int(len(true64)), "stopped_at_eos": len(out_ids) < len(true_ids) + 400,
                 "all_tokens_valid_audio_ids": valid, "match_q0": round(float(same[:, 0].mean()), 4),
                 "match_q1": round(float(same[:, 1].mean()), 4),
                 "first_mismatch_second": None if len(diverge) == 0 else round(float(diverge[0] / (SR / FRAME)), 1),
                 "seconds_to_generate": round(time.time() - t0, 1)}
        log("generated", which=tag, **stats)
        report[f"generation_{tag}"] = stats
        return np.clip(codes, 0, 1023)

    # ---------- 4: original Shao, before any training ----------
    model = load_vanilla_model("backbone", bb_w, bb_a, "cuda", torch.bfloat16)
    lp, hit = score(model, seq, a0, grad=False)
    report["before_training_on_song"] = summarize(lp, hit, len(true_ids))
    log("before_training_score", **report["before_training_on_song"])
    before_codes = None if skip_baseline else generate(model, 1.0, 50, "before_sampled")
    del model
    torch.cuda.empty_cache()

    # ---------- training: whole backbone, fp32 weights + bf16 compute, one song ----------
    model = load_vanilla_model("backbone", bb_w, bb_a, "cuda", torch.float32).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    # a few weight matrices whose per-step change we track (size of update / size of weight)
    watch = {f"layer{i}.{n}": getattr(getattr(model.layers[i], part), n).weight
             for i in (0, 12, 23) for part, n in (("attn", "q_proj"), ("mlp", "down_proj"))}
    warmup, streak = 10, 0
    for step in range(1, max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / warmup)
        t0 = time.time()
        lp, hit = score(model, seq, a0, grad=True)
        loss = -lp.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        before = {k: w.detach().clone() for k, w in watch.items()}
        opt.step()
        ratios = [float((w.detach() - before[k]).norm() / w.detach().norm()) for k, w in watch.items()]
        row = summarize(lp.detach(), hit, len(true_ids)) | {
            "step": step, "lr": opt.param_groups[0]["lr"], "grad_norm": round(gnorm, 3),
            "update_to_weight_mean": float(f"{sum(ratios) / len(ratios):.2e}"), "update_to_weight_max": float(f"{max(ratios):.2e}"),
            "eos_predicted": bool(hit[-1]), "step_seconds": round(time.time() - t0, 2)}
        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                lpo, hito = score(model, other_seq, len(other_prompt), grad=False)
            row["heldout_song_loss"] = round(float(-lpo.mean()), 4)
        log("train_step", **row)
        streak = streak + 1 if float(loss) < target_loss else 0
        if streak >= 3:
            break
    report["training"] = {"steps": step, "final": row, "target_loss": target_loss}
    del opt, before
    model.eval()
    torch.cuda.empty_cache()

    # ---------- checks on the trained weights ----------
    with torch.no_grad():
        lp, hit = score(model, seq, a0, grad=False)                           # training numerics (fp32 weights, bf16 compute)
        report["after_fp32_weights_bf16_compute"] = summarize(lp, hit, len(true_ids)) | {"eos_predicted": bool(hit[-1])}
        lp, hit = score(model, seq, a0, grad=False, autocast=False)           # pure fp32
        report["after_fp32_weights_fp32_compute"] = summarize(lp, hit, len(true_ids)) | {"eos_predicted": bool(hit[-1])}
    log("after_scores", fp32_bf16=report["after_fp32_weights_bf16_compute"], fp32=report["after_fp32_weights_fp32_compute"])

    # save -> reload through Shao's own loader -> must score the same
    from safetensors.torch import save_file
    Path("/data/ckpt_test").mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, "/data/ckpt_test/backbone_fp32.safetensors")
    reloaded = load_vanilla_model("backbone", "/data/ckpt_test/backbone_fp32.safetensors", bb_a, "cuda", torch.float32).eval()
    with torch.no_grad():
        lp_r, hit_r = score(reloaded, seq, a0, grad=False, autocast=False)
    report["save_reload"] = {"loss_before_save": report["after_fp32_weights_fp32_compute"]["loss"],
                             "loss_after_reload": round(float(-lp_r.mean()), 4),
                             "identical_weights": all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values()))}
    log("save_reload", **report["save_reload"])
    del reloaded
    os.remove("/data/ckpt_test/backbone_fp32.safetensors")
    torch.cuda.empty_cache()

    # ---------- 5, 6: overfit Shao sings it back (full precision first, then rounded to bf16) ----------
    greedy_fp32 = generate(model, 0.0, 1, "after_greedy_fp32")
    model = model.to(torch.bfloat16)
    with torch.no_grad():
        lp, hit = score(model, seq, a0, grad=False)
    report["after_bf16_weights"] = summarize(lp, hit, len(true_ids)) | {"eos_predicted": bool(hit[-1])}
    log("after_bf16_score", **report["after_bf16_weights"])
    greedy_codes = generate(model, 0.0, 1, "after_greedy")
    sampled_codes = generate(model, 1.0, 50, "after_sampled")
    del model
    torch.cuda.empty_cache()

    # ---------- super-res (layers 2-63) + decoder, exactly like Shao's worker ----------
    sr = load_vanilla_model("superres", sr_w, sr_a, "cuda", torch.bfloat16)
    sr_prompt = prompt_ids if len(prompt_ids) <= 2048 else prompt_ids[:2047] + [prompt_ids[-1]]

    @torch.inference_mode()
    def superres(codes2, tag):
        torch.manual_seed(seed)
        audio_ids = dc.audio_ids_from_codes(codes2).numpy().reshape(-1, 2)
        text = np.array(sr_prompt, dtype=np.int64)
        L = len(text) + len(audio_ids)
        assert L <= 8192, f"super-res window is 8192 tokens, need {L}"
        tokens = torch.full((L, 2), -1, dtype=torch.long)
        tokens[:len(text), 0] = torch.from_numpy(text)
        tokens[len(text):] = torch.from_numpy(audio_ids)
        attn = torch.zeros(L, dtype=torch.bool)
        loss_mask = (tokens[:, -1] != -1).float()
        full = generate_superres_projection(sr, tokens[None].cuda(), attn[None, None, None].cuda(), loss_mask[None].cuda(),
                                            torch.arange(L)[None].cuda(), len(text), len(audio_ids), 10)
        codes64 = full[:, 0, :].cpu().numpy().T.astype(np.int64)
        codes64 -= (128256 + np.arange(64) * 1024)[None, :]
        assert np.array_equal(codes64[:, :2], codes2), "super-res changed layers 0-1"
        log("superres_done", which=tag, frames=int(len(codes64)), codes_in_range=bool(((codes64 >= 0) & (codes64 < 1024)).all()))
        return codes64

    clips = [("5a_shao_after_overfit_greedy_fullprecision", greedy_fp32),
             ("5b_shao_after_overfit_greedy_bf16", greedy_codes), ("6_shao_after_overfit_sampled_bf16", sampled_codes)]
    if not skip_baseline:
        clips = [("3_true_layers01_plus_shao_superres", true64[:, :2]), ("4_shao_before_training", before_codes)] + clips
    for name, c2 in clips:
        put_mp3(name, decode_codes(superres(np.ascontiguousarray(c2), name)))

    report["minutes_total"] = round((time.time() - t_start) / 60, 1)
    b2.put_object(Bucket=B2_BUCKET, Key=out + "report.json", Body=json.dumps(report, indent=1, ensure_ascii=False).encode())
    log("done", minutes=report["minutes_total"])
    return report


@app.local_entrypoint()
def main(run_name: str = "v1", lr: float = 5e-5, max_steps: int = 400, target_loss: float = 5e-5, skip_baseline: bool = False):
    print(json.dumps(run.remote(run_name, lr=lr, max_steps=max_steps, target_loss=target_loss, skip_baseline=skip_baseline),
                     indent=1, ensure_ascii=False))
