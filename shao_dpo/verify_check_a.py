"""Can "check A" (how surprised Shao is by test songs it never trained on) actually measure progress?

Four tests, all on the same 600 test songs (from artists never used for training, max 3 songs per artist):
  1. repeatability + chance wobble: score original Shao twice (must be identical), then reshuffle which
     artists are in the test set 2,000 times to see how much the average moves by chance alone
  2. real training:     300 normal training songs, scored after every 100 songs (should improve beyond the wobble)
  3. garbage training:  the same 300 songs with each song's moments shuffled out of order (should NOT improve)
  4. cheating on purpose: train on 300 of the test songs themselves (those should improve a lot, the other
     300 test songs much less) - proves the check can see learning when it happens
Every run restarts from the original Shao. Score = average surprise per note (lower = better), per song.

Outputs: B2 erised-sft/check_a_verify/<run>/ : log.jsonl, report.json
Run: MODAL_PROFILE=erised7 modal run --detach shao_dpo/verify_check_a.py --run-name v1   (returns at once; job keeps running)
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
B2_BUCKET, B2_ENDPOINT, TOK_DIR = "erised-sft", "https://s3.us-west-004.backblazeb2.com", "shao_tokens/v1/"
MPS_REPO, EOS = "Vinpolar/Khala-MusicGeneration-v1.0-MPS", 128001

app = modal.App("shao-verify-check-a")
vol = modal.Volume.from_name("shao-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy<2", "boto3", "huggingface_hub", "safetensors", "transformers==4.44.2")
    .add_local_dir(HERE.parent / "khala_runtime", remote_path="/root/khala")
    .add_local_file(HERE / "dpo_common.py", "/root/dpo_common.py")
)


@app.function(image=image, gpu="H100", memory=65536, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("b2-key")], timeout=4 * 3600)
def run(run_name: str, n_eval: int = 600, n_train: int = 300, lr: float = 2e-5, accum: int = 4, seed: int = 0,
        tests: str = "1,2,3,4"):
    import io, random, sys, time
    import boto3, numpy as np, torch
    from torch.utils.checkpoint import checkpoint
    sys.path[:0] = ["/root", "/root/khala"]
    import dpo_common as dc
    from core.khala_runtime import load_vanilla_model
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    t0 = time.time()
    b2 = boto3.client("s3", endpoint_url=B2_ENDPOINT)
    out, log_rows, report = f"check_a_verify/{run_name}/", [], {"run": run_name, "lr": lr, "accum": accum, "seed": seed, "tests": tests}
    todo = {int(x) for x in tests.split(",")}

    def log(event, **kw):
        row = {"t": round(time.time() - t0, 1), "event": event, **kw}
        print(json.dumps(row), flush=True)
        log_rows.append(row)
        b2.put_object(Bucket=B2_BUCKET, Key=out + "log.jsonl", Body="\n".join(json.dumps(r) for r in log_rows).encode())

    # ---------- songs ----------
    get = lambda k: b2.get_object(Bucket=B2_BUCKET, Key=k)["Body"].read()
    rows = [json.loads(l) for l in get(TOK_DIR + "manifest.jsonl").decode().split("\n") if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    per_artist, eval_songs = {}, []
    for r in rows:                                   # test songs, at most 3 per artist so no artist dominates
        if r["split"] == "test" and per_artist.get(r["user_id"], 0) < 3 and len(eval_songs) < n_eval:
            eval_songs.append(r); per_artist[r["user_id"]] = per_artist.get(r["user_id"], 0) + 1
    train_songs = [r for r in rows if r["split"] == "train"][:n_train]
    leak, clean = eval_songs[: n_eval // 2], eval_songs[n_eval // 2:]
    report["songs"] = {"eval": len(eval_songs), "eval_artists": len(per_artist), "train": len(train_songs)}
    log("songs", **report["songs"])

    tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
    cache = {}

    def sequence(r, shuffle_frames=False):
        key = (r["id"], shuffle_frames)
        if key not in cache:
            codes = np.load(io.BytesIO(get(r["q01_key"])))
            if shuffle_frames:
                codes = codes[np.random.default_rng(abs(hash(r["id"])) % 2**32).permutation(len(codes))]
            minutes = int(min(10, max(1, round(r["seconds"] / 60))))
            text = dc.build_prompt_text(r.get("style_prompt") or r.get("description_prompt") or "", r.get("lyrics") or "",
                                        bool(r.get("instrumental")), minutes)
            p = tok.encode(text, add_special_tokens=False)
            if len(p) > 4096:                         # Shao's own limit: keep the start and the final 'song starts here' marker
                p = p[:4095] + [p[-1]]
            ids, a0 = dc.build_sequence(p, codes)
            if len(ids) - a0 == 2 * len(codes) and len(ids) < dc.CONTEXT_LEN:   # whole song fits -> also learn where it ends
                ids = torch.cat([ids, torch.tensor([EOS])])
            cache[key] = (ids, a0)
        ids, a0 = cache[key]
        return ids.cuda(), a0

    def chunk_lp(h, w, t):
        logits = (h @ w.t()).float()
        return logits.gather(-1, t[:, None]).squeeze(-1) - torch.logsumexp(logits, -1)

    def song_logp(model, ids, a0, grad):
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            h = dc.hidden_states(model, ids[None], use_checkpoint=grad)[0]
            pred, tgt, w = h[a0 - 1:-1], ids[a0:], model.lm_head.weight[:dc.REAL_VOCAB]
            parts = [checkpoint(chunk_lp, pred[s:s + 1024], w, tgt[s:s + 1024], use_reentrant=False) if grad
                     else chunk_lp(pred[s:s + 1024], w, tgt[s:s + 1024]) for s in range(0, len(tgt), 1024)]
        return torch.cat(parts)

    @torch.no_grad()
    def evaluate(model, songs):
        """Average surprise per note for each song (audio notes only, not the end marker)."""
        model.eval()
        out = {}
        for r in songs:
            ids, a0 = sequence(r)
            lp = song_logp(model, ids, a0, grad=False)
            n_audio = len(ids) - a0 - (1 if ids[-1] == EOS else 0)
            out[r["id"]] = float(-lp[:n_audio].mean())
        return out

    artist_of = {r["id"]: r["user_id"] for r in eval_songs}

    def compare(base, new, songs):
        """Mean change per song (new - base; negative = less surprised) with a 95% range from reshuffling artists."""
        ids = [r["id"] for r in songs]
        diff = {i: new[i] - base[i] for i in ids}
        artists = sorted({artist_of[i] for i in ids})
        by_artist = {a: [diff[i] for i in ids if artist_of[i] == a] for a in artists}
        brng = np.random.default_rng(1)
        boots = []
        for _ in range(2000):
            pick = brng.choice(len(artists), len(artists), replace=True)
            vals = [v for k in pick for v in by_artist[artists[k]]]
            boots.append(np.mean(vals))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean_change": round(float(np.mean(list(diff.values()))), 5), "range95": [round(float(lo), 5), round(float(hi), 5)],
                "songs_improved": round(float(np.mean([v < 0 for v in diff.values()])), 3)}

    def train(songs, shuffle_frames=False, eval_every=None, label=""):
        model = load_vanilla_model("backbone", bb_w, bb_a, "cuda", torch.float32)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        steps, curve = 0, []
        model.train()
        for i, r in enumerate(songs, 1):
            ids, a0 = sequence(r, shuffle_frames)
            loss = -song_logp(model, ids, a0, grad=True).mean() / accum
            loss.backward()
            if i % accum == 0 or i == len(songs):
                steps += 1
                for g in opt.param_groups:
                    g["lr"] = lr * min(1.0, steps / 10)
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step(); opt.zero_grad(set_to_none=True)
                log("train_step", run=label, songs_seen=i, step=steps, loss=round(float(loss) * accum, 4), grad_norm=round(gn, 3))
            if eval_every and i % eval_every == 0 and i < len(songs):
                scores = evaluate(model, eval_songs)
                c = compare(base, scores, eval_songs)
                curve.append({"songs_seen": i} | c)
                log("eval", run=label, songs_seen=i, **c)
                model.train()
        return model, curve

    # ---------- weights ----------
    bb_w = hf_hub_download(MPS_REPO, "khala_backbone.safetensors", cache_dir="/data/hf")
    bb_a = hf_hub_download(MPS_REPO, "backbone_megatron_args.json", cache_dir="/data/hf")
    vol.commit()

    # ---------- 1. repeatability + chance wobble ----------
    model = load_vanilla_model("backbone", bb_w, bb_a, "cuda", torch.float32)
    base = evaluate(model, eval_songs)
    b2.put_object(Bucket=B2_BUCKET, Key=out + "original_shao_scores.json", Body=json.dumps(base).encode())
    again = evaluate(model, eval_songs) if 1 in todo else base
    del model; torch.cuda.empty_cache()
    if 1 in todo:
        vals = np.array(list(base.values()))
        by_artist = {}
        for i, v in base.items():
            by_artist.setdefault(artist_of[i], []).append(v)
        arts = list(by_artist)
        brng = np.random.default_rng(2)
        boot = [np.mean([v for k in brng.choice(len(arts), len(arts)) for v in by_artist[arts[k]]]) for _ in range(2000)]
        report["test1_original_shao"] = {
            "mean_surprise": round(float(vals.mean()), 4), "spread_between_songs": round(float(vals.std()), 4),
            "repeat_identical": all(base[i] == again[i] for i in base),
            "repeat_max_difference": float(max(abs(base[i] - again[i]) for i in base)),
            "average_range95_from_artist_reshuffling": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])]}
        log("test1", **report["test1_original_shao"])

    # ---------- 2. real training ----------
    if 2 in todo:
        model, curve = train(train_songs, eval_every=100, label="real")
        final = evaluate(model, eval_songs)
        del model; torch.cuda.empty_cache()
        report["test2_real_training"] = curve + [{"songs_seen": len(train_songs)} | compare(base, final, eval_songs)]
        log("test2_done", result=report["test2_real_training"][-1])

    # ---------- 3. garbage training (moments shuffled) ----------
    if 3 in todo:
        model, _ = train(train_songs, shuffle_frames=True, label="garbage")
        garbage = evaluate(model, eval_songs)
        del model; torch.cuda.empty_cache()
        report["test3_garbage_training"] = compare(base, garbage, eval_songs)
        log("test3_done", **report["test3_garbage_training"])

    # ---------- 4. cheating on purpose (train on half of the test songs) ----------
    if 4 in todo:
        model, _ = train(leak, label="cheat")
        cheat = evaluate(model, eval_songs)
        del model; torch.cuda.empty_cache()
        report["test4_cheat"] = {"trained_on_these_test_songs": compare(base, cheat, leak),
                                 "other_test_songs": compare(base, cheat, clean)}
        log("test4_done", **report["test4_cheat"])

    report["minutes"] = round((time.time() - t0) / 60, 1)
    b2.put_object(Bucket=B2_BUCKET, Key=out + "report.json", Body=json.dumps(report, indent=1).encode())
    log("done", minutes=report["minutes"])
    return report


@app.local_entrypoint()
def main(run_name: str = "v1", n_eval: int = 600, n_train: int = 300, lr: float = 2e-5, tests: str = "1,2,3,4"):
    # hand the job to the cloud and return: a laptop that sleeps or disconnects can't cancel it
    call = run.spawn(run_name, n_eval=n_eval, n_train=n_train, lr=lr, tests=tests)
    print(f"launched in the cloud: {call.object_id}; progress in B2 erised-sft/check_a_verify/{run_name}/log.jsonl")
