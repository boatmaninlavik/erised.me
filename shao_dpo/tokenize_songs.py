"""Convert the stage-1 SFT songs (B2 bucket `erised-sft`) into Shao codec tokens, on Modal.

Per song: download from Backblaze -> ffmpeg decode to 44.1 kHz stereo float32 (Suno audio is 48 kHz)
-> Shao's own codec encoder (DacRVQ + golden yaml + dac_rvq_2490000.ckpt from Hugging Face
liujiafeng/Shao-MusicGeneration-v1.0) -> 64 codes per frame, 21.53 frames per second.

Encoding runs in ~30 s windows with ~2 s of extra audio on each side, cut on 2048-sample frame
boundaries, so every kept frame sees the same audio as a whole-song encode (the encoder only looks
~0.6 s around each frame). `smoke` checks this against real whole-song encodes.

Outputs
  Modal volume `shao-tokens`  /data/tokens64/<id>.npy          int16 [frames, 64]  every codec layer
  B2 `erised-sft`             shao_tokens/v1/q01/<id>.npy      int16 [frames, 2]   layers 0-1 (what the backbone trains on)
                              shao_tokens/v1/manifest.jsonl    one line per song: prompt fields, split, frames
                              shao_tokens/v1/source_list.jsonl the frozen song list this run used

Run (workspace erised3):
  MODAL_PROFILE=erised3 modal run shao_dpo/tokenize_songs.py::prepare   # cache codec weights (once)
  MODAL_PROFILE=erised3 modal run shao_dpo/tokenize_songs.py::smoke     # 8 songs: speed, checks, listening clip
  MODAL_PROFILE=erised3 modal run shao_dpo/tokenize_songs.py::full      # launches in the cloud; laptop can close
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
B2_BUCKET, B2_ENDPOINT = "erised-sft", "https://s3.us-west-004.backblazeb2.com"
OUT = "shao_tokens/v1/"
FRAME = 2048                 # codec hop in samples (strides 4*8*8*8)
CORE = 640 * FRAME           # ~29.7 s of frames kept per window
CTX = 44 * FRAME             # ~2.0 s of context on each side
SR = 44100
GPU = "L4"

app = modal.App("shao-tokenize")
vol = modal.Volume.from_name("shao-tokens", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", "numpy<2", "omegaconf", "einops", "lightning", "boto3",
                 "huggingface_hub", "soundfile")
    .add_local_dir(HERE.parent / "khala_runtime/models/Decoder", remote_path="/root/codec")
)
secrets = [modal.Secret.from_name("b2-key")]   # Shao weights are public on Hugging Face


def _b2():
    import boto3
    return boto3.client("s3", endpoint_url=B2_ENDPOINT)


def _read_jsonl(b2, key):
    body = b2.get_object(Bucket=B2_BUCKET, Key=key)["Body"].read().decode()
    return [json.loads(line) for line in body.split("\n") if line.strip()]


def _song_list(b2):
    """Freeze the current selection: newest collector balanced list + the June additions."""
    keys = []
    for page in b2.get_paginator("list_objects_v2").paginate(Bucket=B2_BUCKET, Prefix="selected/balanced_versions/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    newest = max(keys, key=lambda k: int(Path(k).stem))
    songs = _read_jsonl(b2, newest) + _read_jsonl(b2, "june_additions/manifest.jsonl")
    seen, out = set(), []
    for s in songs:
        if s["id"] not in seen:
            seen.add(s["id"])
            out.append(s)
    return out, newest


@app.function(image=image, volumes={"/data": vol}, secrets=secrets, cpu=2, memory=16384, timeout=3600)
def prepare():
    """Download the 3.3 GB codec checkpoint once and keep only the generator weights (~small)."""
    import torch
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("liujiafeng/Shao-MusicGeneration-v1.0", "dac_rvq_2490000.ckpt", cache_dir="/tmp/hf")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    gen = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    Path("/data/codec").mkdir(parents=True, exist_ok=True)
    torch.save(gen, "/data/codec/generator.pt")
    vol.commit()
    print(f"saved {len(gen)} generator tensors, {sum(v.numel() for v in gen.values()) / 1e6:.0f}M numbers")


@app.cls(image=image, gpu=GPU, cpu=4, memory=16384, volumes={"/data": vol}, secrets=secrets, timeout=3600, max_containers=10)
class Tokenizer:
    @modal.enter()
    def load(self):
        import sys
        import torch
        from omegaconf import OmegaConf
        sys.path.insert(0, "/root/codec")
        from dac_rvq import DacRVQ
        self.torch = torch
        dac = DacRVQ(OmegaConf.load("/root/codec/dac_rvq_1024_64_golden.yaml"))
        missing, unexpected = dac.load_state_dict(torch.load("/data/codec/generator.pt"), strict=False)
        # strict=False only tolerates non-weight extras; a missing encoder/quantizer/decoder weight would mean garbage codes
        assert not missing, f"codec weights missing: {missing[:5]}"
        self.dac = dac.eval().cuda()
        self.b2 = _b2()

    def decode_audio(self, key):
        import os, subprocess, tempfile
        import numpy as np
        suffix = Path(key).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(self.b2.get_object(Bucket=B2_BUCKET, Key=key)["Body"].read())
            src = f.name
        try:
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-f", "f32le", "-acodec", "pcm_f32le",
                                  "-ac", "2", "-ar", str(SR), "pipe:1"], capture_output=True, check=True).stdout
        finally:
            os.unlink(src)
        return np.frombuffer(pcm, dtype=np.float32).reshape(-1, 2).T.copy()   # [2, samples]

    def encode(self, wav, whole=False):
        """wav float32 [2, n] -> int16 [frames, 64]."""
        import numpy as np
        torch = self.torch
        frames = -(-wav.shape[1] // FRAME)
        wav = np.pad(wav, ((0, 0), (0, frames * FRAME - wav.shape[1])))
        with torch.no_grad():
            if whole:
                codes = self.dac.encode(torch.from_numpy(wav)[None].cuda())[:, 0]
            else:
                parts, total = [], frames * FRAME
                for s in range(0, total, CORE):
                    a, b = max(0, s - CTX), min(total, s + CORE + CTX)
                    c = self.dac.encode(torch.from_numpy(wav[:, a:b])[None].cuda())   # [64, 1, window frames]
                    f0, nf = (s - a) // FRAME, (min(s + CORE, total) - s) // FRAME
                    parts.append(c[:, 0, f0:f0 + nf])
                codes = torch.cat(parts, dim=1)
        assert codes.shape == (64, frames), codes.shape
        return codes.T.cpu().numpy().astype(np.int16)

    def save(self, song_id, codes):
        import io
        import numpy as np
        Path("/data/tokens64").mkdir(parents=True, exist_ok=True)
        np.save(f"/data/tokens64/{song_id}.npy", codes)
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(codes[:, :2]))
        self.b2.put_object(Bucket=B2_BUCKET, Key=f"{OUT}q01/{song_id}.npy", Body=buf.getvalue())

    @modal.method()
    def run_shard(self, songs):
        import os, time
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time()
        todo = [s for s in songs if not os.path.exists(f"/data/tokens64/{s['id']}.npy")]
        done = [{"id": s["id"], "frames": None, "skipped": True} for s in songs if s not in todo]
        failed, ahead = [], 4
        # download + ffmpeg decode run on CPU threads a few songs ahead, while the GPU encodes the current one
        with ThreadPoolExecutor(3) as pool:
            queue = [pool.submit(self.decode_audio, s["audio"]["key"]) for s in todo[:ahead]]
            for i, s in enumerate(todo):
                if i + ahead < len(todo):
                    queue.append(pool.submit(self.decode_audio, todo[i + ahead]["audio"]["key"]))
                try:
                    codes = self.encode(queue[i].result())
                    self.save(s["id"], codes)
                    done.append({"id": s["id"], "frames": int(codes.shape[0])})
                except Exception as e:  # keep going; failures are listed in failed.json
                    failed.append({"id": s["id"], "error": repr(e)[:300]})
                queue[i] = None
        vol.commit()
        return {"done": done, "failed": failed, "seconds": time.time() - t0}

    @modal.method()
    def smoke(self):
        """8 songs: speed, chunked-vs-whole agreement, and a round-trip listening clip."""
        import io, time
        import numpy as np
        import soundfile as sf
        torch = self.torch
        pool, _ = _song_list(self.b2)
        dur = lambda s: s["audio"].get("duration_seconds") or 0
        songs = ([s for s in pool if dur(s) > 150][:4] + [s for s in pool if 0 < dur(s) <= 100][:3]
                 + [s for s in pool if s["audio"]["key"].endswith(".mp3")][:1])
        report = {"songs": []}
        for s in songs:
            t0 = time.time()
            wav = self.decode_audio(s["audio"]["key"])
            t1 = time.time()
            codes = self.encode(wav)
            t2 = time.time()
            row = {"id": s["id"], "seconds_audio": wav.shape[1] / SR, "decode_s": t1 - t0, "encode_s": t2 - t1,
                   "frames": codes.shape[0], "expected_frames": -(-wav.shape[1] // FRAME)}
            if wav.shape[1] <= 100 * SR:        # whole-song encode fits in memory for shorter songs
                whole = self.encode(wav, whole=True)
                row["chunk_vs_whole_match_by_layer"] = [round(float((whole[:, q] == codes[:, q]).mean()), 5) for q in (0, 1, 2, 8, 32, 63)]
            report["songs"].append(row)
            print(json.dumps(row))
            torch.cuda.empty_cache()
        # round trip on a 60 s excerpt of the first song (decoding a whole song at once needs >22 GB):
        # tokens -> audio, compared with the original over the middle 20 s
        torch.cuda.empty_cache()
        wav = self.decode_audio(songs[0]["audio"]["key"])[:, 20 * SR: 20 * SR + 60 * FRAME * 21]
        codes = self.encode(wav)
        with torch.no_grad():
            rec = self.dac.decode(torch.from_numpy(codes.T.astype(np.int64))[:, None].cuda())[0].cpu().numpy()
        seg = slice(10 * SR, 30 * SR)
        err = wav[:, seg] - rec[:, seg]
        report["round_trip_snr_db"] = float(10 * np.log10((wav[:, seg] ** 2).mean() / max(1e-12, (err ** 2).mean())))
        for name, audio in (("original", wav), ("from_tokens", rec)):
            buf = io.BytesIO()
            sf.write(buf, audio[:, seg].T, SR, format="WAV", subtype="PCM_16")
            self.b2.put_object(Bucket=B2_BUCKET, Key=f"{OUT}checks/{songs[0]['id']}_{name}.wav", Body=buf.getvalue())
        report["listen"] = f"{OUT}checks/{songs[0]['id']}_(original|from_tokens).wav"
        return report


def _done_on_b2(b2):
    """{song id: file size} for every 2-layer token file already in Backblaze (from any workspace)."""
    done = {}
    for page in b2.get_paginator("list_objects_v2").paginate(Bucket=B2_BUCKET, Prefix=OUT + "q01/"):
        for o in page.get("Contents", []):
            done[Path(o["Key"]).stem] = o["Size"]
    return done


@app.function(image=image, volumes={"/data": vol}, secrets=secrets, cpu=1, timeout=12 * 3600)
def orchestrate(shard_size: int = 100, limit: int = 0, refreeze: bool = False, extend_with: str = ""):
    """Runs in the cloud: uses the frozen song list (or freezes a new one), converts only songs not yet in
    Backblaze, fans shards out to GPU workers, then writes the manifest for everything converted so far.
    Resumes across workspaces: 'already done' is read from Backblaze, not from this workspace's volume.
    extend_with=<manifest key>: add that manifest's songs to the frozen list and convert ONLY the added ones
    (so it can run next to another workspace that is still finishing the old list)."""
    import io
    import numpy as np
    b2 = _b2()
    frozen = None if refreeze else b2.list_objects_v2(Bucket=B2_BUCKET, Prefix=f"{OUT}source_list.jsonl").get("KeyCount")
    only = None
    if frozen and extend_with:
        songs = _read_jsonl(b2, f"{OUT}source_list.jsonl")
        old = {s["id"] for s in songs}
        added = [s for s in _read_jsonl(b2, extend_with) if s["id"] not in old]
        songs += added
        only = {s["id"] for s in added}
        b2.put_object(Bucket=B2_BUCKET, Key=f"{OUT}source_list.jsonl",
                      Body="\n".join(json.dumps(s, ensure_ascii=False) for s in songs).encode())
        print(f"frozen list extended with {len(added)} songs from {extend_with} -> {len(songs)} total")
    elif frozen:
        songs = _read_jsonl(b2, f"{OUT}source_list.jsonl")
        print(f"{len(songs)} songs from the existing frozen list")
    else:
        songs, source = _song_list(b2)
        if limit:
            songs = songs[:limit]
        b2.put_object(Bucket=B2_BUCKET, Key=f"{OUT}source_list.jsonl",
                      Body="\n".join(json.dumps(s, ensure_ascii=False) for s in songs).encode())
        print(f"{len(songs)} songs frozen from {source} + june_additions")
    done = _done_on_b2(b2)
    todo = [s for s in songs if s["id"] not in done and (only is None or s["id"] in only)]
    print(f"already converted: {len(songs) - len([s for s in songs if s['id'] not in done])} | to do now: {len(todo)}")
    shards = [todo[i:i + shard_size] for i in range(0, len(todo), shard_size)]
    failed = []
    for k, res in enumerate(Tokenizer().run_shard.map(shards, order_outputs=False) if shards else [], 1):   # Modal's map crashes on an empty list
        failed += res["failed"]
        print(f"shard {k}/{len(shards)}: +{len(res['done'])} ok, +{len(res['failed'])} failed, {res['seconds']:.0f}s")
    done = _done_on_b2(b2)
    # frames from file size: int16 [T, 2] = 4 bytes/frame after the .npy header; checked on 20 real files
    sample = [s["id"] for s in songs if s["id"] in done][:20]
    head = {i: done[i] - 4 * np.load(io.BytesIO(b2.get_object(Bucket=B2_BUCKET, Key=f"{OUT}q01/{i}.npy")["Body"].read())).shape[0]
            for i in sample}
    assert len(set(head.values())) == 1, f"npy header sizes differ: {set(head.values())}"
    header = head[sample[0]]
    rows = []
    for s in songs:
        if s["id"] not in done:
            continue
        n = (done[s["id"]] - header) // 4
        rows.append({"id": s["id"], "split": s.get("split"), "frames": n, "seconds": round(n / (SR / FRAME), 2),
                     "q01_key": f"{OUT}q01/{s['id']}.npy", "style_prompt": s.get("style_prompt"),
                     "description_prompt": s.get("description_prompt"), "lyrics": s.get("lyrics"),
                     "instrumental": s.get("instrumental"), "major_model_version": s.get("major_model_version"),
                     "user_id": s.get("user_id"), "handle": s.get("handle"), "title": s.get("title"),
                     "likes": (s.get("selection_metrics") or {}).get("likes"),
                     "plays": (s.get("selection_metrics") or {}).get("plays"),
                     "source": s.get("source", "sft_collector")})
    b2.put_object(Bucket=B2_BUCKET, Key=f"{OUT}manifest.jsonl",
                  Body="\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode())
    b2.put_object(Bucket=B2_BUCKET, Key=f"{OUT}failed.json", Body=json.dumps(failed).encode())
    print(f"manifest: {len(rows)} songs tokenized, {len(failed)} failed")


@app.local_entrypoint()
def smoke():
    print(json.dumps(Tokenizer().smoke.remote(), indent=1))


@app.local_entrypoint()
def full(limit: int = 0, extend_with: str = ""):
    call = orchestrate.spawn(limit=limit, extend_with=extend_with)
    print(f"launched in the cloud: {call.object_id} (safe to close the laptop)")

