"""Stage-1 SFT of Shao's backbone on the Suno songs, on Modal (workspace erised1).

  prepare  (CPU, once):  pack all 23,131 token files + prompts into one fast file on the volume, cache weights
  smoke    (2 x H100):   20 steps: checks multi-GPU training, checkpoint saving, check A, logging
  full     (8 x H100):   one pass over the training songs (20,816), check A every --eval-every steps,
                         checkpoints every --save-every steps, hard stop at --budget-minutes (cost cap)

Detailed log while it runs:
  MODAL_PROFILE=erised1 modal app logs <app id printed at launch>             (live, every step)
  MODAL_PROFILE=erised1 modal volume get shao-sft sft_runs/<run>/log.jsonl .   (the structured log file)
  B2 erised-sft/sft_runs/<run>/log.jsonl                                       (same file, copied every 20 steps)

Run: MODAL_PROFILE=erised1 modal run shao_dpo/sft_train.py::prepare
     MODAL_PROFILE=erised1 modal run --detach shao_dpo/sft_train.py::smoke
     MODAL_PROFILE=erised1 modal run --detach shao_dpo/sft_train.py::full --run-name sft_v1
     MODAL_PROFILE=erised1 modal run shao_dpo/sft_train.py::upload --run-name sft_v1      (weights -> B2, CPU)
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
B2_BUCKET, B2_ENDPOINT, TOK_DIR = "erised-sft", "https://s3.us-west-004.backblazeb2.com", "shao_tokens/v1/"
MPS_REPO = "Vinpolar/Khala-MusicGeneration-v1.0-MPS"

app = modal.App("shao-sft")
vol = modal.Volume.from_name("shao-sft", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy<2", "boto3", "huggingface_hub", "safetensors", "transformers==4.44.2")
    .add_local_dir(HERE.parent / "khala_runtime", remote_path="/root/khala")
    .add_local_file(HERE / "dpo_common.py", "/root/dpo_common.py")
    .add_local_file(HERE / "sft_worker.py", "/root/sft_worker.py")
)
secrets = [modal.Secret.from_name("b2-key")]


def _b2():
    import boto3
    return boto3.client("s3", endpoint_url=B2_ENDPOINT)


@app.function(image=image, cpu=8, memory=32768, volumes={"/data": vol}, secrets=secrets, timeout=3600)
def prepare():
    import io, shutil, sys
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    sys.path[:0] = ["/root", "/root/khala"]
    import dpo_common as dc

    Path("/data/weights").mkdir(parents=True, exist_ok=True)
    for f in ("khala_backbone.safetensors", "backbone_megatron_args.json"):
        if not Path(f"/data/weights/{f}").exists():
            shutil.copyfile(os.path.realpath(hf_hub_download(MPS_REPO, f, cache_dir="/tmp/hf")), f"/data/weights/{f}")
    b2 = _b2()
    rows = [json.loads(l) for l in b2.get_object(Bucket=B2_BUCKET, Key=TOK_DIR + "manifest.jsonl")["Body"].read().decode().split("\n") if l.strip()]
    print(f"{len(rows)} songs in the manifest")

    def fetch(r):
        a = np.load(io.BytesIO(b2.get_object(Bucket=B2_BUCKET, Key=r["q01_key"])["Body"].read()))
        assert a.shape == (r["frames"], 2) and a.dtype == np.int16, (r["id"], a.shape)
        return a

    t0 = time.time()
    with ThreadPoolExecutor(64) as pool:
        arrays = list(pool.map(fetch, rows))
    print(f"downloaded {len(arrays)} token files in {time.time() - t0:.0f} s")

    tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
    meta, prompts, c0, p0 = [], [], 0, 0
    for r, a in zip(rows, arrays):
        minutes = int(min(10, max(1, round(r["seconds"] / 60))))
        text = dc.build_prompt_text(r.get("style_prompt") or r.get("description_prompt") or "", r.get("lyrics") or "",
                                    bool(r.get("instrumental")), minutes)
        p = tok.encode(text, add_special_tokens=False)
        if len(p) > 4096:                              # Shao's own limit: keep the start and the final marker
            p = p[:4095] + [p[-1]]
        meta.append({"id": r["id"], "split": r["split"], "user_id": r["user_id"], "frames": r["frames"],
                     "c0": c0, "p0": p0, "plen": len(p), "likes": r.get("likes"), "version": r.get("major_model_version")})
        prompts.append(np.array(p, dtype=np.int32))
        c0 += r["frames"]
        p0 += len(p)
    Path("/data/sft_data").mkdir(parents=True, exist_ok=True)
    np.save("/data/sft_data/codes.npy", np.concatenate(arrays))
    np.save("/data/sft_data/prompts.npy", np.concatenate(prompts))
    json.dump(meta, open("/data/sft_data/meta.json", "w"))
    vol.commit()
    print(f"packed: {c0} frames, {p0} prompt tokens, train {sum(m['split'] == 'train' for m in meta)} / "
          f"test {sum(m['split'] == 'test' for m in meta)} songs")


@app.function(image=image, cpu=8, memory=32768, volumes={"/data": vol}, secrets=secrets, timeout=3600)
def prepare_list(list_key: str, out_dir: str = "/data/sft_data_v2"):
    """Pack a song list (e.g. B2 selected/sft2/frozen_*.jsonl, built by build_sft2_list.py) for training.
    Prompts come from the list (cleaned lyrics), tokens from shao_tokens/v1/q01/<id>.npy, the split from the
    list (first-run test creators stay in test). Songs without a token file are skipped and counted."""
    import io, shutil, sys
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    sys.path[:0] = ["/root", "/root/khala"]
    import dpo_common as dc

    Path("/data/weights").mkdir(parents=True, exist_ok=True)
    for f in ("khala_backbone.safetensors", "backbone_megatron_args.json"):
        if not Path(f"/data/weights/{f}").exists():
            shutil.copyfile(os.path.realpath(hf_hub_download(MPS_REPO, f, cache_dir="/tmp/hf")), f"/data/weights/{f}")
    b2 = _b2()
    rows = [json.loads(l) for l in b2.get_object(Bucket=B2_BUCKET, Key=list_key)["Body"].read().decode().split("\n") if l.strip()]

    def fetch(r):
        try:
            a = np.load(io.BytesIO(b2.get_object(Bucket=B2_BUCKET, Key=f"{TOK_DIR}q01/{r['id']}.npy")["Body"].read()))
            assert a.ndim == 2 and a.shape[1] == 2 and a.dtype == np.int16, a.shape
            return a
        except b2.exceptions.NoSuchKey:
            return None

    t0 = time.time()
    with ThreadPoolExecutor(64) as pool:
        arrays = list(pool.map(fetch, rows))
    print(f"downloaded {sum(a is not None for a in arrays)} token files in {time.time() - t0:.0f} s", flush=True)
    tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
    meta, prompts, kept, c0, p0, missing = [], [], [], 0, 0, 0
    for r, a in zip(rows, arrays):
        if a is None:
            missing += 1
            continue
        minutes = int(min(10, max(1, round(len(a) / (44100 / 2048) / 60))))
        text = dc.build_prompt_text(r.get("style_prompt") or r.get("description_prompt") or "", r.get("lyrics") or "",
                                    bool(r.get("instrumental")), minutes)
        p = tok.encode(text, add_special_tokens=False)
        if len(p) > 4096:                              # Shao's own limit: keep the start and the final marker
            p = p[:4095] + [p[-1]]
        meta.append({"id": r["id"], "split": r["split"], "user_id": r["user_id"], "frames": int(len(a)),
                     "c0": c0, "p0": p0, "plen": len(p), "likes": (r.get("selection_metrics") or {}).get("likes"),
                     "version": r.get("major_model_version")})
        prompts.append(np.array(p, dtype=np.int32))
        kept.append(a)
        c0 += len(a)
        p0 += len(p)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/codes.npy", np.concatenate(kept))
    np.save(f"{out_dir}/prompts.npy", np.concatenate(prompts))
    json.dump(meta, open(f"{out_dir}/meta.json", "w"))
    vol.commit()
    summary = {"list": list_key, "songs": len(meta), "missing_tokens": missing, "frames": c0,
               "hours": round(c0 / (44100 / 2048) / 3600, 1), "train": sum(m["split"] == "train" for m in meta),
               "test": sum(m["split"] == "test" for m in meta), "out_dir": out_dir}
    print(json.dumps(summary), flush=True)
    return summary


def _train(gpus, run, lr, max_steps, eval_every, save_every, n_eval, budget_minutes, data="/data/sft_data"):
    env = dict(os.environ, JOB_START=str(time.time()), PYTHONUNBUFFERED="1")
    stop = threading.Event()

    def keep_committing():        # make checkpoints and logs durable even if the job dies mid-way
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:
                print("volume commit failed:", e, flush=True)

    threading.Thread(target=keep_committing, daemon=True).start()
    cmd = ["torchrun", f"--nproc_per_node={gpus}", "/root/sft_worker.py", "--job", run, "--lr", str(lr),
           "--max_steps", str(max_steps), "--eval_every", str(eval_every), "--save_every", str(save_every),
           "--n_eval", str(n_eval), "--budget_minutes", str(budget_minutes), "--data", data]
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env)
    stop.set()
    vol.commit()
    run_dir = Path(f"/data/sft_runs/{run}")
    b2 = _b2()
    for name in ("summary.json", "log.jsonl", "check_a_original.json"):     # small files only; weights via ::upload (CPU)
        p = run_dir / name
        if p.exists():
            try:
                b2.upload_file(str(p), B2_BUCKET, f"sft_runs/{run}/{name}")
                print(f"uploaded {name} to B2", flush=True)
            except Exception as e:
                print(f"B2 upload of {name} failed (kept on the volume): {e}", flush=True)
    if result.returncode:
        raise RuntimeError(f"training exited with code {result.returncode}")


@app.function(image=image, gpu="H100:2", cpu=8, memory=65536, volumes={"/data": vol}, secrets=secrets, timeout=3600)
def train_small(run: str, lr: float, max_steps: int, eval_every: int, save_every: int, n_eval: int, budget_minutes: float,
                data: str = "/data/sft_data"):
    _train(2, run, lr, max_steps, eval_every, save_every, n_eval, budget_minutes, data)


@app.function(image=image, gpu="H100:8", cpu=16, memory=163840, volumes={"/data": vol}, secrets=secrets, timeout=3 * 3600)
def train(run: str, lr: float, max_steps: int, eval_every: int, save_every: int, n_eval: int, budget_minutes: float,
          data: str = "/data/sft_data"):
    _train(8, run, lr, max_steps, eval_every, save_every, n_eval, budget_minutes, data)


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, secrets=secrets, timeout=3600)
def upload_weights(run: str, tag: str = "final"):
    """Copy a saved checkpoint from the volume to B2 on a cheap CPU machine (not on idle GPUs)."""
    p = Path(f"/data/sft_runs/{run}/{tag}.safetensors")
    t0 = time.time()
    _b2().upload_file(str(p), B2_BUCKET, f"sft_runs/{run}/{tag}.safetensors")
    print(f"uploaded {p.name} ({p.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.0f} s")


@app.local_entrypoint()
def upload(run_name: str = "sft_v1", tag: str = "final"):
    upload_weights.remote(run_name, tag)


@app.local_entrypoint()
def smoke():
    call = train_small.spawn("smoke", 2e-5, 20, 10, 10, 40, 20)
    print(f"smoke test launched: {call.object_id}")


@app.local_entrypoint()
def full(run_name: str = "sft_v1", lr: float = 2e-5, budget_minutes: float = 45):
    call = train.spawn(run_name, lr, 0, 325, 650, 300, budget_minutes)
    print(f"full run launched: {call.object_id} (safe to close the laptop)")


# ------------------------------------------------------------------ second run (sft_v2): cleaned-lyrics list
V2_DATA = "/data/sft_data_v2"


@app.local_entrypoint()
def prepare2(list_key: str = "selected/sft2/frozen_2026-10-03.jsonl"):
    print(json.dumps(prepare_list.remote(list_key, V2_DATA), indent=1))


@app.local_entrypoint()
def smoke2():
    call = train_small.spawn("smoke_v2", 2e-5, 20, 10, 10, 40, 20, V2_DATA)
    print(f"v2 smoke test launched: {call.object_id}")


@app.local_entrypoint()
def full2(run_name: str = "sft_v2", lr: float = 2e-5, budget_minutes: float = 90):
    call = train.spawn(run_name, lr, 0, 400, 800, 300, budget_minutes, V2_DATA)
    print(f"v2 full run launched: {call.object_id} (safe to close the laptop)")
