"""Check that the converted Shao tokens are correct and meaningful (reads the files the real run wrote).

Test 1  file check:     B2 q01/<id>.npy must equal columns 0-1 of the full 64-layer file.
Test 2  layer order:    in residual quantization each layer codes what the earlier ones missed, so the vector
                        each layer contributes should shrink with depth, and rebuilding audio from layers 0..k-1
                        should get closer to the real audio as k grows (Shao paper, Sec. 3.1-3.2).
Test 3  Shao judges it: Shao's backbone reads [text] + [q0_1, q1_1, q0_2, q1_2, ...] (paper Sec. 4.2). Real
                        tokens under their own prompt should be far more likely than the same tokens with
                        layers swapped, frames shuffled, random tokens, or another song's tokens.

Run: MODAL_PROFILE=erised3 modal run shao_dpo/verify_tokens.py
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
B2_BUCKET, B2_ENDPOINT, OUT, SR, FRAME = "erised-sft", "https://s3.us-west-004.backblazeb2.com", "shao_tokens/v1/", 44100, 2048

app = modal.App("shao-verify-tokens")
vol = modal.Volume.from_name("shao-tokens")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", "numpy<2", "omegaconf", "einops", "lightning", "boto3", "huggingface_hub",
                 "safetensors", "transformers==4.44.2")
    .add_local_dir(HERE.parent / "khala_runtime", remote_path="/root/khala")
    .add_local_file(HERE / "dpo_common.py", "/root/dpo_common.py")
)


@app.function(image=image, gpu="L4", memory=32768, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("b2-key")], timeout=3600)
def verify():
    import io, os, subprocess, sys, tempfile
    import boto3, numpy as np, torch
    from omegaconf import OmegaConf
    sys.path[:0] = ["/root", "/root/khala", "/root/khala/models/Decoder"]
    import dpo_common as dc
    from dac_rvq import DacRVQ
    from core.khala_runtime import load_vanilla_model
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    b2 = boto3.client("s3", endpoint_url=B2_ENDPOINT)
    get = lambda k: b2.get_object(Bucket=B2_BUCKET, Key=k)["Body"].read()
    source = {json.loads(l)["id"]: json.loads(l) for l in get(OUT + "source_list.jsonl").decode().split("\n") if l.strip()}
    vol.reload()
    done = {p.stem for p in Path("/data/tokens64").glob("*.npy")}
    pick = [s for s in source.values() if s["id"] in done and not s.get("instrumental")
            and len((s.get("lyrics") or "").strip()) > 200 and 150 < (s["audio"].get("duration_seconds") or 0) < 260][:3]
    report = {"songs": [s["id"] for s in pick]}

    # ---- Test 1: saved 2-layer file == first two columns of the 64-layer file ----
    t64 = {s["id"]: np.load(f"/data/tokens64/{s['id']}.npy") for s in pick}
    q01 = {s["id"]: np.load(io.BytesIO(get(f"{OUT}q01/{s['id']}.npy"))) for s in pick}
    report["test1_q01_equals_layers_0_1"] = all(np.array_equal(q01[i], t64[i][:, :2]) for i in t64)
    report["test1_shapes"] = {i: list(q01[i].shape) for i in q01}

    # ---- Test 2: layer strength and rebuild quality by depth ----
    dac = DacRVQ(OmegaConf.load("/root/khala/models/Decoder/dac_rvq_1024_64_golden.yaml"))
    dac.load_state_dict(torch.load("/data/codec/generator.pt"), strict=False)
    dac = dac.eval().cuda()
    s0 = pick[0]
    codes = torch.from_numpy(t64[s0["id"]].T.astype(np.int64)).cuda()          # [64, T]
    with torch.no_grad():
        norms = [float(getattr(dac.quantizer, f"vq_layer_{q}").decode(codes[q][None]).norm(dim=1).mean()) for q in range(64)]
    report["test2_layer_vector_size"] = {f"q{q}": round(norms[q], 3) for q in (0, 1, 2, 3, 7, 15, 31, 63)}
    report["test2_sizes_shrink_with_depth"] = bool(norms[0] > norms[1] > norms[2] and norms[1] > norms[31] > 0)
    with tempfile.NamedTemporaryFile(suffix=Path(s0["audio"]["key"]).suffix) as f:
        f.write(get(s0["audio"]["key"])); f.flush()
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", f.name, "-f", "f32le", "-ac", "2", "-ar", str(SR), "pipe:1"],
                             capture_output=True, check=True).stdout
    wav = np.frombuffer(pcm, dtype=np.float32).reshape(-1, 2).T
    f0, nf = 1400, 1300                                                          # ~65 s to ~125 s of the song
    ref = wav[:, f0 * FRAME:(f0 + nf) * FRAME]
    mid = slice(10 * SR, 50 * SR)
    snr = {}
    with torch.no_grad():
        for k in (1, 2, 4, 8, 16, 32, 64):
            rec = dac.decode(codes[:k, f0:f0 + nf][:, None])[0].cpu().numpy()[:, :ref.shape[1]]
            err = ref[:, mid] - rec[:, mid]
            snr[f"layers_0..{k - 1}"] = round(float(10 * np.log10((ref[:, mid] ** 2).mean() / (err ** 2).mean())), 2)
    report["test2_rebuild_snr_db"] = snr
    del dac
    torch.cuda.empty_cache()

    # ---- Test 3: Shao's backbone scores real vs broken tokens ----
    repo = "Vinpolar/Khala-MusicGeneration-v1.0-MPS"
    model = load_vanilla_model("backbone", hf_hub_download(repo, "khala_backbone.safetensors", cache_dir="/data/hf"),
                               hf_hub_download(repo, "backbone_megatron_args.json", cache_dir="/data/hf"), "cuda", torch.bfloat16)
    tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
    rng = np.random.default_rng(0)

    def prompt_ids(s, lyrics=None):
        minutes = int(min(10, max(1, round((s["audio"]["duration_seconds"] or 60) / 60))))
        text = dc.build_prompt_text(s.get("style_prompt") or s.get("description_prompt") or "",
                                    s["lyrics"] if lyrics is None else lyrics, bool(s.get("instrumental")), minutes)
        return tok.encode(text, add_special_tokens=False)

    def score(p_ids, codes2):
        ids, a0 = dc.build_sequence(p_ids, codes2)
        ids = ids.cuda()
        with torch.no_grad():
            h = dc.hidden_states(model, ids[None], False)[0]
            lp = dc._chunk_logp(h[a0 - 1:-1], model.lm_head.weight[:dc.REAL_VOCAB], ids[a0:])
        return {"all": round(float(lp.mean()), 3), "q0": round(float(lp[0::2].mean()), 3), "q1": round(float(lp[1::2].mean()), 3)}

    results = {}
    for j, s in enumerate(pick):
        other = pick[(j + 1) % len(pick)]
        real = q01[s["id"]]
        n = min(len(real), len(q01[other["id"]]))
        real, oth = real[:n], q01[other["id"]][:n]
        p = prompt_ids(s)
        results[s["id"]] = {
            "real tokens, own prompt": score(p, real),
            "real tokens, lyrics removed": score(prompt_ids(s, lyrics=""), real),
            "real tokens, other song's prompt": score(prompt_ids(other), real),
            "other song's tokens, this prompt": score(p, oth),
            "layers 0/1 swapped": score(p, real[:, ::-1].copy()),
            "frames shuffled in time": score(p, real[rng.permutation(n)]),
            "random tokens": score(p, rng.integers(0, 1024, real.shape).astype(np.int16)),
        }
    report["test3_avg_log_prob_per_audio_token"] = results
    report["test3_note"] = "higher (closer to 0) = Shao finds it more natural; chance level over 130,304 ids = -11.78"
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(verify.remote(), indent=1))
