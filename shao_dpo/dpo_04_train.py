"""Step 4: DPO-train small LoRA add-ons on Shao's backbone, on Modal.

Reads everything from the Modal volume `shao-dpo` (filled by steps 1-2):
  /data/pairs/train.jsonl, /data/pairs/test.jsonl   one pair per line:
      {"pair_id", "creator", "weight",
       "prompt": {"style", "lyrics", "instrumental", "duration_min"},
       "winner": <song id>, "loser": <song id>}
  /data/tokens/<song id>.npy          int16 [frames, 2] = codec layers 0 and 1
  /data/weights/khala_backbone.safetensors, /data/weights/backbone_megatron_args.json
Writes /data/runs/<run>/: config.json, train_log.jsonl (every optimizer step),
  eval.jsonl (held-out check per epoch), lora_ep<N>.pt (~55 MB each).

Smoke test (8 pairs, 1 epoch): modal run shao_dpo/dpo_04_train.py --run-name smoke --smoke
Full run:                      modal run --detach shao_dpo/dpo_04_train.py --run-name r1
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("shao-dpo-train")
vol = modal.Volume.from_name("shao-dpo", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy", "safetensors", "transformers==4.44.2")
    .add_local_dir(HERE.parent / "khala_runtime", remote_path="/root/khala")
    .add_local_file(HERE / "dpo_common.py", "/root/dpo_common.py")
)


@app.function(image=image, gpu="H100", volumes={"/data": vol}, timeout=6 * 3600)
def train(run_name: str, epochs: int = 2, beta: float = 0.1, nll_weight: float = 0.2,
          lr: float = 1e-5, lora_r: int = 16, lora_alpha: float = 32, grad_accum: int = 8,
          smoke: bool = False, max_minutes: float = 150):
    import json, math, random, sys, time
    import numpy as np
    import torch
    sys.path[:0] = ["/root", "/root/khala"]
    import dpo_common as dc
    from core.khala_runtime import load_vanilla_model
    from transformers import AutoTokenizer

    t0 = time.time()
    random.seed(0)
    torch.manual_seed(0)
    out = Path(f"/data/runs/{run_name}")
    out.mkdir(parents=True, exist_ok=True)

    def read_pairs(name):
        return [json.loads(l) for l in open(f"/data/pairs/{name}.jsonl")]

    train_pairs, test_pairs = read_pairs("train"), read_pairs("test")
    if smoke:
        train_pairs, test_pairs, epochs = train_pairs[:8], test_pairs[:4], 1
    mean_w = sum(p.get("weight", 1.0) for p in train_pairs) / len(train_pairs)

    config = dict(run_name=run_name, epochs=epochs, beta=beta, nll_weight=nll_weight, lr=lr,
                  lora_r=lora_r, lora_alpha=lora_alpha, grad_accum=grad_accum, smoke=smoke,
                  n_train=len(train_pairs), n_test=len(test_pairs))
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(config)

    # ---- model: original Shao backbone in bf16 (its training precision) + fresh LoRA ----
    tok = AutoTokenizer.from_pretrained("/root/khala/models/Tokenizer", local_files_only=True)
    model = load_vanilla_model("backbone", "/data/weights/khala_backbone.safetensors",
                               "/data/weights/backbone_megatron_args.json", "cuda", torch.bfloat16)
    loras = dc.add_lora(model, lora_r, lora_alpha)
    params = [p for m in loras for p in (m.A, m.B)]
    print(f"trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.1f}M")

    prompt_cache = {}

    def sequences(pair):
        """(ids, audio_start) for winner and loser; both share the pair's one prompt."""
        key = pair["pair_id"]
        if key not in prompt_cache:
            prompt_cache[key] = tok.encode(dc.build_prompt_text(**pair["prompt"]), add_special_tokens=False)
        res = []
        for side in ("winner", "loser"):
            codes = np.load(f"/data/tokens/{pair[side]}.npy")
            ids, a0 = dc.build_sequence(prompt_cache[key], codes)
            res.append((ids.cuda(), a0))
        return res

    def per_token(ids, a0, grad):
        with torch.set_grad_enabled(grad):
            s, n = dc.audio_logp(model, ids, a0, use_checkpoint=grad)
        return s / n

    # ---- 1. reference scores: the SAME model with adapters off = original Shao ----
    ref = {}
    with torch.no_grad(), dc.lora_disabled(loras):
        for p in train_pairs + test_pairs:
            (w, w0), (l, l0) = sequences(p)
            ref[p["pair_id"]] = (float(per_token(w, w0, False)), float(per_token(l, l0, False)))
    print(f"reference scores for {len(ref)} pairs in {(time.time() - t0) / 60:.1f} min")

    # ---- 2. self-check: fresh adapters must reproduce the reference exactly ----
    (w, w0), _ = sequences(train_pairs[0])
    with torch.no_grad():
        drift0 = abs(float(per_token(w, w0, False)) - ref[train_pairs[0]["pair_id"]][0])
    assert drift0 < 1e-3, f"policy != reference before training ({drift0}); scoring paths disagree"

    def evaluate(pairs):
        """Held-out check. Accuracies compare winner vs loser; drift = how far the model
        moved from original Shao, in log-prob per token (0 = unchanged)."""
        rows = []
        with torch.no_grad():
            for p in pairs:
                (w, w0), (l, l0) = sequences(p)
                pw, pl = float(per_token(w, w0, False)), float(per_token(l, l0, False))
                rw, rl = ref[p["pair_id"]]
                _, st = dc.dpo_loss(torch.tensor(pw), torch.tensor(pl), rw, rl, beta, nll_weight)
                rows.append(dict(reward_correct=st["correct"], margin=st["margin"],
                                 base_prefers_winner=float(rw > rl), tuned_prefers_winner=float(pw > pl),
                                 drift=(abs(pw - rw) + abs(pl - rl)) / 2, change=((pw - rw) + (pl - rl)) / 2))
        return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]} | {"n": len(rows)}

    # ---- 3. training ----
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total_steps = max(1, epochs * math.ceil(len(train_pairs) / grad_accum))
    warmup = max(1, total_steps // 20)

    def lr_at(step):   # linear warm-up, then cosine down to 10% of lr
        if step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    log = open(out / "train_log.jsonl", "a")
    step, window, stop = 0, [], False
    for epoch in range(1, epochs + 1):
        order = train_pairs[:]
        random.shuffle(order)
        for i, p in enumerate(order):
            (w, w0), (l, l0) = sequences(p)
            rw, rl = ref[p["pair_id"]]
            loss, st = dc.dpo_loss(per_token(w, w0, True), per_token(l, l0, True), rw, rl, beta, nll_weight)
            (loss * p.get("weight", 1.0) / mean_w / grad_accum).backward()
            window.append(st)
            if (i + 1) % grad_accum == 0 or i + 1 == len(order):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
                row = {k: sum(s[k] for s in window) / len(window) for k in window[0]}
                row |= dict(step=step, epoch=epoch, lr=lr_at(step), grad_norm=grad_norm,
                            minutes=(time.time() - t0) / 60)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
                step, window = step + 1, []
                if (time.time() - t0) / 60 > max_minutes:
                    print(f"time budget {max_minutes} min reached; stopping early")
                    stop = True
                    break
        torch.save({"lora": dc.lora_state(loras), "config": config, "epoch": epoch,
                    "targets": dc.LORA_TARGETS}, out / f"lora_ep{epoch}.pt")
        ev = evaluate(test_pairs) | {"epoch": epoch, "split": "test"}
        ev_train = evaluate(train_pairs[:len(test_pairs)]) | {"epoch": epoch, "split": "train_sample"}
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps(ev) + "\n" + json.dumps(ev_train) + "\n")
        print("EVAL", json.dumps(ev), "\nEVAL-TRAIN", json.dumps(ev_train))
        vol.commit()
        if stop:
            break
    print(f"done in {(time.time() - t0) / 60:.1f} min")


@app.local_entrypoint()
def main(run_name: str, epochs: int = 2, beta: float = 0.1, nll_weight: float = 0.2,
         lr: float = 1e-5, smoke: bool = False, max_minutes: float = 150):
    train.remote(run_name, epochs=epochs, beta=beta, nll_weight=nll_weight, lr=lr,
                 smoke=smoke, max_minutes=max_minutes)
