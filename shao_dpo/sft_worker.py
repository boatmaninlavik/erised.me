"""One GPU's share of the stage-1 SFT run (launched 8x by torchrun inside sft_train.py).

Every GPU holds a full copy of Shao's backbone. Each step, each GPU trains on one different song;
gradients are averaged across the 8 GPUs (DistributedDataParallel), so one step = 8 songs.
Songs in the same step have similar lengths, so no GPU waits long for the others.

Training = next-token prediction on the audio tokens (prompt is context only), full model,
fp32 weights + bf16 compute, per-layer activation checkpointing (same math as the verified tests).
Check A (surprise on test songs from never-trained artists, paired vs original Shao) runs every
--eval_every steps across all 8 GPUs. GPU 0 writes the log to B2 and saves checkpoints to the volume.
"""
import argparse
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

sys.path[:0] = ["/root", "/root/khala"]
import dpo_common as dc                                       # noqa: E402
from core.khala_runtime import load_vanilla_model             # noqa: E402

EOS = 128001
ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True)    # not "--run": torchrun reads that as its own --run-path
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--max_steps", type=int, default=0)            # 0 = one full pass
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--n_eval", type=int, default=300)
ap.add_argument("--budget_minutes", type=float, default=50)    # hard stop (whole job, all GPUs)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda")
t_start = float(os.environ.get("JOB_START", time.time()))
run_dir = f"/data/sft_runs/{args.job}"
os.makedirs(run_dir, exist_ok=True)

# ---------------- data (packed by sft_train.prepare) ----------------
codes_all = np.load("/data/sft_data/codes.npy", mmap_mode="r")      # shared by all 8 processes, not copied
prompt_all = np.load("/data/sft_data/prompts.npy", mmap_mode="r")
meta = json.load(open("/data/sft_data/meta.json"))


def sequence(i):
    m = meta[i]
    codes = np.array(codes_all[m["c0"]:m["c0"] + m["frames"]])
    prompt = prompt_all[m["p0"]:m["p0"] + m["plen"]].tolist()
    ids, a0 = dc.build_sequence(prompt, codes)
    if len(ids) - a0 == 2 * len(codes) and len(ids) < dc.CONTEXT_LEN:   # whole song fits -> learn where it ends
        ids = torch.cat([ids, torch.tensor([EOS])])
    return ids.to(dev, non_blocking=True), a0


train_idx = [i for i, m in enumerate(meta) if m["split"] == "train"]
rng = random.Random(args.seed)
rng.shuffle(train_idx)
steps = []
for c in range(0, len(train_idx), world * 64):                 # similar lengths within a step
    chunk = sorted(train_idx[c:c + world * 64], key=lambda i: meta[i]["frames"])
    steps += [chunk[j:j + world] for j in range(0, len(chunk), world)]
rng.shuffle(steps)
steps = [s for s in steps if len(s) == world]
if args.max_steps:
    steps = steps[:args.max_steps]

# check A songs: test split (artists never trained on), shuffled with seed 0, at most 3 per artist
test_idx = [i for i, m in enumerate(meta) if m["split"] == "test"]
random.Random(0).shuffle(test_idx)
per, eval_idx = {}, []
for i in test_idx:
    a = meta[i]["user_id"]
    if per.get(a, 0) < 3 and len(eval_idx) < args.n_eval:
        eval_idx.append(i)
        per[a] = per.get(a, 0) + 1


# ---------------- model ----------------
def chunk_lp(h, w, t):
    logits = (h @ w.t()).float()
    return logits.gather(-1, t[:, None]).squeeze(-1) - torch.logsumexp(logits, -1)


class SongLogProb(torch.nn.Module):
    """Wrapper so DistributedDataParallel sees one forward call per song."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, a0, grad=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = dc.hidden_states(self.m, ids[None], use_checkpoint=grad)[0]
            pred, tgt, w = h[a0 - 1:-1], ids[a0:], self.m.lm_head.weight[:dc.REAL_VOCAB]
            parts = [checkpoint(chunk_lp, pred[s:s + 1024], w, tgt[s:s + 1024], use_reentrant=False) if grad
                     else chunk_lp(pred[s:s + 1024], w, tgt[s:s + 1024]) for s in range(0, len(tgt), 1024)]
        return torch.cat(parts)


base_model = load_vanilla_model("backbone", "/data/weights/khala_backbone.safetensors",
                                "/data/weights/backbone_megatron_args.json", "cuda", torch.float32)
net = SongLogProb(base_model)
ddp = DDP(net, device_ids=[dev.index])
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
total = len(steps)


def lr_at(k):          # warm-up, then cosine down to 10%
    if k < args.warmup:
        return args.lr * (k + 1) / args.warmup
    p = (k - args.warmup) / max(1, total - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * p)))


# ---------------- logging (GPU 0 only) ----------------
log_rows = []
if rank == 0:
    import boto3
    b2 = boto3.client("s3", endpoint_url="https://s3.us-west-004.backblazeb2.com")


def log(event, flush=False, **kw):
    if rank != 0:
        return
    row = {"t_min": round((time.time() - t_start) / 60, 2), "event": event, **kw}
    print(json.dumps(row), flush=True)
    log_rows.append(row)
    if flush or len(log_rows) % 20 == 0:
        with open(f"{run_dir}/log.jsonl", "w") as f:
            f.write("\n".join(json.dumps(r) for r in log_rows) + "\n")
        try:
            b2.put_object(Bucket="erised-sft", Key=f"sft_runs/{args.job}/log.jsonl",
                          Body="\n".join(json.dumps(r) for r in log_rows).encode())
        except Exception as e:  # logging must never kill training
            print("B2 log upload failed:", e, flush=True)


# ---------------- check A across all GPUs ----------------
@torch.no_grad()
def check_a():
    net.eval()
    mine = {}
    for i in eval_idx[rank::world]:
        ids, a0 = sequence(i)
        lp = net(ids, a0, grad=False)
        n_audio = len(ids) - a0 - (1 if ids[-1] == EOS else 0)
        mine[meta[i]["id"]] = float(-lp[:n_audio].mean())
    net.train()
    parts = [None] * world
    dist.all_gather_object(parts, mine)
    out = {}
    for p in parts:
        out.update(p)
    return out


def compare(base, new):
    ids = list(base)
    artist = {meta[i]["id"]: meta[i]["user_id"] for i in eval_idx}
    diff = {i: new[i] - base[i] for i in ids}
    arts = sorted({artist[i] for i in ids})
    by = {a: [diff[i] for i in ids if artist[i] == a] for a in arts}
    r = np.random.default_rng(1)
    boots = [np.mean([v for k in r.choice(len(arts), len(arts)) for v in by[arts[k]]]) for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_change": round(float(np.mean(list(diff.values()))), 5), "range95": [round(float(lo), 5), round(float(hi), 5)],
            "songs_improved": round(float(np.mean([v < 0 for v in diff.values()])), 3)}


def save_bf16(tag):
    if rank == 0:
        from safetensors.torch import save_file
        t0 = time.time()
        save_file({k: v.detach().to(torch.bfloat16).contiguous() for k, v in base_model.state_dict().items()},
                  f"{run_dir}/{tag}.safetensors")
        log("checkpoint_saved", flush=True, tag=tag, seconds=round(time.time() - t0, 1))
    dist.barrier()


def weights_in_sync():
    """All 8 copies must stay identical; compare a checksum of a few weight matrices."""
    s = torch.stack([base_model.layers[i].attn.q_proj.weight.float().sum() for i in (0, 12, 23)])
    lo, hi = s.clone(), s.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    return bool(torch.equal(lo, hi))


# ---------------- run ----------------
log("start", flush=True, gpus=world, train_songs=len(train_idx), steps=total, songs_per_step=world,
    eval_songs=len(eval_idx), eval_artists=len(per), lr=args.lr, budget_minutes=args.budget_minutes)
base_scores = check_a()
log("check_a_original_shao", flush=True, mean_surprise=round(float(np.mean(list(base_scores.values()))), 4))
if rank == 0:
    json.dump(base_scores, open(f"{run_dir}/check_a_original.json", "w"))

watch = [base_model.layers[i].attn.q_proj.weight for i in (0, 12, 23)]
best, history, stop_reason = None, [], "finished one full pass"
t_train = time.time()
for k, step_songs in enumerate(steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(k)
    t0 = time.time()
    ids, a0 = sequence(step_songs[rank])
    lp = ddp(ids, a0)
    n_audio = len(ids) - a0 - (1 if ids[-1] == EOS else 0)
    loss = -lp.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
    before = [w.detach().clone() for w in watch] if k % 25 == 0 else None
    opt.step()
    stats = torch.tensor([float(loss), float(-lp[:n_audio][0::2].mean()), float(-lp[:n_audio][1::2].mean()),
                          len(ids)], device=dev)
    dist.all_reduce(stats)
    stats /= world
    row = {"step": k + 1, "songs_seen": (k + 1) * world, "lr": float(f"{lr_at(k):.3e}"), "loss": round(stats[0].item(), 4),
           "loss_q0": round(stats[1].item(), 4), "loss_q1": round(stats[2].item(), 4), "grad_norm": round(gnorm, 3),
           "step_seconds": round(time.time() - t0, 2), "avg_tokens": int(stats[3].item())}
    if before is not None:
        row["update_to_weight"] = float(f"{np.mean([float((w.detach() - b).norm() / w.detach().norm()) for w, b in zip(watch, before)]):.2e}")
    log("train_step", **row)
    if k == 0 or (k + 1) % args.eval_every == 0 or k + 1 == total:
        in_sync = weights_in_sync()
        scores = check_a()
        c = compare(base_scores, scores)
        history.append({"step": k + 1} | c)
        log("check_a", flush=True, step=k + 1, songs_seen=(k + 1) * world, weights_in_sync=in_sync, **c)
        if best is None or c["mean_change"] < best["mean_change"]:
            best = {"step": k + 1} | c
        if k + 1 >= args.eval_every and c["mean_change"] > 0:
            stop_reason = "check A got worse than original Shao"
    if (k + 1) % args.save_every == 0 and k + 1 < total:
        save_bf16(f"step_{k + 1:05d}")
    elapsed = (time.time() - t_start) / 60
    per_step = (time.time() - t_train) / (k + 1) / 60
    stop = torch.tensor([1.0 if (stop_reason != "finished one full pass" or elapsed + per_step + 3 > args.budget_minutes) else 0.0], device=dev)
    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
    if stop.item() > 0 and k + 1 < total:
        if stop_reason == "finished one full pass":
            stop_reason = f"time budget ({args.budget_minutes} min) reached"
        break

save_bf16("final")
log("done", flush=True, stop_reason=stop_reason, steps_done=k + 1, songs_seen=(k + 1) * world, best=best,
    minutes=round((time.time() - t_start) / 60, 1))
if rank == 0:
    json.dump({"run": args.job, "stop_reason": stop_reason, "steps_done": k + 1, "songs_seen": (k + 1) * world,
               "check_a_history": history, "best": best}, open(f"{run_dir}/summary.json", "w"), indent=1)
dist.destroy_process_group()
