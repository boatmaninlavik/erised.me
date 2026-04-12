#!/usr/bin/env python3
"""
Merge original and DPO weights: final = alpha * original + (1-alpha) * DPO.
This reduces distribution drift from partial fine-tuning while keeping DPO preferences.
"""
import os, sys, glob, argparse
from pathlib import Path
from safetensors.torch import load_file, save_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="/workspace/heartlib/ckpt")
    parser.add_argument("--dpo", default="/workspace/dpo_checkpoints_v6/dpo_best")
    parser.add_argument("--output", default="/workspace/dpo_checkpoints_v6_merged")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for original (0.7 = 70%% original, 30%% DPO)")
    args = parser.parse_args()

    print(f"Merging: {args.alpha:.0%} original + {1-args.alpha:.0%} DPO")
    print(f"Original: {args.original}")
    print(f"DPO: {args.dpo}")
    print(f"Output: {args.output}")

    # Load original weights
    orig_files = sorted(glob.glob(os.path.join(args.original, "*.safetensors")))
    if not orig_files:
        orig_files = sorted(glob.glob(os.path.join(args.original, "*", "*.safetensors")))
    print(f"Loading original from {len(orig_files)} files...")
    orig_state = {}
    for f in orig_files:
        orig_state.update(load_file(f))

    # Load DPO weights
    dpo_files = sorted(glob.glob(os.path.join(args.dpo, "*.safetensors")))
    if not dpo_files:
        dpo_files = sorted(glob.glob(os.path.join(args.dpo, "*", "*.safetensors")))
    print(f"Loading DPO from {len(dpo_files)} files...")
    dpo_state = {}
    for f in dpo_files:
        dpo_state.update(load_file(f))

    # Merge: only blend keys that exist in both
    merged = {}
    changed = 0
    for key in dpo_state:
        if key in orig_state and dpo_state[key].shape == orig_state[key].shape:
            # Check if weights actually differ
            diff = (dpo_state[key] - orig_state[key]).abs().max().item()
            if diff > 1e-8:
                merged[key] = args.alpha * orig_state[key] + (1 - args.alpha) * dpo_state[key]
                changed += 1
            else:
                merged[key] = dpo_state[key]
        else:
            merged[key] = dpo_state[key]

    print(f"Blended {changed} tensors that differed between original and DPO")

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "model.safetensors")
    print(f"Saving merged model to {out_path}...")
    save_file(merged, out_path)
    print("Done!")

if __name__ == "__main__":
    main()
