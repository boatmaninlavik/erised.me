#!/usr/bin/env python3
"""
Test streaming speed optimizations.
Run on RunPod GPU pod:
    cd /workspace/erised_repo && python test_streaming_speed.py

Generates songs with different configs and reports timing.
Listen to output WAVs in /workspace/erised_data/test_speed/ to verify quality.
"""

import os
import sys
import time
import logging
import torch

sys.path.insert(0, "/workspace/erised_repo")
sys.path.insert(0, "/workspace/heartlib/src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("test_speed")

from erised.pipeline import ErisedPipeline, ErisedConfig

TEST_PROMPT = "emotional pop ballad with piano and strings"
TEST_LYRICS = """[Verse 1]
Walking through the rain tonight
Every shadow holds a light
Memories like falling stars
Tracing paths between the scars

[Chorus]
Hold me close don't let me go
In the dark you are my glow
Every heartbeat finds its way
Through the night into the day

[Verse 2]
Whispers carried on the wind
Tell me where we should begin
Silent promises we keep
Waking gently from our sleep"""

MAX_SEC = 50  # Long enough for 2+ chunks
OUTPUT_DIR = "/workspace/erised_data/test_speed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_test(pipeline, name, first_chunk=None, lean_gc=False):
    """Generate a song with specific streaming config and measure timing."""
    logger.info("=" * 60)
    logger.info("TEST: %s | first_chunk=%s | lean_gc=%s",
                name, first_chunk or "default(300)", lean_gc)
    logger.info("=" * 60)

    chunk_times = []

    def on_progress(current_frame, total_frames, partial_audio_file=None,
                    partial_version=None, chunk_paths=None):
        if partial_version is not None and partial_version > len(chunk_times):
            chunk_times.append(time.time())
            logger.info("  chunk %d ready at t=%.1fs",
                        partial_version, chunk_times[-1] - t0)

    t0 = time.time()
    result = pipeline.generate(
        prompt=TEST_PROMPT,
        lyrics=TEST_LYRICS,
        max_audio_length_ms=MAX_SEC * 1000,
        streaming_decode=True,
        streaming_first_chunk=first_chunk,
        streaming_lean_gc=lean_gc,
        on_progress=on_progress,
    )
    elapsed = time.time() - t0

    # Copy output to test dir with descriptive name
    import shutil
    dest = os.path.join(OUTPUT_DIR, f"{name}.wav")
    shutil.copy2(result.audio_path, dest)

    logger.info("-" * 40)
    logger.info("  Result: %s", dest)
    logger.info("  Total: %.1fs | Frames: %d", elapsed, result.num_frames)
    if len(chunk_times) >= 2:
        for i in range(1, len(chunk_times)):
            gap = chunk_times[i] - chunk_times[i - 1]
            logger.info("  Chunk %d→%d gap: %.1fs", i, i + 1, gap)
    logger.info("=" * 60)
    return elapsed, chunk_times


def main():
    logger.info("Loading pipeline...")
    config = ErisedConfig(
        model_path="/workspace/heartlib/ckpt",
        output_dir="/workspace/erised_data/test_speed",
    )
    pipeline = ErisedPipeline(config)
    logger.info("Pipeline loaded.\n")

    results = {}

    # Test 1: Baseline (current production settings)
    logger.info("\n>>> TEST 1: BASELINE <<<\n")
    results["baseline"] = run_test(pipeline, "1_baseline")

    # Test 2: Lean GC only
    logger.info("\n>>> TEST 2: LEAN GC <<<\n")
    results["lean_gc"] = run_test(pipeline, "2_lean_gc", lean_gc=True)

    # Test 3: Bigger first chunk (540 frames = 2 codec chunks = ~43s audio)
    logger.info("\n>>> TEST 3: FIRST_CHUNK=540 <<<\n")
    results["big_first"] = run_test(pipeline, "3_big_first_chunk",
                                     first_chunk=540)

    # Test 4: Lean GC + bigger first chunk
    logger.info("\n>>> TEST 4: LEAN GC + FIRST_CHUNK=540 <<<\n")
    results["lean_gc_big"] = run_test(pipeline, "4_lean_gc_big_first",
                                       first_chunk=540, lean_gc=True)

    # Test 5: torch.compile
    logger.info("\n>>> TEST 5: torch.compile <<<\n")
    try:
        logger.info("Compiling backbone...")
        pipeline.compile_backbone()
        logger.info("Compilation successful! Running test...")
        results["compiled"] = run_test(pipeline, "5_compiled",
                                        first_chunk=540, lean_gc=True)
    except Exception as e:
        logger.error("torch.compile FAILED: %s", e)
        logger.error("Skipping compile test.")
        results["compiled"] = None

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, data in results.items():
        if data is None:
            logger.info("  %-20s FAILED", name)
            continue
        elapsed, chunk_times = data
        gaps = []
        for i in range(1, len(chunk_times)):
            gaps.append(chunk_times[i] - chunk_times[i - 1])
        gap_str = ", ".join(f"{g:.1f}s" for g in gaps) if gaps else "N/A"
        logger.info("  %-20s total=%.1fs  chunk_gaps=[%s]", name, elapsed, gap_str)

    logger.info("\nOutput WAVs in: %s", OUTPUT_DIR)
    logger.info("Listen to each to verify quality (no repeated lyrics, no noise).")


if __name__ == "__main__":
    main()
