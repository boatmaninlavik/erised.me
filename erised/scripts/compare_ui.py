#!/usr/bin/env python3
"""
A/B comparison UI for DPO-tuned vs original Erised.

Usage (on RunPod):
    ERISED_MODEL_PATH=/workspace/heartlib/ckpt python erised/scripts/compare_ui.py \
        --dpo-path /workspace/dpo_checkpoints/dpo_3ep_100pairs
"""

import argparse
import glob
import logging
import os
import sys
import time
import threading
from pathlib import Path

import shutil
import tempfile

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("erised.compare_ui")


def load_safetensors_sharded(model, model_path, device="cuda"):
    from safetensors.torch import load_file
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*", "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f, device=str(device)))
    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded %d tensors from %d file(s) in %s", len(state_dict), len(safetensor_files), model_path)


def copy_to_gradio_tmp(src_path):
    gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    os.makedirs(gradio_tmp, exist_ok=True)
    dst = os.path.join(gradio_tmp, os.path.basename(src_path))
    shutil.copy2(src_path, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(description="A/B comparison UI for Erised DPO")
    parser.add_argument("--original-path", type=str, default=None)
    parser.add_argument("--dpo-path", type=str, default="/workspace/dpo_checkpoints/dpo_best")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-length", type=int, default=60)
    args = parser.parse_args()

    import gradio as gr
    import torch
    from erised.config import ErisedConfig
    from erised.pipeline import ErisedPipeline

    config = ErisedConfig.from_env()
    original_model_path = args.original_path or config.model_path
    config.lazy_load = False

    logger.info("Loading pipeline from %s ...", original_model_path)
    pipeline = ErisedPipeline(config)
    device = next(pipeline.pipe.mula.parameters()).device
    logger.info("Pipeline loaded on %s", device)

    gen_lock = threading.Lock()
    current_weights = {"path": original_model_path}

    def swap_weights(target_path):
        if current_weights["path"] == target_path:
            return
        logger.info("Swapping weights -> %s", target_path)
        load_safetensors_sharded(pipeline.pipe.mula, target_path, device=device)
        current_weights["path"] = target_path
        torch.cuda.empty_cache()

    def generate_original(prompt, lyrics, max_sec, progress=gr.Progress(track_tqdm=True)):
        if not prompt.strip():
            raise gr.Error("Please enter a prompt.")
        if not lyrics.strip():
            raise gr.Error("Please enter lyrics.")
        with gen_lock:
            swap_weights(original_model_path)
            t0 = time.time()
            result = pipeline.generate(prompt=prompt, lyrics=lyrics, max_audio_length_ms=int(max_sec * 1000))
            elapsed = time.time() - t0
            logger.info("[original] Generated %s in %.1fs (%d frames)", result.audio_path, elapsed, result.num_frames)
        return copy_to_gradio_tmp(result.audio_path), f"Tags: {result.tags_used}"

    def generate_dpo(prompt, lyrics, max_sec, progress=gr.Progress(track_tqdm=True)):
        if not prompt.strip():
            return None, ""
        with gen_lock:
            swap_weights(args.dpo_path)
            t0 = time.time()
            result = pipeline.generate(prompt=prompt, lyrics=lyrics, max_audio_length_ms=int(max_sec * 1000))
            elapsed = time.time() - t0
            logger.info("[dpo] Generated %s in %.1fs (%d frames)", result.audio_path, elapsed, result.num_frames)
        return copy_to_gradio_tmp(result.audio_path), f"Tags: {result.tags_used}"

    def generate_single(model_choice, prompt, lyrics, max_sec, progress=gr.Progress(track_tqdm=True)):
        if not prompt.strip():
            raise gr.Error("Please enter a prompt.")
        path = args.dpo_path if model_choice == "DPO-tuned" else original_model_path
        with gen_lock:
            swap_weights(path)
            result = pipeline.generate(prompt=prompt, lyrics=lyrics, max_audio_length_ms=int(max_sec * 1000))
        return copy_to_gradio_tmp(result.audio_path), f"Tags: {result.tags_used} | Frames: {result.num_frames}"

    # ── CSS ────────────────────────────────────────────────────────────
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }

    /* Nuke every Gradio background */
    html, body, .gradio-container, .gradio-container *,
    #component-0, .main, .app {
        background: #0a0a0a !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    .gradio-container {
        max-width: 100% !important;
        padding: 40px 60px !important;
    }
    @media (min-width: 1200px) {
        .gradio-container { padding: 40px 10% !important; }
    }

    /* Kill every wrapper div */
    div, .block, .panel, .gr-block, .gr-box, .gr-panel, .gr-form,
    .gr-group, .gr-padded, .contain, .wrap, .gap, .form,
    div[class*="block"], div[class*="panel"], div[class*="group"],
    div[class*="gap"], div[class*="container"], div[class*="wrap"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Title */
    h1, .gr-heading { font-size: 2.25rem !important; font-weight: 700 !important; }

    /* Form card */
    .form-card {
        background: #151515 !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin-bottom: 24px !important;
    }

    /* Audio cards */
    .audio-card-section {
        background: #151515 !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin-bottom: 16px !important;
        border: 2px solid transparent !important;
        transition: border-color 0.2s !important;
    }
    .audio-card-section:hover { border-color: #333 !important; }

    /* Inputs — light grey, cozy */
    textarea, input[type=text] {
        width: 100% !important;
        background: #3d3845 !important;
        border: 1px solid #4a4555 !important;
        color: #e5e5e5 !important;
        padding: 14px !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        line-height: 1.5 !important;
    }
    textarea:focus, input:focus { outline: none !important; border-color: #a855f7 !important; }

    /* Labels — purple, bold, cozy */
    label, label > span, .gr-block-label, .label-wrap > span, .gr-input-label {
        color: #7c3aed !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        text-transform: none !important;
        background: transparent !important;
        letter-spacing: 0.02em !important;
    }

    /* Generate button — white block, purple text */
    #generate-both-btn, #generate-both-btn button, #generate-single-btn, #generate-single-btn button,
    .gen-btn button, .gen-btn .primary {
        width: 100% !important; min-height: 56px !important; padding: 18px !important;
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #7c3aed !important;
        font-size: 18px !important; font-weight: 600 !important;
        border-radius: 8px !important; cursor: pointer !important;
        display: block !important;
    }
    .gen-btn button:hover, .gen-btn .primary:hover { background: #f1f5f9 !important; border-color: #c4b5fd !important; }

    /* Tabs */
    .tab-nav { border-bottom: 1px solid #333 !important; background: transparent !important; }
    .tab-nav button {
        color: #888 !important; font-weight: 500 !important;
        border: none !important; background: transparent !important;
    }
    .tab-nav button.selected { color: #a855f7 !important; border-bottom: 2px solid #a855f7 !important; }

    /* Tags — subdued */
    .tags-text textarea {
        background: transparent !important; border: none !important;
        padding: 0 !important; font-size: 12px !important; color: #64748b !important;
        font-family: monospace !important; font-weight: 400 !important;
    }

    /* Radio */
    input[type=radio]:checked { accent-color: #a855f7 !important; }

    /* Slider — blue filled, light grey uncovered */
    :root, .gradio-container { --color-accent: #38bdf8 !important; }
    input[type=range], .gr-slider input, [class*="slider"] input {
        accent-color: #38bdf8 !important;
    }
    input[type=range]::-webkit-slider-thumb,
    input[type=range]::-moz-range-thumb { background: #38bdf8 !important; }
    input[type=range]::-webkit-slider-runnable-track { background: #e2e8f0 !important; }
    input[type=range]::-moz-range-track { background: #e2e8f0 !important; }
    .gr-slider [class*="range"] > div[style*="width"],
    .gr-slider .bar-container > div:first-child,
    [class*="slider"] [class*="fill"] { background: #38bdf8 !important; }
    .gr-slider .range-wrap, .gr-slider .bar-container { background: #e2e8f0 !important; }

    /* Slider number input — compact */
    .gr-slider input[type=number], .gr-slider input[type="number"],
    input[type=number] {
        max-width: 36px !important; font-size: 11px !important;
        padding: 1px 2px !important; opacity: 0.6 !important;
    }
    .thumb-value-tooltip {
        font-size: 12px !important; font-weight: 600 !important; color: #38bdf8 !important;
    }

    /* Audio */
    audio { width: 100% !important; border-radius: 8px !important; margin: 12px 0 !important; }

    /* Progress — light blue */
    .progress-bar, .progress-level { background: transparent !important; }
    .progress-bar > div { background: linear-gradient(90deg, #38bdf8, #0ea5e9) !important; height: 4px !important; border-radius: 4px !important; }
    .eta-bar { background: linear-gradient(90deg, #38bdf8, #0ea5e9) !important; height: 4px !important; }
    .progress-text { color: #888 !important; font-size: 12px !important; }
    .wrap.generating { background: transparent !important; }

    /* Fix percentage clipping */
    .wrap.generating, .generating, [class*="progress"], [class*="eta"] {
        overflow: visible !important;
        min-height: 32px !important;
        padding: 12px 16px !important;
        line-height: 1.5 !important;
    }
    .wrap.generating > div, .generating > div { overflow: visible !important; }
    .wrap.generating .progress-wrap, .wrap.generating [class*="progress"],
    [class*="generating"] [class*="percentage"] {
        display: flex !important; align-items: center !important; gap: 12px !important;
        flex-wrap: wrap !important; overflow: visible !important; padding: 8px 0 !important;
    }
    [class*="percentage"], .progress-text, span[id*="progress"] {
        overflow: visible !important; padding: 4px 8px !important; min-width: 56px !important;
    }

    footer, .built-with { display: none !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    span.prose, span { color: inherit !important; }
    p { color: #e5e5e5 !important; }
    """

    # Waveform: white for played, grey for unplayed
    waveform_opts = gr.WaveformOptions(
        waveform_color="#555555",
        waveform_progress_color="#ffffff",
    )

    # ── Build UI ──────────────────────────────────────────────────────
    with gr.Blocks(title="Erised", css=css) as demo:
        gr.HTML(
            "<h1 style='text-align:center;color:#a855f7;margin-bottom:8px;font-size:2.5rem;font-weight:700;'>🎵 Erised</h1>"
            "<p style='text-align:center;color:#888;margin-bottom:30px;'>"
            "Compare <b>original</b> vs <b>DPO-tuned</b> generations. "
            "Multiple tabs supported — requests queue automatically.</p>"
        )

        with gr.Tabs():
            with gr.Tab("A/B Compare"):
                with gr.Group(elem_classes=["form-card"]):
                    prompt_ab = gr.Textbox(label="Musical Prompt", placeholder="e.g., emotional pop ballad with piano and strings", lines=3)
                    lyrics_ab = gr.Textbox(label="Lyrics", placeholder="Your lyrics here...", lines=8)
                    max_len_ab = gr.Slider(10, 240, value=args.max_length, step=5, label="Max length (seconds)")

                gen_btn_ab = gr.Button("Generate Both", elem_id="generate-both-btn", elem_classes=["gen-btn"], size="lg")

                with gr.Group(elem_classes=["audio-card-section"]):
                    gr.HTML("<h3 style='color:#a855f7;margin:0 0 12px 0;'>Original Model</h3>")
                    audio_orig = gr.Audio(label="Original", type="filepath", show_label=False, waveform_options=waveform_opts)
                    tags_orig = gr.Textbox(show_label=False, interactive=False, elem_classes=["tags-text"])

                with gr.Group(elem_classes=["audio-card-section"]):
                    gr.HTML("<h3 style='color:#a855f7;margin:0 0 12px 0;'>DPO-Tuned Model</h3>")
                    audio_dpo = gr.Audio(label="DPO", type="filepath", show_label=False, waveform_options=waveform_opts)
                    tags_dpo = gr.Textbox(show_label=False, interactive=False, elem_classes=["tags-text"])

                gen_btn_ab.click(
                    fn=generate_original, inputs=[prompt_ab, lyrics_ab, max_len_ab],
                    outputs=[audio_orig, tags_orig],
                ).then(
                    fn=generate_dpo, inputs=[prompt_ab, lyrics_ab, max_len_ab],
                    outputs=[audio_dpo, tags_dpo],
                )

            with gr.Tab("Single Generate"):
                with gr.Group(elem_classes=["form-card"]):
                    model_choice = gr.Radio(["Original", "DPO-tuned"], value="DPO-tuned", label="Model")
                    prompt_single = gr.Textbox(label="Musical Prompt", placeholder="Describe the music...", lines=3)
                    lyrics_single = gr.Textbox(label="Lyrics", placeholder="Lyrics...", lines=8)
                    max_len_s = gr.Slider(10, 240, value=args.max_length, step=5, label="Max length (seconds)")

                gen_btn_s = gr.Button("Generate", elem_id="generate-single-btn", elem_classes=["gen-btn"], size="lg")

                with gr.Group(elem_classes=["audio-card-section"]):
                    audio_single = gr.Audio(label="Output", type="filepath", show_label=False, waveform_options=waveform_opts)
                    info_single = gr.Textbox(show_label=False, interactive=False, elem_classes=["tags-text"])

                gen_btn_s.click(
                    fn=generate_single, inputs=[model_choice, prompt_single, lyrics_single, max_len_s],
                    outputs=[audio_single, info_single],
                )

        gr.HTML(
            f"<div style='text-align:center;color:#666;font-size:12px;margin-top:24px;padding-top:16px;border-top:1px solid #222;'>"
            f"<b>Original:</b> <code style='color:#a855f7;background:#1a1a1a;padding:2px 6px;border-radius:4px;'>{original_model_path}</code> &nbsp; "
            f"<b>DPO:</b> <code style='color:#a855f7;background:#1a1a1a;padding:2px 6px;border-radius:4px;'>{args.dpo_path}</code></div>"
        )

    # ── JS: slider tooltip on thumb ───────────────────────────────────
    slider_thumb_js = r"""
    function initSliderThumbValue() {
        const setup = () => {
            document.querySelectorAll('.gr-slider, [class*="slider"]').forEach(block => {
                const range = block.querySelector('input[type="range"]');
                const numInput = block.querySelector('input[type="number"]');
                if (!range || range._tooltipInit) return;
                range._tooltipInit = true;
                if (numInput) numInput.style.cssText = 'max-width:36px;font-size:11px;padding:1px 2px;opacity:0.6;';
                let tooltip = block.querySelector('.thumb-value-tooltip');
                if (!tooltip) {
                    tooltip = document.createElement('span');
                    tooltip.className = 'thumb-value-tooltip';
                    tooltip.style.cssText = 'position:absolute;font-size:12px;font-weight:600;color:#38bdf8;background:#1e293b;padding:2px 6px;border-radius:4px;pointer-events:none;z-index:10;white-space:nowrap;';
                    block.style.position = 'relative';
                    block.appendChild(tooltip);
                }
                const update = () => {
                    const val = parseInt(range.value) || 0;
                    tooltip.textContent = val + 's';
                    const pct = (val - parseFloat(range.min)) / (parseFloat(range.max) - parseFloat(range.min));
                    tooltip.style.left = (pct * 100) + '%';
                    tooltip.style.top = '-24px';
                    tooltip.style.transform = 'translateX(-50%)';
                };
                range.addEventListener('input', update);
                range.addEventListener('change', update);
                update();
            });
        };
        if (document.readyState === 'complete') setup();
        else window.addEventListener('load', setup);
        setTimeout(setup, 1500);
        setTimeout(setup, 5000);
    }
    initSliderThumbValue();
    """
    demo.load(js=slider_thumb_js)

    # ── Launch ────────────────────────────────────────────────────────
    gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    os.makedirs(gradio_tmp, exist_ok=True)

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_port=args.port, server_name="0.0.0.0", share=True,
        show_error=True, allowed_paths=[gradio_tmp, "/tmp", "/workspace"],
    )


if __name__ == "__main__":
    main()
