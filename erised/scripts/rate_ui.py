#!/usr/bin/env python3
"""
Gradio-based preference rating UI for DPO training data collection.

Replaces the old ngrok + FastAPI setup with a single script that uses
Gradio's free share tunnel (*.gradio.live).

Usage (on RunPod):
    ERISED_MODEL_PATH=/workspace/heartlib/ckpt python erised/scripts/rate_ui.py
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("erised.rate_ui")


def copy_to_gradio_tmp(src_path):
    """Copy audio file to a Gradio-serveable temp path."""
    gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    os.makedirs(gradio_tmp, exist_ok=True)
    dst = os.path.join(gradio_tmp, os.path.basename(src_path))
    shutil.copy2(src_path, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(description="Preference rating UI for DPO data collection")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-length", type=int, default=60,
                        help="Default max length in seconds")
    args = parser.parse_args()

    import gradio as gr
    import torch

    from erised.config import ErisedConfig
    from erised.pipeline import ErisedPipeline
    from erised.dpo.data import PreferenceStore

    config = ErisedConfig.from_env()
    config.lazy_load = False

    logger.info("Loading pipeline from %s ...", config.model_path)
    pipeline = ErisedPipeline(config)
    logger.info("Pipeline loaded.")

    pref_store = PreferenceStore(config.dpo_db_path)
    logger.info("Preference DB: %s (%d pairs)", config.dpo_db_path, pref_store.count())

    def generate_a(prompt, lyrics, max_sec, progress=gr.Progress(track_tqdm=True)):
        if not prompt.strip():
            raise gr.Error("Please fill in both prompt and lyrics.")
        if not lyrics.strip():
            raise gr.Error("Please fill in both prompt and lyrics.")

        t0 = time.time()
        result_a = pipeline.generate(
            prompt=prompt, lyrics=lyrics,
            max_audio_length_ms=int(max_sec * 1000),
            temperature=0.7, cfg_scale=2.0,
        )
        elapsed = time.time() - t0
        logger.info("[A] Generated %s in %.1fs (%d frames)",
                    result_a.audio_path, elapsed, result_a.num_frames)

        tmp = copy_to_gradio_tmp(result_a.audio_path)
        pair_state = {
            "a_id": result_a.generation_id,
            "a_tokens_path": result_a.tokens_path,
            "prompt": prompt, "lyrics": lyrics,
            "tags": result_a.tags_used,
        }
        return tmp, f"Tags: {result_a.tags_used}", pair_state

    def generate_b(prompt, lyrics, max_sec, pair_state, progress=gr.Progress(track_tqdm=True)):
        if not prompt.strip():
            return None, "", pair_state

        t0 = time.time()
        result_b = pipeline.generate(
            prompt=prompt, lyrics=lyrics,
            max_audio_length_ms=int(max_sec * 1000),
            temperature=1.4, cfg_scale=1.0,
        )
        elapsed = time.time() - t0
        logger.info("[B] Generated %s in %.1fs (%d frames)",
                    result_b.audio_path, elapsed, result_b.num_frames)

        tmp = copy_to_gradio_tmp(result_b.audio_path)
        pair_state["b_id"] = result_b.generation_id
        pair_state["b_tokens_path"] = result_b.tokens_path
        pair_state["pair_id"] = f"{pair_state['a_id']}_{result_b.generation_id}"
        return tmp, f"Tags: {result_b.tags_used}", pair_state

    def rate_preference(choice, pair_state):
        if not pair_state or "pair_id" not in pair_state:
            raise gr.Error("Generate a pair first before rating.")

        if choice == "a":
            winner_id, loser_id = pair_state["a_id"], pair_state["b_id"]
            winner_tokens, loser_tokens = pair_state["a_tokens_path"], pair_state["b_tokens_path"]
        else:
            winner_id, loser_id = pair_state["b_id"], pair_state["a_id"]
            winner_tokens, loser_tokens = pair_state["b_tokens_path"], pair_state["a_tokens_path"]

        pref_store.add_preference(
            pair_id=pair_state["pair_id"], prompt=pair_state["prompt"],
            lyrics=pair_state["lyrics"], winner_id=winner_id, loser_id=loser_id,
            winner_tokens_path=winner_tokens, loser_tokens_path=loser_tokens,
        )
        count = pref_store.count()
        logger.info("Rated pair %s — winner: %s (total: %d)", pair_state["pair_id"], winner_id, count)
        return str(count), {}

    def undo_last():
        all_prefs = pref_store.get_all()
        if not all_prefs:
            return str(pref_store.count())
        pref_store.delete_preference(all_prefs[-1].pair_id)
        return str(pref_store.count())

    def stats_html(count):
        return (
            "<div style='text-align:center;'>"
            "<div style='font-weight:700;color:#a855f7;font-size:16px;'>Preferences collected:</div>"
            f"<div style='font-size:28px;font-weight:700;color:#a855f7;margin-top:6px;'>{count}</div>"
            "<div style='color:#666;font-size:12px;margin-top:8px;'>Aim for 50-100 before running DPO training</div>"
            "</div>"
        )

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

    /* Title — bigger like rate.html */
    h1, .gr-heading { font-size: 2.25rem !important; font-weight: 700 !important; }

    /* Form card */
    .form-card {
        background: #151515 !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin-bottom: 24px !important;
    }

    /* Audio cards — selectable boxes */
    .audio-card-section {
        background: #151515 !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin-bottom: 16px !important;
        border: 2px solid #333 !important;
        cursor: pointer !important;
        transition: border-color 0.2s, background-color 0.2s !important;
    }
    .audio-card-section:hover { border-color: #555 !important; }
    .audio-card-section.selected {
        border-color: #a855f7 !important;
        background: #1a1525 !important;
    }

    /* Vote buttons — side by side, same style as Generate Pair */
    .vote-row { margin-top: 16px !important; gap: 16px !important; }
    .vote-row button, .vote-row .primary,
    .vote-btn-a button, .vote-btn-a .primary,
    .vote-btn-b button, .vote-btn-b .primary {
        width: 100% !important; min-height: 56px !important; padding: 18px !important;
        font-size: 18px !important; font-weight: 600 !important;
        border-radius: 8px !important; cursor: pointer !important;
        display: block !important;
        background: #f8fafc !important; color: #7c3aed !important;
        border: 1px solid #e2e8f0 !important;
    }
    .vote-row button:hover, .vote-btn-a button:hover, .vote-btn-b button:hover {
        background: #f1f5f9 !important; border-color: #c4b5fd !important;
    }

    /* Stats — centered at bottom */
    .stats-card {
        text-align: center !important;
        padding: 20px !important;
        background: #151515 !important;
        border-radius: 12px !important;
        margin-top: 24px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        max-width: 400px !important;
    }

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

    /* Labels — purple, bold, cozy (matching Option A/B style) */
    label, label > span, .gr-block-label, .label-wrap > span, .gr-input-label {
        color: #7c3aed !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        text-transform: none !important;
        background: transparent !important;
        letter-spacing: 0.02em !important;
    }

    /* Generate Pair button — white block, purple text, BIG (matches Submit) */
    .gen-btn-wrap, .gen-btn-wrap .block, .gen-btn-wrap > div {
        margin-bottom: 24px !important;
    }
    #generate-pair-btn, #generate-pair-btn button, .gen-btn-wrap button, .gen-btn button,
    .gen-btn-wrap .primary, .gen-btn .primary {
        width: 100% !important; min-height: 56px !important; padding: 18px !important;
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #7c3aed !important;
        font-size: 18px !important; font-weight: 600 !important;
        border-radius: 8px !important; cursor: pointer !important;
        display: block !important;
    }
    #generate-pair-btn:hover, .gen-btn-wrap button:hover, .gen-btn button:hover {
        background: #f1f5f9 !important; border-color: #c4b5fd !important;
    }

    /* Submit preference button — same as Generate Pair: white block, purple text, BIG */
    .submit-btn-wrap, .submit-row, .submit-btn-wrap .block, .submit-btn-wrap > div {
        width: 100% !important; max-width: 100% !important;
    }
    #submit-preference-btn, #submit-preference-btn button, .submit-btn-wrap button, .submit-row button,
    .submit-btn button, .submit-btn .primary {
        width: 100% !important; min-width: 100% !important; min-height: 56px !important;
        padding: 18px 24px !important;
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #7c3aed !important;
        font-size: 18px !important; font-weight: 600 !important;
        border-radius: 8px !important; cursor: pointer !important;
        display: block !important;
    }
    #submit-preference-btn:hover, .submit-btn-wrap button:hover, .submit-row button:hover,
    .submit-btn button:hover { background: #f1f5f9 !important; border-color: #c4b5fd !important; }
    #submit-preference-btn:disabled, .submit-btn button:disabled, .submit-row button:disabled {
        background: #334155 !important; color: #94a3b8 !important; border-color: #475569 !important;
    }

    /* Toolbar — undo button */
    .toolbar-btn button {
        padding: 10px 20px !important; background: #222 !important;
        border: 1px solid #444 !important; color: #ccc !important;
        border-radius: 6px !important; font-size: 14px !important;
    }
    .toolbar-btn button:hover { background: #333 !important; }

    /* Tags — subdued, not full input styling */
    .tags-text textarea {
        background: transparent !important; border: none !important;
        padding: 0 !important; font-size: 12px !important; color: #64748b !important;
        font-family: monospace !important; font-weight: 400 !important;
    }

    /* Slider — blue filled, white/light grey uncovered */
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

    /* Slider number input — compact, faded (tooltip shows value on thumb) */
    .gr-slider input[type=number], .gr-slider input[type="number"],
    input[type=number] {
        max-width: 36px !important; font-size: 11px !important;
        padding: 1px 2px !important; opacity: 0.6 !important;
    }
    .thumb-value-tooltip {
        font-size: 12px !important; font-weight: 600 !important; color: #38bdf8 !important;
    }

    /* Waveform — Gradio uses WaveSurfer, colors set via waveform_options param.
       Native audio fallback styling: */
    audio, .audio-player audio {
        width: 100% !important; border-radius: 8px !important; margin: 12px 0 !important;
        accent-color: #ffffff !important;
    }
    audio::-webkit-media-controls-panel { background: #1a1a1a !important; }
    audio::-webkit-media-controls-current-time-display,
    audio::-webkit-media-controls-time-remaining-display { color: #e5e5e5 !important; }

    /* Progress — light blue */
    .progress-bar, .progress-level { background: transparent !important; }
    .progress-bar > div { background: linear-gradient(90deg, #38bdf8, #0ea5e9) !important; height: 4px !important; border-radius: 4px !important; }
    .eta-bar { background: linear-gradient(90deg, #38bdf8, #0ea5e9) !important; height: 4px !important; }
    .progress-text { color: #888 !important; font-size: 12px !important; }
    .wrap.generating { background: transparent !important; }

    /* Progress: hide the clipped percentage, hide ghost duplicate, show steps text cleanly */
    .wrap.generating, .generating {
        overflow: visible !important;
        min-height: 40px !important;
        padding: 4px 0 !important;
    }
    /* Hide the percentage text (gets clipped/ugly) */
    .wrap.generating .progress-text, .generating .progress-text,
    .wrap.generating .progress-level .progress-text,
    .generating .progress-level .progress-text {
        display: none !important;
    }
    /* Hide ghost duplicate progress row */
    .wrap.generating .progress-level ~ .progress-level,
    .generating .progress-level ~ .progress-level {
        display: none !important;
    }
    .wrap.generating .progress-level, .generating .progress-level {
        overflow: visible !important;
        white-space: nowrap !important;
    }
    .wrap.generating .progress-level span,
    .generating .progress-level span {
        overflow: visible !important;
        white-space: nowrap !important;
        font-size: 13px !important;
        color: #999 !important;
    }
    /* Hide the percentage span specifically (Gradio renders it as .eta-bar sibling) */
    .wrap.generating .progress-level > .progress-text,
    .generating .progress-level > .progress-text,
    .progress-level > span:first-child {
        display: none !important;
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
    with gr.Blocks(title="Erised - Rate Songs", css=css) as demo:
        gr.HTML(
            "<h1 style='text-align:center;color:#a855f7;margin-bottom:8px;font-size:2.5rem;font-weight:700;'>🎵 Erised RLHF Rating</h1>"
            "<p style='text-align:center;color:#888;margin-bottom:30px;'>Rate which song sounds better to train your model</p>"
        )

        pair_state = gr.State({})

        # ── Form ──
        with gr.Group(elem_classes=["form-card"]):
            prompt = gr.Textbox(
                label="Musical Prompt",
                placeholder="e.g., Male rappers, UK Drill Hip-Hop, 808 bassline, drill hi-hats, confident, luxurious mood",
                lines=3,
            )
            lyrics = gr.Textbox(
                label="Lyrics",
                placeholder="[Verse 1]\nYour lyrics here...\n\n[Chorus]\n...",
                lines=8,
            )
            max_len = gr.Slider(10, 240, value=args.max_length, step=5, label="Max length (seconds)")

        # Generate button
        with gr.Group(elem_classes=["gen-btn-wrap"]):
            gen_btn = gr.Button("Generate Pair", elem_id="generate-pair-btn", elem_classes=["gen-btn"], size="lg")

        # ── Option A card ──
        with gr.Group(elem_classes=["audio-card-section"]):
            gr.HTML("<h3 style='color:#a855f7;margin:0 0 12px 0;'>Option A <span style='font-size:12px;color:#888;font-weight:normal;'>(conservative)</span></h3>")
            audio_a = gr.Audio(label="A", type="filepath", show_label=False, waveform_options=waveform_opts, elem_classes=["audio-player"])
            tags_a = gr.Textbox(show_label=False, interactive=False, elem_classes=["tags-text"])

        # ── Option B card ──
        with gr.Group(elem_classes=["audio-card-section"]):
            gr.HTML("<h3 style='color:#a855f7;margin:0 0 12px 0;'>Option B <span style='font-size:12px;color:#888;font-weight:normal;'>(creative)</span></h3>")
            audio_b = gr.Audio(label="B", type="filepath", show_label=False, waveform_options=waveform_opts, elem_classes=["audio-player"])
            tags_b = gr.Textbox(show_label=False, interactive=False, elem_classes=["tags-text"])

        # ── Vote buttons (one click = select + submit) ──
        gr.HTML("<p style='text-align:center;color:#888;margin:8px 0 4px 0;font-size:14px;'>Which one sounds better?</p>")
        with gr.Row(elem_classes=["vote-row"]):
            vote_a_btn = gr.Button("Option A is better", elem_classes=["vote-btn-a"], size="lg")
            vote_b_btn = gr.Button("Option B is better", elem_classes=["vote-btn-b"], size="lg")

        # ── Stats (centered at bottom) ──
        with gr.Group(elem_classes=["stats-card"]):
            stats_count = gr.HTML(stats_html(pref_store.count()))

        # ── Toolbar ──
        undo_btn = gr.Button("Undo last rating", elem_classes=["toolbar-btn"])

        # ── Wiring ────────────────────────────────────────────────────
        gen_btn.click(
            fn=generate_a, inputs=[prompt, lyrics, max_len],
            outputs=[audio_a, tags_a, pair_state],
            concurrency_limit=1, concurrency_id="gpu",
        ).then(
            fn=generate_b, inputs=[prompt, lyrics, max_len, pair_state],
            outputs=[audio_b, tags_b, pair_state],
            concurrency_limit=1, concurrency_id="gpu",
        )

        def vote_for(choice, ps):
            if not ps or "pair_id" not in ps:
                raise gr.Error("Generate a pair first before voting.")
            c, _ = rate_preference(choice, ps)
            return (
                stats_html(c),
                None, "", None, "",  # audio_a, tags_a, audio_b, tags_b
                {}, "", "",  # pair_state, prompt, lyrics
            )

        def vote_a(ps):
            return vote_for("a", ps)

        def vote_b(ps):
            return vote_for("b", ps)

        def do_undo():
            return stats_html(undo_last())

        vote_a_btn.click(
            fn=vote_a, inputs=[pair_state],
            outputs=[stats_count, audio_a, tags_a, audio_b, tags_b, pair_state, prompt, lyrics],
        )
        vote_b_btn.click(
            fn=vote_b, inputs=[pair_state],
            outputs=[stats_count, audio_a, tags_a, audio_b, tags_b, pair_state, prompt, lyrics],
        )
        undo_btn.click(fn=do_undo, outputs=[stats_count])

    # ── JS: slider tooltip + card click → hidden button trigger ───────
    init_js = r"""
    function initUI() {
        const setup = () => {
            // ── Slider tooltip on thumb ──
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

            // ── Clear form fields on load (prevent browser autocomplete) ──
            document.querySelectorAll('.form-card textarea').forEach(el => {
                el.setAttribute('autocomplete', 'off');
            });
        };
        if (document.readyState === 'complete') setup();
        else window.addEventListener('load', setup);
        setTimeout(setup, 500);
        setTimeout(setup, 2000);
        setTimeout(setup, 5000);
    }
    initUI();
    """
    demo.load(js=init_js)

    # ── Launch ────────────────────────────────────────────────────────
    gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    os.makedirs(gradio_tmp, exist_ok=True)

    demo.queue(default_concurrency_limit=None)
    demo.launch(
        server_port=args.port, server_name="0.0.0.0", share=True,
        show_error=True, allowed_paths=[gradio_tmp, "/tmp", "/workspace"],
    )


if __name__ == "__main__":
    main()
