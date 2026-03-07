#!/usr/bin/env python3
"""
FastAPI-based preference rating UI for DPO training data collection.

Replaces Gradio with a simple REST API + static HTML to avoid
WebSocket/SSE issues (blank pages, connection drops with multiple tabs).

Features:
  - Pre-generation queue: pairs generate in background while you rate
  - Multiple tabs safe: pure HTTP, no WebSocket state
  - 1.25x playback speed option
  - Works with ngrok or RunPod port forwarding

Usage (on RunPod):
    ERISED_DPO_DB=/workspace/heartlib/heartlib/dpo_preferences.db python3 -m erised.scripts.rate_local
    # Then in another terminal: ngrok http 7860
"""

import argparse
import collections
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("erised.rate_local")


# ── HTML UI (inline) ─────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erised - Rate Songs</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0a0a0a; color: #e5e5e5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 40px 60px;
}
@media (min-width: 1200px) { body { padding: 40px 10%; } }
@media (max-width: 768px) { body { padding: 20px; } }

h1 { text-align: center; color: #a855f7; font-size: 2.5rem; font-weight: 700; margin-bottom: 8px; }
.subtitle { text-align: center; color: #888; margin-bottom: 30px; }

.card {
    background: #151515; padding: 24px; border-radius: 12px; margin-bottom: 24px;
}
label {
    display: block; color: #7c3aed; font-weight: 600; font-size: 1.05rem;
    margin-bottom: 6px; letter-spacing: 0.02em;
}
textarea, input[type=text] {
    width: 100%; background: #3d3845; border: 1px solid #4a4555;
    color: #e5e5e5; padding: 14px; border-radius: 8px; font-size: 15px;
    font-weight: 500; font-family: inherit; line-height: 1.5; resize: vertical;
}
textarea:focus, input:focus { outline: none; border-color: #a855f7; }

.row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.row > * { flex: 1; min-width: 120px; }

input[type=range] { accent-color: #38bdf8; width: 100%; }
.slider-wrap { position: relative; }
.slider-val {
    font-size: 12px; font-weight: 600; color: #38bdf8;
    position: absolute; right: 0; top: -20px;
}

.btn {
    width: 100%; min-height: 56px; padding: 18px; font-size: 18px; font-weight: 600;
    border-radius: 8px; cursor: pointer; border: 1px solid #e2e8f0;
    background: #f8fafc; color: #7c3aed; display: block; text-align: center;
    transition: background 0.15s, border-color 0.15s;
}
.btn:hover { background: #f1f5f9; border-color: #c4b5fd; }
.btn:disabled { background: #334155; color: #94a3b8; border-color: #475569; cursor: not-allowed; }
.btn-sm {
    min-height: auto; padding: 10px 20px; font-size: 14px;
    background: #222; border: 1px solid #444; color: #ccc;
}
.btn-sm:hover { background: #333; }

.audio-card {
    background: #151515; padding: 24px; border-radius: 12px; margin-bottom: 16px;
    border: 2px solid #333; transition: border-color 0.2s, background-color 0.2s;
}
.audio-card.selected { border-color: #a855f7; background: #1a1525; }
.audio-card h3 { color: #a855f7; margin: 0 0 12px 0; }
.audio-card h3 span { font-size: 12px; color: #888; font-weight: normal; }
.audio-card audio { width: 100%; margin: 8px 0; border-radius: 8px; }
.tags { font-size: 12px; color: #64748b; font-family: monospace; margin-top: 4px; }

.speed-btns { display: flex; gap: 6px; margin: 8px 0; }
.speed-btn {
    padding: 4px 10px; font-size: 12px; border-radius: 4px; cursor: pointer;
    border: 1px solid #444; background: #222; color: #ccc;
    transition: all 0.15s;
}
.speed-btn:hover { background: #333; }
.speed-btn.active { background: #7c3aed; color: #fff; border-color: #7c3aed; }

.vote-row { display: flex; gap: 16px; margin-top: 16px; }
.vote-row .btn { flex: 1; }

.stats-card {
    text-align: center; padding: 20px; background: #151515;
    border-radius: 12px; margin-top: 24px; max-width: 400px;
    margin-left: auto; margin-right: auto;
}
.stats-num { font-size: 28px; font-weight: 700; color: #a855f7; margin-top: 6px; }
.stats-label { font-weight: 700; color: #a855f7; font-size: 16px; }
.stats-hint { color: #666; font-size: 12px; margin-top: 8px; }

.status-bar {
    display: flex; justify-content: center; gap: 24px; padding: 12px;
    background: #151515; border-radius: 8px; margin-bottom: 24px;
    font-size: 14px; color: #888;
}
.status-bar .val { color: #a855f7; font-weight: 600; }
.status-bar .generating { color: #38bdf8; }

#rating-section { display: none; }
#empty-msg { display: none; text-align: center; color: #666; padding: 40px; font-size: 16px; }

.toolbar { margin-top: 16px; text-align: center; }
</style>
</head>
<body>

<h1>🎵 Erised RLHF Rating</h1>
<p class="subtitle">Rate which song sounds better to train your model</p>

<!-- Status bar -->
<div class="status-bar">
    <div>Queue: <span class="val" id="st-queue">0</span></div>
    <div>Ready: <span class="val" id="st-ready">0</span></div>
    <div id="st-gen-wrap" style="display:none">
        <span class="generating">⏳ Generating...</span>
    </div>
    <div>Rated: <span class="val" id="st-rated">0</span></div>
</div>

<!-- Form -->
<div class="card">
    <div style="margin-bottom: 16px;">
        <label for="prompt">Musical Prompt</label>
        <textarea id="prompt" rows="3"
            placeholder="e.g., Male rappers, UK Drill Hip-Hop, 808 bassline, drill hi-hats, confident, luxurious mood"></textarea>
    </div>
    <div style="margin-bottom: 16px;">
        <label for="lyrics">Lyrics</label>
        <textarea id="lyrics" rows="8"
            placeholder="[Verse 1]&#10;Your lyrics here...&#10;&#10;[Chorus]&#10;Your chorus here..."></textarea>
    </div>
    <div class="row">
        <div class="slider-wrap">
            <label for="max-sec">Max length (seconds)</label>
            <span class="slider-val" id="slider-val">60s</span>
            <input type="range" id="max-sec" min="10" max="240" step="5" value="60">
        </div>
        <div>
            <label for="count">Pairs to queue</label>
            <select id="count" style="background:#3d3845;border:1px solid #4a4555;color:#e5e5e5;padding:14px;border-radius:8px;font-size:15px;width:100%;">
                <option value="1">1 pair</option>
                <option value="2">2 pairs</option>
                <option value="3" selected>3 pairs</option>
                <option value="5">5 pairs</option>
                <option value="10">10 pairs</option>
            </select>
        </div>
    </div>
    <div style="margin-top: 16px;">
        <button class="btn" id="queue-btn" onclick="queuePairs()">Queue Pairs for Generation</button>
    </div>
</div>

<!-- Rating section (shown when a pair is ready) -->
<div id="rating-section">
    <div class="audio-card" id="card-a">
        <h3>Option A <span>(balanced · temp 0.7)</span></h3>
        <audio id="audio-a" controls preload="auto"></audio>
        <div class="speed-btns" data-for="audio-a">
            <button class="speed-btn" data-speed="0.5">0.5x</button>
            <button class="speed-btn active" data-speed="1">1x</button>
            <button class="speed-btn" data-speed="1.25">1.25x</button>
            <button class="speed-btn" data-speed="1.5">1.5x</button>
            <button class="speed-btn" data-speed="2">2x</button>
        </div>
        <div class="tags" id="tags-a"></div>
    </div>

    <div class="audio-card" id="card-b">
        <h3>Option B <span>(creative · temp 1.2)</span></h3>
        <audio id="audio-b" controls preload="auto"></audio>
        <div class="speed-btns" data-for="audio-b">
            <button class="speed-btn" data-speed="0.5">0.5x</button>
            <button class="speed-btn active" data-speed="1">1x</button>
            <button class="speed-btn" data-speed="1.25">1.25x</button>
            <button class="speed-btn" data-speed="1.5">1.5x</button>
            <button class="speed-btn" data-speed="2">2x</button>
        </div>
        <div class="tags" id="tags-b"></div>
    </div>

    <p style="text-align:center;color:#888;font-size:14px;margin:8px 0 4px 0;">Which one sounds better?</p>
    <div class="vote-row">
        <button class="btn" onclick="vote('a')">Option A is better</button>
        <button class="btn" onclick="vote('b')">Option B is better</button>
    </div>

    <div style="text-align:center;margin-top:8px;">
        <span style="color:#666;font-size:12px;" id="pair-prompt-display"></span>
    </div>
</div>

<div id="empty-msg">No pairs ready yet. Queue some pairs above!</div>

<!-- Stats -->
<div class="stats-card">
    <div class="stats-label">Preferences collected:</div>
    <div class="stats-num" id="stats-count">0</div>
    <div class="stats-hint">Aim for 50-100 before running DPO training</div>
</div>

<div class="toolbar">
    <button class="btn btn-sm" onclick="undoLast()">Undo last rating</button>
</div>

<script>
let currentPair = null;

// ── Slider ──
const slider = document.getElementById('max-sec');
const sliderVal = document.getElementById('slider-val');
slider.addEventListener('input', () => { sliderVal.textContent = slider.value + 's'; });

// ── Speed buttons ──
document.querySelectorAll('.speed-btns').forEach(group => {
    group.querySelectorAll('.speed-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const audioId = group.dataset.for;
            const audio = document.getElementById(audioId);
            const speed = parseFloat(btn.dataset.speed);
            audio.playbackRate = speed;
            group.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
});

// ── API calls ──
async function queuePairs() {
    const prompt = document.getElementById('prompt').value.trim();
    const lyrics = document.getElementById('lyrics').value.trim();
    if (!prompt || !lyrics) { alert('Please fill in both prompt and lyrics.'); return; }

    const btn = document.getElementById('queue-btn');
    btn.disabled = true;
    btn.textContent = 'Queuing...';

    try {
        const resp = await fetch('/api/queue', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                prompt, lyrics,
                max_sec: parseInt(slider.value),
                count: parseInt(document.getElementById('count').value),
            }),
        });
        const data = await resp.json();
        if (!resp.ok) { alert(data.detail || 'Error queuing'); return; }
    } finally {
        btn.disabled = false;
        btn.textContent = 'Queue Pairs for Generation';
    }

    // Try to load next pair if we don't have one showing
    if (!currentPair) fetchNext();
}

async function fetchNext() {
    try {
        const resp = await fetch('/api/next');
        const data = await resp.json();
        if (data.status === 'ready') {
            showPair(data.pair);
        } else if (data.status === 'generating') {
            document.getElementById('rating-section').style.display = 'none';
            document.getElementById('empty-msg').style.display = 'block';
            document.getElementById('empty-msg').textContent = '⏳ Generating next pair... hang tight!';
        } else {
            document.getElementById('rating-section').style.display = 'none';
            document.getElementById('empty-msg').style.display = 'block';
            document.getElementById('empty-msg').textContent = 'No pairs ready yet. Queue some pairs above!';
        }
    } catch (e) {
        console.error('fetchNext error:', e);
    }
}

function showPair(pair) {
    currentPair = pair;
    document.getElementById('rating-section').style.display = 'block';
    document.getElementById('empty-msg').style.display = 'none';

    const audioA = document.getElementById('audio-a');
    const audioB = document.getElementById('audio-b');
    audioA.src = '/audio/' + pair.a_audio;
    audioB.src = '/audio/' + pair.b_audio;
    audioA.load();
    audioB.load();

    // Reset speed to 1x
    document.querySelectorAll('.speed-btns').forEach(group => {
        const audio = document.getElementById(group.dataset.for);
        audio.playbackRate = 1;
        group.querySelectorAll('.speed-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.speed === '1');
        });
    });

    document.getElementById('tags-a').textContent = 'Tags: ' + pair.tags_a;
    document.getElementById('tags-b').textContent = 'Tags: ' + pair.tags_b;
    document.getElementById('pair-prompt-display').textContent = 'Prompt: ' + pair.prompt;

    // Deselect cards
    document.getElementById('card-a').classList.remove('selected');
    document.getElementById('card-b').classList.remove('selected');
}

async function vote(choice) {
    if (!currentPair) { alert('No pair to rate!'); return; }

    // Visual feedback
    document.getElementById('card-a').classList.toggle('selected', choice === 'a');
    document.getElementById('card-b').classList.toggle('selected', choice === 'b');

    try {
        const resp = await fetch('/api/rate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ pair_id: currentPair.pair_id, choice }),
        });
        const data = await resp.json();
        if (!resp.ok) { alert(data.detail || 'Error rating'); return; }
        document.getElementById('stats-count').textContent = data.count;
    } catch (e) {
        alert('Error: ' + e.message);
        return;
    }

    currentPair = null;

    // Small delay for visual feedback, then load next
    setTimeout(fetchNext, 300);
}

async function undoLast() {
    try {
        const resp = await fetch('/api/undo', { method: 'POST' });
        const data = await resp.json();
        document.getElementById('stats-count').textContent = data.count;
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Status polling ──
async function pollStatus() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        document.getElementById('st-queue').textContent = data.pending;
        document.getElementById('st-ready').textContent = data.ready;
        document.getElementById('st-rated').textContent = data.rated;
        document.getElementById('st-gen-wrap').style.display = data.generating ? '' : 'none';
        document.getElementById('stats-count').textContent = data.rated;

        // Auto-fetch next pair if we don't have one and pairs are ready
        if (!currentPair && data.ready > 0) {
            fetchNext();
        }
    } catch (e) {}
}
setInterval(pollStatus, 2000);
pollStatus();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="FastAPI preference rating UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-length", type=int, default=60)
    args = parser.parse_args()

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel
    import uvicorn

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

    # ── Pre-generation queue ──────────────────────────────────────────
    gen_queue = queue.Queue()           # pending requests
    ready_pairs = collections.deque()   # completed pairs waiting to be rated
    gen_state = {"generating": False}   # shared state for status

    def generation_worker():
        """Background thread that processes the generation queue."""
        while True:
            try:
                job = gen_queue.get()
                gen_state["generating"] = True
                prompt, lyrics, max_sec = job["prompt"], job["lyrics"], job["max_sec"]

                logger.info("Generating pair for: %s", prompt[:60])
                t0 = time.time()

                result_a = pipeline.generate(
                    prompt=prompt, lyrics=lyrics,
                    max_audio_length_ms=int(max_sec * 1000),
                    temperature=0.7, cfg_scale=1.5,
                )
                logger.info("[A] Generated %s in %.1fs", result_a.audio_path, time.time() - t0)

                t1 = time.time()
                result_b = pipeline.generate(
                    prompt=prompt, lyrics=lyrics,
                    max_audio_length_ms=int(max_sec * 1000),
                    temperature=1.2, cfg_scale=0.8,
                )
                logger.info("[B] Generated %s in %.1fs", result_b.audio_path, time.time() - t1)

                pair_id = f"{result_a.generation_id}_{result_b.generation_id}"
                pair = {
                    "pair_id": pair_id,
                    "prompt": prompt,
                    "lyrics": lyrics,
                    "a_id": result_a.generation_id,
                    "b_id": result_b.generation_id,
                    "a_audio": os.path.basename(result_a.audio_path),
                    "b_audio": os.path.basename(result_b.audio_path),
                    "a_tokens_path": result_a.tokens_path,
                    "b_tokens_path": result_b.tokens_path,
                    "tags_a": result_a.tags_used,
                    "tags_b": result_b.tags_used,
                }
                ready_pairs.append(pair)
                logger.info("Pair ready: %s (total ready: %d)", pair_id, len(ready_pairs))

            except Exception:
                logger.exception("Error in generation worker")
            finally:
                gen_queue.task_done()
                gen_state["generating"] = gen_queue.qsize() > 0

    worker = threading.Thread(target=generation_worker, daemon=True)
    worker.start()

    # ── FastAPI app ───────────────────────────────────────────────────
    app = FastAPI(title="Erised Rating")
    _served_pairs = {}  # track pairs sent to frontend for rating lookup

    class QueueRequest(BaseModel):
        prompt: str
        lyrics: str
        max_sec: int = 60
        count: int = 3

    class RateRequest(BaseModel):
        pair_id: str
        choice: str  # "a" or "b"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_PAGE

    @app.get("/audio/{filename}")
    async def serve_audio(filename: str):
        path = os.path.join(config.output_dir, filename)
        if not os.path.isfile(path):
            raise HTTPException(404, "Audio file not found")
        media = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(path, media_type=media)

    @app.post("/api/queue")
    async def queue_pairs(req: QueueRequest):
        if not req.prompt.strip() or not req.lyrics.strip():
            raise HTTPException(400, "Prompt and lyrics required")
        for _ in range(req.count):
            gen_queue.put({
                "prompt": req.prompt,
                "lyrics": req.lyrics,
                "max_sec": req.max_sec,
            })
        logger.info("Queued %d pairs (queue size: %d)", req.count, gen_queue.qsize())
        return {"queued": req.count, "total_pending": gen_queue.qsize()}

    @app.get("/api/next")
    async def next_pair():
        if ready_pairs:
            pair = ready_pairs.popleft()
            _served_pairs[pair["pair_id"]] = pair
            return {"status": "ready", "pair": pair}
        elif gen_queue.qsize() > 0 or gen_state["generating"]:
            return {"status": "generating"}
        else:
            return {"status": "empty"}

    @app.post("/api/rate")
    async def rate(req: RateRequest):
        if req.choice not in ("a", "b"):
            raise HTTPException(400, "Choice must be 'a' or 'b'")

        # Find the pair data — we need to search ready_pairs or track served pairs
        # Since the pair was already popped from ready_pairs, we need to track it
        pair = _served_pairs.get(req.pair_id)
        if not pair:
            raise HTTPException(400, f"Unknown pair_id: {req.pair_id}")

        if req.choice == "a":
            winner_id, loser_id = pair["a_id"], pair["b_id"]
            winner_tokens, loser_tokens = pair["a_tokens_path"], pair["b_tokens_path"]
        else:
            winner_id, loser_id = pair["b_id"], pair["a_id"]
            winner_tokens, loser_tokens = pair["b_tokens_path"], pair["a_tokens_path"]

        pref_store.add_preference(
            pair_id=req.pair_id,
            prompt=pair["prompt"],
            lyrics=pair["lyrics"],
            winner_id=winner_id,
            loser_id=loser_id,
            winner_tokens_path=winner_tokens,
            loser_tokens_path=loser_tokens,
        )
        count = pref_store.count()
        logger.info("Rated pair %s — winner: %s (total: %d)", req.pair_id, winner_id, count)

        # Clean up served pair
        del _served_pairs[req.pair_id]
        return {"count": count}

    @app.post("/api/undo")
    async def undo():
        all_prefs = pref_store.get_all()
        if all_prefs:
            pref_store.delete_preference(all_prefs[-1].pair_id)
        return {"count": pref_store.count()}

    @app.get("/api/status")
    async def status():
        return {
            "pending": gen_queue.qsize(),
            "ready": len(ready_pairs),
            "generating": gen_state["generating"],
            "rated": pref_store.count(),
        }

    logger.info("Starting server on port %d", args.port)
    logger.info("Access at http://0.0.0.0:%d or via ngrok", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
