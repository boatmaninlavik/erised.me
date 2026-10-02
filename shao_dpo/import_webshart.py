"""Add the 125+-like songs from the public Hugging Face dataset webshart/suno-various-94k to B2 erised-sft.

Nothing here contacts Suno. Likes, plays, lyrics and style captions come from the dataset's own per-song
records (collected by its author in Aug 2026); audio is cut out of its tar shards by byte range.

  scan    read every song's record (85 shard indexes + 94,174 small range reads)
          -> extensions/hf-webshart-94k/metadata_all.jsonl.gz
  select  keep: likes >= 125, has lyrics (vocal), >= 30 s, not already in B2 in any form (songs/<id>/,
          Cursor's ready list, the research extension list, Cursor's upload queue), and at most 2% of the
          combined set per creator -> extensions/hf-webshart-94k/candidates.jsonl
  fetch   for each candidate: byte-range download from Hugging Face, strip embedded cover art, check with
          ffprobe, upload in Cursor's exact layout (songs/<id>/audio.mp3 with sha256 object metadata +
          receipt.json + metadata.json) -> extensions/hf-webshart-94k/ready.jsonl

Why Cursor won't add these again: its upload_song() first reads songs/<id>/receipt.json and, when the audio
object's size and sha256 match, skips the download. We write that receipt only after the audio is verified,
and skip any song that already has a receipt. Our list is separate from selected/ready.jsonl (Cursor's
uploader rewrites that file), so the two lists are merged by song ID at token-conversion time.

Run (workspace erised7):
    MODAL_PROFILE=erised7 modal run shao_dpo/import_webshart.py::scan
    MODAL_PROFILE=erised7 modal run shao_dpo/import_webshart.py::select
    MODAL_PROFILE=erised7 modal run --detach shao_dpo/import_webshart.py::fetch
"""
import gzip
import io
import json
import re
import time

import modal

REPO, REV = "webshart/suno-various-94k", "4ce029abf8fdcdd5fb75712da1afee5e6a5e8bc2"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/original"
SHARDS = [f"suno-various-94k-{i:05d}" for i in range(85)]
SNAPSHOT = "2026-08-31"                      # when the dataset's author collected likes/plays
MIN_LIKES, MIN_SECONDS, CREATOR_SHARE = 125, 30, 0.02
BUCKET, ENDPOINT = "erised-sft", "https://s3.us-west-004.backblazeb2.com"
OUT = "extensions/hf-webshart-94k/"
SOURCE = f"hf:{REPO}@{REV}"

app = modal.App("hf-webshart-import")
image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("boto3", "requests")
secrets = [modal.Secret.from_name("b2-key")]


def _s3():
    import boto3
    return boto3.client("s3", endpoint_url=ENDPOINT)


def _put_json(s3, key, obj, gz=False):
    body = (json.dumps(obj, ensure_ascii=False) if not isinstance(obj, (bytes, str)) else obj)
    body = body.encode() if isinstance(body, str) else body
    if gz:
        body = gzip.compress(body)
    s3.put_object(Bucket=BUCKET, Key=OUT + key, Body=body)


def _cdn(session, shard):
    """Resolve the shard tar to its signed CDN URL once, so the many range reads skip the Hub API."""
    r = session.head(f"{BASE}/{shard}.tar", allow_redirects=False, timeout=60)
    return r.headers.get("Location") or f"{BASE}/{shard}.tar"


def _range(session, url, offset, length, tries=5):
    for k in range(tries):
        try:
            r = session.get(url, headers={"Range": f"bytes={offset}-{offset + length - 1}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == length:
                return r.content
        except Exception:
            pass
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"range read failed at {offset}")


# ------------------------------------------------------------------ 1. every song's record
@app.function(image=image, cpu=4, memory=8192, secrets=secrets, timeout=3600)
def scan_shard(shard: str):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    s = requests.Session()
    idx = s.get(f"{BASE}/{shard}.json", timeout=120).json()
    url = _cdn(s, shard)
    entries = [v for k, v in idx["files"].items() if k.endswith(".mp3") and "json_offset" in v]

    def one(e):
        rec = json.loads(_range(s, url, e["json_offset"], e["json_length"]))
        rec.update(shard=shard, mp3_offset=e["offset"], mp3_length=e["length"])
        return rec

    with ThreadPoolExecutor(32) as pool:
        return list(pool.map(one, entries))


@app.function(image=image, cpu=1, memory=4096, secrets=secrets, timeout=3 * 3600)
def scan_all():
    rows = []
    for part in scan_shard.map(SHARDS, return_exceptions=True):
        if isinstance(part, Exception):
            print("shard failed:", part, flush=True)
            continue
        rows += part
    s3 = _s3()
    _put_json(s3, "metadata_all.jsonl.gz", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), gz=True)
    likes = [int(r.get("upvote_count") or 0) for r in rows]
    summary = {"songs": len(rows), "shards_ok": len({r["shard"] for r in rows}),
               **{f"likes>={t}": sum(x >= t for x in likes) for t in (50, 100, 125, 150, 250, 500, 1000)}}
    _put_json(s3, "scan_summary.json", summary)
    return summary


# ------------------------------------------------------------------ 2. what qualifies and is new
def _vocal(lyrics):
    words = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", lyrics or "")
    return len(re.findall(r"\w+", words)) >= 10 and "instrumental" not in (lyrics or "").lower()[:40]


@app.function(image=image, cpu=2, memory=16384, secrets=secrets, timeout=3600)
def select_new():
    s3 = _s3()
    rows = [json.loads(l) for l in gzip.decompress(s3.get_object(Bucket=BUCKET, Key=OUT + "metadata_all.jsonl.gz")["Body"].read()).splitlines() if l]

    # everything already in the bucket, in any form
    have = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="songs/", Delimiter="/"):
        have.update(p["Prefix"].split("/")[1] for p in page.get("CommonPrefixes", []))
    in_folders = len(have)
    known = {}                                                     # handle -> user_id, from Cursor's list
    creator_counts = {}
    for key in ("selected/ready.jsonl", "extensions/research-2026-10-01/ready.jsonl"):
        try:
            for line in s3.get_object(Bucket=BUCKET, Key=key)["Body"].iter_lines():
                if line:
                    r = json.loads(line)
                    have.add(r["id"])
                    if r.get("handle") and r.get("user_id"):
                        known[r["handle"].lower()] = r["user_id"]
                    creator_counts[r.get("user_id")] = creator_counts.get(r.get("user_id"), 0) + 1
        except s3.exceptions.NoSuchKey:
            pass
    uuid = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    queued = 0
    for prefix in ("queue/upload/", "queue/inflight/"):               # Cursor's in-progress work
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
            for o in page.get("Contents", []):
                body = s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read()
                try:
                    body = gzip.decompress(body)
                except OSError:
                    pass
                ids = {m.decode() for m in uuid.findall(body)}
                queued += len(ids - have)
                have |= ids

    def why_not(r):
        if r["id"] in have:
            return "already_have"
        if int(r.get("upvote_count") or 0) < MIN_LIKES:
            return "below_likes"
        if float(r.get("duration") or 0) < MIN_SECONDS:
            return "too_short"
        if not _vocal(r.get("lyrics")):
            return "no_lyrics"
        return None

    reasons, picked, seen = {}, [], set()
    for r in sorted(rows, key=lambda r: -int(r.get("upvote_count") or 0)):    # most-liked first
        why = why_not(r) or ("duplicate_in_dataset" if r["id"] in seen else None)
        seen.add(r["id"])
        reasons[why or "picked"] = reasons.get(why or "picked", 0) + 1
        if not why:
            r["user_id"] = known.get((r.get("creator") or "").lower()) or f"handle:{(r.get('creator') or '').lower()}"
            picked.append(r)

    # 2% creator cap on the combined set (Cursor's songs + these)
    total = sum(creator_counts.values()) + len(picked)
    cap, final, dropped = int(CREATOR_SHARE * total), [], 0
    for r in picked:
        if creator_counts.get(r["user_id"], 0) >= cap:
            dropped += 1
            continue
        creator_counts[r["user_id"]] = creator_counts.get(r["user_id"], 0) + 1
        final.append(r)
    _put_json(s3, "candidates.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final))
    summary = {"dataset_songs": len(rows), "already_in_bucket_folders": in_folders, "known_ids_total": len(have),
               "cursor_queue_ids_added": queued, "reasons": reasons, "creator_cap": cap,
               "dropped_by_creator_cap": dropped, "candidates": len(final),
               "candidate_hours": round(sum(float(r["duration"]) for r in final) / 3600, 1),
               "candidate_gb_est": round(sum(r["mp3_length"] for r in final) / 1e9, 1)}
    _put_json(s3, "select_summary.json", summary)
    return summary


# ------------------------------------------------------------------ 3. copy the audio in
@app.function(image=image, cpu=2, memory=4096, secrets=secrets, timeout=3 * 3600, max_containers=12)
def fetch_batch(batch: list):
    import hashlib, os, subprocess, tempfile
    import requests
    s3, s = _s3(), requests.Session()
    urls, out = {}, []
    for r in batch:
        cid = r["id"]
        try:
            try:                                             # someone (e.g. Cursor) got here first
                s3.head_object(Bucket=BUCKET, Key=f"songs/{cid}/receipt.json")
                out.append({"id": cid, "status": "already_have"})
                continue
            except Exception:
                pass
            if r["shard"] not in urls:
                urls[r["shard"]] = _cdn(s, r["shard"])
            try:
                raw = _range(s, urls[r["shard"]], r["mp3_offset"], r["mp3_length"])
            except RuntimeError:                             # signed URL expired: resolve again once
                urls[r["shard"]] = _cdn(s, r["shard"])
                raw = _range(s, urls[r["shard"]], r["mp3_offset"], r["mp3_length"])
            with tempfile.TemporaryDirectory() as d:
                open(f"{d}/in.mp3", "wb").write(raw)
                # audio stream only: drops embedded cover art (Cursor's rule) without re-encoding
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/in.mp3", "-map", "0:a:0", "-c:a", "copy",
                                "-map_metadata", "-1", f"{d}/a.mp3"], check=True, timeout=120)
                probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json",
                                                   f"{d}/a.mp3"], capture_output=True, text=True, check=True, timeout=60).stdout)
                streams = probe.get("streams", [])
                dur, expected = float(probe["format"].get("duration") or 0), float(r.get("duration") or 0)
                if not any(x.get("codec_type") == "audio" for x in streams) or any(x.get("codec_type") == "video" for x in streams):
                    out.append({"id": cid, "status": "bad_streams"}); continue
                if dur < MIN_SECONDS or (expected and abs(dur - expected) > max(5, 0.05 * expected)):
                    out.append({"id": cid, "status": "bad_duration", "seconds": dur}); continue
                data = open(f"{d}/a.mp3", "rb").read()
            digest, key = hashlib.sha256(data).hexdigest(), f"songs/{cid}/audio.mp3"
            s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType="audio/mpeg",
                          Metadata={"sha256": digest, "suno-id": cid, "source": "hf-webshart-94k"})
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] != len(data) or head.get("Metadata", {}).get("sha256") != digest:
                out.append({"id": cid, "status": "upload_verify_failed"}); continue
            receipt = {"key": key, "bytes": len(data), "sha256": digest, "duration_seconds": dur,
                       "format": probe["format"]["format_name"],
                       "streams": [{"codec": x.get("codec_name"), "sample_rate": x.get("sample_rate"), "channels": x.get("channels")} for x in streams],
                       "downloaded_at": time.time(), "source_url": f"{BASE}/{r['shard']}.tar#{cid}.mp3",
                       "playback_decoded": False, "source": SOURCE}
            meta = {"id": cid, "title": r.get("title"), "handle": r.get("creator"), "metadata": {
                        "tags": r.get("caption"), "prompt": r.get("lyrics"), "duration": r.get("duration")},
                    "play_count": r.get("play_count"), "upvote_count": r.get("upvote_count"),
                    "model_name": r.get("model_name"), "major_model_version": r.get("major_model_version"),
                    "created_at": r.get("created_at"), "_source": SOURCE, "_observed_at": SNAPSHOT,
                    "_search_term": r.get("search_term")}
            s3.put_object(Bucket=BUCKET, Key=f"songs/{cid}/metadata.json", Body=json.dumps(meta, ensure_ascii=False).encode())
            s3.put_object(Bucket=BUCKET, Key=f"songs/{cid}/receipt.json", Body=json.dumps(receipt).encode())
            likes, plays = int(r.get("upvote_count") or 0), int(r.get("play_count") or 0)
            out.append({"status": "ok", "row": {
                "id": cid, "title": r.get("title"), "user_id": r["user_id"], "handle": r.get("creator"),
                "style_prompt": r.get("caption"), "description_prompt": None, "negative_prompt": None,
                "lyrics": r.get("lyrics"), "instrumental": False, "model_name": r.get("model_name"),
                "major_model_version": r.get("major_model_version"), "created_at": r.get("created_at"),
                "audio": receipt, "metadata_key": f"songs/{cid}/metadata.json",
                "selection_metrics": {"likes": likes, "plays": plays, "like_rate": likes / plays if plays else None},
                "song_url": f"https://suno.com/song/{cid}", "source": SOURCE, "source_snapshot_at": SNAPSHOT,
                "generation_type": "unknown", "split": None, "sft_approved": False,
                "selection_status": "hf_dataset_likes_>=125", "musical_quality_review": "pending"}})
        except Exception as e:
            out.append({"id": cid, "status": "error", "error": str(e)[:200]})
    return out


@app.function(image=image, cpu=1, memory=4096, secrets=secrets, timeout=6 * 3600)
def fetch_all(batch_size: int = 50):
    s3 = _s3()
    cands = [json.loads(l) for l in s3.get_object(Bucket=BUCKET, Key=OUT + "candidates.jsonl")["Body"].iter_lines() if l]
    try:                                                             # resume: keep what is already listed
        ready = {json.loads(l)["id"]: json.loads(l) for l in s3.get_object(Bucket=BUCKET, Key=OUT + "ready.jsonl")["Body"].iter_lines() if l}
    except s3.exceptions.NoSuchKey:
        ready = {}
    todo = [r for r in cands if r["id"] not in ready]
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    counts, t0 = {}, time.time()
    for i, res in enumerate(fetch_batch.map(batches, return_exceptions=True)):
        for x in ([] if isinstance(res, Exception) else res):
            counts[x["status"]] = counts.get(x["status"], 0) + 1
            if x["status"] == "ok":
                ready[x["row"]["id"]] = x["row"]
        if isinstance(res, Exception):
            counts["batch_failed"] = counts.get("batch_failed", 0) + 1
        if i % 10 == 9 or i == len(batches) - 1:                     # single writer for the list
            _put_json(s3, "ready.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ready.values()))
            status = {"phase": "running" if i < len(batches) - 1 else "done", "updated_at": time.time(),
                      "batches_done": i + 1, "batches": len(batches), "ready": len(ready), "counts": counts,
                      "hours": round(sum(r["audio"]["duration_seconds"] for r in ready.values()) / 3600, 1),
                      "elapsed_min": round((time.time() - t0) / 60, 1)}
            _put_json(s3, "status.json", status)
            print(json.dumps(status), flush=True)
    return counts


@app.local_entrypoint()
def scan():
    print(scan_all.remote())


@app.local_entrypoint()
def select():
    print(json.dumps(select_new.remote(), indent=1))


@app.local_entrypoint()
def fetch():
    call = fetch_all.spawn()
    print(f"fetch launched: {call.object_id} (safe to close the laptop); progress: B2 {OUT}status.json")
