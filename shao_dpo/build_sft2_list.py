"""Build (and keep rebuilding) the song list for the second fine-tuning run, from everything in B2 erised-sft.

Every song folder with audio (songs/<id>/audio.*) is checked against Cursor's own eligibility rule
(sft_core.eligible: plain Suno generation `type == "gen"`, no extend/concat/edit/crop/upsample/cover/remix/stem
history or special task, complete, has lyrics and a style prompt, >= 30 s, >= 125 likes, >= 100 plays)
— with three differences: instrumental songs are kept (the first run had 537; Cursor's rule now skips them only for
collecting), persona songs are kept (normal songs sung with a saved voice), and songs marked non-public /
hidden / trashed are kept when we already have their audio (Sean, 2026-10-02: research training data; CDN audio
was already scraped). Then:
  - exact duplicate recordings (same audio fingerprint under two IDs) keep only the most-liked one
  - lyrics get a `lyrics` = cleaned copy (copyright / link / credit / markdown-only lines removed, markdown section
    headings turned into [Section]); the original stays in `lyrics_raw`
  - split: songs from the first run keep their split; new songs from the first run's test creators go to test,
    everyone else to train (the test creators stay exactly the same, so before/after checks stay comparable)
  - at most 2% of the list per creator (most-liked kept), same as Cursor's lists

Output (B2): selected/sft2/list.jsonl (token-converter schema), selected/sft2/summary.json,
             selected/sft2/left_out_ids.json (every left-out song ID, by reason), selected/sft2/rejected_examples.json
Runs every 2 hours once deployed, so Cursor's newly downloaded songs are picked up automatically.

Run:    MODAL_PROFILE=erised7 modal run --detach shao_dpo/build_sft2_list.py      (once now)
        MODAL_PROFILE=erised7 modal deploy shao_dpo/build_sft2_list.py            (every 2 hours)
Stop:   MODAL_PROFILE=erised7 modal app stop sft2-list
"""
import collections
import json
import re
import time

import modal

BUCKET, ENDPOINT = "erised-sft", "https://s3.us-west-004.backblazeb2.com"
OUT = "selected/sft2/"
MIN_LIKES, MIN_PLAYS, MIN_SECONDS, CREATOR_SHARE = 125, 100, 30, 0.02
SPECIAL = ("cover_clip_id", "artist_clip_id", "continue_at", "infill", "upsample_clip_id",
           "edited_clip_id", "overpainting_clip_id", "underpainting_clip_id", "mashup_clip_ids", "stem_from_id",
           "history", "concat_history", "speed_clip_id", "is_remix")
# = Cursor's sft_core.SPECIAL minus persona_id: persona songs (a normal song sung with a voice the creator saved)
# are kept (Sean, 2026-10-02: "persona songs are fine so long as the music is good")

app = modal.App("sft2-list")
image = modal.Image.debian_slim(python_version="3.11").pip_install("boto3")


def meaningful(v):
    return v not in (None, "", False, [], {}, "None", 0, "00000000-0000-0000-0000-000000000000")


def why_not(c, likes, plays):
    """Cursor's eligible(), returning the reason, minus instrumental + not_public exclusions."""
    m = c.get("metadata") or {}
    if not c.get("id") or not c.get("user_id"):
        return "no_creator_id"
    # Keep songs we already have audio for even if Suno marks them private/hidden/trashed.
    if c.get("status") != "complete":
        return "not_complete"
    if m.get("type") != "gen":
        return f"type_{m.get('type') or 'unknown'}"
    if meaningful(m.get("task")):
        return f"task_{m.get('task')}"
    for k in SPECIAL:
        if meaningful(m.get(k)):
            return f"derived_{k}"
    instrumental = bool(m.get("make_instrumental")) or m.get("has_vocal") is False
    if not instrumental and len(re.sub(r"[\W_]+", "", m.get("prompt") or "")) < 3:
        return "no_lyrics"
    if not (m.get("tags") or m.get("gpt_description_prompt") or m.get("description_prompt")):
        return "no_style_prompt"
    if float(m.get("duration") or 0) < MIN_SECONDS:
        return "too_short"
    if likes < MIN_LIKES:
        return "under_125_likes"
    if plays < MIN_PLAYS:
        return "under_100_plays"
    return None


# ------------------------------------------------------------------ lyrics cleaning
SECTION = re.compile(r"^(intro|verse|pre[- ]?chorus|chorus|post[- ]?chorus|bridge|hook|refrain|outro|interlude|"
                     r"breakdown|drop|build[- ]?up|instrumental|solo|rap|spoken(?: word)?|final chorus)\b[\w\s\d'-]*$", re.I)
JUNK = [
    ("copyright", re.compile(r"©|\(c\)\s*\d{4}|all rights reserved|copyright", re.I)),
    ("link", re.compile(r"https?://|www\.|discord\.gg|\b(instagram|tiktok|youtube|spotify|soundcloud|facebook|twitter)\b|"
                        r"\.(com|net|org|io|gg)\b", re.I)),
    # "Lyrics by …", "Music & lyrics: …", or a short non-English credit line like "Testo Francesco Giaguaro"
    # ("by" must be followed by a Capitalized name, an @handle or "me", so "Words by the river" stays a lyric)
    ("credit", re.compile(r"^\s*((?i:lyrics|music|written|words|text|prod(?:uced|\.)?|composed|arranged|mixed)"
                          r"(?:\s*(?:&|and)\s*\w+)?\s*(?:(?i:by)\s+(?:@|[A-Z]|(?i:me|myself)\b)|:)"
                          r"|(?i:testo|musica|letra|paroles|songtext)\b(?=(?:\s+\S+){0,4}\s*$))")),
]


def clean_lyrics(text):
    out, removed = [], collections.Counter()
    for line in (text or "").split("\n"):
        s = line.strip()
        if s and re.fullmatch(r"[-=*_~#·.•\s]{3,}", s):                       # --- / *** / === separators
            removed["separator"] += 1
            continue
        hit = next((name for name, pat in JUNK if pat.search(s)), None)
        if hit:
            removed[hit] += 1
            continue
        bare = re.sub(r"^#+\s*|\*\*|__|:$", "", s).strip()
        if s.startswith("#") or (s.startswith("**") and s.endswith("**")):   # markdown heading
            removed["markdown_heading"] += 1
            out.append(f"[{bare}]" if SECTION.match(bare) else bare)
            continue
        out.append(line.rstrip())
    clean = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return clean, removed


# ------------------------------------------------------------------ the build
@app.function(image=image, cpu=4, memory=16384, secrets=[modal.Secret.from_name("b2-key")], timeout=3600,
              schedule=modal.Period(hours=2))
def build():
    import boto3
    from concurrent.futures import ThreadPoolExecutor
    s3 = boto3.client("s3", endpoint_url=ENDPOINT)
    t0 = time.time()

    def jsonl(key):
        try:
            return [json.loads(l) for l in s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().splitlines() if l.strip()]
        except s3.exceptions.NoSuchKey:
            return []

    # every song folder with audio
    audio = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="songs/"):
        for o in page.get("Contents", []):
            parts = o["Key"].split("/")
            if len(parts) == 3 and parts[2].startswith("audio."):
                audio[parts[1]] = {"key": o["Key"], "bytes": o["Size"], "etag": o["ETag"]}

    # likes/plays as recorded by every list (the highest seen wins), first-run splits and test creators
    seen_likes, seen_plays = collections.Counter(), collections.Counter()
    list_keys = ["selected/ready.jsonl", "extensions/research-2026-10-01/ready.jsonl", "extensions/hf-webshart-94k/ready.jsonl"]
    for p in s3.list_objects_v2(Bucket=BUCKET, Prefix="extensions/", Delimiter="/").get("CommonPrefixes", []):
        list_keys.append(p["Prefix"] + "qualifying.jsonl")
    for key in list_keys:
        for r in jsonl(key):
            cid = r.get("id") or r.get("clip_id")
            sm = r.get("selection_metrics") or {}
            seen_likes[cid] = max(seen_likes[cid], int(sm.get("likes") or r.get("upvote_count") or r.get("likes") or 0))
            seen_plays[cid] = max(seen_plays[cid], int(sm.get("plays") or r.get("play_count") or r.get("plays") or 0))
    v1 = {r["id"]: r for r in jsonl("shao_tokens/v1/manifest.jsonl")}
    test_creators = {r["user_id"] for r in v1.values() if r.get("split") == "test"}

    def meta(cid):
        try:
            return json.loads(s3.get_object(Bucket=BUCKET, Key=f"songs/{cid}/metadata.json")["Body"].read())
        except Exception:
            return None

    def receipt(cid):
        try:
            return json.loads(s3.get_object(Bucket=BUCKET, Key=f"songs/{cid}/receipt.json")["Body"].read())
        except Exception:
            return None

    # Sean 2026-10-02: restore the 863 songs Claude dropped as not_public — keep them even if
    # they also fail type/task filters (they were only labeled not_public because that check ran first).
    try:
        force_include = {
            ln.strip() for ln in s3.get_object(Bucket=BUCKET, Key=OUT + "force_include_ids.txt")["Body"]
            .read().decode().splitlines() if ln.strip()
        }
    except Exception:
        force_include = set()

    ids = sorted(set(audio) | force_include)
    with ThreadPoolExecutor(64) as pool:
        metas = dict(zip(ids, pool.map(meta, ids)))

    reasons, examples, keep = collections.Counter(), collections.defaultdict(list), []
    left_ids = collections.defaultdict(list)                       # reason -> every song ID left out for it
    forced = []
    for cid in ids:
        c = metas[cid]
        if not c:
            reasons["no_metadata_file"] += 1
            left_ids["no_metadata_file"].append(cid)
            continue
        if cid not in audio:
            reasons["force_include_no_audio"] += 1
            left_ids["force_include_no_audio"].append(cid)
            continue
        likes = max(int(c.get("upvote_count") or 0), seen_likes[cid])
        plays = max(int(c.get("play_count") or 0), seen_plays[cid])
        if cid in force_include:
            # force-include skips only the private check; every other rule still applies (Sean 2026-10-02:
            # drop the extended/edited/cover ones among them from the list — audio stays in the bucket)
            why = why_not(dict(c, is_public=True), likes, plays)
            if why:
                reasons[f"force_include_{why}"] += 1
                left_ids[f"force_include_{why}"].append(cid)
                continue
            forced.append((cid, c, likes, plays))
            continue
        why = why_not(c, likes, plays)
        if why:
            reasons[why] += 1
            left_ids[why].append(cid)
            if len(examples[why]) < 5:
                examples[why].append({"id": cid, "likes": likes, "title": c.get("title")})
            continue
        keep.append((cid, c, likes, plays))

    # exact duplicate recordings: keep the most-liked (force-includes exempt — always kept)
    by_etag = collections.defaultdict(list)
    for item in keep:
        by_etag[audio[item[0]]["etag"]].append(item)
    deduped = []
    for group in by_etag.values():
        group.sort(key=lambda x: -x[2])
        deduped.append(group[0])
        reasons["duplicate_recording"] += len(group) - 1
        left_ids["duplicate_recording"] += [x[0] for x in group[1:]]

    # 2% creator cap, most-liked first (force-includes bypass the cap)
    deduped.sort(key=lambda x: -x[2])
    cap = max(1, int(CREATOR_SHARE * (len(deduped) + len(forced))))
    per_creator, final = collections.Counter(), []
    for item in deduped:
        uid = item[1]["user_id"]
        if per_creator[uid] >= cap:
            reasons["over_creator_cap"] += 1
            left_ids["over_creator_cap"].append(item[0])
            continue
        per_creator[uid] += 1
        final.append(item)
    final.extend(forced)

    with ThreadPoolExecutor(64) as pool:
        receipts = dict(zip([x[0] for x in final], pool.map(receipt, [x[0] for x in final])))

    rows, cleaning = [], collections.Counter()
    for cid, c, likes, plays in final:
        m = c.get("metadata") or {}
        rec = receipts.get(cid) or {}
        raw = m.get("prompt") or ""
        instrumental = bool(m.get("make_instrumental")) or m.get("has_vocal") is False
        clean, removed = clean_lyrics(raw) if not instrumental else ("", collections.Counter())
        if not instrumental and len(re.sub(r"[\W_]+", "", clean)) < 3:      # the "lyrics" were only links/credits
            reasons["no_lyrics_after_cleaning"] += 1
            left_ids["no_lyrics_after_cleaning"].append(cid)
            continue
        cleaning.update(removed)
        cleaning["songs_changed"] += int(bool(removed))
        split = v1[cid]["split"] if cid in v1 else ("test" if c["user_id"] in test_creators else "train")
        rows.append({
            "id": cid, "split": split, "in_first_run": cid in v1, "title": c.get("title"),
            "user_id": c["user_id"], "handle": c.get("handle"),
            "style_prompt": m.get("tags") or "", "description_prompt": m.get("gpt_description_prompt"),
            "lyrics": clean, "lyrics_raw": raw, "instrumental": instrumental,
            "major_model_version": c.get("major_model_version"), "model_name": c.get("model_name"),
            "created_at": c.get("created_at"),
            "selection_metrics": {"likes": likes, "plays": plays, "like_rate": likes / plays if plays else None},
            "audio": {"key": audio[cid]["key"], "bytes": audio[cid]["bytes"],
                      "duration_seconds": rec.get("duration_seconds") or float(m.get("duration") or 0)},
            "source": rec.get("source") or "suno_collector",
        })

    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    s3.put_object(Bucket=BUCKET, Key=OUT + "list.jsonl", Body=body, ContentType="application/x-ndjson")
    s3.put_object(Bucket=BUCKET, Key=OUT + f"versions/{int(time.time())}.jsonl", Body=body)
    splits = collections.Counter(r["split"] for r in rows)
    summary = {
        "built_at": time.time(), "seconds_to_build": round(time.time() - t0),
        "songs_with_audio": len(audio), "kept": len(rows),
        "kept_hours": round(sum(r["audio"]["duration_seconds"] for r in rows) / 3600, 1),
        "train": splits["train"], "test": splits["test"],
        "from_first_run": sum(r["in_first_run"] for r in rows), "new": sum(not r["in_first_run"] for r in rows),
        "instrumental": sum(r["instrumental"] for r in rows),
        "force_include": len(forced), "force_include_requested": len(force_include),
        "creators": len(per_creator), "creator_cap": cap, "largest_creator": max(per_creator.values(), default=0),
        "left_out": dict(reasons.most_common()), "lyrics_cleaning": dict(cleaning.most_common()),
    }
    s3.put_object(Bucket=BUCKET, Key=OUT + "summary.json", Body=json.dumps(summary, indent=1).encode())
    s3.put_object(Bucket=BUCKET, Key=OUT + "left_out_ids.json", Body=json.dumps(left_ids).encode())
    s3.put_object(Bucket=BUCKET, Key=OUT + "rejected_examples.json", Body=json.dumps(examples, indent=1, ensure_ascii=False).encode())
    print(json.dumps(summary), flush=True)
    return summary


@app.local_entrypoint()
def main():
    call = build.spawn()
    print(f"build launched: {call.object_id} (safe to close the laptop); result: B2 {OUT}summary.json")
