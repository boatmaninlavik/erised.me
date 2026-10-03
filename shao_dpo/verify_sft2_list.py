"""Independent check of B2 selected/sft2/list.jsonl (written separately from build_sft2_list.py on purpose).

For every song in the list, re-read songs/<id>/metadata.json + receipt.json and the audio object, and check:
plain generation (type gen, no task, none of the derived-from-another-clip fields), >= 125 likes, >= 100 plays,
>= 30 s, lyrics present unless instrumental, a style prompt, status complete, audio present, IDs unique, no identical
recordings, creator share <= 2%. Also flags titles that look like extensions/remixes/covers and lists persona songs
and songs Suno marks as private (both allowed; reported for information).

Run: MODAL_PROFILE=erised7 modal run shao_dpo/verify_sft2_list.py      -> B2 selected/sft2/verify.json
"""
import json

import modal

BUCKET, ENDPOINT = "erised-sft", "https://s3.us-west-004.backblazeb2.com"
DERIVED = ("cover_clip_id", "artist_clip_id", "continue_at", "infill", "upsample_clip_id", "edited_clip_id",
           "overpainting_clip_id", "underpainting_clip_id", "mashup_clip_ids", "stem_from_id", "history",
           "concat_history", "speed_clip_id", "is_remix")

app = modal.App("sft2-verify")
image = modal.Image.debian_slim(python_version="3.11").pip_install("boto3")


def present(v):
    return v not in (None, "", False, [], {}, "None", 0, "00000000-0000-0000-0000-000000000000")


@app.function(image=image, cpu=4, memory=16384, secrets=[modal.Secret.from_name("b2-key")], timeout=3600)
def verify():
    import collections, re
    import boto3
    from concurrent.futures import ThreadPoolExecutor
    s3 = boto3.client("s3", endpoint_url=ENDPOINT)
    rows = [json.loads(l) for l in s3.get_object(Bucket=BUCKET, Key="selected/sft2/list.jsonl")["Body"].read().splitlines() if l.strip()]
    objects = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="songs/"):
        for o in page.get("Contents", []):
            objects[o["Key"]] = (o["Size"], o["ETag"])

    def load(cid):
        out = {}
        for name in ("metadata", "receipt"):
            try:
                out[name] = json.loads(s3.get_object(Bucket=BUCKET, Key=f"songs/{cid}/{name}.json")["Body"].read())
            except Exception:
                out[name] = None
        return out

    with ThreadPoolExecutor(64) as pool:
        files = dict(zip([r["id"] for r in rows], pool.map(load, [r["id"] for r in rows])))

    fail, examples = collections.Counter(), collections.defaultdict(list)
    info = collections.Counter()

    def bad(rule, r, detail=""):
        fail[rule] += 1
        if len(examples[rule]) < 5:
            examples[rule].append({"id": r["id"], "title": r.get("title"), "detail": detail})

    ids = collections.Counter(r["id"] for r in rows)
    for cid, n in ids.items():
        if n > 1:
            bad("id_repeated", {"id": cid}, f"{n} times")
    titles = re.compile(r"\b(extended|extension|remix|cover|remaster(ed)?|edit|sped up|slowed|nightcore|mashup|cut)\b", re.I)
    hashes = collections.defaultdict(list)
    creators = collections.Counter(r["user_id"] for r in rows)
    for r in rows:
        f = files[r["id"]]
        m, rec = f["metadata"], f["receipt"]
        if m is None:
            bad("no_metadata", r); continue
        md = m.get("metadata") or {}
        if md.get("type") != "gen":
            bad("not_plain_generation", r, f"type={md.get('type')}")
        if present(md.get("task")):
            bad("special_task", r, f"task={md.get('task')}")
        for k in DERIVED:
            if present(md.get(k)):
                bad(f"derived:{k}", r, str(md.get(k))[:60])
        likes = int((r.get("selection_metrics") or {}).get("likes") or 0)
        plays = int((r.get("selection_metrics") or {}).get("plays") or 0)
        if likes < 125:
            bad("under_125_likes", r, likes)
        if plays < 100:
            bad("under_100_plays", r, plays)
        if m.get("status") != "complete":
            bad("not_complete", r, m.get("status"))
        audio = objects.get(r["audio"]["key"])
        if not audio or audio[0] < 1000:
            bad("audio_missing", r, r["audio"]["key"])
        dur = float((rec or {}).get("duration_seconds") or r["audio"].get("duration_seconds") or 0)
        if dur < 30:
            bad("under_30_seconds", r, dur)
        if not r.get("instrumental") and len(re.sub(r"[\W_]+", "", r.get("lyrics") or "")) < 3:
            bad("no_lyrics", r)
        if not (r.get("style_prompt") or r.get("description_prompt")):
            bad("no_style_prompt", r)
        if titles.search(r.get("title") or ""):
            info["title_looks_like_extension_remix_cover_edit"] += 1
            if len(examples["title_flag"]) < 12:
                examples["title_flag"].append({"id": r["id"], "title": r.get("title"), "type": md.get("type")})
        if present(m.get("persona")) or present(md.get("persona_id")):
            info["persona_songs"] += 1
        if m.get("is_public") is False or m.get("is_hidden") or m.get("is_trashed"):
            info["private_hidden_or_trashed"] += 1
        if r.get("instrumental"):
            info["instrumental"] += 1
        roots = m.get("clip_roots")
        if present(roots) and any((x.get("id") if isinstance(x, dict) else x) not in (r["id"], None) for x in (roots if isinstance(roots, list) else [roots])):
            info["clip_roots_points_to_another_song"] += 1
            if len(examples["clip_roots"]) < 5:
                examples["clip_roots"].append({"id": r["id"], "title": r.get("title"), "clip_roots": str(roots)[:120]})
        h = (rec or {}).get("sha256") or (audio[1] if audio else None)
        if h:
            hashes[h].append(r["id"])
    for h, group in hashes.items():
        if len(group) > 1:
            bad("identical_recording", {"id": group[0]}, f"{len(group)} copies: {group[:4]}")
    cap = 0.02 * len(rows)
    for uid, n in creators.items():
        if n > cap:
            bad("creator_over_2_percent", {"id": uid}, f"{n} songs > {cap:.0f}")
    result = {"songs_checked": len(rows), "songs_failing_any_rule": None, "rule_failures": dict(fail.most_common()),
              "information_only": dict(info), "largest_creator_share": round(max(creators.values()) / len(rows), 4),
              "splits": dict(collections.Counter(r["split"] for r in rows)), "examples": examples}
    failing = set()
    for rule, lst in examples.items():
        pass
    # count distinct failing songs (re-run the per-song rules cheaply)
    result["songs_failing_any_rule"] = sum(1 for r in rows if files[r["id"]]["metadata"] is None
                                           or ((files[r["id"]]["metadata"].get("metadata") or {}).get("type") != "gen")
                                           or present((files[r["id"]]["metadata"].get("metadata") or {}).get("task"))
                                           or any(present((files[r["id"]]["metadata"].get("metadata") or {}).get(k)) for k in DERIVED))
    s3.put_object(Bucket=BUCKET, Key="selected/sft2/verify.json", Body=json.dumps(result, indent=1, ensure_ascii=False).encode())
    return result


@app.local_entrypoint()
def main():
    r = verify.remote()
    print(json.dumps({k: v for k, v in r.items() if k != "examples"}, indent=1))
    print("title-flag examples:", json.dumps(r["examples"].get("title_flag", [])[:12], ensure_ascii=False))
    print("clip_roots examples:", json.dumps(r["examples"].get("clip_roots", [])[:5], ensure_ascii=False))
