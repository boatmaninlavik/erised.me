"""What did Suno actually sing? Transcribe the start of real Suno songs whose lyrics field contains creator
notes (copyright lines, links, prose) and compare with the lyrics field.

Run: MODAL_PROFILE=erised8 modal run shao_dpo/check_sung_lyrics.py
Output: B2 erised-sft/sft_eval/sft_v1/sung_lyrics_check.json
"""
import json

import modal

B2_BUCKET, B2_ENDPOINT = "erised-sft", "https://s3.us-west-004.backblazeb2.com"
SECONDS = 90
# check B test prompts whose lyrics field starts with notes (pid: song id), see eval_sft.py::pick_prompts
SONGS = {
    1: "credits lines + Italian lyrics",
    2: "markdown headings",
    14: "copyright / legal notice",
    23: "copyright lines + creator's prose note",
    26: "Discord link",
    27: "creator's note + separator line",
}

app = modal.App("shao-sung-lyrics")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", "openai-whisper", "boto3")
)


@app.function(image=image, gpu="L4", memory=16384, secrets=[modal.Secret.from_name("b2-key")], timeout=1800)
def transcribe():
    import subprocess, tempfile
    import boto3, whisper
    b2 = boto3.client("s3", endpoint_url=B2_ENDPOINT)
    prompts = {p["pid"]: p for p in json.loads(
        b2.get_object(Bucket=B2_BUCKET, Key="sft_eval/sft_v1/check_b_prompts.json")["Body"].read())}
    model = whisper.load_model("large-v3", device="cuda")
    out = []
    for pid, why in SONGS.items():
        sid = prompts[pid]["song_id"]
        meta = json.loads(b2.get_object(Bucket=B2_BUCKET, Key=f"songs/{sid}/metadata.json")["Body"].read())
        with tempfile.TemporaryDirectory() as d:
            open(f"{d}/a.m4a", "wb").write(b2.get_object(Bucket=B2_BUCKET, Key=f"songs/{sid}/audio.m4a")["Body"].read())
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/a.m4a", "-t", str(SECONDS), "-ac", "1", "-ar", "16000",
                            f"{d}/a.wav"], check=True)
            r = model.transcribe(f"{d}/a.wav", task="transcribe", condition_on_previous_text=False)
        row = {"pid": pid, "song_id": sid, "why_flagged": why, "language": r["language"],
               "lyrics_field_start": meta["metadata"]["prompt"][:500],
               "sung_first_90s": [{"start": round(s["start"], 1), "text": s["text"].strip()} for s in r["segments"]]}
        out.append(row)
        print(json.dumps({k: row[k] for k in ("pid", "why_flagged", "language")}), flush=True)
        for s in row["sung_first_90s"]:
            print(f"   {s['start']:5.1f}s  {s['text']}", flush=True)
    b2.put_object(Bucket=B2_BUCKET, Key="sft_eval/sft_v1/sung_lyrics_check.json",
                  Body=json.dumps(out, indent=1, ensure_ascii=False).encode())
    return len(out)


@app.local_entrypoint()
def main():
    print(transcribe.remote())
