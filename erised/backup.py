"""
Backup and export preference data for safekeeping.

Usage:
    python -m erised.backup status    # Shows what data exists
    python -m erised.backup export     # Creates a downloadable zip
    python -m erised.backup restore PATH   # Restore from backup zip (e.g. after new pod)
"""

import os
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path


def _get_paths():
    """Use config paths; fall back to common locations if DB not found."""
    try:
        from .config import ErisedConfig
        config = ErisedConfig.from_env()
        db_path = Path(config.dpo_db_path)
        output_dir = Path(config.output_dir)
    except Exception:
        db_path = Path("./dpo_preferences.db")
        output_dir = Path("./outputs")

    # Fallback: if configured path doesn't exist, try project-relative paths
    if not db_path.exists():
        for candidate in [Path("./dpo_preferences.db"), Path("/workspace/heartlib/heartlib/dpo_preferences.db")]:
            if candidate.exists():
                db_path = candidate
                output_dir = candidate.parent / "outputs"
                break

    backup_dir = Path("/workspace/erised_backups")
    return db_path, output_dir, backup_dir


def get_status():
    """Show current data status."""
    db_path, output_dir, backup_dir = _get_paths()
    print("\n=== Erised Data Status ===\n")
    
    # Check DB
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
        conn.close()
        print(f"✓ Preferences database: {db_path}")
        print(f"  → {count} preference pairs collected")
    else:
        print(f"✗ No preferences database found at {db_path}")
    
    # Check outputs
    if output_dir.exists():
        mp3s = list(output_dir.glob("*.mp3"))
        tokens = list(output_dir.glob("*_tokens.pt"))
        print(f"\n✓ Outputs directory: {output_dir}")
        print(f"  → {len(mp3s)} audio files")
        print(f"  → {len(tokens)} token files")
        
        total_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file())
        print(f"  → Total size: {total_size / 1024 / 1024:.1f} MB")
    else:
        print(f"\n✗ No outputs directory found at {output_dir}")
    
    # Check backups
    if backup_dir.exists():
        backups = sorted(backup_dir.glob("*.zip"))
        if backups:
            print(f"\n✓ Backups directory: {backup_dir}")
            for b in backups[-5:]:  # Show last 5
                size = b.stat().st_size / 1024 / 1024
                print(f"  → {b.name} ({size:.1f} MB)")
        else:
            print(f"\n• Backups directory exists but is empty")
    else:
        print(f"\n• No backups created yet (dir: {backup_dir})")
    
    print()


def export_backup(db_path=None, output_dir=None):
    """Create a complete backup zip file.
    When called from server, pass db_path and output_dir to use the same paths."""
    if db_path is not None and output_dir is not None:
        db_path = Path(db_path)
        output_dir = Path(output_dir)
        backup_dir = Path("/workspace/erised_backups")
    else:
        db_path, output_dir, backup_dir = _get_paths()

    if not db_path.exists():
        print("No preference data to backup yet.")
        print(f"  Looked at: {db_path}")
        print("  Also check: ./dpo_preferences.db (relative to cwd)")
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"erised_backup_{timestamp}"
    temp_dir = backup_dir / backup_name
    temp_dir.mkdir(exist_ok=True)

    # Copy database
    shutil.copy(db_path, temp_dir / "preferences.db")

    # Export preferences as JSON (human-readable backup)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT pair_id, prompt, lyrics, winner_id, loser_id, 
               winner_tokens_path, loser_tokens_path, rater_id, timestamp
        FROM preferences ORDER BY timestamp
    """).fetchall()
    conn.close()
    
    preferences_json = []
    for row in rows:
        preferences_json.append({
            "pair_id": row[0],
            "prompt": row[1],
            "lyrics": row[2],
            "winner_id": row[3],
            "loser_id": row[4],
            "winner_tokens_path": row[5],
            "loser_tokens_path": row[6],
            "rater_id": row[7],
            "timestamp": row[8],
        })
    
    with open(temp_dir / "preferences.json", "w") as f:
        json.dump(preferences_json, f, indent=2)
    
    # Copy token files (needed for DPO training)
    tokens_dir = temp_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)
    
    copied_tokens = set()
    for pref in preferences_json:
        for path_key in ["winner_tokens_path", "loser_tokens_path"]:
            token_path = Path(pref[path_key])
            if token_path.exists() and token_path.name not in copied_tokens:
                shutil.copy(token_path, tokens_dir / token_path.name)
                copied_tokens.add(token_path.name)
    
    # Optionally copy audio files (larger, but useful for re-listening)
    audio_dir = temp_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    for pref in preferences_json:
        for id_key in ["winner_id", "loser_id"]:
            audio_file = output_dir / f"{pref[id_key]}.mp3"
            if audio_file.exists():
                shutil.copy(audio_file, audio_dir / audio_file.name)

    # Create zip
    zip_path = backup_dir / backup_name
    shutil.make_archive(str(zip_path), 'zip', temp_dir)
    
    # Cleanup temp dir
    shutil.rmtree(temp_dir)
    
    final_zip = backup_dir / f"{backup_name}.zip"
    size_mb = final_zip.stat().st_size / 1024 / 1024
    
    print(f"\n✓ Backup created: {final_zip}")
    print(f"  → Size: {size_mb:.1f} MB")
    print(f"  → Contains: {len(preferences_json)} preferences, {len(copied_tokens)} token files")
    print(f"\nDownload this file from Jupyter file browser at:")
    print(f"  /workspace/erised_backups/{backup_name}.zip")
    
    return final_zip


def import_restore(
    zip_path: str | Path,
    db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> bool:
    """
    Restore from a backup zip. Use after spinning up a new pod or migrating GPU.

    - Extracts preferences.db, tokens, and audio
    - Copies to target locations (from config if not provided)
    - Rewrites token paths in DB to match new output_dir

    Returns True on success.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"Error: Backup file not found: {zip_path}")
        return False

    try:
        cfg_db, cfg_out, _ = _get_paths()
    except Exception:
        cfg_db = "./dpo_preferences.db"
        cfg_out = "./outputs"

    target_db = Path(db_path) if db_path is not None else Path(cfg_db)
    target_out = Path(output_dir) if output_dir is not None else Path(cfg_out)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        extract_to = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)

        # Expected structure: preferences.db, preferences.json, tokens/, audio/
        src_db = extract_to / "preferences.db"
        if not src_db.exists():
            print("Error: Backup missing preferences.db")
            return False

        tokens_src = extract_to / "tokens"
        audio_src = extract_to / "audio"

        target_out.mkdir(parents=True, exist_ok=True)
        target_db.parent.mkdir(parents=True, exist_ok=True)

        # Back up existing DB if present
        if target_db.exists():
            backup_name = target_db.with_suffix(f".db.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.move(str(target_db), str(backup_name))
            print(f"Backed up existing DB to {backup_name}")

        shutil.copy(src_db, target_db)

        # Copy token files
        copied = 0
        if tokens_src.exists():
            for f in tokens_src.glob("*_tokens.pt"):
                shutil.copy(f, target_out / f.name)
                copied += 1
        print(f"Copied {copied} token files to {target_out}")

        # Copy audio files
        audio_count = 0
        if audio_src.exists():
            for f in audio_src.glob("*.mp3"):
                shutil.copy(f, target_out / f.name)
                audio_count += 1
        if audio_count:
            print(f"Copied {audio_count} audio files to {target_out}")

        # Rewrite token paths in DB to point to new output_dir
        conn = sqlite3.connect(target_db)
        rows = conn.execute(
            "SELECT id, winner_id, loser_id FROM preferences"
        ).fetchall()
        out_dir_str = str(target_out.resolve())
        for row_id, winner_id, loser_id in rows:
            conn.execute(
                "UPDATE preferences SET winner_tokens_path = ?, loser_tokens_path = ? WHERE id = ?",
                (
                    os.path.join(out_dir_str, f"{winner_id}_tokens.pt"),
                    os.path.join(out_dir_str, f"{loser_id}_tokens.pt"),
                    row_id,
                ),
            )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
        conn.close()

        print(f"\n✓ Restored {count} preferences to {target_db}")
        print(f"  Token/output dir: {target_out}")
        return True


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m erised.backup [status|export|restore PATH]")
        return

    cmd = sys.argv[1]
    if cmd == "status":
        get_status()
    elif cmd == "export":
        export_backup()
    elif cmd == "restore":
        if len(sys.argv) < 3:
            print("Usage: python -m erised.backup restore /path/to/erised_backup_YYYYMMDD_HHMMSS.zip [--db PATH] [--output PATH]")
            print("  Use after spinning up a new pod.")
            print("  Optional: --db and --output override ERISED_DPO_DB / ERISED_OUTPUT_DIR")
            return
        zip_arg = sys.argv[2]
        db_arg = output_arg = None
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--db" and i + 1 < len(sys.argv):
                db_arg = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_arg = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        import_restore(zip_arg, db_path=db_arg, output_dir=output_arg)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m erised.backup [status|export|restore PATH]")


if __name__ == "__main__":
    main()
