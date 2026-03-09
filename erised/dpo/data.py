"""
SQLite-backed preference store for RLHF/DPO training data.

Each record stores a (winner, loser) preference pair along with the prompt,
lyrics, and paths to saved token tensors.
"""

import sqlite3
import time
import json
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    pair_id: str
    prompt: str
    lyrics: str
    winner_id: str
    loser_id: str
    winner_tokens_path: str
    loser_tokens_path: str
    rater_id: str
    timestamp: float


class PreferenceStore:
    def __init__(self, db_path: str = "./dpo_preferences.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        logger.info("PreferenceStore initialized at %s", db_path)

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                lyrics TEXT NOT NULL,
                winner_id TEXT NOT NULL,
                loser_id TEXT NOT NULL,
                winner_tokens_path TEXT NOT NULL,
                loser_tokens_path TEXT NOT NULL,
                rater_id TEXT NOT NULL DEFAULT 'anonymous',
                timestamp REAL NOT NULL
            )
        """)
        self._conn.commit()

    def add_preference(
        self,
        pair_id: str,
        prompt: str,
        lyrics: str,
        winner_id: str,
        loser_id: str,
        winner_tokens_path: str,
        loser_tokens_path: str,
        rater_id: str = "anonymous",
    ):
        self._conn.execute(
            """INSERT INTO preferences
               (pair_id, prompt, lyrics, winner_id, loser_id,
                winner_tokens_path, loser_tokens_path, rater_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pair_id, prompt, lyrics, winner_id, loser_id,
             winner_tokens_path, loser_tokens_path, rater_id, time.time()),
        )
        self._conn.commit()
        logger.info("Preference recorded: pair=%s winner=%s loser=%s rater=%s",
                     pair_id, winner_id, loser_id, rater_id)

    def get_all(self) -> List[PreferencePair]:
        rows = self._conn.execute(
            "SELECT pair_id, prompt, lyrics, winner_id, loser_id, "
            "winner_tokens_path, loser_tokens_path, rater_id, timestamp "
            "FROM preferences ORDER BY timestamp"
        ).fetchall()
        return [PreferencePair(*r) for r in rows]

    def delete_preference(self, pair_id: str):
        self._conn.execute("DELETE FROM preferences WHERE pair_id = ?", (pair_id,))
        self._conn.commit()
        logger.info("Deleted preference: pair=%s", pair_id)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]

    def close(self):
        self._conn.close()
