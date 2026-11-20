"""SQLite candidate repository."""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _db_path() -> str:
    from app.config import settings
    return settings.db_path


def get_connection() -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(_db_path())
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        logger.error("DB connection error: %s", exc)
        return None


def init_db() -> None:
    """Create the candidates table if it does not exist."""
    conn = get_connection()
    if not conn:
        return
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                extracted_skills TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        logger.info("SQLite database initialised.")
    except Exception as exc:
        logger.error("DB init error: %s", exc)
    finally:
        conn.close()


def add_candidate(filename: str, content: str, extracted_skills) -> Optional[int]:
    """Insert a new candidate, skipping duplicates by filename."""
    conn = get_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM candidates WHERE filename = ?", (filename,))
        if cur.fetchone():
            logger.info("Candidate '%s' already exists — skipped.", filename)
            return None
        cur.execute(
            "INSERT INTO candidates (filename, content, extracted_skills) VALUES (?, ?, ?)",
            (filename, content, json.dumps(list(extracted_skills))),
        )
        conn.commit()
        return cur.lastrowid
    except Exception as exc:
        logger.error("DB insert error: %s", exc)
        return None
    finally:
        conn.close()


def get_all_candidates() -> List[Dict[str, Any]]:
    """Return all candidates as a list of dicts."""
    conn = get_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, filename, content, extracted_skills FROM candidates ORDER BY upload_date DESC")
        return [dict(row) for row in cur.fetchall()]
    except Exception as exc:
        logger.error("DB fetch error: %s", exc)
        return []
    finally:
        conn.close()
