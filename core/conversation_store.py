"""SQLite-backed conversation & message persistence.

Schema:
  conversations: id, title, agent, created_at, updated_at
  messages: id, conv_id, role, content, agent, time
"""

import sqlite3
import json
import os
import time as _time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "conversations.db"
_lock = Lock()


def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            agent TEXT DEFAULT '',
            pinned INTEGER DEFAULT 0,
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT DEFAULT '',
            agent TEXT DEFAULT '',
            time TEXT DEFAULT '',
            FOREIGN KEY (conv_id) REFERENCES conversations(id)
        );
    """)
    # Migration: add pinned column if missing
    try:
        conn.execute("SELECT pinned FROM conversations LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER DEFAULT 0")
    conn.commit()
    conn.close()


def create_conversation(conv_id: str, title: str = "", agent: str = "", pinned: int = 0) -> Dict[str, Any]:
    _ensure_db()
    now = datetime.now().isoformat()
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO conversations (id, title, agent, pinned, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, title, agent, pinned, now, now),
        )
        conn.commit()
        conn.close()
    return {"id": conv_id, "title": title, "agent": agent, "pinned": pinned, "created_at": now, "updated_at": now}


def get_conversations(limit: int = 50) -> List[Dict[str, Any]]:
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, agent, pinned, created_at, updated_at FROM conversations ORDER BY pinned DESC, updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT id, title, agent, pinned, created_at, updated_at FROM conversations WHERE id=?", (conv_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_conversation(conv_id: str, title: Optional[str] = None, agent: Optional[str] = None, pinned: Optional[int] = None) -> None:
    _ensure_db()
    now = datetime.now().isoformat()
    updates = ["updated_at = ?"]
    params: List[Any] = [now]
    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if agent is not None:
        updates.append("agent = ?")
        params.append(agent)
    if pinned is not None:
        updates.append("pinned = ?")
        params.append(pinned)
    params.append(conv_id)
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
        conn.close()


def delete_conversation(conv_id: str) -> None:
    _ensure_db()
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        conn.close()


def save_message(conv_id: str, role: str, content: str, agent: str = "", msg_time: str = "") -> Dict[str, Any]:
    _ensure_db()
    if not msg_time:
        msg_time = datetime.now().isoformat()
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute(
            "INSERT INTO messages (conv_id, role, content, agent, time) VALUES (?, ?, ?, ?, ?)",
            (conv_id, role, content, agent, msg_time),
        )
        msg_id = cursor.lastrowid
        conn.commit()
        conn.close()
    return {"id": msg_id, "conv_id": conv_id, "role": role, "content": content, "agent": agent, "time": msg_time}


def get_messages(conv_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, conv_id, role, content, agent, time FROM messages WHERE conv_id=? ORDER BY id ASC LIMIT ?",
        (conv_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
