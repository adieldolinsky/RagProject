import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CHAT_DB_NAME, INSTANCE_FOLDER, MAX_HISTORY_MESSAGES


class ChatMemory:
    """SQLite-backed conversational memory keyed by Flask session id."""

    def __init__(self, db_path: str | None = None):
        instance = Path(INSTANCE_FOLDER)
        instance.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(instance / CHAT_DB_NAME)
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    chunks_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_session_created
                ON chat_messages (session_id, created_at)
                """
            )

    @staticmethod
    def new_session_id() -> str:
        return str(uuid.uuid4())

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        chunks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        chunks_json = json.dumps(chunks) if chunks is not None else None

        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, chunks_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content.strip(), chunks_json, created_at),
            )
            message_id = cursor.lastrowid

        self._trim_session(session_id)

        return {
            "id": message_id,
            "role": role,
            "content": content.strip(),
            "chunks": chunks,
            "created_at": created_at,
        }

    def _trim_session(self, session_id: str) -> None:
        with self._connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]

            if count <= MAX_HISTORY_MESSAGES:
                return

            excess = count - MAX_HISTORY_MESSAGES
            conn.execute(
                """
                DELETE FROM chat_messages
                WHERE id IN (
                    SELECT id FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                """,
                (session_id, excess),
            )

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, chunks_json, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()

        messages = []
        for row in rows:
            chunks = None
            if row["chunks_json"]:
                chunks = json.loads(row["chunks_json"])
            messages.append(
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "chunks": chunks,
                    "created_at": row["created_at"],
                }
            )
        return messages

    def clear_session(self, session_id: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ?",
                (session_id,),
            )
