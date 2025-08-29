import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def _init_schema(self):
        with self.connect() as conn:
            c = conn.cursor()
            # Basic person-info store discovered from web
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    title TEXT,
                    email TEXT,
                    phone TEXT,
                    source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Ephemeral plan memory (KV per thread)
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT,
                    key TEXT,
                    value TEXT,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Planner status flag per thread
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS planner_status (
                    thread_id TEXT PRIMARY KEY,
                    is_active INTEGER DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    # --- Persons ---
    def insert_person(self, name: Optional[str], title: Optional[str], email: Optional[str], phone: Optional[str], source: str):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO persons(name, title, email, phone, source) VALUES (?, ?, ?, ?, ?)",
                (name, title, email, phone, source),
            )

    # --- Memory ---
    def remember(self, thread_id: str, key: str, value: str):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO plan_memory(thread_id, key, value) VALUES (?, ?, ?)",
                (thread_id, key, value),
            )

    def recall(self, thread_id: str) -> list[tuple[str, str]]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT key, value FROM plan_memory WHERE thread_id=? ORDER BY ts ASC",
                (thread_id,),
            )
            return list(cur.fetchall())

    # --- Planner status ---
    def set_planner_active(self, thread_id: str, active: bool):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO planner_status(thread_id, is_active) VALUES(?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET is_active=excluded.is_active, updated_at=CURRENT_TIMESTAMP",
                (thread_id, int(active)),
            )

    def is_planner_active(self, thread_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT is_active FROM planner_status WHERE thread_id=?",
                (thread_id,),
            )
            row = cur.fetchone()
            return bool(row[0]) if row else False