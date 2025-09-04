

import sqlite3
import datetime
from typing import Optional, Union

class DBManager:
    """Database manager for user and local_info tables"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE NOT NULL,
              created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS local_info(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              thread_id TEXT NOT NULL,
              obj_info TEXT DEFAULT '',
              emo_info TEXT DEFAULT '',
              created_at TEXT DEFAULT (datetime('now')),
              updated_at TEXT DEFAULT (datetime('now')),
              FOREIGN KEY(user_id) REFERENCES user(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_local_info_uid_thread
            ON local_info(user_id, thread_id);
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_local_info_updated
            AFTER UPDATE ON local_info
            FOR EACH ROW
            BEGIN
                UPDATE local_info
                SET updated_at = datetime('now')
                WHERE id = NEW.id;
            END;
            """
        )
        conn.commit()
        conn.close()

    def list_users(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id, username FROM user ORDER BY created_at ASC")
        rows = cur.fetchall()
        conn.close()
        return rows

    def create_user(self, username: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("INSERT INTO user(username) VALUES(?)", (username,))
        conn.commit()
        conn.close()

    def delete_user(self, user_id: int):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM user WHERE id=?", (user_id,))
        conn.commit()
        conn.close()

    def list_threads(self, user_id: int):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id, thread_id FROM local_info WHERE user_id=?", (user_id,))
        rows = cur.fetchall()
        conn.close()
        return rows

    def update_local_info(self, user_id: int, thread_id: str,
                        obj_info: str, emo_info: str,
                        sep: str = "\n\n---\n", add_timestamp: bool = True):
        obj_info = (obj_info or "").strip()
        emo_info = (emo_info or "").strip()

        if add_timestamp:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if obj_info:
                obj_info = f"[{stamp}] {obj_info}"
            if emo_info:
                emo_info = f"[{stamp}] {emo_info}"

        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
            INSERT OR IGNORE INTO local_info (user_id, thread_id, obj_info, emo_info)
            VALUES (?, ?, '', '');
        """, (user_id, thread_id))

        cur.execute("""
            UPDATE local_info
            SET
            obj_info = CASE
                WHEN ? = '' THEN obj_info
                WHEN obj_info IS NULL OR obj_info = '' THEN ?
                ELSE obj_info || ? || ?
            END,
            emo_info = CASE
                WHEN ? = '' THEN emo_info
                WHEN emo_info IS NULL OR emo_info = '' THEN ?
                ELSE emo_info || ? || ?
            END
            WHERE user_id = ? AND thread_id = ?;
        """, (
            obj_info, obj_info, sep, obj_info,
            emo_info, emo_info, sep, emo_info,
            user_id, thread_id
        ))

        conn.commit()
        conn.close()

    def get_local_info(self, user_id: int, thread_id: str) -> Optional[dict]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT obj_info, emo_info FROM local_info WHERE user_id=? AND thread_id=? LIMIT 1", (user_id, thread_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {"obj_info": row[0] or "", "emo_info": row[1] or ""}