

import sqlite3
from typing import Union

class DBManager:
    """Database manager for user and local_info tables"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS local_info(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER,
              thread_id TEXT,
              obj_info TEXT,
              emo_info TEXT,
              FOREIGN KEY(user_id) REFERENCES user(id) ON DELETE CASCADE
            )
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

    def delete_user(self, user_id: Union[str, int]):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM user WHERE id=?", (user_id,))
        conn.commit()
        conn.close()

    def list_threads(self, user_id: Union[str, int]):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id, thread_id FROM local_info WHERE user_id=?", (user_id,))
        rows = cur.fetchall()
        conn.close()
        return rows

    def update_local_info(self, user_id: Union[str, int], thread_id: str, obj_info: str, emo_info: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO local_info(user_id, thread_id, obj_info, emo_info)
            VALUES(?,?,?,?)
            """,
            (user_id, thread_id, obj_info, emo_info),
        )
        conn.commit()
        conn.close()

    def get_local_info(self, user_id: Union[str, int], thread_id: str) -> dict[str]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT obj_info, emo_info FROM local_info WHERE user_id=? AND thread_id=?", (user_id, thread_id,))
        rows = cur.fetchone()
        conn.close()
        return rows