
import sqlite3
import json
import numpy as np

from pathlib import Path
from datetime import datetime, timezone


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "database" / "objects.db"


def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.commit()


def save_object(object_id, name, image_path, embedding):

    with get_connection() as connection:

        connection.execute("""
            INSERT INTO objects (
                id,
                name,
                image_path,
                embedding,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            object_id,
            name,
            str(image_path),
            json.dumps(embedding.tolist()),
            datetime.now(timezone.utc).isoformat()
        ))

        connection.commit()


def get_all_objects():

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT id, name, image_path, embedding
            FROM objects
        """).fetchall()

    objects = []

    for row in rows:

        objects.append({
            "id": row["id"],
            "name": row["name"],
            "image_path": row["image_path"],
            "embedding": np.asarray(
                json.loads(row["embedding"]),
                dtype=np.float32
            )
        })

    return objects


c = '56'