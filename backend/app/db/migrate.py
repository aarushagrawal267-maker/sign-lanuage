"""Additive, idempotent schema updates for databases created by older versions (no migration framework).

Only adds missing columns (``PRAGMA table_info``); never drops or rewrites data.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

# table -> [(column, SQL type and default)]
COLUMNS: dict[str, list[tuple[str, str]]] = {
    "recognition_events": [
        ("language", "VARCHAR(32) DEFAULT 'asl-wlasl100'"),
        ("detail", "JSON"),
    ],
}


def ensure_columns(engine: Engine) -> list[str]:
    added = []
    with engine.begin() as conn:
        for table, cols in COLUMNS.items():
            have = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            if not have:
                continue  # table does not exist (create_all makes it with every column)
            for name, ddl in cols:
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    added.append(f"{table}.{name}")
    return added
