from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set

from .io_utils import iter_tsv
from .normalize import NormalizedRecord, normalize_record


SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    entity_id TEXT PRIMARY KEY,
    business_name TEXT NOT NULL,
    business_address TEXT NOT NULL,
    country TEXT NOT NULL,
    name_norm TEXT NOT NULL,
    name_compact TEXT NOT NULL,
    name_ascii TEXT NOT NULL,
    name_tokens TEXT NOT NULL,
    address_norm TEXT NOT NULL,
    address_tokens TEXT NOT NULL,
    address_numbers TEXT NOT NULL,
    postal_codes TEXT NOT NULL,
    address_key TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _pack(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _unpack(value: str) -> tuple[str, ...]:
    return tuple(json.loads(value)) if value else ()


def connect(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _remove_database(path: str) -> None:
    """Remove a generated SQLite database and its interrupted-run sidecars."""
    for candidate in (path, path + "-wal", path + "-shm"):
        if os.path.isfile(candidate):
            os.remove(candidate)


def _is_complete_index(db_path: str, source_path: str) -> bool:
    """Reject databases left behind before token indexes finished building."""
    try:
        connection = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        metadata = dict(connection.execute("SELECT key,value FROM meta").fetchall()) if "meta" in tables else {}
        connection.close()
        return (
            {"records", "meta", "token_postings"}.issubset(tables)
            and metadata.get("source_path") == os.path.abspath(source_path)
            and metadata.get("complete") == "1"
        )
    except (OSError, sqlite3.DatabaseError):
        return False


def build_index(source_path: str, db_path: str, rebuild: bool = False, progress_every: int = 50_000) -> str:
    if os.path.isfile(db_path) and not rebuild and _is_complete_index(db_path, source_path):
        return db_path
    if os.path.isfile(db_path) or os.path.isfile(db_path + "-wal") or os.path.isfile(db_path + "-shm"):
        _remove_database(db_path)
    connection = connect(db_path)
    connection.executescript(SCHEMA)
    token_counts: Counter[str] = Counter()
    insert_sql = """INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    batch = []
    count = 0
    started = time.monotonic()
    print(f"[index] {os.path.basename(source_path)}: pass 1/2, normalizing records", flush=True)
    for row in iter_tsv(source_path):
        record = normalize_record(row)
        batch.append((
            record.entity_id, record.business_name, record.business_address, record.country,
            record.name_norm, record.name_compact, record.name_ascii, _pack(record.name_tokens),
            record.address_norm, _pack(record.address_tokens), _pack(record.address_numbers),
            _pack(record.postal_codes), record.address_key,
        ))
        token_counts.update(set(record.name_tokens))
        count += 1
        if len(batch) >= 5000:
            connection.executemany(insert_sql, batch)
            connection.commit()
            batch.clear()
        if count % progress_every == 0:
            elapsed = time.monotonic() - started
            print(f"[index] {os.path.basename(source_path)}: {count:,} rows ({elapsed:.1f}s)", flush=True)
    if batch:
        connection.executemany(insert_sql, batch)
        connection.commit()
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_name_norm ON records(name_norm, country);
        CREATE INDEX IF NOT EXISTS idx_name_compact ON records(name_compact, country);
        CREATE INDEX IF NOT EXISTS idx_name_ascii ON records(name_ascii, country);
        CREATE INDEX IF NOT EXISTS idx_address_key ON records(address_key, country);
        CREATE INDEX IF NOT EXISTS idx_country ON records(country);
        CREATE TABLE IF NOT EXISTS token_postings(token TEXT NOT NULL, entity_id TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_token ON token_postings(token);
        CREATE INDEX IF NOT EXISTS idx_token_entity ON token_postings(token, entity_id);
        """
    )
    connection.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('source_path',?)", (os.path.abspath(source_path),))
    connection.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('row_count',?)", (str(count),))
    connection.commit()
    connection.close()

    # A second pass writes only informative tokens, keeping common words out of the index.
    connection = connect(db_path)
    posting_batch = []
    count = 0
    started = time.monotonic()
    print(f"[index] {os.path.basename(source_path)}: pass 2/2, building rare-token postings", flush=True)
    for row in iter_tsv(source_path):
        record = normalize_record(row)
        count += 1
        for token in set(record.name_tokens):
            if len(token) >= 3 and token_counts[token] <= 50:
                posting_batch.append((token, record.entity_id))
        if len(posting_batch) >= 10000:
            connection.executemany("INSERT INTO token_postings(token,entity_id) VALUES (?,?)", posting_batch)
            connection.commit()
            posting_batch.clear()
        if count % progress_every == 0:
            elapsed = time.monotonic() - started
            print(f"[index] {os.path.basename(source_path)}: {count:,} rows ({elapsed:.1f}s)", flush=True)
    if posting_batch:
        connection.executemany("INSERT INTO token_postings(token,entity_id) VALUES (?,?)", posting_batch)
        connection.commit()
    connection.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('complete', '1')")
    connection.commit()
    connection.close()
    print(f"[index] {os.path.basename(source_path)}: complete ({count:,} rows)", flush=True)
    return db_path


def record_from_row(row: sqlite3.Row) -> NormalizedRecord:
    return NormalizedRecord(
        entity_id=row["entity_id"], business_name=row["business_name"],
        business_address=row["business_address"], country=row["country"],
        name_norm=row["name_norm"], name_compact=row["name_compact"], name_ascii=row["name_ascii"],
        name_tokens=_unpack(row["name_tokens"]), address_norm=row["address_norm"],
        address_tokens=_unpack(row["address_tokens"]), address_numbers=_unpack(row["address_numbers"]),
        postal_codes=_unpack(row["postal_codes"]), address_key=row["address_key"],
    )


class RecordIndex:
    def __init__(self, db_path: str):
        self.connection = connect(db_path)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def _ids(self, sql: str, params: Sequence[str]) -> Set[str]:
        return {row[0] for row in self.connection.execute(sql, params)}

    def candidate_ids(self, query: NormalizedRecord, limit: int = 250) -> Set[str]:
        ids: Set[str] = set()
        lookups = [
            ("SELECT entity_id FROM records WHERE name_norm=? AND country=?", (query.name_norm, query.country)),
            ("SELECT entity_id FROM records WHERE name_norm=?", (query.name_norm,)),
            ("SELECT entity_id FROM records WHERE name_compact=? AND country=?", (query.name_compact, query.country)),
            ("SELECT entity_id FROM records WHERE name_ascii=? AND country=?", (query.name_ascii, query.country)),
        ]
        if query.address_key:
            lookups.append(("SELECT entity_id FROM records WHERE address_key=? AND country=?", (query.address_key, query.country)))
        for sql, params in lookups:
            if params[0]:
                ids.update(self._ids(sql, params))
            if len(ids) >= limit:
                return ids
        informative_tokens = sorted(set(query.name_tokens), key=lambda token: (len(token), token), reverse=True)[:6]
        if informative_tokens:
            placeholders = ",".join("?" for _ in informative_tokens)
            rows = self.connection.execute(
                f"SELECT entity_id FROM token_postings WHERE token IN ({placeholders}) LIMIT ?",
                [*informative_tokens, limit],
            )
            ids.update(row[0] for row in rows)
        return ids

    def records(self, entity_ids: Iterable[str]) -> Dict[str, NormalizedRecord]:
        values = list(entity_ids)
        if not values:
            return {}
        result: Dict[str, NormalizedRecord] = {}
        for offset in range(0, len(values), 400):
            chunk = values[offset:offset + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT * FROM records WHERE entity_id IN ({placeholders})", chunk
            )
            result.update({row["entity_id"]: record_from_row(row) for row in rows})
        return result
