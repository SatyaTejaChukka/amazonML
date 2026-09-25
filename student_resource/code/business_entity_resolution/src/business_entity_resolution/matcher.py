from __future__ import annotations

import hashlib
import csv
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Set, Tuple

from .features import is_high_confidence, pair_features, score_features
from .embeddings import EmbeddingProvider, build_provider
from .index import RecordIndex, build_index
from .io_utils import iter_tsv, write_id_list_row, write_tsv_header
from .metrics import macro_f05, summarize_predictions
from .normalize import normalize_record


@dataclass
class MatcherConfig:
    candidate_cap: int = 40
    raw_candidate_limit: int = 250
    threshold: float = 0.78
    validation_fraction: float = 0.20
    seed: int = 42
    progress_every: int = 5_000
    embedding_model: str = "none"
    embedding_device: str = "auto"
    embedding_batch_size: int = 16


def dataset_root(data_dir: str) -> str:
    nested = os.path.join(data_dir, "dataset")
    return nested if os.path.isdir(os.path.join(nested, "train")) else data_dir


def holdout(entity_id: str, fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(f"{seed}:{entity_id}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    return value < fraction


def read_truth(path: str, fraction: float | None = None, seed: int = 42) -> Dict[str, Set[str]]:
    truth: Dict[str, Set[str]] = {}
    import csv
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source1_id = row["source1_entity_id"]
            if fraction is not None and not holdout(source1_id, fraction, seed):
                continue
            raw = (row.get("matched_entity_ids") or "").strip()
            truth[source1_id] = set(raw.split(",")) if raw else set()
    return truth


def _rank_candidates(query, index: RecordIndex, config: MatcherConfig) -> List[Tuple[str, float, Tuple[float, ...]]]:
    candidates = index.candidate_ids(query, limit=config.raw_candidate_limit)
    records = index.records(candidates)
    ranked = []
    for entity_id, record in records.items():
        features = pair_features(query, record)
        ranked.append((entity_id, score_features(features), features))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:config.candidate_cap]


def generate_candidates(s1_path: str, s2_index: RecordIndex, s3_index: RecordIndex,
                        output_path: str, candidate_db_path: str, config: MatcherConfig,
                        only_holdout: bool = False) -> str:
    if os.path.isfile(candidate_db_path):
        os.remove(candidate_db_path)
    candidate_connection = sqlite3.connect(candidate_db_path)
    candidate_connection.execute(
        "CREATE TABLE candidates(s1_id TEXT NOT NULL, target_id TEXT NOT NULL, rank_score REAL NOT NULL, PRIMARY KEY(s1_id,target_id))"
    )
    started = time.monotonic()
    processed = 0
    scanned = 0
    print(f"[candidates] {os.path.basename(s1_path)}: generating candidates", flush=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        write_tsv_header(handle, ("source1_entity_id", "candidate_entity_ids"))
        for row in iter_tsv(s1_path):
            scanned += 1
            if only_holdout and not holdout(row["entity_id"], config.validation_fraction, config.seed):
                if scanned % config.progress_every == 0:
                    elapsed = time.monotonic() - started
                    print(f"[candidates] scanned {scanned:,} Source 1 rows; selected {processed:,} ({elapsed:.1f}s)", flush=True)
                continue
            processed += 1
            query = normalize_record(row)
            ranked = _rank_candidates(query, s2_index, config) + _rank_candidates(query, s3_index, config)
            ranked.sort(key=lambda item: (-item[1], item[0]))
            deduped: Dict[str, float] = {}
            for entity_id, score, _features in ranked:
                deduped[entity_id] = max(score, deduped.get(entity_id, 0.0))
            values = sorted(deduped.items(), key=lambda item: (-item[1], item[0]))
            candidate_connection.executemany(
                "INSERT INTO candidates(s1_id,target_id,rank_score) VALUES (?,?,?)",
                [(query.entity_id, target_id, score) for target_id, score in values],
            )
            write_id_list_row(handle, query.entity_id, [item[0] for item in values])
            if processed % config.progress_every == 0:
                elapsed = time.monotonic() - started
                print(f"[candidates] {processed:,} Source 1 rows ({elapsed:.1f}s)", flush=True)
        candidate_connection.commit()
    candidate_connection.execute("CREATE INDEX idx_candidates_s1 ON candidates(s1_id, rank_score DESC)")
    candidate_connection.execute("CREATE INDEX idx_candidates_target ON candidates(target_id)")
    candidate_connection.commit()
    candidate_connection.close()
    print(f"[candidates] complete: {processed:,} Source 1 rows", flush=True)
    return candidate_db_path


def score_and_assign(s1_path: str, candidate_db_path: str,
                     s2_index: RecordIndex, s3_index: RecordIndex, config: MatcherConfig,
                     accepted_db_path: str, embedding_provider: EmbeddingProvider | None = None) -> str:
    if os.path.isfile(accepted_db_path):
        os.remove(accepted_db_path)
    connection = sqlite3.connect(accepted_db_path)
    connection.execute("CREATE TABLE edges(s1_id TEXT, target_id TEXT, score REAL, PRIMARY KEY(s1_id,target_id))")
    connection.execute("CREATE INDEX idx_edges_target ON edges(target_id, score DESC)")
    candidate_connection = sqlite3.connect(candidate_db_path)
    candidate_connection.row_factory = sqlite3.Row
    started = time.monotonic()
    processed = 0
    scanned = 0
    print(f"[scoring] {os.path.basename(s1_path)}: scoring accepted edges", flush=True)
    for row in iter_tsv(s1_path):
        scanned += 1
        source1_id = row["entity_id"]
        candidate_rows = candidate_connection.execute(
            "SELECT target_id, rank_score FROM candidates WHERE s1_id=? ORDER BY rank_score DESC, target_id",
            (source1_id,),
        ).fetchall()
        if not candidate_rows:
            if scanned % config.progress_every == 0:
                elapsed = time.monotonic() - started
                print(f"[scoring] scanned {scanned:,} Source 1 rows; {processed:,} had candidates ({elapsed:.1f}s)", flush=True)
            continue
        processed += 1
        query = normalize_record(row)
        s2_records = s2_index.records(row["target_id"] for row in candidate_rows if row["target_id"].startswith("S2-"))
        s3_records = s3_index.records(row["target_id"] for row in candidate_rows if row["target_id"].startswith("S3-"))
        all_records = {**s2_records, **s3_records}
        semantic_scores = embedding_provider.score_records(query, all_records) if embedding_provider else {}
        accepted = []
        for candidate_row in candidate_rows:
            target_id = candidate_row["target_id"]
            record = (s2_records if target_id.startswith("S2-") else s3_records).get(target_id)
            if record is None:
                continue
            features = pair_features(query, record)
            score = score_features(features, semantic_scores.get(target_id))
            if is_high_confidence(features, score, config.threshold):
                accepted.append((target_id, score))
        if accepted:
            connection.executemany("INSERT OR REPLACE INTO edges VALUES (?,?,?)", [(source1_id, target, score) for target, score in accepted])
        if connection.total_changes % 5000 == 0:
            connection.commit()
        if scanned % config.progress_every == 0:
            elapsed = time.monotonic() - started
            print(f"[scoring] scanned {scanned:,} Source 1 rows; {processed:,} had candidates ({elapsed:.1f}s)", flush=True)
    connection.commit()
    # A target record belongs to the highest-scoring S1 owner; S1 remains one-to-many.
    connection.execute(
        """CREATE TABLE owned AS
           SELECT s1_id, target_id FROM (
             SELECT s1_id, target_id, score,
                    ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY score DESC, s1_id) AS rn
             FROM edges
           ) WHERE rn=1"""
    )
    connection.execute("CREATE INDEX idx_owned_s1 ON owned(s1_id)")
    connection.commit()
    candidate_connection.close()
    connection.close()
    print(f"[scoring] complete: {processed:,} Source 1 rows with candidates", flush=True)
    return accepted_db_path


def _predictions_at_threshold(candidate_db_path: str, truth: Mapping[str, Set[str]], threshold: float) -> Dict[str, Set[str]]:
    predictions: Dict[str, Set[str]] = {key: set() for key in truth}
    connection = sqlite3.connect(candidate_db_path)
    for s1_id, target_id, rank_score in connection.execute("SELECT s1_id,target_id,rank_score FROM candidates"):
        if s1_id in predictions and rank_score >= threshold:
            predictions[s1_id].add(target_id)
    connection.close()
    return predictions


def tune_threshold(truth: Mapping[str, Set[str]], candidate_db_path: str,
                   config: MatcherConfig) -> Tuple[float, Dict[str, float]]:
    # Candidate ranks already contain the deterministic score; this grid tunes the final precision gate.
    best_threshold = config.threshold
    best_score = -1.0
    report = {}
    for threshold in [0.70 + i * 0.02 for i in range(16)]:
        predictions = _predictions_at_threshold(candidate_db_path, truth, threshold)
        score = macro_f05(predictions, truth)
        report[f"{threshold:.2f}"] = score
        if score > best_score:
            best_score, best_threshold = score, threshold
    return best_threshold, {"best_threshold": best_threshold, "best_macro_f05": best_score, **report}


def write_matching(path: str, s1_path: str, predictions: Mapping[str, Set[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        write_tsv_header(handle, ("source1_entity_id", "matched_entity_ids"))
        for row in iter_tsv(s1_path):
            values = sorted(predictions.get(row["entity_id"], set()))
            write_id_list_row(handle, row["entity_id"], values)


def write_matching_from_db(path: str, s1_path: str, accepted_db_path: str) -> None:
    connection = sqlite3.connect(accepted_db_path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        write_tsv_header(handle, ("source1_entity_id", "matched_entity_ids"))
        for row in iter_tsv(s1_path):
            ids = [item[0] for item in connection.execute(
                "SELECT target_id FROM owned WHERE s1_id=? ORDER BY target_id", (row["entity_id"],)
            )]
            write_id_list_row(handle, row["entity_id"], ids)
    connection.close()


def run_inference(data_dir: str, work_dir: str, output_dir: str, config: MatcherConfig, rebuild: bool = False) -> Dict[str, str]:
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    test_dir = os.path.join(dataset_root(data_dir), "test")
    print("[infer] building or reusing test indexes", flush=True)
    s2_db = build_index(os.path.join(test_dir, "test_source2.tsv"), os.path.join(work_dir, "test_source2.sqlite"), rebuild, config.progress_every)
    s3_db = build_index(os.path.join(test_dir, "test_source3.tsv"), os.path.join(work_dir, "test_source3.sqlite"), rebuild, config.progress_every)
    s2_index, s3_index = RecordIndex(s2_db), RecordIndex(s3_db)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    candidate_db = generate_candidates(os.path.join(test_dir, "test_source1.tsv"), s2_index, s3_index, candidate_path, os.path.join(work_dir, "candidates_test.sqlite"), config)
    embedding_provider = build_provider(config.embedding_model, config.embedding_device, config.embedding_batch_size)
    accepted_db = score_and_assign(os.path.join(test_dir, "test_source1.tsv"), candidate_db, s2_index, s3_index, config, os.path.join(work_dir, "accepted_test.sqlite"), embedding_provider)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    write_matching_from_db(matching_path, os.path.join(test_dir, "test_source1.tsv"), accepted_db)
    s2_index.close(); s3_index.close()
    return {"matching": matching_path, "candidate": candidate_path}


def run_validation(data_dir: str, work_dir: str, config: MatcherConfig, rebuild: bool = False) -> Dict[str, object]:
    os.makedirs(work_dir, exist_ok=True)
    train_dir = os.path.join(dataset_root(data_dir), "train")
    print("[validate] building or reusing training indexes", flush=True)
    s2_db = build_index(os.path.join(train_dir, "train_source2.tsv"), os.path.join(work_dir, "train_source2.sqlite"), rebuild, config.progress_every)
    s3_db = build_index(os.path.join(train_dir, "train_source3.tsv"), os.path.join(work_dir, "train_source3.sqlite"), rebuild, config.progress_every)
    s2_index, s3_index = RecordIndex(s2_db), RecordIndex(s3_db)
    candidate_path = os.path.join(work_dir, "validation_candidates.tsv")
    candidate_db = generate_candidates(os.path.join(train_dir, "train_source1.tsv"), s2_index, s3_index, candidate_path, os.path.join(work_dir, "candidates_validation.sqlite"), config, only_holdout=True)
    truth = read_truth(os.path.join(train_dir, "train_ground_truth.tsv"), config.validation_fraction, config.seed)
    threshold, tuning = tune_threshold(truth, candidate_db, config)
    config.threshold = threshold
    embedding_provider = build_provider(config.embedding_model, config.embedding_device, config.embedding_batch_size)
    accepted_db = score_and_assign(os.path.join(train_dir, "train_source1.tsv"), candidate_db, s2_index, s3_index, config, os.path.join(work_dir, "accepted_validation.sqlite"), embedding_provider)
    connection = sqlite3.connect(accepted_db)
    predictions = {key: {row[0] for row in connection.execute("SELECT target_id FROM owned WHERE s1_id=?", (key,))} for key in truth}
    connection.close()
    report = summarize_predictions(predictions, truth)
    report.update({"threshold": threshold, "candidate_entities": sum(1 for _ in sqlite3.connect(candidate_db).execute("SELECT 1 FROM candidates")), "tuning": tuning})
    with open(os.path.join(work_dir, "validation_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    s2_index.close(); s3_index.close()
    return report
