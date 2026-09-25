from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from typing import Dict, Iterable, List


FIELDS = ("entity_id", "business_name", "business_address", "country")


def iter_tsv(path: str) -> Iterator[Dict[str, str]]:
    """Yield UTF-8 TSV rows without materialising the file."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Unexpected header in {path}: {reader.fieldnames}")
        for row in reader:
            yield {field: (row.get(field) or "") for field in FIELDS}


def write_tsv_header(handle, fields: Iterable[str]) -> None:
    csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(list(fields))


def write_id_list_row(handle, source1_id: str, ids: List[str]) -> None:
    csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(
        [source1_id, ",".join(ids)]
    )


def count_data_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def require_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path
