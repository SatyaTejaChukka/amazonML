from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Tuple


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
NUMBER_RE = re.compile(r"\d+[a-z]?(?:[-/]\d+[a-z]?)?", re.IGNORECASE)
POSTAL_RE = re.compile(r"\b(?:\d{5}(?:-\d{4})?|\d{6})\b")
DOMAIN_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([^\s/]+\.(?:com|in|net|org))\b", re.I)
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc",
    "ltd", "limited", "pvt", "private", "plc", "llp", "sarl", "sas", "gmbh",
}


def _ascii_fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).encode(
        "ascii", "ignore"
    ).decode("ascii")


def normalize_text(value: str, ascii_fold: bool = False) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    if ascii_fold:
        value = _ascii_fold(value)
    return " ".join(TOKEN_RE.findall(value))


def compact_text(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def strip_domain(value: str) -> str:
    value = DOMAIN_RE.sub(lambda match: match.group(1).rsplit(".", 1)[0], value or "")
    return value


def remove_legal_suffixes(tokens: Iterable[str]) -> Tuple[str, ...]:
    result = tuple(token for token in tokens if token not in LEGAL_SUFFIXES)
    return result or tuple(tokens)


def tokens(value: str, remove_suffixes: bool = False) -> Tuple[str, ...]:
    normalized = normalize_text(strip_domain(value))
    values = tuple(normalized.split())
    if remove_suffixes:
        values = remove_legal_suffixes(values)
    return values


def address_numbers(value: str) -> Tuple[str, ...]:
    return tuple(NUMBER_RE.findall(normalize_text(value)))


def postal_codes(value: str) -> Tuple[str, ...]:
    return tuple(POSTAL_RE.findall(unicodedata.normalize("NFKC", value or "")))


def address_key(value: str) -> str:
    normalized = normalize_text(value, ascii_fold=True)
    parts = normalized.split()
    if not parts:
        return ""
    postal = postal_codes(value)
    number = address_numbers(value)
    tail = parts[-1] if len(parts) > 1 else ""
    anchor = postal[0] if postal else (number[0] if number else "")
    return "|".join(part for part in (anchor, tail) if part)


@dataclass(frozen=True)
class NormalizedRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str
    name_norm: str
    name_compact: str
    name_ascii: str
    name_tokens: Tuple[str, ...]
    address_norm: str
    address_tokens: Tuple[str, ...]
    address_numbers: Tuple[str, ...]
    postal_codes: Tuple[str, ...]
    address_key: str


def normalize_record(row: dict) -> NormalizedRecord:
    raw_name = row.get("business_name", "") or ""
    raw_address = row.get("business_address", "") or ""
    return NormalizedRecord(
        entity_id=row.get("entity_id", "") or "",
        business_name=raw_name,
        business_address=raw_address,
        country=normalize_text(row.get("country", "") or "", ascii_fold=True),
        name_norm=normalize_text(strip_domain(raw_name)),
        name_compact=compact_text(strip_domain(raw_name)),
        name_ascii=normalize_text(strip_domain(raw_name), ascii_fold=True),
        name_tokens=tokens(raw_name, remove_suffixes=True),
        address_norm=normalize_text(raw_address),
        address_tokens=tokens(raw_address),
        address_numbers=address_numbers(raw_address),
        postal_codes=postal_codes(raw_address),
        address_key=address_key(raw_address),
    )
