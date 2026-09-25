from __future__ import annotations

import difflib
from typing import Iterable, Tuple

try:
    from rapidfuzz.fuzz import ratio as _rapid_ratio
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    _rapid_ratio = None

from .normalize import NormalizedRecord


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if _rapid_ratio is not None:
        return float(_rapid_ratio(left, right)) / 100.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def pair_features(left: NormalizedRecord, right: NormalizedRecord) -> Tuple[float, ...]:
    name_exact = float(bool(left.name_norm and left.name_norm == right.name_norm))
    compact_exact = float(bool(left.name_compact and left.name_compact == right.name_compact))
    ascii_exact = float(bool(left.name_ascii and left.name_ascii == right.name_ascii))
    name_ratio = _ratio(left.name_ascii, right.name_ascii)
    name_token_jaccard = _jaccard(left.name_tokens, right.name_tokens)
    name_token_overlap = _overlap(left.name_tokens, right.name_tokens)
    address_ratio = _ratio(left.address_norm, right.address_norm)
    address_jaccard = _jaccard(left.address_tokens, right.address_tokens)
    address_overlap = _overlap(left.address_tokens, right.address_tokens)
    number_overlap = _overlap(left.address_numbers, right.address_numbers)
    postal_overlap = _overlap(left.postal_codes, right.postal_codes)
    address_key_equal = float(bool(left.address_key and left.address_key == right.address_key))
    country_equal = float(bool(left.country and left.country == right.country))
    left_address = float(bool(left.address_norm))
    right_address = float(bool(right.address_norm))
    address_both_present = left_address * right_address
    return (
        name_exact, compact_exact, ascii_exact, name_ratio, name_token_jaccard,
        name_token_overlap, address_ratio, address_jaccard, address_overlap,
        number_overlap, postal_overlap, address_key_equal, country_equal,
        left_address, right_address, address_both_present,
    )


def score_features(features: Tuple[float, ...], semantic_score: float | None = None) -> float:
    (
        name_exact, compact_exact, ascii_exact, name_ratio, name_token_jaccard,
        name_token_overlap, address_ratio, address_jaccard, address_overlap,
        number_overlap, postal_overlap, address_key_equal, country_equal,
        _left_address, _right_address, address_both_present,
    ) = features
    name_score = (
        0.28 * name_exact
        + 0.18 * compact_exact
        + 0.12 * ascii_exact
        + 0.20 * name_ratio
        + 0.12 * name_token_jaccard
        + 0.10 * name_token_overlap
    )
    address_score = (
        0.35 * address_ratio
        + 0.25 * address_jaccard
        + 0.15 * address_overlap
        + 0.15 * number_overlap
        + 0.10 * postal_overlap
    )
    if not address_both_present:
        total = 0.88 * name_score + 0.12 * country_equal
    else:
        total = 0.70 * name_score + 0.25 * address_score + 0.05 * country_equal
    if not country_equal:
        total -= 0.20
    if address_key_equal and country_equal:
        total += 0.08
    if semantic_score is not None:
        # Cosine similarity is normally in [-1, 1]; map it to [0, 1] before blending.
        semantic = max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))
        total = 0.82 * total + 0.18 * semantic
    return max(0.0, min(1.0, total))


def is_high_confidence(features: Tuple[float, ...], score: float, threshold: float) -> bool:
    name_exact, compact_exact, ascii_exact = features[:3]
    country_equal = features[12]
    address_key_equal = features[11]
    if country_equal and (name_exact or compact_exact) and (address_key_equal or features[7] >= 0.25):
        return True
    if country_equal and ascii_exact and score >= threshold - 0.04:
        return True
    return score >= threshold
