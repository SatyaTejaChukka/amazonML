from __future__ import annotations

from typing import Dict, Iterable, Mapping, Set, Tuple


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    return ((1.0 + beta2) * precision * recall / denominator) if denominator else 0.0


def entity_f05(predicted: Set[str], actual: Set[str]) -> float:
    if not predicted and not actual:
        return 1.0
    if not predicted:
        return 0.0
    precision = len(predicted & actual) / len(predicted)
    recall = len(predicted & actual) / len(actual) if actual else 0.0
    return fbeta(precision, recall, beta=0.5)


def macro_f05(predictions: Mapping[str, Set[str]], truth: Mapping[str, Set[str]]) -> float:
    if not truth:
        return 0.0
    return sum(entity_f05(predictions.get(key, set()), value) for key, value in truth.items()) / len(truth)


def summarize_predictions(predictions: Mapping[str, Set[str]], truth: Mapping[str, Set[str]]) -> Dict[str, float]:
    scores = [entity_f05(predictions.get(key, set()), value) for key, value in truth.items()]
    matched_truth = sum(len(value) for value in truth.values())
    matched_pred = sum(len(predictions.get(key, set())) for key in truth)
    true_positive = sum(len(predictions.get(key, set()) & value) for key, value in truth.items())
    return {
        "macro_f05": sum(scores) / len(scores) if scores else 0.0,
        "entities": float(len(scores)),
        "truth_matches": float(matched_truth),
        "predicted_matches": float(matched_pred),
        "true_positive_matches": float(true_positive),
        "singleton_f05": (
            sum(entity_f05(predictions.get(key, set()), value) for key, value in truth.items() if not value)
            / sum(not value for value in truth.values())
            if any(not value for value in truth.values()) else 0.0
        ),
    }
