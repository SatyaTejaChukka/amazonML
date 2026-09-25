import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.features import pair_features, score_features
from business_entity_resolution.metrics import entity_f05, macro_f05
from business_entity_resolution.normalize import address_numbers, normalize_record


def test_unicode_and_domain_normalization():
    record = normalize_record({
        "entity_id": "S3-1",
        "business_name": "https://WilfordHancock.com",
        "business_address": "42 Rue de l'École",
        "country": "France",
    })
    assert record.name_norm == "wilfordhancock"
    assert "école" in record.address_norm
    assert record.name_compact == "wilfordhancock"


def test_address_number_extraction():
    assert address_numbers("No. 10/12, 42A Main Road") == ("10", "12", "42a")


def test_pair_features_and_score():
    left = normalize_record({"entity_id": "S1-1", "business_name": "Prime Money Inc", "business_address": "42 Main Rd, Delhi", "country": "India"})
    right = normalize_record({"entity_id": "S2-1", "business_name": "Prime Money", "business_address": "42 Main Road, Delhi", "country": "India"})
    features = pair_features(left, right)
    assert features[12] == 1.0
    assert score_features(features) > 0.50


def test_macro_f05_singleton_and_partial_match():
    truth = {"S1-1": set(), "S1-2": {"S2-1", "S3-1"}}
    assert entity_f05(set(), set()) == 1.0
    assert entity_f05({"S2-1"}, set()) == 0.0
    assert 0.0 < macro_f05({"S1-1": set(), "S1-2": {"S2-1"}}, truth) < 1.0
