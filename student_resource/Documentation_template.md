# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Fill in team name]  
**Team Members:** [Fill in team members]  
**Submission Date:** [Fill in submission date]

---

## 1. Executive Summary

This solution uses a CPU-first, streaming entity-resolution pipeline for the three-source business matching task. Source 2 and Source 3 records are indexed on disk with normalized names, address anchors, country, and informative name tokens; candidates are then scored with deterministic multilingual-safe lexical and address features and filtered using a validation-tuned F0.5 threshold.

The implementation does not use external identity databases, APIs, geocoders, or runtime model downloads. An embedding-provider interface is included as an optional future extension, but the submitted baseline is self-contained.

## 2. Methodology

### 2.1 Problem Analysis

Source 1 is the deduplicated reference source. Each Source 1 entity may have zero, one, or multiple matches in Source 2 and Source 3. The complete training ground truth contains 2,206,821 Source 1 rows and 7,638,365 matched IDs. The singleton rate is 5.58%, the mean number of matches is 3.46, and the maximum observed number of matches is 11.

Names contain legal-suffix variation, punctuation changes, typos, native scripts, transliteration, and Source 3 domain-style names. Addresses contain reordered components, abbreviations, partial values, numeric variation, and missing values. Test data includes France, which is absent from training, so country is treated as an open string feature rather than a fixed training category.

### 2.2 Solution Strategy

**Approach Type:** Streaming blocking plus deterministic pair scoring  
**Core Innovation:** Disk-backed, source-specific lexical indexes keep the approximately 10M-record S2/S3 test pool out of RAM while retaining exact-name, address-anchor, and rare-token retrieval paths.

The pipeline performs these stages:

1. Read UTF-8 TSV rows in a streaming fashion.
2. Generate Unicode-normalized name and address variants.
3. Build/reuse SQLite indexes for Source 2 and Source 3.
4. Generate candidates separately for each target source.
5. Score candidate pairs using name, address, numeric, postal, country, and missingness features.
6. Tune the acceptance threshold on a stable Source 1 holdout using macro F0.5.
7. Enforce the observed one-owner rule for S2/S3 IDs using highest-confidence ownership.
8. Write one row for every test Source 1 entity.

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** normalized name plus country, compact name, ASCII-folded name, address anchor plus country, and rare informative name tokens.
- **Candidate caps:** up to 40 ranked candidates per target source by default, after a raw retrieval limit of 250.
- **How true matches are protected:** exact normalized-name and compact-name matches are retrieved before token blocking; S2 and S3 are blocked independently; country is used as a feature and lookup aid, not as a fixed vocabulary filter.
- **Candidate output:** `candidate_pairs.tsv` is written from the same disk-backed candidate table that is passed to the scoring stage.

The full training ground truth shows no matched S2/S3 ID reused across different Source 1 entities. This supports the final conflict-resolution pass, but its contribution must still be measured on the generated validation report.

## 4. Matching Model

**Features used:**

- Name features: exact normalized equality, compact equality, ASCII-folded equality, sequence similarity, token Jaccard similarity, and token containment.
- Address features: sequence similarity, token overlap, number overlap, postal-code overlap, and address-anchor equality.
- Other: country equality, address-present indicators, source prefix, missingness-aware scoring, and legal/domain normalization variants.

**Model type:** Deterministic weighted scorer with validation-tuned acceptance threshold  
**Threshold selection method:** Grid search for macro F0.5 on a stable 20% Source 1 holdout. The selected value is written to `work/validation_report.json` and is used automatically by `--mode all`.

No external pretrained model is used by the baseline. `embeddings.py` defines an optional provider interface without downloading or activating a model.

## 5. Results & Error Analysis

- **F0.5 Score (macro):** Run `python -m business_entity_resolution.cli --mode validate`; copy `macro_f05` from `work/validation_report.json` here after execution.
- **Blocking recall:** Copy the measured candidate recall here after execution. This is the ceiling for the final matcher.
- **Common false positives:** Common business names, short names, shared legal suffixes, and records with weak or missing address evidence.
- **Common false negatives:** Native-script/transliteration pairs without shared normalized tokens, heavily corrupted names, and records whose only useful address components were not retrieved by a blocking key.

The validation report should be treated as the source of truth for final numerical claims; no score is stated here before the pipeline is run on the training data.

## 6. Conclusion

The baseline is designed for precision-heavy macro F0.5 evaluation and for the actual 10M-record target pool. Its disk-backed indexes and streaming inference keep memory bounded, while the modular embedding interface leaves room for a later recall-focused extension without changing the required output format.

## Appendix

### A. Code Artefacts

The runnable code is under `code/business_entity_resolution/`:

- `src/business_entity_resolution/normalize.py`: Unicode-safe record normalization.
- `src/business_entity_resolution/index.py`: SQLite record and token indexes.
- `src/business_entity_resolution/features.py`: pair features and deterministic score.
- `src/business_entity_resolution/matcher.py`: validation, blocking, assignment, and output generation.
- `src/business_entity_resolution/cli.py`: command-line entry point.

From `student_resource/code/business_entity_resolution/`, set `PYTHONPATH=src` and run:

```powershell
python -m business_entity_resolution.cli --mode all --data-root ..\..
```

The required files are written to `student_resource/output/`. Validate them from `student_resource/` with `utils/validate_submission.py`.

### B. Additional Results

The full source counts are 2,206,821 / 5,034,616 / 5,285,603 for training S1/S2/S3 and 1,732,544 / 4,887,273 / 5,082,316 for test S1/S2/S3. Test S1 contains 259,452 France records; France does not appear in training.
