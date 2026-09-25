# CPU-First Business Entity Resolution

This package implements the baseline matcher for the ML Challenge 2026 business entity resolution task. It reads the UTF-8 TSV files under `student_resource/dataset`, builds disk-backed SQLite indexes for Source 2 and Source 3, generates lexical candidates, scores them with deterministic multilingual-safe features, applies the one-owner conflict rule, and writes the two required submission files.

The matcher is deliberately streaming and does not load the approximately 10M-record S2/S3 test pool into RAM. `rapidfuzz` is optional; when it is unavailable, the standard-library sequence matcher is used.

## Run

From `student_resource/code/business_entity_resolution`:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src"
python -m business_entity_resolution.cli --mode validate --data-root ..\..
python -m business_entity_resolution.cli --mode infer --data-root ..\..
```

Use `--mode all` to validate first and then run inference. The default output directory is `student_resource/output`; indexes and validation reports are stored in `student_resource/work`.

Useful controls:

```text
--candidate-cap 40          Maximum ranked candidates retained per source
--raw-candidate-limit 250  Maximum raw candidates considered per source
--threshold 0.78            Initial acceptance threshold; validation tunes it
--validation-fraction 0.20 Stable Source 1 holdout fraction
--progress-every 5000      Print progress every N processed rows
--rebuild-index             Recreate SQLite indexes instead of reusing them
--embedding-model none      Optional Sentence-Transformer model for GPU reranking
--device auto               Embedding device: auto, cpu, or cuda
--embedding-batch-size 16   Small batch size suitable for a 4 GB GPU
```

The command prints progress for both index passes, candidate generation, and scoring. The full inference run is expected to be disk- and CPU-intensive. It can be resumed by reusing the generated indexes; use `--rebuild-index` only after an interrupted or stale index build. The default path uses no external API, geocoder, business database, or runtime model download.

## Optional GPU reranking

The RTX GPU path does not accelerate SQLite indexing or Unicode normalization. It reranks already-blocked candidate pairs with a Sentence-Transformer model, avoiding an embedding index for all 10M target records. Install the optional dependencies, then provide a model name or local model directory:

```powershell
pip install -r requirements-gpu.txt

python -m business_entity_resolution.cli `
  --mode infer `
  --data-root $PWD `
  --work-dir "$PWD\work" `
  --output-dir "$PWD\output" `
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --device cuda `
  --embedding-batch-size 8 `
  --progress-every 50000
```

The first use may download the model. For a no-network/fair-play run, replace the model name with an approved local model directory. `--device cuda` fails clearly if the installed PyTorch build cannot see CUDA.

## Outputs

The pipeline writes:

- `output/matching_results.tsv`: one row for every test Source 1 ID.
- `output/candidate_pairs.tsv`: the exact candidate IDs scored before final acceptance.

Validate the result from `student_resource` with:

```powershell
python utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test
```

## Package layout

`src/business_entity_resolution/normalize.py` contains Unicode-safe normalization, `index.py` builds SQLite indexes, `features.py` scores candidate pairs, `matcher.py` orchestrates validation and inference, and `cli.py` exposes the command-line entry point. `embeddings.py` defines an optional provider interface but does not download or activate a model.
