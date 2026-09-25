from __future__ import annotations

import argparse
import json
import os

from .matcher import MatcherConfig, run_inference, run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-first business entity resolution baseline")
    parser.add_argument("--mode", choices=("validate", "infer", "all"), default="all")
    parser.add_argument("--data-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--candidate-cap", type=int, default=40)
    parser.add_argument("--raw-candidate-limit", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument("--embedding-model", default="none", help="Sentence-Transformer model name or local path; use none to disable")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_root = os.path.abspath(args.data_root)
    work_dir = os.path.abspath(args.work_dir or os.path.join(data_root, "work"))
    output_dir = os.path.abspath(args.output_dir or os.path.join(data_root, "output"))
    config = MatcherConfig(
        candidate_cap=args.candidate_cap,
        raw_candidate_limit=args.raw_candidate_limit,
        threshold=args.threshold,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        progress_every=args.progress_every,
        embedding_model=args.embedding_model,
        embedding_device=args.device,
        embedding_batch_size=args.embedding_batch_size,
    )
    if args.mode in ("validate", "all"):
        report = run_validation(data_root, work_dir, config, args.rebuild_index)
        print(json.dumps(report, indent=2))
    if args.mode in ("infer", "all"):
        outputs = run_inference(data_root, work_dir, output_dir, config, args.rebuild_index)
        print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
