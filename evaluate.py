"""Evaluate one generated dataset with paper-aligned and common metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import evaluate_generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Reference split. Use validation for tuning and keep test sealed.",
    )
    parser.add_argument("--metric-samples", type=int, default=1_000)
    parser.add_argument(
        "--transport-samples",
        type=int,
        default=256,
        help="Seeded subsample size for the joint Wasserstein assignment.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_generated(
        args.data,
        args.generated,
        split_name=args.split,
        metric_samples=args.metric_samples,
        transport_samples=args.transport_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
