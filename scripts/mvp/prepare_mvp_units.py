#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpm.data import build_pairwise_negatives, build_response_units, read_jsonl, summarize_units, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MVP LPM response units and pairwise negatives.")
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/demo_passages/candidates.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/mvp")
    parser.add_argument("--max-context-events", type=int, default=16)
    parser.add_argument("--min-context-events", type=int, default=1)
    parser.add_argument("--max-negatives-per-unit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_jsonl(args.candidates)
    units = build_response_units(
        candidates,
        max_context_events=args.max_context_events,
        min_context_events=args.min_context_events,
    )
    pairs = build_pairwise_negatives(
        units,
        max_negatives_per_unit=args.max_negatives_per_unit,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "units.jsonl", units)
    write_jsonl(args.output_dir / "pairs.jsonl", pairs)
    summary = summarize_units(units, pairs)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
