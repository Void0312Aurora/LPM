#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpm.data import read_jsonl, summarize_units, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a pair-closed MVP subset for embedding/ranker validation.")
    parser.add_argument("--units", type=Path, default=ROOT / "data/mvp/units.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/mvp/pairs.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/mvp_qwen9b_smoke")
    parser.add_argument("--max-pairs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def stratified_pairs(pairs: list[dict[str, Any]], max_pairs: int, seed: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_type[str(pair.get("negative_type") or "unknown")].append(pair)
    rng = random.Random(seed)
    for items in by_type.values():
        rng.shuffle(items)

    selected: list[dict[str, Any]] = []
    types = sorted(by_type)
    cursor = 0
    while len(selected) < max_pairs and any(by_type.values()):
        negative_type = types[cursor % len(types)]
        cursor += 1
        if not by_type[negative_type]:
            continue
        selected.append(by_type[negative_type].pop())
    return selected


def remap_pair_ids(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        item = dict(pair)
        item["pair_id"] = f"subset_pair_{index:06d}"
        remapped.append(item)
    return remapped


def main() -> int:
    args = parse_args()
    units = read_jsonl(args.units)
    pairs = read_jsonl(args.pairs)
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    selected_pairs = remap_pair_ids(stratified_pairs(pairs, args.max_pairs, args.seed))
    selected_unit_ids: set[str] = set()
    for pair in selected_pairs:
        selected_unit_ids.add(str(pair["unit_id"]))
        selected_unit_ids.add(str(pair["positive_unit_id"]))
        selected_unit_ids.add(str(pair["negative_unit_id"]))
    selected_units = [unit_by_id[unit_id] for unit_id in sorted(selected_unit_ids) if unit_id in unit_by_id]
    available = {unit["unit_id"] for unit in selected_units}
    selected_pairs = [
        pair
        for pair in selected_pairs
        if pair["unit_id"] in available and pair["positive_unit_id"] in available and pair["negative_unit_id"] in available
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "units.jsonl", selected_units)
    write_jsonl(args.output_dir / "pairs.jsonl", selected_pairs)
    summary = {
        **summarize_units(selected_units, selected_pairs),
        "source_units": str(args.units),
        "source_pairs": str(args.pairs),
        "requested_max_pairs": args.max_pairs,
        "negative_type_counts": dict(sorted(Counter(pair["negative_type"] for pair in selected_pairs).items())),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
