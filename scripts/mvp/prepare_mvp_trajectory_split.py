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

from lpm.data import read_jsonl, write_jsonl


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict trajectory-aware MVP split. A pair is kept only when "
            "its state/positive/negative units all belong to trajectories assigned "
            "to the same split."
        )
    )
    parser.add_argument("--units", type=Path, default=ROOT / "data/mvp/units.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/mvp/pairs.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/mvp_trajectory_split")
    parser.add_argument(
        "--trajectory-field",
        default="auto",
        help="Unit field used as trajectory id. 'auto' tries trajectory_id, trajectory, scene, candidate_id.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--search-iterations", type=int, default=5000)
    return parser.parse_args()


def trajectory_id(unit: dict[str, Any], field: str) -> str:
    if field != "auto":
        value = unit.get(field)
        if value is None or value == "":
            raise ValueError(f"unit {unit.get('unit_id')} is missing trajectory field {field!r}")
        return str(value)
    for key in ("trajectory_id", "trajectory", "scene", "candidate_id"):
        value = unit.get(key)
        if value is not None and value != "":
            return str(value)
    unit_id = str(unit.get("unit_id") or "")
    if "::" in unit_id:
        return unit_id.split("::", 1)[0]
    raise ValueError(f"cannot infer trajectory for unit {unit_id!r}")


def normalize_fractions(args: argparse.Namespace) -> dict[str, float]:
    raw = {
        "train": float(args.train_fraction),
        "val": float(args.val_fraction),
        "test": float(args.test_fraction),
    }
    if any(value < 0 for value in raw.values()):
        raise ValueError(f"fractions must be non-negative: {raw}")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("at least one split fraction must be positive")
    return {key: value / total for key, value in raw.items()}


def pair_trajectories(pair: dict[str, Any], unit_to_trajectory: dict[str, str]) -> set[str] | None:
    ids = (
        str(pair.get("unit_id") or ""),
        str(pair.get("positive_unit_id") or ""),
        str(pair.get("negative_unit_id") or ""),
    )
    trajectories: set[str] = set()
    for unit_id in ids:
        trajectory = unit_to_trajectory.get(unit_id)
        if trajectory is None:
            return None
        trajectories.add(trajectory)
    return trajectories


def split_pair_counts(
    pairs: list[dict[str, Any]],
    pair_to_trajectories: list[set[str] | None],
    assignment: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    split_pairs: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    dropped = 0
    for pair, trajectories in zip(pairs, pair_to_trajectories, strict=True):
        if not trajectories:
            dropped += 1
            continue
        assigned = {assignment.get(trajectory) for trajectory in trajectories}
        if len(assigned) != 1 or None in assigned:
            dropped += 1
            continue
        split_pairs[assigned.pop()].append(pair)
    return split_pairs, dropped


def score_assignment(
    units_by_trajectory: dict[str, list[dict[str, Any]]],
    pairs: list[dict[str, Any]],
    pair_to_trajectories: list[set[str] | None],
    assignment: dict[str, str],
    fractions: dict[str, float],
    negative_types: set[str],
) -> tuple[float, dict[str, list[dict[str, Any]]]]:
    split_pairs, _ = split_pair_counts(pairs, pair_to_trajectories, assignment)
    retained = sum(len(items) for items in split_pairs.values())
    if retained == 0:
        return -1e18, split_pairs

    pair_deviation = 0.0
    for split in SPLITS:
        pair_deviation += abs((len(split_pairs[split]) / retained) - fractions[split])

    total_units = sum(len(items) for items in units_by_trajectory.values())
    units_by_split: Counter[str] = Counter()
    for trajectory, split in assignment.items():
        units_by_split[split] += len(units_by_trajectory[trajectory])
    unit_deviation = sum(abs((units_by_split[split] / total_units) - fractions[split]) for split in SPLITS)

    missing_type_penalty = 0
    for split in SPLITS:
        present = {str(pair.get("negative_type") or "unknown") for pair in split_pairs[split]}
        missing_type_penalty += len(negative_types - present)

    empty_penalty = sum(1 for split in SPLITS if not split_pairs[split]) * retained
    score = retained - (0.75 * retained * pair_deviation) - (0.25 * retained * unit_deviation)
    score -= missing_type_penalty * 25
    score -= empty_penalty
    return score, split_pairs


def random_assignment(
    trajectories: list[str],
    units_by_trajectory: dict[str, list[dict[str, Any]]],
    fractions: dict[str, float],
    rng: random.Random,
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    units_by_split: Counter[str] = Counter()
    total_units = sum(len(items) for items in units_by_trajectory.values())
    ordered = list(trajectories)
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: len(units_by_trajectory[item]), reverse=True)

    for trajectory in ordered:
        size = len(units_by_trajectory[trajectory])
        best_split = None
        best_score = None
        for split in SPLITS:
            trial = Counter(units_by_split)
            trial[split] += size
            deviation = sum(abs((trial[name] / total_units) - fractions[name]) for name in SPLITS)
            jitter = rng.random() * 1e-6
            value = deviation + jitter
            if best_score is None or value < best_score:
                best_score = value
                best_split = split
        assignment[trajectory] = str(best_split)
        units_by_split[str(best_split)] += size
    return assignment


def find_assignment(
    units_by_trajectory: dict[str, list[dict[str, Any]]],
    pairs: list[dict[str, Any]],
    pair_to_trajectories: list[set[str] | None],
    fractions: dict[str, float],
    seed: int,
    iterations: int,
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], float]:
    trajectories = sorted(units_by_trajectory)
    negative_types = {str(pair.get("negative_type") or "unknown") for pair in pairs}
    best_assignment: dict[str, str] | None = None
    best_pairs: dict[str, list[dict[str, Any]]] | None = None
    best_score = -1e18
    rng = random.Random(seed)

    for _ in range(max(1, iterations)):
        assignment = random_assignment(trajectories, units_by_trajectory, fractions, rng)
        score, split_pairs = score_assignment(
            units_by_trajectory,
            pairs,
            pair_to_trajectories,
            assignment,
            fractions,
            negative_types,
        )
        if score > best_score:
            best_score = score
            best_assignment = assignment
            best_pairs = split_pairs

    if best_assignment is None or best_pairs is None:
        raise RuntimeError("failed to build a trajectory split")
    return best_assignment, best_pairs, best_score


def split_units(
    units_by_trajectory: dict[str, list[dict[str, Any]]],
    assignment: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for trajectory in sorted(units_by_trajectory):
        by_split[assignment[trajectory]].extend(units_by_trajectory[trajectory])
    return by_split


def summarize_split(
    split: str,
    units: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    unit_to_trajectory: dict[str, str],
) -> dict[str, Any]:
    unit_ids = {str(unit["unit_id"]) for unit in units}
    referenced_unit_ids: set[str] = set()
    negative_types: Counter[str] = Counter()
    for pair in pairs:
        referenced_unit_ids.update(
            [
                str(pair.get("unit_id") or ""),
                str(pair.get("positive_unit_id") or ""),
                str(pair.get("negative_unit_id") or ""),
            ]
        )
        negative_types[str(pair.get("negative_type") or "unknown")] += 1
    trajectories = sorted({unit_to_trajectory[unit_id] for unit_id in unit_ids if unit_id in unit_to_trajectory})
    return {
        "split": split,
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
        "unit_count": len(units),
        "referenced_unit_count": len(referenced_unit_ids),
        "pair_count": len(pairs),
        "negative_type_counts": dict(sorted(negative_types.items())),
    }


def leakage_report(
    split_units_by_name: dict[str, list[dict[str, Any]]],
    split_pairs: dict[str, list[dict[str, Any]]],
    unit_to_trajectory: dict[str, str],
) -> dict[str, Any]:
    trajectory_sets: dict[str, set[str]] = {}
    unit_sets: dict[str, set[str]] = {}
    for split, units in split_units_by_name.items():
        unit_sets[split] = {str(unit["unit_id"]) for unit in units}
        trajectory_sets[split] = {unit_to_trajectory[unit_id] for unit_id in unit_sets[split]}

    overlaps: dict[str, list[str]] = {}
    split_list = list(SPLITS)
    for left_index, left in enumerate(split_list):
        for right in split_list[left_index + 1 :]:
            overlaps[f"{left}_vs_{right}_trajectories"] = sorted(trajectory_sets[left] & trajectory_sets[right])
            overlaps[f"{left}_vs_{right}_units"] = sorted(unit_sets[left] & unit_sets[right])

    violations: list[dict[str, Any]] = []
    for split, pairs in split_pairs.items():
        allowed = trajectory_sets[split]
        for pair in pairs:
            ids = [str(pair["unit_id"]), str(pair["positive_unit_id"]), str(pair["negative_unit_id"])]
            trajectories = sorted({unit_to_trajectory[unit_id] for unit_id in ids})
            if any(trajectory not in allowed for trajectory in trajectories):
                violations.append({"split": split, "pair_id": pair.get("pair_id"), "trajectories": trajectories})
    return {
        "trajectory_overlaps": {key: value for key, value in overlaps.items() if key.endswith("_trajectories")},
        "unit_overlaps": {key: value for key, value in overlaps.items() if key.endswith("_units")},
        "pair_trajectory_violations": violations,
        "has_leakage": any(overlaps.values()) or bool(violations),
    }


def main() -> int:
    args = parse_args()
    fractions = normalize_fractions(args)
    units = read_jsonl(args.units)
    pairs = read_jsonl(args.pairs)
    unit_to_trajectory = {str(unit["unit_id"]): trajectory_id(unit, args.trajectory_field) for unit in units}
    units_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        units_by_trajectory[unit_to_trajectory[str(unit["unit_id"])]].append(unit)

    pair_to_trajectories = [pair_trajectories(pair, unit_to_trajectory) for pair in pairs]
    assignment, split_pairs, score = find_assignment(
        units_by_trajectory=dict(units_by_trajectory),
        pairs=pairs,
        pair_to_trajectories=pair_to_trajectories,
        fractions=fractions,
        seed=args.seed,
        iterations=args.search_iterations,
    )
    split_units_by_name = split_units(dict(units_by_trajectory), assignment)
    leak = leakage_report(split_units_by_name, split_pairs, unit_to_trajectory)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(args.output_dir / split / "units.jsonl", split_units_by_name[split])
        write_jsonl(args.output_dir / split / "pairs.jsonl", split_pairs[split])

    total_retained = sum(len(items) for items in split_pairs.values())
    total_dropped = len(pairs) - total_retained
    summary = {
        "source_units": str(args.units),
        "source_pairs": str(args.pairs),
        "trajectory_field": args.trajectory_field,
        "fractions": fractions,
        "seed": args.seed,
        "search_iterations": args.search_iterations,
        "score": round(score, 6),
        "source_unit_count": len(units),
        "source_pair_count": len(pairs),
        "trajectory_count": len(units_by_trajectory),
        "retained_pair_count": total_retained,
        "dropped_pair_count": total_dropped,
        "retained_pair_fraction": round(total_retained / len(pairs), 6) if pairs else 0.0,
        "splits": {
            split: summarize_split(split, split_units_by_name[split], split_pairs[split], unit_to_trajectory)
            for split in SPLITS
        },
        "assignment": dict(sorted(assignment.items())),
        "leakage": leak,
    }
    (args.output_dir / "trajectory_split.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if leak["has_leakage"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
