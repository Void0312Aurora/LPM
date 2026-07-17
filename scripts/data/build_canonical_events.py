#!/usr/bin/env python3
"""Build deterministic canonical events from tree_dialogues_v2.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lpm.canonical import build_canonical_events, write_jsonl


DEFAULT_INPUT = REPO_ROOT / "data/upstream/Xi/GalGame/data/tree_dialogues_v2.jsonl"
DEFAULT_RAW_SCENE_DIR = REPO_ROOT / "data/upstream/Xi/GalGame/data/Orin/scene"
DEFAULT_OUTPUT = REPO_ROOT / "data/canonical/events.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data/canonical/report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--raw-scene-dir", type=Path, default=DEFAULT_RAW_SCENE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events, report = build_canonical_events(args.input, args.raw_scene_dir)
    write_jsonl(args.output, events)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(events)} canonical events to {args.output}")
    print(f"wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
