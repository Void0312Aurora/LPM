#!/usr/bin/env python3
"""Build Clean Event v0 files split by trajectory plus a trajectory table."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lpm.canonical import build_canonical_events, write_jsonl


DEFAULT_INPUT = REPO_ROOT / "data/upstream/Xi/GalGame/data/tree_dialogues_v2.jsonl"
DEFAULT_RAW_SCENE_DIR = REPO_ROOT / "data/upstream/Xi/GalGame/data/Orin/scene"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/clean_v0"
DEFAULT_CLEAN_SCHEMA = REPO_ROOT / "schemas/clean_event_v0.schema.json"
DEFAULT_GRAPH_SCHEMA = REPO_ROOT / "schemas/graph_v0.schema.json"

INVALID_TEXT_FLAGS = {"empty_text"}
SYSTEM_FLAGS = {"debug_or_system_scene"}
VALID_JUMP_STATUSES = {"candidate_narrative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--raw-scene-dir", type=Path, default=DEFAULT_RAW_SCENE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean-schema", type=Path, default=DEFAULT_CLEAN_SCHEMA)
    parser.add_argument("--graph-schema", type=Path, default=DEFAULT_GRAPH_SCHEMA)
    parser.add_argument("--max-trajectories", type=int, default=0, help="0 means all trajectories.")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def load_validator(schema_path: Path):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SystemExit("jsonschema is required unless --skip-validation is used") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[\\/:\s]+", "_", value).strip("._")
    return slug or "unknown"


def sorted_flags(flags: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return sorted({str(flag) for flag in flags if str(flag)})


def person(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def slot(values: list[str]) -> str | list[str] | None:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return unique


def participants(*items: str | list[str] | None) -> list[str]:
    values: list[str] = []
    for item in items:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, list):
            values.extend(value for value in item if isinstance(value, str))
    return sorted_flags(values)


def is_text_invalid(flags: list[str], text: str | None) -> bool:
    return text is None or bool(INVALID_TEXT_FLAGS.intersection(flags))


def clean_source(event: dict[str, Any]) -> dict[str, str]:
    return {
        "dataset": str(event.get("source_dataset") or ""),
        "scene": str(event.get("scene") or ""),
        "path": str(event.get("source_path") or ""),
    }


def invalid_clean_event(
    event: dict[str, Any],
    clean_id: int,
    trajectory_id: str,
    flags: list[str],
) -> dict[str, Any]:
    return {
        "id": clean_id,
        "source": clean_source(event),
        "trajectory": trajectory_id,
        "order": int(event.get("local_pos") or 0),
        "kind": "invalid",
        "type": "invalid",
        "action_type": "other",
        "agent": None,
        "target": None,
        "participants": [],
        "text": None,
        "choice": None,
        "valid": False,
        "flag": sorted_flags(flags),
    }


def option_from_branch(branch: dict[str, Any], inherited_flags: list[str]) -> dict[str, Any]:
    text = branch.get("option")
    text = text.strip() if isinstance(text, str) else None
    flags: list[str] = []
    if text is None:
        flags.append("empty_text")
    speaker = person(branch.get("speaker"))
    if text is None:
        option_type = "invalid"
        action_type = "other"
        valid = False
    elif speaker:
        option_type = "utterance"
        action_type = "speak"
        valid = True
    else:
        option_type = "other"
        action_type = "other"
        valid = True
    if SYSTEM_FLAGS.intersection(inherited_flags):
        flags.append("debug_or_system_scene")
        valid = False
    agent = slot([speaker] if speaker else [])
    return {
        "type": option_type,
        "action_type": action_type,
        "agent": agent,
        "target": None,
        "participants": participants(agent),
        "text": text or "",
        "valid": valid,
        "flag": sorted_flags(flags),
    }


def clean_event_from_canonical(event: dict[str, Any], clean_id: int, trajectory_id: str) -> dict[str, Any]:
    flags = sorted_flags(event.get("flags") or [])
    base = {
        "id": clean_id,
        "source": clean_source(event),
        "trajectory": trajectory_id,
        "order": int(event.get("local_pos") or 0),
        "flag": flags,
    }

    event_kind = event.get("event_kind")
    if event_kind == "text":
        text = event.get("text_norm")
        text = text if isinstance(text, str) and text else None
        if is_text_invalid(flags, text):
            return invalid_clean_event(event, clean_id, trajectory_id, flags or ["empty_text"])
        speaker = person(event.get("speaker_norm"))
        agent = slot([speaker] if speaker else [])
        content_type = "utterance" if speaker else "narration"
        action_type = "speak" if speaker else "other"
        valid = not bool(SYSTEM_FLAGS.intersection(flags))
        return {
            **base,
            "kind": "normal",
            "type": content_type,
            "action_type": action_type,
            "agent": agent,
            "target": None,
            "participants": participants(agent),
            "text": text,
            "choice": None,
            "valid": valid,
        }

    if event_kind == "choice":
        choice = event.get("choice") if isinstance(event.get("choice"), dict) else {}
        branches = [branch for branch in (choice.get("branches") or []) if isinstance(branch, dict)]
        if not branches:
            return invalid_clean_event(event, clean_id, trajectory_id, flags or ["degenerate_choice"])
        options: dict[str, dict[str, Any]] = {}
        for index, branch in enumerate(branches):
            options[str(index)] = option_from_branch(branch, flags)
        option_participants: list[str] = []
        for option in options.values():
            option_participants.extend(option.get("participants") or [])
        choice_flags = list(flags)
        if any(not option["valid"] for option in options.values()):
            choice_flags.append("invalid_choice_option")
        valid = not bool(SYSTEM_FLAGS.intersection(choice_flags))
        return {
            **base,
            "kind": "choice",
            "type": None,
            "action_type": "other",
            "agent": None,
            "target": None,
            "participants": sorted_flags(option_participants),
            "text": None,
            "choice": {"options": options},
            "valid": valid,
            "flag": sorted_flags(choice_flags),
        }

    if event_kind == "jump":
        jump = event.get("jump") if isinstance(event.get("jump"), dict) else {}
        status = str(jump.get("status") or "")
        jump_flags = list(flags)
        valid = status in VALID_JUMP_STATUSES and not bool(SYSTEM_FLAGS.intersection(jump_flags))
        if not valid:
            jump_flags.append("invalid_jump_target")
        return {
            **base,
            "kind": "jump",
            "type": None,
            "action_type": "other",
            "agent": None,
            "target": None,
            "participants": [],
            "text": None,
            "choice": None,
            "valid": valid,
            "flag": sorted_flags(jump_flags),
        }

    return invalid_clean_event(event, clean_id, trajectory_id, flags or ["unknown_event_kind"])


def scene_order(canonical_events: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for event in canonical_events:
        scene = str(event.get("scene") or "")
        if scene and scene not in seen:
            ordered.append(scene)
            seen.add(scene)
    return ordered


def trajectory_ids(scenes: list[str]) -> dict[str, str]:
    return {scene: f"t{index:04d}" for index, scene in enumerate(scenes)}


def trajectory_target(
    trajectory_id: str,
    event: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "kind": "trajectory",
        "trajectory_id": trajectory_id,
        "event_id": int(event["id"]) if event else None,
        "order": int(event["order"]) if event else None,
    }


def unresolved_target() -> dict[str, str]:
    return {"kind": "unresolved"}


def terminal_target() -> dict[str, str]:
    return {"kind": "terminal"}


def add_edge(
    edges: dict[str, dict[str, Any]],
    edge_type: str,
    source: dict[str, Any],
    target: dict[str, Any],
    valid: bool,
    flags: list[str] | None = None,
) -> None:
    edge_id = f"edge:{len(edges):06d}"
    edges[edge_id] = {
        "edge_type": edge_type,
        "source": source,
        "target": target,
        "valid": bool(valid),
        "flag": sorted_flags(flags or []),
    }


def edge_source(event: dict[str, Any], option_index: str | None = None) -> dict[str, Any]:
    return {
        "trajectory_id": str(event["trajectory"]),
        "event_id": int(event["id"]),
        "order": int(event["order"]),
        "option_index": option_index,
    }


def build_trajectory_edges(
    clean_events: list[dict[str, Any]],
    canonical_by_clean_id: dict[int, dict[str, Any]],
    text_index_map: dict[str, dict[int, list[dict[str, Any]]]],
    clean_by_trajectory: dict[str, list[dict[str, Any]]],
    scene_to_trajectory: dict[str, str],
) -> dict[str, dict[str, Any]]:
    edges: dict[str, dict[str, Any]] = {}
    dataset_trajectories = set(clean_by_trajectory)

    for event in clean_events:
        trajectory_id = str(event["trajectory"])
        canonical = canonical_by_clean_id[int(event["id"])]
        if event["kind"] == "choice":
            branches = ((canonical.get("choice") or {}).get("branches") or [])
            for option_index, option in (event.get("choice") or {}).get("options", {}).items():
                branch = branches[int(option_index)] if int(option_index) < len(branches) else {}
                leads_to = branch.get("leads_to") if isinstance(branch, dict) else None
                flags = list(option.get("flag") or [])
                if not isinstance(leads_to, dict):
                    add_edge(edges, "choice", edge_source(event, option_index), terminal_target(), False, flags + ["branch_missing_target"])
                    continue
                target_index = leads_to.get("index")
                if not isinstance(target_index, int):
                    add_edge(edges, "choice", edge_source(event, option_index), unresolved_target(), False, flags + ["branch_target_index_missing"])
                    continue
                matches = text_index_map.get(trajectory_id, {}).get(target_index, [])
                if len(matches) == 1:
                    add_edge(
                        edges,
                        "choice",
                        edge_source(event, option_index),
                        trajectory_target(trajectory_id, matches[0]),
                        bool(option["valid"]),
                        flags,
                    )
                elif len(matches) > 1:
                    add_edge(edges, "choice", edge_source(event, option_index), unresolved_target(), False, flags + ["branch_target_ambiguous"])
                else:
                    add_edge(edges, "choice", edge_source(event, option_index), unresolved_target(), False, flags + ["branch_target_missing"])

        if event["kind"] == "jump":
            jump = canonical.get("jump") if isinstance(canonical.get("jump"), dict) else {}
            target_scene = jump.get("target_scene")
            flags = list(event.get("flag") or [])
            target_trajectory = scene_to_trajectory.get(target_scene) if isinstance(target_scene, str) else None
            if target_trajectory and target_trajectory in dataset_trajectories:
                target_rows = clean_by_trajectory.get(target_trajectory) or []
                entry = target_rows[0] if target_rows else None
                add_edge(
                    edges,
                    "jump",
                    edge_source(event),
                    trajectory_target(target_trajectory, entry),
                    bool(event["valid"]),
                    flags,
                )
            else:
                add_edge(edges, "jump", edge_source(event), unresolved_target(), False, flags or ["jump_target_missing"])
    return edges


def write_events_by_trajectory(
    clean_by_trajectory: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    ordered_trajectories: list[str],
) -> dict[str, str]:
    events_dir = output_dir / "events"
    if events_dir.exists():
        shutil.rmtree(events_dir)
    events_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for index, trajectory_id in enumerate(ordered_trajectories):
        rows = clean_by_trajectory.get(trajectory_id) or []
        if not rows:
            continue
        path = events_dir / f"{trajectory_id}.jsonl"
        write_jsonl(path, rows)
        paths[trajectory_id] = str(path.relative_to(output_dir))
    return paths


def build_graph(
    clean_by_trajectory: dict[str, list[dict[str, Any]]],
    event_paths: dict[str, str],
    edges: dict[str, dict[str, Any]],
    ordered_trajectories: list[str],
    trajectory_to_scene: dict[str, str],
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    for trajectory_id in ordered_trajectories:
        rows = clean_by_trajectory.get(trajectory_id) or []
        if not rows:
            continue
        flags = sorted_flags(flag for row in rows for flag in row["flag"] if flag in SYSTEM_FLAGS)
        nodes[trajectory_id] = {
            "scene": trajectory_to_scene[trajectory_id],
            "file": event_paths[trajectory_id],
            "count": len(rows),
            "entry": int(rows[0]["id"]) if rows else None,
            "valid": any(bool(row["valid"]) for row in rows),
            "flag": flags,
        }
    return {
        "nodes": nodes,
        "edges": edges,
        "valid": any(item["valid"] for item in nodes.values()),
        "flag": [],
    }


def validate_rows(rows: list[dict[str, Any]], validator: Any) -> None:
    for index, row in enumerate(rows):
        try:
            validator.validate(row)
        except Exception as exc:
            raise SystemExit(f"clean event validation failed at row {index}: {exc}") from exc


def validate_graph(graph: dict[str, Any], validator: Any) -> None:
    try:
        validator.validate(graph)
    except Exception as exc:
        raise SystemExit(f"graph validation failed: {exc}") from exc


def validate_graph_integrity(graph: dict[str, Any], events_by_id: dict[int, dict[str, Any]]) -> dict[str, int]:
    issues: Counter[str] = Counter()
    nodes = graph["nodes"]
    for trajectory_id, node in nodes.items():
        if node["entry"] is not None and node["entry"] not in events_by_id:
            issues["entry_event_missing"] += 1
    for edge in graph["edges"].values():
        source = edge["source"]
        source_event = events_by_id.get(source["event_id"])
        if source["trajectory_id"] not in nodes:
            issues["source_trajectory_missing"] += 1
        if source_event is None:
            issues["source_event_missing"] += 1
        elif source_event["trajectory"] != source["trajectory_id"]:
            issues["source_event_wrong_trajectory"] += 1
        if source["option_index"] is not None:
            options = ((source_event or {}).get("choice") or {}).get("options") or {}
            if source["option_index"] not in options:
                issues["source_option_missing"] += 1
        target = edge["target"]
        if target["kind"] == "trajectory":
            if target["trajectory_id"] not in nodes:
                issues["target_trajectory_missing"] += 1
            if target["event_id"] is not None:
                target_event = events_by_id.get(target["event_id"])
                if target_event is None:
                    issues["target_event_missing"] += 1
                elif target_event["trajectory"] != target["trajectory_id"]:
                    issues["target_event_wrong_trajectory"] += 1
    return dict(sorted(issues.items()))


def main() -> int:
    args = parse_args()
    canonical_events, canonical_report = build_canonical_events(args.input, args.raw_scene_dir)
    scenes = scene_order(canonical_events)
    if args.max_trajectories:
        selected = set(scenes[: args.max_trajectories])
        canonical_events = [event for event in canonical_events if event.get("scene") in selected]
        scenes = scenes[: args.max_trajectories]
    scene_to_trajectory = trajectory_ids(scenes)
    trajectory_to_scene = {trajectory_id: scene for scene, trajectory_id in scene_to_trajectory.items()}
    trajectories = [scene_to_trajectory[scene] for scene in scenes]

    clean_events: list[dict[str, Any]] = []
    canonical_by_clean_id: dict[int, dict[str, Any]] = {}
    for clean_id, event in enumerate(canonical_events):
        scene = str(event.get("scene") or "")
        clean = clean_event_from_canonical(event, clean_id, scene_to_trajectory[scene])
        clean_events.append(clean)
        canonical_by_clean_id[clean_id] = event

    clean_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in clean_events:
        clean_by_trajectory[str(event["trajectory"])].append(event)
    for rows in clean_by_trajectory.values():
        rows.sort(key=lambda item: (int(item["order"]), int(item["id"])))

    text_index_map: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for clean in clean_events:
        canonical = canonical_by_clean_id[int(clean["id"])]
        text_index = canonical.get("text_index")
        if isinstance(text_index, int):
            text_index_map[str(clean["trajectory"])][text_index].append(clean)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_paths = write_events_by_trajectory(clean_by_trajectory, args.output_dir, trajectories)
    edges = build_trajectory_edges(clean_events, canonical_by_clean_id, text_index_map, clean_by_trajectory, scene_to_trajectory)
    graph = build_graph(clean_by_trajectory, event_paths, edges, trajectories, trajectory_to_scene)
    events_by_id = {int(event["id"]): event for event in clean_events}

    if not args.skip_validation:
        clean_validator = load_validator(args.clean_schema)
        graph_validator = load_validator(args.graph_schema)
        validate_rows(clean_events, clean_validator)
        validate_graph(graph, graph_validator)

    integrity_issues = validate_graph_integrity(graph, events_by_id)

    graph_path = args.output_dir / "graph.json"
    graph_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": {
            "clean_event": str(args.clean_schema),
            "graph": str(args.graph_schema),
        },
        "source": {
            "input": str(args.input),
            "raw_scene_dir": str(args.raw_scene_dir),
        },
        "outputs": {
            "events": str(args.output_dir / "events"),
            "graph": str(graph_path),
        },
        "nodes": [
            {
                "trajectory_id": trajectory_id,
                "scene": trajectory_to_scene[trajectory_id],
                "file": event_paths.get(trajectory_id),
            }
            for trajectory_id in trajectories
            if trajectory_id in event_paths
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    kind_counts = Counter(event["kind"] for event in clean_events)
    type_counts = Counter(str(event["type"]) for event in clean_events)
    edge_counts = Counter(edge["edge_type"] for edge in edges.values())
    invalid_edge_counts = Counter(edge["edge_type"] for edge in edges.values() if not edge["valid"])
    report = {
        "canonical_report": canonical_report,
        "clean_event_count": len(clean_events),
        "node_count": len(event_paths),
        "kind_counts": dict(sorted(kind_counts.items())),
        "type_counts": dict(sorted(type_counts.items())),
        "flag_counts": dict(sorted(Counter(flag for event in clean_events for flag in event["flag"]).items())),
        "invalid_clean_event_count": sum(1 for event in clean_events if not event["valid"]),
        "edge_count": len(edges),
        "edge_counts": dict(sorted(edge_counts.items())),
        "invalid_edge_counts": dict(sorted(invalid_edge_counts.items())),
        "integrity_issues": integrity_issues,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(clean_events)} clean events split into {len(event_paths)} event files")
    print(f"wrote graph to {graph_path}")
    print(f"wrote manifest to {args.output_dir / 'manifest.json'}")
    print(f"wrote report to {report_path}")
    if integrity_issues:
        print(f"graph integrity issues: {json.dumps(integrity_issues, ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
