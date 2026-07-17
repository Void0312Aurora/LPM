#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpm.data import write_jsonl


BEHAVIOR_ACTION_TYPES = {"speak", "action"}
BEHAVIOR_TYPES = {"utterance", "monologue", "action"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare clean_event behavior units. For an agent c, the output is a "
            "current behavior emitted by c; the input context is the previous "
            "events visible to c, where participants/agent/target determine "
            "visibility."
        )
    )
    parser.add_argument("--graph", type=Path, default=ROOT / "data/clean_v0/graph.json")
    parser.add_argument("--events-dir", type=Path, default=ROOT / "data/clean_v0/events")
    parser.add_argument("--event-files", type=Path, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/clean_behavior")
    parser.add_argument("--max-trajectories", type=int, default=0, help="0 means all trajectories.")
    parser.add_argument("--max-units", type=int, default=0, help="0 means all units.")
    parser.add_argument("--max-context-events", type=int, default=32)
    parser.add_argument("--max-agent-context-events", type=int, default=16)
    parser.add_argument("--min-context-events", type=int, default=1)
    parser.add_argument("--max-negatives-per-unit", type=int, default=4)
    parser.add_argument("--same-trajectory-window", type=int, default=24)
    parser.add_argument("--random-negative-candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--strict-participants-only",
        action="store_true",
        help="Do not treat events with no people slots as public context.",
    )
    parser.add_argument(
        "--include-choice-options",
        action="store_true",
        help="Also turn valid choice options with an agent into behavior units.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def as_people(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            if isinstance(item, str) and item.strip() and item.strip() not in seen:
                result.append(item.strip())
                seen.add(item.strip())
        return result
    return []


def unique_people(*values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for person in as_people(value):
            if person not in seen:
                result.append(person)
                seen.add(person)
    return result


def event_people(event: dict[str, Any]) -> list[str]:
    return unique_people(event.get("agent"), event.get("target"), event.get("participants"))


def event_agents(event: dict[str, Any]) -> list[str]:
    return as_people(event.get("agent"))


def is_public_context(event: dict[str, Any]) -> bool:
    return not event_people(event) and bool(event.get("text") or event.get("choice"))


def visible_to(agent: str, event: dict[str, Any], strict_participants_only: bool) -> bool:
    if agent in event_people(event):
        return True
    return (not strict_participants_only) and is_public_context(event)


def behavior_text(event: dict[str, Any]) -> str:
    text = str(event.get("text") or "").strip()
    action_type = str(event.get("action_type") or "other")
    event_type = str(event.get("type") or "unknown")
    if text:
        return f"[{action_type}|{event_type}] {text}"
    return f"[{action_type}|{event_type}]"


def context_text(event: dict[str, Any]) -> str:
    parts = [
        str(event.get("kind") or "unknown"),
        str(event.get("type") or "unknown"),
        str(event.get("action_type") or "other"),
    ]
    agents = "/".join(as_people(event.get("agent"))) or "-"
    targets = "/".join(as_people(event.get("target"))) or "-"
    participants = "/".join(as_people(event.get("participants"))) or "-"
    text = str(event.get("text") or "").strip()
    if event.get("kind") == "choice" and isinstance(event.get("choice"), dict):
        options = event["choice"].get("options") or {}
        option_texts = []
        for index, option in sorted(options.items(), key=lambda item: str(item[0])):
            if isinstance(option, dict) and option.get("text"):
                option_texts.append(f"{index}:{option.get('text')}")
        text = " / ".join(option_texts)
    return f"[{'/'.join(parts)}|agent={agents}|target={targets}|participants={participants}] {text}".strip()


def make_context_text(events: Iterable[dict[str, Any]]) -> str:
    return "\n".join(context_text(event) for event in events)


def is_behavior_event(event: dict[str, Any]) -> bool:
    return (
        bool(event.get("valid", True))
        and event.get("kind") == "normal"
        and event.get("type") in BEHAVIOR_TYPES
        and event.get("action_type") in BEHAVIOR_ACTION_TYPES
        and bool(event_agents(event))
        and bool(str(event.get("text") or "").strip())
    )


def option_behavior_events(event: dict[str, Any]) -> list[dict[str, Any]]:
    if event.get("kind") != "choice" or not isinstance(event.get("choice"), dict):
        return []
    result: list[dict[str, Any]] = []
    options = event["choice"].get("options") or {}
    for index, option in sorted(options.items(), key=lambda item: str(item[0])):
        if not isinstance(option, dict):
            continue
        option_event = {
            **event,
            "kind": "choice_option",
            "type": option.get("type"),
            "action_type": option.get("action_type"),
            "agent": option.get("agent"),
            "target": option.get("target"),
            "participants": option.get("participants") or event.get("participants") or [],
            "text": option.get("text"),
            "valid": option.get("valid", event.get("valid", True)),
            "flag": option.get("flag") or [],
            "choice_option_index": str(index),
            "choice_parent_id": event.get("id"),
        }
        if (
            bool(option_event.get("valid", True))
            and option_event.get("type") in BEHAVIOR_TYPES
            and option_event.get("action_type") in BEHAVIOR_ACTION_TYPES
            and bool(event_agents(option_event))
            and bool(str(option_event.get("text") or "").strip())
        ):
            result.append(option_event)
    return result


def load_event_files(args: argparse.Namespace) -> list[tuple[str, str | None, Path]]:
    if args.event_files:
        return [(path.stem, None, path) for path in args.event_files]

    graph_path = args.graph
    if graph_path.exists():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes = graph.get("nodes") or {}
        items: list[tuple[str, str | None, Path]] = []
        for trajectory_id, node in sorted(nodes.items(), key=lambda item: int(item[1].get("entry", 0))):
            file_value = node.get("file")
            if not file_value:
                continue
            path = args.events_dir / Path(str(file_value)).name
            items.append((str(trajectory_id), node.get("scene"), path))
        return items

    return [(path.stem, None, path) for path in sorted(args.events_dir.glob("*.jsonl"))]


def normalize_event(raw: dict[str, Any], trajectory_id: str, scene: str | None) -> dict[str, Any]:
    event = dict(raw)
    event.setdefault("trajectory", trajectory_id)
    event.setdefault("order", int(event.get("id") or 0))
    if scene and not event.get("scene"):
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        event["scene"] = source.get("scene") or scene
    elif not event.get("scene"):
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        event["scene"] = source.get("scene")
    return event


def build_units_for_trajectory(
    trajectory_id: str,
    scene: str | None,
    events: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    sorted_events = sorted(events, key=lambda event: int(event.get("order") or 0))

    for raw_event in sorted_events:
        candidate_events = [raw_event]
        if args.include_choice_options:
            candidate_events.extend(option_behavior_events(raw_event))

        for event in candidate_events:
            if not is_behavior_event(event) and event.get("kind") != "choice_option":
                continue
            if event.get("kind") == "choice_option" and not event_agents(event):
                continue
            for agent in event_agents(event):
                visible_context = [
                    item for item in history if visible_to(agent, item, args.strict_participants_only)
                ][-args.max_context_events :]
                if len(visible_context) < args.min_context_events:
                    continue
                agent_history = [
                    item for item in visible_context if agent in as_people(item.get("agent"))
                ][-args.max_agent_context_events :]
                participant_context = [
                    item for item in visible_context if agent in as_people(item.get("participants"))
                ][-args.max_context_events :]
                order = int(event.get("order") or 0)
                option_suffix = ""
                if event.get("kind") == "choice_option":
                    option_suffix = f"::option{event.get('choice_option_index')}"
                unit_id = f"{trajectory_id}::order{order:06d}{option_suffix}::{agent}"
                participants = unique_people(event.get("participants"), event.get("agent"), event.get("target"))
                units.append(
                    {
                        "unit_id": unit_id,
                        "unit_type": "behavior",
                        "trajectory_id": trajectory_id,
                        "trajectory": trajectory_id,
                        "candidate_id": trajectory_id,
                        "candidate_type": "clean_behavior",
                        "scene": event.get("scene") or scene,
                        "order": order,
                        "target_character": agent,
                        "agent": event.get("agent"),
                        "target": event.get("target"),
                        "participants": participants,
                        "action_type": event.get("action_type"),
                        "event_type": event.get("type"),
                        "event_kind": event.get("kind"),
                        "response_pos": order,
                        "response_text": behavior_text(event),
                        "action_text": behavior_text(event),
                        "response_event": event,
                        "action_event": event,
                        "context_events": visible_context,
                        "visible_context_events": visible_context,
                        "character_context_events": agent_history,
                        "participant_context_events": participant_context,
                        "global_context_text": make_context_text(visible_context),
                        "character_context_text": make_context_text(agent_history),
                        "participant_context_text": make_context_text(participant_context),
                    }
                )

        history.append(raw_event)
    return units


def sample_random_other_agent(
    units: list[dict[str, Any]],
    unit: dict[str, Any],
    rng: random.Random,
    attempts: int,
) -> list[dict[str, Any]]:
    if len(units) <= 1:
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = {str(unit["unit_id"])}
    for _ in range(attempts):
        candidate = units[rng.randrange(len(units))]
        candidate_id = str(candidate["unit_id"])
        if candidate_id in seen:
            continue
        if candidate["target_character"] == unit["target_character"]:
            continue
        result.append(candidate)
        seen.add(candidate_id)
    return result


def build_pairs(units: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_trajectory[str(unit["trajectory_id"])].append(unit)
        by_agent[str(unit["target_character"])].append(unit)

    for trajectory_units in by_trajectory.values():
        trajectory_units.sort(key=lambda item: int(item.get("order") or 0))

    pairs: list[dict[str, Any]] = []
    for unit in units:
        unit_id = str(unit["unit_id"])
        agent = str(unit["target_character"])
        trajectory_id = str(unit["trajectory_id"])
        order = int(unit.get("order") or 0)

        choices: list[tuple[str, dict[str, Any]]] = []
        same_trajectory = by_trajectory.get(trajectory_id, [])
        window_other = [
            item
            for item in same_trajectory
            if item["unit_id"] != unit_id
            and item["target_character"] != agent
            and abs(int(item.get("order") or 0) - order) <= args.same_trajectory_window
        ]
        choices.extend(("same_trajectory_other_agent", item) for item in window_other)

        same_agent_other_state = [
            item
            for item in by_agent.get(agent, [])
            if item["unit_id"] != unit_id and item["trajectory_id"] != trajectory_id
        ]
        if len(same_agent_other_state) > args.random_negative_candidates:
            same_agent_other_state = rng.sample(same_agent_other_state, args.random_negative_candidates)
        choices.extend(("same_agent_temporal_mismatch", item) for item in same_agent_other_state)

        random_other = sample_random_other_agent(units, unit, rng, args.random_negative_candidates)
        choices.extend(("random_other_agent", item) for item in random_other)

        rng.shuffle(choices)
        seen_negatives: set[str] = set()
        for negative_type, negative in choices:
            negative_id = str(negative["unit_id"])
            if negative_id in seen_negatives or negative_id == unit_id:
                continue
            seen_negatives.add(negative_id)
            pairs.append(
                {
                    "pair_id": f"pair_{len(pairs):08d}",
                    "unit_id": unit_id,
                    "positive_unit_id": unit_id,
                    "negative_unit_id": negative_id,
                    "negative_type": negative_type,
                    "target_character": agent,
                    "trajectory_id": trajectory_id,
                    "candidate_id": trajectory_id,
                }
            )
            if len(seen_negatives) >= args.max_negatives_per_unit:
                break
    return pairs


def summarize(units: list[dict[str, Any]], pairs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    agents = Counter(str(unit["target_character"]) for unit in units)
    trajectories = Counter(str(unit["trajectory_id"]) for unit in units)
    action_types = Counter(str(unit.get("action_type") or "unknown") for unit in units)
    event_types = Counter(str(unit.get("event_type") or "unknown") for unit in units)
    negative_types = Counter(str(pair.get("negative_type") or "unknown") for pair in pairs)
    context_lengths = [len(unit.get("context_events") or []) for unit in units]
    participant_context_lengths = [len(unit.get("participant_context_events") or []) for unit in units]
    agent_context_lengths = [len(unit.get("character_context_events") or []) for unit in units]
    return {
        "unit_count": len(units),
        "pair_count": len(pairs),
        "trajectory_count": len(trajectories),
        "agent_count": len(agents),
        "agents": dict(agents.most_common(32)),
        "trajectories": dict(trajectories.most_common(32)),
        "action_types": dict(sorted(action_types.items())),
        "event_types": dict(sorted(event_types.items())),
        "negative_types": dict(sorted(negative_types.items())),
        "context": {
            "max_context_events": args.max_context_events,
            "max_agent_context_events": args.max_agent_context_events,
            "min_context_events": args.min_context_events,
            "strict_participants_only": args.strict_participants_only,
            "mean_visible_context": round(sum(context_lengths) / len(context_lengths), 6) if context_lengths else 0.0,
            "mean_agent_context": round(sum(agent_context_lengths) / len(agent_context_lengths), 6) if agent_context_lengths else 0.0,
            "mean_participant_context": round(sum(participant_context_lengths) / len(participant_context_lengths), 6)
            if participant_context_lengths
            else 0.0,
        },
    }


def main() -> int:
    args = parse_args()
    event_files = load_event_files(args)
    if args.max_trajectories > 0:
        event_files = event_files[: args.max_trajectories]
    if not event_files:
        raise SystemExit("no clean event files found")

    units: list[dict[str, Any]] = []
    for fallback_trajectory_id, scene, path in event_files:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        trajectory_id = str(rows[0].get("trajectory") or fallback_trajectory_id) if rows else fallback_trajectory_id
        normalized = [normalize_event(row, trajectory_id, scene) for row in rows]
        units.extend(build_units_for_trajectory(trajectory_id, scene, normalized, args))
        if args.max_units > 0 and len(units) >= args.max_units:
            units = units[: args.max_units]
            break

    pairs = build_pairs(units, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "units.jsonl", units)
    write_jsonl(args.output_dir / "pairs.jsonl", pairs)
    summary = {
        **summarize(units, pairs, args),
        "source": {
            "graph": str(args.graph),
            "events_dir": str(args.events_dir),
            "event_files": [str(path) for _, _, path in event_files],
        },
        "pairing": {
            "max_negatives_per_unit": args.max_negatives_per_unit,
            "same_trajectory_window": args.same_trajectory_window,
            "random_negative_candidates": args.random_negative_candidates,
            "seed": args.seed,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
