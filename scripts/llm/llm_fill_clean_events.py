#!/usr/bin/env python3
"""LLM-fill mutable fields in Clean Event v0 rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lpm.canonical import read_jsonl, write_jsonl


DEFAULT_INPUT = REPO_ROOT / "data/clean_v0/events/t0000.jsonl"
DEFAULT_SCHEMA = REPO_ROOT / "schemas/clean_event_v0.schema.json"
DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/enriched/clean_event_fill"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

NORMAL_TYPES = ["utterance", "action", "monologue", "narration", "other"]
OPTION_TYPES = ["utterance", "action", "monologue", "narration", "other", "invalid"]
ACTION_TYPES = ["speak", "action", "other"]
PATCH_FLAGS = [
    "ambiguous_actor",
    "ambiguous_target",
    "participant_inferred",
    "target_inferred",
    "viewpoint_mediated",
    "text_noise",
    "system_or_debug",
    "needs_review",
    "low_confidence",
    "branch_incomplete",
]
TARGET_EVIDENCE = [
    "explicit_name_or_vocative",
    "imperative_or_request_recipient",
    "explicit_second_person",
    "direct_reply_or_turn_taking",
    "direct_physical_interaction",
    "none",
]
FIELD_DEFINITIONS = {
    "type": "Observation category for the event, not a rewrite of text.",
    "action_type": "Coarse behavior category derived from type.",
    "agent": "Speaker, actor, or experiencer directly supported by text/context.",
    "target": "Interactive counterpart of this event: the character(s) directly addressed, replied to, requested from, or physically interacted with. Not a topic, mentioned person, merely affected subject, or mere co-present character.",
    "participants": "Characters who participate in, can perceive, or are directly affected by this event; do not include people who are merely mentioned or only thought about. For private monologue, this is normally only the experiencer.",
    "valid": "Whether the row is usable as a narrative event.",
    "flag": "Closed-set uncertainty or quality notes.",
}
REQUIRED_PATCH_EXAMPLE = {
    "event_patches": [
        {
            "id": "integer id of the requested target event",
            "type": "for kind=normal only; one of normal_type",
            "action_type": "one of action_type",
            "agent": "string, array of strings, or null",
            "target": "string, array of strings, or null",
            "participants": [],
            "valid": True,
            "flag": [],
            "target_evidence": "one of target_evidence; use none when target is null",
            "choice_options": {
                "0": {
                    "type": "for kind=choice options only; one of choice_option_type",
                    "action_type": "one of action_type",
                    "agent": "string, array of strings, or null",
                    "target": "string, array of strings, or null",
                    "participants": [],
                    "valid": True,
                    "flag": [],
                    "target_evidence": "one of target_evidence; use none when target is null",
                }
            },
            "confidence": 0.0,
            "short_reason": "brief reason; do not rewrite source text",
        }
    ]
}
FILL_RULES = [
    "Annotate every requested target event exactly once; use history/context events only as context.",
    "If future_context is provided, use it only as right-side context for interpreting the current event; do not annotate future_context events and do not copy their draft fields.",
    "Draft fields are deterministic or previous-pass hints. They may be incomplete or wrong; do not copy them blindly.",
    "History events may include type, action_type, and agent from the previous pass; use them to track speaker/actor continuity and active interlocutors.",
    "Do not invent character names or aliases when a canonical name is available in history or draft fields.",
    "Never rewrite source text and never change id, source, trajectory, order, kind, text, or choice option indexes.",
    "For kind=normal, choose type from utterance/action/monologue/narration/other; do not use invalid.",
    "agent is the speaker, actor, or experiencer when locally supported; otherwise null.",
    "target is an interaction role, not a generic conversation partner, topic, mentioned person, or broad affected-subject role.",
    "Fill target only when the event itself has a directed interaction with that character.",
    "utterance means outward speech; action_type must be speak.",
    "For utterance, target is the direct interaction counterpart of the speech act, not the person being discussed.",
    "For utterance, fill target when there is a vocative, imperative/request with a supported recipient, explicit second-person reference, clearly named addressee, or direct reply/turn-taking relationship.",
    "A unique active interlocutor is not enough by itself; use direct_reply_or_turn_taking only when the utterance clearly functions as a reply, acknowledgement, correction, answer, greeting, request, or backchannel to that interlocutor.",
    "Leave target null for comments, narration-like remarks, self-directed speech, topic statements, or ambiguous utterances even in a two-person scene; use participants for co-present/visible people.",
    "Do not fill target from mere co-presence, mere mention, narration, monologue, or a multi-party exchange with ambiguous interaction counterpart.",
    "Every non-null target must be supported by target_evidence other than none.",
    "When target is inferred rather than explicitly named, add target_inferred to flag.",
    "If target is uncertain, leave target null and use participants plus ambiguous_target or needs_review when useful.",
    "A merely mentioned third party is neither target nor participant unless that character is present, can perceive the event, participates in it, or is directly affected by it.",
    "action means concrete behavior by one or more actors; action_type must be action.",
    "For action, target is the direct interaction counterpart when clear; many actions have no target.",
    "For action, fill target for events like handing something to a character, looking at a character, touching/hitting/helping/pulling a character, or moving toward a character.",
    "For action, leave target null for self-contained actions, object-only interactions, scene changes, body states, or descriptions where a character is only the subject/topic.",
    "monologue means a character's unspoken inner thought, subjective feeling, motivation, self-question, self-decision, or internal psychological reaction; action_type must be other.",
    "Classify internal states like feeling strange, mission-like resolve, hesitation, realization, embarrassment, or self-directed questions as monologue even when phrased in third person or without explicit words like 'I think' or 'inside'.",
    "For monologue, agent is the experiencer/viewpoint character and participants must normally contain that same character; target must normally be null.",
    "For private monologue, participants should be exactly the experiencer/agent unless the text explicitly states a shared mental experience; characters merely mentioned, imagined, remembered, compared, or thought about must not be added to participants.",
    "For monologue with first-person wording such as 我, resolve agent to the locally supported viewpoint character from context; do not output 我 as a character name.",
    "For monologue with implicit experiencer, infer the most plausible character from first-person wording, nearby subjective narration, scene viewpoint, current speaker continuity, source speaker labels, and future_context when provided; add participant_inferred, and add ambiguous_actor when uncertainty remains.",
    "After a quoted line, a speakerless psychological beat may describe the quoted speaker's visible/expressed state or the viewpoint character's private reaction; decide from local semantic anchoring, not adjacency alone.",
    "Previous speaker continuity is valid evidence only when the beat elaborates that speaker's stated intent, expression, or observable state; otherwise check broader viewpoint evidence.",
    "Consecutive speakerless fragments that elaborate the same private feeling, personal evaluation, self-question, or lived situation usually form one monologue chain; do not split a middle fragment into narration unless it is clearly objective scene description.",
    "Do not leave monologue agent/participants empty unless no plausible experiencer exists after using local context; empty monologue participants should be rare and require needs_review.",
    "narration means external scene/state description, objective visible behavior, transitions, or plot exposition; action_type must be other.",
    "For narration, use participants for characters present, observed, involved, directly affected, or jointly affected by a local interaction beat/transition; do not add people who are merely mentioned in narration.",
    "For pure object/background description, participants should be empty unless a character is directly observing, interacting with, or affected by it.",
    "For speakerless private reflection, subjective feeling, psychological state, rhetorical questions, self-evaluation, or self-decision, prefer monologue over narration.",
    "For observable behavior by a clear actor, including completed behavior mentioned in narration, prefer action/action.",
    "For kind=choice, keep top-level type null in the merged row; fill only choice_options and top participants/valid/flag.",
    "Use target only when the interactive counterpart is locally supported; otherwise null plus ambiguous_target if important.",
    "participants must include all clear agents and targets, plus other characters present, visible, involved, or directly affected in the event; mere mention is insufficient.",
    "valid=false only for system/debug/noise/unusable rows; ambiguous but usable rows stay valid=true with needs_review or ambiguous_* flags.",
    "If confidence is below 0.7, add low_confidence or needs_review.",
]


def make_system_prompt() -> str:
    spec = {
        "role": "strict structured data cleaner for Clean Event v0",
        "task": "Fill only mutable Clean Event v0 fields for GalGame LPM data cleaning. Return JSON only.",
        "closed_sets": {
            "normal_type": NORMAL_TYPES,
            "choice_option_type": OPTION_TYPES,
            "action_type": ACTION_TYPES,
            "flag": PATCH_FLAGS,
            "target_evidence": TARGET_EVIDENCE,
        },
        "field_definitions": FIELD_DEFINITIONS,
        "required_output": REQUIRED_PATCH_EXAMPLE,
        "rules": FILL_RULES,
    }
    return (
        "You are a strict structured data cleaner for Clean Event v0. "
        "You only return machine-readable JSON patches. You preserve provenance and source text.\n"
        + json.dumps(spec, ensure_ascii=False, indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["sequential", "window"],
        default="sequential",
        help="sequential fills one event at a time with long trajectory history; window keeps the legacy batch mode.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--event-files",
        type=Path,
        nargs="*",
        default=None,
        help="Clean Event jsonl files to process together in sequential mode.",
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=None,
        help="Directory of per-trajectory Clean Event jsonl files for sequential mode.",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="Maximum number of event files to process in sequential mode; 0 means no file-count limit.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=32,
        help="Maximum number of target events to prepare in sequential mode; 0 means all selected events.",
    )
    parser.add_argument(
        "--max-events-per-trajectory",
        type=int,
        default=0,
        help="Maximum target events per trajectory in sequential mode; 0 means no per-trajectory limit.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=0,
        help="Maximum previous events kept as sequential history per trajectory; 0 means all previous events.",
    )
    parser.add_argument(
        "--startup-lookahead",
        type=int,
        default=4,
        help="Number of future raw context events included for the first events of each trajectory; 0 disables startup lookahead.",
    )
    parser.add_argument(
        "--future-context-limit",
        type=int,
        default=8,
        help="Number of future raw context events included for each sequential target event; 0 disables general right-side context.",
    )
    parser.add_argument(
        "--history-field-mode",
        choices=["raw", "filled"],
        default="raw",
        help="raw keeps text plus type/agent hints; filled also exposes already accepted filled fields.",
    )
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=6)
    parser.add_argument("--max-windows", type=int, default=1, help="0 means prepare all windows.")
    parser.add_argument("--annotate", action="store_true", help="Call the LLM for prepared requests.")
    parser.add_argument(
        "--llm-event-limit",
        type=int,
        default=32,
        help="Maximum sequential requests to annotate when --annotate is set; 0 means all prepared events.",
    )
    parser.add_argument("--llm-window-limit", type=int, default=1, help="0 means annotate all prepared windows.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args()


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def load_validator(schema_path: Path):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SystemExit("jsonschema is required for validating filled clean events") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def normalize_model_id(model: str) -> str:
    stripped = model.strip()
    if stripped.lower().startswith("deepseek"):
        return stripped.lower()
    return stripped


def output_prefix(input_path: Path) -> str:
    stem = input_path.stem
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._") or "events"


def output_prefix_for_sources(sources: list[tuple[Path, list[dict[str, Any]]]]) -> str:
    if len(sources) == 1:
        return output_prefix(sources[0][0])
    return f"events_batch_{len(sources)}"


def system_prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def sorted_flags(flags: list[Any] | tuple[Any, ...] | set[Any]) -> list[str]:
    return sorted({str(flag) for flag in flags if str(flag)})


def slot(value: Any) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        items: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if stripped and stripped not in seen:
                items.append(stripped)
                seen.add(stripped)
        if not items:
            return None
        return items[0] if len(items) == 1 else items
    return None


def list_slot(value: Any) -> list[str]:
    normalized = slot(value)
    if normalized is None:
        return []
    if isinstance(normalized, str):
        return [normalized]
    return normalized


def coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def consistent_action_type(event_type: str | None, proposed: Any) -> str:
    if event_type == "utterance":
        return "speak"
    if event_type == "action":
        return "action"
    if event_type in {"monologue", "narration", "other", "invalid"}:
        return "other"
    return proposed if proposed in ACTION_TYPES else "other"


def participants_from(agent: Any, target: Any, participants: Any) -> list[str]:
    return sorted_flags([*list_slot(agent), *list_slot(target), *list_slot(participants)])


def flags_from_patch_target(target: Any, patch: dict[str, Any]) -> list[str]:
    flags = list(patch.get("flag") or [])
    if target is None:
        return sorted_flags(flags)
    evidence = patch.get("target_evidence")
    if evidence == "explicit_name_or_vocative":
        return sorted_flags(flags)
    if evidence in TARGET_EVIDENCE and evidence != "none":
        flags.append("target_inferred")
    else:
        flags.extend(["ambiguous_target", "needs_review"])
    return sorted_flags(flags)


def compact_text(text: Any, max_chars: int = 220) -> str | None:
    if not isinstance(text, str):
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def event_sort_key(event: dict[str, Any]) -> tuple[str, int, int]:
    trajectory = str(event.get("trajectory") or "")
    try:
        order = int(event.get("order") or 0)
    except (TypeError, ValueError):
        order = 0
    try:
        event_id = int(event.get("id") or 0)
    except (TypeError, ValueError):
        event_id = 0
    return trajectory, order, event_id


def compact_choice_text(choice: Any) -> dict[str, Any] | None:
    if not isinstance(choice, dict):
        return None
    options = choice.get("options")
    if not isinstance(options, dict):
        return {"options": {}}
    compact_options: dict[str, Any] = {}
    for index, option in options.items():
        if not isinstance(option, dict):
            continue
        compact_option = {"text": compact_text(option.get("text"))}
        for source_key in ["type", "action_type", "agent"]:
            value = option.get(source_key)
            if value is not None:
                compact_option[source_key] = value
        compact_options[str(index)] = compact_option
    return {"options": compact_options}


def compact_filled_fields(event: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "type": event.get("type"),
        "action_type": event.get("action_type"),
        "agent": event.get("agent"),
        "target": event.get("target"),
        "participants": event.get("participants"),
        "valid": event.get("valid"),
        "flag": event.get("flag") or [],
    }
    if event.get("kind") == "choice":
        options: dict[str, Any] = {}
        for index, option in ((event.get("choice") or {}).get("options") or {}).items():
            if not isinstance(option, dict):
                continue
            options[str(index)] = {
                "type": option.get("type"),
                "action_type": option.get("action_type"),
                "agent": option.get("agent"),
                "target": option.get("target"),
                "participants": option.get("participants"),
                "valid": option.get("valid"),
                "flag": option.get("flag") or [],
            }
        fields["choice_options"] = options
    return fields


def compact_history_event(event: dict[str, Any], *, include_filled_fields: bool = False) -> dict[str, Any]:
    compact = {
        "id": event.get("id"),
        "order": event.get("order"),
        "kind": event.get("kind"),
    }
    for source_key in ["type", "action_type", "agent"]:
        value = event.get(source_key)
        if value is not None:
            compact[source_key] = value
    text = compact_text(event.get("text"), max_chars=360)
    if text is not None:
        compact["text"] = text
    choice = compact_choice_text(event.get("choice"))
    if choice is not None:
        compact["choice"] = choice
    if include_filled_fields:
        compact["filled_fields"] = compact_filled_fields(event)
    return compact


def make_history_events(
    events: list[dict[str, Any]],
    current_index: int,
    history_limit: int,
    history_field_mode: str,
) -> list[dict[str, Any]]:
    previous = events[:current_index]
    if history_limit > 0:
        previous = previous[-history_limit:]
    include_filled_fields = history_field_mode == "filled"
    return [compact_history_event(event, include_filled_fields=include_filled_fields) for event in previous]


def make_future_context_events(
    events: list[dict[str, Any]],
    current_index: int,
    startup_lookahead: int,
    future_context_limit: int,
) -> list[dict[str, Any]]:
    context_limit = max(0, future_context_limit)
    if startup_lookahead > 0 and current_index < startup_lookahead:
        context_limit = max(context_limit, startup_lookahead)
    if context_limit <= 0:
        return []
    right = min(len(events), current_index + 1 + context_limit)
    return [compact_history_event(event, include_filled_fields=False) for event in events[current_index + 1 : right]]


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "id": event.get("id"),
        "order": event.get("order"),
        "kind": event.get("kind"),
        "draft_fields": {
            "type": event.get("type"),
            "action_type": event.get("action_type"),
            "agent": event.get("agent"),
            "target": event.get("target"),
            "participants": event.get("participants"),
            "valid": event.get("valid"),
            "flag": event.get("flag") or [],
        },
    }
    text = compact_text(event.get("text"))
    if text is not None:
        compact["text"] = text
    if event.get("kind") == "choice":
        options: dict[str, Any] = {}
        for index, option in ((event.get("choice") or {}).get("options") or {}).items():
            options[str(index)] = {
                "text": compact_text(option.get("text")),
                "draft_fields": {
                    "type": option.get("type"),
                    "action_type": option.get("action_type"),
                    "agent": option.get("agent"),
                    "target": option.get("target"),
                    "participants": option.get("participants"),
                    "valid": option.get("valid"),
                    "flag": option.get("flag") or [],
                },
            }
        compact["choice_options"] = options
    return compact


def make_windows(events: list[dict[str, Any]], target_size: int, context_size: int, max_windows: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for start in range(0, len(events), target_size):
        end = min(len(events), start + target_size)
        left = max(0, start - context_size)
        right = min(len(events), end + context_size)
        target_events = events[start:end]
        window = {
            "window_id": f"window_{len(windows):04d}",
            "target_event_ids": [int(event["id"]) for event in target_events],
            "context_before": [compact_event(event) for event in events[left:start]],
            "target_events": [compact_event(event) for event in target_events],
            "context_after": [compact_event(event) for event in events[end:right]],
        }
        windows.append(window)
        if max_windows and len(windows) >= max_windows:
            break
    return windows


def resolve_event_files(args: argparse.Namespace) -> list[Path]:
    if args.event_files is not None:
        if not args.event_files:
            raise SystemExit("--event-files was provided but no files were listed")
        paths = list(args.event_files)
    elif args.events_dir is not None:
        paths = sorted(args.events_dir.glob("*.jsonl"))
        if not paths:
            raise SystemExit(f"No jsonl files found in --events-dir {args.events_dir}")
    else:
        paths = [args.input]
    if args.max_trajectories > 0:
        paths = paths[: args.max_trajectories]
    return paths


def load_event_sources(args: argparse.Namespace) -> list[tuple[Path, list[dict[str, Any]]]]:
    sources: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in resolve_event_files(args):
        events = sorted(read_jsonl(path), key=event_sort_key)
        sources.append((path, events))
    return sources


def grouped_by_trajectory(events: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for event in events:
        trajectory = str(event.get("trajectory") or "")
        if trajectory not in by_trajectory:
            by_trajectory[trajectory] = []
            order.append(trajectory)
        by_trajectory[trajectory].append(event)
    for trajectory in order:
        groups.append((trajectory, by_trajectory[trajectory]))
    return groups


def make_sequential_fill_request(
    event: dict[str, Any],
    history_events: list[dict[str, Any]],
    future_context_events: list[dict[str, Any]],
) -> dict[str, Any]:
    request = {
        "history": history_events,
        "current_event": compact_event(event),
    }
    if future_context_events:
        request["future_context"] = future_context_events
    return request


def source_scene(event: dict[str, Any]) -> str | None:
    source = event.get("source")
    if not isinstance(source, dict):
        return None
    scene = source.get("scene")
    return str(scene) if scene is not None else None


def make_sequential_request_item(
    path: Path,
    event: dict[str, Any],
    history_events: list[dict[str, Any]],
    future_context_events: list[dict[str, Any]],
    sequence_index: int,
) -> dict[str, Any]:
    return {
        "request_id": f"{output_prefix(path)}::{event.get('trajectory')}::e{int(event['id']):06d}",
        "request_type": "clean_event_fill_sequential",
        "source_file": str(path),
        "trajectory": event.get("trajectory"),
        "scene": source_scene(event),
        "sequence_index": sequence_index,
        "target_event_ids": [int(event["id"])],
        "model": None,
        "request": make_sequential_fill_request(event, history_events, future_context_events),
    }


def make_sequential_requests(
    sources: list[tuple[Path, list[dict[str, Any]]]],
    max_events: int,
    max_events_per_trajectory: int,
    history_limit: int,
    history_field_mode: str,
    startup_lookahead: int,
    future_context_limit: int,
) -> list[dict[str, Any]]:
    requests_out: list[dict[str, Any]] = []
    sequence_index = 0
    for path, events in sources:
        for _trajectory, trajectory_events in grouped_by_trajectory(events):
            for index, event in enumerate(trajectory_events):
                if max_events_per_trajectory > 0 and index >= max_events_per_trajectory:
                    break
                if max_events > 0 and len(requests_out) >= max_events:
                    return requests_out
                history_events = make_history_events(trajectory_events, index, history_limit, history_field_mode)
                future_context_events = make_future_context_events(
                    trajectory_events,
                    index,
                    startup_lookahead,
                    future_context_limit,
                )
                requests_out.append(
                    make_sequential_request_item(
                        path,
                        event,
                        history_events,
                        future_context_events,
                        sequence_index,
                    )
                )
                sequence_index += 1
    return requests_out


def make_fill_request(input_path: Path, window: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Fill mutable Clean Event v0 fields for GalGame LPM data cleaning. Return JSON only.",
        "source_file": str(input_path),
        "window_id": window["window_id"],
        "target_event_ids": window["target_event_ids"],
        "closed_sets": {
            "normal_type": NORMAL_TYPES,
            "choice_option_type": OPTION_TYPES,
            "action_type": ACTION_TYPES,
            "flag": PATCH_FLAGS,
            "target_evidence": TARGET_EVIDENCE,
        },
        "field_definitions": FIELD_DEFINITIONS,
        "required_output": {
            "window_id": window["window_id"],
            **REQUIRED_PATCH_EXAMPLE,
        },
        "rules": FILL_RULES,
        "context_before": window["context_before"],
        "target_events": window["target_events"],
        "context_after": window["context_after"],
    }


def chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_request: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_request, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:1200].replace("\n", " ")
        raise RuntimeError(f"HTTP {response.status_code}: {body}") from exc
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def validate_patch_batch(parsed: dict[str, Any], expected_ids: list[int]) -> dict[str, Any]:
    patches = parsed.get("event_patches")
    if not isinstance(patches, list):
        return {"ok": False, "missing": expected_ids, "extra": [], "duplicate": [], "error": "event_patches_not_list"}
    seen: list[int] = []
    for patch in patches:
        if isinstance(patch, dict) and isinstance(patch.get("id"), int):
            seen.append(int(patch["id"]))
    expected = set(expected_ids)
    seen_set = set(seen)
    duplicate = sorted({event_id for event_id in seen if seen.count(event_id) > 1})
    return {
        "ok": expected == seen_set and not duplicate,
        "missing": sorted(expected - seen_set),
        "extra": sorted(seen_set - expected),
        "duplicate": duplicate,
    }


def apply_normal_patch(event: dict[str, Any], patch: dict[str, Any]) -> None:
    event_type = patch.get("type")
    if event_type in NORMAL_TYPES:
        event["type"] = event_type
    event["action_type"] = consistent_action_type(event.get("type"), patch.get("action_type"))
    agent = slot(patch.get("agent"))
    target = slot(patch.get("target"))
    event["agent"] = agent
    event["target"] = target
    event["participants"] = participants_from(agent, target, patch.get("participants"))
    event["valid"] = coerce_bool(patch.get("valid"), bool(event.get("valid", True)))
    event["flag"] = sorted_flags([*(event.get("flag") or []), *flags_from_patch_target(target, patch)])
    if event["type"] == "monologue" and not event["participants"]:
        event["flag"] = sorted_flags([*(event.get("flag") or []), "ambiguous_actor", "needs_review"])


def apply_choice_patch(event: dict[str, Any], patch: dict[str, Any]) -> None:
    event["type"] = None
    event["action_type"] = "other"
    event["agent"] = None
    event["target"] = None
    event["text"] = None
    choice = event.get("choice") if isinstance(event.get("choice"), dict) else {}
    options = choice.get("options") if isinstance(choice.get("options"), dict) else {}
    patched_options = patch.get("choice_options") if isinstance(patch.get("choice_options"), dict) else {}
    for option_index, option_patch in patched_options.items():
        option = options.get(str(option_index))
        if not isinstance(option, dict) or not isinstance(option_patch, dict):
            continue
        option_type = option_patch.get("type")
        if option_type in OPTION_TYPES:
            option["type"] = option_type
        option["action_type"] = consistent_action_type(option.get("type"), option_patch.get("action_type"))
        agent = slot(option_patch.get("agent"))
        target = slot(option_patch.get("target"))
        option["agent"] = agent
        option["target"] = target
        option["participants"] = participants_from(agent, target, option_patch.get("participants"))
        option["valid"] = coerce_bool(option_patch.get("valid"), bool(option.get("valid", True)))
        option["flag"] = sorted_flags([*(option.get("flag") or []), *flags_from_patch_target(target, option_patch)])
    event["participants"] = sorted_flags(
        participant
        for option in options.values()
        if isinstance(option, dict)
        for participant in option.get("participants", [])
    )
    event["valid"] = coerce_bool(patch.get("valid"), all(bool(option.get("valid", True)) for option in options.values()))
    event["flag"] = sorted_flags([*(event.get("flag") or []), *(patch.get("flag") or [])])


def apply_patch_to_event(event: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    filled = copy.deepcopy(event)
    kind = filled.get("kind")
    if kind == "normal":
        apply_normal_patch(filled, patch)
    elif kind == "choice":
        apply_choice_patch(filled, patch)
    elif kind == "jump":
        filled["valid"] = coerce_bool(patch.get("valid"), bool(filled.get("valid", True)))
        filled["flag"] = sorted_flags([*(filled.get("flag") or []), *(patch.get("flag") or [])])
    elif kind == "invalid":
        filled["type"] = "invalid"
        filled["action_type"] = "other"
        filled["valid"] = False
        filled["flag"] = sorted_flags(patch.get("flag") or filled.get("flag") or [])
    return filled


def validate_rows(rows: list[dict[str, Any]], validator: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        for error in row_errors:
            errors.append({"index": index, "id": row.get("id"), "path": list(error.path), "message": error.message})
    return errors


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "kind_counts": dict(sorted(Counter(str(row.get("kind")) for row in rows).items())),
        "type_counts": dict(sorted(Counter(str(row.get("type")) for row in rows).items())),
        "action_type_counts": dict(sorted(Counter(str(row.get("action_type")) for row in rows).items())),
        "target_non_null_count": sum(1 for row in rows if row.get("target") is not None),
        "invalid_count": sum(1 for row in rows if not row.get("valid")),
        "flag_counts": dict(sorted(Counter(flag for row in rows for flag in row.get("flag", [])).items())),
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def timing_summary(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(
        float(annotation["elapsed_seconds"])
        for annotation in annotations
        if isinstance(annotation.get("elapsed_seconds"), int | float)
    )
    if not values:
        return {"count": 0}
    total = sum(values)
    return {
        "count": len(values),
        "total_seconds": round(total, 6),
        "mean_seconds": round(total / len(values), 6),
        "min_seconds": round(values[0], 6),
        "p50_seconds": round(percentile(values, 0.50), 6),
        "p90_seconds": round(percentile(values, 0.90), 6),
        "max_seconds": round(values[-1], 6),
    }


def resolve_api_key(env: dict[str, str], env_path: Path) -> str:
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("API_KEY")
        or env.get("DEEPSEEK_API_KEY")
        or env.get("API_KEY")
    )
    if not api_key:
        raise SystemExit(f"No API key found in environment or {env_path}")
    return api_key


def main() -> int:
    args = parse_args()
    env = load_env(args.env)
    model = normalize_model_id(args.model or os.environ.get("MODEL") or env.get("MODEL") or DEFAULT_MODEL)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = make_system_prompt()
    prompt_hash = system_prompt_hash(system_prompt)

    sources = load_event_sources(args) if args.mode == "sequential" else [(args.input, read_jsonl(args.input))]
    prefix = output_prefix_for_sources(sources)
    system_prompt_path = args.output_dir / f"{prefix}_system_prompt.txt"
    system_prompt_path.write_text(system_prompt + "\n", encoding="utf-8")

    if args.mode == "window":
        events = sources[0][1]
        windows = make_windows(events, args.target_size, args.context_size, args.max_windows)
        requests_out = [
            {
                "request_id": f"{output_prefix(args.input)}::{window['window_id']}",
                "request_type": "clean_event_fill",
                "target_event_ids": window["target_event_ids"],
                "model": model,
                "system_prompt_sha256": prompt_hash,
                "request": make_fill_request(args.input, window),
            }
            for window in windows
        ]
        prepared_count = len(windows)
    else:
        requests_out = make_sequential_requests(
            sources,
            max_events=args.max_events,
            max_events_per_trajectory=args.max_events_per_trajectory,
            history_limit=args.history_limit,
            history_field_mode=args.history_field_mode,
            startup_lookahead=args.startup_lookahead,
            future_context_limit=args.future_context_limit,
        )
        for item in requests_out:
            item["model"] = model
            item["system_prompt_sha256"] = prompt_hash
        prepared_count = len(requests_out)

    requests_path = args.output_dir / f"{prefix}_requests.jsonl"
    write_jsonl(requests_path, requests_out)

    summary: dict[str, Any] = {
        "mode": args.mode,
        "input": str(args.input),
        "source_files": [str(path) for path, _events in sources],
        "model": model,
        "system_prompt_path": str(system_prompt_path),
        "system_prompt_sha256": prompt_hash,
        "input_event_count": sum(len(events) for _path, events in sources),
        "prepared_request_count": prepared_count,
        "requests_path": str(requests_path),
        "target_size": args.target_size,
        "context_size": args.context_size,
        "max_events": args.max_events,
        "max_events_per_trajectory": args.max_events_per_trajectory,
        "history_limit": args.history_limit,
        "history_field_mode": args.history_field_mode,
        "startup_lookahead": args.startup_lookahead,
        "future_context_limit": args.future_context_limit,
    }

    annotations: list[dict[str, Any]] = []
    filled_rows: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    if args.annotate:
        api_key = resolve_api_key(env, args.env)
        events_by_id = {int(event["id"]): event for _path, events in sources for event in events}
        annotation_started_at = time.perf_counter()

        if args.mode == "window":
            llm_limit = len(requests_out) if args.llm_window_limit == 0 else min(args.llm_window_limit, len(requests_out))
            for item in requests_out[:llm_limit]:
                request = item["request"]
                started_at = time.perf_counter()
                try:
                    parsed = chat_completion(api_key, args.base_url, model, system_prompt, request, args.timeout)
                    elapsed_seconds = time.perf_counter() - started_at
                    validation = validate_patch_batch(parsed, item["target_event_ids"])
                    annotations.append(
                        {
                            "request_id": item["request_id"],
                            "request_type": item["request_type"],
                            "model": model,
                            "elapsed_seconds": round(elapsed_seconds, 6),
                            "annotation": parsed,
                            "validation": validation,
                        }
                    )
                    if validation["ok"]:
                        for patch in parsed.get("event_patches", []):
                            event_id = int(patch["id"])
                            filled_rows.append(apply_patch_to_event(events_by_id[event_id], patch))
                except Exception as exc:
                    elapsed_seconds = time.perf_counter() - started_at
                    annotations.append(
                        {
                            "request_id": item["request_id"],
                            "request_type": item["request_type"],
                            "model": model,
                            "elapsed_seconds": round(elapsed_seconds, 6),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                time.sleep(args.sleep)
        elif args.history_field_mode == "filled":
            requests_out = []
            sequence_index = 0
            llm_limit = None if args.llm_event_limit == 0 else args.llm_event_limit
            for path, source_events in sources:
                working_events = copy.deepcopy(source_events)
                for _trajectory, trajectory_events in grouped_by_trajectory(working_events):
                    for index, event in enumerate(trajectory_events):
                        if args.max_events_per_trajectory > 0 and index >= args.max_events_per_trajectory:
                            break
                        if args.max_events > 0 and sequence_index >= args.max_events:
                            break
                        if llm_limit is not None and len(annotations) >= llm_limit:
                            break
                        history_events = make_history_events(
                            trajectory_events,
                            index,
                            args.history_limit,
                            args.history_field_mode,
                        )
                        future_context_events = make_future_context_events(
                            trajectory_events,
                            index,
                            args.startup_lookahead,
                            args.future_context_limit,
                        )
                        item = make_sequential_request_item(
                            path,
                            event,
                            history_events,
                            future_context_events,
                            sequence_index,
                        )
                        item["model"] = model
                        item["system_prompt_sha256"] = prompt_hash
                        request = item["request"]
                        requests_out.append(item)
                        started_at = time.perf_counter()
                        try:
                            parsed = chat_completion(api_key, args.base_url, model, system_prompt, request, args.timeout)
                            elapsed_seconds = time.perf_counter() - started_at
                            validation = validate_patch_batch(parsed, item["target_event_ids"])
                            annotations.append(
                                {
                                    "request_id": item["request_id"],
                                    "request_type": item["request_type"],
                                    "model": model,
                                    "elapsed_seconds": round(elapsed_seconds, 6),
                                    "annotation": parsed,
                                    "validation": validation,
                                }
                            )
                            if validation["ok"]:
                                patch = parsed.get("event_patches", [])[0]
                                filled = apply_patch_to_event(event, patch)
                                trajectory_events[index] = filled
                                filled_rows.append(filled)
                        except Exception as exc:
                            elapsed_seconds = time.perf_counter() - started_at
                            annotations.append(
                                {
                                    "request_id": item["request_id"],
                                    "request_type": item["request_type"],
                                    "model": model,
                                    "elapsed_seconds": round(elapsed_seconds, 6),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        sequence_index += 1
                        time.sleep(args.sleep)
                    if args.max_events > 0 and sequence_index >= args.max_events:
                        break
                    if llm_limit is not None and len(annotations) >= llm_limit:
                        break
                if args.max_events > 0 and sequence_index >= args.max_events:
                    break
                if llm_limit is not None and len(annotations) >= llm_limit:
                    break
            write_jsonl(requests_path, requests_out)
            summary["prepared_request_count"] = len(requests_out)
        else:
            llm_limit = len(requests_out) if args.llm_event_limit == 0 else min(args.llm_event_limit, len(requests_out))
            for item in requests_out[:llm_limit]:
                request = item["request"]
                started_at = time.perf_counter()
                try:
                    parsed = chat_completion(api_key, args.base_url, model, system_prompt, request, args.timeout)
                    elapsed_seconds = time.perf_counter() - started_at
                    validation = validate_patch_batch(parsed, item["target_event_ids"])
                    annotations.append(
                        {
                            "request_id": item["request_id"],
                            "request_type": item["request_type"],
                            "model": model,
                            "elapsed_seconds": round(elapsed_seconds, 6),
                            "annotation": parsed,
                            "validation": validation,
                        }
                    )
                    if validation["ok"]:
                        patch = parsed.get("event_patches", [])[0]
                        event_id = int(patch["id"])
                        filled_rows.append(apply_patch_to_event(events_by_id[event_id], patch))
                except Exception as exc:
                    elapsed_seconds = time.perf_counter() - started_at
                    annotations.append(
                        {
                            "request_id": item["request_id"],
                            "request_type": item["request_type"],
                            "model": model,
                            "elapsed_seconds": round(elapsed_seconds, 6),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                time.sleep(args.sleep)

        annotation_wall_seconds = time.perf_counter() - annotation_started_at
        annotations_path = args.output_dir / f"{prefix}_annotations.jsonl"
        write_jsonl(annotations_path, annotations)
        filled_rows.sort(key=event_sort_key)
        filled_path = args.output_dir / f"{prefix}_filled.jsonl"
        write_jsonl(filled_path, filled_rows)

        validator = load_validator(args.schema)
        validation_errors = validate_rows(filled_rows, validator)
        summary.update(
            {
                "annotations_path": str(annotations_path),
                "filled_path": str(filled_path),
                "annotated_request_count": len(annotations),
                "annotation_error_count": sum(1 for annotation in annotations if "error" in annotation),
                "annotation_timing": timing_summary(annotations),
                "annotation_wall_seconds": round(annotation_wall_seconds, 6),
                "filled_event_count": len(filled_rows),
                "validation_error_count": len(validation_errors),
                "input_summary_for_filled_ids": summarize_rows(
                    [events_by_id[int(row["id"])] for row in filled_rows if int(row["id"]) in events_by_id]
                ),
                "filled_summary": summarize_rows(filled_rows),
            }
        )
        if validation_errors:
            errors_path = args.output_dir / f"{prefix}_validation_errors.json"
            errors_path.write_text(json.dumps(validation_errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary["validation_errors_path"] = str(errors_path)

    summary_path = args.output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {summary['prepared_request_count']} clean-event fill requests in {args.mode} mode")
    print(f"wrote requests to {requests_path}")
    print(f"wrote system prompt to {system_prompt_path}")
    if args.annotate:
        print(f"wrote {len(annotations)} LLM annotation batches")
        print(f"wrote {len(filled_rows)} filled rows")
        if validation_errors:
            print(f"validation errors: {len(validation_errors)}")
    print(f"wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
