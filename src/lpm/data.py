from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Event:
    candidate_id: str
    scene: str
    pos: int
    kind: str
    speaker: str | None
    text: str

    @classmethod
    def from_candidate_event(cls, candidate_id: str, raw: dict[str, Any]) -> "Event":
        return cls(
            candidate_id=candidate_id,
            scene=str(raw.get("scene") or ""),
            pos=int(raw.get("pos") or 0),
            kind=str(raw.get("kind") or "unknown"),
            speaker=raw.get("speaker"),
            text=str(raw.get("text") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "scene": self.scene,
            "pos": self.pos,
            "kind": self.kind,
            "speaker": self.speaker,
            "text": self.text,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def event_text(event: dict[str, Any]) -> str:
    speaker = event.get("speaker") or "NARRATION"
    kind = event.get("kind") or "unknown"
    text = event.get("text") or ""
    return f"[{kind}|{speaker}] {text}"


def make_context_text(events: list[dict[str, Any]]) -> str:
    return "\n".join(event_text(event) for event in events if event.get("text"))


def build_response_units(
    candidates: list[dict[str, Any]],
    max_context_events: int = 16,
    min_context_events: int = 1,
    min_response_chars: int = 1,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        raw_events = candidate.get("events") or []
        events = [Event.from_candidate_event(candidate_id, raw).to_json() for raw in raw_events]
        for index, event in enumerate(events):
            speaker = event.get("speaker")
            text = str(event.get("text") or "")
            if event.get("kind") != "dialogue" or not speaker or len(text) < min_response_chars:
                continue
            left = events[max(0, index - max_context_events) : index]
            if len(left) < min_context_events:
                continue
            char_left = [item for item in left if item.get("speaker") == speaker]
            unit_id = f"{candidate_id}::pos{event['pos']}::{speaker}"
            units.append(
                {
                    "unit_id": unit_id,
                    "candidate_id": candidate_id,
                    "candidate_type": candidate.get("candidate_type"),
                    "scene": candidate.get("scene"),
                    "target_character": speaker,
                    "response_pos": event["pos"],
                    "response_text": text,
                    "response_event": event,
                    "context_events": left,
                    "character_context_events": char_left,
                    "global_context_text": make_context_text(left),
                    "character_context_text": make_context_text(char_left),
                }
            )
    return units


def build_pairwise_negatives(
    units: list[dict[str, Any]],
    max_negatives_per_unit: int = 4,
    seed: int = 7,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    by_character: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        by_candidate.setdefault(unit["candidate_id"], []).append(unit)
        by_character.setdefault(unit["target_character"], []).append(unit)

    pairs: list[dict[str, Any]] = []
    all_units = list(units)
    for unit in units:
        choices: list[tuple[str, dict[str, Any]]] = []

        same_candidate_other = [
            item
            for item in by_candidate.get(unit["candidate_id"], [])
            if item["unit_id"] != unit["unit_id"] and item["target_character"] != unit["target_character"]
        ]
        choices.extend(("same_window_other_character", item) for item in same_candidate_other)

        same_character_temporal = [
            item
            for item in by_character.get(unit["target_character"], [])
            if item["unit_id"] != unit["unit_id"] and item["candidate_id"] != unit["candidate_id"]
        ]
        choices.extend(("same_character_temporal_mismatch", item) for item in same_character_temporal)

        other_character = [
            item
            for item in all_units
            if item["unit_id"] != unit["unit_id"] and item["target_character"] != unit["target_character"]
        ]
        if other_character:
            choices.extend(("random_other_character", item) for item in rng.sample(other_character, k=min(8, len(other_character))))

        rng.shuffle(choices)
        seen_negatives: set[str] = set()
        used = 0
        for negative_type, negative in choices:
            if negative["unit_id"] in seen_negatives:
                continue
            seen_negatives.add(negative["unit_id"])
            pairs.append(
                {
                    "pair_id": f"pair_{len(pairs):06d}",
                    "unit_id": unit["unit_id"],
                    "positive_unit_id": unit["unit_id"],
                    "negative_unit_id": negative["unit_id"],
                    "negative_type": negative_type,
                    "target_character": unit["target_character"],
                    "candidate_id": unit["candidate_id"],
                }
            )
            used += 1
            if used >= max_negatives_per_unit:
                break
    return pairs


def summarize_units(units: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    characters: dict[str, int] = {}
    negative_types: dict[str, int] = {}
    for unit in units:
        characters[unit["target_character"]] = characters.get(unit["target_character"], 0) + 1
    for pair in pairs:
        key = pair["negative_type"]
        negative_types[key] = negative_types.get(key, 0) + 1
    return {
        "unit_count": len(units),
        "pair_count": len(pairs),
        "character_count": len(characters),
        "characters": dict(sorted(characters.items(), key=lambda item: (-item[1], item[0]))[:32]),
        "negative_types": dict(sorted(negative_types.items())),
    }
