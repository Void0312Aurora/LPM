from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SYSTEM_SCENE_RE = re.compile(r"(デバッグ|ボイスチェック|CS用処理|シナリオフロー|^_|_var|_task|_scr|_build|マップ選択)")
TEXT_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u061f": "?",
    }
)


def normalize_text_punctuation(value: str) -> str:
    return value.translate(TEXT_PUNCTUATION_TRANSLATION)


def clean_speaker(value: Any) -> tuple[str | None, list[str]]:
    flags: list[str] = []
    if not isinstance(value, str) or not value.strip():
        return None, flags
    stripped = value.strip()
    normalized = normalize_text_punctuation(stripped)
    if normalized != stripped:
        flags.append("speaker_punctuation_normalized")
    return normalized or None, flags


def clean_text(value: Any) -> tuple[str | None, list[str]]:
    flags: list[str] = []
    if not isinstance(value, str):
        return None, flags
    stripped = value.strip()
    cleaned = CONTROL_CHARS.sub("", stripped)
    if cleaned != stripped:
        flags.append("control_char_removed")
    normalized = normalize_text_punctuation(cleaned)
    if normalized != cleaned:
        flags.append("text_punctuation_normalized")
    cleaned = normalized
    if not cleaned:
        flags.append("empty_text")
        return None, flags
    return cleaned, flags


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


def event_id(scene: str, source_path: str) -> str:
    return f"tree_dialogues_v2::{scene}::{source_path}"


def is_system_scene(scene: str | None) -> bool:
    return bool(scene and SYSTEM_SCENE_RE.search(scene))


def is_bad_jump_target(target: Any) -> bool:
    if not isinstance(target, str) or not target:
        return True
    if any(ord(ch) < 32 for ch in target):
        return True
    return any(token in target for token in ("==", "(", ")", "$"))


def jump_status(
    source_scene: str,
    target_scene: Any,
    branch_index: Any,
    dataset_scene_ids: set[str],
    raw_scene_ids: set[str],
) -> tuple[str, list[str]]:
    flags: list[str] = []
    target = target_scene if isinstance(target_scene, str) else None
    branch_ok = isinstance(branch_index, int) and 0 <= branch_index <= 1000
    target_in_dataset = target in dataset_scene_ids
    target_in_raw = target in raw_scene_ids

    if not branch_ok:
        flags.append("suspicious_branch_index")
    if is_system_scene(source_scene):
        flags.append("debug_or_system_scene")
        return "system_source", flags
    if is_bad_jump_target(target):
        flags.append("parse_garbage_target")
        return "parse_garbage", flags
    if is_system_scene(target):
        flags.append("system_target")
        return "system_target", flags
    if target_in_dataset and branch_ok:
        return "candidate_narrative", flags
    if target_in_raw:
        flags.append("target_only_in_raw_scene_dir")
        return "raw_only_target", flags
    flags.append("target_missing")
    return "missing_target", flags


def build_canonical_events(input_path: Path, raw_scene_dir: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenes = read_jsonl(input_path)
    dataset_scene_ids = {str(scene.get("scene")) for scene in scenes if scene.get("scene")}
    raw_scene_ids: set[str] = set()
    if raw_scene_dir and raw_scene_dir.exists():
        raw_scene_ids = {path.name for path in raw_scene_dir.iterdir() if path.is_file()}

    speaker_counts: Counter[str] = Counter()
    text_index_counts: dict[str, Counter[int]] = {}
    for scene_obj in scenes:
        scene = str(scene_obj.get("scene") or "")
        counter: Counter[int] = Counter()
        for item in scene_obj.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                speaker, _speaker_flags = clean_speaker(item.get("speaker"))
                if speaker:
                    speaker_counts[speaker] += 1
                index = item.get("index")
                if isinstance(index, int):
                    counter[index] += 1
        text_index_counts[scene] = counter

    events: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    previous_text_index: dict[str, int] = {}
    for source_line, scene_obj in enumerate(scenes, start=1):
        scene = str(scene_obj.get("scene") or "")
        scene_flags = ["debug_or_system_scene"] if is_system_scene(scene) else []
        content = scene_obj.get("content") or []
        for local_pos, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("type") or "unknown")
            source_path = f"content[{local_pos}]"
            flags = list(scene_flags)
            text_index = item.get("index") if isinstance(item.get("index"), int) else None
            if text_index is not None:
                if text_index_counts.get(scene, Counter()).get(text_index, 0) > 1:
                    flags.append("duplicate_text_index")
                prev = previous_text_index.get(scene)
                if prev is not None and text_index < prev:
                    flags.append("index_order_regression")
                previous_text_index[scene] = text_index

            if raw_type == "text":
                text_norm, text_flags = clean_text(item.get("text"))
                flags.extend(text_flags)
                speaker_raw = item.get("speaker")
                speaker_norm, speaker_flags = clean_speaker(speaker_raw)
                flags.extend(speaker_flags)
                if speaker_norm and speaker_counts[speaker_norm] <= 2:
                    flags.append("low_frequency_speaker")
                observation_channel = "direct_dialogue" if speaker_norm else "narration"
                event = {
                    "event_id": event_id(scene, source_path),
                    "source_dataset": "tree_dialogues_v2",
                    "source_file": str(input_path),
                    "source_line": source_line,
                    "source_path": source_path,
                    "trajectory_id": scene,
                    "scene": scene,
                    "local_pos": local_pos,
                    "text_index": text_index,
                    "raw_type": raw_type,
                    "event_kind": "text",
                    "observation_channel": observation_channel,
                    "speaker_raw": speaker_raw,
                    "speaker_norm": speaker_norm,
                    "text_raw": item.get("text"),
                    "text_norm": text_norm,
                    "choice": None,
                    "jump": None,
                    "flags": sorted(set(flags)),
                }
                events.append(event)
                stats[observation_channel] += 1
                continue

            if raw_type == "choice":
                branches: list[dict[str, Any]] = []
                for branch in item.get("branches") or []:
                    if not isinstance(branch, dict):
                        continue
                    option_text, option_flags = clean_text(branch.get("option"))
                    if option_flags:
                        flags.extend(f"choice_{flag}" for flag in option_flags)
                    speaker_norm, speaker_flags = clean_speaker(branch.get("speaker"))
                    if speaker_flags:
                        flags.extend(f"choice_{flag}" for flag in speaker_flags)
                    leads_to = branch.get("leads_to") if isinstance(branch.get("leads_to"), dict) else None
                    if leads_to is None:
                        flags.append("branch_missing_target")
                    branches.append(
                        {
                            "option_value": branch.get("option_value"),
                            "speaker": speaker_norm,
                            "option": option_text,
                            "has_leads_to": leads_to is not None,
                            "leads_to": leads_to,
                        }
                    )
                if not branches:
                    flags.append("degenerate_choice")
                text_norm = " / ".join(branch["option"] or "" for branch in branches).strip() or None
                event = {
                    "event_id": event_id(scene, source_path),
                    "source_dataset": "tree_dialogues_v2",
                    "source_file": str(input_path),
                    "source_line": source_line,
                    "source_path": source_path,
                    "trajectory_id": scene,
                    "scene": scene,
                    "local_pos": local_pos,
                    "text_index": None,
                    "raw_type": raw_type,
                    "event_kind": "choice",
                    "observation_channel": "choice",
                    "speaker_raw": None,
                    "speaker_norm": None,
                    "text_raw": None,
                    "text_norm": text_norm,
                    "choice": {"var_id": item.get("var_id"), "branches": branches},
                    "jump": None,
                    "flags": sorted(set(flags)),
                }
                events.append(event)
                stats["choice"] += 1

        jump_base_pos = len(content)
        for jump_pos, jump in enumerate(scene_obj.get("scene_jumps") or []):
            if not isinstance(jump, dict):
                continue
            source_path = f"scene_jumps[{jump_pos}]"
            target_scene = jump.get("target")
            branch_index = jump.get("branch_index")
            status, flags = jump_status(scene, target_scene, branch_index, dataset_scene_ids, raw_scene_ids)
            flags.extend(scene_flags)
            target = target_scene if isinstance(target_scene, str) else None
            event = {
                "event_id": event_id(scene, source_path),
                "source_dataset": "tree_dialogues_v2",
                "source_file": str(input_path),
                "source_line": source_line,
                "source_path": source_path,
                "trajectory_id": scene,
                "scene": scene,
                "local_pos": jump_base_pos + jump_pos,
                "text_index": None,
                "raw_type": "scene_jump",
                "event_kind": "jump",
                "observation_channel": "scene_jump",
                "speaker_raw": None,
                "speaker_norm": None,
                "text_raw": None,
                "text_norm": None,
                "choice": None,
                "jump": {
                    "target_scene": target_scene,
                    "branch_index": branch_index,
                    "target_in_dataset": target in dataset_scene_ids,
                    "target_in_raw_scene_dir": target in raw_scene_ids,
                    "status": status,
                },
                "flags": sorted(set(flags)),
            }
            events.append(event)
            stats[f"jump:{status}"] += 1

    report = {
        "source_file": str(input_path),
        "scene_count": len(scenes),
        "event_count": len(events),
        "text_event_count": stats.get("direct_dialogue", 0) + stats.get("narration", 0),
        "observation_channel_counts": {
            "direct_dialogue": stats.get("direct_dialogue", 0),
            "narration": stats.get("narration", 0),
            "choice": stats.get("choice", 0),
            "scene_jump": sum(value for key, value in stats.items() if key.startswith("jump:")),
        },
        "jump_status_counts": {
            key.removeprefix("jump:"): value for key, value in sorted(stats.items()) if key.startswith("jump:")
        },
        "speaker_count": len(speaker_counts),
        "low_frequency_speaker_count": sum(1 for count in speaker_counts.values() if count <= 2),
        "flag_counts": dict(sorted(Counter(flag for event in events for flag in event["flags"]).items())),
    }
    return events, report
