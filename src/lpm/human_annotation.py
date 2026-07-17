from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import threading
import uuid
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


NORMAL_TYPES = ("utterance", "action", "monologue", "narration", "other")
OPTION_TYPES = (*NORMAL_TYPES, "invalid")
ANNOTATION_STATUSES = ("accepted", "needs_review", "skipped")
FLAG_SUGGESTIONS = (
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
    "source_alignment_checked",
)


class AnnotationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__("; ".join(str(error.get("message")) for error in errors))


class RevisionConflict(RuntimeError):
    def __init__(self, current_revision: int) -> None:
        self.current_revision = current_revision
        super().__init__(f"annotation revision changed to {current_revision}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temp, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temp, path)


def normalize_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace("，", ",").replace("、", ",").replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def normalize_slot(value: Any) -> str | list[str] | None:
    values = normalize_strings(value)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def slot_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return normalize_strings(value)
    return normalize_strings(value)


def sorted_unique(values: Iterable[Any]) -> list[str]:
    result: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            result.update(normalize_strings(value))
        elif value is not None:
            text = str(value).strip()
            if text:
                result.add(text)
    return sorted(result)


def action_type_for(event_type: str | None) -> str:
    if event_type == "utterance":
        return "speak"
    if event_type == "action":
        return "action"
    return "other"


def event_source_key(event: dict[str, Any]) -> tuple[str, str]:
    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    return str(source.get("scene") or ""), str(source.get("path") or "")


class EvidenceIndex:
    def __init__(self, canonical_events: Path | None = None, aligned_dialogues: Path | None = None) -> None:
        self.canonical_by_source: dict[tuple[str, str], dict[str, Any]] = {}
        self.aligned_by_scene_index: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.character_counts: Counter[str] = Counter()
        self.canonical_row_count = 0
        self.aligned_row_count = 0
        if canonical_events is not None and canonical_events.exists():
            self._load_canonical(canonical_events)
        if aligned_dialogues is not None and aligned_dialogues.exists():
            self._load_aligned(aligned_dialogues)

    def _load_canonical(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid canonical JSONL at {path}:{line_number}: {exc}") from exc
                scene = str(row.get("scene") or "")
                source_path = str(row.get("source_path") or "")
                if not scene or not source_path:
                    continue
                speaker = row.get("speaker_norm")
                if isinstance(speaker, str) and speaker.strip():
                    self.character_counts[speaker.strip()] += 1
                self.canonical_by_source[(scene, source_path)] = {
                    "event_id": row.get("event_id"),
                    "source_file": row.get("source_file"),
                    "source_line": row.get("source_line"),
                    "source_path": source_path,
                    "scene": scene,
                    "local_pos": row.get("local_pos"),
                    "text_index": row.get("text_index"),
                    "raw_type": row.get("raw_type"),
                    "event_kind": row.get("event_kind"),
                    "observation_channel": row.get("observation_channel"),
                    "speaker_raw": row.get("speaker_raw"),
                    "speaker_norm": row.get("speaker_norm"),
                    "text_raw": row.get("text_raw"),
                    "text_norm": row.get("text_norm"),
                    "flags": row.get("flags") or [],
                }
                self.canonical_row_count += 1

    def _load_aligned(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid aligned JSONL at {path}:{line_number}: {exc}") from exc
                scene = row.get("scene_file")
                text_index = row.get("text_index")
                if not isinstance(scene, str) or not isinstance(text_index, int):
                    continue
                compact = {
                    "scene_file": scene,
                    "text_index": text_index,
                    "speaker": row.get("speaker"),
                    "text": row.get("text"),
                    "text_type": row.get("text_type"),
                    "zh_speaker": row.get("zh_speaker"),
                    "zh_text": row.get("zh_text"),
                    "en_text": row.get("en_text"),
                }
                self.aligned_by_scene_index.setdefault((scene, text_index), []).append(compact)
                self.aligned_row_count += 1

    def for_event(self, event: dict[str, Any]) -> dict[str, Any]:
        canonical = self.canonical_by_source.get(event_source_key(event))
        aligned: list[dict[str, Any]] = []
        if canonical is not None and isinstance(canonical.get("text_index"), int):
            aligned = self.aligned_by_scene_index.get(
                (str(canonical.get("scene") or ""), int(canonical["text_index"])),
                [],
            )
        speaker = canonical.get("speaker_norm") if canonical else None
        direct_dialogue = bool(
            canonical
            and (
                canonical.get("observation_channel") == "direct_dialogue"
                or (isinstance(speaker, str) and speaker.strip())
            )
        )
        locked_fields: dict[str, Any] = {}
        if event.get("kind") == "normal" and direct_dialogue and isinstance(speaker, str) and speaker.strip():
            locked_fields = {
                "type": "utterance",
                "action_type": "speak",
                "agent": speaker.strip(),
            }
        return {
            "canonical": canonical,
            "aligned": aligned,
            "locked_fields": locked_fields,
            "speaker_locked": bool(locked_fields),
        }

    def characters(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "count": count}
            for name, count in sorted(self.character_counts.items(), key=lambda item: (-item[1], item[0]))
        ]


class EventRepository:
    def __init__(
        self,
        graph_path: Path,
        events_dir: Path,
        evidence: EvidenceIndex | None = None,
        cache_size: int = 8,
    ) -> None:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes = graph.get("nodes")
        if not isinstance(nodes, dict):
            raise ValueError(f"graph nodes must be an object: {graph_path}")
        self.graph_path = graph_path
        self.events_dir = events_dir
        self.evidence = evidence or EvidenceIndex()
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, tuple[list[dict[str, Any]], dict[int, int]]] = OrderedDict()
        self.trajectories: list[dict[str, Any]] = []
        self.trajectory_by_id: dict[str, dict[str, Any]] = {}
        for trajectory_id, node in sorted(nodes.items(), key=lambda item: int(item[1].get("entry", 0))):
            meta = {
                "trajectory_id": str(trajectory_id),
                "scene": node.get("scene"),
                "file": node.get("file"),
                "count": int(node.get("count") or 0),
                "entry": node.get("entry"),
                "valid": bool(node.get("valid", True)),
                "flag": node.get("flag") or [],
            }
            self.trajectories.append(meta)
            self.trajectory_by_id[str(trajectory_id)] = meta

    def _event_path(self, trajectory_id: str) -> Path:
        meta = self.trajectory_by_id.get(trajectory_id)
        if meta is None:
            raise KeyError(f"unknown trajectory: {trajectory_id}")
        filename = Path(str(meta.get("file") or f"{trajectory_id}.jsonl")).name
        return self.events_dir / filename

    def load_trajectory(self, trajectory_id: str) -> tuple[list[dict[str, Any]], dict[int, int]]:
        cached = self._cache.get(trajectory_id)
        if cached is not None:
            self._cache.move_to_end(trajectory_id)
            return cached
        path = self._event_path(trajectory_id)
        events = sorted(read_jsonl(path), key=lambda event: (int(event.get("order") or 0), int(event.get("id") or 0)))
        id_to_index = {int(event["id"]): index for index, event in enumerate(events)}
        value = (events, id_to_index)
        self._cache[trajectory_id] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def get_event(self, trajectory_id: str, event_id: int | None = None) -> tuple[dict[str, Any], int, int]:
        events, id_to_index = self.load_trajectory(trajectory_id)
        if not events:
            raise KeyError(f"trajectory has no events: {trajectory_id}")
        if event_id is None:
            index = 0
        else:
            if event_id not in id_to_index:
                raise KeyError(f"unknown event {trajectory_id}:{event_id}")
            index = id_to_index[event_id]
        return events[index], index, len(events)

    def context(self, trajectory_id: str, event_id: int, radius: int) -> list[dict[str, Any]]:
        events, id_to_index = self.load_trajectory(trajectory_id)
        index = id_to_index[event_id]
        radius = max(0, radius)
        window_size = min(len(events), radius * 2 + 1)
        left = max(0, min(index - radius, len(events) - window_size))
        right = min(len(events), left + window_size)
        return events[left:right]

    def neighbor_id(self, trajectory_id: str, event_id: int, direction: int) -> int | None:
        events, id_to_index = self.load_trajectory(trajectory_id)
        index = id_to_index[event_id] + direction
        if index < 0 or index >= len(events):
            return None
        return int(events[index]["id"])

    def next_matching_id(
        self,
        trajectory_id: str,
        event_id: int,
        predicate: Any,
        direction: int = 1,
    ) -> int | None:
        events, id_to_index = self.load_trajectory(trajectory_id)
        index = id_to_index[event_id] + direction
        while 0 <= index < len(events):
            candidate = events[index]
            if predicate(candidate):
                return int(candidate["id"])
            index += direction
        return None


def default_patch(event: dict[str, Any], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence = evidence or {}
    locked = evidence.get("locked_fields") if isinstance(evidence.get("locked_fields"), dict) else {}
    kind = event.get("kind")
    if kind == "normal":
        event_type = locked.get("type") or event.get("type")
        agent = locked.get("agent") if "agent" in locked else event.get("agent")
        return {
            "type": event_type,
            "action_type": action_type_for(str(event_type) if event_type is not None else None),
            "agent": copy.deepcopy(agent),
            "target": copy.deepcopy(event.get("target")),
            "participants": copy.deepcopy(event.get("participants") or []),
            "valid": bool(event.get("valid", True)),
            "flag": copy.deepcopy(event.get("flag") or []),
        }
    if kind == "choice":
        options: dict[str, Any] = {}
        for index, option in ((event.get("choice") or {}).get("options") or {}).items():
            options[str(index)] = {
                "type": option.get("type"),
                "action_type": action_type_for(option.get("type")),
                "agent": copy.deepcopy(option.get("agent")),
                "target": copy.deepcopy(option.get("target")),
                "participants": copy.deepcopy(option.get("participants") or []),
                "valid": bool(option.get("valid", True)),
                "flag": copy.deepcopy(option.get("flag") or []),
            }
        return {
            "participants": copy.deepcopy(event.get("participants") or []),
            "valid": bool(event.get("valid", True)),
            "flag": copy.deepcopy(event.get("flag") or []),
            "choice_options": options,
        }
    return {
        "valid": bool(event.get("valid", kind != "invalid")),
        "flag": copy.deepcopy(event.get("flag") or []),
    }


def _normalize_option_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    event_type = str(patch.get("type") or base.get("type") or "other")
    if event_type not in OPTION_TYPES:
        event_type = str(base.get("type") or "other")
    agent = normalize_slot(patch.get("agent", base.get("agent")))
    target = normalize_slot(patch.get("target", base.get("target")))
    participants = sorted_unique(
        [
            *normalize_strings(base.get("participants")),
            *normalize_strings(patch.get("participants")),
            *slot_values(agent),
            *slot_values(target),
        ]
    )
    if event_type == "monologue":
        target = None
        participants = sorted_unique([*participants, *slot_values(agent)])
    return {
        "type": event_type,
        "action_type": action_type_for(event_type),
        "agent": agent,
        "target": target,
        "participants": participants,
        "valid": bool(patch.get("valid", base.get("valid", True))),
        "flag": sorted_unique([*normalize_strings(base.get("flag")), *normalize_strings(patch.get("flag"))]),
    }


def normalize_patch(
    event: dict[str, Any],
    patch: dict[str, Any] | None,
    evidence: dict[str, Any] | None = None,
    status: str = "accepted",
) -> dict[str, Any]:
    patch = patch if isinstance(patch, dict) else {}
    evidence = evidence or {}
    base = default_patch(event, evidence)
    kind = event.get("kind")
    if kind == "normal":
        locked = evidence.get("locked_fields") if isinstance(evidence.get("locked_fields"), dict) else {}
        event_type = str(locked.get("type") or patch.get("type") or base.get("type") or "other")
        if event_type not in NORMAL_TYPES:
            event_type = str(base.get("type") or "other")
        agent = normalize_slot(locked.get("agent") if "agent" in locked else patch.get("agent", base.get("agent")))
        target = normalize_slot(patch.get("target", base.get("target")))
        if event_type == "monologue":
            target = None
        participants = sorted_unique(
            [
                *normalize_strings(base.get("participants")),
                *normalize_strings(patch.get("participants")),
                *slot_values(agent),
                *slot_values(target),
            ]
        )
        flags = sorted_unique([*normalize_strings(base.get("flag")), *normalize_strings(patch.get("flag"))])
        if status == "needs_review":
            flags = sorted_unique([*flags, "needs_review"])
        return {
            "type": event_type,
            "action_type": action_type_for(event_type),
            "agent": agent,
            "target": target,
            "participants": participants,
            "valid": bool(patch.get("valid", base.get("valid", True))),
            "flag": flags,
        }
    if kind == "choice":
        base_options = base.get("choice_options") or {}
        patch_options = patch.get("choice_options") if isinstance(patch.get("choice_options"), dict) else {}
        options = {
            str(index): _normalize_option_patch(option_base, patch_options.get(str(index), {}))
            for index, option_base in base_options.items()
        }
        participants = sorted_unique(
            participant
            for option in options.values()
            for participant in option.get("participants", [])
        )
        flags = sorted_unique([*normalize_strings(base.get("flag")), *normalize_strings(patch.get("flag"))])
        if status == "needs_review":
            flags = sorted_unique([*flags, "needs_review"])
        return {
            "participants": participants,
            "valid": bool(patch.get("valid", all(option.get("valid", True) for option in options.values()))),
            "flag": flags,
            "choice_options": options,
        }
    flags = sorted_unique([*normalize_strings(base.get("flag")), *normalize_strings(patch.get("flag"))])
    if status == "needs_review":
        flags = sorted_unique([*flags, "needs_review"])
    return {
        "valid": False if kind == "invalid" else bool(patch.get("valid", base.get("valid", True))),
        "flag": flags,
    }


def apply_patch(event: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    filled = copy.deepcopy(event)
    kind = filled.get("kind")
    if kind == "normal":
        for field in ("type", "action_type", "agent", "target", "participants", "valid", "flag"):
            filled[field] = copy.deepcopy(patch[field])
    elif kind == "choice":
        filled["type"] = None
        filled["action_type"] = "other"
        filled["agent"] = None
        filled["target"] = None
        filled["text"] = None
        options = ((filled.get("choice") or {}).get("options") or {})
        for index, option_patch in (patch.get("choice_options") or {}).items():
            option = options.get(str(index))
            if not isinstance(option, dict):
                continue
            for field in ("type", "action_type", "agent", "target", "participants", "valid", "flag"):
                option[field] = copy.deepcopy(option_patch[field])
        filled["participants"] = copy.deepcopy(patch.get("participants") or [])
        filled["valid"] = bool(patch.get("valid", True))
        filled["flag"] = copy.deepcopy(patch.get("flag") or [])
    elif kind == "jump":
        filled["valid"] = bool(patch.get("valid", filled.get("valid", True)))
        filled["flag"] = copy.deepcopy(patch.get("flag") or [])
    elif kind == "invalid":
        filled["type"] = "invalid"
        filled["action_type"] = "other"
        filled["valid"] = False
        filled["flag"] = copy.deepcopy(patch.get("flag") or [])
    return filled


def validate_annotation(
    event: dict[str, Any],
    patch: dict[str, Any],
    status: str,
    note: str,
    validator: Any | None = None,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if status not in ANNOTATION_STATUSES:
        errors.append({"path": ["status"], "message": f"unsupported annotation status: {status}"})
    kind = event.get("kind")
    if status == "needs_review" and not note.strip():
        errors.append({"path": ["note"], "message": "needs_review requires a short note"})
    if status == "accepted" and kind == "normal":
        event_type = patch.get("type")
        if event_type in {"utterance", "action", "monologue"} and not slot_values(patch.get("agent")):
            errors.append({"path": ["patch", "agent"], "message": f"{event_type} requires an agent"})
        if event_type == "monologue" and patch.get("target") is not None:
            errors.append({"path": ["patch", "target"], "message": "monologue target must be null"})
    if status == "accepted" and not patch.get("valid", True) and not patch.get("flag"):
        errors.append({"path": ["patch", "flag"], "message": "invalid accepted rows require at least one flag"})
    filled = apply_patch(event, patch)
    if validator is not None:
        for error in sorted(validator.iter_errors(filled), key=lambda item: list(item.path)):
            errors.append({
                "path": ["filled_event", *list(error.path)],
                "message": error.message,
            })
    return errors


class AnnotationStore:
    def __init__(self, history_path: Path) -> None:
        self.history_path = history_path
        self._lock = threading.RLock()
        self.latest: dict[tuple[str, int], dict[str, Any]] = {}
        self.load_errors: list[str] = []
        self.record_count = 0
        self._load()

    def _load(self) -> None:
        if not self.history_path.exists():
            return
        with self.history_path.open("r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    trajectory = str(record["trajectory"])
                    event_id = int(record["event_id"])
                    revision = int(record["revision"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    self.load_errors.append(f"{self.history_path}:{line_number}: {exc}")
                    continue
                key = (trajectory, event_id)
                current = self.latest.get(key)
                if current is None or revision >= int(current.get("revision") or 0):
                    self.latest[key] = record
                self.record_count += 1

    def get(self, trajectory: str, event_id: int) -> dict[str, Any] | None:
        record = self.latest.get((trajectory, int(event_id)))
        return copy.deepcopy(record) if record is not None else None

    def append(
        self,
        *,
        trajectory: str,
        event: dict[str, Any],
        patch: dict[str, Any],
        status: str,
        annotator: str,
        note: str,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        key = (trajectory, int(event["id"]))
        with self._lock:
            current = self.latest.get(key)
            current_revision = int(current.get("revision") or 0) if current else 0
            if expected_revision is not None and int(expected_revision) != current_revision:
                raise RevisionConflict(current_revision)
            revision = current_revision + 1
            record = {
                "record_version": "human_annotation_v0",
                "annotation_id": str(uuid.uuid4()),
                "trajectory": trajectory,
                "event_id": int(event["id"]),
                "source": copy.deepcopy(event.get("source") or {}),
                "base_sha256": json_sha256(event),
                "revision": revision,
                "status": status,
                "annotator": annotator.strip() or "anonymous",
                "note": note.strip(),
                "patch": copy.deepcopy(patch),
                "filled_sha256": json_sha256(apply_patch(event, patch)),
                "created_at": utc_now(),
            }
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as fp:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
                fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                fp.flush()
                os.fsync(fp.fileno())
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            self.latest[key] = record
            self.record_count += 1
            return copy.deepcopy(record)

    def status_for(self, trajectory: str, event_id: int) -> str | None:
        record = self.latest.get((trajectory, int(event_id)))
        return str(record.get("status")) if record else None

    def counts(self, trajectory: str | None = None) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for (record_trajectory, _), record in self.latest.items():
            if trajectory is not None and record_trajectory != trajectory:
                continue
            counts[str(record.get("status") or "unknown")] += 1
        return dict(sorted(counts.items()))

    def trajectory_counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, Counter[str]] = {}
        for (trajectory, _), record in self.latest.items():
            result.setdefault(trajectory, Counter())[str(record.get("status") or "unknown")] += 1
        return {trajectory: dict(sorted(counts.items())) for trajectory, counts in result.items()}

    def latest_records(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(record) for _, record in sorted(self.latest.items())]


def trajectory_summaries(repository: EventRepository, store: AnnotationStore) -> list[dict[str, Any]]:
    counts = store.trajectory_counts()
    summaries: list[dict[str, Any]] = []
    for meta in repository.trajectories:
        trajectory_id = str(meta["trajectory_id"])
        status_counts = counts.get(trajectory_id, {})
        annotated = sum(status_counts.values())
        summaries.append(
            {
                **copy.deepcopy(meta),
                "status_counts": status_counts,
                "annotated_count": annotated,
                "unannotated_count": max(0, int(meta.get("count") or 0) - annotated),
            }
        )
    return summaries


def compact_context_event(event: dict[str, Any], status: str | None = None) -> dict[str, Any]:
    return {
        "id": event.get("id"),
        "order": event.get("order"),
        "kind": event.get("kind"),
        "type": event.get("type"),
        "action_type": event.get("action_type"),
        "agent": event.get("agent"),
        "target": event.get("target"),
        "participants": event.get("participants") or [],
        "text": event.get("text"),
        "valid": event.get("valid"),
        "flag": event.get("flag") or [],
        "annotation_status": status,
    }


def export_snapshot(
    repository: EventRepository,
    store: AnnotationStore,
    output_dir: Path,
    validator: Any | None = None,
) -> dict[str, Any]:
    latest = store.latest_records()
    accepted_events: list[dict[str, Any]] = []
    needs_review: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for record in latest:
        status = str(record.get("status") or "unknown")
        status_counts[status] += 1
        trajectory = str(record["trajectory"])
        event_id = int(record["event_id"])
        try:
            event, _, _ = repository.get_event(trajectory, event_id)
        except KeyError as exc:
            stale.append({"trajectory": trajectory, "event_id": event_id, "reason": str(exc)})
            continue
        if json_sha256(event) != record.get("base_sha256"):
            stale.append({"trajectory": trajectory, "event_id": event_id, "reason": "base_sha256_mismatch"})
            continue
        if status == "needs_review":
            needs_review.append(record)
        if status != "accepted":
            continue
        filled = apply_patch(event, record.get("patch") or {})
        row_errors = []
        if validator is not None:
            row_errors = list(validator.iter_errors(filled))
        if row_errors:
            validation_errors.append(
                {
                    "trajectory": trajectory,
                    "event_id": event_id,
                    "errors": [error.message for error in row_errors],
                }
            )
            continue
        accepted_events.append(
            {
                "annotation": {
                    "annotation_id": record.get("annotation_id"),
                    "revision": record.get("revision"),
                    "annotator": record.get("annotator"),
                    "created_at": record.get("created_at"),
                    "note": record.get("note"),
                },
                "event": filled,
            }
        )

    write_jsonl_atomic(output_dir / "annotations.jsonl", latest)
    write_jsonl_atomic(output_dir / "accepted_events.jsonl", accepted_events)
    write_jsonl_atomic(output_dir / "needs_review.jsonl", needs_review)
    summary = {
        "format": "human_annotation_v0",
        "generated_at": utc_now(),
        "source": {
            "graph": str(repository.graph_path),
            "events_dir": str(repository.events_dir),
            "history": str(store.history_path),
        },
        "latest_annotation_count": len(latest),
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_event_count": len(accepted_events),
        "needs_review_count": len(needs_review),
        "stale_count": len(stale),
        "stale": stale,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "outputs": {
            "annotations": str(output_dir / "annotations.jsonl"),
            "accepted_events": str(output_dir / "accepted_events.jsonl"),
            "needs_review": str(output_dir / "needs_review.jsonl"),
        },
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary
