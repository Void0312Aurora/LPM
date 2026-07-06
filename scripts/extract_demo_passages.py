#!/usr/bin/env python3
"""Extract representative LPM demo passages and optionally annotate them with DeepSeek."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


XI_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = XI_ROOT.parent
DEFAULT_INPUT = XI_ROOT / "GalGame" / "data" / "tree_dialogues_v2.jsonl"
DEFAULT_OUTPUT_DIR = XI_ROOT / "LPM" / "data" / "demo_passages"
DEFAULT_ENV = XI_ROOT / "LPM" / ".env"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_EXCLUDE_SCENE_RE = r"デバッグ|ボイスチェック|CS用処理|_var|_task|_scr|_build|シナリオフロー"

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class Event:
    scene: str
    pos: int
    kind: str
    index: int | None
    speaker: str | None
    text: str
    raw_type: str
    branch_count: int = 0
    branches: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = {
            "scene": self.scene,
            "pos": self.pos,
            "kind": self.kind,
            "index": self.index,
            "speaker": self.speaker,
            "text": self.text,
            "raw_type": self.raw_type,
        }
        if self.branch_count:
            data["branch_count"] = self.branch_count
        if self.branches:
            data["branches"] = self.branches
        return data


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return CONTROL_CHARS.sub("", value).strip()


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


def load_scenes(path: Path) -> list[dict[str, Any]]:
    scenes = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            scenes.append(json.loads(line))
    return scenes


def normalize_event(scene: str, pos: int, item: dict[str, Any]) -> Event | None:
    raw_type = str(item.get("type"))
    if raw_type == "text":
        text = clean_text(item.get("text"))
        if not text:
            return None
        speaker = item.get("speaker")
        kind = "narration" if speaker is None else "dialogue"
        return Event(
            scene=scene,
            pos=pos,
            kind=kind,
            index=item.get("index"),
            speaker=speaker,
            text=text,
            raw_type=raw_type,
        )
    if raw_type == "choice":
        branches = []
        for branch in item.get("branches") or []:
            if not isinstance(branch, dict):
                continue
            branches.append(
                {
                    "option_value": branch.get("option_value"),
                    "speaker": branch.get("speaker"),
                    "option": clean_text(branch.get("option")),
                    "has_leads_to": isinstance(branch.get("leads_to"), dict),
                }
            )
        text = " / ".join(b["option"] for b in branches if b.get("option"))
        if not text:
            text = "<choice>"
        return Event(
            scene=scene,
            pos=pos,
            kind="choice",
            index=None,
            speaker=None,
            text=text,
            raw_type=raw_type,
            branch_count=len(branches),
            branches=branches,
        )
    return None


def scene_events(scene_obj: dict[str, Any]) -> list[Event]:
    scene = scene_obj["scene"]
    events: list[Event] = []
    for pos, item in enumerate(scene_obj.get("content") or []):
        if not isinstance(item, dict):
            continue
        event = normalize_event(scene, pos, item)
        if event is not None:
            events.append(event)
    return events


def passage_metrics(events: list[Event]) -> dict[str, Any]:
    speakers = Counter(e.speaker for e in events if e.speaker)
    kinds = Counter(e.kind for e in events)
    text_chars = sum(len(e.text) for e in events)
    max_same_speaker_run = 0
    current_speaker = None
    current_run = 0
    for event in events:
        if event.speaker and event.speaker == current_speaker:
            current_run += 1
        else:
            current_speaker = event.speaker
            current_run = 1 if event.speaker else 0
        max_same_speaker_run = max(max_same_speaker_run, current_run)
    return {
        "event_count": len(events),
        "text_chars": text_chars,
        "speaker_count": len(speakers),
        "speakers": dict(speakers.most_common(12)),
        "kind_counts": dict(kinds),
        "narration_ratio": round(kinds.get("narration", 0) / max(len(events), 1), 4),
        "choice_count": kinds.get("choice", 0),
        "max_same_speaker_run": max_same_speaker_run,
    }


def make_candidate(
    scene: str,
    events: list[Event],
    start: int,
    end: int,
    candidate_type: str,
    reason: str,
    score: float,
) -> dict[str, Any] | None:
    start = max(0, start)
    end = min(len(events), end)
    if end <= start:
        return None
    window = events[start:end]
    metrics = passage_metrics(window)
    if metrics["event_count"] < 4 or metrics["text_chars"] < 20:
        return None
    return {
        "candidate_id": "",
        "source_dataset": str(DEFAULT_INPUT.relative_to(WORKSPACE_ROOT)),
        "scene": scene,
        "start_pos": window[0].pos,
        "end_pos": window[-1].pos,
        "candidate_type": candidate_type,
        "selection_reason": reason,
        "score": round(score, 4),
        "metrics": metrics,
        "events": [event.as_dict() for event in window],
    }


def add_unique(candidates: list[dict[str, Any]], seen: set[tuple], candidate: dict[str, Any] | None) -> None:
    if candidate is None:
        return
    key = (candidate["scene"], candidate["start_pos"], candidate["end_pos"], candidate["candidate_type"])
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def select_candidates(
    scenes: list[dict[str, Any]],
    radius: int,
    max_candidates: int,
    exclude_scene_pattern: str,
) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    major_speakers = Counter()
    scene_data = []
    exclude_scene_re = re.compile(exclude_scene_pattern) if exclude_scene_pattern else None

    for scene_obj in scenes:
        scene = scene_obj["scene"]
        if exclude_scene_re and exclude_scene_re.search(scene):
            continue
        events = scene_events(scene_obj)
        if not events:
            continue
        scene_data.append((scene, events))
        for event in events:
            if event.speaker:
                major_speakers[event.speaker] += 1

        for i, event in enumerate(events):
            if event.kind == "choice":
                score = 20 + event.branch_count * 5 + passage_metrics(events[max(0, i - radius): i + radius + 1])["speaker_count"]
                cand = make_candidate(
                    scene,
                    events,
                    i - radius,
                    i + radius + 1,
                    "choice_window",
                    "Window centered on a choice node.",
                    score,
                )
                if cand:
                    by_type["choice_window"].append(cand)

        window = max(8, radius * 2)
        for start in range(0, max(len(events) - window + 1, 0), max(1, window // 2)):
            end = start + window
            chunk = events[start:end]
            metrics = passage_metrics(chunk)
            if metrics["speaker_count"] >= 3 and metrics["narration_ratio"] <= 0.25 and metrics["choice_count"] == 0:
                score = metrics["speaker_count"] * 5 + metrics["event_count"] - metrics["narration_ratio"] * 10
                cand = make_candidate(
                    scene,
                    events,
                    start,
                    end,
                    "multi_speaker_dialogue",
                    "Dense cross-speaker dialogue window.",
                    score,
                )
                if cand:
                    by_type["multi_speaker_dialogue"].append(cand)
            if metrics["narration_ratio"] >= 0.45 and metrics["speaker_count"] >= 1:
                score = metrics["narration_ratio"] * 20 + metrics["speaker_count"] * 2
                cand = make_candidate(
                    scene,
                    events,
                    start,
                    end,
                    "narration_mediated",
                    "Narration-heavy window for viewpoint-observation tests.",
                    score,
                )
                if cand:
                    by_type["narration_mediated"].append(cand)

        run_start = None
        run_speaker = None
        for i, event in enumerate(events + [Event(scene, len(events), "sentinel", None, None, "", "sentinel")]):
            if event.speaker and event.speaker == run_speaker:
                continue
            if run_speaker is not None and run_start is not None:
                run_len = i - run_start
                if run_len >= 2:
                    score = run_len * 10
                    cand = make_candidate(
                        scene,
                        events,
                        run_start - radius // 2,
                        i + radius // 2,
                        "consecutive_same_speaker",
                        "Window around consecutive outputs by the same speaker.",
                        score,
                    )
                    if cand:
                        by_type["consecutive_same_speaker"].append(cand)
            run_start = i if event.speaker else None
            run_speaker = event.speaker

    for scene, events in sorted(scene_data, key=lambda item: len(item[1]), reverse=True)[:8]:
        center = len(events) // 2
        score = len(events)
        cand = make_candidate(
            scene,
            events,
            center - radius,
            center + radius,
            "long_scene_middle",
            "Middle slice from a long scene.",
            score,
        )
        if cand:
            by_type["long_scene_middle"].append(cand)

    target_speakers = [speaker for speaker, _ in major_speakers.most_common(8)]
    for scene, events in scene_data:
        for i, event in enumerate(events):
            if event.speaker not in target_speakers:
                continue
            left = events[max(0, i - radius): i]
            has_external = any(e.speaker != event.speaker for e in left if e.kind in {"dialogue", "narration"})
            if not has_external:
                continue
            score = major_speakers[event.speaker] / 1000 + passage_metrics(left + [event])["speaker_count"]
            cand = make_candidate(
                scene,
                events,
                i - radius,
                i + 1,
                "target_response_unit",
                "Major-character response after external input.",
                score,
            )
            if cand:
                by_type["target_response_unit"].append(cand)
                break

    caps = {
        "choice_window": 6,
        "narration_mediated": 6,
        "multi_speaker_dialogue": 6,
        "consecutive_same_speaker": 5,
        "long_scene_middle": 4,
        "target_response_unit": 8,
    }
    ranked_by_type: dict[str, list[dict[str, Any]]] = {}
    category_order = [
        "choice_window",
        "narration_mediated",
        "multi_speaker_dialogue",
        "target_response_unit",
        "consecutive_same_speaker",
        "long_scene_middle",
    ]
    for candidate_type, items in by_type.items():
        items = sorted(items, key=lambda item: (-item["score"], item["scene"], item["start_pos"]))
        ranked_by_type[candidate_type] = items[: caps.get(candidate_type, 4)]

    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    round_index = 0
    while len(candidates) < max_candidates:
        added = False
        for candidate_type in category_order:
            bucket = ranked_by_type.get(candidate_type, [])
            if round_index >= len(bucket):
                continue
            before = len(candidates)
            add_unique(candidates, seen, bucket[round_index])
            added = added or len(candidates) > before
            if len(candidates) >= max_candidates:
                break
        if not added:
            break
        round_index += 1

    for i, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"demo_{i:04d}_{candidate['candidate_type']}"
    return candidates


def candidate_for_prompt(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "scene": candidate["scene"],
        "candidate_type": candidate["candidate_type"],
        "selection_reason": candidate["selection_reason"],
        "metrics": candidate["metrics"],
        "events": candidate["events"],
    }


def annotate_candidate(candidate: dict[str, Any], api_key: str, base_url: str, model: str, timeout: int) -> dict[str, Any]:
    system = (
        "You are reviewing GalGame narrative passages for a Latent Persona Modeling dataset. "
        "Do not rewrite the source text. Do not infer psychological truth from narration. "
        "Treat narration as viewpoint-mediated observation. Return strict JSON only."
    )
    user = {
        "task": "Evaluate whether this passage is useful as a demonstration/test passage for LPM.",
        "required_json_fields": [
            "candidate_id",
            "selected",
            "quality_score_1_to_5",
            "recommended_uses",
            "observation_channels",
            "viewpoint_mediation_notes",
            "risks",
            "short_reason",
        ],
        "passage": candidate_for_prompt(candidate),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return {
        "candidate_id": candidate["candidate_id"],
        "model": model,
        "annotation": parsed,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument(
        "--exclude-scene-re",
        default=DEFAULT_EXCLUDE_SCENE_RE,
        help="Regex for scenes excluded from demo extraction.",
    )
    parser.add_argument("--annotate", action="store_true", help="Call DeepSeek and write annotations.jsonl.")
    parser.add_argument("--llm-limit", type=int, default=8, help="Maximum candidates to annotate.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = load_scenes(args.input)
    candidates = select_candidates(
        scenes,
        radius=args.radius,
        max_candidates=args.max_candidates,
        exclude_scene_pattern=args.exclude_scene_re,
    )
    candidates_path = args.output_dir / "candidates.jsonl"
    write_jsonl(candidates_path, candidates)
    print(f"wrote {len(candidates)} candidates to {candidates_path}")

    if not args.annotate:
        return 0

    local_env = load_env(args.env)
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("API_KEY")
        or local_env.get("DEEPSEEK_API_KEY")
        or local_env.get("API_KEY")
    )
    if not api_key:
        raise SystemExit(f"No DeepSeek API key found in environment or {args.env}")

    annotations: list[dict[str, Any]] = []
    for candidate in candidates[: args.llm_limit]:
        try:
            annotation = annotate_candidate(
                candidate,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
            )
        except Exception as exc:
            annotation = {
                "candidate_id": candidate["candidate_id"],
                "model": args.model,
                "error": f"{type(exc).__name__}: {exc}",
            }
        annotations.append(annotation)
        time.sleep(args.sleep)

    annotations_path = args.output_dir / "annotations.jsonl"
    write_jsonl(annotations_path, annotations)
    print(f"wrote {len(annotations)} annotations to {annotations_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
