#!/usr/bin/env python3
"""Run LLM-assisted scene cleaning over canonical events."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lpm.canonical import read_jsonl


DEFAULT_CANONICAL = REPO_ROOT / "data/canonical/events.jsonl"
DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/enriched/scene_cleaning"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

EVENT_LABELS = [
    "dialogue",
    "internal_monologue",
    "narration_action",
    "narration_scene",
    "narration_subjective",
    "choice_option",
    "system_or_debug",
    "unclear",
]
OBSERVATION_CHANNELS = [
    "direct_dialogue",
    "narration",
    "choice",
    "scene_jump",
    "system_or_debug",
    "unknown",
]
OBSERVERS = ["not_applicable", "protagonist", "speaker", "external_narrator", "unknown"]
EVENT_ISSUES = [
    "viewpoint_bias",
    "ambiguous_actor",
    "ambiguous_speaker",
    "fragment_without_context",
    "system_or_debug",
    "branch_incomplete",
    "text_noise",
    "translation_or_alignment_risk",
    "label_mismatch",
    "chain_break",
]
JUMP_NATURALNESS = [
    "natural_route_branch",
    "natural_scene_transition",
    "system_or_debug",
    "parse_error",
    "raw_only_unresolved",
    "missing_target",
    "needs_review",
]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_scene_slug(scene: str, scene_index: int) -> str:
    cleaned = re.sub(r"[\\/:\s]+", "_", scene).strip("_")
    return f"scene_{scene_index:03d}_{cleaned or 'unknown'}"


def normalize_model_id(model: str) -> str:
    stripped = model.strip()
    if stripped.lower().startswith("deepseek"):
        return stripped.lower()
    return stripped


def scene_order(events: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    known: set[str] = set()
    for event in events:
        scene = str(event.get("scene") or "")
        if scene and scene not in known:
            seen.append(scene)
            known.add(scene)
    return seen


def select_scene(events: list[dict[str, Any]], scene: str | None, scene_index: int) -> tuple[str, int]:
    scenes = scene_order(events)
    if scene:
        if scene not in scenes:
            raise SystemExit(f"Scene not found: {scene}")
        return scene, scenes.index(scene)
    if scene_index < 0 or scene_index >= len(scenes):
        raise SystemExit(f"Scene index out of range: {scene_index}; total scenes={len(scenes)}")
    return scenes[scene_index], scene_index


def compact_event(event: dict[str, Any], max_chars: int = 220) -> dict[str, Any]:
    text = event.get("text_norm")
    if isinstance(text, str) and len(text) > max_chars:
        text = text[: max_chars - 1] + "..."
    compact = {
        "event_id": event.get("event_id"),
        "source_path": event.get("source_path"),
        "local_pos": event.get("local_pos"),
        "text_index": event.get("text_index"),
        "event_kind": event.get("event_kind"),
        "deterministic_observation_channel": event.get("observation_channel"),
        "speaker": event.get("speaker_norm"),
        "text": text,
        "flags": event.get("flags") or [],
    }
    if event.get("choice"):
        compact["choice"] = event.get("choice")
    if event.get("jump"):
        compact["jump"] = event.get("jump")
    return compact


def make_event_windows(
    scene_events: list[dict[str, Any]],
    target_size: int,
    context_size: int,
    max_windows: int,
) -> list[dict[str, Any]]:
    text_like = [event for event in scene_events if event.get("event_kind") in {"text", "choice"}]
    windows: list[dict[str, Any]] = []
    for start in range(0, len(text_like), target_size):
        end = min(len(text_like), start + target_size)
        left = max(0, start - context_size)
        right = min(len(text_like), end + context_size)
        target_events = text_like[start:end]
        context_events = text_like[left:right]
        window_id = f"window_{len(windows):04d}"
        windows.append(
            {
                "window_id": window_id,
                "target_start": start,
                "target_end": end - 1,
                "target_event_ids": [str(event["event_id"]) for event in target_events],
                "context_events": [compact_event(event) for event in context_events],
            }
        )
        if max_windows and len(windows) >= max_windows:
            break
    return windows


def make_event_review_request(scene: str, window: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Review canonical GalGame events for LPM data cleaning. Do not rewrite source text.",
        "scene": scene,
        "window_id": window["window_id"],
        "target_event_ids": window["target_event_ids"],
        "closed_sets": {
            "semantic_label": EVENT_LABELS,
            "corrected_observation_channel": OBSERVATION_CHANNELS,
            "observer": OBSERVERS,
            "issue_codes": EVENT_ISSUES,
        },
        "required_output": {
            "scene": scene,
            "window_id": window["window_id"],
            "event_annotations": [
                {
                    "event_id": "must be one of target_event_ids",
                    "semantic_label": "one closed-set label",
                    "deterministic_label_ok": True,
                    "corrected_observation_channel": "one closed-set observation channel",
                    "viewpoint_mediated": None,
                    "observer": "one closed-set observer",
                    "actor_candidates": [],
                    "addressed_to_candidates": [],
                    "issue_codes": [],
                    "confidence": 0.0,
                    "short_reason": "short Chinese or English reason without rewriting the source text",
                }
            ],
            "chain_review": {
                "coherent": True,
                "break_points": [],
                "missing_context": False,
                "issue_codes": [],
                "summary": "brief statement about local narrative continuity",
            },
        },
        "rules": [
            "Annotate every target_event_id exactly once; use context_events only as context.",
            "If speaker is present and the line is spoken outwardly, use dialogue/direct_dialogue.",
            "For outward dialogue with no narrative observer, use observer=not_applicable.",
            "Use internal_monologue only when the text presents inner thought rather than spoken exchange.",
            "Use narration_* labels for speakerless action, scene, or subjective viewpoint descriptions.",
            "Narration may be viewpoint-mediated; do not treat it as psychological ground truth.",
            "Flag label_mismatch when deterministic_observation_channel is too coarse or wrong.",
            "Return strict JSON only.",
        ],
        "context_events": window["context_events"],
    }


def make_jump_review_request(
    scene: str,
    scene_events: list[dict[str, Any]],
    all_events_by_scene: dict[str, list[dict[str, Any]]],
    preview_events: int,
) -> dict[str, Any] | None:
    jumps = [event for event in scene_events if event.get("event_kind") == "jump"]
    if not jumps:
        return None
    text_like = [event for event in scene_events if event.get("event_kind") in {"text", "choice"}]
    targets: dict[str, Any] = {}
    for jump in jumps:
        target = ((jump.get("jump") or {}).get("target_scene"))
        if not isinstance(target, str) or target in targets:
            continue
        target_events = [
            event
            for event in all_events_by_scene.get(target, [])
            if event.get("event_kind") in {"text", "choice"}
        ][:preview_events]
        targets[target] = [compact_event(event) for event in target_events]
    return {
        "task": "Review whether scene jumps are narrative transitions, route branches, system/debug jumps, or parse errors.",
        "scene": scene,
        "closed_sets": {
            "naturalness": JUMP_NATURALNESS,
            "issue_codes": EVENT_ISSUES + ["jump_target_unavailable", "jump_target_system", "jump_parse_noise"],
        },
        "required_output": {
            "scene": scene,
            "jump_reviews": [
                {
                    "event_id": "jump event_id",
                    "target_scene": "copied target",
                    "is_narrative_jump": True,
                    "naturalness": "one closed-set value",
                    "issue_codes": [],
                    "confidence": 0.0,
                    "short_reason": "brief reason",
                }
            ],
            "scene_level_summary": "brief statement about jump structure",
        },
        "rules": [
            "Do not invent missing targets.",
            "Treat deterministic jump.status as evidence, not as something to overwrite silently.",
            "System/debug targets are not narrative continuity.",
            "Route variants such as _u7/_u9 may be natural route branches if target previews support it.",
            "Return strict JSON only.",
        ],
        "source_tail_events": [compact_event(event) for event in text_like[-preview_events:]],
        "jump_events": [compact_event(event) for event in jumps],
        "target_previews": targets,
    }


def chat_completion(api_key: str, base_url: str, model: str, user_request: dict[str, Any], timeout: int) -> dict[str, Any]:
    system = (
        "You are a careful data-cleaning reviewer for a GalGame Latent Persona Modeling dataset. "
        "You classify observations and local continuity. You never rewrite source text, never invent missing plot, "
        "and return strict JSON only."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
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


def validate_event_review(parsed: dict[str, Any], expected_ids: list[str]) -> dict[str, Any]:
    annotations = parsed.get("event_annotations")
    if not isinstance(annotations, list):
        return {"ok": False, "missing": expected_ids, "extra": [], "duplicate": [], "error": "event_annotations_not_list"}
    seen: list[str] = []
    for item in annotations:
        if isinstance(item, dict):
            seen.append(str(item.get("event_id")))
    expected = set(expected_ids)
    seen_set = set(seen)
    duplicate = sorted({event_id for event_id in seen if seen.count(event_id) > 1})
    return {
        "ok": expected == seen_set and not duplicate,
        "missing": sorted(expected - seen_set),
        "extra": sorted(seen_set - expected),
        "duplicate": duplicate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=6)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all windows are prepared.")
    parser.add_argument("--annotate", action="store_true", help="Call the LLM for prepared windows.")
    parser.add_argument("--llm-window-limit", type=int, default=1, help="0 means all prepared windows when --annotate is set.")
    parser.add_argument("--include-jumps", action="store_true", help="Prepare and optionally annotate scene-level jump review.")
    parser.add_argument("--preview-events", type=int, default=8)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = load_env(args.env)
    model = normalize_model_id(args.model or os.environ.get("MODEL") or env.get("MODEL") or DEFAULT_MODEL)
    events = read_jsonl(args.canonical)
    scene, scene_index = select_scene(events, args.scene, args.scene_index)
    slug = safe_scene_slug(scene, scene_index)

    all_events_by_scene: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        all_events_by_scene.setdefault(str(event.get("scene") or ""), []).append(event)
    scene_events = all_events_by_scene[scene]
    windows = make_event_windows(scene_events, args.target_size, args.context_size, args.max_windows)
    requests_out = [
        {
            "request_id": f"{slug}::{window['window_id']}",
            "request_type": "event_window_review",
            "model": model,
            "request": make_event_review_request(scene, window),
        }
        for window in windows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = args.output_dir / f"{slug}_requests.jsonl"
    write_jsonl(requests_path, requests_out)

    jump_request = None
    jump_request_path = None
    if args.include_jumps:
        jump_request = make_jump_review_request(scene, scene_events, all_events_by_scene, args.preview_events)
        if jump_request is not None:
            jump_request_path = args.output_dir / f"{slug}_jump_request.json"
            jump_request_path.write_text(json.dumps(jump_request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "scene": scene,
        "scene_index": scene_index,
        "model": model,
        "canonical_event_count": len(scene_events),
        "text_or_choice_event_count": sum(1 for event in scene_events if event.get("event_kind") in {"text", "choice"}),
        "jump_event_count": sum(1 for event in scene_events if event.get("event_kind") == "jump"),
        "prepared_window_count": len(windows),
        "requests_path": str(requests_path),
        "jump_request_path": str(jump_request_path) if jump_request_path else None,
    }

    annotations: list[dict[str, Any]] = []
    jump_review: dict[str, Any] | None = None
    if args.annotate:
        api_key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("API_KEY")
            or env.get("DEEPSEEK_API_KEY")
            or env.get("API_KEY")
        )
        if not api_key:
            raise SystemExit(f"No API key found in environment or {args.env}")
        llm_limit = len(requests_out) if args.llm_window_limit == 0 else min(args.llm_window_limit, len(requests_out))
        for item in requests_out[:llm_limit]:
            request = item["request"]
            try:
                parsed = chat_completion(api_key, args.base_url, model, request, args.timeout)
                validation = validate_event_review(parsed, request["target_event_ids"])
                annotations.append(
                    {
                        "request_id": item["request_id"],
                        "request_type": item["request_type"],
                        "model": model,
                        "annotation": parsed,
                        "validation": validation,
                    }
                )
            except Exception as exc:
                annotations.append(
                    {
                        "request_id": item["request_id"],
                        "request_type": item["request_type"],
                        "model": model,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            time.sleep(args.sleep)

        if jump_request is not None:
            try:
                jump_review = {
                    "request_type": "jump_review",
                    "model": model,
                    "annotation": chat_completion(api_key, args.base_url, model, jump_request, args.timeout),
                }
            except Exception as exc:
                jump_review = {
                    "request_type": "jump_review",
                    "model": model,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        annotations_path = args.output_dir / f"{slug}_annotations.jsonl"
        write_jsonl(annotations_path, annotations)
        summary["annotations_path"] = str(annotations_path)
        summary["annotated_window_count"] = len(annotations)
        if jump_review is not None:
            jump_review_path = args.output_dir / f"{slug}_jump_review.json"
            jump_review_path.write_text(json.dumps(jump_review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary["jump_review_path"] = str(jump_review_path)

    summary_path = args.output_dir / f"{slug}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {len(windows)} event-review requests for scene {scene}")
    print(f"wrote requests to {requests_path}")
    if args.annotate:
        print(f"wrote {len(annotations)} LLM annotations")
    if jump_request_path:
        print(f"wrote jump request to {jump_request_path}")
    print(f"wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
