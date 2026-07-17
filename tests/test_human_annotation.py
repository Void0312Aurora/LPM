from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from lpm.human_annotation import (
    AnnotationStore,
    EventRepository,
    EvidenceIndex,
    RevisionConflict,
    apply_patch,
    export_snapshot,
    normalize_patch,
    validate_annotation,
)


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class HumanAnnotationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events_dir = self.root / "events"
        self.events = [
            {
                "id": 0,
                "source": {"dataset": "tree_dialogues_v2", "scene": "scene_a", "path": "content[0]"},
                "trajectory": "t0000",
                "order": 0,
                "kind": "normal",
                "type": "utterance",
                "action_type": "speak",
                "agent": "羽未",
                "target": None,
                "participants": ["羽未"],
                "text": "“早上好。”",
                "choice": None,
                "valid": True,
                "flag": [],
            },
            {
                "id": 1,
                "source": {"dataset": "tree_dialogues_v2", "scene": "scene_a", "path": "content[1]"},
                "trajectory": "t0000",
                "order": 1,
                "kind": "normal",
                "type": "narration",
                "action_type": "other",
                "agent": None,
                "target": None,
                "participants": [],
                "text": "心里感到有些奇怪。",
                "choice": None,
                "valid": True,
                "flag": [],
            },
        ]
        write_jsonl(self.events_dir / "t0000.jsonl", self.events)
        self.graph_path = self.root / "graph.json"
        self.graph_path.write_text(
            json.dumps(
                {
                    "nodes": {
                        "t0000": {
                            "scene": "scene_a",
                            "file": "events/t0000.jsonl",
                            "count": 2,
                            "entry": 0,
                            "valid": True,
                            "flag": [],
                        }
                    },
                    "edges": {},
                    "valid": True,
                    "flag": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        canonical_path = self.root / "canonical.jsonl"
        write_jsonl(
            canonical_path,
            [
                {
                    "event_id": "scene_a::content[0]",
                    "source_file": "source.jsonl",
                    "source_line": 1,
                    "source_path": "content[0]",
                    "scene": "scene_a",
                    "local_pos": 0,
                    "text_index": 10,
                    "raw_type": "text",
                    "event_kind": "text",
                    "observation_channel": "direct_dialogue",
                    "speaker_raw": "羽未",
                    "speaker_norm": "羽未",
                    "text_raw": "“早上好。”",
                    "text_norm": "“早上好。”",
                    "flags": [],
                },
                {
                    "event_id": "scene_a::content[1]",
                    "source_file": "source.jsonl",
                    "source_line": 1,
                    "source_path": "content[1]",
                    "scene": "scene_a",
                    "local_pos": 1,
                    "text_index": 11,
                    "raw_type": "text",
                    "event_kind": "text",
                    "observation_channel": "narration",
                    "speaker_raw": None,
                    "speaker_norm": None,
                    "text_raw": "心里感到有些奇怪。",
                    "text_norm": "心里感到有些奇怪。",
                    "flags": [],
                },
            ],
        )
        aligned_path = self.root / "aligned.jsonl"
        write_jsonl(
            aligned_path,
            [
                {
                    "scene_file": "scene_a",
                    "text_index": 10,
                    "speaker": "うみ",
                    "text": "おはよう。",
                    "text_type": "dialogue",
                    "zh_speaker": "羽未",
                    "zh_text": "“早上好。”",
                    "en_text": "@Umi@ Good morning.",
                }
            ],
        )
        self.evidence = EvidenceIndex(canonical_path, aligned_path)
        self.repository = EventRepository(self.graph_path, self.events_dir, self.evidence)
        schema = json.loads((ROOT / "schemas/clean_event_v0.schema.json").read_text(encoding="utf-8"))
        self.validator = Draft202012Validator(schema)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_source_speaker_locks_direct_dialogue_fields(self) -> None:
        event = self.events[0]
        evidence = self.evidence.for_event(event)
        patch = normalize_patch(
            event,
            {
                "type": "narration",
                "agent": "羽依里",
                "target": "羽依里",
                "participants": [],
                "valid": True,
                "flag": [],
            },
            evidence,
            status="accepted",
        )
        self.assertEqual(patch["type"], "utterance")
        self.assertEqual(patch["action_type"], "speak")
        self.assertEqual(patch["agent"], "羽未")
        self.assertEqual(patch["target"], "羽依里")
        self.assertEqual(patch["participants"], ["羽依里", "羽未"])
        self.assertEqual(validate_annotation(event, patch, "accepted", "", self.validator), [])

    def test_context_window_keeps_full_size_near_trajectory_edges(self) -> None:
        events_dir = self.root / "window-events"
        window_events = [
            {
                "id": index,
                "source": {"dataset": "test", "scene": "window_scene", "path": f"content[{index}]"},
                "trajectory": "t0001",
                "order": index,
                "kind": "normal",
                "type": "narration",
                "action_type": "other",
                "agent": None,
                "target": None,
                "participants": [],
                "text": f"event {index}",
                "choice": None,
                "valid": True,
                "flag": [],
            }
            for index in range(10)
        ]
        write_jsonl(events_dir / "t0001.jsonl", window_events)
        graph_path = self.root / "window-graph.json"
        graph_path.write_text(
            json.dumps(
                {
                    "nodes": {
                        "t0001": {
                            "scene": "window_scene",
                            "file": "window-events/t0001.jsonl",
                            "count": 10,
                            "entry": 0,
                            "valid": True,
                            "flag": [],
                        }
                    },
                    "edges": {},
                    "valid": True,
                    "flag": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        repository = EventRepository(graph_path, events_dir, EvidenceIndex())

        self.assertEqual([event["id"] for event in repository.context("t0001", 0, 2)], [0, 1, 2, 3, 4])
        self.assertEqual([event["id"] for event in repository.context("t0001", 5, 2)], [3, 4, 5, 6, 7])
        self.assertEqual([event["id"] for event in repository.context("t0001", 9, 2)], [5, 6, 7, 8, 9])

    def test_monologue_normalization_and_review_note(self) -> None:
        event = self.events[1]
        patch = normalize_patch(
            event,
            {
                "type": "monologue",
                "agent": "羽依里",
                "target": "羽未",
                "participants": [],
                "valid": True,
                "flag": [],
            },
            self.evidence.for_event(event),
            status="needs_review",
        )
        self.assertIsNone(patch["target"])
        self.assertEqual(patch["participants"], ["羽依里"])
        self.assertIn("needs_review", patch["flag"])
        errors = validate_annotation(event, patch, "needs_review", "", self.validator)
        self.assertTrue(any(error["path"] == ["note"] for error in errors))
        self.assertEqual(validate_annotation(event, patch, "needs_review", "需核对视角", self.validator), [])

    def test_append_only_store_detects_revision_conflict(self) -> None:
        store = AnnotationStore(self.root / "history.jsonl")
        event = self.events[0]
        patch = normalize_patch(event, {}, self.evidence.for_event(event), status="accepted")
        record = store.append(
            trajectory="t0000",
            event=event,
            patch=patch,
            status="accepted",
            annotator="tester",
            note="",
            expected_revision=0,
        )
        self.assertEqual(record["revision"], 1)
        with self.assertRaises(RevisionConflict):
            store.append(
                trajectory="t0000",
                event=event,
                patch=patch,
                status="accepted",
                annotator="tester",
                note="stale tab",
                expected_revision=0,
            )
        reloaded = AnnotationStore(self.root / "history.jsonl")
        self.assertEqual(reloaded.get("t0000", 0)["revision"], 1)

    def test_export_contains_only_current_valid_accepted_events(self) -> None:
        store = AnnotationStore(self.root / "history.jsonl")
        accepted_patch = normalize_patch(
            self.events[0],
            {"target": "羽依里", "participants": ["羽依里"]},
            self.evidence.for_event(self.events[0]),
            status="accepted",
        )
        store.append(
            trajectory="t0000",
            event=self.events[0],
            patch=accepted_patch,
            status="accepted",
            annotator="tester",
            note="aligned checked",
            expected_revision=0,
        )
        review_patch = normalize_patch(
            self.events[1],
            {"type": "monologue", "agent": "羽依里"},
            self.evidence.for_event(self.events[1]),
            status="needs_review",
        )
        store.append(
            trajectory="t0000",
            event=self.events[1],
            patch=review_patch,
            status="needs_review",
            annotator="tester",
            note="viewpoint uncertain",
            expected_revision=0,
        )
        output = self.root / "export"
        summary = export_snapshot(self.repository, store, output, self.validator)
        self.assertEqual(summary["accepted_event_count"], 1)
        self.assertEqual(summary["needs_review_count"], 1)
        accepted_rows = [json.loads(line) for line in (output / "accepted_events.jsonl").read_text().splitlines()]
        self.assertEqual(accepted_rows[0]["event"]["target"], "羽依里")
        self.assertEqual(list(self.validator.iter_errors(accepted_rows[0]["event"])), [])


if __name__ == "__main__":
    unittest.main()
