#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpm.human_annotation import (  # noqa: E402
    ANNOTATION_STATUSES,
    FLAG_SUGGESTIONS,
    NORMAL_TYPES,
    AnnotationError,
    AnnotationStore,
    EventRepository,
    EvidenceIndex,
    RevisionConflict,
    apply_patch,
    compact_context_event,
    default_patch,
    export_snapshot,
    json_sha256,
    normalize_patch,
    trajectory_summaries,
    validate_annotation,
)


STATIC_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY_BYTES = 2 * 1024 * 1024


def load_validator(schema_path: Path) -> Any:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SystemExit("jsonschema is required; install the project requirements first") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def as_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"expected an integer, got {value!r}") from exc


class AnnotationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        repository: EventRepository,
        store: AnnotationStore,
        validator: Any,
        annotator: str,
        export_dir: Path,
        default_context: int,
    ) -> None:
        super().__init__(address, AnnotationRequestHandler)
        self.repository = repository
        self.store = store
        self.validator = validator
        self.annotator = annotator
        self.export_dir = export_dir
        self.default_context = default_context

    def item_payload(self, trajectory: str, event_id: int | None, radius: int) -> dict[str, Any]:
        event, position, total = self.repository.get_event(trajectory, event_id)
        resolved_id = int(event["id"])
        evidence = self.repository.evidence.for_event(event)
        record = self.store.get(trajectory, resolved_id)
        base_patch = default_patch(event, evidence)
        if record is not None:
            patch = record.get("patch") or base_patch
            stale = record.get("base_sha256") != json_sha256(event)
        else:
            patch = base_patch
            stale = False
        preview = apply_patch(event, patch)
        context_rows = [
            compact_context_event(
                context_event,
                self.store.status_for(trajectory, int(context_event["id"])),
            )
            for context_event in self.repository.context(trajectory, resolved_id, radius)
        ]
        next_unannotated = self.repository.next_matching_id(
            trajectory,
            resolved_id,
            lambda candidate: self.store.status_for(trajectory, int(candidate["id"])) is None,
        )
        return {
            "trajectory": self.repository.trajectory_by_id[trajectory],
            "event": event,
            "position": position,
            "total": total,
            "context": context_rows,
            "context_radius": radius,
            "evidence": evidence,
            "policy": {
                "locked_fields": evidence.get("locked_fields") or {},
                "speaker_locked": bool(evidence.get("speaker_locked")),
                "base_flags": event.get("flag") or [],
            },
            "default_patch": base_patch,
            "patch": patch,
            "preview": preview,
            "annotation": record,
            "stale_annotation": stale,
            "navigation": {
                "previous_id": self.repository.neighbor_id(trajectory, resolved_id, -1),
                "next_id": self.repository.neighbor_id(trajectory, resolved_id, 1),
                "next_unannotated_id": next_unannotated,
            },
            "trajectory_status_counts": self.store.counts(trajectory),
        }


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    server: AnnotationHTTPServer

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write(f"[annotator] {self.address_string()} {format_string % args}\n")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.send_header("Cache-Control", "no-store")

    def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, *, errors: list[dict[str, Any]] | None = None) -> None:
        payload: dict[str, Any] = {"ok": False, "error": message}
        if errors is not None:
            payload["errors"] = errors
        self._json(payload, status=status)

    def _read_json(self) -> dict[str, Any]:
        content_length = as_int(self.headers.get("Content-Length"), 0) or 0
        if content_length <= 0:
            return {}
        if content_length > MAX_BODY_BYTES:
            raise ValueError(f"request body exceeds {MAX_BODY_BYTES} bytes")
        body = self.rfile.read(content_length)
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _check_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        return parsed.netloc == self.headers.get("Host") and parsed.scheme in {"http", "https"}

    def _serve_static(self, name: str) -> None:
        allowed = {"index.html", "app.js", "styles.css"}
        if name not in allowed:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = STATIC_DIR / name
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._serve_static("index.html")
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self._security_headers()
                self.end_headers()
                return
            if parsed.path in {"/app.js", "/styles.css"}:
                self._serve_static(parsed.path.lstrip("/"))
                return
            if parsed.path == "/api/bootstrap":
                trajectories = trajectory_summaries(self.server.repository, self.server.store)
                self._json(
                    {
                        "ok": True,
                        "annotator": self.server.annotator,
                        "normal_types": NORMAL_TYPES,
                        "annotation_statuses": ANNOTATION_STATUSES,
                        "flag_suggestions": FLAG_SUGGESTIONS,
                        "characters": self.server.repository.evidence.characters(),
                        "trajectories": trajectories,
                        "global_status_counts": self.server.store.counts(),
                        "history_record_count": self.server.store.record_count,
                        "history_load_errors": self.server.store.load_errors,
                        "evidence": {
                            "canonical_rows": self.server.repository.evidence.canonical_row_count,
                            "aligned_rows": self.server.repository.evidence.aligned_row_count,
                        },
                        "default_context": self.server.default_context,
                    }
                )
                return
            if parsed.path == "/api/item":
                trajectory = (query.get("trajectory") or [None])[0]
                if not trajectory:
                    trajectory = str(self.server.repository.trajectories[0]["trajectory_id"])
                event_id = as_int((query.get("event_id") or [None])[0], None)
                radius = as_int(
                    (query.get("context") or [str(self.server.default_context)])[0],
                    self.server.default_context,
                )
                radius = max(0, min(64, int(radius or 0)))
                self._json({"ok": True, "item": self.server.item_payload(str(trajectory), event_id, radius)})
                return
            if parsed.path == "/api/stats":
                self._json(
                    {
                        "ok": True,
                        "global_status_counts": self.server.store.counts(),
                        "trajectories": trajectory_summaries(self.server.repository, self.server.store),
                        "history_record_count": self.server.store.record_count,
                    }
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self.log_error("GET failed: %s", exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if not self._check_origin():
            self._error(HTTPStatus.FORBIDDEN, "origin mismatch")
            return
        try:
            payload = self._read_json()
            if parsed.path == "/api/annotations":
                trajectory = str(payload.get("trajectory") or "")
                event_id = as_int(str(payload.get("event_id")) if payload.get("event_id") is not None else None)
                if not trajectory or event_id is None:
                    raise ValueError("trajectory and event_id are required")
                event, _, _ = self.server.repository.get_event(trajectory, event_id)
                status = str(payload.get("status") or "accepted")
                note = str(payload.get("note") or "")
                evidence = self.server.repository.evidence.for_event(event)
                patch = normalize_patch(event, payload.get("patch"), evidence, status=status)
                errors = validate_annotation(event, patch, status, note, self.server.validator)
                if errors:
                    raise AnnotationError(errors)
                expected_revision = payload.get("expected_revision")
                record = self.server.store.append(
                    trajectory=trajectory,
                    event=event,
                    patch=patch,
                    status=status,
                    annotator=str(payload.get("annotator") or self.server.annotator),
                    note=note,
                    expected_revision=int(expected_revision) if expected_revision is not None else None,
                )
                next_unannotated = self.server.repository.next_matching_id(
                    trajectory,
                    event_id,
                    lambda candidate: self.server.store.status_for(trajectory, int(candidate["id"])) is None,
                )
                self._json(
                    {
                        "ok": True,
                        "record": record,
                        "next_unannotated_id": next_unannotated,
                        "trajectory_status_counts": self.server.store.counts(trajectory),
                        "global_status_counts": self.server.store.counts(),
                    }
                )
                return
            if parsed.path == "/api/export":
                summary = export_snapshot(
                    self.server.repository,
                    self.server.store,
                    self.server.export_dir,
                    self.server.validator,
                )
                self._json({"ok": True, "summary": summary, "output_dir": str(self.server.export_dir)})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except AnnotationError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "annotation validation failed", errors=exc.errors)
        except RevisionConflict as exc:
            self._json(
                {
                    "ok": False,
                    "error": "revision_conflict",
                    "current_revision": exc.current_revision,
                },
                status=HTTPStatus.CONFLICT,
            )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self.log_error("POST failed: %s", exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")


def add_common_arguments(parser: argparse.ArgumentParser, *, evidence: bool = True) -> None:
    parser.add_argument("--graph", type=Path, default=ROOT / "data/clean_v0/graph.json")
    parser.add_argument("--events-dir", type=Path, default=ROOT / "data/clean_v0/events")
    parser.add_argument("--event-schema", type=Path, default=ROOT / "schemas/clean_event_v0.schema.json")
    parser.add_argument(
        "--history",
        type=Path,
        default=ROOT / "data/human_annotations/history.jsonl",
        help="Append-only annotation audit log.",
    )
    if evidence:
        parser.add_argument("--canonical-events", type=Path, default=ROOT / "data/canonical/events.jsonl")
        parser.add_argument(
            "--aligned-dialogues",
            type=Path,
            default=ROOT / "data/upstream/Xi/GalGame/data/aligned_dialogues.jsonl",
        )
        parser.add_argument("--no-aligned-evidence", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LPM local human clean-event annotation GUI and export tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the local annotation web GUI.")
    add_common_arguments(serve, evidence=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--annotator", default=os.environ.get("USER", "anonymous"))
    serve.add_argument("--context", type=int, default=16)
    serve.add_argument("--cache-trajectories", type=int, default=8)
    serve.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    serve.add_argument(
        "--export-dir",
        type=Path,
        default=ROOT / "data/enriched/human_clean_event_v0",
    )

    stats = subparsers.add_parser("stats", help="Print current annotation coverage.")
    add_common_arguments(stats, evidence=False)

    export = subparsers.add_parser("export", help="Export the latest human annotation snapshot.")
    add_common_arguments(export, evidence=False)
    export.add_argument("--output-dir", type=Path, default=ROOT / "data/enriched/human_clean_event_v0")
    return parser.parse_args()


def build_repository(args: argparse.Namespace, *, load_evidence: bool) -> EventRepository:
    if load_evidence:
        aligned = None if args.no_aligned_evidence else args.aligned_dialogues
        print(f"loading canonical evidence from {args.canonical_events}", flush=True)
        if aligned is not None:
            print(f"loading aligned evidence from {aligned}", flush=True)
        evidence = EvidenceIndex(args.canonical_events, aligned)
        print(
            f"loaded canonical={evidence.canonical_row_count} aligned={evidence.aligned_row_count} "
            f"characters={len(evidence.character_counts)}",
            flush=True,
        )
    else:
        evidence = EvidenceIndex()
    return EventRepository(
        args.graph,
        args.events_dir,
        evidence=evidence,
        cache_size=getattr(args, "cache_trajectories", 8),
    )


def main() -> int:
    args = parse_args()
    validator = load_validator(args.event_schema)
    store = AnnotationStore(args.history)

    if args.command == "stats":
        repository = build_repository(args, load_evidence=False)
        payload = {
            "history": str(args.history),
            "history_record_count": store.record_count,
            "latest_annotation_count": len(store.latest),
            "global_status_counts": store.counts(),
            "load_errors": store.load_errors,
            "trajectories": [
                item for item in trajectory_summaries(repository, store) if item["annotated_count"] > 0
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export":
        repository = build_repository(args, load_evidence=False)
        summary = export_snapshot(repository, store, args.output_dir, validator)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not summary["validation_error_count"] and not summary["stale_count"] else 2

    repository = build_repository(args, load_evidence=True)
    server = AnnotationHTTPServer(
        (args.host, args.port),
        repository=repository,
        store=store,
        validator=validator,
        annotator=args.annotator,
        export_dir=args.export_dir,
        default_context=max(0, min(64, args.context)),
    )
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}/"
    print(f"LPM human annotator: {url}", flush=True)
    print(f"history: {args.history}", flush=True)
    print(f"export:  {args.export_dir}", flush=True)
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nstopping annotation server", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
