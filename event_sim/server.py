from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


SERVICE_NAME = "real-world-event-decision-sim"

EVENT_FIELDS = ("eventId", "organizationId", "type", "occurredAt", "payload")
IDENTIFIER_FIELDS = ("eventId", "organizationId", "type")


class EventLedger:
    """In-process store for events; data lives only for the server's lifetime."""

    def __init__(self) -> None:
        self._events: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add(
        self, event: dict[str, Any]
    ) -> tuple[HTTPStatus, dict[str, Any] | None]:
        with self._lock:
            existing = self._events.get(event["eventId"])
            if existing is not None:
                if existing == event:
                    return HTTPStatus.OK, existing
                return HTTPStatus.CONFLICT, None
            stored = {field: event[field] for field in EVENT_FIELDS}
            self._events[stored["eventId"]] = stored
            return HTTPStatus.CREATED, stored

    def list_for(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            events = [
                dict(event)
                for event in self._events.values()
                if event["organizationId"] == organization_id
            ]
        return sorted(events, key=lambda event: (event["occurredAt"], event["eventId"]))


def validate_event(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != set(EVENT_FIELDS):
        return False
    for field in IDENTIFIER_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            return False
    occurred_at = data["occurredAt"]
    if isinstance(occurred_at, bool) or not isinstance(occurred_at, int):
        return False
    if occurred_at < 0:
        return False
    if not isinstance(data["payload"], dict):
        return False
    return True


def _handler_type() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "EventSim/0.1"

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            path = urlsplit(self.path)
            if path.path == "/health":
                self._write_json(
                    HTTPStatus.OK,
                    {"service": SERVICE_NAME, "status": "ok"},
                )
                return
            if path.path == "/events":
                self._list_events(path.query)
                return
            self._not_found()

        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            path = urlsplit(self.path)
            if path.path == "/events":
                self._create_event()
                return
            self._not_found()

        def _create_event(self) -> None:
            if self.headers.get_content_type() != "application/json":
                self._write_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": "unsupported_media_type"},
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            raw_body = self.rfile.read(max(length, 0))
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except ValueError:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_json"},
                )
                return
            if not validate_event(data):
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "validation_error"},
                )
                return
            status, event = self.server.ledger.add(data)  # type: ignore[attr-defined]
            if status is HTTPStatus.CONFLICT:
                self._write_json(
                    HTTPStatus.CONFLICT,
                    {"error": "event_id_conflict"},
                )
                return
            self._write_json(status, event)

        def _list_events(self, query: str) -> None:
            values = parse_qs(query, keep_blank_values=True).get("organizationId")
            if not values or len(values) != 1 or not values[0].strip():
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "validation_error"},
                )
                return
            organization_id = values[0]
            events = self.server.ledger.list_for(organization_id)  # type: ignore[attr-defined]
            self._write_json(
                HTTPStatus.OK,
                {"organizationId": organization_id, "events": events},
            )

        def _not_found(self) -> None:
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "path": self.path},
            )

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


class EventSimHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ledger = EventLedger()


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return EventSimHTTPServer((host, port), _handler_type())


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = create_server(host, port)
    try:
        print(f"{SERVICE_NAME} listening on http://{host}:{server.server_port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
