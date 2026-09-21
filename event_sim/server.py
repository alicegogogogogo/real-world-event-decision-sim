from __future__ import annotations

import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


SERVICE_NAME = "real-world-event-decision-sim"

EVENT_FIELDS = ("eventId", "organizationId", "type", "occurredAt", "payload")

_POSITIVE_INTEGER_TEXT = re.compile(r"[1-9][0-9]*")
_NON_NEGATIVE_INTEGER_TEXT = re.compile(r"(?:0|[1-9][0-9]*)")


class EventValidationError(ValueError):
    """The submitted event object fails the ledger's field rules."""


class EventLedger:
    """In-process store of events, keyed by eventId.

    Data lives only for the lifetime of this instance: a new server starts
    with an empty ledger.
    """

    def __init__(self) -> None:
        self._events: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add(
        self, event: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Insert an event or reconcile a replay.

        Returns ``(status, event)`` where status is ``"created"`` for a new
        eventId, ``"exists"`` for an identical replay, or ``"conflict"`` when
        the same eventId carries different fields.
        """
        event_id = event["eventId"]
        with self._lock:
            existing = self._events.get(event_id)
            if existing is None:
                stored = {field: event[field] for field in EVENT_FIELDS}
                self._events[event_id] = stored
                return "created", stored
            if existing == {field: event[field] for field in EVENT_FIELDS}:
                return "exists", existing
            return "conflict", existing

    def list_for_organization(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            events = [
                dict(event)
                for event in self._events.values()
                if event["organizationId"] == organization_id
            ]
        events.sort(key=lambda event: (event["occurredAt"], event["eventId"]))
        return events

    def aggregate(
        self,
        organization_id: str,
        event_type: str,
        window_size: int,
        start: int | None,
        end: int | None,
    ) -> list[dict[str, int]]:
        """Count matching events per fixed-width window.

        Windows start at 0 and are ``window_size`` wide; each window covers
        ``window_start <= occurredAt < window_end``. When ``start`` and ``end``
        are given, every window intersecting the closed range ``[start, end]``
        is returned (empty windows included); otherwise only windows that
        contain a matching event are returned.
        """
        counts: dict[int, int] = {}
        with self._lock:
            timestamps = sorted(
                event["occurredAt"]
                for event in self._events.values()
                if event["organizationId"] == organization_id
                and event["type"] == event_type
                and (start is None or start <= event["occurredAt"] <= end)
            )

        for occurred_at in timestamps:
            window_start = (occurred_at // window_size) * window_size
            counts[window_start] = counts.get(window_start, 0) + 1

        if start is None:
            return [
                {
                    "start": window_start,
                    "end": window_start + window_size,
                    "count": counts[window_start],
                }
                for window_start in sorted(counts)
            ]

        first_window = (start // window_size) * window_size
        last_window = (end // window_size) * window_size
        return [
            {
                "start": window_start,
                "end": window_start + window_size,
                "count": counts.get(window_start, 0),
            }
            for window_start in range(
                first_window, last_window + window_size, window_size
            )
        ]


def validate_event(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body against the five-field event contract."""
    if not isinstance(data, dict):
        raise EventValidationError("event body must be a JSON object")

    keys = set(data)
    expected = set(EVENT_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in ("eventId", "organizationId", "type"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")

    occurred_at = data["occurredAt"]
    # bool is a subclass of int; floats are explicitly rejected.
    if (
        not isinstance(occurred_at, int)
        or isinstance(occurred_at, bool)
        or isinstance(occurred_at, float)
        or occurred_at < 0
    ):
        raise EventValidationError("occurredAt must be a non-negative integer")

    if not isinstance(data["payload"], dict):
        raise EventValidationError("payload must be a JSON object")

    return {field: data[field] for field in EVENT_FIELDS}


def _organization_id_from_query(query: str) -> str:
    """Extract a single, non-empty organizationId from a query string."""
    values = parse_qs(query, keep_blank_values=True).get("organizationId")
    if not values or len(values) != 1:
        raise EventValidationError(
            "organizationId query parameter is required exactly once"
        )
    value = values[0]
    if not value.strip():
        raise EventValidationError("organizationId must be non-empty")
    return value


def _single_non_empty(values: list[str] | None, name: str) -> str:
    if not values or len(values) != 1:
        raise EventValidationError(f"{name} query parameter is required exactly once")
    value = values[0]
    if not value.strip():
        raise EventValidationError(f"{name} must be non-empty")
    return value


def _integer_text(values: list[str] | None, name: str, pattern: re.Pattern[str]) -> int:
    value = _single_non_empty(values, name)
    if not pattern.fullmatch(value):
        kind = "positive" if pattern is _POSITIVE_INTEGER_TEXT else "non-negative"
        raise EventValidationError(f"{name} must be a {kind} integer")
    return int(value)


def _aggregate_params_from_query(
    query: str,
) -> tuple[str, str, int, int | None, int | None]:
    """Validate the aggregate query string into typed parameter values."""
    params = parse_qs(query, keep_blank_values=True)
    organization_id = _single_non_empty(params.get("organizationId"), "organizationId")
    event_type = _single_non_empty(params.get("type"), "type")
    window_size = _integer_text(
        params.get("windowSize"), "windowSize", _POSITIVE_INTEGER_TEXT
    )

    has_from = "from" in params
    has_to = "to" in params
    if has_from != has_to:
        raise EventValidationError("from and to query parameters must be provided together")
    start = end = None
    if has_from:
        start = _integer_text(params.get("from"), "from", _NON_NEGATIVE_INTEGER_TEXT)
        end = _integer_text(params.get("to"), "to", _NON_NEGATIVE_INTEGER_TEXT)
        if start > end:
            raise EventValidationError("from must be less than or equal to to")

    return organization_id, event_type, window_size, start, end


class Handler(BaseHTTPRequestHandler):
    server_version = "EventSim/0.1"

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {"service": SERVICE_NAME, "status": "ok"},
            )
            return
        if parsed.path == "/events":
            self._list_events(parsed.query)
            return
        if parsed.path == "/events/aggregate":
            self._aggregate_events(parsed.query)
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": self.path},
        )

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        if parsed.path != "/events":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "path": self.path},
            )
            return

        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            self._write_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "unsupported_media_type"},
            )
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        raw_body = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_json"},
            )
            return

        try:
            event = validate_event(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return

        status, stored = self.server.ledger.add(event)  # type: ignore[attr-defined]
        if status == "created":
            self._write_json(HTTPStatus.CREATED, stored)
        elif status == "exists":
            self._write_json(HTTPStatus.OK, stored)
        else:
            self._write_json(
                HTTPStatus.CONFLICT,
                {"error": "event_id_conflict"},
            )

    def _list_events(self, query: str) -> None:
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return
        events = self.server.ledger.list_for_organization(  # type: ignore[attr-defined]
            organization_id
        )
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": organization_id, "events": events},
        )

    def _aggregate_events(self, query: str) -> None:
        try:
            (
                organization_id,
                event_type,
                window_size,
                start,
                end,
            ) = _aggregate_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return
        windows = self.server.ledger.aggregate(  # type: ignore[attr-defined]
            organization_id, event_type, window_size, start, end
        )
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": organization_id,
                "type": event_type,
                "windowSize": window_size,
                "from": start,
                "to": end,
                "windows": windows,
            },
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


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.ledger = EventLedger()  # type: ignore[attr-defined]
    return server


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = create_server(host, port)
    try:
        print(f"{SERVICE_NAME} listening on http://{host}:{server.server_port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
