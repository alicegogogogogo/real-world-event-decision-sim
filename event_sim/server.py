from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


SERVICE_NAME = "real-world-event-decision-sim"

EVENT_FIELDS = ("eventId", "organizationId", "type", "occurredAt", "payload")

DECISION_REQUIRED_FIELDS = ("organizationId", "type", "windowSize", "threshold")
DECISION_ALLOWED_FIELDS = DECISION_REQUIRED_FIELDS + ("from", "to")


class EventValidationError(ValueError):
    """The submitted event object fails the ledger's field rules."""


class DecisionValidationError(ValueError):
    """The submitted decision request fails the evaluation field rules."""


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

    def occurred_at_values(
        self, organization_id: str, event_type: str
    ) -> list[int]:
        """Sorted occurredAt values of events matching organization and type."""
        with self._lock:
            values = [
                event["occurredAt"]
                for event in self._events.values()
                if event["organizationId"] == organization_id
                and event["type"] == event_type
            ]
        values.sort()
        return values


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


def _integer_text(value: str) -> int | None:
    """Parse ASCII decimal integer text; return None for anything else."""
    if not value or not value.isascii() or not value.isdigit():
        return None
    return int(value)


def _aggregate_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /events/aggregate query string.

    organizationId, type and windowSize are each required exactly once;
    from and to must be both absent or both present exactly once.
    """
    params = parse_qs(query, keep_blank_values=True)

    def single_text(name: str) -> str:
        values = params.get(name)
        if not values or len(values) != 1:
            raise EventValidationError(
                f"{name} query parameter is required exactly once"
            )
        value = values[0]
        if not value.strip():
            raise EventValidationError(f"{name} must be non-empty")
        return value

    organization_id = single_text("organizationId")
    event_type = single_text("type")

    window_size_text = single_text("windowSize")
    window_size = _integer_text(window_size_text)
    if window_size is None or window_size <= 0:
        raise EventValidationError("windowSize must be a positive integer")

    from_values = params.get("from")
    to_values = params.get("to")
    from_value: int | None = None
    to_value: int | None = None
    if from_values is not None or to_values is not None:
        if (
            not from_values
            or len(from_values) != 1
            or not to_values
            or len(to_values) != 1
        ):
            raise EventValidationError(
                "from and to must be omitted together or each appear exactly once"
            )
        from_value = _integer_text(from_values[0])
        to_value = _integer_text(to_values[0])
        if from_value is None or to_value is None:
            raise EventValidationError("from and to must be non-negative integers")
        if from_value > to_value:
            raise EventValidationError("from must be less than or equal to to")

    return {
        "organizationId": organization_id,
        "type": event_type,
        "windowSize": window_size,
        "from": from_value,
        "to": to_value,
    }


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_integer(value: Any) -> bool:
    # bool is a subclass of int; floats are explicitly rejected.
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and not isinstance(value, float)
        and value > 0
    )


def _non_negative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and not isinstance(value, float)
        and value >= 0
    )


def validate_decision(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body against the decision-request contract."""
    if not isinstance(data, dict):
        raise DecisionValidationError("decision body must be a JSON object")

    keys = set(data)
    allowed = set(DECISION_ALLOWED_FIELDS)
    required = set(DECISION_REQUIRED_FIELDS)
    missing = required - keys
    unknown = keys - allowed
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(sorted(unknown))}")
        raise DecisionValidationError("; ".join(detail))

    if not _non_empty_string(data["organizationId"]):
        raise DecisionValidationError("organizationId must be a non-empty string")
    if not _non_empty_string(data["type"]):
        raise DecisionValidationError("type must be a non-empty string")
    if not _positive_integer(data["windowSize"]):
        raise DecisionValidationError("windowSize must be a positive integer")
    if not _positive_integer(data["threshold"]):
        raise DecisionValidationError("threshold must be a positive integer")

    has_from = "from" in data
    has_to = "to" in data
    if has_from != has_to:
        raise DecisionValidationError(
            "from and to must be omitted together or supplied together"
        )
    from_value = None
    to_value = None
    if has_from:
        if not _non_negative_integer(data["from"]):
            raise DecisionValidationError("from must be a non-negative integer")
        if not _non_negative_integer(data["to"]):
            raise DecisionValidationError("to must be a non-negative integer")
        from_value = data["from"]
        to_value = data["to"]
        if from_value > to_value:
            raise DecisionValidationError("from must be less than or equal to to")

    return {
        "organizationId": data["organizationId"],
        "type": data["type"],
        "windowSize": data["windowSize"],
        "threshold": data["threshold"],
        "from": from_value,
        "to": to_value,
    }


def evaluate_decision(params: dict[str, Any], occurred: list[int]) -> dict[str, Any]:
    """Compute the peak window and resulting action over matching timestamps.

    ``occurred`` is a single consistent snapshot from the ledger; this
    function never touches or mutates ledger state.
    """
    window_size = params["windowSize"]
    from_value = params["from"]
    to_value = params["to"]

    counts: dict[int, int] = {}
    for timestamp in occurred:
        if from_value is not None and not (from_value <= timestamp <= to_value):
            continue
        start = (timestamp // window_size) * window_size
        counts[start] = counts.get(start, 0) + 1

    if from_value is None:
        # Only windows actually covered by matching events.
        starts = sorted(counts)
    else:
        # Every window intersecting the closed interval [from, to], kept so
        # empty windows remain available for auditing.
        first = (from_value // window_size) * window_size
        last = (to_value // window_size) * window_size
        starts = list(range(first, last + 1, window_size))

    peak_start = None
    peak_count = 0
    # starts is ascending, so the earliest start wins ties by only taking
    # strict improvements.
    for start in starts:
        count = counts.get(start, 0)
        if count > peak_count:
            peak_count = count
            peak_start = start

    action = "escalate" if peak_count >= params["threshold"] else "observe"
    return {
        "organizationId": params["organizationId"],
        "type": params["type"],
        "windowSize": window_size,
        "from": from_value,
        "to": to_value,
        "peakStart": peak_start,
        "peakCount": peak_count,
        "action": action,
    }


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
        if parsed.path == "/decisions/evaluate":
            self._evaluate_decision()
            return
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
            params = _aggregate_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return

        window_size = params["windowSize"]
        from_value = params["from"]
        to_value = params["to"]
        occurred = self.server.ledger.occurred_at_values(  # type: ignore[attr-defined]
            params["organizationId"], params["type"]
        )

        counts: dict[int, int] = {}
        for timestamp in occurred:
            if from_value is not None and not (from_value <= timestamp <= to_value):
                continue
            start = (timestamp // window_size) * window_size
            counts[start] = counts.get(start, 0) + 1

        if from_value is None:
            # Only windows actually covered by matching events.
            starts = sorted(counts)
        else:
            # Every window intersecting the closed interval [from, to].
            first = (from_value // window_size) * window_size
            last = (to_value // window_size) * window_size
            starts = list(range(first, last + 1, window_size))

        windows = [
            {"start": start, "end": start + window_size, "count": counts.get(start, 0)}
            for start in starts
        ]
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "type": params["type"],
                "windowSize": window_size,
                "from": from_value,
                "to": to_value,
                "windows": windows,
            },
        )

    def _evaluate_decision(self) -> None:
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
            params = validate_decision(data)
        except DecisionValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return

        # One locked read produces a consistent snapshot for this request;
        # the evaluation itself never writes back to the ledger.
        occurred = self.server.ledger.occurred_at_values(  # type: ignore[attr-defined]
            params["organizationId"], params["type"]
        )
        result = evaluate_decision(params, occurred)
        self._write_json(HTTPStatus.OK, result)

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
