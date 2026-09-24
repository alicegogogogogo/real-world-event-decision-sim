from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


SERVICE_NAME = "real-world-event-decision-sim"

EVENT_FIELDS = ("eventId", "organizationId", "type", "occurredAt", "payload")

DECISION_REQUIRED_FIELDS = ("organizationId", "type", "windowSize", "threshold")
DECISION_OPTIONAL_FIELDS = ("from", "to")
DECISION_FIELDS = DECISION_REQUIRED_FIELDS + DECISION_OPTIONAL_FIELDS

ALERT_REQUIRED_FIELDS = (
    "organizationId",
    "type",
    "windowSize",
    "threshold",
    "suppressionWindow",
)
ALERT_OPTIONAL_FIELDS = ("from", "to")
ALERT_FIELDS = ALERT_REQUIRED_FIELDS + ALERT_OPTIONAL_FIELDS

ALLOCATION_REQUIRED_FIELDS = ("organizationId", "demands", "resources")
DEMAND_FIELDS = ("demandId", "units", "priority")
RESOURCE_FIELDS = ("resourceId", "capacity")

RESERVATION_FIELDS = (
    "organizationId",
    "reservationId",
    "resourceId",
    "quantity",
    "capacity",
)

# Sentinel returned by Handler._json_request_body after it has already
# written the 415/400 error response.
_BODY_ERROR = object()


class EventValidationError(ValueError):
    """The submitted event object fails the ledger's field rules."""


def event_region(event: dict[str, Any]) -> str | None:
    """The region an event is attributed to, or ``None``.

    Only a non-empty string stored under the payload's ``region`` key
    attributes the event; a missing key, an empty string, or a non-string
    value means the event has no region.
    """
    value = event["payload"].get("region")
    if isinstance(value, str) and value:
        return value
    return None


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

    def list_for_region(
        self, organization_id: str, region: str
    ) -> list[dict[str, Any]]:
        """Events of one organization attributed to ``region``, sorted as
        the plain organization listing."""
        with self._lock:
            events = [
                dict(event)
                for event in self._events.values()
                if event["organizationId"] == organization_id
                and event_region(event) == region
            ]
        events.sort(key=lambda event: (event["occurredAt"], event["eventId"]))
        return events

    def occurred_at_values_for_region(
        self, organization_id: str, region: str, event_type: str
    ) -> list[int]:
        """Sorted occurredAt values matching organization, region, and type.

        Like ``occurred_at_values``, the values are copied under the lock so
        the returned list is a consistent snapshot.
        """
        with self._lock:
            values = [
                event["occurredAt"]
                for event in self._events.values()
                if event["organizationId"] == organization_id
                and event["type"] == event_type
                and event_region(event) == region
            ]
        values.sort()
        return values

    def occurred_at_values(
        self, organization_id: str, event_type: str
    ) -> list[int]:
        """Sorted occurredAt values of events matching organization and type.

        The matching values are copied while holding the lock, so the
        returned list is a consistent snapshot safe against concurrent writes.
        """
        with self._lock:
            values = [
                event["occurredAt"]
                for event in self._events.values()
                if event["organizationId"] == organization_id
                and event["type"] == event_type
            ]
        values.sort()
        return values

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a deep, locked copy of every stored event."""
        with self._lock:
            return {
                event_id: dict(event) for event_id, event in self._events.items()
            }

    def restore(self, state: dict[str, dict[str, Any]]) -> None:
        """Replace all stored events with a deep copy of ``state``."""
        with self._lock:
            self._events = {
                event_id: dict(event) for event_id, event in state.items()
            }


class ReservationInventory:
    """In-process reservation ledger with per-resource capacity control.

    Resource capacities are fixed by the first reservation that names the
    resource and are never rewritten. The balance check and the deduction
    happen inside a single lock, so concurrent requests can neither oversell
    a resource nor lose each other's writes. State lives only for the
    lifetime of this instance.
    """

    def __init__(self) -> None:
        self._capacities: dict[str, int] = {}
        self._reservations: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _occupied_locked(self, resource_id: str) -> int:
        return sum(
            record["quantity"]
            for record in self._reservations.values()
            if record["resourceId"] == resource_id
        )

    def _view_locked(self, record: dict[str, Any]) -> dict[str, Any]:
        capacity = self._capacities[record["resourceId"]]
        occupied = self._occupied_locked(record["resourceId"])
        return {
            "organizationId": record["organizationId"],
            "reservationId": record["reservationId"],
            "resourceId": record["resourceId"],
            "quantity": record["quantity"],
            "capacity": capacity,
            "occupied": occupied,
            "remaining": capacity - occupied,
        }

    def reserve(
        self, reservation: dict[str, Any]
    ) -> tuple[str, dict[str, Any] | None]:
        """Commit a reservation or reconcile a replay, atomically.

        Returns ``(status, view)`` where status is ``"created"`` for a new
        reservationId, ``"exists"`` for an identical replay (nothing is
        counted twice), ``"reservation_conflict"`` when the reservationId
        exists with different fields, ``"capacity_conflict"`` when the
        resource's recorded capacity differs from the declared one, or
        ``"capacity_exceeded"`` when the remaining balance cannot cover the
        quantity. Only ``"created"`` mutates state; the view carries the
        post-commit occupied and remaining balances.
        """
        with self._lock:
            existing = self._reservations.get(reservation["reservationId"])
            if existing is not None:
                if all(
                    existing[field] == reservation[field]
                    for field in RESERVATION_FIELDS
                ):
                    return "exists", self._view_locked(existing)
                return "reservation_conflict", None

            resource_id = reservation["resourceId"]
            recorded = self._capacities.get(resource_id)
            if recorded is not None and recorded != reservation["capacity"]:
                return "capacity_conflict", None
            capacity = recorded if recorded is not None else reservation["capacity"]

            occupied = self._occupied_locked(resource_id)
            if reservation["quantity"] > capacity - occupied:
                return "capacity_exceeded", None

            stored = {field: reservation[field] for field in RESERVATION_FIELDS}
            self._capacities[resource_id] = capacity
            self._reservations[stored["reservationId"]] = stored
            return "created", self._view_locked(stored)

    def list_for_organization(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            views = [
                self._view_locked(record)
                for record in self._reservations.values()
                if record["organizationId"] == organization_id
            ]
        views.sort(key=lambda view: (view["resourceId"], view["reservationId"]))
        return views

    def capacity_count(self) -> int:
        with self._lock:
            return len(self._capacities)

    def reservation_count(self) -> int:
        with self._lock:
            return len(self._reservations)

    def snapshot(self) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
        """Return deep, locked copies of capacities and reservations."""
        with self._lock:
            capacities = dict(self._capacities)
            reservations = {
                reservation_id: dict(record)
                for reservation_id, record in self._reservations.items()
            }
        return capacities, reservations

    def restore(
        self,
        capacities: dict[str, int],
        reservations: dict[str, dict[str, Any]],
    ) -> None:
        """Replace capacities and reservations with deep copies."""
        with self._lock:
            self._capacities = dict(capacities)
            self._reservations = {
                reservation_id: dict(record)
                for reservation_id, record in reservations.items()
            }


class Snapshot:
    """An immutable, deep-copied capture of one main-state point in time.

    Holds every event plus reservation capacities and reservations.
    """

    def __init__(
        self,
        snapshot_id: str,
        events: dict[str, dict[str, Any]],
        capacities: dict[str, int],
        reservations: dict[str, dict[str, Any]],
    ) -> None:
        self.snapshot_id = snapshot_id
        self._events = {
            event_id: dict(event) for event_id, event in events.items()
        }
        self._capacities = dict(capacities)
        self._reservations = {
            reservation_id: dict(record)
            for reservation_id, record in reservations.items()
        }

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def capacity_count(self) -> int:
        return len(self._capacities)

    @property
    def reservation_count(self) -> int:
        return len(self._reservations)

    def materialize(self) -> tuple[EventLedger, ReservationInventory]:
        """Build fresh ledger and inventory instances from this snapshot."""
        ledger = EventLedger()
        ledger.restore(self._events)
        reservations = ReservationInventory()
        reservations.restore(self._capacities, self._reservations)
        return ledger, reservations


class SnapshotStore:
    """In-process registry of named snapshots keyed by snapshotId."""

    def __init__(self) -> None:
        self._snapshots: dict[str, Snapshot] = {}
        self._lock = threading.Lock()

    def create(
        self,
        snapshot_id: str,
        ledger: EventLedger,
        reservations: ReservationInventory,
    ) -> tuple[str, Snapshot]:
        """Capture ``ledger`` and ``reservations`` under ``snapshot_id``.

        Returns ``("created", snapshot)`` or ``("conflict", snapshot)`` when
        the name already exists; an existing snapshot is never replaced.
        """
        with self._lock:
            existing = self._snapshots.get(snapshot_id)
            if existing is not None:
                return "conflict", existing
            snapshot = Snapshot(
                snapshot_id,
                ledger.snapshot(),
                *reservations.snapshot(),
            )
            self._snapshots[snapshot_id] = snapshot
            return "created", snapshot

    def get(self, snapshot_id: str) -> Snapshot | None:
        with self._lock:
            return self._snapshots.get(snapshot_id)

    def summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            snapshots = list(self._snapshots.values())
        snapshots.sort(key=lambda snapshot: snapshot.snapshot_id)
        return [
            {
                "snapshotId": snapshot.snapshot_id,
                "events": snapshot.event_count,
                "resources": snapshot.capacity_count,
                "reservations": snapshot.reservation_count,
            }
            for snapshot in snapshots
        ]


class Branch:
    """An isolated fork of the main service state built from one snapshot.

    A branch owns its own ledger and reservation inventory; writes land only
    in the branch and never touch the main service or any other branch.
    """

    def __init__(self, branch_id: str, snapshot_id: str, snapshot: Snapshot) -> None:
        self.branch_id = branch_id
        self.snapshot_id = snapshot_id
        self.ledger, self.reservations = snapshot.materialize()

    def summary(self) -> dict[str, Any]:
        return {
            "branchId": self.branch_id,
            "snapshotId": self.snapshot_id,
            "events": self.ledger.count(),
            "resources": self.reservations.capacity_count(),
            "reservations": self.reservations.reservation_count(),
        }


class BranchStore:
    """In-process registry of isolated branches keyed by branchId."""

    def __init__(self) -> None:
        self._branches: dict[str, Branch] = {}
        self._lock = threading.Lock()

    def create(self, branch_id: str, snapshot: Snapshot) -> tuple[str, Branch]:
        """Fork ``snapshot`` under ``branch_id``.

        Returns ``("created", branch)`` or ``("conflict", branch)`` when a
        branch with that name already exists.
        """
        with self._lock:
            existing = self._branches.get(branch_id)
            if existing is not None:
                return "conflict", existing
            branch = Branch(branch_id, snapshot.snapshot_id, snapshot)
            self._branches[branch_id] = branch
            return "created", branch

    def get(self, branch_id: str) -> Branch | None:
        with self._lock:
            return self._branches.get(branch_id)


class AlertStore:
    """In-process alert ledger with per-organization/type suppression.

    An alert is raised at a peak window start when the peak count reaches the
    request threshold. A repeat peak within the suppression window of a prior
    alert's peak start increments that alert's suppressed count instead of
    creating a new alert; a peak at or beyond the window opens a new alert.
    Suppression checks and creation happen under one lock, so concurrent
    evaluations cannot create or suppress against a stale view. State lives
    only for the lifetime of this instance and is cleared on restart.
    """

    def __init__(self) -> None:
        self._alerts: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def record(
        self,
        organization_id: str,
        event_type: str,
        peak_start: int,
        threshold: int,
        suppression_window: int,
    ) -> tuple[str, dict[str, Any]]:
        """Raise or suppress an alert for an organization/type peak.

        Returns ``(status, alert)`` where status is ``"escalate"`` for a new
        alert or ``"suppress"`` when the peak falls inside the suppression
        window of the most recent prior alert. A suppressed peak increments
        that alert's ``suppressedCount``.
        """
        with self._lock:
            by_type = self._alerts.setdefault(organization_id, {})
            alerts = by_type.setdefault(event_type, [])
            if alerts:
                latest = alerts[-1]
                if peak_start - latest["peakStart"] < suppression_window:
                    latest["suppressedCount"] += 1
                    return "suppress", latest
            self._counter += 1
            alert = {
                "alertId": f"alert-{self._counter}",
                "organizationId": organization_id,
                "type": event_type,
                "peakStart": peak_start,
                "threshold": threshold,
                "suppressedCount": 0,
            }
            alerts.append(alert)
            return "escalate", alert

    def list_for_organization(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            by_type = self._alerts.get(organization_id, {})
            alerts = [dict(alert) for group in by_type.values() for alert in group]
        # Peak start ascending, then alertId in Unicode code-point order.
        alerts.sort(key=lambda alert: (alert["peakStart"], alert["alertId"]))
        return alerts


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


def _single_query_text(params: dict[str, list[str]], name: str) -> str:
    """Extract one query parameter that must appear exactly once, non-blank."""
    values = params.get(name)
    if not values or len(values) != 1:
        raise EventValidationError(
            f"{name} query parameter is required exactly once"
        )
    value = values[0]
    if not value.strip():
        raise EventValidationError(f"{name} must be non-empty")
    return value


def _region_params_from_query(query: str) -> tuple[str, str]:
    """Validate the /events/region query string.

    organizationId and region are each required exactly once and must be
    non-empty; unrelated parameters are ignored, as in the plain listing.
    """
    params = parse_qs(query, keep_blank_values=True)
    organization_id = _single_query_text(params, "organizationId")
    region = _single_query_text(params, "region")
    return organization_id, region


def _region_aggregate_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /events/region/aggregate query string.

    organizationId, region, type and windowSize are each required exactly
    once; from and to must be both absent or both present exactly once.
    """
    params = parse_qs(query, keep_blank_values=True)

    organization_id = _single_query_text(params, "organizationId")
    region = _single_query_text(params, "region")
    event_type = _single_query_text(params, "type")

    window_size_text = _single_query_text(params, "windowSize")
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
        "region": region,
        "type": event_type,
        "windowSize": window_size,
        "from": from_value,
        "to": to_value,
    }


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


def _is_positive_integer(value: Any) -> bool:
    """True only for plain positive ints (booleans and floats rejected)."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and not isinstance(value, float)
        and value > 0
    )


def _is_non_negative_integer(value: Any) -> bool:
    """True only for plain non-negative ints (booleans and floats rejected)."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and not isinstance(value, float)
        and value >= 0
    )


def validate_decision_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /decisions/evaluate.

    Required fields: organizationId, type, windowSize, threshold.
    Optional fields: from and to, which must appear together. No other
    fields are allowed.
    """
    if not isinstance(data, dict):
        raise EventValidationError("decision body must be a JSON object")

    keys = set(data)
    required = set(DECISION_REQUIRED_FIELDS)
    allowed = set(DECISION_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "type"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")

    for field in ("windowSize", "threshold"):
        if not _is_positive_integer(data[field]):
            raise EventValidationError(f"{field} must be a positive integer")

    # from and to are either both present or both omitted.
    if "from" in keys or "to" in keys:
        if "from" not in keys or "to" not in keys:
            raise EventValidationError(
                "from and to must be omitted together or both be present"
            )
        from_value = data["from"]
        to_value = data["to"]
        if not _is_non_negative_integer(from_value) or not _is_non_negative_integer(
            to_value
        ):
            raise EventValidationError("from and to must be non-negative integers")
        if from_value > to_value:
            raise EventValidationError("from must be less than or equal to to")
    else:
        from_value = None
        to_value = None

    return {
        "organizationId": data["organizationId"],
        "type": data["type"],
        "windowSize": data["windowSize"],
        "threshold": data["threshold"],
        "from": from_value,
        "to": to_value,
    }


def evaluate_decision(occurred: list[int], params: dict[str, Any]) -> dict[str, Any]:
    """Compute the peak window and resulting action from matching timestamps.

    ``occurred`` is a consistent snapshot (already filtered by organization
    and type). Window boundaries are left-closed/right-open and start at
    zero. Ties on the peak count resolve to the earliest window start.
    """
    window_size = params["windowSize"]
    threshold = params["threshold"]
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
        # Every window intersecting the closed interval [from, to], kept
        # (including empty ones) so the audit can reconcile the result.
        first = (from_value // window_size) * window_size
        last = (to_value // window_size) * window_size
        starts = list(range(first, last + 1, window_size))

    peak_start = None
    peak_count = 0
    for start in starts:
        count = counts.get(start, 0)
        if count > peak_count:
            peak_count = count
            peak_start = start

    action = "escalate" if peak_count >= threshold else "observe"
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


def validate_alert_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /alerts/evaluate.

    Required fields: organizationId, type, windowSize, threshold, and
    suppressionWindow. Optional fields: from and to, which must appear
    together. No other fields are allowed.
    """
    if not isinstance(data, dict):
        raise EventValidationError("alert body must be a JSON object")

    keys = set(data)
    required = set(ALERT_REQUIRED_FIELDS)
    allowed = set(ALERT_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "type"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")

    for field in ("windowSize", "threshold", "suppressionWindow"):
        if not _is_positive_integer(data[field]):
            raise EventValidationError(f"{field} must be a positive integer")

    # from and to are either both present or both omitted.
    if "from" in keys or "to" in keys:
        if "from" not in keys or "to" not in keys:
            raise EventValidationError(
                "from and to must be omitted together or both be present"
            )
        from_value = data["from"]
        to_value = data["to"]
        if not _is_non_negative_integer(from_value) or not _is_non_negative_integer(
            to_value
        ):
            raise EventValidationError("from and to must be non-negative integers")
        if from_value > to_value:
            raise EventValidationError("from must be less than or equal to to")
    else:
        from_value = None
        to_value = None

    return {
        "organizationId": data["organizationId"],
        "type": data["type"],
        "windowSize": data["windowSize"],
        "threshold": data["threshold"],
        "suppressionWindow": data["suppressionWindow"],
        "from": from_value,
        "to": to_value,
    }


def validate_allocation_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /decisions/allocate.

    Required fields: organizationId, demands, resources, and no others.
    Each demand/resource element must be an object with exactly its fixed
    fields; identifiers must be unique within the request and non-blank.
    """
    if not isinstance(data, dict):
        raise EventValidationError("allocation body must be a JSON object")

    keys = set(data)
    required = set(ALLOCATION_REQUIRED_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    organization_id = data["organizationId"]
    if not isinstance(organization_id, str) or not organization_id.strip():
        raise EventValidationError("organizationId must be a non-empty string")

    raw_demands = data["demands"]
    raw_resources = data["resources"]
    if not isinstance(raw_demands, list) or not isinstance(raw_resources, list):
        raise EventValidationError("demands and resources must be arrays")

    demands: list[dict[str, Any]] = []
    seen_demand_ids: set[str] = set()
    for index, element in enumerate(raw_demands):
        if not isinstance(element, dict):
            raise EventValidationError(f"demands[{index}] must be a JSON object")
        element_keys = set(element)
        expected = set(DEMAND_FIELDS)
        if element_keys != expected:
            element_missing = sorted(expected - element_keys)
            element_unknown = sorted(element_keys - expected)
            element_detail = []
            if element_missing:
                element_detail.append(
                    f"missing fields: {', '.join(element_missing)}"
                )
            if element_unknown:
                element_detail.append(
                    f"unexpected fields: {', '.join(element_unknown)}"
                )
            raise EventValidationError(
                f"demands[{index}]: {'; '.join(element_detail)}"
            )
        demand_id = element["demandId"]
        if not isinstance(demand_id, str) or not demand_id.strip():
            raise EventValidationError(
                f"demands[{index}].demandId must be a non-empty string"
            )
        if demand_id in seen_demand_ids:
            raise EventValidationError(
                f"duplicate demandId: {demand_id}"
            )
        if not _is_positive_integer(element["units"]):
            raise EventValidationError(
                f"demands[{index}].units must be a positive integer"
            )
        if not _is_non_negative_integer(element["priority"]):
            raise EventValidationError(
                f"demands[{index}].priority must be a non-negative integer"
            )
        seen_demand_ids.add(demand_id)
        demands.append(
            {
                "demandId": demand_id,
                "units": element["units"],
                "priority": element["priority"],
            }
        )

    resources: dict[str, int] = {}
    for index, element in enumerate(raw_resources):
        if not isinstance(element, dict):
            raise EventValidationError(f"resources[{index}] must be a JSON object")
        element_keys = set(element)
        expected = set(RESOURCE_FIELDS)
        if element_keys != expected:
            element_missing = sorted(expected - element_keys)
            element_unknown = sorted(element_keys - expected)
            element_detail = []
            if element_missing:
                element_detail.append(
                    f"missing fields: {', '.join(element_missing)}"
                )
            if element_unknown:
                element_detail.append(
                    f"unexpected fields: {', '.join(element_unknown)}"
                )
            raise EventValidationError(
                f"resources[{index}]: {'; '.join(element_detail)}"
            )
        resource_id = element["resourceId"]
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise EventValidationError(
                f"resources[{index}].resourceId must be a non-empty string"
            )
        if resource_id in resources:
            raise EventValidationError(
                f"duplicate resourceId: {resource_id}"
            )
        if not _is_positive_integer(element["capacity"]):
            raise EventValidationError(
                f"resources[{index}].capacity must be a positive integer"
            )
        resources[resource_id] = element["capacity"]

    return {
        "organizationId": organization_id,
        "demands": demands,
        "resources": resources,
    }


def validate_reservation_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /reservations.

    Required fields: organizationId, reservationId, resourceId, quantity,
    capacity — and no others. Identifiers must be non-empty strings;
    quantity and capacity must be positive integers.
    """
    if not isinstance(data, dict):
        raise EventValidationError("reservation body must be a JSON object")

    keys = set(data)
    expected = set(RESERVATION_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "reservationId", "resourceId"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")

    for field in ("quantity", "capacity"):
        if not _is_positive_integer(data[field]):
            raise EventValidationError(f"{field} must be a positive integer")

    return {field: data[field] for field in RESERVATION_FIELDS}


def _validate_single_identifier_body(
    data: Any, field: str, *, body_kind: str
) -> str:
    """Validate a body containing exactly one non-empty string field."""
    if not isinstance(data, dict):
        raise EventValidationError(f"{body_kind} body must be a JSON object")

    keys = set(data)
    expected = {field}
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise EventValidationError(f"{field} must be a non-empty string")
    return value


def validate_snapshot_request(data: Any) -> str:
    """Validate a decoded JSON body for POST /snapshots."""
    return _validate_single_identifier_body(
        data, "snapshotId", body_kind="snapshot"
    )


def validate_branch_request(data: Any) -> tuple[str, str]:
    """Validate a decoded JSON body for POST /branches."""
    if not isinstance(data, dict):
        raise EventValidationError("branch body must be a JSON object")

    keys = set(data)
    expected = {"branchId", "snapshotId"}
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in ("branchId", "snapshotId"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return data["branchId"], data["snapshotId"]


def plan_allocation(params: dict[str, Any]) -> dict[str, Any]:
    """Allocate whole demands to resources by deterministic rules.

    Demands are handled by priority descending, then demandId in Unicode
    code-point order. A demand goes wholly to the lexicographically smallest
    resourceId with enough remaining capacity; otherwise it is unassigned.
    """
    demands = params["demands"]
    remaining = dict(params["resources"])

    ordered_demands = sorted(
        demands, key=lambda demand: (-demand["priority"], demand["demandId"])
    )
    ordered_resources = sorted(remaining)

    assignments: list[dict[str, Any]] = []
    unassigned: list[str] = []
    total_units = 0
    for demand in ordered_demands:
        demand_id = demand["demandId"]
        units = demand["units"]
        chosen = None
        for resource_id in ordered_resources:
            if remaining[resource_id] >= units:
                chosen = resource_id
                break
        if chosen is None:
            unassigned.append(demand_id)
            continue
        remaining[chosen] -= units
        assignments.append(
            {"demandId": demand_id, "resourceId": chosen, "units": units}
        )
        total_units += units

    unassigned.sort()
    return {
        "organizationId": params["organizationId"],
        "assignments": assignments,
        "unassigned": unassigned,
        "totalUnits": total_units,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "EventSim/0.1"

    # ------------------------------------------------------------------ GET

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parsed.query

        if path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {"service": SERVICE_NAME, "status": "ok"},
            )
            return

        if path == "/snapshots":
            self._list_snapshots()
            return

        if path.startswith("/branches/"):
            remainder = path[len("/branches/"):]
            segments = remainder.split("/")
            if len(segments) == 1 and segments[0]:
                self._get_branch(unquote(segments[0]))
                return
            if len(segments) >= 2 and segments[0]:
                branch_id = unquote(segments[0])
                branch = self.server.branches.get(  # type: ignore[attr-defined]
                    branch_id
                )
                if branch is None:
                    self._branch_not_found()
                    return
                if len(segments) == 2:
                    if segments[1] == "events":
                        self._list_events(branch.ledger, query, newline=True)
                        return
                    if segments[1] == "reservations":
                        self._list_reservations(
                            branch.reservations, query, newline=True
                        )
                        return
                if (
                    len(segments) == 3
                    and segments[1] == "events"
                    and segments[2] == "aggregate"
                ):
                    self._aggregate_events(branch.ledger, query, newline=True)
                    return
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "path": self.path},
                newline=True,
            )
            return

        if path == "/events":
            self._list_events(self.server.ledger, query)  # type: ignore[attr-defined]
            return
        if path == "/events/aggregate":
            self._aggregate_events(
                self.server.ledger,  # type: ignore[attr-defined]
                query,
            )
            return
        if path == "/events/region":
            self._list_events_by_region(
                self.server.ledger,  # type: ignore[attr-defined]
                query,
            )
            return
        if path == "/events/region/aggregate":
            self._aggregate_events_by_region(
                self.server.ledger,  # type: ignore[attr-defined]
                query,
            )
            return
        if path == "/reservations":
            self._list_reservations(
                self.server.reservations,  # type: ignore[attr-defined]
                query,
                newline=True,
            )
            return
        if path == "/alerts":
            self._list_alerts(query)
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": self.path},
        )

    # ----------------------------------------------------------------- POST

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        path = parsed.path

        if path == "/snapshots":
            self._create_snapshot()
            return
        if path == "/branches":
            self._create_branch()
            return

        if path.startswith("/branches/"):
            remainder = path[len("/branches/"):]
            segments = remainder.split("/")
            if len(segments) >= 2 and segments[0]:
                branch_id = unquote(segments[0])
                branch = self.server.branches.get(  # type: ignore[attr-defined]
                    branch_id
                )
                if branch is None:
                    self._branch_not_found()
                    return
                if len(segments) == 2:
                    if segments[1] == "events":
                        self._create_event(branch.ledger, newline=True)
                        return
                    if segments[1] == "reservations":
                        self._create_reservation(branch.reservations, newline=True)
                        return
                if (
                    len(segments) == 3
                    and segments[1] == "decisions"
                    and segments[2] == "evaluate"
                ):
                    self._evaluate_decision(branch.ledger, newline=True)
                    return
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "path": self.path},
                newline=True,
            )
            return

        if path == "/events":
            self._create_event(self.server.ledger)  # type: ignore[attr-defined]
            return
        if path == "/reservations":
            self._create_reservation(
                self.server.reservations,  # type: ignore[attr-defined]
                newline=True,
            )
            return
        if path == "/decisions/evaluate":
            self._evaluate_decision(self.server.ledger)  # type: ignore[attr-defined]
            return
        if path == "/alerts/evaluate":
            self._evaluate_alert()
            return
        if path == "/decisions/allocate":
            self._allocate_decision()
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": self.path},
        )

    # ------------------------------------------------------------- snapshots

    def _create_snapshot(self) -> None:
        data = self._json_request_body()
        if data is _BODY_ERROR:
            return

        try:
            snapshot_id = validate_snapshot_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return

        status, snapshot = self.server.snapshots.create(  # type: ignore[attr-defined]
            snapshot_id,
            self.server.ledger,  # type: ignore[attr-defined]
            self.server.reservations,  # type: ignore[attr-defined]
        )
        if status == "created":
            self._write_json(
                HTTPStatus.CREATED,
                {
                    "snapshotId": snapshot_id,
                    "events": snapshot.event_count,
                    "resources": snapshot.capacity_count,
                    "reservations": snapshot.reservation_count,
                },
            )
        else:
            self._write_json(
                HTTPStatus.CONFLICT,
                {"error": "snapshot_conflict"},
            )

    def _list_snapshots(self) -> None:
        self._write_json(
            HTTPStatus.OK,
            {"snapshots": self.server.snapshots.summaries()},  # type: ignore[attr-defined]
        )

    # --------------------------------------------------------------- branches

    def _create_branch(self) -> None:
        data = self._json_request_body()
        if data is _BODY_ERROR:
            return

        try:
            branch_id, snapshot_id = validate_branch_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return

        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "snapshot_not_found"},
            )
            return

        status, branch = self.server.branches.create(  # type: ignore[attr-defined]
            branch_id, snapshot
        )
        if status == "created":
            self._write_json(HTTPStatus.CREATED, branch.summary())
        else:
            self._write_json(
                HTTPStatus.CONFLICT,
                {"error": "branch_conflict"},
            )

    def _get_branch(self, branch_id: str) -> None:
        branch = self.server.branches.get(branch_id)  # type: ignore[attr-defined]
        if branch is None:
            self._branch_not_found()
            return
        self._write_json(HTTPStatus.OK, branch.summary())

    def _branch_not_found(self) -> None:
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "branch_not_found"},
            newline=True,
        )

    # ----------------------------------------------------------------- events

    def _create_event(self, ledger: EventLedger, *, newline: bool = False) -> None:
        data = self._json_request_body(newline=newline)
        if data is _BODY_ERROR:
            return

        try:
            event = validate_event(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return

        status, stored = ledger.add(event)
        if status == "created":
            self._write_json(HTTPStatus.CREATED, stored, newline=newline)
        elif status == "exists":
            self._write_json(HTTPStatus.OK, stored, newline=newline)
        else:
            self._write_json(
                HTTPStatus.CONFLICT,
                {"error": "event_id_conflict"},
                newline=newline,
            )

    def _list_events(
        self, ledger: EventLedger, query: str, *, newline: bool = False
    ) -> None:
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return
        events = ledger.list_for_organization(organization_id)
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": organization_id, "events": events},
            newline=newline,
        )

    def _aggregate_events(
        self, ledger: EventLedger, query: str, *, newline: bool = False
    ) -> None:
        try:
            params = _aggregate_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return

        window_size = params["windowSize"]
        from_value = params["from"]
        to_value = params["to"]
        occurred = ledger.occurred_at_values(
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
            newline=newline,
        )

    # ------------------------------------------------------------ event regions

    def _list_events_by_region(self, ledger: EventLedger, query: str) -> None:
        try:
            organization_id, region = _region_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return
        # Read-only: the ledger is only read, never mutated, and an unknown
        # region simply matches nothing.
        events = ledger.list_for_region(organization_id, region)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": organization_id,
                "region": region,
                "events": events,
            },
            newline=True,
        )

    def _aggregate_events_by_region(self, ledger: EventLedger, query: str) -> None:
        try:
            params = _region_aggregate_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        window_size = params["windowSize"]
        from_value = params["from"]
        to_value = params["to"]
        # Read-only: counts come from a single locked snapshot of the
        # matching timestamps; the ledger is never mutated.
        occurred = ledger.occurred_at_values_for_region(
            params["organizationId"], params["region"], params["type"]
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
                "region": params["region"],
                "type": params["type"],
                "windowSize": window_size,
                "from": from_value,
                "to": to_value,
                "windows": windows,
            },
            newline=True,
        )

    # --------------------------------------------------------------- decisions

    def _evaluate_decision(
        self, ledger: EventLedger, *, newline: bool = False
    ) -> None:
        data = self._json_request_body(newline=newline)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_decision_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return

        # Read-only: the ledger is never mutated by a decision request, and
        # the snapshot is taken in a single locked copy for consistency.
        occurred = ledger.occurred_at_values(
            params["organizationId"], params["type"]
        )
        result = evaluate_decision(occurred, params)
        self._write_json(HTTPStatus.OK, result, newline=newline)

    def _allocate_decision(self) -> None:
        data = self._json_request_body()
        if data is _BODY_ERROR:
            return

        try:
            params = validate_allocation_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return

        # Read-only and stateless: the plan is computed from the request body
        # alone, so identical submissions and concurrent requests never
        # interfere with each other or with the ledger.
        result = plan_allocation(params)
        self._write_json(HTTPStatus.OK, result)

    # ------------------------------------------------------------------ alerts

    def _evaluate_alert(self) -> None:
        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_alert_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        # The ledger is never mutated by an alert request; the peak is read
        # from a single locked snapshot of the matching events.
        occurred = self.server.ledger.occurred_at_values(  # type: ignore[attr-defined]
            params["organizationId"], params["type"]
        )
        peak = evaluate_decision(occurred, params)
        peak_start = peak["peakStart"]
        peak_count = peak["peakCount"]

        if peak_count < params["threshold"]:
            alert_id = None
            suppressed_count = None
            action = "observe"
        else:
            # The suppression decision and the create/commit run in one lock,
            # so concurrent threshold hits cannot both open an alert.
            action, alert = self.server.alerts.record(  # type: ignore[attr-defined]
                params["organizationId"],
                params["type"],
                peak_start,
                params["threshold"],
                params["suppressionWindow"],
            )
            alert_id = alert["alertId"]
            suppressed_count = alert["suppressedCount"]

        result = {
            "organizationId": params["organizationId"],
            "type": params["type"],
            "windowSize": params["windowSize"],
            "threshold": params["threshold"],
            "suppressionWindow": params["suppressionWindow"],
            "from": params["from"],
            "to": params["to"],
            "peakStart": peak_start,
            "peakCount": peak_count,
            "action": action,
            "alertId": alert_id,
            "suppressedCount": suppressed_count,
        }
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _list_alerts(self, query: str) -> None:
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return
        alerts = self.server.alerts.list_for_organization(  # type: ignore[attr-defined]
            organization_id
        )
        entries = [
            {
                "alertId": alert["alertId"],
                "type": alert["type"],
                "peakStart": alert["peakStart"],
                "threshold": alert["threshold"],
                "suppressedCount": alert["suppressedCount"],
            }
            for alert in alerts
        ]
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": organization_id, "alerts": entries},
            newline=True,
        )

    # ----------------------------------------------------------- reservations

    def _create_reservation(
        self,
        reservations: ReservationInventory,
        *,
        newline: bool = False,
    ) -> None:
        data = self._json_request_body(newline=newline)
        if data is _BODY_ERROR:
            return

        try:
            reservation = validate_reservation_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return

        status, view = reservations.reserve(reservation)
        if status == "created":
            self._write_json(HTTPStatus.CREATED, view, newline=newline)
        elif status == "exists":
            self._write_json(HTTPStatus.OK, view, newline=newline)
        else:
            self._write_json(HTTPStatus.CONFLICT, {"error": status}, newline=newline)

    def _list_reservations(
        self,
        reservations: ReservationInventory,
        query: str,
        *,
        newline: bool = False,
    ) -> None:
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return
        views = reservations.list_for_organization(organization_id)
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": organization_id, "reservations": views},
            newline=newline,
        )

    # ------------------------------------------------------------------ wiring

    def _json_request_body(self, *, newline: bool = False) -> Any:
        """Validate the media type and decode the request body as JSON.

        On failure the 415/400 response is written here and the ``_BODY_ERROR``
        sentinel is returned; callers must return immediately.
        """
        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            self._write_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "unsupported_media_type"},
                newline=newline,
            )
            return _BODY_ERROR

        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        raw_body = self.rfile.read(length) if length > 0 else b""
        try:
            return json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_json"},
                newline=newline,
            )
            return _BODY_ERROR

    def _write_json(
        self, status: HTTPStatus, payload: dict[str, Any], *, newline: bool = False
    ) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        if newline:
            body += b"\n"
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
    server.reservations = ReservationInventory()  # type: ignore[attr-defined]
    server.snapshots = SnapshotStore()  # type: ignore[attr-defined]
    server.branches = BranchStore()  # type: ignore[attr-defined]
    server.alerts = AlertStore()  # type: ignore[attr-defined]
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
