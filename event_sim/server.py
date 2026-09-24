from __future__ import annotations

import copy
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

SNAPSHOT_FIELDS = ("snapshotId",)
BRANCH_FIELDS = ("branchId", "snapshotId")

# Sentinel returned by Handler._json_request_body after it has already
# written the 415/400 error response.
_BODY_ERROR = object()


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

    def clone(self) -> "EventLedger":
        """Deep-copy the ledger so the copy shares no mutable state."""
        with self._lock:
            events = copy.deepcopy(self._events)
        ledger = EventLedger()
        ledger._events = events
        return ledger


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

    def count(self) -> int:
        with self._lock:
            return len(self._reservations)

    def clone(self) -> "ReservationInventory":
        """Deep-copy capacities and reservations; the copy is independent."""
        with self._lock:
            capacities = dict(self._capacities)
            reservations = copy.deepcopy(self._reservations)
        inventory = ReservationInventory()
        inventory._capacities = capacities
        inventory._reservations = reservations
        return inventory


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


class IsolatedState:
    """A self-contained copy of the event ledger and reservation inventory.

    Snapshots and branches each hold one of these; nothing inside is shared
    with the main service or with any other snapshot or branch.
    """

    def __init__(
        self,
        ledger: EventLedger,
        reservations: ReservationInventory,
        snapshot_id: str | None = None,
    ) -> None:
        self.ledger = ledger
        self.reservations = reservations
        self.snapshot_id = snapshot_id


def validate_snapshot_request(data: Any) -> str:
    """Validate a decoded JSON body for POST /snapshots.

    The body must be an object with exactly one field, ``snapshotId``,
    holding a non-empty string.
    """
    if not isinstance(data, dict):
        raise EventValidationError("snapshot body must be a JSON object")

    keys = set(data)
    expected = set(SNAPSHOT_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    snapshot_id = data["snapshotId"]
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise EventValidationError("snapshotId must be a non-empty string")
    return snapshot_id


def validate_branch_request(data: Any) -> tuple[str, str]:
    """Validate a decoded JSON body for POST /branches.

    The body must be an object with exactly ``branchId`` and
    ``snapshotId``, both non-empty strings.
    """
    if not isinstance(data, dict):
        raise EventValidationError("branch body must be a JSON object")

    keys = set(data)
    expected = set(BRANCH_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    branch_id = data["branchId"]
    snapshot_id = data["snapshotId"]
    for field, value in (("branchId", branch_id), ("snapshotId", snapshot_id)):
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return branch_id, snapshot_id


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


def _branch_route(path: str) -> tuple[str, str] | None:
    """Split ``/branches/{branchId}[/sub...]`` into ``(branchId, subpath)``.

    Returns ``None`` for paths outside the branch prefix; the subpath is
    ``""`` for the bare branch resource itself.
    """
    if not path.startswith("/branches/"):
        return None
    rest = path[len("/branches/"):]
    branch_id, _, sub = rest.partition("/")
    if not branch_id:
        return None
    return unquote(branch_id), "/" + sub if sub else ""


def _snapshot_summary(snapshot_id: str, state: IsolatedState) -> dict[str, Any]:
    return {
        "snapshotId": snapshot_id,
        "events": state.ledger.count(),
        "reservations": state.reservations.count(),
    }


def _branch_summary(branch_id: str, state: IsolatedState) -> dict[str, Any]:
    return {
        "branchId": branch_id,
        "snapshotId": state.snapshot_id,
        "events": state.ledger.count(),
        "reservations": state.reservations.count(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "EventSim/0.1"

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {"service": SERVICE_NAME, "status": "ok"},
            )
            return
        if path == "/snapshots":
            self._list_snapshots()
            return
        branch = _branch_route(path)
        if branch is not None:
            branch_id, subpath = branch
            self._dispatch_branch_get(branch_id, subpath, parsed.query)
            return
        if path == "/events":
            self._list_events(parsed.query, self.server.ledger)  # type: ignore[attr-defined]
            return
        if path == "/events/aggregate":
            self._aggregate_events(parsed.query, self.server.ledger)  # type: ignore[attr-defined]
            return
        if path == "/reservations":
            self._list_reservations(parsed.query, self.server.reservations)  # type: ignore[attr-defined]
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": self.path},
        )

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/snapshots":
            self._create_snapshot()
            return
        if path == "/branches":
            self._create_branch()
            return
        branch = _branch_route(path)
        if branch is not None:
            branch_id, subpath = branch
            self._dispatch_branch_post(branch_id, subpath)
            return
        if path == "/reservations":
            self._create_reservation(self.server.reservations)  # type: ignore[attr-defined]
            return
        if path == "/decisions/evaluate":
            self._evaluate_decision(self.server.ledger)  # type: ignore[attr-defined]
            return
        if path == "/decisions/allocate":
            self._allocate_decision()
            return
        if path == "/events":
            self._create_event(self.server.ledger)  # type: ignore[attr-defined]
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": self.path},
        )

    def _create_event(self, ledger: EventLedger) -> None:
        data = self._json_request_body()
        if data is _BODY_ERROR:
            return

        try:
            event = validate_event(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return

        status, stored = ledger.add(event)
        if status == "created":
            self._write_json(HTTPStatus.CREATED, stored)
        elif status == "exists":
            self._write_json(HTTPStatus.OK, stored)
        else:
            self._write_json(
                HTTPStatus.CONFLICT,
                {"error": "event_id_conflict"},
            )

    def _evaluate_decision(self, ledger: EventLedger) -> None:
        data = self._json_request_body()
        if data is _BODY_ERROR:
            return

        try:
            params = validate_decision_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return

        # Read-only: the ledger is never mutated by a decision request, and
        # the snapshot is taken in a single locked copy for consistency.
        occurred = ledger.occurred_at_values(
            params["organizationId"], params["type"]
        )
        result = evaluate_decision(occurred, params)
        self._write_json(HTTPStatus.OK, result)

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

    def _create_reservation(self, inventory: ReservationInventory) -> None:
        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            reservation = validate_reservation_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        status, view = inventory.reserve(reservation)
        if status == "created":
            self._write_json(HTTPStatus.CREATED, view, newline=True)
        elif status == "exists":
            self._write_json(HTTPStatus.OK, view, newline=True)
        else:
            self._write_json(HTTPStatus.CONFLICT, {"error": status}, newline=True)

    def _list_reservations(
        self, query: str, inventory: ReservationInventory
    ) -> None:
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return
        reservations = inventory.list_for_organization(organization_id)
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": organization_id, "reservations": reservations},
            newline=True,
        )

    def _list_events(self, query: str, ledger: EventLedger) -> None:
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
            )
            return
        events = ledger.list_for_organization(organization_id)
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": organization_id, "events": events},
        )

    def _aggregate_events(self, query: str, ledger: EventLedger) -> None:
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
        )

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

        # Copy first, then publish under the name lock: a failed or
        # conflicting request never touches the main state, and a stored
        # snapshot is never rewritten.
        state = IsolatedState(
            self.server.ledger.clone(),  # type: ignore[attr-defined]
            self.server.reservations.clone(),  # type: ignore[attr-defined]
        )
        with self.server.states_lock:  # type: ignore[attr-defined]
            snapshots = self.server.snapshots  # type: ignore[attr-defined]
            if snapshot_id in snapshots:
                self._write_json(
                    HTTPStatus.CONFLICT,
                    {"error": "snapshot_conflict"},
                )
                return
            snapshots[snapshot_id] = state
        self._write_json(
            HTTPStatus.CREATED,
            _snapshot_summary(snapshot_id, state),
        )

    def _list_snapshots(self) -> None:
        with self.server.states_lock:  # type: ignore[attr-defined]
            items = sorted(self.server.snapshots.items())  # type: ignore[attr-defined]
        self._write_json(
            HTTPStatus.OK,
            {
                "snapshots": [
                    _snapshot_summary(snapshot_id, state)
                    for snapshot_id, state in items
                ]
            },
        )

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

        with self.server.states_lock:  # type: ignore[attr-defined]
            branches = self.server.branches  # type: ignore[attr-defined]
            if branch_id in branches:
                self._write_json(
                    HTTPStatus.CONFLICT,
                    {"error": "branch_conflict"},
                )
                return
            snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
            if snapshot is None:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "snapshot_not_found"},
                )
                return
            # The branch deep-copies the snapshot so later writes on either
            # side never share mutable state.
            state = IsolatedState(
                snapshot.ledger.clone(),
                snapshot.reservations.clone(),
                snapshot_id=snapshot_id,
            )
            branches[branch_id] = state
        self._write_json(
            HTTPStatus.CREATED,
            _branch_summary(branch_id, state),
        )

    def _branch_state(self, branch_id: str) -> IsolatedState | None:
        with self.server.states_lock:  # type: ignore[attr-defined]
            return self.server.branches.get(branch_id)  # type: ignore[attr-defined]

    def _dispatch_branch_get(
        self, branch_id: str, subpath: str, query: str
    ) -> None:
        state = self._branch_state(branch_id)
        if state is None:
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "branch_not_found"},
            )
            return
        if subpath == "":
            self._write_json(HTTPStatus.OK, _branch_summary(branch_id, state))
            return
        if subpath == "/events":
            self._list_events(query, state.ledger)
            return
        if subpath == "/events/aggregate":
            self._aggregate_events(query, state.ledger)
            return
        if subpath == "/reservations":
            self._list_reservations(query, state.reservations)
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": self.path},
        )

    def _dispatch_branch_post(self, branch_id: str, subpath: str) -> None:
        state = self._branch_state(branch_id)
        if state is None:
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "branch_not_found"},
            )
            return
        if subpath == "/events":
            self._create_event(state.ledger)
            return
        if subpath == "/decisions/evaluate":
            self._evaluate_decision(state.ledger)
            return
        if subpath == "/reservations":
            self._create_reservation(state.reservations)
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": self.path},
        )

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
    # Snapshots and branches live only in this process: a restart begins
    # with none of them, exactly like the ledger and the inventory.
    server.snapshots = {}  # type: ignore[attr-defined]
    server.branches = {}  # type: ignore[attr-defined]
    server.states_lock = threading.Lock()  # type: ignore[attr-defined]
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
