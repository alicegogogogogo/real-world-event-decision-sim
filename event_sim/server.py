from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, NamedTuple
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

TOKEN_FIELDS = ("token", "organizationId", "role")
ROLES = ("read", "write")

BRANCH_COMPARE_REQUIRED_FIELDS = (
    "organizationId",
    "left",
    "right",
    "type",
    "windowSize",
    "threshold",
)
BRANCH_COMPARE_OPTIONAL_FIELDS = ("from", "to")
BRANCH_COMPARE_FIELDS = (
    BRANCH_COMPARE_REQUIRED_FIELDS + BRANCH_COMPARE_OPTIONAL_FIELDS
)

BRANCH_EVENT_COMPARE_FIELDS = ("organizationId", "left", "right")

BRANCH_RESERVATION_COMPARE_FIELDS = ("organizationId", "left", "right")

SNAPSHOT_EVENT_COMPARE_FIELDS = ("organizationId", "left", "right")

SNAPSHOT_RESERVATION_COMPARE_FIELDS = ("organizationId", "left", "right")

# Reservations align by reservationId; only these four fields participate in
# the same/diff decision for an identifier present on both sides.
RESERVATION_COMPARE_FIELDS = (
    "organizationId",
    "resourceId",
    "quantity",
    "capacity",
)

# Sentinel returned by Handler._json_request_body after it has already
# written the 415/400 error response.
_BODY_ERROR = object()


class Subject(NamedTuple):
    """The identity bound to a registered token: one organization and role."""

    organization_id: str
    role: str


class TokenRegistry:
    """In-process map of bearer tokens to an organization and a role.

    Credentials live only for the lifetime of this instance: a restart
    starts with an empty registry and no protected entry point accepts any
    token. Registration is idempotent for an identical record; a token can
    never be rebound to a different organization or role. The same lock that
    guards the map also wraps the decide-and-commit sequence of a write
    request, so the authorization decision and the ledger/inventory mutation
    are indivisible.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, Subject] = {}
        self._lock = threading.Lock()

    def register(
        self, token: str, organization_id: str, role: str
    ) -> tuple[str, Subject]:
        """Register a token or reconcile an identical resubmission.

        Returns ``("created", subject)`` for a new token, ``("exists",
        subject)`` for a resubmission carrying the same organization and
        role, or ``("conflict", existing)`` when the token is already bound
        to different credentials. Only ``"created"`` adds a record.
        """
        with self._lock:
            existing = self._tokens.get(token)
            if existing is None:
                subject = Subject(organization_id, role)
                self._tokens[token] = subject
                return "created", subject
            if existing.organization_id == organization_id and existing.role == role:
                return "exists", existing
            return "conflict", existing

    def lookup(self, token: str) -> Subject | None:
        with self._lock:
            return self._tokens.get(token)

    def run_locked(self, commit: Callable[[], Any]) -> Any:
        """Run ``commit`` holding the registry lock.

        Snapshot and branch creation make their single role decision earlier
        (from the request-scoped subject) and then run the capture/fork
        sequence here, so it stays indivisible from every other
        registry-guarded write without re-reading a credential.
        """
        with self._lock:
            return commit()

    def commit_write(
        self,
        token: str,
        organization_id: str,
        commit: Callable[[], Any],
    ) -> tuple[str, Any]:
        """Authorize a write and run ``commit`` under one lock.

        The token is re-checked against the registry, its organization must
        match the request's organization, and its role must be ``write``;
        only then does ``commit`` run, still holding the registry lock. The
        status is ``"ok"`` with the commit result or ``"forbidden"`` (the
        commit is never invoked). Reaching the store this way makes a
        rejected request incapable of changing the ledger or inventory.
        """
        with self._lock:
            subject = self._tokens.get(token)
            if (
                subject is None
                or subject.organization_id != organization_id
                or subject.role != "write"
            ):
                return "forbidden", None
            return "ok", commit()


class EventValidationError(ValueError):
    """The submitted event object fails the ledger's field rules."""


def event_region(event: dict[str, Any]) -> str | None:
    """Return an event's region attribution, or None when it has none.

    The region is the payload's ``region`` value: only a non-empty string
    attributes the event to a region. A missing key, an empty string, or any
    non-string value means the event has no region. Matching uses the value
    verbatim, with no normalization.
    """
    region = event["payload"].get("region")
    if isinstance(region, str) and region != "":
        return region
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

    def list_for_organization_as_of(
        self, organization_id: str, as_of: int
    ) -> list[dict[str, Any]]:
        """List one organization's events with occurredAt not after ``as_of``.

        Ordering matches :meth:`list_for_organization`; the result is a
        consistent locked snapshot, so a replay never observes a partial
        concurrent write.
        """
        with self._lock:
            events = [
                dict(event)
                for event in self._events.values()
                if event["organizationId"] == organization_id
                and event["occurredAt"] <= as_of
            ]
        events.sort(key=lambda event: (event["occurredAt"], event["eventId"]))
        return events

    def list_for_organization_region(
        self, organization_id: str, region: str
    ) -> list[dict[str, Any]]:
        """List one organization's events attributed to one region.

        Only events whose payload ``region`` is a non-empty string equal to
        ``region`` (compared verbatim) are included; ordering matches
        :meth:`list_for_organization`.
        """
        with self._lock:
            events = [
                dict(event)
                for event in self._events.values()
                if event["organizationId"] == organization_id
                and event_region(event) == region
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

    def occurred_at_values_for_region(
        self, organization_id: str, event_type: str, region: str
    ) -> list[int]:
        """Sorted occurredAt values of an org/type's events in one region.

        Consistent locked snapshot, like :meth:`occurred_at_values`; events
        without a non-empty string payload ``region`` never match.
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

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a deep, locked copy of every stored event."""
        with self._lock:
            return {
                event_id: dict(event) for event_id, event in self._events.items()
            }

    def snapshot_for_organization(
        self, organization_id: str
    ) -> dict[str, dict[str, Any]]:
        """Return a deep, locked copy of one organization's stored events."""
        with self._lock:
            return {
                event_id: dict(event)
                for event_id, event in self._events.items()
                if event["organizationId"] == organization_id
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

    def snapshot_for_organization(
        self, organization_id: str
    ) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
        """Locked copies of capacities/reservations visible to one org.

        Capacities are included only for resources the organization itself
        reserves; reservations of other organizations are never captured.
        """
        with self._lock:
            reservations = {
                reservation_id: dict(record)
                for reservation_id, record in self._reservations.items()
                if record["organizationId"] == organization_id
            }
            capacities = {
                resource_id: capacity
                for resource_id, capacity in self._capacities.items()
                if resource_id in {r["resourceId"] for r in reservations.values()}
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
        organization_id: str,
        events: dict[str, dict[str, Any]],
        capacities: dict[str, int],
        reservations: dict[str, dict[str, Any]],
    ) -> None:
        self.snapshot_id = snapshot_id
        self.organization_id = organization_id
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

    def events_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the captured events, keyed by eventId.

        A snapshot holds only its owning organization's events, so the copy
        needs no further organization filtering. It is immutable after
        capture; the copy keeps a read-only comparison from sharing mutable
        references with the snapshot.
        """
        return {event_id: dict(event) for event_id, event in self._events.items()}

    def reservations_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the captured reservations, keyed by id.

        A snapshot holds only its owning organization's reservations, so the
        copy needs no further organization filtering. It is immutable after
        capture; the copy keeps a read-only comparison from sharing mutable
        references with the snapshot.
        """
        return {
            reservation_id: dict(record)
            for reservation_id, record in self._reservations.items()
        }


class SnapshotStore:
    """In-process registry of named snapshots keyed by snapshotId."""

    def __init__(self) -> None:
        self._snapshots: dict[str, Snapshot] = {}
        self._lock = threading.Lock()

    def create(
        self,
        snapshot_id: str,
        organization_id: str,
        ledger: EventLedger,
        reservations: ReservationInventory,
    ) -> tuple[str, Snapshot]:
        """Capture one organization's state under ``snapshot_id``.

        Returns ``("created", snapshot)`` or ``("conflict", snapshot)`` when
        the name already exists; an existing snapshot is never replaced.
        """
        with self._lock:
            existing = self._snapshots.get(snapshot_id)
            if existing is not None:
                return "conflict", existing
            snapshot = Snapshot(
                snapshot_id,
                organization_id,
                ledger.snapshot_for_organization(organization_id),
                *reservations.snapshot_for_organization(organization_id),
            )
            self._snapshots[snapshot_id] = snapshot
            return "created", snapshot

    def get(self, snapshot_id: str) -> Snapshot | None:
        with self._lock:
            return self._snapshots.get(snapshot_id)

    def summaries_for_organization(
        self, organization_id: str
    ) -> list[dict[str, Any]]:
        """List one organization's snapshots, sorted by snapshotId."""
        with self._lock:
            snapshots = [
                snapshot
                for snapshot in self._snapshots.values()
                if snapshot.organization_id == organization_id
            ]
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

    def __init__(self, branch_id: str, snapshot: Snapshot) -> None:
        self.branch_id = branch_id
        self.organization_id = snapshot.organization_id
        self.snapshot_id = snapshot.snapshot_id
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
            branch = Branch(branch_id, snapshot)
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


def _single_non_empty_text(params: dict[str, list[str]], name: str) -> str:
    """Extract one exactly-once, non-blank text parameter from parsed query."""
    values = params.get(name)
    if not values or len(values) != 1:
        raise EventValidationError(
            f"{name} query parameter is required exactly once"
        )
    value = values[0]
    if not value.strip():
        raise EventValidationError(f"{name} must be non-empty")
    return value


def _region_list_params_from_query(query: str) -> dict[str, str]:
    """Validate the /events/region query string.

    organizationId and region are each required exactly once and non-empty;
    matching keeps the raw region text verbatim.
    """
    params = parse_qs(query, keep_blank_values=True)
    return {
        "organizationId": _single_non_empty_text(params, "organizationId"),
        "region": _single_non_empty_text(params, "region"),
    }


def _region_aggregate_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /events/region/aggregate query string.

    organizationId, region, type and windowSize are each required exactly
    once; from and to must be both absent or both present exactly once. The
    rules mirror /events/aggregate, with an added region parameter.
    """
    params = parse_qs(query, keep_blank_values=True)

    organization_id = _single_non_empty_text(params, "organizationId")
    region = _single_non_empty_text(params, "region")
    event_type = _single_non_empty_text(params, "type")

    window_size_text = _single_non_empty_text(params, "windowSize")
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


def _integer_text(value: str) -> int | None:
    """Parse ASCII decimal integer text; return None for anything else."""
    if not value or not value.isascii() or not value.isdigit():
        return None
    return int(value)


def _non_negative_integer_param(params: dict[str, list[str]], name: str) -> int:
    """Extract one exactly-once, non-blank, non-negative integer parameter."""
    text = _single_non_empty_text(params, name)
    value = _integer_text(text)
    if value is None:
        raise EventValidationError(f"{name} must be a non-negative integer")
    return value


def _replay_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /events/replay query string.

    organizationId and asOf are each required exactly once and non-empty;
    asOf must be non-negative integer text.
    """
    params = parse_qs(query, keep_blank_values=True)
    return {
        "organizationId": _single_non_empty_text(params, "organizationId"),
        "asOf": _non_negative_integer_param(params, "asOf"),
    }


def _replay_compare_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /events/replay/compare query string.

    organizationId, fromAsOf and toAsOf are each required exactly once and
    non-empty; both time points must be non-negative integer text. The two
    time points may relate to each other in either direction.
    """
    params = parse_qs(query, keep_blank_values=True)
    return {
        "organizationId": _single_non_empty_text(params, "organizationId"),
        "fromAsOf": _non_negative_integer_param(params, "fromAsOf"),
        "toAsOf": _non_negative_integer_param(params, "toAsOf"),
    }


def compare_replays(
    from_events: list[dict[str, Any]], to_events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Diff two replayed event lists by eventId.

    ``added`` holds identifiers only in the to-replay, ``removed`` those
    only in the from-replay; identifiers present in both count toward
    ``unchangedCount`` when every field except eventId matches, and land in
    ``changed`` otherwise. Each identifier group is sorted in Unicode
    code-point order.
    """
    from_by_id = {event["eventId"]: event for event in from_events}
    to_by_id = {event["eventId"]: event for event in to_events}
    added = sorted(set(to_by_id) - set(from_by_id))
    removed = sorted(set(from_by_id) - set(to_by_id))
    changed: list[str] = []
    unchanged_count = 0
    for event_id in sorted(set(from_by_id) & set(to_by_id)):
        before = from_by_id[event_id]
        after = to_by_id[event_id]
        if all(
            before[field] == after[field]
            for field in EVENT_FIELDS
            if field != "eventId"
        ):
            unchanged_count += 1
        else:
            changed.append(event_id)
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchangedCount": unchanged_count,
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


def validate_token_request(data: Any) -> dict[str, str]:
    """Validate a decoded JSON body for POST /auth/tokens.

    Exactly the ``token``, ``organizationId`` and ``role`` fields may be
    present; the first two must be non-empty strings and role must be
    ``read`` or ``write``. A blank (whitespace-only) string is rejected just
    like an empty one.
    """
    if not isinstance(data, dict):
        raise EventValidationError("token body must be a JSON object")

    keys = set(data)
    expected = set(TOKEN_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in ("token", "organizationId"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")

    if data["role"] not in ROLES:
        raise EventValidationError("role must be read or write")

    return {field: data[field] for field in TOKEN_FIELDS}


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


def validate_branch_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /branches/compare.

    Required fields: organizationId, left, right, type, windowSize,
    threshold. Optional fields: from and to, which must appear together. No
    other fields are allowed. ``left`` and ``right`` are branch names; the
    same name on both sides is legal.
    """
    if not isinstance(data, dict):
        raise EventValidationError("branch comparison body must be a JSON object")

    keys = set(data)
    required = set(BRANCH_COMPARE_REQUIRED_FIELDS)
    allowed = set(BRANCH_COMPARE_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "left", "right", "type"):
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
        "left": data["left"],
        "right": data["right"],
        "type": data["type"],
        "windowSize": data["windowSize"],
        "threshold": data["threshold"],
        "from": from_value,
        "to": to_value,
    }


def validate_branch_event_compare_request(data: Any) -> dict[str, str]:
    """Validate a decoded JSON body for POST /branches/compare/events.

    Exactly ``organizationId``, ``left`` and ``right`` may be present, each
    a non-empty string. ``left`` and ``right`` are branch names; the same
    name on both sides is legal.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "branch event comparison body must be a JSON object"
        )

    keys = set(data)
    expected = set(BRANCH_EVENT_COMPARE_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in BRANCH_EVENT_COMPARE_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return {field: data[field] for field in BRANCH_EVENT_COMPARE_FIELDS}


def validate_branch_reservation_compare_request(data: Any) -> dict[str, str]:
    """Validate a decoded JSON body for POST /branches/compare/reservations.

    Exactly ``organizationId``, ``left`` and ``right`` may be present, each
    a non-empty string. ``left`` and ``right`` are branch names; the same
    name on both sides is legal.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "branch reservation comparison body must be a JSON object"
        )

    keys = set(data)
    expected = set(BRANCH_RESERVATION_COMPARE_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in BRANCH_RESERVATION_COMPARE_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return {field: data[field] for field in BRANCH_RESERVATION_COMPARE_FIELDS}


def validate_snapshot_event_compare_request(data: Any) -> dict[str, str]:
    """Validate a decoded JSON body for POST /snapshots/compare/events.

    Exactly ``organizationId``, ``left`` and ``right`` may be present, each
    a non-empty string. ``left`` and ``right`` are snapshot names; the same
    name on both sides is legal.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "snapshot event comparison body must be a JSON object"
        )

    keys = set(data)
    expected = set(SNAPSHOT_EVENT_COMPARE_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in SNAPSHOT_EVENT_COMPARE_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return {field: data[field] for field in SNAPSHOT_EVENT_COMPARE_FIELDS}


def validate_snapshot_reservation_compare_request(data: Any) -> dict[str, str]:
    """Validate a decoded JSON body for POST /snapshots/compare/reservations.

    Exactly ``organizationId``, ``left`` and ``right`` may be present, each
    a non-empty string. ``left`` and ``right`` are snapshot names; the same
    name on both sides is legal.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "snapshot reservation comparison body must be a JSON object"
        )

    keys = set(data)
    expected = set(SNAPSHOT_RESERVATION_COMPARE_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in SNAPSHOT_RESERVATION_COMPARE_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return {field: data[field] for field in SNAPSHOT_RESERVATION_COMPARE_FIELDS}


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


def _windows_from_occurred(
    occurred: list[int],
    window_size: int,
    from_value: int | None,
    to_value: int | None,
) -> list[dict[str, int]]:
    """Build aggregate windows from a sorted timestamp snapshot.

    Windows start at zero and cover ``[start, start + windowSize)``. Without
    a range only windows actually hit by events are returned; with a range
    every window intersecting the closed interval ``[from, to]`` is kept,
    empty ones included. Windows are ordered by start ascending.
    """
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

    return [
        {"start": start, "end": start + window_size, "count": counts.get(start, 0)}
        for start in starts
    ]


def _branch_window_counts(
    occurred: list[int], params: dict[str, Any]
) -> dict[int, int]:
    """Count one side's matching events per window start, with range filter.

    Shares the window division and closed-interval filtering used by the
    aggregate and decision entry points.
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
    return counts


def _branch_window_starts(params: dict[str, Any]) -> list[int]:
    """Window starts a comparison must align, using the aggregate contract.

    Without a range there is no fixed grid; the caller unions the two sides'
    hit windows. With a range every window intersecting the closed interval
    is present, so both sides share the same start list.
    """
    if params["from"] is None:
        return []
    window_size = params["windowSize"]
    first = (params["from"] // window_size) * window_size
    last = (params["to"] // window_size) * window_size
    return list(range(first, last + 1, window_size))


def compare_branches(
    left_occurred: list[int],
    right_occurred: list[int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Diff two branches' window counts and peaks from timestamp snapshots.

    Each ``occurred`` list is a single locked snapshot of one branch's
    matching events, so the window rows and the peak for a side derive from
    the same read; nothing is written. Without a range, rows cover the union
    of the two sides' hit windows; with a range, rows cover every window
    intersecting ``[from, to]``, empty windows kept. Rows align by start
    ascending and carry each side's count plus an ``equal`` marker. The
    ``decision`` block holds each side's peak and action and a marker saying
    whether the two peak results agree.
    """
    left_counts = _branch_window_counts(left_occurred, params)
    right_counts = _branch_window_counts(right_occurred, params)

    range_starts = _branch_window_starts(params)
    if range_starts:
        starts = range_starts
    else:
        starts = sorted(set(left_counts) | set(right_counts))

    windows = []
    for start in starts:
        left_count = left_counts.get(start, 0)
        right_count = right_counts.get(start, 0)
        windows.append(
            {
                "start": start,
                "leftCount": left_count,
                "rightCount": right_count,
                "equal": left_count == right_count,
            }
        )

    left_peak = evaluate_decision(left_occurred, params)
    right_peak = evaluate_decision(right_occurred, params)
    left_decision = {
        "peakStart": left_peak["peakStart"],
        "peakCount": left_peak["peakCount"],
        "action": left_peak["action"],
    }
    right_decision = {
        "peakStart": right_peak["peakStart"],
        "peakCount": right_peak["peakCount"],
        "action": right_peak["action"],
    }
    decision = {
        "left": left_decision,
        "right": right_decision,
        "equal": left_decision == right_decision,
    }

    return {
        "organizationId": params["organizationId"],
        "left": params["left"],
        "right": params["right"],
        "type": params["type"],
        "windowSize": params["windowSize"],
        "threshold": params["threshold"],
        "from": params["from"],
        "to": params["to"],
        "decision": decision,
        "windows": windows,
    }


def _compare_events(
    left_by_id: dict[str, dict[str, Any]],
    right_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Diff two event mappings aligned by eventId.

    Each mapping is a single locked snapshot of one side's events for one
    organization, keyed by eventId, so the four groups derive from consistent
    reads and nothing is written. Identifiers only on the left land in
    ``leftOnly``, only on the right in ``rightOnly``. An identifier on both
    sides counts toward ``same`` when every field except eventId matches
    (payloads compare by content) and lands in ``diff`` otherwise, as
    ``{"eventId", "fields"}`` naming the mismatched fields. Identifiers and
    field names are sorted in Unicode code-point order; each group also gets
    a ``<group>Count`` key.
    """
    left_only = sorted(set(left_by_id) - set(right_by_id))
    right_only = sorted(set(right_by_id) - set(left_by_id))
    same: list[str] = []
    diff: list[dict[str, Any]] = []
    for event_id in sorted(set(left_by_id) & set(right_by_id)):
        left = left_by_id[event_id]
        right = right_by_id[event_id]
        fields = sorted(
            field
            for field in EVENT_FIELDS
            if field != "eventId" and left[field] != right[field]
        )
        if fields:
            diff.append({"eventId": event_id, "fields": fields})
        else:
            same.append(event_id)

    return {
        "leftOnly": left_only,
        "leftOnlyCount": len(left_only),
        "rightOnly": right_only,
        "rightOnlyCount": len(right_only),
        "same": same,
        "sameCount": len(same),
        "diff": diff,
        "diffCount": len(diff),
    }


def compare_branch_events(
    left_events: list[dict[str, Any]], right_events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Diff two branches' events aligned by eventId.

    Delegates to the shared :func:`_compare_events`; each event list is a
    single locked snapshot of one branch's ledger, so the four groups derive
    from consistent reads and nothing is written.
    """
    left_by_id = {event["eventId"]: event for event in left_events}
    right_by_id = {event["eventId"]: event for event in right_events}
    return _compare_events(left_by_id, right_by_id)


def compare_snapshot_events(
    left_events: dict[str, dict[str, Any]],
    right_events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Diff two snapshots' events aligned by eventId.

    Delegates to the shared :func:`_compare_events`; each mapping is a deep
    copy of one snapshot's captured events (already scoped to one
    organization at capture time).
    """
    return _compare_events(left_events, right_events)


def _compare_reservations(
    left_reservations: dict[str, dict[str, Any]],
    right_reservations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Diff two reservation mappings aligned by reservationId.

    Each mapping is a single locked snapshot of one side's reservations for
    one organization, keyed by reservationId, so the four groups derive from
    consistent reads and nothing is written. Identifiers only on the left
    land in ``leftOnly``, only on the right in ``rightOnly``. An identifier
    on both sides counts toward ``same`` when its organization, resource,
    quantity and capacity all match, and lands in ``diff`` otherwise, as
    ``{"reservationId", "fields"}`` naming the mismatched fields. No other
    content participates in the comparison. Identifiers and field names are
    sorted in Unicode code-point order; each group also gets a
    ``<group>Count`` key.
    """
    left_only = sorted(set(left_reservations) - set(right_reservations))
    right_only = sorted(set(right_reservations) - set(left_reservations))
    same: list[str] = []
    diff: list[dict[str, Any]] = []
    for reservation_id in sorted(set(left_reservations) & set(right_reservations)):
        left = left_reservations[reservation_id]
        right = right_reservations[reservation_id]
        fields = sorted(
            field
            for field in RESERVATION_COMPARE_FIELDS
            if left[field] != right[field]
        )
        if fields:
            diff.append({"reservationId": reservation_id, "fields": fields})
        else:
            same.append(reservation_id)

    return {
        "leftOnly": left_only,
        "leftOnlyCount": len(left_only),
        "rightOnly": right_only,
        "rightOnlyCount": len(right_only),
        "same": same,
        "sameCount": len(same),
        "diff": diff,
        "diffCount": len(diff),
    }


def compare_branch_reservations(
    left_reservations: dict[str, dict[str, Any]],
    right_reservations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Diff two branches' reservations aligned by reservationId.

    Delegates to the shared :func:`_compare_reservations`; each mapping is a
    locked snapshot of one branch's reservations for one organization.
    """
    return _compare_reservations(left_reservations, right_reservations)


def compare_snapshot_reservations(
    left_reservations: dict[str, dict[str, Any]],
    right_reservations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Diff two snapshots' reservations aligned by reservationId.

    Delegates to the shared :func:`_compare_reservations`; each mapping is a
    deep copy of one snapshot's captured reservations (already scoped to one
    organization at capture time).
    """
    return _compare_reservations(left_reservations, right_reservations)


class Handler(BaseHTTPRequestHandler):
    server_version = "EventSim/0.1"

    # Set by _require_subject for every authenticated request.
    token: str = ""

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
            subject = self._require_subject(newline=True)
            if subject is None:
                return
            if len(segments) == 1 and segments[0]:
                self._get_branch(unquote(segments[0]), subject)
                return
            if len(segments) >= 2 and segments[0]:
                branch_id = unquote(segments[0])
                branch = self.server.branches.get(  # type: ignore[attr-defined]
                    branch_id
                )
                if branch is None:
                    self._branch_not_found()
                    return
                # The branch belongs to one organization; a credential for
                # any other organization is rejected before the query is even
                # parsed.
                if not self._allow_branch(subject, branch, newline=True):
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
        if path == "/events/replay":
            self._replay_events(
                self.server.ledger,  # type: ignore[attr-defined]
                query,
            )
            return
        if path == "/events/replay/compare":
            self._compare_replays(
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
        if path == "/events/aggregate":
            self._aggregate_events(
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

        if path == "/auth/tokens":
            self._register_token()
            return

        if path == "/snapshots":
            self._create_snapshot()
            return
        if path == "/branches":
            self._create_branch()
            return
        if path == "/branches/compare":
            self._compare_branches()
            return
        if path == "/branches/compare/events":
            self._compare_branch_events()
            return
        if path == "/branches/compare/reservations":
            self._compare_branch_reservations()
            return
        if path == "/snapshots/compare/events":
            self._compare_snapshot_events()
            return
        if path == "/snapshots/compare/reservations":
            self._compare_snapshot_reservations()
            return

        if path.startswith("/branches/"):
            remainder = path[len("/branches/"):]
            segments = remainder.split("/")
            subject = self._require_subject(newline=True)
            if subject is None:
                return
            if len(segments) >= 2 and segments[0]:
                branch_id = unquote(segments[0])
                branch = self.server.branches.get(  # type: ignore[attr-defined]
                    branch_id
                )
                if branch is None:
                    self._branch_not_found()
                    return
                if not self._allow_branch(subject, branch, newline=True):
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

    # ------------------------------------------------------------ registration

    def _register_token(self) -> None:
        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            credentials = validate_token_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        status, _subject = self.server.tokens.register(  # type: ignore[attr-defined]
            credentials["token"],
            credentials["organizationId"],
            credentials["role"],
        )
        if status == "created":
            self._write_json(HTTPStatus.CREATED, credentials, newline=True)
        elif status == "exists":
            # Idempotent resubmission: the record is not added a second time
            # and the same three fields are echoed back.
            self._write_json(HTTPStatus.OK, credentials, newline=True)
        else:
            self._write_json(
                HTTPStatus.CONFLICT,
                {"error": "auth_conflict"},
                newline=True,
            )

    # ------------------------------------------------------------- snapshots

    def _create_snapshot(self) -> None:
        subject = self._require_subject(newline=False)
        if subject is None:
            return

        # The single permission decision for this request: the write role is
        # settled once, before the body is read (403 outranks 415/400/422).
        # The token is never re-read afterward; the capture below only runs
        # under the registry lock.
        if subject.role != "write":
            self._forbidden(newline=False)
            return

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

        # The snapshot belongs to the creator's organization and captures only
        # that organization's state. The capture and the insert run under the
        # registry lock, so concurrent requests serialize on a name and a
        # rejected validation request stores nothing.
        def capture() -> tuple[str, Snapshot]:
            return self.server.snapshots.create(  # type: ignore[attr-defined]
                snapshot_id,
                subject.organization_id,
                self.server.ledger,  # type: ignore[attr-defined]
                self.server.reservations,  # type: ignore[attr-defined]
            )

        create_status, snapshot = self.server.tokens.run_locked(  # type: ignore[attr-defined]
            capture
        )
        if create_status == "created":
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
        subject = self._require_subject(newline=False)
        if subject is None:
            return
        self._write_json(
            HTTPStatus.OK,
            {
                "snapshots": self.server.snapshots.summaries_for_organization(  # type: ignore[attr-defined]
                    subject.organization_id
                )
            },
        )

    # --------------------------------------------------------------- branches

    def _create_branch(self) -> None:
        subject = self._require_subject(newline=False)
        if subject is None:
            return
        # The one permission decision for this request: a write credential is
        # required, checked before the body is read. The token is never
        # re-read afterward; snapshot ownership and the duplicate-name insert
        # below run together under the registry lock.
        if subject.role != "write":
            self._forbidden(newline=False)
            return

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

        # The snapshot lookup, the ownership decision, and the duplicate-name
        # insert are one indivisible step under the registry lock: a snapshot
        # owned by another organization is forbidden before the branch name is
        # ever compared, a missing snapshot is reported as not found, and a
        # rejected request can neither create a branch nor race its way past
        # the ownership check.
        def fork() -> tuple[str, Branch | None]:
            snapshot = self.server.snapshots.get(  # type: ignore[attr-defined]
                snapshot_id
            )
            if snapshot is None:
                return "snapshot_missing", None
            if snapshot.organization_id != subject.organization_id:
                return "forbidden", None
            return self.server.branches.create(  # type: ignore[attr-defined]
                branch_id, snapshot
            )

        create_status, branch = self.server.tokens.run_locked(  # type: ignore[attr-defined]
            fork
        )
        if create_status == "forbidden":
            self._forbidden(newline=False)
            return
        if create_status == "snapshot_missing":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "snapshot_not_found"},
            )
            return
        if create_status == "created":
            self._write_json(HTTPStatus.CREATED, branch.summary())
        else:
            self._write_json(
                HTTPStatus.CONFLICT,
                {"error": "branch_conflict"},
            )

    def _get_branch(self, branch_id: str, subject: Subject) -> None:
        branch = self.server.branches.get(branch_id)  # type: ignore[attr-defined]
        if branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, branch, newline=True):
            return
        # The branch summary success body keeps its original byte contract:
        # compact JSON with no trailing newline (the 404/403 errors do use one).
        self._write_json(HTTPStatus.OK, branch.summary())

    def _branch_not_found(self) -> None:
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "branch_not_found"},
            newline=True,
        )

    def _snapshot_not_found(self) -> None:
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "snapshot_not_found"},
            newline=True,
        )

    def _compare_branches(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_branch_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]

        # Read-only and organization-scoped. The organization decision
        # happens before either branch name is inspected, so a foreign
        # organization cannot probe which branch names exist.
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Both branches must already exist; the fixed left-then-right order
        # makes the verdict deterministic, and a missing name is reported as
        # branch_not_found without ever creating a branch.
        left_branch = self.server.branches.get(  # type: ignore[attr-defined]
            params["left"]
        )
        if left_branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, left_branch, newline=True):
            return
        right_branch = self.server.branches.get(  # type: ignore[attr-defined]
            params["right"]
        )
        if right_branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, right_branch, newline=True):
            return

        # Each side reads its own ledger in one locked snapshot, then both
        # sides are recomputed purely from those copies; nothing is written,
        # so identical submissions return byte-identical JSON.
        left_occurred = left_branch.ledger.occurred_at_values(
            organization_id, params["type"]
        )
        right_occurred = right_branch.ledger.occurred_at_values(
            organization_id, params["type"]
        )
        result = compare_branches(left_occurred, right_occurred, params)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_branch_events(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_branch_event_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]

        # Read-only and organization-scoped, with the same verdict order as
        # the window comparison: the organization decision happens before
        # either branch name is inspected, then left before right.
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Both branches must already exist; a comparison never implicitly
        # creates a branch.
        left_branch = self.server.branches.get(  # type: ignore[attr-defined]
            params["left"]
        )
        if left_branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, left_branch, newline=True):
            return
        right_branch = self.server.branches.get(  # type: ignore[attr-defined]
            params["right"]
        )
        if right_branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, right_branch, newline=True):
            return

        # Each side reads its own ledger in one locked snapshot, then the
        # diff is computed purely from those copies; nothing is written, so
        # identical submissions return byte-identical JSON.
        left_events = list(
            left_branch.ledger.snapshot_for_organization(organization_id).values()
        )
        right_events = list(
            right_branch.ledger.snapshot_for_organization(organization_id).values()
        )
        result = compare_branch_events(left_events, right_events)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_branch_reservations(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_branch_reservation_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]

        # Read-only and organization-scoped, with the same verdict order as
        # the other branch comparisons: the organization decision happens
        # before either branch name is inspected, then left before right.
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Both branches must already exist; a comparison never implicitly
        # creates a branch.
        left_branch = self.server.branches.get(  # type: ignore[attr-defined]
            params["left"]
        )
        if left_branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, left_branch, newline=True):
            return
        right_branch = self.server.branches.get(  # type: ignore[attr-defined]
            params["right"]
        )
        if right_branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, right_branch, newline=True):
            return

        # Each side reads its own inventory in one locked snapshot (only the
        # requested organization's reservations are captured), then the diff
        # is computed purely from those copies; neither branch, the main
        # service, nor any alert state is written, so identical submissions
        # return byte-identical JSON.
        _left_capacities, left_reservations = (
            left_branch.reservations.snapshot_for_organization(organization_id)
        )
        _right_capacities, right_reservations = (
            right_branch.reservations.snapshot_for_organization(organization_id)
        )
        result = compare_branch_reservations(left_reservations, right_reservations)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_snapshot_events(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_snapshot_event_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]

        # Read-only and organization-scoped, with the same verdict order as
        # the other snapshot comparisons: the organization decision happens
        # before either snapshot name is inspected, then left before right.
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Both snapshots must already exist; a comparison never implicitly
        # creates one. The fixed left-then-right order makes the verdict
        # deterministic: a missing left outranks any problem on the right,
        # and a foreign snapshot is forbidden before the other name is even
        # looked up.
        left_snapshot = self.server.snapshots.get(  # type: ignore[attr-defined]
            params["left"]
        )
        if left_snapshot is None:
            self._snapshot_not_found()
            return
        if left_snapshot.organization_id != organization_id:
            self._forbidden(newline=True)
            return
        right_snapshot = self.server.snapshots.get(  # type: ignore[attr-defined]
            params["right"]
        )
        if right_snapshot is None:
            self._snapshot_not_found()
            return
        if right_snapshot.organization_id != organization_id:
            self._forbidden(newline=True)
            return

        # Each side's events were already scoped to the owning organization
        # at capture time; deep copies keep the comparison from sharing
        # mutable snapshot state. Nothing is written, so identical
        # submissions return byte-identical JSON.
        left_events = left_snapshot.events_snapshot()
        right_events = right_snapshot.events_snapshot()
        result = compare_snapshot_events(left_events, right_events)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_snapshot_reservations(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_snapshot_reservation_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]

        # Read-only and organization-scoped, with the same verdict order as
        # the branch comparisons: the organization decision happens before
        # either snapshot name is inspected, then left before right.
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Both snapshots must already exist; a comparison never implicitly
        # creates one. The fixed left-then-right order makes the verdict
        # deterministic: a missing left outranks any problem on the right,
        # and a foreign snapshot is forbidden before the other name is even
        # looked up.
        left_snapshot = self.server.snapshots.get(  # type: ignore[attr-defined]
            params["left"]
        )
        if left_snapshot is None:
            self._snapshot_not_found()
            return
        if left_snapshot.organization_id != organization_id:
            self._forbidden(newline=True)
            return
        right_snapshot = self.server.snapshots.get(  # type: ignore[attr-defined]
            params["right"]
        )
        if right_snapshot is None:
            self._snapshot_not_found()
            return
        if right_snapshot.organization_id != organization_id:
            self._forbidden(newline=True)
            return

        # Each side's reservations were already scoped to the owning
        # organization at capture time; deep copies keep the comparison from
        # sharing mutable snapshot state. Nothing is written, so identical
        # submissions return byte-identical JSON.
        left_reservations = left_snapshot.reservations_snapshot()
        right_reservations = right_snapshot.reservations_snapshot()
        result = compare_snapshot_reservations(
            left_reservations, right_reservations
        )
        self._write_json(HTTPStatus.OK, result, newline=True)

    # ----------------------------------------------------------------- events

    def _create_event(self, ledger: EventLedger, *, newline: bool = False) -> None:
        subject = self._require_subject(newline=newline)
        if subject is None:
            return
        # A write entry point is out of scope for a read credential even
        # before the body is read; the locked commit below re-checks it.
        if subject.role != "write":
            self._forbidden(newline=newline)
            return

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

        organization_id = event["organizationId"]

        def commit() -> tuple[str, dict[str, Any]]:
            return ledger.add(event)

        # The organization/role decision and the ledger mutation are one
        # indivisible step: a forbidden request can never reach the ledger.
        status, result = self.server.tokens.commit_write(  # type: ignore[attr-defined]
            self.token, organization_id, commit
        )
        if status == "forbidden":
            self._forbidden(newline=newline)
            return
        add_status, stored = result
        if add_status == "created":
            self._write_json(HTTPStatus.CREATED, stored, newline=newline)
        elif add_status == "exists":
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
        subject = self._require_subject(newline=newline)
        if subject is None:
            return
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return
        if not self._authorize_organization(
            subject, organization_id, newline=newline
        ):
            return
        events = ledger.list_for_organization(organization_id)
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": organization_id, "events": events},
            newline=newline,
        )

    def _replay_events(self, ledger: EventLedger, query: str) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _replay_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return
        if not self._authorize_organization(
            subject, params["organizationId"], newline=True
        ):
            return
        # Read-only: the replay is a single locked snapshot of the ledger.
        events = ledger.list_for_organization_as_of(
            params["organizationId"], params["asOf"]
        )
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": params["organizationId"], "events": events},
            newline=True,
        )

    def _compare_replays(self, ledger: EventLedger, query: str) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _replay_compare_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return
        if not self._authorize_organization(
            subject, params["organizationId"], newline=True
        ):
            return
        # Read-only: both replays read the same current ledger and nothing
        # is written back, so identical requests return identical bytes.
        from_events = ledger.list_for_organization_as_of(
            params["organizationId"], params["fromAsOf"]
        )
        to_events = ledger.list_for_organization_as_of(
            params["organizationId"], params["toAsOf"]
        )
        result = compare_replays(from_events, to_events)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "fromAsOf": params["fromAsOf"],
                "toAsOf": params["toAsOf"],
                "added": result["added"],
                "removed": result["removed"],
                "changed": result["changed"],
                "unchangedCount": result["unchangedCount"],
            },
            newline=True,
        )

    def _aggregate_events(
        self, ledger: EventLedger, query: str, *, newline: bool = False
    ) -> None:
        subject = self._require_subject(newline=newline)
        if subject is None:
            return
        try:
            params = _aggregate_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return
        if not self._authorize_organization(
            subject, params["organizationId"], newline=newline
        ):
            return

        window_size = params["windowSize"]
        from_value = params["from"]
        to_value = params["to"]
        occurred = ledger.occurred_at_values(
            params["organizationId"], params["type"]
        )
        windows = _windows_from_occurred(
            occurred, window_size, from_value, to_value
        )
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

    def _list_events_by_region(
        self, ledger: EventLedger, query: str
    ) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _region_list_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return
        if not self._authorize_organization(
            subject, params["organizationId"], newline=True
        ):
            return
        events = ledger.list_for_organization_region(
            params["organizationId"], params["region"]
        )
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "region": params["region"],
                "events": events,
            },
            newline=True,
        )

    def _aggregate_events_by_region(
        self, ledger: EventLedger, query: str
    ) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _region_aggregate_params_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return
        if not self._authorize_organization(
            subject, params["organizationId"], newline=True
        ):
            return

        window_size = params["windowSize"]
        from_value = params["from"]
        to_value = params["to"]
        # Read-only: a single locked snapshot of the region's matching events.
        occurred = ledger.occurred_at_values_for_region(
            params["organizationId"], params["type"], params["region"]
        )
        windows = _windows_from_occurred(
            occurred, window_size, from_value, to_value
        )
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
        subject = self._require_subject(newline=newline)
        if subject is None:
            return

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

        if not self._authorize_organization(
            subject, params["organizationId"], newline=newline
        ):
            return

        # Read-only: the ledger is never mutated by a decision request, and
        # the snapshot is taken in a single locked copy for consistency.
        occurred = ledger.occurred_at_values(
            params["organizationId"], params["type"]
        )
        result = evaluate_decision(occurred, params)
        self._write_json(HTTPStatus.OK, result, newline=newline)

    def _allocate_decision(self) -> None:
        subject = self._require_subject(newline=False)
        if subject is None:
            return

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

        if not self._authorize_organization(
            subject, params["organizationId"], newline=False
        ):
            return

        # Read-only and stateless: the plan is computed from the request body
        # alone, so identical submissions and concurrent requests never
        # interfere with each other or with the ledger.
        result = plan_allocation(params)
        self._write_json(HTTPStatus.OK, result)

    # ------------------------------------------------------------------ alerts

    def _evaluate_alert(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        if subject.role != "write":
            self._forbidden(newline=True)
            return

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

        organization_id = params["organizationId"]

        def commit() -> dict[str, Any]:
            # The ledger is never modified; the peak is a single locked
            # snapshot of the matching events. The role/org decision and the
            # alert record share the registry lock, so a read-only credential
            # is rejected before any record call and concurrent threshold
            # hits cannot both open an alert.
            occurred = self.server.ledger.occurred_at_values(  # type: ignore[attr-defined]
                organization_id, params["type"]
            )
            peak = evaluate_decision(occurred, params)
            peak_start = peak["peakStart"]
            peak_count = peak["peakCount"]

            if peak_count < params["threshold"]:
                action = "observe"
                alert_id = None
                suppressed_count = None
            else:
                action, alert = self.server.alerts.record(  # type: ignore[attr-defined]
                    organization_id,
                    params["type"],
                    peak_start,
                    params["threshold"],
                    params["suppressionWindow"],
                )
                alert_id = alert["alertId"]
                suppressed_count = alert["suppressedCount"]

            return {
                "organizationId": organization_id,
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

        status, result = self.server.tokens.commit_write(  # type: ignore[attr-defined]
            self.token, organization_id, commit
        )
        if status == "forbidden":
            self._forbidden(newline=True)
            return
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _list_alerts(self, query: str) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return
        if not self._authorize_organization(subject, organization_id, newline=True):
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
        subject = self._require_subject(newline=newline)
        if subject is None:
            return
        if subject.role != "write":
            self._forbidden(newline=newline)
            return

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

        organization_id = reservation["organizationId"]

        def commit() -> tuple[str, dict[str, Any] | None]:
            return reservations.reserve(reservation)

        # The balance check/deduction stays in the inventory lock; the
        # organization/role decision is wrapped in the same outer registry
        # lock, so a forbidden request never reaches the inventory and cannot
        # race its way past a read-only credential.
        status, result = self.server.tokens.commit_write(  # type: ignore[attr-defined]
            self.token, organization_id, commit
        )
        if status == "forbidden":
            self._forbidden(newline=newline)
            return
        reserve_status, view = result
        if reserve_status == "created":
            self._write_json(HTTPStatus.CREATED, view, newline=newline)
        elif reserve_status == "exists":
            self._write_json(HTTPStatus.OK, view, newline=newline)
        else:
            self._write_json(
                HTTPStatus.CONFLICT, {"error": reserve_status}, newline=newline
            )

    def _list_reservations(
        self,
        reservations: ReservationInventory,
        query: str,
        *,
        newline: bool = False,
    ) -> None:
        subject = self._require_subject(newline=newline)
        if subject is None:
            return
        try:
            organization_id = _organization_id_from_query(query)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return
        if not self._authorize_organization(
            subject, organization_id, newline=newline
        ):
            return
        views = reservations.list_for_organization(organization_id)
        self._write_json(
            HTTPStatus.OK,
            {"organizationId": organization_id, "reservations": views},
            newline=newline,
        )

    # ------------------------------------------------------------------ wiring

    def _require_subject(self, *, newline: bool = True) -> Subject | None:
        """Authenticate the ``Authorization: Bearer ...`` credential.

        Returns the registered :class:`Subject` and remembers the presented
        token on ``self.token``; on any failure writes the 401 response and
        returns ``None``. A missing header, a non-Bearer scheme, an empty or
        malformed token, or an unregistered token are all equally
        unauthenticated.
        """
        header = self.headers.get("Authorization")
        token: str | None = None
        if isinstance(header, str):
            parts = header.split(" ")
            if len(parts) == 2 and parts[0] == "Bearer" and parts[1]:
                token = parts[1]
        if token is None:
            self._unauthorized(newline=newline)
            return None
        subject = self.server.tokens.lookup(token)  # type: ignore[attr-defined]
        if subject is None:
            self._unauthorized(newline=newline)
            return None
        self.token = token
        return subject

    def _authorize_organization(
        self, subject: Subject, organization_id: str, *, newline: bool = True
    ) -> bool:
        """Return True when the subject's organization matches the request."""
        if subject.organization_id != organization_id:
            self._forbidden(newline=newline)
            return False
        return True

    def _allow_branch(
        self, subject: Subject, branch: Branch, *, newline: bool = True
    ) -> bool:
        """Return True when the branch belongs to the subject's organization."""
        return self._authorize_organization(
            subject, branch.organization_id, newline=newline
        )

    def _unauthorized(self, *, newline: bool = True) -> None:
        self._write_json(
            HTTPStatus.UNAUTHORIZED,
            {"error": "unauthorized"},
            newline=newline,
        )

    def _forbidden(self, *, newline: bool = True) -> None:
        self._write_json(
            HTTPStatus.FORBIDDEN,
            {"error": "forbidden"},
            newline=newline,
        )

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
    server.tokens = TokenRegistry()  # type: ignore[attr-defined]
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
