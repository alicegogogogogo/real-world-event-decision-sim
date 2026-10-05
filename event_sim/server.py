from __future__ import annotations

import heapq
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

REGION_ALERT_REQUIRED_FIELDS = (
    "organizationId",
    "region",
    "type",
    "windowSize",
    "threshold",
    "suppressionWindow",
)
REGION_ALERT_OPTIONAL_FIELDS = ("from", "to")
REGION_ALERT_FIELDS = REGION_ALERT_REQUIRED_FIELDS + REGION_ALERT_OPTIONAL_FIELDS

ALLOCATION_REQUIRED_FIELDS = ("organizationId", "demands", "resources")
DEMAND_FIELDS = ("demandId", "units", "priority")
RESOURCE_FIELDS = ("resourceId", "capacity")

ROUTE_ALLOCATION_REQUIRED_FIELDS = (
    "organizationId",
    "edges",
    "demands",
    "resources",
)
ROUTE_EDGE_FIELDS = ("from", "to", "cost")
ROUTE_DEMAND_FIELDS = ("demandId", "location", "units", "priority")
ROUTE_RESOURCE_FIELDS = ("resourceId", "location", "capacity")

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

SNAPSHOT_COMPARE_REQUIRED_FIELDS = (
    "organizationId",
    "left",
    "right",
    "type",
    "windowSize",
    "threshold",
)
SNAPSHOT_COMPARE_OPTIONAL_FIELDS = ("from", "to")
SNAPSHOT_COMPARE_FIELDS = (
    SNAPSHOT_COMPARE_REQUIRED_FIELDS + SNAPSHOT_COMPARE_OPTIONAL_FIELDS
)

REGION_ALERT_COMPARE_REQUIRED_FIELDS = (
    "organizationId",
    "left",
    "right",
    "type",
    "windowSize",
    "threshold",
    "suppressionWindow",
)
REGION_ALERT_COMPARE_OPTIONAL_FIELDS = ("from", "to")
REGION_ALERT_COMPARE_FIELDS = (
    REGION_ALERT_COMPARE_REQUIRED_FIELDS + REGION_ALERT_COMPARE_OPTIONAL_FIELDS
)

# The type-dimension counterpart names the two sides' event types directly,
# so the body carries left/right types but no single shared "type" field.
TYPE_ALERT_COMPARE_REQUIRED_FIELDS = (
    "organizationId",
    "left",
    "right",
    "windowSize",
    "threshold",
    "suppressionWindow",
)
TYPE_ALERT_COMPARE_OPTIONAL_FIELDS = ("from", "to")
TYPE_ALERT_COMPARE_FIELDS = (
    TYPE_ALERT_COMPARE_REQUIRED_FIELDS + TYPE_ALERT_COMPARE_OPTIONAL_FIELDS
)

# The event (non-alert) replay type-dimension comparison carries no
# suppressionWindow and no shared "type" field: left/right name the two
# event types directly, exactly like its alert counterpart.
EVENT_TYPE_COMPARE_REQUIRED_FIELDS = (
    "organizationId",
    "left",
    "right",
    "windowSize",
    "threshold",
)
EVENT_TYPE_COMPARE_OPTIONAL_FIELDS = ("from", "to")
EVENT_TYPE_COMPARE_FIELDS = (
    EVENT_TYPE_COMPARE_REQUIRED_FIELDS + EVENT_TYPE_COMPARE_OPTIONAL_FIELDS
)

# The event (non-alert) region-dimension comparisons — the single-shot
# window comparison and the step-by-step replay comparison — carry no
# suppressionWindow and no shared "type" field: like the type-dimension event
# comparison, left/right name the two sides directly — here the two region
# names, matched verbatim by payload attribution downstream.
EVENT_REGION_COMPARE_REQUIRED_FIELDS = (
    "organizationId",
    "left",
    "right",
    "windowSize",
    "threshold",
)
EVENT_REGION_COMPARE_OPTIONAL_FIELDS = ("from", "to")
EVENT_REGION_COMPARE_FIELDS = (
    EVENT_REGION_COMPARE_REQUIRED_FIELDS + EVENT_REGION_COMPARE_OPTIONAL_FIELDS
)

BRANCH_EVENT_COMPARE_FIELDS = ("organizationId", "left", "right")

BRANCH_RESERVATION_COMPARE_FIELDS = ("organizationId", "left", "right")

BRANCH_RESOURCE_COMPARE_FIELDS = ("organizationId", "left", "right")

SNAPSHOT_EVENT_COMPARE_FIELDS = ("organizationId", "left", "right")

SNAPSHOT_RESERVATION_COMPARE_FIELDS = ("organizationId", "left", "right")

SNAPSHOT_RESOURCE_COMPARE_FIELDS = ("organizationId", "left", "right")

# Resource balances align by resourceId; only these three balances
# participate in the same/diff decision for an identifier present on a side.
RESOURCE_COMPARE_FIELDS = ("capacity", "occupied", "remaining")

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


class DuplicateJsonKeyError(ValueError):
    """A JSON object carried the same key more than once."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that refuses an object with a repeated key."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate field: {key}")
        result[key] = value
    return result


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
    a resource nor lose each other's writes. A cancelled reservation is
    removed from the active records but leaves a tombstone behind, so its
    reservationId stays reserved forever and a repeated cancel replays the
    same cancelled view. State lives only for the lifetime of this instance.
    """

    def __init__(self) -> None:
        self._capacities: dict[str, int] = {}
        self._reservations: dict[str, dict[str, Any]] = {}
        self._cancelled: dict[str, dict[str, Any]] = {}
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
        exists with different fields, ``"reservation_cancelled"`` when the
        reservationId was cancelled before (it is never reusable),
        ``"capacity_conflict"`` when the
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
            if reservation["reservationId"] in self._cancelled:
                # A cancelled reservationId is never reusable, whatever
                # fields the resubmission carries.
                return "reservation_cancelled", None

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

    @staticmethod
    def _cancelled_view(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "organizationId": record["organizationId"],
            "reservationId": record["reservationId"],
            "resourceId": record["resourceId"],
            "quantity": record["quantity"],
            "status": "cancelled",
        }

    def cancel(
        self, organization_id: str, reservation_id: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Cancel a reservation or reconcile a replay, atomically.

        Returns ``(status, view)`` where status is ``"cancelled"`` (the
        reservation was active and is now released, or it was already
        cancelled and the same view is replayed without touching capacity
        again), ``"forbidden"`` when the reservationId belongs to another
        organization, or ``"not_found"`` when the reservationId never
        existed. Only the first ``"cancelled"`` mutates state: the record
        leaves the active reservations, its quantity is released, and a
        tombstone keeps the reservationId permanently unusable. The
        resource's recorded capacity is never rewritten.
        """
        with self._lock:
            record = self._reservations.get(reservation_id)
            if record is None:
                record = self._cancelled.get(reservation_id)
            if record is None:
                return "not_found", None
            if record["organizationId"] != organization_id:
                return "forbidden", None
            self._reservations.pop(reservation_id, None)
            self._cancelled[reservation_id] = record
            return "cancelled", self._cancelled_view(record)

    def cancelled_for_organization(
        self, organization_id: str
    ) -> dict[str, dict[str, Any]]:
        """Locked deep copies of one organization's cancellation tombstones.

        Snapshots carry these so a branch derived from a snapshot inherits
        the cancelled state of its organization's reservationIds; the
        tombstones never appear in listings, balances, or counts.
        """
        with self._lock:
            return {
                reservation_id: dict(record)
                for reservation_id, record in self._cancelled.items()
                if record["organizationId"] == organization_id
            }

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

    def resource_balances_for_organization(
        self, organization_id: str
    ) -> dict[str, dict[str, int]]:
        """Locked copy of one organization's per-resource balances.

        Each resource the organization itself reserves maps to its
        ``capacity``, ``occupied`` (the total quantity reserved against it by
        that organization's records) and ``remaining`` (``capacity -
        occupied``). The balances are recomputed from deep copies taken under
        one lock, so a read-only comparison sees a consistent snapshot and
        shares no mutable inventory state. Reservations of other
        organizations never contribute.
        """
        with self._lock:
            capacities = {
                resource_id: capacity
                for resource_id, capacity in self._capacities.items()
                if any(
                    record["resourceId"] == resource_id
                    and record["organizationId"] == organization_id
                    for record in self._reservations.values()
                )
            }
            reservations = {
                reservation_id: dict(record)
                for reservation_id, record in self._reservations.items()
                if record["organizationId"] == organization_id
            }
        balances: dict[str, dict[str, int]] = {}
        for resource_id, capacity in capacities.items():
            occupied = sum(
                record["quantity"]
                for record in reservations.values()
                if record["resourceId"] == resource_id
            )
            balances[resource_id] = {
                "capacity": capacity,
                "occupied": occupied,
                "remaining": capacity - occupied,
            }
        return balances

    def restore(
        self,
        capacities: dict[str, int],
        reservations: dict[str, dict[str, Any]],
        cancelled: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Replace capacities, reservations, and tombstones with deep copies."""
        with self._lock:
            self._capacities = dict(capacities)
            self._reservations = {
                reservation_id: dict(record)
                for reservation_id, record in reservations.items()
            }
            self._cancelled = {
                reservation_id: dict(record)
                for reservation_id, record in (cancelled or {}).items()
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
        cancelled: dict[str, dict[str, Any]] | None = None,
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
        # Cancellation tombstones are captured so a derived branch inherits
        # the cancelled state; they never count as reservations or balances.
        self._cancelled = {
            reservation_id: dict(record)
            for reservation_id, record in (cancelled or {}).items()
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
        reservations.restore(self._capacities, self._reservations, self._cancelled)
        return ledger, reservations

    def events_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the captured events, keyed by eventId.

        A snapshot holds only its owning organization's events, so the copy
        needs no further organization filtering. It is immutable after
        capture; the copy keeps a read-only comparison from sharing mutable
        references with the snapshot.
        """
        return {event_id: dict(event) for event_id, event in self._events.items()}

    def occurred_at_values(self, event_type: str) -> list[int]:
        """Sorted occurredAt values of the captured events of one type.

        A snapshot holds only its owning organization's events and is
        immutable after capture, so the filtered copy needs neither a lock
        nor an organization filter.
        """
        values = [
            event["occurredAt"]
            for event in self._events.values()
            if event["type"] == event_type
        ]
        values.sort()
        return values

    def events_snapshot_for_region(
        self, region: str
    ) -> dict[str, dict[str, Any]]:
        """Return a deep copy of the captured events attributed to one region.

        Region attribution follows the same verbatim rule as the main
        ledger: only a non-empty string payload ``region`` equal to
        ``region`` matches, and an unknown region simply matches zero
        events. A snapshot holds only its owning organization's events and
        is immutable after capture, so the filtered copy needs neither a
        lock nor an organization filter — the same captured-event contract
        the region aggregate counts from, so the listing and the aggregate
        hit exactly the same events.
        """
        return {
            event_id: dict(event)
            for event_id, event in self._events.items()
            if event_region(event) == region
        }

    def occurred_at_values_for_region(
        self, event_type: str, region: str
    ) -> list[int]:
        """Sorted occurredAt values of the captured events of one type in one region.

        A snapshot holds only its owning organization's events and is
        immutable after capture, so the filtered copy needs neither a lock
        nor an organization filter. Region attribution follows the same
        verbatim rule as the main ledger: only a non-empty string payload
        ``region`` equal to ``region`` matches, and an unknown region simply
        matches zero events.
        """
        values = [
            event["occurredAt"]
            for event in self._events.values()
            if event["type"] == event_type and event_region(event) == region
        ]
        values.sort()
        return values

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

    def resource_balances(self) -> dict[str, dict[str, int]]:
        """Return a deep copy of the captured per-resource balances.

        Each resource with a recorded capacity maps to its ``capacity``,
        ``occupied`` (the total captured quantity reserved against it) and
        ``remaining`` (``capacity - occupied``). A snapshot holds only its
        owning organization's resources, so the copy needs no further
        organization filtering; it is immutable after capture, so the copy
        keeps a read-only comparison from sharing mutable snapshot state.
        """
        balances: dict[str, dict[str, int]] = {}
        for resource_id, capacity in self._capacities.items():
            occupied = sum(
                record["quantity"]
                for record in self._reservations.values()
                if record["resourceId"] == resource_id
            )
            balances[resource_id] = {
                "capacity": capacity,
                "occupied": occupied,
                "remaining": capacity - occupied,
            }
        return balances


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
                reservations.cancelled_for_organization(organization_id),
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

    Region-dimension alerts live in a parallel namespace keyed by
    organization, region, and type: the two alert kinds never suppress each
    other, but both draw their identifiers from one global counter shared
    across all organizations, regions, and types.
    """

    def __init__(self) -> None:
        self._alerts: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._region_alerts: dict[
            str, dict[str, dict[str, list[dict[str, Any]]]]
        ] = {}
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

    def record_for_region(
        self,
        organization_id: str,
        region: str,
        event_type: str,
        peak_start: int,
        threshold: int,
        suppression_window: int,
    ) -> tuple[str, dict[str, Any]]:
        """Raise or suppress a region-dimension alert.

        Mirrors :meth:`record` with the suppression scope narrowed to one
        organization, region, and type. Region alerts share the single
        global identifier counter with organization-dimension alerts but
        never suppress — and are never suppressed by — that other kind.
        """
        with self._lock:
            by_region = self._region_alerts.setdefault(organization_id, {})
            by_type = by_region.setdefault(region, {})
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
                "region": region,
                "type": event_type,
                "peakStart": peak_start,
                "threshold": threshold,
                "suppressedCount": 0,
            }
            alerts.append(alert)
            return "escalate", alert

    def list_for_organization_region(
        self, organization_id: str, region: str
    ) -> list[dict[str, Any]]:
        """List one organization's region-dimension alerts for one region.

        An unknown organization or region simply has no alerts; the region
        is matched verbatim. Ordering matches :meth:`list_for_organization`.
        """
        with self._lock:
            by_region = self._region_alerts.get(organization_id, {})
            by_type = by_region.get(region, {})
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


def _replay_decision_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /events/replay/decisions query string.

    organizationId, type, windowSize and threshold are each required
    exactly once and non-empty; windowSize and threshold must be positive
    integer text. from and to must be both absent or both present exactly
    once, each a non-negative integer, with from <= to. The rules mirror
    /events/aggregate plus a threshold parameter, expressed as query text.
    """
    params = parse_qs(query, keep_blank_values=True)

    organization_id = _single_non_empty_text(params, "organizationId")
    event_type = _single_non_empty_text(params, "type")

    window_size = _integer_text(_single_non_empty_text(params, "windowSize"))
    if window_size is None or window_size <= 0:
        raise EventValidationError("windowSize must be a positive integer")
    threshold = _integer_text(_single_non_empty_text(params, "threshold"))
    if threshold is None or threshold <= 0:
        raise EventValidationError("threshold must be a positive integer")

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
        "threshold": threshold,
        "from": from_value,
        "to": to_value,
    }


def _region_replay_decision_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /events/region/replay/decisions query string.

    The region-dimension counterpart of
    :func:`_replay_decision_params_from_query`: organizationId, region, type,
    windowSize and threshold are each required exactly once and non-empty;
    windowSize and threshold must be positive integer text. from and to must
    be both absent or both present exactly once, each a non-negative integer,
    with from <= to. The region is matched verbatim downstream.
    """
    params = parse_qs(query, keep_blank_values=True)

    organization_id = _single_non_empty_text(params, "organizationId")
    region = _single_non_empty_text(params, "region")
    event_type = _single_non_empty_text(params, "type")

    window_size = _integer_text(_single_non_empty_text(params, "windowSize"))
    if window_size is None or window_size <= 0:
        raise EventValidationError("windowSize must be a positive integer")
    threshold = _integer_text(_single_non_empty_text(params, "threshold"))
    if threshold is None or threshold <= 0:
        raise EventValidationError("threshold must be a positive integer")

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
        "threshold": threshold,
        "from": from_value,
        "to": to_value,
    }


def _region_alert_replay_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /alerts/region/replay/decisions query string.

    The alert-replay counterpart of
    :func:`_region_replay_decision_params_from_query`: organizationId,
    region, type, windowSize, threshold and suppressionWindow are each
    required exactly once and non-empty; windowSize, threshold and
    suppressionWindow must be positive integer text. from and to must be
    both absent or both present exactly once, each a non-negative integer,
    with from <= to. The region is matched verbatim downstream.
    """
    params = parse_qs(query, keep_blank_values=True)

    organization_id = _single_non_empty_text(params, "organizationId")
    region = _single_non_empty_text(params, "region")
    event_type = _single_non_empty_text(params, "type")

    window_size = _integer_text(_single_non_empty_text(params, "windowSize"))
    if window_size is None or window_size <= 0:
        raise EventValidationError("windowSize must be a positive integer")
    threshold = _integer_text(_single_non_empty_text(params, "threshold"))
    if threshold is None or threshold <= 0:
        raise EventValidationError("threshold must be a positive integer")
    suppression_window = _integer_text(
        _single_non_empty_text(params, "suppressionWindow")
    )
    if suppression_window is None or suppression_window <= 0:
        raise EventValidationError("suppressionWindow must be a positive integer")

    from_value, to_value = _range_pair_from_params(params)

    return {
        "organizationId": organization_id,
        "region": region,
        "type": event_type,
        "windowSize": window_size,
        "threshold": threshold,
        "suppressionWindow": suppression_window,
        "from": from_value,
        "to": to_value,
    }


def _alert_replay_params_from_query(query: str) -> dict[str, Any]:
    """Validate the /alerts/replay/decisions query string.

    The organization-dimension alert-replay counterpart of
    :func:`_replay_decision_params_from_query`: organizationId, type,
    windowSize, threshold and suppressionWindow are each required exactly
    once and non-empty; windowSize, threshold and suppressionWindow must be
    positive integer text. from and to must be both absent or both present
    exactly once, each a non-negative integer, with from <= to.
    """
    params = parse_qs(query, keep_blank_values=True)

    organization_id = _single_non_empty_text(params, "organizationId")
    event_type = _single_non_empty_text(params, "type")

    window_size = _integer_text(_single_non_empty_text(params, "windowSize"))
    if window_size is None or window_size <= 0:
        raise EventValidationError("windowSize must be a positive integer")
    threshold = _integer_text(_single_non_empty_text(params, "threshold"))
    if threshold is None or threshold <= 0:
        raise EventValidationError("threshold must be a positive integer")
    suppression_window = _integer_text(
        _single_non_empty_text(params, "suppressionWindow")
    )
    if suppression_window is None or suppression_window <= 0:
        raise EventValidationError("suppressionWindow must be a positive integer")

    from_value, to_value = _range_pair_from_params(params)

    return {
        "organizationId": organization_id,
        "type": event_type,
        "windowSize": window_size,
        "threshold": threshold,
        "suppressionWindow": suppression_window,
        "from": from_value,
        "to": to_value,
    }


def _range_pair_from_params(
    params: dict[str, list[str]],
) -> tuple[int | None, int | None]:
    """Validate an optional paired ``from``/``to`` range from parsed query.

    Both must be absent or each present exactly once, each non-negative
    integer text, with from <= to. Shared by the two alert-replay query
    parsers so their range contract and wording stay identical.
    """
    from_values = params.get("from")
    to_values = params.get("to")
    if from_values is None and to_values is None:
        return None, None
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
    return from_value, to_value


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
    zero; the rows come from the same helper as the aggregate entry point
    and the peak from the shared tie-breaking helper, so the aggregate,
    decision, and replay-step views agree item by item on the same state.
    """
    window_size = params["windowSize"]
    threshold = params["threshold"]
    from_value = params["from"]
    to_value = params["to"]

    windows = _windows_from_occurred(occurred, window_size, from_value, to_value)
    peak = _peak_from_windows(windows, threshold)
    return {
        "organizationId": params["organizationId"],
        "type": params["type"],
        "windowSize": window_size,
        "from": from_value,
        "to": to_value,
        "peakStart": peak["peakStart"],
        "peakCount": peak["peakCount"],
        "action": peak["action"],
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


def validate_region_alert_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /alerts/region/evaluate.

    Required fields: organizationId, region, type, windowSize, threshold,
    and suppressionWindow. Optional fields: from and to, which must appear
    together. No other fields are allowed. The rules mirror
    ``POST /alerts/evaluate`` with an added ``region`` field; the region is
    matched verbatim downstream.
    """
    if not isinstance(data, dict):
        raise EventValidationError("region alert body must be a JSON object")

    keys = set(data)
    required = set(REGION_ALERT_REQUIRED_FIELDS)
    allowed = set(REGION_ALERT_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "region", "type"):
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
        "region": data["region"],
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


def validate_route_allocation_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /decisions/route-allocate.

    Required fields: organizationId, edges, demands, resources, and no
    others. Each edge/demand/resource element must be an object with exactly
    its fixed fields; demand and resource identifiers must be unique within
    the request and non-blank, and no two edges may share the same directed
    (from, to) endpoint pair.
    """
    if not isinstance(data, dict):
        raise EventValidationError("route allocation body must be a JSON object")

    keys = set(data)
    required = set(ROUTE_ALLOCATION_REQUIRED_FIELDS)
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

    raw_edges = data["edges"]
    raw_demands = data["demands"]
    raw_resources = data["resources"]
    if (
        not isinstance(raw_edges, list)
        or not isinstance(raw_demands, list)
        or not isinstance(raw_resources, list)
    ):
        raise EventValidationError("edges, demands and resources must be arrays")

    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, element in enumerate(raw_edges):
        if not isinstance(element, dict):
            raise EventValidationError(f"edges[{index}] must be a JSON object")
        element_keys = set(element)
        expected = set(ROUTE_EDGE_FIELDS)
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
                f"edges[{index}]: {'; '.join(element_detail)}"
            )
        from_location = element["from"]
        to_location = element["to"]
        if not isinstance(from_location, str) or not from_location.strip():
            raise EventValidationError(
                f"edges[{index}].from must be a non-empty string"
            )
        if not isinstance(to_location, str) or not to_location.strip():
            raise EventValidationError(
                f"edges[{index}].to must be a non-empty string"
            )
        if not _is_positive_integer(element["cost"]):
            raise EventValidationError(
                f"edges[{index}].cost must be a positive integer"
            )
        pair = (from_location, to_location)
        if pair in seen_pairs:
            raise EventValidationError(
                f"duplicate edge endpoints: {from_location} -> {to_location}"
            )
        seen_pairs.add(pair)
        edges.append(
            {"from": from_location, "to": to_location, "cost": element["cost"]}
        )

    demands: list[dict[str, Any]] = []
    seen_demand_ids: set[str] = set()
    for index, element in enumerate(raw_demands):
        if not isinstance(element, dict):
            raise EventValidationError(f"demands[{index}] must be a JSON object")
        element_keys = set(element)
        expected = set(ROUTE_DEMAND_FIELDS)
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
        location = element["location"]
        if not isinstance(location, str) or not location.strip():
            raise EventValidationError(
                f"demands[{index}].location must be a non-empty string"
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
                "location": location,
                "units": element["units"],
                "priority": element["priority"],
            }
        )

    resources: list[dict[str, Any]] = []
    seen_resource_ids: set[str] = set()
    for index, element in enumerate(raw_resources):
        if not isinstance(element, dict):
            raise EventValidationError(f"resources[{index}] must be a JSON object")
        element_keys = set(element)
        expected = set(ROUTE_RESOURCE_FIELDS)
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
        if resource_id in seen_resource_ids:
            raise EventValidationError(
                f"duplicate resourceId: {resource_id}"
            )
        location = element["location"]
        if not isinstance(location, str) or not location.strip():
            raise EventValidationError(
                f"resources[{index}].location must be a non-empty string"
            )
        if not _is_positive_integer(element["capacity"]):
            raise EventValidationError(
                f"resources[{index}].capacity must be a positive integer"
            )
        seen_resource_ids.add(resource_id)
        resources.append(
            {
                "resourceId": resource_id,
                "location": location,
                "capacity": element["capacity"],
            }
        )

    return {
        "organizationId": organization_id,
        "edges": edges,
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


def validate_reservation_cancel_request(data: Any) -> tuple[str, str]:
    """Validate a decoded JSON body for POST /reservations/cancel.

    Exactly the ``organizationId`` and ``reservationId`` fields may be
    present, and both must be non-empty strings.
    """
    if not isinstance(data, dict):
        raise EventValidationError("reservation cancel body must be a JSON object")

    keys = set(data)
    expected = {"organizationId", "reservationId"}
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "reservationId"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return data["organizationId"], data["reservationId"]


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


def _validate_window_compare_request(
    data: Any,
    *,
    body_kind: str,
    required_fields: tuple[str, ...],
    allowed_fields: tuple[str, ...],
    positive_fields: tuple[str, ...] = ("windowSize", "threshold"),
) -> dict[str, Any]:
    """Validate a decoded JSON body for a window-decision comparison.

    Shared by the branch and snapshot window comparisons, whose request
    contracts are identical: required fields organizationId, left, right,
    type, windowSize, threshold; optional fields from and to, which must
    appear together. No other fields are allowed. ``left`` and ``right``
    are branch or snapshot names; the same name on both sides is legal.

    The region alert replay comparison adds the required positive-integer
    field ``suppressionWindow`` via ``positive_fields``; that field is only
    validated/echoed when the caller lists it.
    """
    if not isinstance(data, dict):
        raise EventValidationError(f"{body_kind} body must be a JSON object")

    keys = set(data)
    required = set(required_fields)
    allowed = set(allowed_fields)
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

    for field in positive_fields:
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

    result = {
        "organizationId": data["organizationId"],
        "left": data["left"],
        "right": data["right"],
        "type": data["type"],
        "from": from_value,
        "to": to_value,
    }
    for field in positive_fields:
        result[field] = data[field]
    return result


def validate_branch_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /branches/compare.

    Required fields: organizationId, left, right, type, windowSize,
    threshold. Optional fields: from and to, which must appear together. No
    other fields are allowed. ``left`` and ``right`` are branch names; the
    same name on both sides is legal.
    """
    return _validate_window_compare_request(
        data,
        body_kind="branch comparison",
        required_fields=BRANCH_COMPARE_REQUIRED_FIELDS,
        allowed_fields=BRANCH_COMPARE_FIELDS,
    )


def validate_branch_replay_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /branches/compare/replay/decisions.

    The request contract is identical to ``POST /branches/compare``:
    required fields organizationId, left, right, type, windowSize,
    threshold; optional fields from and to, which must appear together. No
    other fields are allowed. ``left`` and ``right`` are branch names; the
    same name on both sides is legal.
    """
    return _validate_window_compare_request(
        data,
        body_kind="branch replay comparison",
        required_fields=BRANCH_COMPARE_REQUIRED_FIELDS,
        allowed_fields=BRANCH_COMPARE_FIELDS,
    )


def validate_snapshot_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /snapshots/compare.

    Required fields: organizationId, left, right, type, windowSize,
    threshold. Optional fields: from and to, which must appear together. No
    other fields are allowed. ``left`` and ``right`` are snapshot names; the
    same name on both sides is legal.
    """
    return _validate_window_compare_request(
        data,
        body_kind="snapshot comparison",
        required_fields=SNAPSHOT_COMPARE_REQUIRED_FIELDS,
        allowed_fields=SNAPSHOT_COMPARE_FIELDS,
    )


def validate_snapshot_replay_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /snapshots/compare/replay/decisions.

    The request contract is identical to ``POST /snapshots/compare``:
    required fields organizationId, left, right, type, windowSize,
    threshold; optional fields from and to, which must appear together. No
    other fields are allowed. ``left`` and ``right`` are snapshot names; the
    same name on both sides is legal.
    """
    return _validate_window_compare_request(
        data,
        body_kind="snapshot replay comparison",
        required_fields=SNAPSHOT_COMPARE_REQUIRED_FIELDS,
        allowed_fields=SNAPSHOT_COMPARE_FIELDS,
    )


def validate_region_alert_replay_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /alerts/regions/compare/replay/decisions.

    The region alert replay counterpart of the window comparisons: required
    fields organizationId, left, right, type, windowSize, threshold, and
    suppressionWindow; optional fields from and to, which must appear
    together. No other fields are allowed. ``left`` and ``right`` are region
    names matched verbatim downstream; the same region name on both sides is
    legal, and an unknown region simply matches zero events.
    """
    return _validate_window_compare_request(
        data,
        body_kind="region alert replay comparison",
        required_fields=REGION_ALERT_COMPARE_REQUIRED_FIELDS,
        allowed_fields=REGION_ALERT_COMPARE_FIELDS,
        positive_fields=("windowSize", "threshold", "suppressionWindow"),
    )


def validate_type_alert_replay_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /alerts/types/compare/replay/decisions.

    The type-dimension counterpart of
    :func:`validate_region_alert_replay_compare_request`: required fields
    organizationId, left, right, windowSize, threshold, and
    suppressionWindow; optional fields from and to, which must appear
    together. Unlike the region comparison there is no shared ``type``
    field — ``left`` and ``right`` themselves name the two event types,
    matched verbatim downstream; the same type name on both sides is
    legal, and an unknown type simply matches zero events.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "type alert replay comparison body must be a JSON object"
        )

    keys = set(data)
    required = set(TYPE_ALERT_COMPARE_REQUIRED_FIELDS)
    allowed = set(TYPE_ALERT_COMPARE_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "left", "right"):
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
        "left": data["left"],
        "right": data["right"],
        "windowSize": data["windowSize"],
        "threshold": data["threshold"],
        "suppressionWindow": data["suppressionWindow"],
        "from": from_value,
        "to": to_value,
    }


def validate_event_type_replay_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /events/types/compare/replay/decisions.

    The event-replay counterpart of
    :func:`validate_type_alert_replay_compare_request` minus the alert
    machinery: required fields organizationId, left, right, windowSize and
    threshold; optional fields from and to, which must appear together.
    Unlike the region comparisons there is no shared ``type`` field —
    ``left`` and ``right`` themselves name the two event types, matched
    verbatim downstream; the same type name on both sides is legal, and an
    unknown type simply matches zero events.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "event type replay comparison body must be a JSON object"
        )

    keys = set(data)
    required = set(EVENT_TYPE_COMPARE_REQUIRED_FIELDS)
    allowed = set(EVENT_TYPE_COMPARE_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "left", "right"):
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
        "windowSize": data["windowSize"],
        "threshold": data["threshold"],
        "from": from_value,
        "to": to_value,
    }


def validate_event_region_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /events/regions/compare.

    The single-shot counterpart of
    :func:`validate_event_region_replay_compare_request` with the identical
    field contract: required fields organizationId, left, right, windowSize
    and threshold; optional fields from and to, which must appear together.
    ``left`` and ``right`` name the two sides' regions directly (there is no
    shared ``type`` field); the region names are matched verbatim against
    payload attribution downstream, the same region name on both sides is
    legal, and an unknown region simply matches zero events.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "event region comparison body must be a JSON object"
        )

    keys = set(data)
    required = set(EVENT_REGION_COMPARE_REQUIRED_FIELDS)
    allowed = set(EVENT_REGION_COMPARE_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "left", "right"):
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
        "windowSize": data["windowSize"],
        "threshold": data["threshold"],
        "from": from_value,
        "to": to_value,
    }


def validate_event_region_replay_compare_request(data: Any) -> dict[str, Any]:
    """Validate a decoded JSON body for POST /events/regions/compare/replay/decisions.

    The region counterpart of
    :func:`validate_event_type_replay_compare_request`: required fields
    organizationId, left, right, windowSize and threshold; optional fields
    from and to, which must appear together. ``left`` and ``right`` name the
    two sides' regions directly (there is no shared ``type`` field); the
    region names are matched verbatim against payload attribution downstream,
    the same region name on both sides is legal, and an unknown region
    simply matches zero events.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "event region replay comparison body must be a JSON object"
        )

    keys = set(data)
    required = set(EVENT_REGION_COMPARE_REQUIRED_FIELDS)
    allowed = set(EVENT_REGION_COMPARE_FIELDS)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    detail = []
    if missing:
        detail.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        detail.append(f"unexpected fields: {', '.join(unknown)}")
    if detail:
        raise EventValidationError("; ".join(detail))

    for field in ("organizationId", "left", "right"):
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


def validate_branch_resource_compare_request(data: Any) -> dict[str, str]:
    """Validate a decoded JSON body for POST /branches/compare/resources.

    Exactly ``organizationId``, ``left`` and ``right`` may be present, each
    a non-empty string. ``left`` and ``right`` are branch names; the same
    name on both sides is legal.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "branch resource comparison body must be a JSON object"
        )

    keys = set(data)
    expected = set(BRANCH_RESOURCE_COMPARE_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in BRANCH_RESOURCE_COMPARE_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return {field: data[field] for field in BRANCH_RESOURCE_COMPARE_FIELDS}


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


def validate_snapshot_resource_compare_request(data: Any) -> dict[str, str]:
    """Validate a decoded JSON body for POST /snapshots/compare/resources.

    Exactly ``organizationId``, ``left`` and ``right`` may be present, each
    a non-empty string. ``left`` and ``right`` are snapshot names; the same
    name on both sides is legal.
    """
    if not isinstance(data, dict):
        raise EventValidationError(
            "snapshot resource comparison body must be a JSON object"
        )

    keys = set(data)
    expected = set(SNAPSHOT_RESOURCE_COMPARE_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unexpected fields: {', '.join(unknown)}")
        raise EventValidationError("; ".join(detail))

    for field in SNAPSHOT_RESOURCE_COMPARE_FIELDS:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise EventValidationError(f"{field} must be a non-empty string")
    return {field: data[field] for field in SNAPSHOT_RESOURCE_COMPARE_FIELDS}


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


def _shortest_route(
    adjacency: dict[str, list[tuple[str, int]]], source: str, target: str
) -> tuple[list[str] | None, int | None]:
    """Return the deterministic shortest directed route from source to target.

    The result is the lowest-cost path; among equal-cost paths the node
    sequence that is smallest in Unicode code-point order wins. A source
    equal to the target routes through itself alone at zero cost. When the
    target is unreachable the result is ``(None, None)``.
    """
    if source == target:
        return [source], 0
    # Heap entries are (cost, path, node); the path's last element is the
    # node, so entries never compare equal and the pop order is fully
    # deterministic. Positive edge costs make the first settled label of a
    # node the lexicographically smallest (cost, path) over all its routes.
    heap: list[tuple[int, tuple[str, ...], str]] = [(0, (source,), source)]
    settled: set[str] = set()
    while heap:
        cost, path, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        if node == target:
            return list(path), cost
        for neighbor, weight in adjacency.get(node, ()):
            if neighbor not in settled:
                heapq.heappush(heap, (cost + weight, path + (neighbor,), neighbor))
    return None, None


def plan_route_allocation(params: dict[str, Any]) -> dict[str, Any]:
    """Allocate whole demands to resources under directed routing constraints.

    Demands are handled by priority descending, then demandId in Unicode
    code-point order. A demand goes wholly to the reachable resource with
    enough remaining capacity whose shortest route from its location to the
    demand's destination costs the least; ties resolve by resourceId in
    code-point order. Capacity is deducted immediately and never oversold.
    """
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for edge in params["edges"]:
        adjacency.setdefault(edge["from"], []).append((edge["to"], edge["cost"]))
    for neighbors in adjacency.values():
        neighbors.sort()

    remaining = {
        resource["resourceId"]: resource["capacity"]
        for resource in params["resources"]
    }
    locations = {
        resource["resourceId"]: resource["location"]
        for resource in params["resources"]
    }
    ordered_resource_ids = sorted(remaining)
    ordered_demands = sorted(
        params["demands"],
        key=lambda demand: (-demand["priority"], demand["demandId"]),
    )

    assignments: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    total_units = 0
    total_travel_cost = 0
    for demand in ordered_demands:
        demand_id = demand["demandId"]
        units = demand["units"]
        capacity_available = False
        best: tuple[int, str, list[str]] | None = None
        for resource_id in ordered_resource_ids:
            if remaining[resource_id] < units:
                continue
            capacity_available = True
            path, cost = _shortest_route(
                adjacency, locations[resource_id], demand["location"]
            )
            if path is None or cost is None:
                continue
            if best is None or (cost, resource_id) < best[:2]:
                best = (cost, resource_id, path)
        if best is None:
            reason = (
                "unreachable" if capacity_available else "insufficient_capacity"
            )
            unassigned.append({"demandId": demand_id, "reason": reason})
            continue
        cost, resource_id, path = best
        remaining[resource_id] -= units
        assignments.append(
            {
                "demandId": demand_id,
                "resourceId": resource_id,
                "units": units,
                "path": path,
                "travelCost": cost,
            }
        )
        total_units += units
        total_travel_cost += cost

    return {
        "organizationId": params["organizationId"],
        "assignments": assignments,
        "unassigned": unassigned,
        "totalUnits": total_units,
        "totalTravelCost": total_travel_cost,
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


def _peak_from_windows(
    windows: list[dict[str, int]], threshold: int
) -> dict[str, Any]:
    """Pick the peak from aggregate-style window rows.

    The peak is the largest count; ties resolve to the earliest start,
    which is the first row to reach that count because window rows are
    sorted by start ascending. With no rows (or only zero counts) the
    peak count is zero, the start is null, and the action is ``observe``.
    """
    peak_start: int | None = None
    peak_count = 0
    for window in windows:
        if window["count"] > peak_count:
            peak_count = window["count"]
            peak_start = window["start"]
    action = "escalate" if peak_count >= threshold else "observe"
    return {
        "peakStart": peak_start,
        "peakCount": peak_count,
        "action": action,
    }


def replay_decision_steps(
    events: list[dict[str, Any]], params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replay matching events one at a time, recomputing counts and the peak.

    ``events`` is a consistent locked snapshot already filtered to one
    organization (and, for the region entry points, one region). Events of
    other types are skipped without opening a step; the matching events are
    re-sorted here by ``occurredAt`` ascending and then ``eventId`` in
    Unicode code-point order — the exact tie order used by the alert
    replays, so both replay kinds line up identifier by identifier on the
    same state. Each remaining event is added in that replay order and the
    window rows and peak decision are recomputed from the accumulated
    prefix through the same helpers as the aggregate and decision entry
    points, so every step agrees item by item with what those endpoints
    return on the same state.
    """
    window_size = params["windowSize"]
    threshold = params["threshold"]
    from_value = params["from"]
    to_value = params["to"]

    matching = [event for event in events if event["type"] == params["type"]]
    matching.sort(key=lambda event: (event["occurredAt"], event["eventId"]))

    steps: list[dict[str, Any]] = []
    accumulated: list[int] = []
    for event in matching:
        accumulated.append(event["occurredAt"])
        windows = _windows_from_occurred(
            accumulated, window_size, from_value, to_value
        )
        peak = _peak_from_windows(windows, threshold)
        steps.append(
            {
                "eventId": event["eventId"],
                "occurredAt": event["occurredAt"],
                "windows": windows,
                "peakStart": peak["peakStart"],
                "peakCount": peak["peakCount"],
                "action": peak["action"],
            }
        )
    return steps


def alert_replay_steps(
    events: list[dict[str, Any]], params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replay matching events, simulating the alert suppression rule.

    Shared by the organization- and region-dimension alert replays, so the
    two entry points share one ordering, window, peak, and suppression
    contract. ``events`` is a consistent locked snapshot already filtered to
    the caller's organization (and, for the region dimension, the verbatim
    region); events of other types are skipped without opening a step. The
    matching events are re-sorted here by ``occurredAt`` ascending and then
    ``eventId`` in Unicode code-point order — the exact tie order of
    ``GET /events`` — so the two dimensions enter the replay in one identical
    order regardless of the list that supplied them.

    Each matching event is added in that replay order and the window rows and
    peak are recomputed from the accumulated prefix through the same helpers
    as the aggregate and decision entry points, then the exact suppression
    rule of ``POST /alerts/evaluate`` is layered on against the most recent
    alert *within this replay* for the same dimension scope. The simulation
    never touches the alert store: alert identifiers run their own one-based
    sequence and the suppressed count is the running total against the
    simulated prior alert, so the replay is read-only and repeated calls are
    byte-for-byte identical.

    A step whose peak is below the threshold is ``observe`` with null alert
    identity. A threshold hit with no prior simulated alert, or whose peak
    start is at least ``suppressionWindow`` after the prior one, opens a new
    simulated alert (``escalate`` with the next ``alert-N`` id and a zero
    suppressed count); a closer peak is ``suppress`` against the prior alert
    and carries the new cumulative suppressed count.
    """
    window_size = params["windowSize"]
    threshold = params["threshold"]
    suppression_window = params["suppressionWindow"]
    from_value = params["from"]
    to_value = params["to"]

    matching = [event for event in events if event["type"] == params["type"]]
    matching.sort(key=lambda event: (event["occurredAt"], event["eventId"]))

    steps: list[dict[str, Any]] = []
    accumulated: list[int] = []
    prior_start: int | None = None
    prior_alert_id: str | None = None
    suppressed_count = 0
    alert_counter = 0
    for event in matching:
        accumulated.append(event["occurredAt"])
        windows = _windows_from_occurred(
            accumulated, window_size, from_value, to_value
        )
        peak = _peak_from_windows(windows, threshold)
        peak_start = peak["peakStart"]
        peak_count = peak["peakCount"]

        if peak_count < threshold:
            action = "observe"
            alert_id = None
            step_suppressed_count = None
        elif prior_start is None or peak_start - prior_start >= suppression_window:
            alert_counter += 1
            alert_id = f"alert-{alert_counter}"
            action = "escalate"
            step_suppressed_count = 0
            suppressed_count = 0
            prior_start = peak_start
            prior_alert_id = alert_id
        else:
            action = "suppress"
            alert_id = prior_alert_id
            suppressed_count += 1
            step_suppressed_count = suppressed_count

        steps.append(
            {
                "eventId": event["eventId"],
                "occurredAt": event["occurredAt"],
                "windows": windows,
                "peakStart": peak_start,
                "peakCount": peak_count,
                "action": action,
                "alertId": alert_id,
                "suppressedCount": step_suppressed_count,
            }
        )
    return steps


def _aligned_replay_window_rows(
    left_occurred: list[int],
    right_occurred: list[int],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Align one replay step's window rows for the two sides.

    Each ``occurred`` list holds one side's timestamps accumulated up to and
    including the current step. The rows use the same division as the window
    comparison: without a range they cover the union of the two sides' hit
    windows; with a range they cover every window intersecting the closed
    interval ``[from, to]``, intersecting empty windows kept. Rows align by
    start ascending; a side with no event in a window reports a zero count,
    and ``equal`` is true exactly when the two counts agree.
    """
    left_counts = _branch_window_counts(left_occurred, params)
    right_counts = _branch_window_counts(right_occurred, params)

    range_starts = _branch_window_starts(params)
    if range_starts:
        starts = range_starts
    else:
        starts = sorted(set(left_counts) | set(right_counts))

    return [
        {
            "start": start,
            "leftCount": left_counts.get(start, 0),
            "rightCount": right_counts.get(start, 0),
            "equal": left_counts.get(start, 0) == right_counts.get(start, 0),
        }
        for start in starts
    ]


def _aligned_replay_decision(
    left_occurred: list[int],
    right_occurred: list[int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build one replay step's paired peak decisions and equality marker."""
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
    return {
        "left": left_decision,
        "right": right_decision,
        "equal": left_decision == right_decision,
    }


def compare_branch_replay_decisions(
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Align two branches' step-by-step replay decisions by event id.

    Each ``events`` list is a single locked snapshot of one branch's events
    for the caller's organization, sorted by ``occurredAt`` then ``eventId``
    — the replay order the single-branch replay uses. Events of other types
    never open a step. Each side accumulates its own matching events in that
    order, and the two replays are aligned by ``eventId``: an identifier
    reached by both sides is one shared step on which both prefixes extend;
    an identifier reached by only one side is its own step on which the
    other side stays put. Aligned steps are ordered by occurred time and
    then identifier in Unicode code-point order, exactly like one merged
    replay; when the two sides disagree on a shared identifier's time the
    earlier one is reported. Every step reports the event id and time, the
    window rows of the two accumulated prefixes aligned by start (a side
    with no event in a window counts zero), and each side's peak decision
    plus an ``equal`` marker. When neither side holds a matching event there
    are no steps at all; nothing is written.
    """
    left_stream = [event for event in left_events if event["type"] == params["type"]]
    right_stream = [
        event for event in right_events if event["type"] == params["type"]
    ]

    left_by_id = {event["eventId"]: event for event in left_stream}
    right_by_id = {event["eventId"]: event for event in right_stream}

    # One aligned step per identifier in the union; each side's own matching
    # event supplies that identifier's time on that side, and a shared
    # identifier's reported time is the earlier of the two sides' times.
    step_times: dict[str, int] = {}
    for event in left_stream:
        step_times[event["eventId"]] = event["occurredAt"]
    for event in right_stream:
        event_id = event["eventId"]
        if event_id in step_times:
            step_times[event_id] = min(step_times[event_id], event["occurredAt"])
        else:
            step_times[event_id] = event["occurredAt"]

    # Walk the aligned identifiers in step order; each side accumulates its
    # own event when the aligned replay reaches that identifier and stays
    # put for an identifier only the other side reaches.
    steps: list[dict[str, Any]] = []
    left_accumulated: list[int] = []
    right_accumulated: list[int] = []
    for event_id in sorted(step_times, key=lambda eid: (step_times[eid], eid)):
        left_event = left_by_id.get(event_id)
        right_event = right_by_id.get(event_id)
        if left_event is not None:
            left_accumulated.append(left_event["occurredAt"])
        if right_event is not None:
            right_accumulated.append(right_event["occurredAt"])

        # The window rows align the two prefixes row by row, and the peak
        # for each side is exactly what the single-branch replay step shows
        # on that side's own accumulated prefix.
        windows = _aligned_replay_window_rows(
            left_accumulated, right_accumulated, params
        )
        decision = _aligned_replay_decision(
            left_accumulated, right_accumulated, params
        )
        steps.append(
            {
                "eventId": event_id,
                "occurredAt": step_times[event_id],
                "windows": windows,
                "decision": decision,
            }
        )

    return {
        "organizationId": params["organizationId"],
        "left": params["left"],
        "right": params["right"],
        "type": params["type"],
        "windowSize": params["windowSize"],
        "threshold": params["threshold"],
        "from": params["from"],
        "to": params["to"],
        "steps": steps,
    }


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


def compare_snapshots(
    left_occurred: list[int],
    right_occurred: list[int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Diff two snapshots' window counts and peaks from captured timestamps.

    Delegates to :func:`compare_branches`; each ``occurred`` list derives
    from one snapshot's immutable captured events (already scoped to one
    organization at capture time), so the window rows and the peak for a
    side are recomputed purely from those copies and nothing is written.
    """
    return compare_branches(left_occurred, right_occurred, params)


def compare_snapshot_replay_decisions(
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Align two snapshots' step-by-step replay decisions by event id.

    Delegates to :func:`compare_branch_replay_decisions`; each ``events``
    list is one immutable snapshot's captured events of the caller's
    organization (a snapshot holds only its owning organization's events),
    sorted by ``occurredAt`` then ``eventId`` — the replay order the
    single-snapshot replay uses. The two replays align exactly like the
    branch replay comparison, and nothing is written.
    """
    return compare_branch_replay_decisions(left_events, right_events, params)


def _prefix_peak(
    accumulated: list[int], params: dict[str, Any]
) -> dict[str, Any]:
    """The peak of one side's accumulated timestamps for an alert replay.

    Uses the same window rows and peak tie-break as every other replay
    entry point, but unlike :func:`evaluate_decision` it neither reads nor
    echoes a ``type`` field, so it serves the region comparison (whose
    params carry one shared type) and the type comparison (whose params
    carry only the two side names) identically. The caller derives the
    suppression action from the returned start/count itself.
    """
    windows = _windows_from_occurred(
        accumulated,
        params["windowSize"],
        params["from"],
        params["to"],
    )
    return _peak_from_windows(windows, params["threshold"])


def _empty_alert_side_state() -> dict[str, Any]:
    """One side's private simulated-alert state before its first event."""
    return {
        "accumulated": [],
        "prior_start": None,
        "prior_alert_id": None,
        "suppressed_count": 0,
        "alert_counter": 0,
        "last": None,
    }


def _alert_replay_side_decision(
    state: dict[str, Any], event: dict[str, Any] | None, params: dict[str, Any]
) -> dict[str, Any]:
    """Advance (or freeze) one side and return its alert replay decision.

    Shared by the region- and type-dimension alert replay comparisons; the
    suppression mechanics are identical for both. With ``event`` set the
    side accumulates that event and runs the exact suppression rule of
    :func:`alert_replay_steps` against the most recent simulated alert
    *within this side's own replay*; simulated identifiers run a private
    one-based sequence. With ``event`` None the aligned step belongs to the
    other side, so this side neither accumulates nor re-runs the rule — its
    prefix is not cleared and the step reports the decision of its own last
    replay step (the initial observe state when it has not moved yet).
    """
    accumulated = state["accumulated"]
    if event is None:
        last = state["last"]
        if last is not None:
            return dict(last)
        peak = _prefix_peak(accumulated, params)
        return {
            "peakStart": peak["peakStart"],
            "peakCount": peak["peakCount"],
            "action": "observe",
            "alertId": None,
            "suppressedCount": None,
        }

    accumulated.append(event["occurredAt"])
    peak = _prefix_peak(accumulated, params)
    peak_start = peak["peakStart"]
    peak_count = peak["peakCount"]

    if peak_count < params["threshold"]:
        decision = {
            "peakStart": peak_start,
            "peakCount": peak_count,
            "action": "observe",
            "alertId": None,
            "suppressedCount": None,
        }
    elif (
        state["prior_start"] is None
        or peak_start - state["prior_start"] >= params["suppressionWindow"]
    ):
        state["alert_counter"] += 1
        decision = {
            "peakStart": peak_start,
            "peakCount": peak_count,
            "action": "escalate",
            "alertId": f"alert-{state['alert_counter']}",
            "suppressedCount": 0,
        }
        state["suppressed_count"] = 0
        state["prior_start"] = peak_start
        state["prior_alert_id"] = decision["alertId"]
    else:
        state["suppressed_count"] += 1
        decision = {
            "peakStart": peak_start,
            "peakCount": peak_count,
            "action": "suppress",
            "alertId": state["prior_alert_id"],
            "suppressedCount": state["suppressed_count"],
        }

    state["last"] = dict(decision)
    return decision


def _align_alert_replay_decisions(
    left_stream: list[dict[str, Any]],
    right_stream: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Align two sides' step-by-step alert replay decisions by event id.

    Dimension-neutral core shared by the region and type alert replay
    comparisons. Each ``stream`` is already filtered to one side's matching
    events (the caller scopes the region and/or type) for the caller's
    organization. Each side accumulates its own events in ``occurredAt``
    then ``eventId`` order and runs the suppression simulation of
    :func:`alert_replay_steps` with **private** state: an identifier
    reached by both sides is one shared step on which both simulations
    advance, an identifier reached by only one side is its own step on
    which the other side stays put with its prefix and prior simulated
    alert untouched. Simulated alert identifiers restart at ``alert-1``
    independently on each side; the alert store, the service-wide
    counter, the ledger, and reservations are never read or written, so
    repeated calls are byte-for-byte identical.

    Aligned steps are ordered by occurred time and then identifier in
    Unicode code-point order; a shared identifier whose sides disagree on
    time reports the earlier one. Every step carries the event id/time,
    the two prefixes' window rows aligned by start (a missing side counts
    zero), and ``decision`` with ``left``/``right`` blocks of
    ``peakStart``, ``peakCount``, ``action``, ``alertId`` and
    ``suppressedCount``; ``equal`` considers only the peak start, count,
    and action, never the alert identity. The echoed body carries the
    common fields; a dimension that also names a shared scope (the region
    comparison's single ``type``) adds that key itself.
    """
    left_by_id = {event["eventId"]: event for event in left_stream}
    right_by_id = {event["eventId"]: event for event in right_stream}

    # One aligned step per identifier in the union; a shared identifier's
    # reported time is the earlier of the two sides' times.
    step_times: dict[str, int] = {}
    for event in left_stream:
        step_times[event["eventId"]] = event["occurredAt"]
    for event in right_stream:
        event_id = event["eventId"]
        if event_id in step_times:
            step_times[event_id] = min(step_times[event_id], event["occurredAt"])
        else:
            step_times[event_id] = event["occurredAt"]

    left_state = _empty_alert_side_state()
    right_state = _empty_alert_side_state()
    steps: list[dict[str, Any]] = []
    for event_id in sorted(step_times, key=lambda eid: (step_times[eid], eid)):
        left_event = left_by_id.get(event_id)
        right_event = right_by_id.get(event_id)
        left_decision = _alert_replay_side_decision(
            left_state, left_event, params
        )
        right_decision = _alert_replay_side_decision(
            right_state, right_event, params
        )

        windows = _aligned_replay_window_rows(
            left_state["accumulated"], right_state["accumulated"], params
        )
        equal = (
            left_decision["peakStart"],
            left_decision["peakCount"],
            left_decision["action"],
        ) == (
            right_decision["peakStart"],
            right_decision["peakCount"],
            right_decision["action"],
        )
        steps.append(
            {
                "eventId": event_id,
                "occurredAt": step_times[event_id],
                "windows": windows,
                "decision": {
                    "left": left_decision,
                    "right": right_decision,
                    "equal": equal,
                },
            }
        )

    return {
        "organizationId": params["organizationId"],
        "left": params["left"],
        "right": params["right"],
        "windowSize": params["windowSize"],
        "threshold": params["threshold"],
        "suppressionWindow": params["suppressionWindow"],
        "from": params["from"],
        "to": params["to"],
        "steps": steps,
    }


def compare_region_alert_replay_decisions(
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Align two regions' step-by-step alert replay decisions by event id.

    The region-dimension wrapper over :func:`_align_alert_replay_decisions`.
    Each ``events`` list is a single locked snapshot of one region's events
    for the caller's organization, already filtered to the verbatim region;
    events of other types never open a step. Both sides share the one
    request ``type``, so the result additionally echoes it.
    """
    left_stream = [event for event in left_events if event["type"] == params["type"]]
    right_stream = [
        event for event in right_events if event["type"] == params["type"]
    ]
    result = _align_alert_replay_decisions(left_stream, right_stream, params)
    result["type"] = params["type"]
    return result


def compare_type_alert_replay_decisions(
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Align two event types' step-by-step alert replay decisions by event id.

    The type-dimension counterpart of
    :func:`compare_region_alert_replay_decisions`: the two sides name two
    event types directly rather than two regions, so the left stream keeps
    ``params["left"]`` events and the right stream keeps ``params["right"]``
    events. Each ``events`` list is a single locked snapshot of one
    organization's events (region attribution is irrelevant here); events
    of the other side's type never open a step on this side. There is no
    single shared ``type`` field, so the result echoes only the two type
    names. Suppression state, simulated identifier numbering, window rows,
    alignment order, the equality marker, and the read-only contract are
    all identical to the region comparison.
    """
    left_stream = [event for event in left_events if event["type"] == params["left"]]
    right_stream = [
        event for event in right_events if event["type"] == params["right"]
    ]
    return _align_alert_replay_decisions(left_stream, right_stream, params)


def compare_event_type_replay_decisions(
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Align two event types' step-by-step event replay decisions by event id.

    The non-alert counterpart of
    :func:`compare_type_alert_replay_decisions`: the two sides name two
    event types directly, so the left stream keeps ``params["left"]``
    events and the right stream keeps ``params["right"]`` events. Each
    ``events`` list is a single locked snapshot of one organization's
    events; events of the other side's type never open a step on this
    side. There is no single shared ``type`` field, so the result echoes
    only the two type names.

    Each side accumulates its own matching events in ``occurredAt`` then
    ``eventId`` order and recomputes its peak purely from that prefix —
    below the threshold is ``observe``, otherwise ``escalate``; there is
    no alert identity or suppression state on this dimension. An
    identifier reached by both sides is one shared step on which both
    prefixes extend; an identifier reached by only one side is its own
    step on which the other side keeps its prefix untouched (its frozen
    decision is re-reported, the initial empty-prefix observe state when
    it has not moved yet). Aligned steps are ordered by occurred time and
    then identifier in Unicode code-point order; a shared identifier
    whose sides disagree on time reports the earlier one. Window rows,
    the peak tie-break, the equality markers, and the read-only
    byte-identical contract are all identical to the branch replay
    comparison.
    """
    left_stream = [event for event in left_events if event["type"] == params["left"]]
    right_stream = [
        event for event in right_events if event["type"] == params["right"]
    ]

    left_by_id = {event["eventId"]: event for event in left_stream}
    right_by_id = {event["eventId"]: event for event in right_stream}

    # One aligned step per identifier in the union; a shared identifier's
    # reported time is the earlier of the two sides' times.
    step_times: dict[str, int] = {}
    for event in left_stream:
        step_times[event["eventId"]] = event["occurredAt"]
    for event in right_stream:
        event_id = event["eventId"]
        if event_id in step_times:
            step_times[event_id] = min(step_times[event_id], event["occurredAt"])
        else:
            step_times[event_id] = event["occurredAt"]

    steps: list[dict[str, Any]] = []
    left_accumulated: list[int] = []
    right_accumulated: list[int] = []
    for event_id in sorted(step_times, key=lambda eid: (step_times[eid], eid)):
        left_event = left_by_id.get(event_id)
        right_event = right_by_id.get(event_id)
        if left_event is not None:
            left_accumulated.append(left_event["occurredAt"])
        if right_event is not None:
            right_accumulated.append(right_event["occurredAt"])

        # The peak is a pure function of the accumulated prefix, so a side
        # that stays put on this step re-reports exactly its prior decision
        # (the initial empty-prefix observe state until it has moved).
        windows = _aligned_replay_window_rows(
            left_accumulated, right_accumulated, params
        )
        left_decision = _prefix_peak(left_accumulated, params)
        right_decision = _prefix_peak(right_accumulated, params)
        steps.append(
            {
                "eventId": event_id,
                "occurredAt": step_times[event_id],
                "windows": windows,
                "decision": {
                    "left": left_decision,
                    "right": right_decision,
                    "equal": left_decision == right_decision,
                },
            }
        )

    return {
        "organizationId": params["organizationId"],
        "left": params["left"],
        "right": params["right"],
        "windowSize": params["windowSize"],
        "threshold": params["threshold"],
        "from": params["from"],
        "to": params["to"],
        "steps": steps,
    }


def compare_event_regions(
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Diff two regions' window counts and peaks in one shot.

    The region counterpart of :func:`compare_branches` on the main ledger
    and the single-shot counterpart of
    :func:`compare_event_region_replay_decisions`: instead of aligning
    step-by-step prefixes it counts each side's full event set once. Each
    ``events`` list is a single locked snapshot already scoped to the
    caller's organization and one region by the ledger; the attribution
    rule is re-asserted here, so only a verbatim non-empty string payload
    region match counts and events of every type participate. There is no
    shared ``type`` field, so the result echoes only the two region names.

    Window rows follow the aggregate contract exactly: without a range
    they cover the union of the two sides' hit windows; with a range they
    cover every window intersecting the closed interval ``[from, to]``,
    intersecting empty windows kept with count zero. Rows align by start
    ascending and carry each side's count plus an ``equal`` marker that is
    true exactly when the two counts agree. The ``decision`` block holds
    each side's peak (largest count, ties to the earliest start; an empty
    side reports a null start, a zero count, and ``observe``) and its
    threshold-only action, plus a marker saying whether the two sides
    agree on all three. Nothing is written.
    """
    left_occurred = [
        event["occurredAt"]
        for event in left_events
        if event_region(event) == params["left"]
    ]
    right_occurred = [
        event["occurredAt"]
        for event in right_events
        if event_region(event) == params["right"]
    ]

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

    left_peak = _prefix_peak(left_occurred, params)
    right_peak = _prefix_peak(right_occurred, params)
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
        "windowSize": params["windowSize"],
        "threshold": params["threshold"],
        "from": params["from"],
        "to": params["to"],
        "decision": decision,
        "windows": windows,
    }


def compare_event_region_replay_decisions(
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Align two regions' plain event replay decisions by event id.

    The region counterpart of :func:`compare_event_type_replay_decisions`:
    the two sides name two regions directly, so the left stream keeps the
    verbatim ``params["left"]`` region events and the right stream keeps the
    ``params["right"]`` region events. Each ``events`` list is a single
    locked snapshot already scoped to the caller's organization and one
    region by the ledger; events attributed to another region never open a
    step on this side, and events without a non-empty string payload region
    never match either side. There is no shared ``type`` field, so the
    result echoes only the two region names.

    Each side accumulates its own region's events in ``occurredAt`` then
    ``eventId`` order and recomputes its peak purely from that prefix —
    below the threshold is ``observe``, otherwise ``escalate``; there is
    no alert identity or suppression state on this dimension. An
    identifier reached by both sides is one shared step on which both
    prefixes extend; an identifier reached by only one side is its own
    step on which the other side keeps its prefix untouched (its frozen
    decision is re-reported, the initial empty-prefix observe state when
    it has not moved yet). Aligned steps are ordered by occurred time and
    then identifier in Unicode code-point order; a shared identifier
    whose sides disagree on time reports the earlier one. Window rows,
    the peak tie-break, the equality markers, and the read-only
    byte-identical contract are all identical to the type replay
    comparison.
    """
    left_stream = [
        event for event in left_events if event_region(event) == params["left"]
    ]
    right_stream = [
        event for event in right_events if event_region(event) == params["right"]
    ]

    left_by_id = {event["eventId"]: event for event in left_stream}
    right_by_id = {event["eventId"]: event for event in right_stream}

    # One aligned step per identifier in the union; a shared identifier's
    # reported time is the earlier of the two sides' times.
    step_times: dict[str, int] = {}
    for event in left_stream:
        step_times[event["eventId"]] = event["occurredAt"]
    for event in right_stream:
        event_id = event["eventId"]
        if event_id in step_times:
            step_times[event_id] = min(step_times[event_id], event["occurredAt"])
        else:
            step_times[event_id] = event["occurredAt"]

    steps: list[dict[str, Any]] = []
    left_accumulated: list[int] = []
    right_accumulated: list[int] = []
    for event_id in sorted(step_times, key=lambda eid: (step_times[eid], eid)):
        left_event = left_by_id.get(event_id)
        right_event = right_by_id.get(event_id)
        if left_event is not None:
            left_accumulated.append(left_event["occurredAt"])
        if right_event is not None:
            right_accumulated.append(right_event["occurredAt"])

        # The peak is a pure function of the accumulated prefix, so a side
        # that stays put on this step re-reports exactly its prior decision
        # (the initial empty-prefix observe state until it has moved).
        windows = _aligned_replay_window_rows(
            left_accumulated, right_accumulated, params
        )
        left_decision = _prefix_peak(left_accumulated, params)
        right_decision = _prefix_peak(right_accumulated, params)
        steps.append(
            {
                "eventId": event_id,
                "occurredAt": step_times[event_id],
                "windows": windows,
                "decision": {
                    "left": left_decision,
                    "right": right_decision,
                    "equal": left_decision == right_decision,
                },
            }
        )

    return {
        "organizationId": params["organizationId"],
        "left": params["left"],
        "right": params["right"],
        "windowSize": params["windowSize"],
        "threshold": params["threshold"],
        "from": params["from"],
        "to": params["to"],
        "steps": steps,
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


def _compare_resources(
    left_balances: dict[str, dict[str, int]],
    right_balances: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compare two resource-balance mappings aligned by resourceId.

    Each mapping is a consistent, locked or deep-copied snapshot of one side's
    per-resource ``capacity``/``occupied``/``remaining`` balances for one
    organization, so the rows and four groups derive from consistent reads
    and nothing is written. The union of the two sides' resource ids
    produces one ``resources`` row each, sorted by resourceId in Unicode
    code-point order; a resource missing on a side is treated as absent and
    that side carries all three balances as zero. A row's ``equal`` marker
    is true exactly when the two sides' three balances all agree.

    Identifiers only on the left land in ``leftOnly``, only on the right in
    ``rightOnly``. An identifier on both sides counts toward ``same`` when
    all three balances match and lands in ``diff`` otherwise, as
    ``{"resourceId", "fields"}`` naming the mismatched balances, drawn only
    from ``capacity``, ``occupied`` and ``remaining``. Identifiers and field
    names are sorted in Unicode code-point order; each group also gets a
    ``<group>Count`` key. When neither side holds a resource, every group
    (and the row array) is empty and every count is zero.
    """
    zero_balances = {"capacity": 0, "occupied": 0, "remaining": 0}

    left_only = sorted(set(left_balances) - set(right_balances))
    right_only = sorted(set(right_balances) - set(left_balances))
    same: list[str] = []
    diff: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []

    for resource_id in sorted(set(left_balances) | set(right_balances)):
        left = left_balances.get(resource_id, zero_balances)
        right = right_balances.get(resource_id, zero_balances)
        fields = sorted(
            field
            for field in RESOURCE_COMPARE_FIELDS
            if left[field] != right[field]
        )
        resources.append(
            {
                "resourceId": resource_id,
                "left": {field: left[field] for field in RESOURCE_COMPARE_FIELDS},
                "right": {field: right[field] for field in RESOURCE_COMPARE_FIELDS},
                "equal": not fields,
            }
        )
        if resource_id in left_balances and resource_id in right_balances:
            if fields:
                diff.append({"resourceId": resource_id, "fields": fields})
            else:
                same.append(resource_id)

    return {
        "resources": resources,
        "leftOnly": left_only,
        "leftOnlyCount": len(left_only),
        "rightOnly": right_only,
        "rightOnlyCount": len(right_only),
        "same": same,
        "sameCount": len(same),
        "diff": diff,
        "diffCount": len(diff),
    }


def resource_balance_rows(
    balances: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    """Build the sorted resource rows for one branch's balance listing.

    ``balances`` is a consistent, locked snapshot of one branch's
    per-resource balances for one organization (the same mapping the
    resource comparison aligns on), so the rows derive from a consistent
    read and nothing is written. Each resource maps to one row carrying
    its ``capacity``, ``occupied`` and ``remaining`` balances; rows are
    sorted by resourceId in Unicode code-point order, and an empty mapping
    yields an empty row array.
    """
    return [
        {
            "resourceId": resource_id,
            "capacity": balances[resource_id]["capacity"],
            "occupied": balances[resource_id]["occupied"],
            "remaining": balances[resource_id]["remaining"],
        }
        for resource_id in sorted(balances)
    ]


def event_rows(events: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the sorted event rows for one snapshot's listing.

    ``events`` is a deep copy of one snapshot's captured events for one
    organization (the same mapping the event comparison aligns on), so the
    rows derive from a consistent read and nothing is written. Each captured
    event maps to one row carrying exactly the five event fields
    (``eventId``, ``organizationId``, ``type``, ``occurredAt``, ``payload``);
    rows are sorted by ``occurredAt`` then ``eventId`` in Unicode code-point
    order, matching ``GET /events``, and an empty mapping yields an empty row
    array.
    """
    rows = [
        {field: event[field] for field in EVENT_FIELDS}
        for event in events.values()
    ]
    rows.sort(key=lambda row: (row["occurredAt"], row["eventId"]))
    return rows


def reservation_rows(
    reservations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the sorted reservation rows for one snapshot's listing.

    ``reservations`` is a deep copy of one snapshot's captured reservations
    for one organization (the same mapping the reservation comparison
    aligns on), so the rows derive from a consistent read and nothing is
    written. Each captured reservation maps to one row carrying exactly the
    five reservation fields (``organizationId``, ``reservationId``,
    ``resourceId``, ``quantity``, ``capacity``); rows are sorted by
    ``resourceId`` then ``reservationId`` in Unicode code-point order, and
    an empty mapping yields an empty row array.
    """
    rows = [
        {field: record[field] for field in RESERVATION_FIELDS}
        for record in reservations.values()
    ]
    rows.sort(key=lambda row: (row["resourceId"], row["reservationId"]))
    return rows


def compare_branch_resources(
    left_balances: dict[str, dict[str, int]],
    right_balances: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compare two branches' resource balances aligned by resourceId.

    Delegates to the shared :func:`_compare_resources`; each mapping is a
    locked snapshot of one branch's per-resource balances for one
    organization.
    """
    return _compare_resources(left_balances, right_balances)


def compare_snapshot_resources(
    left_balances: dict[str, dict[str, int]],
    right_balances: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Compare two snapshots' resource balances aligned by resourceId.

    Delegates to the shared :func:`_compare_resources`; each mapping is a
    deep copy of one snapshot's captured per-resource
    ``capacity``/``occupied``/``remaining`` balances (already scoped to one
    organization at capture time), so the rows and four groups derive from
    immutable copies and nothing is written. See :func:`_compare_resources`
    for the row and group contract.
    """
    return _compare_resources(left_balances, right_balances)


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

        if path.startswith("/snapshots/"):
            remainder = path[len("/snapshots/"):]
            segments = remainder.split("/")
            if (
                len(segments) == 2
                and segments[0]
                and segments[1] == "resources"
            ):
                # The resource-balance listing has its own verdict order
                # (credential, then query shape, then organization, then
                # snapshot), so it is routed directly; every other unknown
                # /snapshots/... sub-path keeps the generic 404.
                self._list_snapshot_resources(unquote(segments[0]), query)
                return
            if (
                len(segments) == 2
                and segments[0]
                and segments[1] == "reservations"
            ):
                # The reservation listing shares the resource listing's
                # verdict order, so it is routed directly too.
                self._list_snapshot_reservations(unquote(segments[0]), query)
                return
            if (
                len(segments) == 2
                and segments[0]
                and segments[1] == "events"
            ):
                # The event listing shares the same verdict order as the
                # resource and reservation listings.
                self._list_snapshot_events(unquote(segments[0]), query)
                return
            if (
                len(segments) == 3
                and segments[0]
                and segments[1] == "events"
                and segments[2] == "region"
            ):
                # The single-snapshot region event listing shares the same
                # verdict order as the other snapshot-prefixed queries.
                self._list_snapshot_events_by_region(
                    unquote(segments[0]), query
                )
                return
            if (
                len(segments) == 3
                and segments[0]
                and segments[1] == "events"
                and segments[2] == "aggregate"
            ):
                # The single-snapshot event aggregate shares the event
                # listing's verdict order (credential, query shape,
                # organization, then snapshot ownership).
                self._aggregate_snapshot_events(unquote(segments[0]), query)
                return
            if (
                len(segments) == 4
                and segments[0]
                and segments[1] == "events"
                and segments[2] == "region"
                and segments[3] == "aggregate"
            ):
                # The single-snapshot region aggregate shares the same
                # verdict order as the other snapshot-prefixed queries.
                self._aggregate_snapshot_events_by_region(
                    unquote(segments[0]), query
                )
                return
            if (
                len(segments) == 5
                and segments[0]
                and segments[1] == "events"
                and segments[2] == "region"
                and segments[3] == "replay"
                and segments[4] == "decisions"
            ):
                # The single-snapshot region replay decision shares the same
                # verdict order as the other snapshot-prefixed queries.
                self._replay_snapshot_region_decisions(
                    unquote(segments[0]), query
                )
                return
            if (
                len(segments) == 4
                and segments[0]
                and segments[1] == "events"
                and segments[2] == "replay"
                and segments[3] == "decisions"
            ):
                # The single-snapshot replay decision shares the same
                # verdict order as the other snapshot-prefixed queries.
                self._replay_snapshot_decisions(unquote(segments[0]), query)
                return
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "path": self.path},
            )
            return

        if path.startswith("/branches/"):
            remainder = path[len("/branches/"):]
            segments = remainder.split("/")
            if (
                len(segments) == 2
                and segments[0]
                and segments[1] == "resources"
            ):
                # The resource-balance listing has its own verdict order
                # (credential, then query shape, then organization, then
                # branch), so it is routed before the generic branch block,
                # which resolves the branch name first.
                self._list_branch_resources(unquote(segments[0]), query)
                return
            if (
                len(segments) == 4
                and segments[0]
                and segments[1] == "events"
                and segments[2] == "replay"
                and segments[3] == "decisions"
            ):
                # The branch replay decision shares the resource listing's
                # verdict order, so it too is routed before the generic
                # branch block.
                self._replay_branch_decisions(unquote(segments[0]), query)
                return
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
        if path == "/events/replay/decisions":
            self._replay_decisions(
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
        if path == "/events/region/replay/decisions":
            self._replay_region_decisions(
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
        if path == "/alerts/replay/decisions":
            self._replay_alert_decisions(query)
            return
        if path == "/alerts/region/replay/decisions":
            self._replay_region_alert_decisions(query)
            return
        if path == "/alerts/region":
            self._list_region_alerts(query)
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
        if path == "/branches/compare/resources":
            self._compare_branch_resources()
            return
        if path == "/branches/compare/replay/decisions":
            self._compare_branch_replay_decisions()
            return
        if path == "/snapshots/compare":
            self._compare_snapshots()
            return
        if path == "/snapshots/compare/events":
            self._compare_snapshot_events()
            return
        if path == "/snapshots/compare/reservations":
            self._compare_snapshot_reservations()
            return
        if path == "/snapshots/compare/resources":
            self._compare_snapshot_resources()
            return
        if path == "/snapshots/compare/replay/decisions":
            self._compare_snapshot_replay_decisions()
            return
        if path == "/alerts/regions/compare/replay/decisions":
            self._compare_region_alert_replay_decisions()
            return
        if path == "/alerts/types/compare/replay/decisions":
            self._compare_type_alert_replay_decisions()
            return
        if path == "/events/types/compare/replay/decisions":
            self._compare_event_type_replay_decisions()
            return
        if path == "/events/regions/compare":
            self._compare_event_regions()
            return
        if path == "/events/regions/compare/replay/decisions":
            self._compare_event_region_replay_decisions()
            return

        if path.startswith("/snapshots/"):
            remainder = path[len("/snapshots/"):]
            segments = remainder.split("/")
            if (
                len(segments) == 3
                and segments[0]
                and segments[1] == "decisions"
                and segments[2] == "evaluate"
            ):
                # The single-snapshot decision has its own verdict order
                # (credential, then body validation, then organization, then
                # snapshot), so it is routed directly; every other unknown
                # /snapshots/... sub-path keeps the generic 404.
                self._evaluate_snapshot_decision(unquote(segments[0]))
                return
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "path": self.path},
            )
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
                    and segments[1] == "reservations"
                    and segments[2] == "cancel"
                ):
                    self._cancel_reservation(branch.reservations, newline=True)
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
        if path == "/reservations/cancel":
            self._cancel_reservation(
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
        if path == "/alerts/region/evaluate":
            self._evaluate_region_alert()
            return
        if path == "/decisions/allocate":
            self._allocate_decision()
            return
        if path == "/decisions/route-allocate":
            self._route_allocate_decision()
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
            # The creation success body carries the same trailing newline as
            # the other write entry points (event and reservation commits).
            self._write_json(
                HTTPStatus.CREATED,
                {
                    "snapshotId": snapshot_id,
                    "events": snapshot.event_count,
                    "resources": snapshot.capacity_count,
                    "reservations": snapshot.reservation_count,
                },
                newline=True,
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

    def _list_snapshot_resources(self, snapshot_id: str, query: str) -> None:
        """Serve GET /snapshots/{snapshotId}/resources.

        Read-only per-resource balance listing for one immutable snapshot.
        The verdict order is fixed: the credential is authenticated first,
        then the query shape is validated, then the requested organization
        is compared with the credential's, and only then is the snapshot
        name resolved (missing snapshot, then foreign snapshot). Both
        ``read`` and ``write`` credentials may call it.
        """
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
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return
        # The snapshot must already exist; a listing never implicitly creates
        # one, and a foreign snapshot is forbidden rather than not found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != organization_id:
            self._forbidden(newline=True)
            return
        # The snapshot already scopes its captured capacities and
        # reservations to the owning organization at capture time; the
        # balances are recomputed from deep copies — the same mapping the
        # snapshot resource comparison aligns on, so a snapshot compared
        # with itself agrees row by row. Nothing is written, so identical
        # requests return byte-identical JSON.
        balances = snapshot.resource_balances()
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": organization_id,
                "snapshotId": snapshot.snapshot_id,
                "resources": resource_balance_rows(balances),
            },
            newline=True,
        )

    def _list_snapshot_reservations(self, snapshot_id: str, query: str) -> None:
        """Serve GET /snapshots/{snapshotId}/reservations.

        Read-only reservation listing for one immutable snapshot. The
        verdict order is fixed: the credential is authenticated first,
        then the query shape is validated, then the requested organization
        is compared with the credential's, and only then is the snapshot
        name resolved (missing snapshot, then foreign snapshot). Both
        ``read`` and ``write`` credentials may call it.
        """
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
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return
        # The snapshot must already exist; a listing never implicitly creates
        # one, and a foreign snapshot is forbidden rather than not found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != organization_id:
            self._forbidden(newline=True)
            return
        # The snapshot already scopes its captured reservations to the
        # owning organization at capture time; the rows are built from a
        # deep copy — the same mapping the snapshot reservation comparison
        # aligns on, so a snapshot compared with itself agrees item by
        # item. Nothing is written, so identical requests return
        # byte-identical JSON.
        reservations = snapshot.reservations_snapshot()
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": organization_id,
                "snapshotId": snapshot.snapshot_id,
                "reservations": reservation_rows(reservations),
            },
            newline=True,
        )

    def _list_snapshot_events(self, snapshot_id: str, query: str) -> None:
        """Serve GET /snapshots/{snapshotId}/events.

        Read-only listing of one immutable snapshot's captured events for the
        caller's organization — the single-snapshot counterpart of
        ``POST /snapshots/compare/events``, built from the exact same
        captured-event contract. The verdict order is fixed: the credential
        is authenticated first, then the query shape is validated, then the
        requested organization is compared with the credential's, and only
        then is the snapshot name resolved (missing snapshot, then foreign
        snapshot). Both ``read`` and ``write`` credentials may call it.
        """
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
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return
        # The snapshot must already exist; a listing never implicitly creates
        # one, and a foreign snapshot is forbidden rather than not found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != organization_id:
            self._forbidden(newline=True)
            return
        # The snapshot already scopes its captured events to the owning
        # organization at capture time; the rows are built from a deep copy —
        # the same mapping the snapshot event comparison aligns on, so a
        # snapshot compared with itself lists every event in ``same`` with
        # zero diffs, item by item. Nothing is written, so identical requests
        # return byte-identical JSON.
        events = snapshot.events_snapshot()
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": organization_id,
                "snapshotId": snapshot.snapshot_id,
                "events": event_rows(events),
            },
            newline=True,
        )

    def _list_snapshot_events_by_region(
        self, snapshot_id: str, query: str
    ) -> None:
        """Serve GET /snapshots/{snapshotId}/events/region.

        Read-only listing of one immutable snapshot's captured events
        attributed to one region for the caller's organization — the
        single-snapshot counterpart of the main ``GET /events/region``,
        with the exact region attribution rule the snapshot region
        aggregate counts from. The verdict order is fixed: the credential
        is authenticated first, then the query shape is validated, then
        the requested organization is compared with the credential's, and
        only then is the snapshot name resolved (missing snapshot, then
        foreign snapshot). Both ``read`` and ``write`` credentials may
        call it.
        """
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
        # The snapshot must already exist; a listing never implicitly creates
        # one, and a foreign snapshot is forbidden rather than not found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != params["organizationId"]:
            self._forbidden(newline=True)
            return
        # The snapshot already scopes its captured events to the owning
        # organization at capture time; the rows are built from a deep copy
        # filtered by the same verbatim region rule the snapshot region
        # aggregate counts from, so the listing and the aggregate hit
        # exactly the same captured events. Nothing is written, so
        # identical requests return byte-identical JSON.
        events = snapshot.events_snapshot_for_region(params["region"])
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "snapshotId": snapshot.snapshot_id,
                "region": params["region"],
                "events": event_rows(events),
            },
            newline=True,
        )

    def _aggregate_snapshot_events(self, snapshot_id: str, query: str) -> None:
        """Serve GET /snapshots/{snapshotId}/events/aggregate.

        Read-only window aggregation over one immutable snapshot's captured
        events — the single-snapshot counterpart of the two-snapshot window
        comparison, with the exact window division the main event aggregate
        uses. The verdict order is fixed: the credential is authenticated
        first, then the query shape is validated, then the requested
        organization is compared with the credential's, and only then is the
        snapshot name resolved (missing snapshot, then foreign snapshot).
        Both ``read`` and ``write`` credentials may call it.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _aggregate_params_from_query(query)
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
        # The snapshot must already exist; an aggregate never implicitly
        # creates one, and a foreign snapshot is forbidden rather than not
        # found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != params["organizationId"]:
            self._forbidden(newline=True)
            return
        # The snapshot already scopes its captured events to the owning
        # organization at capture time; the window rows are recomputed from a
        # fresh filtered copy using the same start grid the main aggregate and
        # the snapshot window comparison use, so a snapshot compared left
        # against itself matches the comparison's window rows item by item.
        # Nothing is written, so identical requests return byte-identical
        # JSON.
        occurred = snapshot.occurred_at_values(params["type"])
        windows = _windows_from_occurred(
            occurred, params["windowSize"], params["from"], params["to"]
        )
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "snapshotId": snapshot.snapshot_id,
                "type": params["type"],
                "windowSize": params["windowSize"],
                "from": params["from"],
                "to": params["to"],
                "windows": windows,
            },
            newline=True,
        )

    def _aggregate_snapshot_events_by_region(
        self, snapshot_id: str, query: str
    ) -> None:
        """Serve GET /snapshots/{snapshotId}/events/region/aggregate.

        Read-only region window aggregation over one immutable snapshot's
        captured events — the single-snapshot counterpart of the main
        ``GET /events/region/aggregate``, with the exact window division the
        main event aggregate uses. The verdict order is fixed: the credential
        is authenticated first, then the query shape is validated, then the
        requested organization is compared with the credential's, and only
        then is the snapshot name resolved (missing snapshot, then foreign
        snapshot). Both ``read`` and ``write`` credentials may call it.
        """
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
        # The snapshot must already exist; an aggregate never implicitly
        # creates one, and a foreign snapshot is forbidden rather than not
        # found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != params["organizationId"]:
            self._forbidden(newline=True)
            return
        # The snapshot already scopes its captured events to the owning
        # organization at capture time; the window rows are recomputed from a
        # fresh filtered copy using the same verbatim region rule and the
        # same start grid the main region aggregate uses. Nothing is written,
        # so identical requests return byte-identical JSON.
        occurred = snapshot.occurred_at_values_for_region(
            params["type"], params["region"]
        )
        windows = _windows_from_occurred(
            occurred, params["windowSize"], params["from"], params["to"]
        )
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "snapshotId": snapshot.snapshot_id,
                "region": params["region"],
                "type": params["type"],
                "windowSize": params["windowSize"],
                "from": params["from"],
                "to": params["to"],
                "windows": windows,
            },
            newline=True,
        )

    def _replay_snapshot_decisions(self, snapshot_id: str, query: str) -> None:
        """Serve GET /snapshots/{snapshotId}/events/replay/decisions.

        Read-only, step-by-step replay of one immutable snapshot's captured
        events — the single-snapshot counterpart of
        ``GET /events/replay/decisions``, lowering the exact window and peak
        contract of the main replay-decision query onto the events the
        snapshot captured at its creation time. The verdict order is fixed:
        the credential is authenticated first, then the query shape is
        validated, then the requested organization is compared with the
        credential's, and only then is the snapshot name resolved (missing
        snapshot, then foreign snapshot). Both ``read`` and ``write``
        credentials may call it.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _replay_decision_params_from_query(query)
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
        # The snapshot must already exist; a replay never implicitly creates
        # one, and a foreign snapshot is forbidden rather than not found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != params["organizationId"]:
            self._forbidden(newline=True)
            return
        # The snapshot already scopes its captured events to the owning
        # organization at capture time and is immutable afterwards, so other
        # organizations' data and events committed after the capture never
        # enter the replay. The rows are sorted by occurredAt then eventId —
        # the replay order the main replay-decision query uses — and each
        # step recomputes the window rows and peak through the same helpers
        # as the aggregate and decision entry points, so every step agrees
        # item by item with what those endpoints return on the same
        # accumulated prefix. Nothing is written, so identical requests
        # return byte-identical JSON.
        events = event_rows(snapshot.events_snapshot())
        steps = replay_decision_steps(events, params)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "snapshotId": snapshot.snapshot_id,
                "type": params["type"],
                "windowSize": params["windowSize"],
                "threshold": params["threshold"],
                "from": params["from"],
                "to": params["to"],
                "steps": steps,
            },
            newline=True,
        )

    def _replay_snapshot_region_decisions(
        self, snapshot_id: str, query: str
    ) -> None:
        """Serve GET /snapshots/{snapshotId}/events/region/replay/decisions.

        Read-only, step-by-step region replay of one immutable snapshot's
        captured events — the single-snapshot counterpart of
        ``GET /events/region/replay/decisions``, lowering the exact region,
        window and peak contract of the main region replay-decision query
        onto the events the snapshot captured at its creation time. The
        verdict order is fixed: the credential is authenticated first, then
        the query shape is validated, then the requested organization is
        compared with the credential's, and only then is the snapshot name
        resolved (missing snapshot, then foreign snapshot). Both ``read``
        and ``write`` credentials may call it.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _region_replay_decision_params_from_query(query)
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
        # The snapshot must already exist; a replay never implicitly creates
        # one, and a foreign snapshot is forbidden rather than not found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != params["organizationId"]:
            self._forbidden(newline=True)
            return
        # The snapshot already scopes its captured events to the owning
        # organization at capture time and is immutable afterwards, so other
        # organizations' data and events committed after the capture never
        # enter the replay. The region filter uses the same verbatim rule the
        # main region replay uses (only a non-empty string payload ``region``
        # matches), and the rows are sorted by occurredAt then eventId — the
        # replay order the main replay-decision query uses — before each step
        # recomputes the window rows and peak through the same helpers as the
        # aggregate and decision entry points. Nothing is written, so
        # identical requests return byte-identical JSON.
        events = event_rows(
            snapshot.events_snapshot_for_region(params["region"])
        )
        steps = replay_decision_steps(events, params)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "snapshotId": snapshot.snapshot_id,
                "region": params["region"],
                "type": params["type"],
                "windowSize": params["windowSize"],
                "threshold": params["threshold"],
                "from": params["from"],
                "to": params["to"],
                "steps": steps,
            },
            newline=True,
        )

    def _evaluate_snapshot_decision(self, snapshot_id: str) -> None:
        """Serve POST /snapshots/{snapshotId}/decisions/evaluate.

        Read-only peak decision over one immutable snapshot's captured
        events — the single-snapshot counterpart of the two-snapshot
        window decision (``POST /snapshots/compare``), lowering the exact
        window and peak contract of ``POST /decisions/evaluate`` onto the
        events the snapshot captured at its creation time. The verdict
        order is fixed: the credential is authenticated first, then the
        request body is validated, then the requested organization is
        compared with the credential's, and only then is the snapshot
        name resolved (missing snapshot, then foreign snapshot). Both
        ``read`` and ``write`` credentials may call it.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_decision_request(data)
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

        # The snapshot must already exist; an evaluation never implicitly
        # creates one, and a foreign snapshot is forbidden rather than not
        # found.
        snapshot = self.server.snapshots.get(snapshot_id)  # type: ignore[attr-defined]
        if snapshot is None:
            self._snapshot_not_found()
            return
        if snapshot.organization_id != params["organizationId"]:
            self._forbidden(newline=True)
            return

        # The snapshot already scopes its captured events to the owning
        # organization at capture time; the peak is recomputed from a fresh
        # filtered copy using the same window division the main decision and
        # the snapshot window comparison use, so a snapshot compared with
        # itself reports this exact peak on both sides. Nothing is written —
        # neither the snapshot, nor the main-service ledger, inventory, or
        # alert state — so identical requests return byte-identical JSON.
        occurred = snapshot.occurred_at_values(params["type"])
        result = evaluate_decision(occurred, params)
        result["snapshotId"] = snapshot.snapshot_id
        self._write_json(HTTPStatus.OK, result, newline=True)

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

    def _list_branch_resources(self, branch_id: str, query: str) -> None:
        """Serve GET /branches/{branchId}/resources.

        Read-only per-resource balance listing for one branch. The verdict
        order is fixed: the credential is authenticated first, then the
        query shape is validated, then the requested organization is
        compared with the credential's, and only then is the branch name
        resolved (missing branch, then foreign branch). Both ``read`` and
        ``write`` credentials may call it.
        """
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
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return
        # The branch must already exist; a listing never implicitly creates
        # one, and a foreign branch is forbidden rather than not found.
        branch = self.server.branches.get(branch_id)  # type: ignore[attr-defined]
        if branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, branch, newline=True):
            return
        # The balances are recomputed from one locked snapshot of the
        # branch's own inventory (only the requested organization's
        # resources contribute) — the same mapping the resource comparison
        # aligns on, so a branch compared with itself agrees row by row.
        # Nothing is written, so identical requests return byte-identical
        # JSON.
        balances = branch.reservations.resource_balances_for_organization(
            organization_id
        )
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": organization_id,
                "branchId": branch.branch_id,
                "resources": resource_balance_rows(balances),
            },
            newline=True,
        )

    def _replay_branch_decisions(self, branch_id: str, query: str) -> None:
        """Serve GET /branches/{branchId}/events/replay/decisions.

        Read-only, step-by-step replay of one branch's own events for the
        caller's organization — the single-branch counterpart of
        ``GET /events/replay/decisions``, lowering the exact window and peak
        contract of the main replay-decision query onto the events that live
        in that branch (its forked snapshot plus branch-only commits). The
        verdict order is fixed: the credential is authenticated first, then
        the query shape is validated, then the requested organization is
        compared with the credential's, and only then is the branch name
        resolved (missing branch, then foreign branch). Both ``read`` and
        ``write`` credentials may call it.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _replay_decision_params_from_query(query)
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
        # The branch must already exist; a replay never implicitly creates
        # one, and a foreign branch is forbidden rather than not found.
        branch = self.server.branches.get(branch_id)  # type: ignore[attr-defined]
        if branch is None:
            self._branch_not_found()
            return
        if not self._allow_branch(subject, branch, newline=True):
            return
        # The branch ledger already scopes its events to the forked
        # snapshot's owning organization; other organizations' data and
        # writes committed elsewhere never enter the replay, and branch
        # reads never touch the main service or any other branch. The rows
        # are sorted by occurredAt then eventId — the replay order the main
        # replay-decision query uses — and each step recomputes the window
        # rows and peak through the same helpers as the aggregate and
        # decision entry points, so every step agrees item by item with what
        # the branch aggregate and decision endpoints return on the same
        # accumulated prefix. Nothing is written, so identical requests
        # return byte-identical JSON.
        events = branch.ledger.list_for_organization(params["organizationId"])
        steps = replay_decision_steps(events, params)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "branchId": branch.branch_id,
                "type": params["type"],
                "windowSize": params["windowSize"],
                "threshold": params["threshold"],
                "from": params["from"],
                "to": params["to"],
                "steps": steps,
            },
            newline=True,
        )

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

    def _compare_branch_resources(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_branch_resource_compare_request(data)
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

        # Each side recomputes its per-resource balances from one locked
        # snapshot of its own inventory (only the requested organization's
        # resources contribute), then the rows and groups are computed purely
        # from those copies; neither branch, the main service, nor any alert
        # state is written, so identical submissions return byte-identical
        # JSON.
        left_balances = left_branch.reservations.resource_balances_for_organization(
            organization_id
        )
        right_balances = (
            right_branch.reservations.resource_balances_for_organization(
                organization_id
            )
        )
        result = compare_branch_resources(left_balances, right_balances)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_branch_replay_decisions(self) -> None:
        """Serve POST /branches/compare/replay/decisions.

        Read-only, step-by-step alignment of two branches' replay decisions.
        The verdict order matches the other branch comparisons: the
        credential is authenticated first, then the media type and body
        shape are validated, then the requested organization is compared
        with the credential's, and only then are the branch names resolved
        (left before right; a missing name is branch_not_found, a foreign
        branch is forbidden). Both ``read`` and ``write`` credentials may
        call it; a comparison never writes either branch, the main service,
        inventory, or alerts and never implicitly creates a branch.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(
            newline=True, reject_duplicate_keys=True
        )
        if data is _BODY_ERROR:
            return

        try:
            params = validate_branch_replay_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]

        # The organization decision happens before either branch name is
        # inspected, so a foreign organization cannot probe branch names.
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Both branches must already exist; left is resolved before right so
        # a missing left outranks any problem on the right.
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

        # Each side reads its own ledger in one locked snapshot (the rows
        # are already scoped to the organization and sorted by occurredAt
        # then eventId — the single-branch replay order), then the aligned
        # steps are recomputed purely from those copies. Nothing is written,
        # so identical submissions return byte-identical JSON.
        left_events = left_branch.ledger.list_for_organization(organization_id)
        right_events = right_branch.ledger.list_for_organization(organization_id)
        result = compare_branch_replay_decisions(
            left_events, right_events, params
        )
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_snapshots(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_snapshot_compare_request(data)
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

        # Each side recomputes its window counts and peak from its own
        # immutable captured events; nothing in either snapshot, in any
        # branch, or in the main service is written, so identical
        # submissions return byte-identical JSON.
        left_occurred = left_snapshot.occurred_at_values(params["type"])
        right_occurred = right_snapshot.occurred_at_values(params["type"])
        result = compare_snapshots(left_occurred, right_occurred, params)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_snapshot_replay_decisions(self) -> None:
        """Serve POST /snapshots/compare/replay/decisions.

        Read-only, step-by-step alignment of two snapshots' replay
        decisions. The verdict order matches the other snapshot
        comparisons: the credential is authenticated first, then the media
        type and body shape are validated, then the requested organization
        is compared with the credential's, and only then are the snapshot
        names resolved (left before right; a missing name is
        snapshot_not_found, a foreign snapshot is forbidden). Both
        ``read`` and ``write`` credentials may call it; a comparison never
        writes either snapshot, any branch, the main service, inventory,
        or alerts and never implicitly creates a snapshot.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(
            newline=True, reject_duplicate_keys=True
        )
        if data is _BODY_ERROR:
            return

        try:
            params = validate_snapshot_replay_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]

        # The organization decision happens before either snapshot name is
        # inspected, so a foreign organization cannot probe snapshot names.
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Both snapshots must already exist; left is resolved before right
        # so a missing left outranks any problem on the right, and a foreign
        # snapshot is forbidden before the other name is even looked up.
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
        # at capture time; the deep copies are sorted by occurredAt then
        # eventId — the single-snapshot replay order — and the aligned steps
        # are recomputed purely from those copies through the same helper as
        # the branch replay comparison. Snapshots are immutable and events
        # committed after a capture never enter, so identical submissions
        # return byte-identical JSON and nothing is written.
        left_events = event_rows(left_snapshot.events_snapshot())
        right_events = event_rows(right_snapshot.events_snapshot())
        result = compare_snapshot_replay_decisions(
            left_events, right_events, params
        )
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_region_alert_replay_decisions(self) -> None:
        """Serve POST /alerts/regions/compare/replay/decisions.

        Read-only, step-by-step alignment of two regions' alert replay
        decisions: the region-alert counterpart of the branch/snapshot
        replay comparisons. The verdict order matches the other POST
        comparisons: the credential is authenticated first, then the media
        type and body shape are validated, and only then is the requested
        organization compared with the credential's. Region names are
        matched verbatim against the ledger and are never existence-checked
        or implicitly created: an unknown region simply contributes no
        events, and using the same region name for both sides is legal.
        Both ``read`` and ``write`` credentials may call it; neither side
        touches the alert store, the ledger, or the reservation inventory,
        and identical submissions return byte-identical JSON.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(
            newline=True, reject_duplicate_keys=True
        )
        if data is _BODY_ERROR:
            return

        try:
            params = validate_region_alert_replay_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Each side takes one locked ledger snapshot scoped to the verbatim
        # region; an unknown region just returns no rows. The aligned alert
        # simulation is then recomputed purely from those copies with
        # private per-side state, so the alert store and its counters are
        # never read or advanced and repeated calls are byte-identical.
        left_events = self.server.ledger.list_for_organization_region(  # type: ignore[attr-defined]
            organization_id, params["left"]
        )
        right_events = self.server.ledger.list_for_organization_region(  # type: ignore[attr-defined]
            organization_id, params["right"]
        )
        result = compare_region_alert_replay_decisions(
            left_events, right_events, params
        )
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_type_alert_replay_decisions(self) -> None:
        """Serve POST /alerts/types/compare/replay/decisions.

        Read-only, step-by-step alignment of two event types' alert replay
        decisions: the type-alert counterpart of the region comparison. The
        verdict order matches the other POST comparisons: the credential is
        authenticated first, then the media type and body shape are
        validated, and only then is the requested organization compared
        with the credential's. Type names are matched verbatim against the
        ledger and are never existence-checked or implicitly created: an
        unknown type simply contributes no events, and using the same type
        name for both sides is legal. Both ``read`` and ``write``
        credentials may call it; neither side touches the alert store, the
        ledger, or the reservation inventory, and identical submissions
        return byte-identical JSON.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(
            newline=True, reject_duplicate_keys=True
        )
        if data is _BODY_ERROR:
            return

        try:
            params = validate_type_alert_replay_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Each side takes one locked ledger snapshot of the organization's
        # events and keeps only its own verbatim type; an unknown type just
        # yields no rows. The aligned alert simulation is then recomputed
        # purely from those copies with private per-side state, so the
        # alert store and its counters are never read or advanced and
        # repeated calls are byte-identical.
        events = self.server.ledger.list_for_organization(  # type: ignore[attr-defined]
            organization_id
        )
        result = compare_type_alert_replay_decisions(events, events, params)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_event_type_replay_decisions(self) -> None:
        """Serve POST /events/types/compare/replay/decisions.

        Read-only, step-by-step alignment of two event types' plain event
        replay decisions: the non-alert counterpart of
        ``POST /alerts/types/compare/replay/decisions``, layering the type
        dimension onto ``GET /events/replay/decisions`` exactly the way the
        alert entry point layers it onto ``GET /alerts/replay/decisions``.
        The verdict order matches the other POST comparisons: the
        credential is authenticated first, then the media type and body
        shape are validated, and only then is the requested organization
        compared with the credential's. Type names are matched verbatim
        against the ledger and are never existence-checked or implicitly
        created: an unknown type simply contributes no events, and using
        the same type name for both sides is legal. Both ``read`` and
        ``write`` credentials may call it; the decision is threshold-only
        (observe versus escalate) with no alert identity or suppression
        count, and neither the alert store, the ledger, nor the
        reservation inventory is touched. Identical submissions return
        byte-identical JSON.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(
            newline=True, reject_duplicate_keys=True
        )
        if data is _BODY_ERROR:
            return

        try:
            params = validate_event_type_replay_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # One locked ledger snapshot of the organization's events; the pure
        # alignment function keeps each side's verbatim type and recomputes
        # both prefixes from copies, so the read is repeatable with no
        # writes to the ledger, inventory, or alert store.
        events = self.server.ledger.list_for_organization(  # type: ignore[attr-defined]
            organization_id
        )
        result = compare_event_type_replay_decisions(events, events, params)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_event_regions(self) -> None:
        """Serve POST /events/regions/compare.

        Read-only, single-shot window comparison of two regions on the
        main ledger: the region counterpart of ``POST /branches/compare``
        and the single-shot counterpart of
        ``POST /events/regions/compare/replay/decisions``, counting each
        side's full event set once instead of aligning step-by-step
        prefixes. The verdict order matches the other POST comparisons:
        the credential is authenticated first, then the media type and
        body shape are validated, and only then is the requested
        organization compared with the credential's. Region names follow
        the public attribution rule — only a non-empty string payload
        region attributes an event, matched verbatim — and are never
        existence-checked or implicitly created: an unknown region simply
        contributes zero events, and using the same region name for both
        sides is legal. Both ``read`` and ``write`` credentials may call
        it; the decision is threshold-only (observe versus escalate) with
        no alert identity or suppression count, and neither the alert
        store, the ledger, nor the reservation inventory is touched.
        Identical submissions return byte-identical JSON.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(
            newline=True, reject_duplicate_keys=True
        )
        if data is _BODY_ERROR:
            return

        try:
            params = validate_event_region_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Each side takes one locked ledger snapshot scoped to the verbatim
        # region; an unknown region just returns no rows. The pure
        # comparison function re-asserts the same attribution rule and
        # recomputes both sides from copies, so the read is repeatable with
        # no writes to the ledger, inventory, or alert store.
        left_events = self.server.ledger.list_for_organization_region(  # type: ignore[attr-defined]
            organization_id, params["left"]
        )
        right_events = self.server.ledger.list_for_organization_region(  # type: ignore[attr-defined]
            organization_id, params["right"]
        )
        result = compare_event_regions(left_events, right_events, params)
        self._write_json(HTTPStatus.OK, result, newline=True)

    def _compare_event_region_replay_decisions(self) -> None:
        """Serve POST /events/regions/compare/replay/decisions.

        Read-only, step-by-step alignment of two regions' plain event
        replay decisions: the region counterpart of
        ``POST /events/types/compare/replay/decisions`` and the non-alert
        counterpart of
        ``POST /alerts/regions/compare/replay/decisions``. The verdict
        order matches the other POST comparisons: the credential is
        authenticated first, then the media type and body shape are
        validated, and only then is the requested organization compared
        with the credential's. Region names follow the public attribution
        rule — only a non-empty string payload region attributes an
        event, matched verbatim — and are never existence-checked or
        implicitly created: an unknown region simply contributes no
        events, and using the same region name for both sides is legal.
        Both ``read`` and ``write`` credentials may call it; the decision
        is threshold-only (observe versus escalate) with no alert
        identity or suppression count, and neither the alert store, the
        ledger, nor the reservation inventory is touched. Identical
        submissions return byte-identical JSON.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(
            newline=True, reject_duplicate_keys=True
        )
        if data is _BODY_ERROR:
            return

        try:
            params = validate_event_region_replay_compare_request(data)
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=True,
            )
            return

        organization_id = params["organizationId"]
        if not self._authorize_organization(
            subject, organization_id, newline=True
        ):
            return

        # Each side takes one locked ledger snapshot scoped to the verbatim
        # region; an unknown region just returns no rows. The pure alignment
        # function re-asserts the same attribution rule and recomputes both
        # prefixes from copies, so the read is repeatable with no writes to
        # the ledger, inventory, or alert store.
        left_events = self.server.ledger.list_for_organization_region(  # type: ignore[attr-defined]
            organization_id, params["left"]
        )
        right_events = self.server.ledger.list_for_organization_region(  # type: ignore[attr-defined]
            organization_id, params["right"]
        )
        result = compare_event_region_replay_decisions(
            left_events, right_events, params
        )
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

    def _compare_snapshot_resources(self) -> None:
        subject = self._require_subject(newline=True)
        if subject is None:
            return

        data = self._json_request_body(newline=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_snapshot_resource_compare_request(data)
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

        # Each side's capacities and reservations were already scoped to the
        # owning organization at capture time; the balances are recomputed
        # from deep copies, so a comparison never shares mutable snapshot
        # state and nothing is written, making identical submissions
        # byte-identical JSON.
        left_balances = left_snapshot.resource_balances()
        right_balances = right_snapshot.resource_balances()
        result = compare_snapshot_resources(left_balances, right_balances)
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

    def _replay_decisions(self, ledger: EventLedger, query: str) -> None:
        # Verdict order matches the other replay queries: the credential is
        # checked first, then the query shape, then the organization. Both
        # read and write credentials may call it; the read is one locked
        # snapshot and nothing is written, so repeated calls return the
        # same bytes and never disturb the ledger, inventory, or alerts.
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _replay_decision_params_from_query(query)
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

        # A single locked snapshot of the organization's events, already
        # scoped by the ledger to this organization and sorted by occurredAt
        # then eventId; replay_decision_steps narrows to the requested type.
        events = ledger.list_for_organization(params["organizationId"])
        steps = replay_decision_steps(events, params)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "type": params["type"],
                "windowSize": params["windowSize"],
                "threshold": params["threshold"],
                "from": params["from"],
                "to": params["to"],
                "steps": steps,
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

    def _replay_region_decisions(self, ledger: EventLedger, query: str) -> None:
        """Serve GET /events/region/replay/decisions.

        Read-only, step-by-step replay scoped to one region — the
        region-dimension counterpart of ``GET /events/replay/decisions``,
        applying the exact verbatim region attribution rule of
        ``GET /events/region`` before the same type, window, peak, and range
        contract. The verdict order matches the other replay queries: the
        credential is checked first, then the query shape, then the
        organization. Both read and write credentials may call it; the read
        is one locked snapshot and nothing is written, so repeated calls
        return the same bytes and never disturb the ledger, inventory, or
        alerts.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _region_replay_decision_params_from_query(query)
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

        # A single locked snapshot of the organization's region events,
        # already scoped by the ledger to this organization and the verbatim
        # region and sorted by occurredAt then eventId; replay_decision_steps
        # narrows to the requested type. Events without a non-empty string
        # payload region never match, and other organizations' data never
        # enters the result.
        events = ledger.list_for_organization_region(
            params["organizationId"], params["region"]
        )
        steps = replay_decision_steps(events, params)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "region": params["region"],
                "type": params["type"],
                "windowSize": params["windowSize"],
                "threshold": params["threshold"],
                "from": params["from"],
                "to": params["to"],
                "steps": steps,
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

    def _route_allocate_decision(self) -> None:
        subject = self._require_subject(newline=False)
        if subject is None:
            return

        data = self._json_request_body(reject_duplicate_keys=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_route_allocation_request(data)
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
        # alone; events, reservations, alerts, snapshots, and branches are
        # never touched, and capacity deductions live only for this call.
        result = plan_route_allocation(params)
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

    def _evaluate_region_alert(self) -> None:
        """Serve POST /alerts/region/evaluate.

        The region-dimension counterpart of ``POST /alerts/evaluate``: the
        same write-role, media-type, body, and suppression contract, with
        the peak computed only from the organization's events whose payload
        ``region`` is a non-empty string equal to the requested region
        (matched verbatim). An unknown region simply matches zero events.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        if subject.role != "write":
            self._forbidden(newline=True)
            return

        data = self._json_request_body(newline=True, reject_duplicate_keys=True)
        if data is _BODY_ERROR:
            return

        try:
            params = validate_region_alert_request(data)
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
            # snapshot of the matching region events. The role/org decision
            # and the alert record share the registry lock, so a read-only
            # credential is rejected before any record call and concurrent
            # threshold hits cannot both open an alert.
            occurred = self.server.ledger.occurred_at_values_for_region(  # type: ignore[attr-defined]
                organization_id, params["type"], params["region"]
            )
            peak = evaluate_decision(occurred, params)
            peak_start = peak["peakStart"]
            peak_count = peak["peakCount"]

            if peak_count < params["threshold"]:
                action = "observe"
                alert_id = None
                suppressed_count = None
            else:
                action, alert = self.server.alerts.record_for_region(  # type: ignore[attr-defined]
                    organization_id,
                    params["region"],
                    params["type"],
                    peak_start,
                    params["threshold"],
                    params["suppressionWindow"],
                )
                alert_id = alert["alertId"]
                suppressed_count = alert["suppressedCount"]

            return {
                "organizationId": organization_id,
                "region": params["region"],
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

    def _replay_alert_decisions(self, query: str) -> None:
        """Serve GET /alerts/replay/decisions.

        Read-only, step-by-step organization-dimension alert replay: the
        alert-suppression counterpart of ``GET /events/replay/decisions``.
        Matching events enter the replay one at a time and each step reports
        the accumulated window rows and peak plus the action and alert
        identity the suppression rule would produce *within the replay*. The
        alert store is never read or written, so simulated identifiers and
        suppression counts are independent of committed alerts and repeated
        calls are byte-for-byte identical. The verdict order matches the
        other query entry points: credential, then query shape, then
        organization; both read and write credentials may call it.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _alert_replay_params_from_query(query)
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

        # A single locked snapshot of the organization's events, sorted by
        # occurredAt then eventId; alert_replay_steps narrows to the requested
        # type (re-asserting that same order) and runs the whole suppression
        # simulation without touching the alert store, the ledger, or the
        # reservation inventory. An unknown organization simply yields no
        # events, and other organizations' data never enters the result.
        events = self.server.ledger.list_for_organization(  # type: ignore[attr-defined]
            params["organizationId"]
        )
        steps = alert_replay_steps(events, params)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "type": params["type"],
                "windowSize": params["windowSize"],
                "threshold": params["threshold"],
                "suppressionWindow": params["suppressionWindow"],
                "from": params["from"],
                "to": params["to"],
                "steps": steps,
            },
            newline=True,
        )

    def _replay_region_alert_decisions(self, query: str) -> None:
        """Serve GET /alerts/region/replay/decisions.

        Read-only, step-by-step region alert replay: the region-alert
        counterpart of ``GET /events/region/replay/decisions``. Matching
        events enter the replay one at a time and each step reports the
        accumulated window rows and peak plus the action and alert identity
        the suppression rule would produce *within the replay*. The alert
        store is never read or written, so simulated identifiers and
        suppression counts are independent of committed alerts and repeated
        calls are byte-for-byte identical. The verdict order matches the
        other query entry points: credential, then query shape, then
        organization; both read and write credentials may call it.
        """
        subject = self._require_subject(newline=True)
        if subject is None:
            return
        try:
            params = _region_alert_replay_params_from_query(query)
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

        # A single locked snapshot of the organization's region events,
        # scoped to the verbatim region and sorted by occurredAt then
        # eventId; alert_replay_steps narrows to the requested type
        # (re-asserting that same order) and runs the whole suppression
        # simulation without touching the alert store, the ledger, or the
        # reservation inventory.
        events = self.server.ledger.list_for_organization_region(  # type: ignore[attr-defined]
            params["organizationId"], params["region"]
        )
        steps = alert_replay_steps(events, params)
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "region": params["region"],
                "type": params["type"],
                "windowSize": params["windowSize"],
                "threshold": params["threshold"],
                "suppressionWindow": params["suppressionWindow"],
                "from": params["from"],
                "to": params["to"],
                "steps": steps,
            },
            newline=True,
        )

    def _list_region_alerts(self, query: str) -> None:
        """Serve GET /alerts/region.

        Read-only region-dimension alert listing; both ``read`` and
        ``write`` credentials may call it. The verdict order matches the
        other query entry points: the credential is checked first, then the
        query shape, then the organization. An unknown region returns an
        empty alert array and is never implicitly created.
        """
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
        alerts = self.server.alerts.list_for_organization_region(  # type: ignore[attr-defined]
            params["organizationId"], params["region"]
        )
        entries = [
            {
                "type": alert["type"],
                "peakStart": alert["peakStart"],
                "threshold": alert["threshold"],
                "suppressedCount": alert["suppressedCount"],
            }
            for alert in alerts
        ]
        self._write_json(
            HTTPStatus.OK,
            {
                "organizationId": params["organizationId"],
                "region": params["region"],
                "alerts": entries,
            },
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

    def _cancel_reservation(
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
            organization_id, reservation_id = validate_reservation_cancel_request(
                data
            )
        except EventValidationError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return

        def commit() -> tuple[str, dict[str, Any] | None]:
            return reservations.cancel(organization_id, reservation_id)

        # The cancel and the capacity release commit under the same lock
        # pair as every other write: the organization/role decision is made
        # by the registry lock and the inventory mutation by the inventory
        # lock, so a rejected request changes nothing and a concurrent
        # reservation cannot observe a half-released balance.
        status, result = self.server.tokens.commit_write(  # type: ignore[attr-defined]
            self.token, organization_id, commit
        )
        if status == "forbidden":
            self._forbidden(newline=newline)
            return
        cancel_status, view = result
        if cancel_status == "forbidden":
            self._forbidden(newline=newline)
            return
        if cancel_status == "not_found":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "reservation_not_found"},
                newline=newline,
            )
            return
        self._write_json(HTTPStatus.OK, view, newline=newline)

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

    def _json_request_body(
        self, *, newline: bool = False, reject_duplicate_keys: bool = False
    ) -> Any:
        """Validate the media type and decode the request body as JSON.

        On failure the 415/400 response is written here and the ``_BODY_ERROR``
        sentinel is returned; callers must return immediately. A duplicated
        JSON field is syntactically valid JSON, so when
        ``reject_duplicate_keys`` is set (an entry point whose contract makes
        a repeated field a validation failure) it is reported here as the 422
        validation error rather than the 400 syntax error.
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
            decoded = raw_body.decode("utf-8")
        except UnicodeDecodeError:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_json"},
                newline=newline,
            )
            return _BODY_ERROR
        try:
            return json.loads(
                decoded,
                object_pairs_hook=(
                    _reject_duplicate_keys if reject_duplicate_keys else None
                ),
            )
        except DuplicateJsonKeyError as exc:
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "validation_error", "message": str(exc)},
                newline=newline,
            )
            return _BODY_ERROR
        except json.JSONDecodeError:
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
