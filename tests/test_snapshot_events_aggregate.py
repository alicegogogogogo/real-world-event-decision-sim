"""Regression tests for GET /snapshots/{snapshotId}/events/aggregate.

The main event aggregate and the two-snapshot window comparison already
existed; this locks down the new read-only entry point that applies the
event aggregate's window discipline to one snapshot's captured events:

- ``organizationId``, ``type`` and ``windowSize`` are each required exactly
  once, ``windowSize`` must be positive integer text, and ``from``/``to``
  must be absent together or present together as non-negative integers with
  ``from <= to``; windows start at zero with fixed width and are left-closed
  and right-open, exactly like ``GET /events/aggregate``;
- without a range only windows covered by matching captured events are
  returned (empty match -> ``windows: []``); with a range every window
  intersecting the closed interval is returned, empty windows kept with
  ``count: 0``; rows are ordered by ``start`` ascending and the response
  echoes organization, snapshot, type, width, and range;
- only events the snapshot captured for that organization contribute: later
  main-service writes, other types, and other organizations never enter the
  result; a snapshot compared with itself via ``POST /snapshots/compare``
  agrees window row by window row;
- the response is compact, key-sorted JSON with integer values and one
  trailing newline; identical requests are byte-for-byte identical and reads
  never pollute each other or any state;
- the verdict order is fixed: 401 (credential) before 422 (query shape)
  before 403 (organization, then foreign snapshot) before 404
  (snapshot_not_found); both roles may call it, nothing is ever written or
  implicitly created, and restarting clears snapshots and events.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"
EVENT_TYPE = "incident.created"


def event_body(
    event_id: str,
    *,
    occurred_at: int,
    event_type: str = EVENT_TYPE,
    payload: dict[str, Any] | None = None,
    organization_id: str = ORG1,
) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": event_type,
        "occurredAt": occurred_at,
        "payload": {} if payload is None else payload,
    }


class SnapshotEventAggregateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.register("w1", ORG1, "write")
        self.register("w2", ORG2, "write")
        self.register("r1", ORG1, "read")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    # -------------------------------------------------------------- low level

    def raw(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        token: str | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, bytes]:
        headers: dict[str, str] = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except HTTPError as error:
            try:
                return error.code, error.read()
            finally:
                error.close()

    def call(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Any = None,
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        status, raw = self.raw(path, method=method, body=body, token=token)
        return status, json.loads(raw)

    def register(self, token: str, organization_id: str, role: str) -> None:
        status, _ = self.call(
            "/auth/tokens",
            method="POST",
            token=None,
            payload={
                "token": token,
                "organizationId": organization_id,
                "role": role,
            },
        )
        self.assertIn(status, (200, 201))

    # ------------------------------------------------------------- scaffolding

    def add_event(self, event_id: str, token: str = "w1", **kwargs: Any) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        status, _ = self.call(
            "/events",
            method="POST",
            token=token,
            payload=event_body(
                event_id, organization_id=organization_id, **kwargs
            ),
        )
        self.assertEqual(status, 201)

    def capture(
        self, snapshot_id: str = "s1", *, token: str = "w1"
    ) -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def aggregate(
        self,
        query: str,
        *,
        snapshot: str = "s1",
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/events/aggregate?{query}", token=token
        )

    def main_events(self, org: str = ORG1, *, token: str | None = "w1") -> list:
        status, body = self.call(f"/events?organizationId={org}", token=token)
        self.assertEqual(status, 200)
        return body["events"]

    # ------------------------------------------------------------- happy paths

    def test_without_range_returns_only_covered_windows(self) -> None:
        # Two events in [0,60), two in [60,120), one in [180,240).
        for event_id, occurred_at in (
            ("evt-a", 5),
            ("evt-b", 59),
            ("evt-c", 60),
            ("evt-d", 119),
            ("evt-e", 200),
        ):
            self.add_event(event_id, occurred_at=occurred_at)
        self.capture()

        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": ORG1,
                "snapshotId": "s1",
                "type": EVENT_TYPE,
                "windowSize": 60,
                "from": None,
                "to": None,
                "windows": [
                    {"start": 0, "end": 60, "count": 2},
                    {"start": 60, "end": 120, "count": 2},
                    {"start": 180, "end": 240, "count": 1},
                ],
            },
        )

    def test_boundary_event_belongs_to_upper_window(self) -> None:
        self.add_event("evt-a", occurred_at=59)
        self.add_event("evt-b", occurred_at=60)
        self.capture()
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
            ],
        )

    def test_with_range_keeps_empty_windows_and_filters_events(self) -> None:
        # Counts: two events in [0,60), two in [60,120), one at 200.
        for event_id, occurred_at in (
            ("evt-a", 5),
            ("evt-b", 59),
            ("evt-c", 60),
            ("evt-d", 119),
            ("evt-e", 200),
        ):
            self.add_event(event_id, occurred_at=occurred_at)
        self.capture()

        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&from=59&to=180"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 59)
        self.assertEqual(body["to"], 180)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 2},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 0},
            ],
        )

    def test_range_equal_endpoints_keeps_intersecting_window(self) -> None:
        self.add_event("evt-a", occurred_at=60)
        self.capture()
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created"
            "&windowSize=60&from=60&to=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 60, "end": 120, "count": 1}]
        )

    def test_no_matching_events_without_range_is_empty_array(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture()
        status, body = self.aggregate(
            "organizationId=org-1&type=no.such.type&windowSize=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

        # An empty snapshot is an empty match as well.
        self.register("w0", "org-empty", "write")
        self.capture("s2", token="w0")
        status, body = self.aggregate(
            "organizationId=org-empty&type=incident.created&windowSize=10",
            snapshot="s2",
            token="w0",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

    def test_no_matching_events_with_range_returns_zero_windows(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture()
        status, body = self.aggregate(
            "organizationId=org-1&type=no.such.type&windowSize=10&from=5&to=25"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 10, "count": 0},
                {"start": 10, "end": 20, "count": 0},
                {"start": 20, "end": 30, "count": 0},
            ],
        )

    # ------------------------------------------------- snapshot scoping rules

    def test_only_captured_events_contribute(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture()
        # Events committed after capture never enter the aggregate, even when
        # their windows would otherwise be returned.
        self.add_event("evt-b", occurred_at=20)
        self.add_event("evt-c", occurred_at=100)
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 0, "end": 60, "count": 1}]
        )

    def test_only_matching_type_contributes(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.add_event("evt-b", occurred_at=20, event_type="other.kind")
        self.add_event("evt-c", occurred_at=70, event_type="other.kind")
        self.capture()
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 0, "end": 60, "count": 1}]
        )

    def test_other_organizations_never_contribute(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture("s1")
        self.add_event("evt-x", token="w2", occurred_at=10)
        self.add_event("evt-y", token="w2", occurred_at=999)
        self.capture("s2", token="w2")

        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 0, "end": 60, "count": 1}]
        )

        status, body = self.aggregate(
            "organizationId=org-2&type=incident.created&windowSize=60",
            snapshot="s2",
            token="w2",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG2)
        self.assertEqual(body["snapshotId"], "s2")
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 960, "end": 1020, "count": 1},
            ],
        )

    def test_windows_match_main_aggregate_at_capture_time(self) -> None:
        for event_id, occurred_at in (
            ("evt-a", 5),
            ("evt-b", 59),
            ("evt-c", 200),
        ):
            self.add_event(event_id, occurred_at=occurred_at)
        self.capture()
        self.add_event("evt-later", occurred_at=70)

        query = (
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&from=0&to=240"
        )
        status, snapshot_body = self.aggregate(query)
        self.assertEqual(status, 200)
        status, main_body = self.call(f"/events/aggregate?{query}")
        self.assertEqual(status, 200)
        # The main ledger has gained evt-later since capture; the snapshot
        # aggregate must not.
        self.assertNotEqual(snapshot_body["windows"], main_body["windows"])
        self.assertEqual(
            snapshot_body["windows"],
            [
                {"start": 0, "end": 60, "count": 2},
                {"start": 60, "end": 120, "count": 0},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 1},
                {"start": 240, "end": 300, "count": 0},
            ],
        )
        self.assertEqual(
            main_body["windows"],
            [
                {"start": 0, "end": 60, "count": 2},
                {"start": 60, "end": 120, "count": 1},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 1},
                {"start": 240, "end": 300, "count": 0},
            ],
        )

    # ------------------------------------------- consistency with comparison

    def test_self_comparison_agrees_window_by_window(self) -> None:
        for event_id, occurred_at in (
            ("evt-a", 5),
            ("evt-b", 60),
            ("evt-c", 125),
        ):
            self.add_event(event_id, occurred_at=occurred_at)
        self.capture()

        query = (
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&from=0&to=180"
        )
        status, aggregate_body = self.aggregate(query)
        self.assertEqual(status, 200)
        status, comparison = self.call(
            "/snapshots/compare",
            method="POST",
            payload={
                "organizationId": ORG1,
                "left": "s1",
                "right": "s1",
                "type": EVENT_TYPE,
                "windowSize": 60,
                "threshold": 3,
                "from": 0,
                "to": 180,
            },
        )
        self.assertEqual(status, 200)
        # Each aggregate window row has an identical comparison row, item by
        # item in the same order, with equal left/right counts.
        self.assertEqual(
            [
                {"start": row["start"], "count": row["count"]}
                for row in aggregate_body["windows"]
            ],
            [
                {"start": row["start"], "count": row["leftCount"]}
                for row in comparison["windows"]
            ],
        )
        self.assertTrue(
            all(
                row["leftCount"] == row["rightCount"] and row["equal"]
                for row in comparison["windows"]
            )
        )
        self.assertEqual(
            [row["count"] for row in aggregate_body["windows"]],
            [1, 1, 1, 0],
        )

    def test_self_comparison_without_range_agrees_window_by_window(self) -> None:
        self.add_event("evt-a", occurred_at=5)
        self.add_event("evt-b", occurred_at=200)
        self.capture()

        status, aggregate_body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        status, comparison = self.call(
            "/snapshots/compare",
            method="POST",
            payload={
                "organizationId": ORG1,
                "left": "s1",
                "right": "s1",
                "type": EVENT_TYPE,
                "windowSize": 60,
                "threshold": 3,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [(row["start"], row["count"]) for row in aggregate_body["windows"]],
            [
                (row["start"], row["leftCount"]) for row in comparison["windows"]
            ],
        )

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event("evt-a", occurred_at=100)
        self.capture()

        status, raw = self.raw(
            "/snapshots/s1/events/aggregate"
            "?organizationId=org-1&type=incident.created&windowSize=60",
            token="w1",
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Widths and timestamps stay integers.
        self.assertIn(b'"windowSize":60', raw)
        self.assertIn(b'"start":60', raw)
        # Keys are sorted by code point at every level.
        body = json.loads(raw)
        self.assertEqual(
            list(body),
            ["from", "organizationId", "snapshotId", "to", "type",
             "windowSize", "windows"],
        )
        self.assertEqual(list(body["windows"][0]), ["count", "end", "start"])

    def test_repeated_requests_are_byte_identical_and_do_not_pollute(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.add_event("evt-b", occurred_at=80)
        self.capture()
        path = (
            "/snapshots/s1/events/aggregate"
            "?organizationId=org-1&type=incident.created&windowSize=60"
            "&from=0&to=120"
        )
        first = self.raw(path, token="w1")[1]
        # Main-service writes between reads must not perturb bytes or leak
        # into the snapshot; neither read affects the other.
        self.add_event("evt-c", occurred_at=11)
        rest = [self.raw(path, token="r1")[1] for _ in range(3)]
        self.assertTrue(all(chunk == first for chunk in rest))

    # -------------------------------------------------------------- read-only

    def test_aggregate_is_read_only(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture()

        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, _ = self.aggregate(
                "organizationId=org-1&type=incident.created&windowSize=60"
            )
            self.assertEqual(status, 200)
        status, after = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

        # Main-service ledger, reservations and alerts are untouched.
        self.assertEqual(len(self.main_events()), 1)
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_aggregate(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture()
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60",
            token="r1",
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 0, "end": 60, "count": 1}]
        )

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture()
        path = (
            "/snapshots/s1/events/aggregate"
            "?organizationId=org-1&type=incident.created&windowSize=60"
        )
        # Missing header.
        status, body = self.raw(path, token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"error": "unauthorized"})
        # Unregistered token.
        status, body = self.raw(path, token="forged")
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"error": "unauthorized"})
        # Malformed Authorization header (non-Bearer scheme).
        request = Request(
            f"{self.base_url}{path}",
            headers={"Authorization": "Basic abc"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=5) as response:
                status, body = response.status, response.read()
        except HTTPError as error:
            status, body = error.code, error.read()
            error.close()
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"error": "unauthorized"})

    def test_credential_outranks_query_validation(self) -> None:
        self.capture()
        # A missing credential is 401 even when the query is also invalid.
        status, body = self.raw(
            "/snapshots/s1/events/aggregate?organizationId=org-1", token=None
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- parameter 422

    def test_missing_duplicated_or_blank_parameters_are_422(self) -> None:
        self.capture()
        base = "/snapshots/s1/events/aggregate"
        for query in (
            "",
            "organizationId=org-1&type=incident.created",
            "organizationId=org-1&windowSize=60",
            "type=incident.created&windowSize=60",
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&windowSize=30",
            "organizationId=org-1&organizationId=org-1"
            "&type=incident.created&windowSize=60",
            "organizationId=org-1&type=incident.created&type=x&windowSize=60",
            "organizationId=&type=t&windowSize=1",
            "organizationId=%20%20&type=t&windowSize=1",
            "organizationId=o&type=&windowSize=1",
            "organizationId=o&type=t&windowSize=",
        ):
            with self.subTest(query=query):
                status, body = self.raw(f"{base}?{query}", token="w1")
                self.assertEqual(status, 422)
                self.assertEqual(json.loads(body)["error"], "validation_error")

    def test_invalid_values_are_422(self) -> None:
        self.capture()
        base = "/snapshots/s1/events/aggregate"
        for query in (
            "organizationId=o&type=t&windowSize=0",
            "organizationId=o&type=t&windowSize=-5",
            "organizationId=o&type=t&windowSize=1.5",
            "organizationId=o&type=t&windowSize=abc",
            "organizationId=o&type=t&windowSize=%2B5",
            "organizationId=o&type=t&windowSize=10&from=0",
            "organizationId=o&type=t&windowSize=10&to=0",
            "organizationId=o&type=t&windowSize=10&from=0&from=1&to=2",
            "organizationId=o&type=t&windowSize=10&from=-1&to=2",
            "organizationId=o&type=t&windowSize=10&from=0&to=x",
            "organizationId=o&type=t&windowSize=10&from=&to=2",
            "organizationId=o&type=t&windowSize=10&from=10&to=5",
        ):
            with self.subTest(query=query):
                status, body = self.raw(f"{base}?{query}", token="w1")
                self.assertEqual(status, 422)
                self.assertEqual(json.loads(body)["error"], "validation_error")

        # No failed validation created or altered anything.
        self.assertEqual(self.call("/snapshots")[1]["snapshots"][0]["events"], 0)

    def test_query_validation_outranks_organization_and_snapshot(self) -> None:
        self.capture()
        # A malformed query is 422 even when the organization would also
        # mismatch and the snapshot name is unknown.
        status, body = self.raw(
            "/snapshots/ghost/events/aggregate"
            "?organizationId=org-2&organizationId=org-2&type=t&windowSize=10",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.capture()
        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the snapshot name does not exist anywhere.
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60",
            snapshot="ghost",
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        # The organization matches the credential, but the snapshot belongs
        # to another organization.
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60",
            snapshot="s2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60",
            snapshot="ghost",
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        # The failed lookup did not implicitly create the snapshot.
        self.assertEqual(self.call("/snapshots")[1]["snapshots"], [])

    def test_snapshot_lookup_outranks_snapshot_ownership(self) -> None:
        self.capture("s2", token="w2")
        # A missing snapshot is 404 even though a foreign snapshot exists.
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60",
            snapshot="ghost",
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    # ------------------------------------------------------- unknown sub-path

    def test_unknown_snapshot_subpath_is_generic_not_found(self) -> None:
        self.capture()
        status, body = self.call(
            "/snapshots/s1/events/aggregate/extra?organizationId=org-1",
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    # ---------------------------------------------------------------- restart

    def test_new_server_instance_has_no_snapshots_or_events(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture()

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{fresh.server_port}"
            register = Request(
                f"{base_url}/auth/tokens",
                data=json.dumps(
                    {"token": "tok-fresh", "organizationId": ORG1, "role": "read"}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(register, timeout=5) as response:
                self.assertEqual(response.status, 201)
            request = Request(
                f"{base_url}/snapshots/s1/events/aggregate"
                "?organizationId=org-1&type=incident.created&windowSize=60",
                headers={"Authorization": "Bearer tok-fresh"},
                method="GET",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    status = response.status
            except HTTPError as error:
                status = error.code
                self.assertEqual(
                    json.loads(error.read())["error"], "snapshot_not_found"
                )
                error.close()
            self.assertEqual(status, 404)
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
