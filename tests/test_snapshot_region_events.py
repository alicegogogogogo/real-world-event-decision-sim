"""Regression tests for GET /snapshots/{snapshotId}/events/region.

The baseline already had the single-snapshot event listing and the
single-snapshot region aggregate, but no single-snapshot region event
listing; this locks down the new read-only entry point that lowers the main
service's region listing contract onto the events one snapshot captured:

- only events captured into the caller's own organization snapshot whose
  payload ``region`` is a non-empty string equal to the requested region
  (compared verbatim) are listed; a missing key, an empty string, or a
  non-string value never attributes an event, an unknown region matches zero
  events and is never implicitly created, and other organizations' data and
  events committed after capture never enter the list;
- each row carries exactly the five event fields and rows are sorted by
  ``occurredAt`` ascending and then ``eventId`` in Unicode code-point order,
  exactly like ``GET /events`` and ``GET /events/region``;
- the listed event set is exactly the set counted by the snapshot region
  aggregate, so the two endpoints reconcile event by event;
- the response echoes organization, snapshot, and region, is compact
  key-sorted JSON with integer values and one trailing newline, and
  identical requests are byte-for-byte identical without polluting each
  other;
- both ``read`` and ``write`` credentials may call it; the verdict order is
  fixed — 401 (credential) before 422 (query shape) before 403
  (organization, then foreign snapshot) before 404 (snapshot_not_found) —
  nothing is ever written or implicitly created, and restarting clears
  snapshots and events.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"
EVENT_TYPE = "incident.created"
REGION = "north"


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


class SnapshotRegionEventsTest(unittest.TestCase):
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

    def list_region(
        self,
        query: str,
        *,
        snapshot: str = "s1",
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/events/region?{query}", token=token
        )

    def list_region_raw(
        self, query: str, *, snapshot: str = "s1", token: str | None = "w1"
    ) -> tuple[int, bytes]:
        return self.raw(
            f"/snapshots/{snapshot}/events/region?{query}", token=token
        )

    @staticmethod
    def query(
        *,
        org: str = ORG1,
        region: str = REGION,
    ) -> str:
        return urlencode({"organizationId": org, "region": region})

    @staticmethod
    def row(
        event_id: str,
        occurred_at: int,
        *,
        event_type: str = EVENT_TYPE,
        **payload: Any,
    ) -> dict[str, Any]:
        return {
            "eventId": event_id,
            "organizationId": ORG1,
            "type": event_type,
            "occurredAt": occurred_at,
            "payload": dict(payload),
        }

    # ------------------------------------------------------------- happy paths

    def test_lists_only_captured_events_attributed_to_region(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=50, payload={"region": "north"})
        self.add_event("evt-3", occurred_at=70, payload={"region": "south"})
        self.add_event(
            "evt-4", occurred_at=60, event_type="other.kind",
            payload={"region": "north"},
        )
        self.capture()

        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(body["region"], REGION)
        # Region listing is not type-scoped: evt-4 (a different type) still
        # matches its region; the south event never does.
        self.assertEqual(
            body["events"],
            [
                self.row("evt-2", 50, region="north"),
                self.row("evt-4", 60, event_type="other.kind", region="north"),
                self.row("evt-1", 100, region="north"),
            ],
        )

    def test_rows_carry_exactly_the_five_event_fields(self) -> None:
        self.add_event(
            "evt-1",
            occurred_at=10,
            payload={"region": "north", "severity": "low", "note": "x"},
        )
        self.capture()
        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)
        self.assertEqual(
            list(body["events"][0]),
            ["eventId", "occurredAt", "organizationId", "payload", "type"],
        )
        self.assertEqual(
            body["events"][0],
            {
                "eventId": "evt-1",
                "organizationId": ORG1,
                "type": EVENT_TYPE,
                "occurredAt": 10,
                "payload": {"region": "north", "severity": "low", "note": "x"},
            },
        )

    def test_rows_sort_by_occurred_at_then_event_id(self) -> None:
        # Insert deliberately out of order, with an occurredAt tie.
        self.add_event("evt-9", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=50, payload={"region": "north"})
        self.add_event("evt-1", occurred_at=50, payload={"region": "north"})
        self.add_event("evt-10", occurred_at=50, payload={"region": "north"})
        self.capture()

        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["evt-1", "evt-10", "evt-2", "evt-9"],
        )
        self.assertEqual(
            [event["occurredAt"] for event in body["events"]],
            [50, 50, 50, 100],
        )

    def test_only_non_empty_string_region_attributes_an_event(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=10, payload={})
        self.add_event("evt-3", occurred_at=10, payload={"region": ""})
        self.add_event("evt-4", occurred_at=10, payload={"region": 7})
        self.add_event("evt-5", occurred_at=10, payload={"region": None})
        self.add_event("evt-6", occurred_at=10, payload={"region": ["north"]})
        self.add_event(
            "evt-7", occurred_at=10, payload={"region": "north", "other": None}
        )
        self.capture()

        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["evt-1", "evt-7"],
        )

    def test_region_matches_verbatim_without_normalization(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "North"})
        self.add_event("evt-2", occurred_at=10, payload={"region": "north "})
        self.add_event("evt-3", occurred_at=10, payload={"region": " north"})
        self.add_event("evt-4", occurred_at=10, payload={"region": "north"})
        self.capture()

        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual([event["eventId"] for event in body["events"]], ["evt-4"])

        status, body = self.list_region(self.query(region="North"))
        self.assertEqual(status, 200)
        self.assertEqual([event["eventId"] for event in body["events"]], ["evt-1"])

        status, body = self.list_region(self.query(region="north "))
        self.assertEqual(status, 200)
        self.assertEqual([event["eventId"] for event in body["events"]], ["evt-2"])

    def test_unknown_region_is_zero_match_and_never_created(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.capture()

        status, body = self.list_region(self.query(region="atlantis"))
        self.assertEqual(status, 200)
        self.assertEqual(body["region"], "atlantis")
        self.assertEqual(body["events"], [])

        # The unknown region still matches nothing afterwards, and the known
        # region is unaffected.
        status, body = self.list_region(self.query(region="atlantis"))
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)

    def test_empty_snapshot_returns_empty_array(self) -> None:
        self.capture()
        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": ORG1,
                "snapshotId": "s1",
                "region": REGION,
                "events": [],
            },
        )

    def test_events_committed_after_capture_never_enter(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.capture()
        self.add_event("evt-2", occurred_at=5, payload={"region": "north"})
        self.add_event("evt-3", occurred_at=200, payload={"region": "north"})
        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual([event["eventId"] for event in body["events"]], ["evt-1"])

    def test_other_organizations_never_contribute(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.capture("s1")
        self.add_event("evt-a", token="w2", occurred_at=5,
                       payload={"region": "north"})
        self.add_event("evt-b", token="w2", occurred_at=200,
                       payload={"region": "north"})
        self.capture("s2", token="w2")

        status, body = self.list_region(self.query(), snapshot="s1")
        self.assertEqual(status, 200)
        self.assertEqual([event["eventId"] for event in body["events"]], ["evt-1"])

        status, body = self.list_region(
            self.query(org=ORG2), snapshot="s2", token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]], ["evt-a", "evt-b"]
        )
        self.assertTrue(
            all(event["organizationId"] == ORG2 for event in body["events"])
        )

    # ----------------------------- consistency with the region aggregate

    def test_listing_matches_region_aggregate_event_set(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=70, payload={"region": "north"})
        self.add_event("evt-3", occurred_at=130, payload={"region": "north"})
        self.add_event("evt-4", occurred_at=10, payload={})
        self.add_event("evt-5", occurred_at=20, payload={"region": "south"})
        self.capture()

        status, listing = self.list_region(self.query())
        self.assertEqual(status, 200)
        aggregate_query = urlencode(
            {
                "organizationId": ORG1,
                "region": REGION,
                "type": EVENT_TYPE,
                "windowSize": 60,
            }
        )
        status, aggregate = self.call(
            f"/snapshots/s1/events/region/aggregate?{aggregate_query}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            len(listing["events"]),
            sum(window["count"] for window in aggregate["windows"]),
        )
        self.assertEqual(
            [event["eventId"] for event in listing["events"]],
            ["evt-1", "evt-2", "evt-3"],
        )
        self.assertEqual(
            aggregate["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
                {"start": 120, "end": 180, "count": 1},
            ],
        )

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()

        status, raw = self.list_region_raw("organizationId=org-1&region=north")
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertIn(b'"occurredAt":100', raw)
        body = json.loads(raw)
        self.assertEqual(
            list(body), ["events", "organizationId", "region", "snapshotId"]
        )
        self.assertEqual(
            list(body["events"][0]),
            ["eventId", "occurredAt", "organizationId", "payload", "type"],
        )
        self.assertEqual(
            raw,
            b'{"events":[{"eventId":"evt-1","occurredAt":100,'
            b'"organizationId":"org-1","payload":{"region":"north"},'
            b'"type":"incident.created"}],"organizationId":"org-1",'
            b'"region":"north","snapshotId":"s1"}\n',
        )

    def test_empty_match_byte_contract(self) -> None:
        self.capture()
        status, raw = self.list_region_raw("organizationId=org-1&region=north")
        self.assertEqual(status, 200)
        self.assertEqual(
            raw,
            b'{"events":[],"organizationId":"org-1","region":"north",'
            b'"snapshotId":"s1"}\n',
        )

    def test_repeated_requests_are_byte_identical_and_do_not_pollute(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=50, payload={"region": "north"})
        self.capture()
        query = self.query()
        first = self.list_region_raw(query)[1]
        # Main-service writes between reads must not perturb bytes or leak
        # into the snapshot; neither read affects the other.
        self.add_event("evt-3", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-4", occurred_at=200, payload={"region": "north"})
        rest = [self.list_region_raw(query)[1] for _ in range(3)]
        self.assertTrue(all(chunk == first for chunk in rest))

    # -------------------------------------------------------------- read-only

    def test_listing_is_read_only(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()
        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, _ = self.list_region(self.query())
            self.assertEqual(status, 200)
        status, after = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

        status, snapshot_events = self.call(
            "/snapshots/s1/events?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(snapshot_events["events"]), 1)
        status, events = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(events["events"]), 1)
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_list(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()
        status, body = self.list_region(self.query(), token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture()
        path = "/snapshots/s1/events/region?organizationId=org-1&region=north"

        status, body = self.raw(path, token=None)
        self.assertEqual(status, 401)
        self.assertEqual(body, b'{"error":"unauthorized"}\n')

        status, body = self.raw(path, token="forged")
        self.assertEqual(status, 401)
        self.assertEqual(body, b'{"error":"unauthorized"}\n')

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
        self.assertEqual(body, b'{"error":"unauthorized"}\n')

    def test_credential_outranks_query_validation(self) -> None:
        self.capture()
        # Missing credential is 401 even though the query is also invalid.
        status, body = self.raw("/snapshots/s1/events/region", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- parameter 422

    def test_query_shape_errors_are_422(self) -> None:
        self.capture()
        base = "/snapshots/s1/events/region"
        bad_queries = [
            "",
            "organizationId=org-1",
            "region=north",
            "organizationId=&region=north",
            "organizationId=%20%20&region=north",
            "organizationId=org-1&organizationId=org-1&region=north",
            "organizationId=org-1&region=",
            "organizationId=org-1&region=%20",
            "organizationId=org-1&region=north&region=south",
        ]
        for query in bad_queries:
            with self.subTest(query=query):
                status, body = self.raw(f"{base}?{query}", token="w1")
                self.assertEqual(status, 422)
                self.assertEqual(
                    json.loads(body)["error"], "validation_error"
                )

        # No failed validation created or altered anything.
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(len(snapshots["snapshots"]), 1)
        self.assertEqual(snapshots["snapshots"][0]["events"], 0)

    def test_unrelated_query_parameters_are_not_special(self) -> None:
        # Extra parameters do not satisfy the required ones; an unknown
        # parameter alongside a valid shape is simply ignored (mirroring the
        # main region listing, which parses named parameters).
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.capture()
        status, body = self.list_region(
            "organizationId=org-1&region=north&type=incident.created"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)

    def test_query_validation_outranks_organization_and_snapshot(self) -> None:
        self.capture()
        status, body = self.raw(
            "/snapshots/ghost/events/region"
            "?organizationId=org-2&organizationId=org-2&region=north",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.capture()
        status, body = self.list_region(
            self.query(org=ORG1), token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # The organization decision precedes even the snapshot lookup: an
        # unknown snapshot name is still 403 for a foreign organization.
        status, body = self.list_region(
            self.query(org=ORG1), snapshot="ghost", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        status, body = self.list_region(self.query(), snapshot="s2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.list_region(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(snapshots["snapshots"], [])

    def test_snapshot_lookup_outranks_snapshot_ownership(self) -> None:
        self.capture("s2", token="w2")
        # A missing snapshot is 404 even though a foreign snapshot exists.
        status, body = self.list_region(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    def test_failed_requests_leave_no_trace(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.capture()
        for _ in range(2):
            self.raw(
                "/snapshots/s1/events/region?organizationId=org-1&region=north",
                token=None,
            )
            self.raw(
                "/snapshots/ghost/events/region?organizationId=org-1",
                token="w1",
            )
            self.list_region(
                self.query(org=ORG2), snapshot="s1", token="w2"
            )
        status, body = self.list_region(self.query())
        self.assertEqual(status, 200)
        self.assertEqual([event["eventId"] for event in body["events"]], ["evt-1"])

    # ---------------------------------------------------------------- restart

    def test_new_server_instance_has_no_snapshots_or_events(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{fresh.server_port}"
            payload = json.dumps(
                {"token": "tok-fresh", "organizationId": ORG1, "role": "read"}
            ).encode()
            request = Request(
                f"{base_url}/auth/tokens",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 201)
            request = Request(
                f"{base_url}/snapshots/s1/events/region"
                "?organizationId=org-1&region=north",
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
