"""Regression tests for GET /snapshots/{snapshotId}/events/region.

The baseline already had the single-snapshot event listing and the main
service's region event list, but no snapshot-dimension region event list;
this locks down the new read-only entry point that lowers the region
attribution contract onto the events one snapshot captured:

- only the snapshot's captured events of the caller's organization
  attributed to the requested region are listed; region attribution
  follows the main rule (only a non-empty string payload ``region``,
  matched verbatim), an unknown region matches zero events and is never
  implicitly created, and other organizations' data — including events
  under the same region name — and events committed after capture never
  enter the result;
- each row reports the five event fields (``eventId``, ``organizationId``,
  ``type``, ``occurredAt``, ``payload``), rows are sorted by
  ``occurredAt`` ascending then ``eventId`` in Unicode code-point order,
  and the listed events are exactly the captured events the snapshot
  region aggregate counts for the same snapshot and region;
- the response echoes organization, snapshot and region, is compact
  key-sorted JSON with integer timestamps and one trailing newline, and
  identical requests are byte-for-byte identical without polluting each
  other;
- both ``read`` and ``write`` credentials may call it; the verdict order
  is fixed — 401 (credential) before 422 (query shape) before 403
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


class SnapshotRegionEventListTest(unittest.TestCase):
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

    def list_events(
        self,
        query: str,
        *,
        snapshot: str = "s1",
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/events/region?{query}", token=token
        )

    def list_raw(
        self, query: str, *, snapshot: str = "s1", token: str | None = "w1"
    ) -> tuple[int, bytes]:
        return self.raw(
            f"/snapshots/{snapshot}/events/region?{query}", token=token
        )

    @staticmethod
    def query(*, org: str = ORG1, region: str = REGION) -> str:
        return urlencode({"organizationId": org, "region": region})

    # ------------------------------------------------------------- happy paths

    def test_rows_echo_identifiers_and_report_five_fields(self) -> None:
        self.add_event(
            "evt-1", occurred_at=100, payload={"region": "north", "sev": "low"}
        )
        self.add_event(
            "evt-2",
            occurred_at=200,
            event_type="incident.updated",
            payload={"region": "north"},
        )
        self.capture()

        status, body = self.list_events(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(body["region"], REGION)
        self.assertEqual(
            body["events"],
            [
                {
                    "eventId": "evt-1",
                    "organizationId": ORG1,
                    "type": "incident.created",
                    "occurredAt": 100,
                    "payload": {"region": "north", "sev": "low"},
                },
                {
                    "eventId": "evt-2",
                    "organizationId": ORG1,
                    "type": "incident.updated",
                    "occurredAt": 200,
                    "payload": {"region": "north"},
                },
            ],
        )

    def test_only_matching_region_events_are_listed(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=20, payload={"region": "south"})
        self.add_event("evt-3", occurred_at=30, payload={})
        self.add_event("evt-4", occurred_at=40, payload={"region": "north"})
        self.capture()

        status, body = self.list_events(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["eventId"] for row in body["events"]], ["evt-1", "evt-4"]
        )

    def test_rows_sort_by_occurred_at_then_event_id(self) -> None:
        # Insert deliberately out of order, including equal timestamps.
        self.add_event("evt-c", occurred_at=200, payload={"region": "north"})
        self.add_event("evt-a", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-b2", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-b1", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-B", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-1", occurred_at=50, payload={"region": "north"})
        self.capture()

        status, body = self.list_events(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [(row["occurredAt"], row["eventId"]) for row in body["events"]],
            [
                (50, "evt-1"),
                (100, "evt-B"),
                (100, "evt-a"),
                (100, "evt-b1"),
                (100, "evt-b2"),
                (200, "evt-c"),
            ],
        )

    def test_only_non_empty_string_region_attributes_an_event(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=10, payload={})
        self.add_event("evt-3", occurred_at=10, payload={"region": ""})
        self.add_event("evt-4", occurred_at=10, payload={"region": 7})
        self.add_event("evt-5", occurred_at=10, payload={"region": None})
        self.add_event("evt-6", occurred_at=10, payload={"region": ["north"]})
        self.capture()

        status, body = self.list_events(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["eventId"] for row in body["events"]], ["evt-1"]
        )

    def test_region_matches_verbatim_without_normalization(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "North"})
        self.add_event("evt-2", occurred_at=10, payload={"region": "north "})
        self.add_event("evt-3", occurred_at=10, payload={"region": " north"})
        self.capture()

        status, body = self.list_events(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])

        status, body = self.list_events(self.query(region="North"))
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["eventId"] for row in body["events"]], ["evt-1"]
        )

    def test_unknown_region_is_empty_array_and_never_created(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.capture()

        status, body = self.list_events(self.query(region="atlantis"))
        self.assertEqual(status, 200)
        self.assertEqual(body["region"], "atlantis")
        self.assertEqual(body["events"], [])

        # The unknown region still matches nothing afterwards, and the
        # known region is unaffected.
        status, body = self.list_events(self.query(region="atlantis"))
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])
        status, body = self.list_events(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["eventId"] for row in body["events"]], ["evt-1"]
        )

    def test_empty_snapshot_returns_empty_array(self) -> None:
        self.capture()
        status, body = self.list_events(self.query())
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
        self.add_event("evt-2", occurred_at=20, payload={"region": "north"})
        self.add_event("evt-3", occurred_at=5, payload={"region": "north"})

        status, body = self.list_events(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["eventId"] for row in body["events"]], ["evt-1"]
        )

    def test_other_organizations_never_contribute(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.capture("s1")
        # ORG2 commits its own events under the same region name and
        # captures its own snapshot; nothing leaks into ORG1's listing.
        self.add_event(
            "evt-a", token="w2", occurred_at=10, payload={"region": "north"}
        )
        self.add_event(
            "evt-b", token="w2", occurred_at=200, payload={"region": "north"}
        )
        self.capture("s2", token="w2")

        status, body = self.list_events(self.query(), snapshot="s1")
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["eventId"] for row in body["events"]], ["evt-1"]
        )
        self.assertTrue(
            all(row["organizationId"] == ORG1 for row in body["events"])
        )

        # ORG2's own snapshot listing sees only ORG2 captured events.
        status, body = self.list_events(
            self.query(org=ORG2), snapshot="s2", token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG2)
        self.assertEqual(body["snapshotId"], "s2")
        self.assertEqual(
            [row["eventId"] for row in body["events"]], ["evt-a", "evt-b"]
        )

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()

        status, raw = self.list_raw("organizationId=org-1&region=north")
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Timestamps stay integers.
        self.assertIn(b'"occurredAt":100', raw)
        # Keys are sorted by code point at every level.
        body = json.loads(raw)
        self.assertEqual(
            list(body), ["events", "organizationId", "region", "snapshotId"]
        )
        self.assertEqual(
            list(body["events"][0]),
            ["eventId", "occurredAt", "organizationId", "payload", "type"],
        )

    def test_repeated_requests_are_byte_identical_and_do_not_pollute(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=50, payload={"region": "north"})
        self.capture()
        query = self.query()
        first = self.list_raw(query)[1]
        # Main-service writes between reads must not perturb bytes or leak
        # into the snapshot; neither read affects the other.
        self.add_event("evt-3", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-4", occurred_at=200, payload={"region": "north"})
        rest = [self.list_raw(query)[1] for _ in range(3)]
        self.assertTrue(all(chunk == first for chunk in rest))

    # -------------------------------------- consistency with region aggregate

    def test_listing_and_region_aggregate_hit_the_same_events(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=70, payload={"region": "north"})
        self.add_event(
            "evt-3",
            occurred_at=70,
            event_type="other.kind",
            payload={"region": "north"},
        )
        self.add_event("evt-4", occurred_at=70, payload={"region": "south"})
        self.capture()

        status, listing = self.list_events(self.query())
        self.assertEqual(status, 200)

        # For each type, the region aggregate's window counts sum to the
        # number of listed rows of that type — the two queries hit exactly
        # the same captured events.
        for event_type in (EVENT_TYPE, "other.kind"):
            status, aggregate = self.call(
                "/snapshots/s1/events/region/aggregate?"
                + urlencode(
                    {
                        "organizationId": ORG1,
                        "region": REGION,
                        "type": event_type,
                        "windowSize": 60,
                    }
                )
            )
            self.assertEqual(status, 200)
            listed = [
                row for row in listing["events"] if row["type"] == event_type
            ]
            self.assertEqual(
                sum(window["count"] for window in aggregate["windows"]),
                len(listed),
            )

    # -------------------------------------------------------------- read-only

    def test_listing_is_read_only(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()
        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, _ = self.list_events(self.query())
            self.assertEqual(status, 200)
        status, after = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

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
        status, body = self.list_events(self.query(), token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture()
        path = "/snapshots/s1/events/region?organizationId=org-1&region=north"
        status, body = self.raw(path, token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"error": "unauthorized"})

        status, body = self.raw(path, token="forged")
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"error": "unauthorized"})

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
            "organizationId=org-1&organizationId=org-2&region=north",
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
        status, body = self.list_events(self.query(org=ORG1), token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # The organization decision precedes even the snapshot lookup: an
        # unknown snapshot name is still 403 for a foreign organization.
        status, body = self.list_events(
            self.query(org=ORG1), snapshot="ghost", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        status, body = self.list_events(self.query(), snapshot="s2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.list_events(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(snapshots["snapshots"], [])

    def test_snapshot_lookup_outranks_snapshot_ownership(self) -> None:
        self.capture("s2", token="w2")
        # A missing snapshot is 404 even though a foreign snapshot exists.
        status, body = self.list_events(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

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
