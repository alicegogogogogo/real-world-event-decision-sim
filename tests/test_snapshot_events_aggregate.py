"""Regression tests for GET /snapshots/{snapshotId}/events/aggregate.

The baseline already had the single-snapshot event listing and the
two-snapshot window decision comparison, but no snapshot-dimension window
aggregate; this locks down the new read-only entry point that lowers the
event-aggregation window contract onto the events one snapshot captured:

- only the snapshot's captured events of the caller's organization and the
  requested ``type`` are counted; other organizations' data and events
  committed after capture never enter the result;
- the window division is exactly the main aggregate's: windows start at zero
  and cover ``[start, start + windowSize)``; without ``from``/``to`` only
  windows hit by matching events are returned (``windows: []`` with none),
  and with a range every window intersecting the closed interval is kept,
  including empty windows counted as zero;
- the response echoes organization, snapshot, type, window width and range,
  is compact key-sorted JSON with integer values and one trailing newline,
  and identical requests are byte-for-byte identical without polluting each
  other;
- a snapshot aggregated against itself agrees row by row with the two
  snapshot window comparison's window rows;
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

    def aggregate_raw(
        self, query: str, *, snapshot: str = "s1", token: str | None = "w1"
    ) -> tuple[int, bytes]:
        return self.raw(
            f"/snapshots/{snapshot}/events/aggregate?{query}", token=token
        )

    @staticmethod
    def query(
        *,
        org: str = ORG1,
        event_type: str = EVENT_TYPE,
        window_size: int = 60,
        from_to: tuple[int, int] | None = None,
    ) -> str:
        from urllib.parse import urlencode

        params: dict[str, Any] = {
            "organizationId": org,
            "type": event_type,
            "windowSize": window_size,
        }
        if from_to is not None:
            params["from"], params["to"] = from_to
        return urlencode(params)

    # ------------------------------------------------------------- happy paths

    def test_aggregates_only_captured_events_into_windows(self) -> None:
        self.add_event("evt-1", occurred_at=0)
        self.add_event("evt-2", occurred_at=59)
        self.add_event("evt-3", occurred_at=60)
        self.add_event("evt-4", occurred_at=125, event_type="other.kind")
        self.capture()

        status, body = self.aggregate(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(body["type"], EVENT_TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 2},
                {"start": 60, "end": 120, "count": 1},
            ],
        )

    def test_window_is_left_closed_right_open_at_every_boundary(self) -> None:
        # An event exactly on a boundary belongs to the upper window, and an
        # event one tick before it to the lower one — same division as the
        # main event aggregate.
        for index, timestamp in enumerate((0, 59, 60, 119, 120)):
            self.add_event(f"evt-{index}", occurred_at=timestamp)
        self.capture()

        status, body = self.aggregate(self.query(window_size=60))
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 2},
                {"start": 60, "end": 120, "count": 2},
                {"start": 120, "end": 180, "count": 1},
            ],
        )

    def test_no_matching_events_without_range_is_empty_array(self) -> None:
        self.add_event("evt-1", occurred_at=10, event_type="other.kind")
        self.capture()
        status, body = self.aggregate(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

    def test_empty_snapshot_without_range_is_empty_array(self) -> None:
        self.capture()
        status, body = self.aggregate(self.query())
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
                "windows": [],
            },
        )

    def test_range_keeps_every_intersecting_window_including_empty(self) -> None:
        self.add_event("evt-1", occurred_at=65)
        self.add_event("evt-2", occurred_at=181)
        self.capture()

        status, body = self.aggregate(self.query(from_to=(30, 180)))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 30)
        self.assertEqual(body["to"], 180)
        # [30,180] intersects windows starting 0, 60, 120, 180; the event at
        # 181 falls outside the closed interval and its window (180) is kept
        # but counted as zero.
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 0},
                {"start": 60, "end": 120, "count": 1},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 0},
            ],
        )

    def test_range_filters_events_by_closed_interval(self) -> None:
        for index, timestamp in enumerate((0, 10, 59, 60, 61)):
            self.add_event(f"evt-{index}", occurred_at=timestamp)
        self.capture()

        status, body = self.aggregate(self.query(from_to=(10, 60)))
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 2},  # 10 and 59
                {"start": 60, "end": 120, "count": 1},  # 60 inclusive
            ],
        )

    def test_range_with_zero_matches_still_returns_empty_windows(self) -> None:
        self.add_event("evt-1", occurred_at=1000)
        self.capture()
        status, body = self.aggregate(self.query(from_to=(0, 120)))
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 0},
                {"start": 60, "end": 120, "count": 0},
                {"start": 120, "end": 180, "count": 0},
            ],
        )

    def test_from_equal_to_to_counts_one_boundary_point(self) -> None:
        self.add_event("evt-1", occurred_at=60)
        self.add_event("evt-2", occurred_at=59)
        self.capture()
        status, body = self.aggregate(self.query(from_to=(60, 60)))
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 60, "end": 120, "count": 1}],
        )

    def test_windows_sort_by_start_ascending_regardless_of_insertion(self) -> None:
        self.add_event("evt-c", occurred_at=181)
        self.add_event("evt-a", occurred_at=1)
        self.add_event("evt-b", occurred_at=61)
        self.capture()
        status, body = self.aggregate(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["start"] for row in body["windows"]],
            [0, 60, 180],
        )

    def test_events_committed_after_capture_never_enter(self) -> None:
        self.add_event("evt-1", occurred_at=10)
        self.capture()
        self.add_event("evt-2", occurred_at=10)
        self.add_event("evt-3", occurred_at=200)
        status, body = self.aggregate(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 0, "end": 60, "count": 1}],
        )

    def test_other_organizations_never_contribute(self) -> None:
        self.add_event("evt-1", occurred_at=10)
        self.capture("s1")
        self.add_event("evt-a", token="w2", occurred_at=10)
        self.add_event("evt-b", token="w2", occurred_at=200)
        self.capture("s2", token="w2")

        status, body = self.aggregate(self.query(), snapshot="s1")
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 0, "end": 60, "count": 1}],
        )

        status, body = self.aggregate(
            self.query(org=ORG2), snapshot="s2", token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 180, "end": 240, "count": 1},
            ],
        )

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()

        status, raw = self.aggregate_raw(
            "organizationId=org-1&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertIn(b'"start":60', raw)
        self.assertIn(b'"windowSize":60', raw)
        body = json.loads(raw)
        self.assertEqual(
            list(body),
            ["from", "organizationId", "snapshotId", "to", "type",
             "windowSize", "windows"],
        )
        self.assertEqual(
            list(body["windows"][0]), ["count", "end", "start"]
        )

    def test_repeated_requests_are_byte_identical_and_do_not_pollute(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.add_event("evt-2", occurred_at=50)
        self.capture()
        query = self.query(from_to=(0, 120))
        first = self.aggregate_raw(query)[1]
        # Main-service writes between reads must not perturb bytes or leak
        # into the snapshot; neither read affects the other.
        self.add_event("evt-3", occurred_at=10)
        self.add_event("evt-4", occurred_at=200)
        rest = [self.aggregate_raw(query)[1] for _ in range(3)]
        self.assertTrue(all(chunk == first for chunk in rest))

    # ------------------------------------------- consistency with comparison

    def test_self_comparison_window_rows_match_item_by_item(self) -> None:
        self.add_event("evt-1", occurred_at=5)
        self.add_event("evt-2", occurred_at=65)
        self.add_event("evt-3", occurred_at=125)
        self.add_event("evt-other", occurred_at=5, event_type="x.raised")
        self.capture()

        for from_to in (None, (0, 180), (30, 130)):
            with self.subTest(from_to=from_to):
                query = self.query(from_to=from_to)
                status, aggregated = self.aggregate(query)
                self.assertEqual(status, 200)

                compare_payload: dict[str, Any] = {
                    "organizationId": ORG1,
                    "left": "s1",
                    "right": "s1",
                    "type": EVENT_TYPE,
                    "windowSize": 60,
                    "threshold": 2,
                }
                if from_to is not None:
                    compare_payload["from"], compare_payload["to"] = from_to
                status, comparison = self.call(
                    "/snapshots/compare",
                    method="POST",
                    payload=compare_payload,
                )
                self.assertEqual(status, 200)

                # Every aggregate window row matches the self-comparison
                # row at the same start, count for count, in the same order;
                # both sides are equal by construction.
                self.assertEqual(
                    [row["start"] for row in comparison["windows"]],
                    [row["start"] for row in aggregated["windows"]],
                )
                for aggregate_row, compare_row in zip(
                    aggregated["windows"], comparison["windows"]
                ):
                    self.assertEqual(
                        aggregate_row["start"], compare_row["start"]
                    )
                    self.assertEqual(
                        aggregate_row["end"],
                        compare_row["start"] + 60,
                    )
                    self.assertEqual(
                        aggregate_row["count"], compare_row["leftCount"]
                    )
                    self.assertEqual(
                        compare_row["leftCount"], compare_row["rightCount"]
                    )
                    self.assertTrue(compare_row["equal"])

                # And the comparison's own peak agrees with the largest
                # aggregate row.
                peak = max(
                    (row["count"] for row in aggregated["windows"]),
                    default=0,
                )
                self.assertEqual(
                    comparison["decision"]["left"]["peakCount"], peak
                )
                self.assertTrue(comparison["decision"]["equal"])

    def test_matches_main_aggregate_at_capture_time_only(self) -> None:
        self.add_event("evt-1", occurred_at=10)
        self.add_event("evt-2", occurred_at=70)
        self.capture()
        self.add_event("evt-3", occurred_at=10)

        status, main_body = self.call(
            f"/events/aggregate?{self.query()}", token="w1"
        )
        self.assertEqual(status, 200)
        status, snapshot_body = self.aggregate(self.query())
        self.assertEqual(status, 200)
        # The main ledger has since moved on; the snapshot stays at capture.
        self.assertEqual(
            main_body["windows"],
            [
                {"start": 0, "end": 60, "count": 2},
                {"start": 60, "end": 120, "count": 1},
            ],
        )
        self.assertEqual(
            snapshot_body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
            ],
        )

    # -------------------------------------------------------------- read-only

    def test_aggregate_is_read_only(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()
        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, _ = self.aggregate(self.query())
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

    def test_read_credential_may_aggregate(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()
        status, body = self.aggregate(self.query(), token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 60, "end": 120, "count": 1}],
        )

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture()
        path = (
            "/snapshots/s1/events/aggregate"
            "?organizationId=org-1&type=incident.created&windowSize=60"
        )
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
        status, body = self.raw(
            "/snapshots/s1/events/aggregate", token=None
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- parameter 422

    def test_query_shape_errors_are_422(self) -> None:
        self.capture()
        base = "/snapshots/s1/events/aggregate"
        bad_queries = [
            "",
            "organizationId=org-1",
            "organizationId=org-1&type=incident.created",
            "organizationId=org-1&windowSize=60",
            "type=incident.created&windowSize=60",
            "organizationId=&type=incident.created&windowSize=60",
            "organizationId=%20%20&type=incident.created&windowSize=60",
            "organizationId=org-1&organizationId=org-1"
            "&type=incident.created&windowSize=60",
            "organizationId=org-1&type=&windowSize=60",
            "organizationId=org-1&type=%20&windowSize=60",
            "organizationId=org-1&type=incident.created&type=x&windowSize=60",
            "organizationId=org-1&type=incident.created",
            "organizationId=org-1&type=incident.created&windowSize=",
            "organizationId=org-1&type=incident.created&windowSize=%20",
            "organizationId=org-1&type=incident.created&windowSize=0",
            "organizationId=org-1&type=incident.created&windowSize=-3",
            "organizationId=org-1&type=incident.created&windowSize=1.5",
            "organizationId=org-1&type=incident.created&windowSize=abc",
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&windowSize=30",
            "organizationId=org-1&type=incident.created&windowSize=60&from=0",
            "organizationId=org-1&type=incident.created&windowSize=60&to=60",
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&from=-1&to=60",
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&from=0&to=-1",
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&from=abc&to=60",
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&from=61&to=60",
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&from=0&to=60&from=10",
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
            "/snapshots/ghost/events/aggregate"
            "?organizationId=org-2&organizationId=org-2&type=t&windowSize=60",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.capture()
        status, body = self.aggregate(
            self.query(org=ORG1), token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # The organization decision precedes even the snapshot lookup: an
        # unknown snapshot name is still 403 for a foreign organization.
        status, body = self.aggregate(
            self.query(org=ORG1), snapshot="ghost", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        status, body = self.aggregate(self.query(), snapshot="s2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.aggregate(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(snapshots["snapshots"], [])

    def test_snapshot_lookup_outranks_snapshot_ownership(self) -> None:
        self.capture("s2", token="w2")
        # A missing snapshot is 404 even though a foreign snapshot exists.
        status, body = self.aggregate(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    # ---------------------------------------------------------------- restart

    def test_new_server_instance_has_no_snapshots_or_events(self) -> None:
        self.add_event("evt-1", occurred_at=100)
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
