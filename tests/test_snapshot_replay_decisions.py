"""Regression tests for GET /snapshots/{snapshotId}/events/replay/decisions.

The baseline already had the main-service replay decision query
(``GET /events/replay/decisions``) and several snapshot-dimension reads,
but no snapshot-dimension step-by-step replay; this locks down the new
read-only entry point that lowers the replay-decision contract onto the
events one snapshot captured:

- only the snapshot's captured events of the caller's organization and the
  requested ``type`` are replayed; other organizations' data and events
  committed after the capture never enter the result;
- captured events arrive ordered by ``occurredAt`` then ``eventId``; each
  step reports the event, the window rows accumulated so far (the exact
  main-aggregate window division, ranged empty windows kept), and the peak
  decision (largest count, ties to the earliest start, ``escalate`` at the
  threshold, ``observe`` otherwise);
- the response echoes organization, snapshot, type, window width,
  threshold and range, is compact key-sorted JSON with integer values and
  one trailing newline, and identical requests are byte-for-byte identical
  without polluting each other;
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


class SnapshotReplayDecisionsTest(unittest.TestCase):
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

    def capture(self, snapshot_id: str = "s1", *, token: str = "w1") -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def decisions(
        self,
        query: str,
        *,
        snapshot: str = "s1",
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/events/replay/decisions?{query}",
            token=token,
        )

    def decisions_raw(
        self,
        query: str,
        *,
        snapshot: str = "s1",
        token: str | None = "w1",
    ) -> tuple[int, bytes]:
        return self.raw(
            f"/snapshots/{snapshot}/events/replay/decisions?{query}",
            token=token,
        )

    @staticmethod
    def query(
        *,
        org: str = ORG1,
        event_type: str = EVENT_TYPE,
        window_size: int = 60,
        threshold: int = 3,
        from_to: tuple[int, int] | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "organizationId": org,
            "type": event_type,
            "windowSize": window_size,
            "threshold": threshold,
        }
        if from_to is not None:
            params["from"], params["to"] = from_to
        return urlencode(params)

    def seed_and_capture(self) -> None:
        self.add_event("evt-b", occurred_at=200)
        self.add_event("evt-a", occurred_at=200)
        self.add_event("evt-c", occurred_at=100)
        self.add_event("evt-d", occurred_at=0)
        self.add_event("evt-e", occurred_at=300)
        # A different type never opens a replay step for this query.
        self.add_event("evt-z", event_type="other.kind", occurred_at=100)
        # Another organization's events never enter the replay.
        self.add_event("evt-x", token="w2", occurred_at=10)
        self.capture()

    # ----------------------------------------------------------------- 200 shape

    def test_steps_follow_replay_order_with_event_id_and_time(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(body["type"], EVENT_TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 3)
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [
                ("evt-d", 0),
                ("evt-c", 100),
                ("evt-a", 200),
                ("evt-b", 200),
                ("evt-e", 300),
            ],
        )
        # No other-type and no other-organization event became a step.
        self.assertNotIn("evt-z", json.dumps(body))
        self.assertNotIn("evt-x", json.dumps(body))

    def test_each_step_accumulates_window_counts(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(self.query())
        self.assertEqual(status, 200)
        expected_counts = [
            {0: 1},
            {0: 1, 60: 1},
            {0: 1, 60: 1, 180: 1},
            {0: 1, 60: 1, 180: 2},
            {0: 1, 60: 1, 180: 2, 300: 1},
        ]
        self.assertEqual(len(body["steps"]), len(expected_counts))
        for step, counts in zip(body["steps"], expected_counts):
            self.assertEqual(
                [
                    (row["start"], row["end"], row["count"])
                    for row in step["windows"]
                ],
                [
                    (start, start + 60, count)
                    for start, count in sorted(counts.items())
                ],
            )
            peak_count = max(counts.values())
            peak_start = min(
                start for start, count in counts.items() if count == peak_count
            )
            self.assertEqual(step["peakCount"], peak_count)
            self.assertEqual(step["peakStart"], peak_start)
            self.assertEqual(
                step["action"],
                "escalate" if peak_count >= 3 else "observe",
            )

    def test_action_escalates_when_the_peak_reaches_the_threshold(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(self.query(threshold=2))
        self.assertEqual(status, 200)
        # The fourth step (evt-b) is the first to put two events in one window.
        self.assertEqual(
            [step["action"] for step in body["steps"]],
            ["observe", "observe", "observe", "escalate", "escalate"],
        )
        self.assertEqual(body["steps"][3]["peakStart"], 180)
        self.assertEqual(body["steps"][3]["peakCount"], 2)

    def test_peak_ties_resolve_to_the_earliest_start(self) -> None:
        for event_id, occurred_at in (
            ("evt-a", 0),
            ("evt-b", 100),
            ("evt-c", 200),
        ):
            self.add_event(event_id, occurred_at=occurred_at)
        self.capture()
        status, body = self.decisions(self.query(threshold=5))
        self.assertEqual(status, 200)
        for step in body["steps"][1:]:
            self.assertEqual(step["peakCount"], 1)
            self.assertEqual(step["peakStart"], 0)
            self.assertEqual(step["action"], "observe")

    def test_no_matching_events_produces_no_steps(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(self.query(event_type="never.seen"))
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": ORG1,
                "snapshotId": "s1",
                "type": "never.seen",
                "windowSize": 60,
                "threshold": 3,
                "from": None,
                "to": None,
                "steps": [],
            },
        )

    def test_events_committed_after_capture_never_enter(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture()
        # Writes after the capture belong to the main ledger only.
        self.add_event("evt-b", occurred_at=20)
        self.add_event("evt-c", occurred_at=30)
        status, body = self.decisions(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["evt-a"]
        )
        # The main-service replay sees all three events.
        status, main = self.call(
            f"/events/replay/decisions?{self.query()}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(main["steps"]), 3)

    # ------------------------------------------------------------- range semantics

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(self.query(from_to=(0, 180)))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 180)
        grid = [(0, 60), (60, 120), (120, 180), (180, 240)]
        for step in body["steps"]:
            self.assertEqual(
                [(row["start"], row["end"]) for row in step["windows"]], grid
            )
        # The events at 200 and 300 are outside [0, 180], so the 180 window
        # never gains a count here.
        self.assertEqual(
            [row["count"] for row in body["steps"][-1]["windows"]],
            [1, 1, 0, 0],
        )
        # Out-of-range events still arrive as replay steps in order.
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["evt-d", "evt-c", "evt-a", "evt-b", "evt-e"],
        )

    def test_step_outside_the_range_has_zero_count_null_start_observe(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.add_event("evt-b", occurred_at=70)
        self.capture()
        status, body = self.decisions(self.query(from_to=(60, 120)))
        self.assertEqual(status, 200)
        first, second = body["steps"]
        # The first accumulated event (at 10) counts in no ranged window.
        self.assertEqual(first["eventId"], "evt-a")
        self.assertEqual(
            first["windows"],
            [
                {"start": 60, "end": 120, "count": 0},
                {"start": 120, "end": 180, "count": 0},
            ],
        )
        self.assertEqual(first["peakCount"], 0)
        self.assertIsNone(first["peakStart"])
        self.assertEqual(first["action"], "observe")
        # The second event lands inside the range and opens the peak.
        self.assertEqual(second["peakCount"], 1)
        self.assertEqual(second["peakStart"], 60)
        self.assertEqual(second["action"], "observe")

    # --------------------------------------------- consistency with aggregate/decision

    def test_final_step_matches_snapshot_aggregate_and_decision(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(self.query(threshold=2))
        self.assertEqual(status, 200)

        status, aggregate = self.call(
            "/snapshots/s1/events/aggregate?"
            + urlencode(
                {
                    "organizationId": ORG1,
                    "type": EVENT_TYPE,
                    "windowSize": 60,
                }
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"][-1]["windows"], aggregate["windows"])

        status, decision = self.call(
            "/snapshots/s1/decisions/evaluate",
            method="POST",
            payload={
                "organizationId": ORG1,
                "type": EVENT_TYPE,
                "windowSize": 60,
                "threshold": 2,
            },
        )
        self.assertEqual(status, 200)
        last = body["steps"][-1]
        self.assertEqual(last["peakStart"], decision["peakStart"])
        self.assertEqual(last["peakCount"], decision["peakCount"])
        self.assertEqual(last["action"], decision["action"])

    # --------------------------------------------------------- serialization / roles

    def test_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_and_capture()
        status, raw = self.decisions_raw(self.query())
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        body = json.loads(raw)
        self.assertEqual(
            set(body),
            {
                "organizationId",
                "snapshotId",
                "type",
                "windowSize",
                "threshold",
                "from",
                "to",
                "steps",
            },
        )
        self.assertEqual(
            set(body["steps"][0]),
            {"eventId", "occurredAt", "windows", "peakStart", "peakCount", "action"},
        )
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )
        # Integers stay integers rather than spilling into floats/strings.
        self.assertIsInstance(body["windowSize"], int)
        self.assertIsInstance(body["threshold"], int)
        self.assertIsInstance(body["steps"][0]["occurredAt"], int)

    def test_repeated_requests_are_byte_identical_and_isolated(self) -> None:
        self.seed_and_capture()
        query = self.query(threshold=2)
        raws = [self.decisions_raw(query)[1] for _ in range(3)]
        self.assertEqual(raws[0], raws[1])
        self.assertEqual(raws[1], raws[2])
        # A ranged read between them does not change any later result.
        ranged = self.decisions_raw(self.query(threshold=2, from_to=(0, 180)))[1]
        self.assertNotEqual(ranged, raws[0])
        self.assertEqual(self.decisions_raw(query)[1], raws[0])

    def test_read_and_write_credentials_may_both_query(self) -> None:
        self.seed_and_capture()
        query = self.query()
        read_status, read_body = self.decisions(query, token="r1")
        write_status, write_body = self.decisions(query, token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    def test_query_is_read_only(self) -> None:
        self.seed_and_capture()
        self.decisions(self.query(from_to=(0, 180)))
        self.decisions(self.query(threshold=1))
        # The snapshot listing is unchanged and no snapshot was added.
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(
            snapshots["snapshots"],
            [{"snapshotId": "s1", "events": 6, "resources": 0, "reservations": 0}],
        )
        # The main ledger and alerts are untouched.
        status, listing = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    def test_fresh_server_has_no_snapshot_and_no_events(self) -> None:
        self.seed_and_capture()
        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            fresh_base = f"http://127.0.0.1:{fresh.server_port}"
            payload = json.dumps(
                {"token": "tok-f", "organizationId": ORG1, "role": "write"}
            ).encode()
            request = Request(
                f"{fresh_base}/auth/tokens",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 201)
            request = Request(
                f"{fresh_base}/snapshots/s1/events/replay/decisions"
                f"?{self.query()}",
                headers={"Authorization": "Bearer tok-f"},
            )
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=2)
            self.assertEqual(caught.exception.code, 404)
            self.assertEqual(
                json.loads(caught.exception.read()),
                {"error": "snapshot_not_found"},
            )
            caught.exception.close()
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # ----------------------------------------------------------------- 401 / 403 / 404

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.seed_and_capture()
        path = f"/snapshots/s1/events/replay/decisions?{self.query()}"
        for headers in (
            {},
            {"Authorization": "Bearer"},
            {"Authorization": "Bearer "},
            {"Authorization": "Basic w1"},
            {"Authorization": "Bearer ghost-token"},
        ):
            with self.subTest(headers=headers):
                request = Request(
                    f"{self.base_url}{path}", headers=headers, method="GET"
                )
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request, timeout=5)
                error = caught.exception
                raw = error.read()
                self.assertEqual(error.code, 401)
                self.assertEqual(json.loads(raw), {"error": "unauthorized"})
                error.close()

    def test_credential_is_checked_before_query_shape(self) -> None:
        request = Request(
            f"{self.base_url}/snapshots/s1/events/replay/decisions"
            "?windowSize=not-a-number",
            headers={"Authorization": "Bearer ghost-token"},
            method="GET",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 401)
        caught.exception.close()

    def test_query_shape_is_checked_before_organization(self) -> None:
        # The credential passes and the query shape fails before the
        # organization is ever compared, so a foreign credential with a
        # malformed query gets 422, not 403.
        status, body = self.decisions(
            "organizationId=&windowSize=x", token="w2"
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_organization_is_checked_before_snapshot_name(self) -> None:
        # A foreign organization gets 403 even when the snapshot name has
        # never existed.
        status, body = self.decisions(
            self.query(org=ORG1), snapshot="ghost", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_foreign_organization_is_403_and_leaves_no_trace(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(self.query(), token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        status, listing = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.seed_and_capture()
        # org-2's credential names org-1's snapshot: forbidden, never 404.
        status, body = self.decisions(
            self.query(org=ORG2), snapshot="s1", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "snapshot_not_found"})
        # The failed read did not implicitly create the snapshot.
        status, again = self.decisions(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(again, {"error": "snapshot_not_found"})
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(
            [entry["snapshotId"] for entry in snapshots["snapshots"]], ["s1"]
        )

    # ---------------------------------------------------------------------- 422

    def test_each_required_parameter_must_appear_exactly_once(self) -> None:
        self.seed_and_capture()
        base = self.query()
        for query in (
            "",
            "type=t&windowSize=60&threshold=3",
            "organizationId=org-1&windowSize=60&threshold=3",
            "organizationId=org-1&type=t&threshold=3",
            "organizationId=org-1&type=t&windowSize=60",
            f"{base}&organizationId=org-1",
            f"{base}&type={EVENT_TYPE}",
            f"{base}&windowSize=60",
            f"{base}&threshold=3",
        ):
            with self.subTest(query=query):
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_values_are_422(self) -> None:
        self.seed_and_capture()
        for query in (
            f"organizationId=&type={EVENT_TYPE}&windowSize=60&threshold=3",
            f"organizationId=%20&type={EVENT_TYPE}&windowSize=60&threshold=3",
            "organizationId=org-1&type=&windowSize=60&threshold=3",
            "organizationId=org-1&type=%20&windowSize=60&threshold=3",
            "organizationId=org-1&type=t&windowSize=&threshold=3",
            "organizationId=org-1&type=t&windowSize=%20&threshold=3",
            "organizationId=org-1&type=t&windowSize=60&threshold=",
            "organizationId=org-1&type=t&windowSize=60&threshold=%20",
        ):
            with self.subTest(query=query):
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_window_size_and_threshold_must_be_positive_integers(self) -> None:
        self.seed_and_capture()
        for bad in ("0", "-1", "1.5", "abc", "1e3", "+5", "true"):
            for name, other in (
                ("windowSize", "threshold=3"),
                ("threshold", "windowSize=60"),
            ):
                query = (
                    f"organizationId=org-1&type={EVENT_TYPE}"
                    f"&{name}={bad}&{other}"
                )
                with self.subTest(name=name, bad=bad):
                    status, body = self.decisions(query)
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_range_must_be_paired_non_negative_and_ordered(self) -> None:
        self.seed_and_capture()
        base = self.query()
        for query in (
            f"{base}&from=0",
            f"{base}&to=10",
            f"{base}&from=0&to=10&from=5",
            f"{base}&from=-1&to=10",
            f"{base}&from=0&to=-10",
            f"{base}&from=1.5&to=10",
            f"{base}&from=abc&to=10",
            f"{base}&from=10&to=9",
        ):
            with self.subTest(query=query):
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_from_equal_to_to_is_legal(self) -> None:
        self.add_event("evt-a", occurred_at=30)
        self.capture()
        status, body = self.decisions(self.query(from_to=(30, 30)))
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(
            body["steps"][0]["windows"],
            [{"start": 0, "end": 60, "count": 1}],
        )

    def test_validation_failure_writes_nothing(self) -> None:
        self.seed_and_capture()
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=x&threshold=3"
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")
        status, listing = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(
            [entry["snapshotId"] for entry in snapshots["snapshots"]], ["s1"]
        )


if __name__ == "__main__":
    unittest.main()
