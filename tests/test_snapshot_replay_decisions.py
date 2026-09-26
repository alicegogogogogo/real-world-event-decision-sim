"""Regression tests for GET /snapshots/{snapshotId}/events/replay/decisions.

The baseline already had the main-service step replay
(``GET /events/replay/decisions``) plus the single-snapshot aggregate and
peak decision, but no snapshot-dimension step-by-step recomputation; this
locks down the new read-only entry point that lowers the exact replay,
window, and peak contract of the main replay onto the events one snapshot
captured:

- only the snapshot's captured events of the caller's organization and the
  requested ``type`` are replayed; other organizations' data, other types,
  and events committed after capture never enter the result;
- matching events arrive ordered by ``occurredAt`` then ``eventId`` and
  accumulate one at a time; each step reports the event id and time, the
  window-count rows for the accumulated prefix, and the peak decision, all
  computed through the same helpers as the main replay;
- without ``from``/``to`` only windows hit by accumulated events appear;
  with a range out-of-range events still arrive as steps while every
  intersecting window (empty ones included) is retained at every step;
- the response echoes organization, snapshot, type, window width,
  threshold, and the (possibly null) range, is compact key-sorted JSON with
  integer values and one trailing newline, and identical requests are
  byte-for-byte identical without polluting each other;
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

_UNSET = object()


def make_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "eventId": "evt-1",
        "organizationId": ORG1,
        "type": EVENT_TYPE,
        "occurredAt": 100,
        "payload": {"severity": "low"},
    }
    event.update(overrides)
    return event


def query(
    *,
    org: str = ORG1,
    event_type: str = EVENT_TYPE,
    window_size: Any = 60,
    threshold: Any = 3,
    from_to: tuple[int, int] | None = None,
) -> str:
    text = (
        f"organizationId={org}&type={event_type}"
        f"&windowSize={window_size}&threshold={threshold}"
    )
    if from_to is not None:
        text += f"&from={from_to[0]}&to={from_to[1]}"
    return text


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
        token: str | None = "w1",
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
        self, path: str, *, token: str | None = "w1"
    ) -> tuple[int, Any]:
        status, raw = self.raw(path, token=token)
        return status, json.loads(raw)

    def register(self, token: str, organization_id: str, role: str) -> None:
        body = json.dumps(
            {
                "token": token,
                "organizationId": organization_id,
                "role": role,
            }
        ).encode()
        status, _ = self.raw(
            "/auth/tokens", method="POST", body=body, token=None
        )
        self.assertIn(status, (200, 201))

    def post_event(self, event: dict[str, Any], *, token: str | None = None) -> None:
        if token is None:
            token = "w1" if event["organizationId"] == ORG1 else "w2"
        status, _ = self.raw(
            "/events",
            method="POST",
            body=json.dumps(event).encode(),
            token=token,
        )
        self.assertEqual(status, 201)

    def capture(self, snapshot_id: str = "s1", *, token: str = "w1") -> None:
        status, _ = self.raw(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": snapshot_id}).encode(),
            token=token,
        )
        self.assertEqual(status, 201)

    def replay(
        self,
        query_text: str | None = None,
        *,
        snapshot: str = "s1",
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        if query_text is None:
            query_text = query()
        return self.call(
            f"/snapshots/{snapshot}/events/replay/decisions?{query_text}",
            token=token,
        )

    def replay_raw(
        self, query_text: str | None = None, *, snapshot: str = "s1"
    ) -> tuple[int, bytes]:
        if query_text is None:
            query_text = query()
        return self.raw(
            f"/snapshots/{snapshot}/events/replay/decisions?{query_text}"
        )

    def seed_events(self) -> None:
        events = [
            make_event(eventId="evt-b", occurredAt=200),
            make_event(eventId="evt-a", occurredAt=200),
            make_event(eventId="evt-c", occurredAt=100),
            make_event(eventId="evt-d", occurredAt=0),
            make_event(eventId="evt-e", occurredAt=300),
            # A different type never opens a replay step for this query.
            make_event(eventId="evt-z", type="other.kind", occurredAt=100),
            # Another organization's events never enter the replay.
            make_event(eventId="evt-x", organizationId=ORG2, occurredAt=10),
        ]
        for event in events:
            self.post_event(event)

    # ------------------------------------------------------------- happy paths

    def test_steps_follow_replay_order_with_event_id_and_time(self) -> None:
        self.seed_events()
        self.capture()
        status, body = self.replay()
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
        self.seed_events()
        self.capture()
        status, body = self.replay(query(threshold=3))
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
            rows = step["windows"]
            self.assertEqual(
                [(row["start"], row["end"], row["count"]) for row in rows],
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
        self.seed_events()
        self.capture()
        status, body = self.replay(query(threshold=2))
        self.assertEqual(status, 200)
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
            self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))
        self.capture()
        status, body = self.replay(query(threshold=5))
        self.assertEqual(status, 200)
        for step in body["steps"][1:]:
            self.assertEqual(step["peakCount"], 1)
            self.assertEqual(step["peakStart"], 0)
            self.assertEqual(step["action"], "observe")

    def test_no_matching_type_has_no_steps(self) -> None:
        self.seed_events()
        self.capture()
        status, body = self.replay(query(event_type="never.seen"))
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

    def test_empty_snapshot_has_no_steps(self) -> None:
        self.capture()
        status, body = self.replay()
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])
        self.assertEqual(body["snapshotId"], "s1")

    # -------------------------------------------------------- capture boundary

    def test_post_capture_writes_never_enter_the_steps(self) -> None:
        self.post_event(make_event(eventId="evt-a", occurredAt=10))
        self.capture()
        self.post_event(make_event(eventId="evt-b", occurredAt=10))
        self.post_event(make_event(eventId="evt-c", occurredAt=10))
        status, body = self.replay(query(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["evt-a"])
        self.assertEqual(body["steps"][0]["peakCount"], 1)
        self.assertEqual(body["steps"][0]["action"], "observe")

    def test_other_organizations_never_enter_and_foreign_snapshot_replays(
        self,
    ) -> None:
        self.post_event(make_event(eventId="evt-a", occurredAt=10))
        self.capture("s1")
        for index in range(3):
            self.post_event(
                make_event(eventId=f"evt-b{index}", organizationId=ORG2),
                token="w2",
            )
        self.capture("s2", token="w2")

        status, body = self.replay(query(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(body["steps"][0]["peakCount"], 1)

        status, body = self.replay(
            query(org=ORG2, threshold=2), snapshot="s2", token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["snapshotId"], "s2")
        self.assertEqual(len(body["steps"]), 3)
        self.assertEqual(body["steps"][-1]["peakCount"], 3)
        self.assertEqual(body["steps"][-1]["action"], "escalate")

    # ------------------------------------------------------------- range semantics

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.seed_events()
        self.capture()
        status, body = self.replay(query(threshold=3, from_to=(0, 180)))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 180)
        grid = [(0, 60), (60, 120), (120, 180), (180, 240)]
        for step in body["steps"]:
            self.assertEqual(
                [(row["start"], row["end"]) for row in step["windows"]], grid
            )
        # Events at 200 and 300 are outside [0, 180], so the 180 window
        # never gains a count, but those events still arrive as steps.
        self.assertEqual(
            [row["count"] for row in body["steps"][-1]["windows"]],
            [1, 1, 0, 0],
        )
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["evt-d", "evt-c", "evt-a", "evt-b", "evt-e"],
        )

    def test_range_step_with_no_counted_window_is_zero_null_observe(self) -> None:
        self.post_event(make_event(eventId="evt-late", occurredAt=100))
        self.capture()
        status, body = self.replay(query(from_to=(0, 0)))
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        step = body["steps"][0]
        # The event at 100 falls outside [0, 0]; the only intersecting
        # window is [0, 60) and it stays empty.
        self.assertEqual(
            step["windows"], [{"start": 0, "end": 60, "count": 0}]
        )
        self.assertEqual(step["peakCount"], 0)
        self.assertIsNone(step["peakStart"])
        self.assertEqual(step["action"], "observe")

    def test_range_boundaries_are_inclusive(self) -> None:
        for event_id, occurred_at in (
            ("evt-a", 59),
            ("evt-b", 60),
            ("evt-c", 119),
            ("evt-d", 120),
        ):
            self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))
        self.capture()
        status, body = self.replay(
            query(threshold=2, from_to=(60, 120))
        )
        self.assertEqual(status, 200)
        last = body["steps"][-1]
        # The closed interval includes both 60 and 120; the event at 59
        # never counts.
        self.assertEqual(
            [(row["start"], row["count"]) for row in last["windows"]],
            [(60, 2), (120, 1)],
        )
        self.assertEqual(last["peakStart"], 60)
        self.assertEqual(last["peakCount"], 2)
        self.assertEqual(last["action"], "escalate")

    # --------------------------------------------- consistency with snapshot views

    def test_final_step_matches_snapshot_aggregate_and_decision(self) -> None:
        self.seed_events()
        self.capture()
        for from_to in (None, (0, 180), (30, 130)):
            with self.subTest(from_to=from_to):
                status, body = self.replay(
                    query(threshold=2, from_to=from_to)
                )
                self.assertEqual(status, 200)
                last = body["steps"][-1]

                aggregate_query = (
                    f"organizationId={ORG1}&type={EVENT_TYPE}&windowSize=60"
                )
                if from_to is not None:
                    aggregate_query += f"&from={from_to[0]}&to={from_to[1]}"
                status, aggregate = self.call(
                    f"/snapshots/s1/events/aggregate?{aggregate_query}"
                )
                self.assertEqual(status, 200)
                self.assertEqual(last["windows"], aggregate["windows"])

                decision_payload: dict[str, Any] = {
                    "organizationId": ORG1,
                    "type": EVENT_TYPE,
                    "windowSize": 60,
                    "threshold": 2,
                }
                if from_to is not None:
                    decision_payload["from"] = from_to[0]
                    decision_payload["to"] = from_to[1]
                status, decision_bytes = self.raw(
                    "/snapshots/s1/decisions/evaluate",
                    method="POST",
                    body=json.dumps(decision_payload).encode(),
                )
                self.assertEqual(status, 200)
                decision = json.loads(decision_bytes)
                self.assertEqual(last["peakStart"], decision["peakStart"])
                self.assertEqual(last["peakCount"], decision["peakCount"])
                self.assertEqual(last["action"], decision["action"])

    # --------------------------------------------------------- serialization / roles

    def test_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_events()
        self.capture()
        status, raw = self.replay_raw()
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        body = json.loads(raw)
        self.assertEqual(
            list(body),
            [
                "from",
                "organizationId",
                "snapshotId",
                "steps",
                "threshold",
                "to",
                "type",
                "windowSize",
            ],
        )
        self.assertEqual(
            list(body["steps"][0]),
            [
                "action",
                "eventId",
                "occurredAt",
                "peakCount",
                "peakStart",
                "windows",
            ],
        )
        self.assertEqual(
            list(body["steps"][0]["windows"][0]), ["count", "end", "start"]
        )
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )
        self.assertIsInstance(body["windowSize"], int)
        self.assertIsInstance(body["threshold"], int)
        self.assertIsInstance(body["steps"][0]["occurredAt"], int)

    def test_repeated_requests_are_byte_identical_and_isolated(self) -> None:
        self.seed_events()
        self.capture()
        query_text = query(threshold=2)
        first = self.replay_raw(query_text)[1]
        # Main-service writes after capture and a ranged read between
        # replays must not perturb any later bytes.
        self.post_event(make_event(eventId="evt-after-1", occurredAt=10))
        rest = [self.replay_raw(query_text)[1] for _ in range(3)]
        self.assertTrue(all(chunk == first for chunk in rest))
        ranged = self.replay_raw(query(threshold=2, from_to=(0, 180)))[1]
        self.assertNotEqual(ranged, first)
        self.assertEqual(self.replay_raw(query_text)[1], first)
        self.post_event(make_event(eventId="evt-after-2", occurredAt=20))

    def test_read_and_write_credentials_may_both_query(self) -> None:
        self.seed_events()
        self.capture()
        read_status, read_body = self.replay(token="r1")
        write_status, write_body = self.replay(token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    def test_query_is_read_only(self) -> None:
        self.seed_events()
        self.capture()
        self.replay(query(threshold=2, from_to=(0, 180)))
        self.replay(query(threshold=1))
        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            self.assertEqual(self.replay()[0], 200)
        status, after = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)
        status, events = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(events["events"]), 6)
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    def test_fresh_server_starts_with_no_snapshot_and_no_steps(self) -> None:
        self.post_event(make_event())
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
                f"{base_url}/snapshots/s1/events/replay/decisions?{query()}",
                headers={"Authorization": "Bearer tok-fresh"},
                method="GET",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    status = response.status
                    body = json.loads(response.read())
            except HTTPError as error:
                status = error.code
                body = json.loads(error.read())
                error.close()
            # Restart cleared both the events and the snapshots: the name
            # has never existed on this instance.
            self.assertEqual(status, 404)
            self.assertEqual(body, {"error": "snapshot_not_found"})
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # ----------------------------------------------------------------- 401

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture()
        valid_query = query()
        for headers in (
            {},
            {"Authorization": "Bearer"},
            {"Authorization": "Bearer "},
            {"Authorization": "Basic w1"},
            {"Authorization": "Bearer ghost-token"},
        ):
            with self.subTest(headers=headers):
                request = Request(
                    f"{self.base_url}/snapshots/s1/events/replay/decisions"
                    f"?{valid_query}",
                    headers=headers,
                    method="GET",
                )
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request, timeout=5)
                error = caught.exception
                self.assertEqual(error.code, 401)
                self.assertEqual(
                    json.loads(error.read()), {"error": "unauthorized"}
                )
                error.close()

    def test_credential_is_checked_before_query_shape(self) -> None:
        self.capture()
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

    # ----------------------------------------------------------------- 403 / 404

    def test_foreign_organization_is_403_before_snapshot_lookup(self) -> None:
        self.capture()
        status, body = self.replay(token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        # The organization decision precedes even the snapshot lookup: an
        # unknown snapshot name is still 403 for a foreign organization.
        status, body = self.replay(
            query(org=ORG1), snapshot="ghost", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        status, body = self.replay(snapshot="s2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.replay(snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "snapshot_not_found"})
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(snapshots["snapshots"], [])

    # ---------------------------------------------------------------------- 422

    def test_each_required_parameter_must_appear_exactly_once(self) -> None:
        self.capture()
        base = query()
        for bad_query in (
            "",
            f"type={EVENT_TYPE}&windowSize=60&threshold=3",
            "organizationId=org-1&windowSize=60&threshold=3",
            f"organizationId=org-1&type={EVENT_TYPE}&threshold=3",
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60",
            f"{base}&organizationId=org-1",
            f"{base}&type={EVENT_TYPE}",
            f"{base}&windowSize=60",
            f"{base}&threshold=3",
        ):
            with self.subTest(bad_query=bad_query):
                status, body = self.replay(bad_query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_values_are_422(self) -> None:
        self.capture()
        for bad_query in (
            f"organizationId=&type={EVENT_TYPE}&windowSize=60&threshold=3",
            f"organizationId=%20&type={EVENT_TYPE}&windowSize=60&threshold=3",
            "organizationId=org-1&type=&windowSize=60&threshold=3",
            "organizationId=org-1&type=%20&windowSize=60&threshold=3",
            "organizationId=org-1&type=t&windowSize=&threshold=3",
            "organizationId=org-1&type=t&windowSize=%20&threshold=3",
            "organizationId=org-1&type=t&windowSize=60&threshold=",
            "organizationId=org-1&type=t&windowSize=60&threshold=%20",
        ):
            with self.subTest(bad_query=bad_query):
                status, body = self.replay(bad_query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_window_size_and_threshold_must_be_positive_integers(self) -> None:
        self.capture()
        for bad in ("0", "-1", "1.5", "abc", "1e3", "+5", "true"):
            for name, other in (
                ("windowSize", "threshold=3"),
                ("threshold", "windowSize=60"),
            ):
                bad_query = (
                    f"organizationId=org-1&type={EVENT_TYPE}"
                    f"&{name}={bad}&{other}"
                )
                with self.subTest(name=name, bad=bad):
                    status, body = self.replay(bad_query)
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_leading_zero_integer_text_is_accepted(self) -> None:
        self.seed_events()
        self.capture()
        status, body = self.replay(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=060&threshold=003"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 3)

    def test_range_must_be_paired_non_negative_and_ordered(self) -> None:
        self.capture()
        base = query()
        for bad_query in (
            f"{base}&from=0",
            f"{base}&to=10",
            f"{base}&from=0&to=10&from=5",
            f"{base}&from=-1&to=10",
            f"{base}&from=0&to=-10",
            f"{base}&from=1.5&to=10",
            f"{base}&from=abc&to=10",
            f"{base}&from=10&to=9",
        ):
            with self.subTest(bad_query=bad_query):
                status, body = self.replay(bad_query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_from_equal_to_to_is_legal(self) -> None:
        self.post_event(make_event(eventId="evt-a", occurredAt=30))
        self.capture()
        status, body = self.replay(query(from_to=(30, 30)))
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(
            body["steps"][0]["windows"],
            [{"start": 0, "end": 60, "count": 1}],
        )

    def test_validation_failure_writes_nothing(self) -> None:
        self.seed_events()
        self.capture()
        status, _ = self.replay(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=x&threshold=3"
        )
        self.assertEqual(status, 422)
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(len(snapshots["snapshots"]), 1)
        status, events = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(events["events"]), 6)


if __name__ == "__main__":
    unittest.main()
