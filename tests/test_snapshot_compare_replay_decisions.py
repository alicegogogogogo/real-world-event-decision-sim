"""Regression tests for POST /snapshots/compare/replay/decisions.

The baseline already had the single-snapshot step-by-step replay
(``GET /snapshots/{snapshotId}/events/replay/decisions``) and the one-shot
snapshot window comparison (``POST /snapshots/compare``), but no way to
align two snapshots' replays. This locks down the new read-only entry
point:

- each side replays only that snapshot's captured events of the caller's
  organization and the requested type, ordered by occurredAt then
  eventId; events committed after a capture never enter it;
- the two replays align by eventId: a shared identifier is one shared
  step, an identifier on only one side is its own step on which the other
  side stays put and counts zero, and the aligned steps run by occurred
  time then identifier;
- every step carries the event id/time, window rows aligned by start
  (ranged empty windows kept), and the two sides' peak decisions with an
  equality marker; thresholds decide escalate/observe per side;
- using the same snapshot on both sides is legal and equal by
  construction, and neither side holding a matching event yields no
  steps;
- the response is compact key-sorted JSON with integer/boolean values
  and one trailing newline; identical submissions are byte-for-byte
  identical;
- 401 / 415 / 400 / 422 / 403 / 404 follow the fixed ordering, left is
  resolved before right, and nothing is ever written or implicitly
  created.
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
PATH = "/snapshots/compare/replay/decisions"


def event_body(
    event_id: str,
    occurred_at: int,
    organization_id: str = ORG1,
    event_type: str = EVENT_TYPE,
) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": event_type,
        "occurredAt": occurred_at,
        "payload": {},
    }


class SnapshotCompareReplayDecisionsTest(unittest.TestCase):
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
        raw_body: bytes | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, Any]:
        body = (
            raw_body
            if raw_body is not None
            else json.dumps(payload).encode() if payload is not None
            else None
        )
        status, raw = self.raw(
            path,
            method=method,
            body=body,
            token=token,
            content_type=content_type,
        )
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

    def add_main_event(
        self,
        event_id: str,
        occurred_at: int,
        token: str = "w1",
        event_type: str = EVENT_TYPE,
    ) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        status, _ = self.call(
            "/events",
            method="POST",
            token=token,
            payload=event_body(event_id, occurred_at, organization_id, event_type),
        )
        self.assertEqual(status, 201)

    def capture(self, snapshot_id: str, token: str = "w1") -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def compare_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": ORG1,
            "left": "snap-left",
            "right": "snap-right",
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 3,
        }
        payload.update(overrides)
        return payload

    def compare(
        self, payload: dict[str, Any] | None = None, *, token: str | None = "w1"
    ) -> tuple[int, Any]:
        return self.call(
            PATH,
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    @staticmethod
    def peak(start: Any, count: int, action: str) -> dict[str, Any]:
        return {"action": action, "peakCount": count, "peakStart": start}

    @staticmethod
    def row(
        start: int, left_count: int, right_count: int, equal: bool
    ) -> dict[str, Any]:
        return {
            "start": start,
            "leftCount": left_count,
            "rightCount": right_count,
            "equal": equal,
        }

    # ------------------------------------------------------------- happy paths

    def test_shared_events_align_into_one_step_each(self) -> None:
        self.add_main_event("e1", 5)
        self.add_main_event("e2", 65)
        self.capture("snap-left")
        self.capture("snap-right")

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["e1", "e2"])
        self.assertEqual(
            [step["occurredAt"] for step in body["steps"]], [5, 65]
        )
        first, second = body["steps"]
        self.assertEqual(first["windows"], [self.row(0, 1, 1, True)])
        self.assertEqual(
            first["decision"],
            {
                "left": self.peak(0, 1, "observe"),
                "right": self.peak(0, 1, "observe"),
                "equal": True,
            },
        )
        self.assertEqual(
            second["windows"],
            [self.row(0, 1, 1, True), self.row(60, 1, 1, True)],
        )
        self.assertTrue(second["decision"]["equal"])

    def test_one_sided_identifier_is_its_own_step_with_other_side_frozen(
        self,
    ) -> None:
        # Snapshots are cumulative captures: snap-left holds the earlier two
        # events; snap-right is taken later and also holds only-right, which
        # exists on no earlier snapshot, so it is right-only in the alignment.
        self.add_main_event("shared", 10)
        self.add_main_event("only-left", 20)
        self.capture("snap-left")
        self.add_main_event("only-right", 70)
        self.capture("snap-right")

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["shared", "only-left", "only-right"],
        )

        shared, only_left, only_right = body["steps"]
        # The shared step extends both prefixes together.
        self.assertEqual(shared["windows"], [self.row(0, 1, 1, True)])
        # only-left is captured by both snapshots, so it stays shared; the
        # first right-only step arrives last and freezes the left prefix at
        # two events while the right prefix grows into the next window.
        self.assertEqual(only_left["windows"], [self.row(0, 2, 2, True)])
        self.assertEqual(
            only_left["decision"]["left"], self.peak(0, 2, "observe")
        )
        self.assertEqual(
            only_left["decision"]["right"], self.peak(0, 2, "observe")
        )
        self.assertTrue(only_left["decision"]["equal"])
        # The right-only step leaves the left prefix at two events while the
        # right prefix holds three, one of them in the [60, 120) window.
        self.assertEqual(
            only_right["windows"],
            [self.row(0, 2, 2, True), self.row(60, 0, 1, False)],
        )
        self.assertEqual(
            only_right["decision"]["left"], self.peak(0, 2, "observe")
        )
        self.assertEqual(
            only_right["decision"]["right"], self.peak(0, 2, "observe")
        )

    def test_steps_ordered_by_time_then_identifier(self) -> None:
        # Two captures split the events: snap-left holds evt-b (100) and
        # evt-a (100, committed first); snap-right is taken after evt-c at 50
        # is added too. Union ordering ties on time and breaks by id.
        self.add_main_event("evt-b", 100)
        self.add_main_event("evt-a", 100)
        self.capture("snap-left")
        self.add_main_event("evt-c", 50)
        self.capture("snap-right")

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("evt-c", 50), ("evt-a", 100), ("evt-b", 100)],
        )

    def test_shared_identifier_disagreement_on_time_reports_earlier(self) -> None:
        # Two captures cannot themselves move an event id to another time on
        # the main ledger, so exercise the pure merge function directly for
        # the same-id-different-time data shape.
        from event_sim.server import compare_snapshot_replay_decisions

        def ev(event_id: str, occurred_at: int) -> dict[str, Any]:
            return {
                "eventId": event_id,
                "organizationId": ORG1,
                "type": EVENT_TYPE,
                "occurredAt": occurred_at,
                "payload": {},
            }

        params = self.compare_payload(left="L", right="R")
        params["from"] = None
        params["to"] = None
        result = compare_snapshot_replay_decisions(
            [ev("dup", 100), ev("late-l", 150)],
            [ev("dup", 40), ev("early-r", 10)],
            params,
        )
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in result["steps"]],
            [("early-r", 10), ("dup", 40), ("late-l", 150)],
        )
        dup = result["steps"][1]
        # The right prefix now holds both its earlier event and its own dup
        # in window 0; the left prefix holds only its dup in window 60.
        self.assertEqual(
            dup["windows"],
            [
                self.row(0, 0, 2, False),   # right's early-r and dup at 40
                self.row(60, 1, 0, False),  # left's dup at 100
            ],
        )

    def test_window_rows_align_union_of_hit_windows_with_missing_side_zero(
        self,
    ) -> None:
        self.add_main_event("l1", 5)
        self.add_main_event("l2", 10)
        self.capture("snap-left")
        self.add_main_event("r1", 125)
        self.capture("snap-right")

        status, body = self.compare()
        self.assertEqual(status, 200)
        last = body["steps"][-1]
        # The earlier two events are in both captures; r1 joined after
        # snap-left and lands in the [120,180) window on the right only.
        self.assertEqual(
            last["windows"],
            [
                self.row(0, 2, 2, True),
                self.row(120, 0, 1, False),
            ],
        )

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.add_main_event("l1", 5)
        self.add_main_event("l2", 65)  # outside [0, 60]
        self.capture("snap-left")
        self.add_main_event("r1", 120)  # outside [0, 60]
        self.capture("snap-right")

        payload = self.compare_payload(**{"from": 0, "to": 60})
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 60)
        # Out-of-range events still arrive as steps in time order; every
        # window intersecting [0, 60] is present at every step, including the
        # empty [60, 120) intersection window.
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["l1", "l2", "r1"],
        )
        for step in body["steps"]:
            self.assertEqual(
                [row["start"] for row in step["windows"]], [0, 60]
            )
        first, second, third = body["steps"]
        self.assertEqual(
            first["windows"],
            [self.row(0, 1, 1, True), self.row(60, 0, 0, True)],
        )
        # l2 at 65 is outside the range and counts on neither window.
        self.assertEqual(
            second["windows"],
            [self.row(0, 1, 1, True), self.row(60, 0, 0, True)],
        )
        # r1 at 120 is outside [0, 60] as well; it still arrives as a
        # right-only step but contributes zero counts.
        self.assertEqual(
            third["windows"],
            [self.row(0, 1, 1, True), self.row(60, 0, 0, True)],
        )

    def test_threshold_escalate_is_decided_per_side_per_step(self) -> None:
        # snap-left captures the single shared event; snap-right is taken
        # after a second event in the same window joins the ledger. The
        # right-only second step pushes only the right peak to the
        # threshold while the left prefix stays frozen at one.
        self.add_main_event("shared", 5)
        self.capture("snap-left")
        self.add_main_event("r2", 10)
        self.capture("snap-right")

        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["shared", "r2"]
        )
        # The shared first step leaves both sides at one.
        self.assertEqual(body["steps"][0]["decision"]["left"]["action"], "observe")
        # The right-only second step pushes the right peak to the threshold.
        self.assertEqual(
            body["steps"][1]["decision"]["right"]["action"], "escalate"
        )
        self.assertEqual(body["steps"][1]["decision"]["right"]["peakCount"], 2)
        self.assertEqual(
            body["steps"][1]["decision"]["left"]["action"], "observe"
        )
        self.assertFalse(body["steps"][1]["decision"]["equal"])

    def test_peak_tie_resolves_to_earliest_start(self) -> None:
        self.add_main_event("a", 5)
        self.add_main_event("b", 65)
        self.capture("snap-solo")
        status, body = self.compare(
            self.compare_payload(left="snap-solo", right="snap-solo")
        )
        self.assertEqual(status, 200)
        final = body["steps"][-1]
        self.assertEqual(final["decision"]["left"]["peakStart"], 0)
        self.assertEqual(final["decision"]["left"]["peakCount"], 1)

    def test_same_snapshot_on_both_sides_is_equal_by_construction(self) -> None:
        self.add_main_event("a", 5)
        self.add_main_event("b", 70)
        self.capture("snap-solo")
        status, body = self.compare(
            self.compare_payload(left="snap-solo", right="snap-solo")
        )
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["a", "b"])
        for step in body["steps"]:
            self.assertTrue(step["decision"]["equal"])
            self.assertEqual(
                step["decision"]["left"], step["decision"]["right"]
            )
            self.assertTrue(all(row["equal"] for row in step["windows"]))

    def test_other_types_and_later_main_writes_never_enter_the_replay(
        self,
    ) -> None:
        self.add_main_event("other-1", 5, event_type="other.kind")
        self.capture("snap-left")
        self.add_main_event("other-2", 6, event_type="other.kind")
        self.capture("snap-right")
        # Events committed after both captures (either type) never enter.
        self.add_main_event("other-3", 7, event_type="other.kind")
        self.add_main_event("main-late", 7)

        # The requested type matches no captured event; the post-capture
        # main write and other-type events are all excluded.
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])
        status, body = self.compare(self.compare_payload(type="never.seen"))
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])
        # The other type's captured events do replay when it is requested:
        # other-1 is shared (in both captures) and other-2 is right-only,
        # freezing the left prefix at one; other-3 (committed after
        # snap-right) never enters.
        status, body = self.compare(self.compare_payload(type="other.kind"))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["other-1", "other-2"]
        )
        self.assertEqual(
            body["steps"][0]["windows"], [self.row(0, 1, 1, True)]
        )
        self.assertEqual(
            body["steps"][1]["windows"], [self.row(0, 1, 2, False)]
        )

    def test_neither_side_holding_matching_events_yields_no_steps(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])

    def test_response_echoes_all_request_parameters(self) -> None:
        self.add_main_event("a", 5)
        self.capture("snap-left")
        self.capture("snap-right")
        payload = self.compare_payload(
            threshold=4, type=EVENT_TYPE, **{"from": 0, "to": 120}
        )
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["left"], "snap-left")
        self.assertEqual(body["right"], "snap-right")
        self.assertEqual(body["type"], EVENT_TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 4)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 120)

    # ----------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_boolean_integer_and_newline_terminated(
        self,
    ) -> None:
        # snap-a is the earlier capture; naming the later snap-b as ``left``
        # makes its later events left-only, so the union rows carry a
        # right-zero side as well as false equality markers.
        self.add_main_event("l1", 5)
        self.capture("snap-a")
        self.add_main_event("r1", 5)
        self.add_main_event("r2", 65)
        self.capture("snap-b")

        raw_body = json.dumps(
            self.compare_payload(left="snap-b", right="snap-a")
        ).encode()
        status, raw = self.raw(PATH, method="POST", body=raw_body, token="w1")
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        text = raw.decode()
        self.assertIn('"leftCount":1', text)
        self.assertIn('"rightCount":0', text)
        self.assertIn('"equal":false', text)

        body = json.loads(raw)
        self.assertEqual(
            list(body),
            [
                "from",
                "left",
                "organizationId",
                "right",
                "steps",
                "threshold",
                "to",
                "type",
                "windowSize",
            ],
        )
        self.assertEqual(
            list(body["steps"][0]),
            ["decision", "eventId", "occurredAt", "windows"],
        )
        self.assertEqual(
            list(body["steps"][0]["decision"]), ["equal", "left", "right"]
        )
        self.assertEqual(
            list(body["steps"][0]["decision"]["left"]),
            ["action", "peakCount", "peakStart"],
        )
        self.assertEqual(
            list(body["steps"][0]["windows"][0]),
            ["equal", "leftCount", "rightCount", "start"],
        )

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.add_main_event("l1", 5)
        self.add_main_event("l2", 65)
        self.capture("snap-left")
        self.add_main_event("r1", 120)
        self.capture("snap-right")
        raw_body = json.dumps(
            self.compare_payload(**{"from": 0, "to": 180})
        ).encode()
        raws = [
            self.raw(PATH, method="POST", body=raw_body, token="w1")[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # -------------------------------------------------------------- read-only

    def test_comparison_is_read_only(self) -> None:
        self.add_main_event("l1", 5)
        self.capture("snap-left")
        self.add_main_event("r1", 10)
        self.capture("snap-right")

        def summaries() -> Any:
            return self.call("/snapshots")[1]

        before = summaries()
        for _ in range(3):
            status, _ = self.compare(self.compare_payload(**{"from": 0, "to": 60}))
            self.assertEqual(status, 200)
        self.assertEqual(before, summaries())

        # The main-service ledger, reservations, and alerts are untouched.
        status, events = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            sorted(e["eventId"] for e in events["events"]), ["l1", "r1"]
        )
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    def test_failed_comparison_creates_no_snapshot(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        for payload in (
            self.compare_payload(left="ghost"),
            self.compare_payload(right="ghost"),
            self.compare_payload(**{"windowSize": 0}),
        ):
            status, _ = self.compare(payload)
            self.assertIn(status, (404, 422))
        snapshot_ids = [
            entry["snapshotId"] for entry in self.call("/snapshots")[1]["snapshots"]
        ]
        self.assertEqual(sorted(snapshot_ids), ["snap-left", "snap-right"])

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_compare(self) -> None:
        self.add_main_event("l1", 5)
        self.capture("snap-left")
        self.capture("snap-right")
        status, body = self.compare(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["l1"])

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        raw_body = json.dumps(self.compare_payload()).encode()
        for token, headers_extra in (
            (None, {}),
            ("forged", {}),
            ("w1", {"Authorization": "Bearer"}),
            ("w1", {"Authorization": "Basic abc"}),
            ("w1", {"Authorization": "Bearerer w1"}),
        ):
            with self.subTest(token=token, headers_extra=headers_extra):
                headers = {"Content-Type": "application/json"}
                if token is not None:
                    headers["Authorization"] = f"Bearer {token}"
                headers.update(headers_extra)
                request = Request(
                    f"{self.base_url}{PATH}",
                    data=raw_body,
                    headers=headers,
                    method="POST",
                )
                try:
                    with urlopen(request, timeout=5) as response:
                        status = response.status
                        raw = response.read()
                except HTTPError as error:
                    status = error.code
                    raw = error.read()
                    error.close()
                self.assertEqual(status, 401)
                self.assertEqual(json.loads(raw)["error"], "unauthorized")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        self.capture("foreign", token="w2")

        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the snapshot names do not exist anywhere.
        status, body = self.compare(
            self.compare_payload(left="nope", right="also-nope"), token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

        # An ORG1 credential cannot read ORG2's snapshot on either side.
        status, body = self.compare(
            self.compare_payload(left="foreign", right="snap-right")
        )
        self.assertEqual(status, 403)
        status, body = self.compare(
            self.compare_payload(left="snap-left", right="foreign")
        )
        self.assertEqual(status, 403)

    def test_snapshot_existence_checked_left_then_right(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        self.capture("foreign", token="w2")

        status, body = self.compare(self.compare_payload(left="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        status, body = self.compare(self.compare_payload(right="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

        # A missing left outranks a foreign right.
        status, body = self.compare(
            self.compare_payload(left="ghost", right="foreign")
        )
        self.assertEqual(status, 404)
        # ...and a foreign left outranks a missing right.
        status, body = self.compare(
            self.compare_payload(left="foreign", right="ghost")
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_organizations_are_kept_isolated_in_the_replay(self) -> None:
        self.add_main_event("l1", 5)
        self.capture("snap-left")
        self.add_main_event("r1", 10)
        self.capture("snap-right")
        self.add_main_event("f1", 65, token="w2")
        self.capture("foreign-a", token="w2")
        self.add_main_event("f2", 125, token="w2")
        self.capture("foreign-b", token="w2")

        # ORG2's own two snapshots align within ORG2 only; f1 is in both
        # captures (cumulative) and f2 is right-only.
        payload = {
            "organizationId": ORG2,
            "left": "foreign-a",
            "right": "foreign-b",
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 1,
        }
        status, body = self.call(PATH, method="POST", payload=payload, token="w2")
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["f1", "f2"]
        )
        # f1 is shared and already at the threshold: both sides escalate.
        self.assertEqual(
            body["steps"][0]["windows"], [self.row(60, 1, 1, True)]
        )
        self.assertEqual(
            body["steps"][0]["decision"]["left"]["action"], "escalate"
        )
        # f2 is right-only; the left prefix stays frozen at f1.
        self.assertEqual(
            body["steps"][1]["windows"],
            [self.row(60, 1, 1, True), self.row(120, 0, 1, False)],
        )

        # ORG1's comparison never sees ORG2 events. Its two captures are
        # cumulative too: l1 is shared, r1 is right-only.
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["l1", "r1"]
        )
        self.assertEqual(
            body["steps"][1]["windows"], [self.row(0, 1, 2, False)]
        )

    # ----------------------------------------------------------- 422 / 415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {k: v for k, v in valid.items() if k != "type"},
            {k: v for k, v in valid.items() if k != "windowSize"},
            {k: v for k, v in valid.items() if k != "threshold"},
            {**valid, "extra": 1},
            {**valid, "left": ""},
            {**valid, "right": "   "},
            {**valid, "left": 7},
            {**valid, "right": None},
            {**valid, "type": ""},
            {**valid, "organizationId": ""},
            {**valid, "windowSize": 0},
            {**valid, "windowSize": -3},
            {**valid, "windowSize": 1.5},
            {**valid, "windowSize": True},
            {**valid, "windowSize": "60"},
            {**valid, "threshold": 0},
            {**valid, "threshold": False},
            {**valid, "threshold": 2.0},
            {**valid, "from": 10},
            {**valid, "to": 10},
            {**valid, "from": -1, "to": 10},
            {**valid, "from": 10, "to": 9},
            {**valid, "from": "0", "to": 9},
            {**valid, "from": 0, "to": False},
            [],
            "x",
            42,
            True,
            None,
        ]
        for bad_payload in bad_payloads:
            with self.subTest(bad_payload=bad_payload):
                status, parsed = self.call(
                    PATH,
                    method="POST",
                    raw_body=json.dumps(bad_payload).encode(),
                    token="w1",
                )
                self.assertEqual(status, 422)
                self.assertEqual(parsed["error"], "validation_error")

        snapshot_ids = [
            entry["snapshotId"] for entry in self.call("/snapshots")[1]["snapshots"]
        ]
        self.assertEqual(sorted(snapshot_ids), ["snap-left", "snap-right"])

    def test_duplicate_json_keys_are_422_validation_error_and_write_nothing(
        self,
    ) -> None:
        # A body with a repeated field is syntactically valid JSON but
        # violates the exact-fields contract, so it is 422 (never 400) and
        # creates no snapshot and mutates no state.
        self.capture("snap-left")
        self.capture("snap-right")
        raw_body = (
            b'{"organizationId":"org-1","left":"snap-left","right":"snap-right",'
            b'"type":"incident.created","windowSize":60,"threshold":3,'
            b'"left":"ghost"}'
        )
        status, parsed = self.call(
            PATH, method="POST", raw_body=raw_body, token="w1"
        )
        self.assertEqual(status, 422)
        self.assertEqual(parsed["error"], "validation_error")
        snapshot_ids = [
            entry["snapshotId"] for entry in self.call("/snapshots")[1]["snapshots"]
        ]
        self.assertNotIn("ghost", snapshot_ids)

    def test_media_type_and_json_errors(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        raw_body = json.dumps(self.compare_payload()).encode()

        status, body = self.call(
            PATH, method="POST", raw_body=raw_body, content_type=None
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            PATH, method="POST", raw_body=raw_body, content_type="text/plain"
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            PATH,
            method="POST",
            raw_body=b'{"left": ',
            content_type="application/json",
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

        self.assertEqual(self.call("/snapshots")[0], 200)

    def test_only_post_is_served_and_no_snapshot_sub_path_is_added(self) -> None:
        self.capture("snap-left")
        # GET falls through to the generic JSON 404 under /snapshots/...
        status, body = self.call(PATH)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        # An unknown extra sub-path stays 404 under the generic route.
        status, body = self.call("/snapshots/compare/replay/decisions/extra")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
