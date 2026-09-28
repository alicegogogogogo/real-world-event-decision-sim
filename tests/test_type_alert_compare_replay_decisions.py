"""Regression tests for POST /alerts/types/compare/replay/decisions.

The baseline already had the region-dimension step-by-step alert replay
comparison (``POST /alerts/regions/compare/replay/decisions``) but no way
to align two event types' alert replays. This locks down the new
read-only entry point:

- each side replays only its own type's events of the caller's
  organization, ordered by occurredAt then eventId (payload region is
  irrelevant on this dimension);
- the two replays align by eventId: a shared identifier is one shared
  step, an identifier on only one side is its own step on which the
  other side stays put with its prefix and suppression state untouched,
  and the aligned steps run by occurred time then identifier;
- every step carries the event id/time, window rows aligned by start
  (ranged empty windows kept), and the two sides' peak/alert decisions
  with an equality marker that considers peakStart, peakCount, and
  action only;
- simulated alert identifiers restart at ``alert-1`` independently on
  each side, the alert store is never read or written and the
  service-wide counter never advances;
- the response is compact key-sorted JSON with integer/boolean values
  and one trailing newline; identical submissions are byte-for-byte
  identical, both roles may call it, and 401 / 415 / 400 / 422 / 403
  follow the fixed ordering.

On the main ledger eventId is globally unique, so two distinct types
hold disjoint event ids; the shared-identifier merge is reached over
HTTP by naming the same type on both sides (legal) and is also
exercised directly against the pure alignment function, the same way
the region comparison tests cover that shape.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from event_sim.server import compare_type_alert_replay_decisions, create_server

ORG1 = "org-1"
ORG2 = "org-2"
LEFT = "incident.created"
RIGHT = "incident.updated"
PATH = "/alerts/types/compare/replay/decisions"


def event_body(
    event_id: str,
    occurred_at: int,
    *,
    event_type: str = LEFT,
    organization_id: str = ORG1,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": event_type,
        "occurredAt": occurred_at,
        "payload": {} if payload is None else payload,
    }


class TypeAlertCompareReplayDecisionsTest(unittest.TestCase):
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

    def add_event(self, event: dict[str, Any], *, token: str = "w1") -> None:
        status, _ = self.call("/events", method="POST", payload=event, token=token)
        self.assertEqual(status, 201)

    def compare_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": ORG1,
            "left": LEFT,
            "right": RIGHT,
            "windowSize": 60,
            "threshold": 3,
            "suppressionWindow": 120,
        }
        payload.update(overrides)
        return payload

    def compare(
        self,
        payload: dict[str, Any] | None = None,
        *,
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        return self.call(
            PATH,
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    def type_replay(self, event_type: str, **overrides: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "organizationId": ORG1,
            "type": event_type,
            "windowSize": 60,
            "threshold": 3,
            "suppressionWindow": 120,
        }
        params.update(overrides)
        status, body = self.call(f"/alerts/replay/decisions?{urlencode(params)}")
        self.assertEqual(status, 200)
        return body

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

    @staticmethod
    def side(
        start: Any,
        count: int,
        action: str,
        alert_id: Any,
        suppressed: Any,
    ) -> dict[str, Any]:
        return {
            "action": action,
            "alertId": alert_id,
            "peakCount": count,
            "peakStart": start,
            "suppressedCount": suppressed,
        }

    # ------------------------------------------------------------- alignment

    def test_same_type_on_both_sides_merges_every_identifier(self) -> None:
        self.add_event(event_body("e1", 5))
        self.add_event(event_body("e2", 65))

        status, body = self.compare(self.compare_payload(right=LEFT))
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["e1", "e2"])
        self.assertEqual([step["occurredAt"] for step in body["steps"]], [5, 65])
        first, second = body["steps"]
        self.assertEqual(first["windows"], [self.row(0, 1, 1, True)])
        self.assertEqual(
            first["decision"],
            {
                "left": self.side(0, 1, "observe", None, None),
                "right": self.side(0, 1, "observe", None, None),
                "equal": True,
            },
        )
        self.assertEqual(
            second["windows"],
            [self.row(0, 1, 1, True), self.row(60, 1, 1, True)],
        )
        self.assertTrue(second["decision"]["equal"])
        self.assertEqual(
            second["decision"]["left"], second["decision"]["right"]
        )

    def test_one_sided_identifier_freezes_other_side_without_reset(self) -> None:
        # Left opens alert-1 on its first two events; the right type's only
        # event is later in the next window. On the left steps the right
        # side holds the initial observe state, and on the right step the
        # left side is frozen mid-escalation: its prefix and suppression
        # state are neither advanced nor cleared.
        self.add_event(event_body("l1", 5, event_type=LEFT))
        self.add_event(event_body("l2", 10, event_type=LEFT))
        self.add_event(event_body("r1", 70, event_type=RIGHT))

        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["l1", "l2", "r1"]
        )
        l1, l2, r1 = body["steps"]
        self.assertEqual(l1["windows"], [self.row(0, 1, 0, False)])
        self.assertEqual(
            l1["decision"]["left"], self.side(0, 1, "observe", None, None)
        )
        self.assertEqual(
            l1["decision"]["right"], self.side(None, 0, "observe", None, None)
        )
        self.assertFalse(l1["decision"]["equal"])
        self.assertEqual(
            l2["decision"]["left"], self.side(0, 2, "escalate", "alert-1", 0)
        )
        self.assertEqual(
            l2["decision"]["right"], self.side(None, 0, "observe", None, None)
        )
        # The right-only step leaves the left decision exactly as it was;
        # the right event at 70 counts in window 60, whose lone row is the
        # right side's peak (start 60, count 1).
        self.assertEqual(
            r1["decision"]["left"], self.side(0, 2, "escalate", "alert-1", 0)
        )
        self.assertEqual(
            r1["decision"]["right"], self.side(60, 1, "observe", None, None)
        )
        self.assertFalse(r1["decision"]["equal"])
        self.assertEqual(
            r1["windows"],
            [self.row(0, 2, 0, False), self.row(60, 0, 1, False)],
        )

    def test_steps_ordered_by_time_then_identifier(self) -> None:
        self.add_event(event_body("l-b", 100, event_type=LEFT))
        self.add_event(event_body("l-a", 100, event_type=LEFT))
        self.add_event(event_body("r-c", 50, event_type=RIGHT))

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("r-c", 50), ("l-a", 100), ("l-b", 100)],
        )

    def test_shared_and_unique_identifiers_merge_through_pure_function(self) -> None:
        def ev(event_id: str, occurred_at: int, event_type: str = LEFT) -> dict[str, Any]:
            return {
                "eventId": event_id,
                "organizationId": ORG1,
                "type": event_type,
                "occurredAt": occurred_at,
                "payload": {},
            }

        params = self.compare_payload(threshold=2)
        params["from"] = params["to"] = None
        result = compare_type_alert_replay_decisions(
            [ev("shared", 10), ev("l-only", 20)],
            [ev("shared", 10, RIGHT), ev("r-only", 70, RIGHT)],
            params,
        )
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in result["steps"]],
            [("shared", 10), ("l-only", 20), ("r-only", 70)],
        )
        shared, l_only, r_only = result["steps"]
        self.assertEqual(shared["windows"], [self.row(0, 1, 1, True)])
        # The left-only step extends only the left prefix.
        self.assertEqual(l_only["windows"], [self.row(0, 2, 1, False)])
        self.assertEqual(
            l_only["decision"]["left"], self.side(0, 2, "escalate", "alert-1", 0)
        )
        self.assertEqual(
            l_only["decision"]["right"], self.side(0, 1, "observe", None, None)
        )
        # The right-only step freezes the left side mid-escalation.
        self.assertEqual(
            r_only["decision"]["left"], self.side(0, 2, "escalate", "alert-1", 0)
        )
        self.assertEqual(
            r_only["windows"],
            [self.row(0, 2, 1, False), self.row(60, 0, 1, False)],
        )

    def test_shared_identifier_disagreement_on_time_reports_earlier(self) -> None:
        def ev(event_id: str, occurred_at: int, event_type: str = LEFT) -> dict[str, Any]:
            return {
                "eventId": event_id,
                "organizationId": ORG1,
                "type": event_type,
                "occurredAt": occurred_at,
                "payload": {},
            }

        params = self.compare_payload()
        params["from"] = params["to"] = None
        result = compare_type_alert_replay_decisions(
            [ev("dup", 100, LEFT), ev("late-l", 150, LEFT)],
            [ev("dup", 40, RIGHT), ev("early-r", 10, RIGHT)],
            params,
        )
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in result["steps"]],
            [("early-r", 10), ("dup", 40), ("late-l", 150)],
        )
        dup = result["steps"][1]
        self.assertEqual(
            dup["windows"],
            [
                self.row(0, 0, 2, False),
                self.row(60, 1, 0, False),
            ],
        )
        # Each side accumulated its own copy of the shared id at its own
        # time; neither prefix contains the other side's timestamp.
        self.assertEqual(dup["decision"]["left"]["peakStart"], 60)
        self.assertEqual(dup["decision"]["right"]["peakStart"], 0)

    def test_window_rows_align_union_of_hit_windows_with_missing_side_zero(
        self,
    ) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
        self.add_event(event_body("l2", 10, event_type=LEFT))
        self.add_event(event_body("r1", 125, event_type=RIGHT))

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["steps"][-1]["windows"],
            [
                self.row(0, 2, 0, False),
                self.row(120, 0, 1, False),
            ],
        )

    # ----------------------------------------------------- suppression rules

    def test_suppression_state_is_private_to_each_side(self) -> None:
        # windowSize 60, threshold 2, suppressionWindow 60.
        # Left:  10,20 open alert-1 at start 0; 60,61 are suppressed; 62
        # makes window 60 strictly the largest and opens alert-2 at the
        # exact suppression distance. Right: 10,20 open its own alert-1 and
        # then nothing, so its id numbering never catches up while the
        # left side moves alone.
        for event_id, timestamp in (
            ("l-a", 10),
            ("l-b", 20),
            ("l-c", 60),
            ("l-d", 61),
            ("l-e", 62),
        ):
            self.add_event(event_body(event_id, timestamp, event_type=LEFT))
        for event_id, timestamp in (("r-a", 10), ("r-b", 20)):
            self.add_event(event_body(event_id, timestamp, event_type=RIGHT))

        status, body = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=60)
        )
        self.assertEqual(status, 200)
        by_id = {step["eventId"]: step for step in body["steps"]}
        self.assertEqual(by_id["l-a"]["decision"]["left"]["action"], "observe")
        self.assertEqual(by_id["l-b"]["decision"]["left"]["alertId"], "alert-1")
        self.assertEqual(by_id["l-c"]["decision"]["left"]["action"], "suppress")
        self.assertEqual(by_id["l-d"]["decision"]["left"]["action"], "suppress")
        self.assertEqual(
            (
                by_id["l-e"]["decision"]["left"]["action"],
                by_id["l-e"]["decision"]["left"]["alertId"],
                by_id["l-e"]["decision"]["left"]["suppressedCount"],
                by_id["l-e"]["decision"]["left"]["peakStart"],
            ),
            ("escalate", "alert-2", 0, 60),
        )
        # On every left-only step the right side re-reports its own last
        # decision, which opened its own independent alert-1.
        self.assertEqual(
            by_id["l-e"]["decision"]["right"],
            self.side(0, 2, "escalate", "alert-1", 0),
        )
        self.assertFalse(by_id["l-e"]["decision"]["equal"])

    def test_suppress_carries_prior_id_and_cumulative_count_per_side(self) -> None:
        # Each side opens alert-1 on its own two close events and then takes
        # a private suppression; the frozen side never advances its count.
        self.add_event(event_body("l-a", 10, event_type=LEFT))
        self.add_event(event_body("l-b", 20, event_type=LEFT))
        self.add_event(event_body("r-a", 10, event_type=RIGHT))
        self.add_event(event_body("r-b", 20, event_type=RIGHT))
        self.add_event(event_body("l-c", 21, event_type=LEFT))
        self.add_event(event_body("r-c", 22, event_type=RIGHT))

        status, body = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=60)
        )
        self.assertEqual(status, 200)
        by_id = {step["eventId"]: step for step in body["steps"]}
        self.assertEqual(
            (
                by_id["l-c"]["decision"]["left"]["action"],
                by_id["l-c"]["decision"]["left"]["alertId"],
                by_id["l-c"]["decision"]["left"]["suppressedCount"],
            ),
            ("suppress", "alert-1", 1),
        )
        # The right side has not moved yet: still its own escalate with 0.
        self.assertEqual(by_id["l-c"]["decision"]["right"]["suppressedCount"], 0)
        self.assertEqual(
            (
                by_id["r-c"]["decision"]["right"]["action"],
                by_id["r-c"]["decision"]["right"]["alertId"],
                by_id["r-c"]["decision"]["right"]["suppressedCount"],
            ),
            ("suppress", "alert-1", 1),
        )
        # The left side is frozen on its suppression count of 1.
        self.assertEqual(by_id["r-c"]["decision"]["left"]["suppressedCount"], 1)

    def test_equal_marker_ignores_suppressed_count(self) -> None:
        # Both sides reach the same peak triple (start 0, count 3,
        # suppress); the left then adds a trailing-window event that does
        # not overtake the peak but advances its suppressed count to 2,
        # while the frozen right stays at 1. The equal marker must remain
        # true because it compares peak start/count/action only.
        self.add_event(event_body("l-a", 10, event_type=LEFT))
        self.add_event(event_body("l-b", 20, event_type=LEFT))
        self.add_event(event_body("r-a", 10, event_type=RIGHT))
        self.add_event(event_body("r-b", 20, event_type=RIGHT))
        self.add_event(event_body("l-c", 21, event_type=LEFT))
        self.add_event(event_body("r-c", 22, event_type=RIGHT))
        self.add_event(event_body("l-tail", 60, event_type=LEFT))

        status, body = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=60)
        )
        self.assertEqual(status, 200)
        final = body["steps"][-1]
        self.assertEqual(final["eventId"], "l-tail")
        self.assertEqual(
            (
                final["decision"]["left"]["action"],
                final["decision"]["right"]["action"],
            ),
            ("suppress", "suppress"),
        )
        self.assertEqual(
            (
                final["decision"]["left"]["peakStart"],
                final["decision"]["right"]["peakStart"],
            ),
            (0, 0),
        )
        self.assertEqual(
            (
                final["decision"]["left"]["peakCount"],
                final["decision"]["right"]["peakCount"],
            ),
            (3, 3),
        )
        self.assertEqual(
            (
                final["decision"]["left"]["suppressedCount"],
                final["decision"]["right"]["suppressedCount"],
            ),
            (2, 1),
        )
        self.assertTrue(final["decision"]["equal"])

    def test_equal_marker_is_false_when_peak_triple_differs(self) -> None:
        self.add_event(event_body("l-a", 10, event_type=LEFT))
        self.add_event(event_body("l-b", 20, event_type=LEFT))
        self.add_event(event_body("r-a", 10, event_type=RIGHT))
        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        # l-a first alone (right side is the empty initial state:
        # peakStart null), then r-a ties both sides, then l-b pushes only
        # the left side over the threshold.
        self.assertEqual(
            [step["decision"]["equal"] for step in body["steps"]],
            [False, True, False],
        )

    def test_observe_step_carries_null_alert_identity(self) -> None:
        self.add_event(event_body("l-a", 10, event_type=LEFT))
        self.add_event(event_body("r-a", 10, event_type=RIGHT))
        status, body = self.compare(self.compare_payload(threshold=3))
        self.assertEqual(status, 200)
        l_step, r_step = body["steps"]
        # Before the other type has moved, its decision is the initial
        # empty-prefix observe state (null peak start, zero count).
        self.assertEqual(
            l_step["decision"]["left"], self.side(0, 1, "observe", None, None)
        )
        self.assertEqual(
            l_step["decision"]["right"],
            self.side(None, 0, "observe", None, None),
        )
        # Once both have their one event they agree.
        self.assertEqual(
            r_step["decision"]["left"], self.side(0, 1, "observe", None, None)
        )
        self.assertEqual(
            r_step["decision"]["right"], self.side(0, 1, "observe", None, None)
        )
        self.assertTrue(r_step["decision"]["equal"])

    def test_each_side_matches_single_type_replay_item_by_item(self) -> None:
        self.add_event(event_body("l-1", 5, event_type=LEFT))
        self.add_event(event_body("l-2", 10, event_type=LEFT))
        self.add_event(event_body("l-3", 63, event_type=LEFT))
        self.add_event(event_body("r-1", 5, event_type=RIGHT))
        self.add_event(event_body("r-2", 70, event_type=RIGHT))

        status, body = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=60)
        )
        self.assertEqual(status, 200)
        left_replay = self.type_replay(LEFT, threshold=2, suppressionWindow=60)
        right_replay = self.type_replay(
            RIGHT, threshold=2, suppressionWindow=60
        )

        left_expected = {step["eventId"]: step for step in left_replay["steps"]}
        right_expected = {
            step["eventId"]: step for step in right_replay["steps"]
        }
        fields = (
            "peakStart",
            "peakCount",
            "action",
            "alertId",
            "suppressedCount",
        )
        for step in body["steps"]:
            event_id = step["eventId"]
            if event_id in left_expected:
                for field in fields:
                    self.assertEqual(
                        step["decision"]["left"][field],
                        left_expected[event_id][field],
                    )
            if event_id in right_expected:
                for field in fields:
                    self.assertEqual(
                        step["decision"]["right"][field],
                        right_expected[event_id][field],
                    )

    # ------------------------------------------------------------- attribution

    def test_attribution_matches_types_verbatim_and_ignores_region(self) -> None:
        # The left type's events carry varied payload regions; region is
        # irrelevant on this dimension, so every LEFT event counts.
        self.add_event(event_body("l1", 5, payload={"region": "north"}))
        self.add_event(event_body("l2", 6, payload={"region": "south"}))
        self.add_event(event_body("l3", 7, payload={}))
        # A different type and another organization never enter.
        self.add_event(event_body("other-1", 8, event_type="other.kind"))
        self.add_event(
            event_body("org2", 9, event_type=LEFT, organization_id=ORG2),
            token="w2",
        )

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["l1", "l2", "l3"]
        )

        # An unknown type on either side simply yields no steps for it.
        status, body = self.compare(self.compare_payload(right="unseen.kind"))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["l1", "l2", "l3"]
        )
        self.assertEqual(
            body["steps"][-1]["windows"], [self.row(0, 3, 0, False)]
        )

        status, body = self.compare(
            self.compare_payload(left="ghost.kind", right="also.ghost")
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])

        # Other organizations' data never enters even with the same type.
        status, body = self.compare(
            self.compare_payload(organizationId=ORG2, right=LEFT), token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["org2"])

    # ------------------------------------------------------------- range

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
        self.add_event(event_body("l2", 65, event_type=LEFT))  # outside [0, 60]
        self.add_event(event_body("r1", 120, event_type=RIGHT))  # outside too

        status, body = self.compare(
            self.compare_payload(**{"from": 0, "to": 60})
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 60)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["l1", "l2", "r1"],
        )
        for step in body["steps"]:
            self.assertEqual(
                [row["start"] for row in step["windows"]], [0, 60]
            )
        self.assertEqual(
            body["steps"][-1]["windows"],
            [
                self.row(0, 1, 0, False),
                self.row(60, 0, 0, True),
            ],
        )

    def test_range_runs_suppression_only_from_ranged_peaks(self) -> None:
        # The left event at 10 lies outside [60, 120]; the two in-range
        # events on each side open that side's first simulated alert at
        # start 60.
        self.add_event(event_body("l-a", 10, event_type=LEFT))
        self.add_event(event_body("l-b", 60, event_type=LEFT))
        self.add_event(event_body("l-c", 61, event_type=LEFT))
        self.add_event(event_body("r-b", 60, event_type=RIGHT))
        self.add_event(event_body("r-c", 61, event_type=RIGHT))

        status, body = self.compare(
            self.compare_payload(
                threshold=2, suppressionWindow=60, **{"from": 60, "to": 120}
            )
        )
        self.assertEqual(status, 200)
        by_id = {step["eventId"]: step for step in body["steps"]}
        self.assertEqual(by_id["l-a"]["decision"]["left"]["action"], "observe")
        self.assertIsNone(by_id["l-a"]["decision"]["left"]["peakStart"])
        self.assertEqual(by_id["l-a"]["decision"]["left"]["peakCount"], 0)
        self.assertEqual(by_id["l-b"]["decision"]["left"]["action"], "observe")
        self.assertEqual(
            (
                by_id["l-c"]["decision"]["left"]["action"],
                by_id["l-c"]["decision"]["left"]["alertId"],
                by_id["l-c"]["decision"]["left"]["suppressedCount"],
            ),
            ("escalate", "alert-1", 0),
        )
        self.assertEqual(by_id["l-c"]["decision"]["left"]["peakStart"], 60)
        # The right side only reaches its matching threshold hit on r-c,
        # where both sides report the same peak triple.
        self.assertFalse(by_id["l-c"]["decision"]["equal"])
        self.assertTrue(by_id["r-c"]["decision"]["equal"])
        self.assertEqual(
            by_id["r-c"]["decision"]["right"]["alertId"], "alert-1"
        )

    # ------------------------------------------------------------- echo/shape

    def test_response_echoes_all_request_parameters(self) -> None:
        self.add_event(event_body("a", 5))
        payload = self.compare_payload(
            threshold=4, suppressionWindow=90, **{"from": 0, "to": 120}
        )
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["left"], LEFT)
        self.assertEqual(body["right"], RIGHT)
        self.assertNotIn("type", body)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 4)
        self.assertEqual(body["suppressionWindow"], 90)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 120)

    def test_no_steps_when_neither_type_matches(self) -> None:
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])

    def test_response_is_compact_sorted_boolean_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
        self.add_event(event_body("r1", 5, event_type=RIGHT))
        self.add_event(event_body("r2", 65, event_type=RIGHT))

        raw_body = json.dumps(self.compare_payload()).encode()
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
                "suppressionWindow",
                "threshold",
                "to",
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
            [
                "action",
                "alertId",
                "peakCount",
                "peakStart",
                "suppressedCount",
            ],
        )
        self.assertEqual(
            list(body["steps"][0]["windows"][0]),
            ["equal", "leftCount", "rightCount", "start"],
        )
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )
        self.assertIsInstance(body["windowSize"], int)
        self.assertIsInstance(body["threshold"], int)
        self.assertIsInstance(body["suppressionWindow"], int)
        self.assertIsInstance(body["steps"][0]["occurredAt"], int)
        self.assertIsInstance(body["steps"][0]["decision"]["equal"], bool)

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
        self.add_event(event_body("l2", 65, event_type=LEFT))
        self.add_event(event_body("r1", 120, event_type=RIGHT))
        raw_body = json.dumps(
            self.compare_payload(**{"from": 0, "to": 180})
        ).encode()
        raws = [
            self.raw(PATH, method="POST", body=raw_body, token="w1")[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # -------------------------------------------------------------- read-only

    def test_comparison_never_reads_or_writes_alert_store(self) -> None:
        self.add_event(event_body("l-a", 10, event_type=LEFT))
        self.add_event(event_body("l-b", 20, event_type=LEFT))

        # Commit a real organization alert first; it advances the
        # service-wide counter and records a suppression.
        evaluate_payload = {
            "organizationId": ORG1,
            "type": LEFT,
            "windowSize": 60,
            "threshold": 2,
            "suppressionWindow": 10000,
        }
        status, first = self.call(
            "/alerts/evaluate", method="POST", payload=evaluate_payload
        )
        self.assertEqual(
            (status, first["action"], first["alertId"]),
            (200, "escalate", "alert-1"),
        )
        status, second = self.call(
            "/alerts/evaluate", method="POST", payload=evaluate_payload
        )
        self.assertEqual(
            (status, second["action"], second["alertId"]),
            (200, "suppress", "alert-1"),
        )

        # The comparison's simulated sequences restart at alert-1 on each
        # side regardless of the committed alert and its suppression state.
        status, body = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=10000)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [
                (
                    step["decision"]["left"]["action"],
                    step["decision"]["left"]["alertId"],
                    step["decision"]["left"]["suppressedCount"],
                )
                for step in body["steps"]
            ],
            [("observe", None, None), ("escalate", "alert-1", 0)],
        )

        # The committed alert is untouched and the global counter did not
        # advance: another real evaluation suppresses as alert-1 with the
        # next cumulative count.
        status, after = self.call(
            "/alerts/evaluate", method="POST", payload=evaluate_payload
        )
        self.assertEqual(status, 200)
        self.assertEqual(after["alertId"], "alert-1")
        self.assertEqual(after["suppressedCount"], 2)

    def test_comparison_is_read_only(self) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
        self.add_event(event_body("r1", 10, event_type=RIGHT))

        def listings() -> tuple[Any, Any]:
            events = self.call(f"/events?organizationId={ORG1}")[1]
            org_alerts = self.call(f"/alerts?organizationId={ORG1}")[1]
            return events, org_alerts

        before = listings()
        for _ in range(3):
            status, _ = self.compare(
                self.compare_payload(**{"from": 0, "to": 60})
            )
            self.assertEqual(status, 200)
        self.assertEqual(before, listings())

    def test_read_and_write_credentials_may_both_compare(self) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
        payload = self.compare_payload()
        read_status, read_body = self.compare(payload, token="r1")
        write_status, write_body = self.compare(payload, token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    # ----------------------------------------------------------------- auth

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
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
                self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_credential_checked_before_body_shape(self) -> None:
        # No Authorization header at all -> 401 even with garbage body.
        status, raw = self.raw(PATH, method="POST", body=b"not json", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_foreign_organization_is_403_before_type_inspection(self) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
        # An ORG2 credential naming ORG1 is forbidden regardless of type
        # names, even ones that exist nowhere.
        status, body = self.compare(
            self.compare_payload(left="nope.kind", right="also.nope"),
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        status, events_doc = self.call(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(events_doc["events"]), 1)

    # ----------------------------------------------------------- 422/415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {k: v for k, v in valid.items() if k != "windowSize"},
            {k: v for k, v in valid.items() if k != "threshold"},
            {k: v for k, v in valid.items() if k != "suppressionWindow"},
            # A shared "type" field has no place on this entry point.
            {**valid, "type": LEFT},
            {**valid, "extra": 1},
            {**valid, "left": ""},
            {**valid, "right": "   "},
            {**valid, "left": 7},
            {**valid, "right": None},
            {**valid, "organizationId": ""},
            {**valid, "windowSize": 0},
            {**valid, "windowSize": -3},
            {**valid, "windowSize": 1.5},
            {**valid, "windowSize": True},
            {**valid, "windowSize": "60"},
            {**valid, "threshold": 0},
            {**valid, "threshold": False},
            {**valid, "threshold": 2.0},
            {**valid, "suppressionWindow": 0},
            {**valid, "suppressionWindow": -1},
            {**valid, "suppressionWindow": 1.5},
            {**valid, "suppressionWindow": "60"},
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

        status, events_doc = self.call(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(events_doc["events"]), 1)

    def test_duplicate_json_keys_are_422_validation_error(self) -> None:
        raw_body = (
            b'{"organizationId":"org-1","left":"incident.created",'
            b'"right":"incident.updated","windowSize":60,"threshold":3,'
            b'"suppressionWindow":120,"left":"ghost.kind"}'
        )
        status, parsed = self.call(
            PATH, method="POST", raw_body=raw_body, token="w1"
        )
        self.assertEqual(status, 422)
        self.assertEqual(parsed["error"], "validation_error")

    def test_media_type_and_json_errors(self) -> None:
        self.add_event(event_body("l1", 5, event_type=LEFT))
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

    def test_only_post_is_served_and_no_type_sub_path_is_added(self) -> None:
        status, body = self.call(PATH)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, body = self.call(
            "/alerts/types/incident.created/compare/replay/decisions",
            method="POST",
            payload=self.compare_payload(),
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
