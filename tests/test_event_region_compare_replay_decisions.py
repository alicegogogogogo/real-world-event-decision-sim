"""Regression tests for POST /events/regions/compare/replay/decisions.

The baseline already had the region-dimension step-by-step **alert** replay
comparison (``POST /alerts/regions/compare/replay/decisions``) and the
type-dimension plain event replay comparison
(``POST /events/types/compare/replay/decisions``), but no way to align two
regions' plain event replays. This locks down the new read-only entry
point:

- each side replays only its own region's events of the caller's
  organization (region attribution follows the public verbatim
  payload-region rule), ordered by occurredAt then eventId;
- the two replays align by eventId: a shared identifier is one shared
  step, an identifier on only one side is its own step on which the
  other side stays put keeping its accumulated prefix, and the aligned
  steps run by occurred time then identifier;
- every step carries the event id/time, window rows aligned by start
  (ranged empty windows kept), and the two sides' threshold-only peak
  decisions (``observe``/``escalate`` — no alert identity, no
  suppression count) with an equality marker that considers peakStart,
  peakCount, and action only;
- the response is compact key-sorted JSON with integer/boolean values
  and one trailing newline; identical submissions are byte-for-byte
  identical, both roles may call it, and 401 / 415 / 400 / 422 / 403
  follow the fixed ordering.

On the main ledger every event is attributed to at most one region, so
two distinct region names hold disjoint event ids; the shared-identifier
merge is reached over HTTP by naming the same region on both sides
(legal) and is also exercised directly against the pure alignment
function, the same way the alert comparison tests cover that shape.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from event_sim.server import (
    compare_event_region_replay_decisions,
    create_server,
)

ORG1 = "org-1"
ORG2 = "org-2"
LEFT = "north"
RIGHT = "south"
EVENT_TYPE = "incident.created"
PATH = "/events/regions/compare/replay/decisions"


def event_body(
    event_id: str,
    occurred_at: int,
    *,
    region: str | None = LEFT,
    event_type: str = EVENT_TYPE,
    organization_id: str = ORG1,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_payload: dict[str, Any] = {} if payload is None else dict(payload)
    if region is not None and "region" not in event_payload:
        event_payload["region"] = region
    return {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": event_type,
        "occurredAt": occurred_at,
        "payload": event_payload,
    }


class EventRegionCompareReplayDecisionsTest(unittest.TestCase):
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

    def region_replay(self, region: str, **overrides: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "organizationId": ORG1,
            "region": region,
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 3,
        }
        params.update(overrides)
        status, body = self.call(
            f"/events/region/replay/decisions?{urlencode(params)}"
        )
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
    def side(start: Any, count: int, action: str) -> dict[str, Any]:
        return {"action": action, "peakCount": count, "peakStart": start}

    # ------------------------------------------------------------- alignment

    def test_same_region_on_both_sides_merges_every_identifier(self) -> None:
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
                "left": self.side(0, 1, "observe"),
                "right": self.side(0, 1, "observe"),
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

    def test_one_sided_identifier_freezes_other_side_prefix(self) -> None:
        # Two north events in window 0 escalate the left side at threshold
        # 2; the south event lands in the next window. On the north steps the
        # right side holds the initial empty-prefix observe state, and on the
        # south step the left side is frozen mid-escalation: its prefix is
        # neither advanced nor cleared.
        self.add_event(event_body("n1", 5, region=LEFT))
        self.add_event(event_body("n2", 10, region=LEFT))
        self.add_event(event_body("s1", 70, region=RIGHT))

        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["n1", "n2", "s1"]
        )
        n1, n2, s1 = body["steps"]
        self.assertEqual(n1["windows"], [self.row(0, 1, 0, False)])
        self.assertEqual(n1["decision"]["left"], self.side(0, 1, "observe"))
        self.assertEqual(n1["decision"]["right"], self.side(None, 0, "observe"))
        self.assertFalse(n1["decision"]["equal"])
        self.assertEqual(n2["decision"]["left"], self.side(0, 2, "escalate"))
        self.assertEqual(n2["decision"]["right"], self.side(None, 0, "observe"))
        # The right-only step leaves the left decision exactly as it was;
        # the south event at 70 counts in window 60, whose lone row is the
        # right side's peak (start 60, count 1).
        self.assertEqual(s1["decision"]["left"], self.side(0, 2, "escalate"))
        self.assertEqual(s1["decision"]["right"], self.side(60, 1, "observe"))
        self.assertFalse(s1["decision"]["equal"])
        self.assertEqual(
            s1["windows"],
            [self.row(0, 2, 0, False), self.row(60, 0, 1, False)],
        )

    def test_steps_ordered_by_time_then_identifier(self) -> None:
        self.add_event(event_body("n-b", 100, region=LEFT))
        self.add_event(event_body("n-a", 100, region=LEFT))
        self.add_event(event_body("s-c", 50, region=RIGHT))

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("s-c", 50), ("n-a", 100), ("n-b", 100)],
        )

    def test_shared_and_unique_identifiers_merge_through_pure_function(self) -> None:
        def ev(event_id: str, occurred_at: int, region: str = LEFT) -> dict[str, Any]:
            return {
                "eventId": event_id,
                "organizationId": ORG1,
                "type": EVENT_TYPE,
                "occurredAt": occurred_at,
                "payload": {"region": region},
            }

        params = self.compare_payload(threshold=2)
        params["from"] = params["to"] = None
        result = compare_event_region_replay_decisions(
            [ev("shared", 10), ev("n-only", 20)],
            [ev("shared", 10, RIGHT), ev("s-only", 70, RIGHT)],
            params,
        )
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in result["steps"]],
            [("shared", 10), ("n-only", 20), ("s-only", 70)],
        )
        shared, n_only, s_only = result["steps"]
        self.assertEqual(shared["windows"], [self.row(0, 1, 1, True)])
        # The left-only step extends only the left prefix.
        self.assertEqual(n_only["windows"], [self.row(0, 2, 1, False)])
        self.assertEqual(n_only["decision"]["left"], self.side(0, 2, "escalate"))
        self.assertEqual(
            n_only["decision"]["right"], self.side(0, 1, "observe")
        )
        # The right-only step freezes the left side mid-escalation.
        self.assertEqual(s_only["decision"]["left"], self.side(0, 2, "escalate"))
        self.assertEqual(
            s_only["windows"],
            [self.row(0, 2, 1, False), self.row(60, 0, 1, False)],
        )

    def test_shared_identifier_disagreement_on_time_reports_earlier(self) -> None:
        def ev(event_id: str, occurred_at: int, region: str = LEFT) -> dict[str, Any]:
            return {
                "eventId": event_id,
                "organizationId": ORG1,
                "type": EVENT_TYPE,
                "occurredAt": occurred_at,
                "payload": {"region": region},
            }

        params = self.compare_payload()
        params["from"] = params["to"] = None
        result = compare_event_region_replay_decisions(
            [ev("dup", 100, LEFT), ev("late-n", 150, LEFT)],
            [ev("dup", 40, RIGHT), ev("early-s", 10, RIGHT)],
            params,
        )
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in result["steps"]],
            [("early-s", 10), ("dup", 40), ("late-n", 150)],
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
        self.add_event(event_body("n1", 5, region=LEFT))
        self.add_event(event_body("n2", 10, region=LEFT))
        self.add_event(event_body("s1", 125, region=RIGHT))

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["steps"][-1]["windows"],
            [
                self.row(0, 2, 0, False),
                self.row(120, 0, 1, False),
            ],
        )

    # --------------------------------------------------------- peak/action

    def test_threshold_decides_escalate_or_observe_per_side(self) -> None:
        self.add_event(event_body("n-a", 10, region=LEFT))
        self.add_event(event_body("n-b", 20, region=LEFT))
        self.add_event(event_body("s-a", 10, region=RIGHT))

        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        # n-a first alone (right side is the empty initial state:
        # peakStart null), then s-a ties both sides, then n-b pushes only
        # the left side over the threshold.
        self.assertEqual(
            [step["decision"]["equal"] for step in body["steps"]],
            [False, True, False],
        )
        by_id = {step["eventId"]: step for step in body["steps"]}
        self.assertEqual(
            by_id["n-b"]["decision"]["left"], self.side(0, 2, "escalate")
        )
        self.assertEqual(
            by_id["n-b"]["decision"]["right"], self.side(0, 1, "observe")
        )

    def test_decision_carries_no_alert_identity_or_suppression_count(self) -> None:
        self.add_event(event_body("n-a", 10, region=LEFT))
        self.add_event(event_body("n-b", 20, region=LEFT))

        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        for step in body["steps"]:
            self.assertEqual(
                set(step["decision"]["left"]),
                {"action", "peakCount", "peakStart"},
            )
            self.assertEqual(
                set(step["decision"]["right"]),
                {"action", "peakCount", "peakStart"},
            )

    def test_each_side_matches_single_region_replay_item_by_item(self) -> None:
        self.add_event(event_body("n-1", 5, region=LEFT))
        self.add_event(event_body("n-2", 10, region=LEFT))
        self.add_event(event_body("n-3", 63, region=LEFT))
        self.add_event(event_body("s-1", 5, region=RIGHT))
        self.add_event(event_body("s-2", 70, region=RIGHT))

        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        left_replay = self.region_replay(LEFT, threshold=2)
        right_replay = self.region_replay(RIGHT, threshold=2)

        left_expected = {step["eventId"]: step for step in left_replay["steps"]}
        right_expected = {
            step["eventId"]: step for step in right_replay["steps"]
        }
        fields = ("peakStart", "peakCount", "action")
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

    def test_attribution_matches_regions_verbatim(self) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
        self.add_event(event_body("n2", 6, region=LEFT))
        self.add_event(event_body("n3", 7, region=LEFT))
        # A different region, an event with no region at all, an empty-string
        # region, and a non-string region never enter either side.
        self.add_event(event_body("s-other", 8, region=RIGHT))
        self.add_event(event_body("none-1", 9, region=None))
        self.add_event(
            event_body("empty-1", 11, payload={"region": ""}),
        )
        self.add_event(
            event_body("number-1", 12, payload={"region": 7}),
        )
        self.add_event(
            event_body("org2", 13, region=LEFT, organization_id=ORG2),
            token="w2",
        )

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["n1", "n2", "n3", "s-other"],
        )

        # Region names are matched verbatim with no normalization: a
        # whitespace-decorated name is its own (unknown) region.
        status, body = self.compare(
            self.compare_payload(left=" north", right=RIGHT)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["s-other"]
        )

        # An unknown region on either side simply yields no steps for it.
        status, body = self.compare(self.compare_payload(right="nowhere"))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["n1", "n2", "n3"]
        )
        self.assertEqual(
            body["steps"][-1]["windows"], [self.row(0, 3, 0, False)]
        )

        status, body = self.compare(
            self.compare_payload(left="ghost", right="also-ghost")
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])

        # Other organizations' data never enters even with the same region.
        status, body = self.compare(
            self.compare_payload(organizationId=ORG2, right=LEFT), token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["org2"])

    def test_all_event_types_in_a_region_enter_the_replay(self) -> None:
        # Unlike the single-region replay, the comparison carries no shared
        # type field: every type attributed to the region enters.
        self.add_event(event_body("a1", 5, region=LEFT, event_type="kind.one"))
        self.add_event(event_body("a2", 6, region=LEFT, event_type="kind.two"))
        self.add_event(event_body("b1", 7, region=RIGHT, event_type="kind.one"))
        # An event with no region is excluded no matter its type.
        self.add_event(
            event_body("none-1", 8, region=None, event_type="kind.one")
        )

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["a1", "a2", "b1"]
        )
        self.assertEqual(
            body["steps"][-1]["windows"], [self.row(0, 2, 1, False)]
        )

    # ------------------------------------------------------------- range

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
        self.add_event(event_body("n2", 65, region=LEFT))  # outside [0, 60]
        self.add_event(event_body("s1", 120, region=RIGHT))  # outside too

        status, body = self.compare(
            self.compare_payload(**{"from": 0, "to": 60})
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 60)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["n1", "n2", "s1"],
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

    def test_range_counts_only_in_range_events_into_peak(self) -> None:
        # The north event at 10 lies outside [60, 120]; the two in-range
        # events escalate the left side only at n-c with peak start 60.
        self.add_event(event_body("n-a", 10, region=LEFT))
        self.add_event(event_body("n-b", 60, region=LEFT))
        self.add_event(event_body("n-c", 61, region=LEFT))
        self.add_event(event_body("s-b", 60, region=RIGHT))
        self.add_event(event_body("s-c", 61, region=RIGHT))

        status, body = self.compare(
            self.compare_payload(threshold=2, **{"from": 60, "to": 120})
        )
        self.assertEqual(status, 200)
        by_id = {step["eventId"]: step for step in body["steps"]}
        self.assertEqual(by_id["n-a"]["decision"]["left"]["action"], "observe")
        self.assertIsNone(by_id["n-a"]["decision"]["left"]["peakStart"])
        self.assertEqual(by_id["n-a"]["decision"]["left"]["peakCount"], 0)
        self.assertEqual(by_id["n-b"]["decision"]["left"]["action"], "observe")
        self.assertEqual(
            (
                by_id["n-c"]["decision"]["left"]["action"],
                by_id["n-c"]["decision"]["left"]["peakStart"],
            ),
            ("escalate", 60),
        )
        self.assertFalse(by_id["n-c"]["decision"]["equal"])
        self.assertTrue(by_id["s-c"]["decision"]["equal"])

    # ------------------------------------------------------------- echo/shape

    def test_response_echoes_all_request_parameters(self) -> None:
        self.add_event(event_body("a", 5))
        payload = self.compare_payload(threshold=4, **{"from": 0, "to": 120})
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["left"], LEFT)
        self.assertEqual(body["right"], RIGHT)
        self.assertNotIn("type", body)
        self.assertNotIn("suppressionWindow", body)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 4)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 120)

    def test_no_steps_when_neither_region_matches(self) -> None:
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])

    def test_response_is_compact_sorted_boolean_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
        self.add_event(event_body("s1", 5, region=RIGHT))
        self.add_event(event_body("s2", 65, region=RIGHT))

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
            ["action", "peakCount", "peakStart"],
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
        self.assertIsInstance(body["steps"][0]["occurredAt"], int)
        self.assertIsInstance(body["steps"][0]["decision"]["equal"], bool)

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
        self.add_event(event_body("n2", 65, region=LEFT))
        self.add_event(event_body("s1", 120, region=RIGHT))
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
        self.add_event(event_body("n1", 5, region=LEFT))
        self.add_event(event_body("s1", 10, region=RIGHT))

        def listings() -> tuple[Any, Any, Any]:
            events = self.call(f"/events?organizationId={ORG1}")[1]
            org_alerts = self.call(f"/alerts?organizationId={ORG1}")[1]
            reservations = self.call(f"/reservations?organizationId={ORG1}")[1]
            return events, org_alerts, reservations

        before = listings()
        for _ in range(3):
            status, _ = self.compare(
                self.compare_payload(**{"from": 0, "to": 60})
            )
            self.assertEqual(status, 200)
        self.assertEqual(before, listings())

    def test_comparison_never_writes_alerts(self) -> None:
        self.add_event(event_body("n-a", 10, region=LEFT))
        self.add_event(event_body("n-b", 20, region=LEFT))

        # The plain replay escalates without ever opening an alert.
        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["decision"]["left"]["action"] for step in body["steps"]],
            ["observe", "escalate"],
        )
        status, org_alerts = self.call(f"/alerts?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(org_alerts["alerts"], [])

    def test_read_and_write_credentials_may_both_compare(self) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
        payload = self.compare_payload()
        read_status, read_body = self.compare(payload, token="r1")
        write_status, write_body = self.compare(payload, token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    # ----------------------------------------------------------------- auth

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
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

    def test_foreign_organization_is_403_before_region_inspection(self) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
        # An ORG2 credential naming ORG1 is forbidden regardless of region
        # names, even ones that exist nowhere.
        status, body = self.compare(
            self.compare_payload(left="nowhere", right="also-nowhere"),
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        status, events_doc = self.call(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(events_doc["events"]), 1)

    # ----------------------------------------------------------- 422/415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {k: v for k, v in valid.items() if k != "windowSize"},
            {k: v for k, v in valid.items() if k != "threshold"},
            # Neither a shared "type" field nor a suppression window has any
            # place on this entry point.
            {**valid, "type": EVENT_TYPE},
            {**valid, "suppressionWindow": 120},
            {**valid, "region": LEFT},
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
            b'{"organizationId":"org-1","left":"north",'
            b'"right":"south","windowSize":60,"threshold":3,'
            b'"left":"nowhere"}'
        )
        status, parsed = self.call(
            PATH, method="POST", raw_body=raw_body, token="w1"
        )
        self.assertEqual(status, 422)
        self.assertEqual(parsed["error"], "validation_error")

    def test_media_type_and_json_errors(self) -> None:
        self.add_event(event_body("n1", 5, region=LEFT))
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

    def test_only_post_is_served_and_no_region_sub_path_is_added(self) -> None:
        status, body = self.call(PATH)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, body = self.call(
            "/events/regions/north/compare/replay/decisions",
            method="POST",
            payload=self.compare_payload(),
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
