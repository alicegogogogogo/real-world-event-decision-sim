"""Regression tests for POST /alerts/regions/compare/replay/decisions.

The baseline already had the region alert replay
(``GET /alerts/region/replay/decisions``) but no aligned comparison of two
regions' replays. This locks down the new read-only entry point:

- each side simulates the published region alert replay over the caller's
  organization's events attributed to its own verbatim region and type,
  ordered by occurredAt then eventId, with the same window, peak, range,
  threshold, and suppression contract;
- the two replays align by eventId: a shared identifier is one shared
  step, an identifier on only one side is its own step on which the other
  side's prefix and suppression state are neither cleared nor advanced,
  and the aligned steps run by occurred time then identifier, reporting
  the earlier time when the sides disagree on a shared identifier;
- every step carries the event id/time, window rows aligned by start
  (ranged empty windows kept, a missing side counts zero), each side's
  peak start/count/action/alert identity, and an ``equal`` marker that
  looks only at peak start, peak count and action;
- simulated alert identifiers are numbered independently on each side
  from alert-1; the alert store is never read or written and the global
  counter never advances, so identical submissions are byte-for-byte
  identical and nothing is implicitly created;
- the same region name on both sides is legal and equal by construction;
  an unknown region matches zero events;
- the response is compact key-sorted JSON with integer values and one
  trailing newline; 401 / 415 / 400 / 422 / 403 follow the fixed
  ordering and both read and write credentials may call it.

The event ledger keys events globally by eventId and one event carries
exactly one payload region, so a shared identifier across two *distinct*
regions cannot be built over HTTP (the second commit conflicts); those
alignment shapes are exercised on the pure merge function directly, the
same way the branch replay comparison tests do. Over HTTP, shared
identifier steps occur when both sides name the same region.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import compare_region_alert_replay_decisions
from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"
LEFT = "north"
RIGHT = "south"
EVENT_TYPE = "incident.created"
PATH = "/alerts/regions/compare/replay/decisions"


def event_body(
    event_id: str,
    occurred_at: int,
    *,
    region: str = LEFT,
    organization_id: str = ORG1,
    event_type: str = EVENT_TYPE,
) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": event_type,
        "occurredAt": occurred_at,
        "payload": {"region": region},
    }


def unattributed_event(
    event_id: str, occurred_at: int, payload: Any
) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "organizationId": ORG1,
        "type": EVENT_TYPE,
        "occurredAt": occurred_at,
        "payload": payload,
    }


class RegionAlertCompareReplayDecisionsTest(unittest.TestCase):
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
            "type": EVENT_TYPE,
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

    def compare_raw(
        self, payload: dict[str, Any] | None = None, *, token: str | None = "w1"
    ) -> tuple[int, bytes]:
        return self.raw(
            PATH,
            method="POST",
            body=json.dumps(
                self.compare_payload() if payload is None else payload
            ).encode(),
            token=token,
        )

    @staticmethod
    def row(start: int, left_count: int, right_count: int) -> dict[str, Any]:
        return {"leftCount": left_count, "rightCount": right_count, "start": start}

    @staticmethod
    def side(
        peak_start: Any,
        peak_count: int,
        action: str,
        alert_id: Any = None,
        suppressed_count: Any = None,
    ) -> dict[str, Any]:
        return {
            "action": action,
            "alertId": alert_id,
            "peakCount": peak_count,
            "peakStart": peak_start,
            "suppressedCount": suppressed_count,
        }

    def seed_distinct_region_events(self) -> None:
        # North: n1@5, n2@10, n3@65. South: s1@12, s2@70. All identifiers
        # distinct (the ledger keys an event globally, so one id carries one
        # region); aligned order is n1, n2, s1, n3, s2.
        for event in (
            event_body("n1", 5, region=LEFT),
            event_body("n2", 10, region=LEFT),
            event_body("n3", 65, region=LEFT),
            event_body("s1", 12, region=RIGHT),
            event_body("s2", 70, region=RIGHT),
        ):
            self.add_event(event)

    # ------------------------------------------------------------- happy paths

    def test_aligned_steps_run_by_time_then_identifier(self) -> None:
        self.add_event(event_body("n-b", 100, region=LEFT))
        self.add_event(event_body("n-a", 100, region=LEFT))
        self.add_event(event_body("s-c", 50, region=RIGHT))

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("s-c", 50), ("n-a", 100), ("n-b", 100)],
        )

    def test_one_sided_identifier_freezes_other_side_without_suppressing(
        self,
    ) -> None:
        # threshold 2, suppression window 100.
        # left:  la@5, lb@10, lc@65
        # right: ra@15, rb@70, rc@75
        for event in (
            event_body("la", 5, region=LEFT),
            event_body("lb", 10, region=LEFT),
            event_body("lc", 65, region=LEFT),
            event_body("ra", 15, region=RIGHT),
            event_body("rb", 70, region=RIGHT),
            event_body("rc", 75, region=RIGHT),
        ):
            self.add_event(event)

        status, body = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=100)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["la", "lb", "ra", "lc", "rb", "rc"],
        )
        la, lb, ra, lc, rb, rc = body["steps"]

        # la@5: left takes its first step; the right prefix is empty and
        # reports the empty-prefix decision.
        self.assertEqual(la["left"], self.side(0, 1, "observe"))
        self.assertEqual(la["right"], self.side(None, 0, "observe"))
        self.assertEqual(la["windows"], [self.row(0, 1, 0)])

        # lb@10: the left side opens its first simulated alert; the frozen
        # right side stays at the empty-prefix decision and does not advance.
        self.assertEqual(lb["left"], self.side(0, 2, "escalate", "alert-1", 0))
        self.assertEqual(lb["right"], self.side(None, 0, "observe"))
        self.assertFalse(lb["equal"])

        # ra@15: left frozen at its escalate state; right takes its first.
        self.assertEqual(ra["left"], self.side(0, 2, "escalate", "alert-1", 0))
        self.assertEqual(ra["right"], self.side(0, 1, "observe"))

        # lc@65: left reaches window 60 but the peak tie keeps start 0, so it
        # suppresses against its own alert-1; the frozen right side must NOT
        # fabricate a suppression and stays observe with one event.
        self.assertEqual(lc["left"], self.side(0, 2, "suppress", "alert-1", 1))
        self.assertEqual(lc["right"], self.side(0, 1, "observe"))
        self.assertEqual(
            lc["windows"], [self.row(0, 2, 1), self.row(60, 1, 0)]
        )

        # rb@70: left frozen with suppressedCount 1; right still below the
        # threshold, so it observes.
        self.assertEqual(rb["left"], self.side(0, 2, "suppress", "alert-1", 1))
        self.assertEqual(rb["right"], self.side(0, 1, "observe"))

        # rc@75: the right side reaches count 2 in window 60 and opens its
        # own alert-1 (independent numbering); the left side remains frozen.
        self.assertEqual(rc["left"], self.side(0, 2, "suppress", "alert-1", 1))
        self.assertEqual(rc["right"], self.side(60, 2, "escalate", "alert-1", 0))
        self.assertFalse(rc["equal"])
        self.assertEqual(
            rc["windows"], [self.row(0, 2, 1), self.row(60, 1, 2)]
        )

    def test_simulated_identifiers_are_independent_per_side(self) -> None:
        # Each side opens its own alert-1 from its independent sequence; a
        # frozen side never re-decides or advances its counter.
        self.add_event(event_body("l1", 5, region=LEFT))
        self.add_event(event_body("l2", 10, region=LEFT))
        self.add_event(event_body("r1", 15, region=RIGHT))
        self.add_event(event_body("r2", 20, region=RIGHT))

        status, body = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=1000)
        )
        self.assertEqual(status, 200)
        l1, l2, r1, r2 = body["steps"]
        self.assertEqual(l1["left"], self.side(0, 1, "observe"))
        self.assertEqual(l1["right"], self.side(None, 0, "observe"))
        self.assertEqual(l2["left"], self.side(0, 2, "escalate", "alert-1", 0))
        self.assertEqual(l2["right"], self.side(None, 0, "observe"))
        # The frozen left keeps its escalate result on the right-only steps.
        self.assertEqual(r1["left"], self.side(0, 2, "escalate", "alert-1", 0))
        self.assertEqual(r1["right"], self.side(0, 1, "observe"))
        # Right opens its own alert-1, numbered from the beginning on its side.
        self.assertEqual(r2["left"], self.side(0, 2, "escalate", "alert-1", 0))
        self.assertEqual(r2["right"], self.side(0, 2, "escalate", "alert-1", 0))
        self.assertTrue(r2["equal"])

    def test_window_rows_align_union_with_missing_side_zero(self) -> None:
        self.add_event(event_body("l1", 5, region=LEFT))
        self.add_event(event_body("l2", 10, region=LEFT))
        self.add_event(event_body("r1", 125, region=RIGHT))

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["steps"][-1]["windows"],
            [self.row(0, 2, 0), self.row(120, 0, 1)],
        )

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.add_event(event_body("l1", 5, region=LEFT))
        self.add_event(event_body("l2", 65, region=LEFT))  # outside [0, 60]
        self.add_event(event_body("r1", 120, region=RIGHT))  # outside [0, 60]

        status, body = self.compare(
            self.compare_payload(**{"from": 0, "to": 60})
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 60)
        # Out-of-range events still arrive as steps in time order.
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
            first["windows"], [self.row(0, 1, 0), self.row(60, 0, 0)]
        )
        # l2 at 65 is outside the range and counts on neither window.
        self.assertEqual(
            second["windows"], [self.row(0, 1, 0), self.row(60, 0, 0)]
        )
        # r1 at 120 is outside the range as well.
        self.assertEqual(
            third["windows"], [self.row(0, 1, 0), self.row(60, 0, 0)]
        )

    def test_range_suppression_simulated_only_from_ranged_peaks(self) -> None:
        # Left-only stream: an out-of-range event at 10 then two in-range
        # events at 60/61; the right prefix is empty throughout.
        self.add_event(event_body("la", 10, region=LEFT))
        self.add_event(event_body("lb", 60, region=LEFT))
        self.add_event(event_body("lc", 61, region=LEFT))

        status, body = self.compare(
            self.compare_payload(
                threshold=2, suppressionWindow=60, **{"from": 60, "to": 120}
            )
        )
        self.assertEqual(status, 200)
        first, second, third = body["steps"]
        # The out-of-range event counts in no ranged window: zero peak,
        # observe, no alert.
        self.assertEqual(
            first["windows"],
            [self.row(60, 0, 0), self.row(120, 0, 0)],
        )
        self.assertEqual(first["left"], self.side(None, 0, "observe"))
        self.assertEqual(first["right"], self.side(None, 0, "observe"))
        self.assertEqual(second["left"]["action"], "observe")
        self.assertEqual(
            third["left"], self.side(60, 2, "escalate", "alert-1", 0)
        )
        # The frozen empty right side never changes.
        for step in body["steps"]:
            self.assertEqual(step["right"], self.side(None, 0, "observe"))

    def test_same_region_on_both_sides_is_legal_and_equal(self) -> None:
        self.seed_distinct_region_events()
        status, body = self.compare(self.compare_payload(right=LEFT))
        self.assertEqual(status, 200)
        self.assertEqual(body["left"], LEFT)
        self.assertEqual(body["right"], LEFT)
        # Only the north events replay; the south events never enter, and
        # every shared identifier is one step equal on both sides.
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("n1", 5), ("n2", 10), ("n3", 65)],
        )
        for step in body["steps"]:
            self.assertTrue(step["equal"])
            self.assertEqual(step["left"], step["right"])
            self.assertTrue(
                all(row["leftCount"] == row["rightCount"] for row in step["windows"])
            )

    def test_unknown_region_matches_zero_events(self) -> None:
        self.seed_distinct_region_events()
        status, body = self.compare(self.compare_payload(left="nowhere"))
        self.assertEqual(status, 200)
        # Only south events produce steps; the left side stays empty/frozen.
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("s1", 12), ("s2", 70)],
        )
        for step in body["steps"]:
            self.assertEqual(step["left"], self.side(None, 0, "observe"))
            self.assertFalse(step["equal"])

        status, body = self.compare(
            self.compare_payload(left="nowhere", right="also-nowhere")
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])

    def test_other_regions_types_and_organizations_never_enter(self) -> None:
        self.seed_distinct_region_events()
        # East region never enters either side.
        self.add_event(event_body("e1", 5, region="east"))
        # A matching region but different type never opens a step.
        self.add_event(
            event_body("u1", 6, region=LEFT, event_type="incident.updated")
        )
        # No region attribution never matches.
        self.add_event(unattributed_event("z1", 7, {}))
        self.add_event(unattributed_event("z2", 8, {"region": ""}))
        self.add_event(unattributed_event("z3", 9, {"region": 7}))
        # Another organization's same-named regions never enter.
        self.add_event(
            event_body("x1", 8, region=LEFT, organization_id=ORG2),
            token="w2",
        )
        status, body = self.compare()
        self.assertEqual(status, 200)
        serialized = json.dumps(body)
        for excluded in ("e1", "u1", "z1", "z2", "z3", "x1"):
            self.assertNotIn(excluded, serialized)

    def test_neither_side_holding_matching_events_yields_no_steps(self) -> None:
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])

    def test_response_echoes_all_request_parameters(self) -> None:
        self.add_event(event_body("l1", 5, region=LEFT))
        payload = self.compare_payload(
            threshold=4,
            suppressionWindow=90,
            type=EVENT_TYPE,
            **{"from": 0, "to": 120},
        )
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["left"], LEFT)
        self.assertEqual(body["right"], RIGHT)
        self.assertEqual(body["type"], EVENT_TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 4)
        self.assertEqual(body["suppressionWindow"], 90)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 120)

    def test_equal_marker_looks_only_at_peak_start_count_action(self) -> None:
        # Distinct identifiers across regions, so the final aligned step has
        # both prefixes fully accumulated with the same peak start/count/
        # action but different simulated alert identities:
        # left:  a@0 (esc L1@0), b@60 (suppress L1, peak tie keeps start 0),
        #        c@61 (window 60 count 2 moves the peak, esc L2@60),
        #        z-l@62 (suppress L2 #1)
        # right: p@60 (esc R1@60), q@61 (suppress R1 #1),
        #        z-r@62 (suppress R1 #2)
        for event in (
            event_body("a", 0, region=LEFT),
            event_body("b", 60, region=LEFT),
            event_body("c", 61, region=LEFT),
            event_body("p", 60, region=RIGHT),
            event_body("q", 61, region=RIGHT),
            event_body("z-l", 62, region=LEFT),
            event_body("z-r", 62, region=RIGHT),
        ):
            self.add_event(event)

        status, body = self.compare(
            self.compare_payload(threshold=1, suppressionWindow=60)
        )
        self.assertEqual(status, 200)
        final = body["steps"][-1]
        self.assertEqual(final["eventId"], "z-r")
        # On this final right-only step the left side is frozen at its
        # z-l result; both sides nevertheless agree on the three compared
        # fields despite different alert identities.
        self.assertEqual(final["left"]["peakStart"], 60)
        self.assertEqual(final["right"]["peakStart"], 60)
        self.assertEqual(final["left"]["peakCount"], 3)
        self.assertEqual(final["right"]["peakCount"], 3)
        self.assertEqual(final["left"]["action"], "suppress")
        self.assertEqual(final["right"]["action"], "suppress")
        self.assertEqual(final["left"]["alertId"], "alert-2")
        self.assertEqual(final["right"]["alertId"], "alert-1")
        self.assertTrue(final["equal"])
        self.assertEqual(
            final["windows"], [self.row(0, 1, 0), self.row(60, 3, 3)]
        )

    # --------------------------------------- pure-function cross-region shapes

    @staticmethod
    def _pure_event(event_id: str, occurred_at: int) -> dict[str, Any]:
        return {
            "eventId": event_id,
            "organizationId": ORG1,
            "type": EVENT_TYPE,
            "occurredAt": occurred_at,
            "payload": {},
        }

    def _pure_params(self, **overrides: Any) -> dict[str, Any]:
        # Normalize through the request validator so the optional from/to
        # keys are present exactly as the handler supplies them.
        from event_sim.server import validate_region_alert_replay_compare_request

        return validate_region_alert_replay_compare_request(
            self.compare_payload(**overrides)
        )

    def test_pure_shared_identifier_is_one_step_accumulating_both_sides(self) -> None:
        params = self._pure_params(threshold=1, suppressionWindow=1000)
        result = compare_region_alert_replay_decisions(
            [self._pure_event("dup", 5)],
            [self._pure_event("dup", 5)],
            params,
        )
        self.assertEqual(len(result["steps"]), 1)
        step = result["steps"][0]
        self.assertEqual(step["eventId"], "dup")
        self.assertEqual(step["occurredAt"], 5)
        self.assertEqual(step["windows"], [self.row(0, 1, 1)])
        self.assertEqual(
            step["left"], self.side(0, 1, "escalate", "alert-1", 0)
        )
        self.assertEqual(
            step["right"], self.side(0, 1, "escalate", "alert-1", 0)
        )
        self.assertTrue(step["equal"])

    def test_pure_shared_identifier_disagreement_on_time_reports_earlier(
        self,
    ) -> None:
        params = self._pure_params(left="L", right="R")
        result = compare_region_alert_replay_decisions(
            [self._pure_event("dup", 100), self._pure_event("late-l", 150)],
            [self._pure_event("dup", 40), self._pure_event("early-r", 10)],
            params,
        )
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in result["steps"]],
            [("early-r", 10), ("dup", 40), ("late-l", 150)],
        )
        dup = result["steps"][1]
        # Right holds early-r and its own dup in window 0; left holds only
        # its dup in window 60.
        self.assertEqual(
            dup["windows"],
            [self.row(0, 0, 2), self.row(60, 1, 0)],
        )

    def test_pure_per_side_matches_single_region_replay(self) -> None:
        from event_sim.server import alert_replay_steps

        params = self._pure_params(threshold=2, suppressionWindow=60)
        left = [
            self._pure_event("l1", 5),
            self._pure_event("l2", 60),
            self._pure_event("l3", 63),
        ]
        right = [
            self._pure_event("r1", 10),
            self._pure_event("r2", 70),
        ]
        result = compare_region_alert_replay_decisions(left, right, params)

        left_single = {
            (step["eventId"], step["occurredAt"]): step
            for step in alert_replay_steps(left, params)
        }
        right_single = {
            (step["eventId"], step["occurredAt"]): step
            for step in alert_replay_steps(right, params)
        }
        for step in result["steps"]:
            key = (step["eventId"], step["occurredAt"])
            if key in left_single:
                own = left_single[key]
                self.assertEqual(
                    (
                        step["left"]["peakStart"],
                        step["left"]["peakCount"],
                        step["left"]["action"],
                        step["left"]["alertId"],
                        step["left"]["suppressedCount"],
                    ),
                    (
                        own["peakStart"],
                        own["peakCount"],
                        own["action"],
                        own["alertId"],
                        own["suppressedCount"],
                    ),
                )
            if key in right_single:
                own = right_single[key]
                self.assertEqual(
                    (
                        step["right"]["peakStart"],
                        step["right"]["peakCount"],
                        step["right"]["action"],
                        step["right"]["alertId"],
                        step["right"]["suppressedCount"],
                    ),
                    (
                        own["peakStart"],
                        own["peakCount"],
                        own["action"],
                        own["alertId"],
                        own["suppressedCount"],
                    ),
                )

    # ----------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(self) -> None:
        self.seed_distinct_region_events()
        status, raw = self.compare_raw(
            self.compare_payload(threshold=2, suppressionWindow=100)
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        text = raw.decode()
        self.assertIn('"leftCount":1', text)
        self.assertIn('"rightCount":0', text)

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
                "type",
                "windowSize",
            ],
        )
        self.assertEqual(
            list(body["steps"][0]),
            ["equal", "eventId", "left", "occurredAt", "right", "windows"],
        )
        self.assertEqual(
            list(body["steps"][0]["left"]),
            ["action", "alertId", "peakCount", "peakStart", "suppressedCount"],
        )
        self.assertEqual(
            list(body["steps"][0]["windows"][0]),
            ["leftCount", "rightCount", "start"],
        )
        # Window rows carry no per-row equality marker.
        self.assertNotIn("equal", body["steps"][0]["windows"][0])
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )
        self.assertIsInstance(body["windowSize"], int)
        self.assertIsInstance(body["threshold"], int)
        self.assertIsInstance(body["suppressionWindow"], int)
        self.assertIsInstance(body["steps"][0]["occurredAt"], int)
        self.assertIsInstance(body["steps"][0]["windows"][0]["leftCount"], int)

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.seed_distinct_region_events()
        payload = self.compare_payload(
            threshold=2, suppressionWindow=100, **{"from": 0, "to": 180}
        )
        raws = [self.compare_raw(payload)[1] for _ in range(3)]
        self.assertEqual(len(set(raws)), 1)

    # -------------------------------------------------------------- read-only

    def test_comparison_ignores_committed_alerts_and_writes_none(self) -> None:
        self.add_event(event_body("n1", 10, region=LEFT))
        self.add_event(event_body("n2", 20, region=LEFT))
        evaluate_payload = {
            "organizationId": ORG1,
            "region": LEFT,
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 2,
            "suppressionWindow": 10000,
        }
        status, committed = self.call(
            "/alerts/region/evaluate",
            method="POST",
            payload=evaluate_payload,
        )
        self.assertEqual(
            (status, committed["action"], committed["alertId"]),
            (200, "escalate", "alert-1"),
        )

        # The comparison's simulated ids restart at alert-1 on each side
        # regardless of the committed alert.
        status, body = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=10000)
        )
        self.assertEqual(status, 200)
        n1, n2 = body["steps"]
        self.assertEqual(n1["left"]["action"], "observe")
        self.assertEqual(n2["left"]["alertId"], "alert-1")
        self.assertEqual(n2["left"]["suppressedCount"], 0)

        # The committed region alert is untouched.
        status, listing = self.call(
            f"/alerts/region?organizationId={ORG1}&region={LEFT}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["alerts"]), 1)
        self.assertEqual(listing["alerts"][0]["suppressedCount"], 0)

        # The service-wide counter is unaffected: a real organization alert
        # takes the next global id.
        status, org_alert = self.call(
            "/alerts/evaluate",
            method="POST",
            payload={
                "organizationId": ORG1,
                "type": EVENT_TYPE,
                "windowSize": 60,
                "threshold": 1,
                "suppressionWindow": 10000,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(org_alert["alertId"], "alert-2")

        # And the comparison is unchanged after the committed writes.
        status, again = self.compare(
            self.compare_payload(threshold=2, suppressionWindow=10000)
        )
        self.assertEqual(status, 200)
        self.assertEqual(again, body)

    def test_comparison_is_read_only(self) -> None:
        self.seed_distinct_region_events()
        for _ in range(3):
            status, _ = self.compare(
                self.compare_payload(**{"from": 0, "to": 180})
            )
            self.assertEqual(status, 200)
        status, events = self.call(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(events["events"]), 5)
        status, alerts = self.call(f"/alerts?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])
        status, region_alerts = self.call(
            f"/alerts/region?organizationId={ORG1}&region={LEFT}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(region_alerts["alerts"], [])

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_compare(self) -> None:
        self.seed_distinct_region_events()
        read_status, read_body = self.compare(token="r1")
        write_status, write_body = self.compare(token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
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

    def test_credential_checked_before_media_type_and_body(self) -> None:
        # No credential with a wrong media type is still 401.
        status, raw = self.raw(
            PATH,
            method="POST",
            body=b"not json at all",
            token=None,
            content_type="text/plain",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

        # An unregistered token with syntactically invalid JSON is 401 too.
        status, raw = self.raw(
            PATH,
            method="POST",
            body=b'{"left": ',
            token="ghost-token",
            content_type="application/json",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_foreign_organization_is_403(self) -> None:
        self.seed_distinct_region_events()
        status, body = self.compare(token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        # The event ledger is untouched.
        status, events = self.call(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(events["events"]), 5)

    def test_organization_checked_after_body_validation(self) -> None:
        # A foreign credential with a malformed body gets 422, not 403.
        status, body = self.call(
            PATH,
            method="POST",
            raw_body=b'{"organizationId":"org-1"}',
            token="w2",
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_organization_two_kept_isolated(self) -> None:
        self.seed_distinct_region_events()
        self.add_event(
            event_body("f1", 5, region=LEFT, organization_id=ORG2),
            token="w2",
        )
        self.add_event(
            event_body("f2", 65, region=RIGHT, organization_id=ORG2),
            token="w2",
        )
        payload = {
            "organizationId": ORG2,
            "left": LEFT,
            "right": RIGHT,
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 1,
            "suppressionWindow": 1000,
        }
        status, body = self.call(PATH, method="POST", payload=payload, token="w2")
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("f1", 5), ("f2", 65)],
        )
        self.assertEqual(
            body["steps"][0]["windows"], [self.row(0, 1, 0)]
        )

    # ----------------------------------------------------------- 422 / 415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {k: v for k, v in valid.items() if k != "type"},
            {k: v for k, v in valid.items() if k != "windowSize"},
            {k: v for k, v in valid.items() if k != "threshold"},
            {k: v for k, v in valid.items() if k != "suppressionWindow"},
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
            {**valid, "suppressionWindow": 0},
            {**valid, "suppressionWindow": True},
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

        # Nothing was written by the failed requests.
        status, events = self.call(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(events["events"], [])

    def test_duplicate_json_keys_are_422_validation_error(self) -> None:
        raw_body = (
            b'{"organizationId":"org-1","left":"north","right":"south",'
            b'"type":"incident.created","windowSize":60,"threshold":3,'
            b'"suppressionWindow":120,"left":"ghost"}'
        )
        status, parsed = self.call(
            PATH, method="POST", raw_body=raw_body, token="w1"
        )
        self.assertEqual(status, 422)
        self.assertEqual(parsed["error"], "validation_error")

    def test_media_type_and_json_errors(self) -> None:
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

    def test_get_is_not_served(self) -> None:
        status, body = self.call(PATH)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
