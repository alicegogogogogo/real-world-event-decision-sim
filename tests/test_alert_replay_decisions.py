"""Regression tests for GET /alerts/replay/decisions.

The baseline already had organization alert evaluation and the
organization window/peak replay, but no organization *alert* replay entry
point; this locks down the new read-only query that replays matching
events one at a time and simulates the organization alert
suppression/escalation rule within the replay:

- only the caller's organization's events whose ``type`` matches are
  replayed; other types and other organizations' data never enter the
  result, and an unknown organization (registered to the caller) matches
  zero events;
- matching events arrive ordered by ``occurredAt`` then ``eventId`` — the
  same order the region alert replay uses — and each step reports the
  event, the window rows accumulated so far (the exact aggregate window
  division, ranged empty windows kept), the peak (largest count, ties to
  the earliest start), and the action and alert identity the suppression
  rule produces within the replay: ``observe`` below the threshold, then
  ``escalate`` for the first threshold hit or one at/after the
  suppression distance, and ``suppress`` against the prior simulated
  alert with the new cumulative count inside the window;
- the simulation never reads or writes the alert store: simulated ids
  start at ``alert-1`` on every call regardless of committed alerts, the
  service-wide counter never advances, and identical requests are
  byte-for-byte identical without polluting the ledger, inventory, or
  alerts;
- both ``read`` and ``write`` credentials may call it; the verdict order
  is fixed — 401 (credential) before 422 (query shape) before 403
  (organization) — and restarting clears the ledger. The alert replay is
  main-only: no snapshot- or branch-prefixed path exists.
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
from tests import _support

ORG1 = "org-1"
ORG2 = "org-2"
EVENT_TYPE = "incident.created"

# Sentinel: a request with token=_UNSET uses the shared auto-credential
# helper, token=None sends no Authorization header at all.
_UNSET = object()


def make_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "eventId": "evt-1",
        "organizationId": ORG1,
        "type": EVENT_TYPE,
        "occurredAt": 100,
        "payload": {},
    }
    event.update(overrides)
    return event


class AlertReplayDecisionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.token_cache: dict[str, str] = {}
        self.register("w1", ORG1, "write")
        self.register("w2", ORG2, "write")
        self.register("r1", ORG1, "read")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    # -------------------------------------------------------------- low level

    def request_raw(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        token: Any = _UNSET,
    ) -> tuple[int, bytes, Any]:
        headers: dict[str, str] = (
            {"Content-Type": "application/json"} if body is not None else {}
        )
        if token is _UNSET:
            headers.update(
                _support.authorization_header(
                    self.token_cache, self.base_url, path, body
                )
            )
        elif token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=5) as response:
                raw = response.read()
                return response.status, raw, json.loads(raw)
        except HTTPError as error:
            raw = error.read()
            try:
                return error.code, raw, json.loads(raw)
            finally:
                error.close()

    def replay(
        self, query: str, *, token: Any = _UNSET
    ) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            f"/alerts/replay/decisions?{query}", token=token
        )
        return status, body

    def replay_raw(
        self, query: str, *, token: Any = _UNSET
    ) -> tuple[int, bytes]:
        status, raw, _ = self.request_raw(
            f"/alerts/replay/decisions?{query}", token=token
        )
        return status, raw

    def register(self, token: str, organization_id: str, role: str) -> None:
        status, _, _ = self.request_raw(
            "/auth/tokens",
            method="POST",
            body=json.dumps(
                {
                    "token": token,
                    "organizationId": organization_id,
                    "role": role,
                }
            ).encode(),
            token=None,
        )
        self.assertIn(status, (200, 201))

    # ------------------------------------------------------------- scaffolding

    def post_event(self, event: dict[str, Any]) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            "/events", method="POST", body=json.dumps(event).encode()
        )
        return status, body

    def post_json(self, path: str, payload: Any) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            path, method="POST", body=json.dumps(payload).encode()
        )
        return status, body

    def seed_events(self) -> None:
        events = [
            make_event(eventId="evt-b", occurredAt=200),
            make_event(eventId="evt-a", occurredAt=200),
            make_event(eventId="evt-c", occurredAt=100),
            make_event(eventId="evt-d", occurredAt=0),
            make_event(eventId="evt-e", occurredAt=300),
            # A different type never opens a step.
            make_event(eventId="evt-u1", type="incident.updated", occurredAt=10),
            # Another organization's events never enter.
            make_event(eventId="evt-x", organizationId=ORG2, occurredAt=1),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

    @staticmethod
    def query(
        *,
        org: str = ORG1,
        event_type: str = EVENT_TYPE,
        window_size: int = 60,
        threshold: int = 3,
        suppression_window: int = 120,
        from_to: tuple[int, int] | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "organizationId": org,
            "type": event_type,
            "windowSize": window_size,
            "threshold": threshold,
            "suppressionWindow": suppression_window,
        }
        if from_to is not None:
            params["from"], params["to"] = from_to
        return urlencode(params)

    # --------------------------------------------------------- response shape

    def test_steps_follow_replay_order_and_type_attribution(self) -> None:
        self.seed_events()
        status, body = self.replay(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["type"], EVENT_TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 3)
        self.assertEqual(body["suppressionWindow"], 120)
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
        serialized = json.dumps(body)
        for excluded in ("evt-u1", "evt-x"):
            self.assertNotIn(excluded, serialized)
        for step in body["steps"]:
            self.assertEqual(
                set(step),
                {
                    "eventId",
                    "occurredAt",
                    "windows",
                    "peakStart",
                    "peakCount",
                    "action",
                    "alertId",
                    "suppressedCount",
                },
            )

    def test_each_step_accumulates_window_counts(self) -> None:
        self.seed_events()
        status, body = self.replay(self.query())
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

    # ------------------------------------------------- suppression simulation

    def test_below_threshold_is_observe_with_null_identity(self) -> None:
        for index, timestamp in enumerate((10, 20)):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=f"evt-{index}", occurredAt=timestamp)
                )[0],
                201,
            )
        status, body = self.replay(self.query(threshold=3))
        self.assertEqual(status, 200)
        for step in body["steps"]:
            self.assertEqual(step["action"], "observe")
            self.assertIsNone(step["alertId"])
            self.assertIsNone(step["suppressedCount"])

    def test_first_threshold_hit_escalates_alert_one(self) -> None:
        for index, timestamp in enumerate((10, 20)):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=f"evt-{index}", occurredAt=timestamp)
                )[0],
                201,
            )
        status, body = self.replay(self.query(threshold=2, suppression_window=100))
        self.assertEqual(status, 200)
        first, second = body["steps"]
        self.assertEqual(first["action"], "observe")
        self.assertIsNone(first["alertId"])
        self.assertEqual(second["action"], "escalate")
        self.assertEqual(second["alertId"], "alert-1")
        self.assertEqual(second["suppressedCount"], 0)
        self.assertEqual(second["peakStart"], 0)
        self.assertEqual(second["peakCount"], 2)

    def test_full_lifecycle_escalate_suppress_then_escalate_again(self) -> None:
        # windowSize 60, threshold 2, suppressionWindow 60:
        #   10  observe          (window 0 count 1)
        #   20  escalate alert-1 (window 0 count 2, peak start 0)
        #   30  suppress alert-1 (count 3, same peak start)
        #   60  suppress alert-1 (window 60 count 1, peak still start 0)
        #   61  suppress alert-1 (counts 3 vs 2, tie keeps earliest start)
        #   62  suppress alert-1 (counts 3 vs 3, tie keeps earliest start)
        #   63  escalate alert-2 (window 60 count 4, distance exactly 60)
        #  130  suppress alert-2 (peak stays start 60, distance 0)
        events = (
            ("evt-a", 10),
            ("evt-b", 20),
            ("evt-c", 30),
            ("evt-d", 60),
            ("evt-e", 61),
            ("evt-f", 62),
            ("evt-g", 63),
            ("evt-h", 130),
        )
        for event_id, timestamp in events:
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=timestamp)
                )[0],
                201,
            )
        status, body = self.replay(
            self.query(threshold=2, suppression_window=60)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [
                (
                    step["action"],
                    step["alertId"],
                    step["suppressedCount"],
                    step["peakStart"],
                )
                for step in body["steps"]
            ],
            [
                ("observe", None, None, 0),
                ("escalate", "alert-1", 0, 0),
                ("suppress", "alert-1", 1, 0),
                ("suppress", "alert-1", 2, 0),
                ("suppress", "alert-1", 3, 0),
                ("suppress", "alert-1", 4, 0),
                ("escalate", "alert-2", 0, 60),
                ("suppress", "alert-2", 1, 60),
            ],
        )

    def test_threshold_one_escalates_then_suppresses_every_followup(self) -> None:
        for event_id, timestamp in (("evt-a", 0), ("evt-b", 10), ("evt-c", 20)):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=timestamp)
                )[0],
                201,
            )
        status, body = self.replay(
            self.query(threshold=1, suppression_window=60)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["action"], step["alertId"], step["suppressedCount"]) for step in body["steps"]],
            [
                ("escalate", "alert-1", 0),
                ("suppress", "alert-1", 1),
                ("suppress", "alert-1", 2),
            ],
        )

    def test_suppression_resets_count_under_the_new_alert(self) -> None:
        events = (
            ("evt-a", 0),
            ("evt-b", 1),
            ("evt-c", 60),
            ("evt-d", 61),
            ("evt-e", 62),
        )
        for event_id, timestamp in events:
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=timestamp)
                )[0],
                201,
            )
        status, body = self.replay(
            self.query(threshold=2, suppression_window=60)
        )
        self.assertEqual(status, 200)
        last = body["steps"][-1]
        self.assertEqual(last["action"], "escalate")
        self.assertEqual(last["alertId"], "alert-2")
        self.assertEqual(last["suppressedCount"], 0)

    # -------------------------------------------- independence from alert store

    def test_replay_ignores_committed_alerts_and_writes_none(self) -> None:
        for index, timestamp in enumerate((10, 20)):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=f"evt-{index}", occurredAt=timestamp)
                )[0],
                201,
            )
        evaluate_payload = {
            "organizationId": ORG1,
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 2,
            "suppressionWindow": 100,
        }
        status, first = self.post_json("/alerts/evaluate", evaluate_payload)
        self.assertEqual((status, first["action"], first["alertId"]), (200, "escalate", "alert-1"))
        status, second = self.post_json("/alerts/evaluate", evaluate_payload)
        self.assertEqual((status, second["action"], second["alertId"]), (200, "suppress", "alert-1"))

        # The replay's simulated sequence restarts at alert-1 regardless of
        # the committed alerts, and never sees their suppression state.
        status, body = self.replay(
            self.query(threshold=2, suppression_window=100)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["action"], step["alertId"]) for step in body["steps"]],
            [("observe", None), ("escalate", "alert-1")],
        )
        self.assertEqual(body["steps"][-1]["suppressedCount"], 0)

        # The committed alert is untouched, with the single suppression.
        status, _, listing = self.request_raw(f"/alerts?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["alerts"]), 1)
        self.assertEqual(listing["alerts"][0]["suppressedCount"], 1)

        # The service-wide counter is unaffected: another real evaluation
        # still suppresses against the same committed alert.
        status, after = self.post_json("/alerts/evaluate", evaluate_payload)
        self.assertEqual(status, 200)
        self.assertEqual(after["alertId"], "alert-1")
        self.assertEqual(after["suppressedCount"], 2)

        # And the replay is still unchanged after the committed writes.
        status, body = self.replay(
            self.query(threshold=2, suppression_window=100)
        )
        self.assertEqual(
            [(step["action"], step["alertId"]) for step in body["steps"]],
            [("observe", None), ("escalate", "alert-1")],
        )

    def test_organization_and_region_replays_share_one_order(self) -> None:
        # Events carrying a region are also organization events; the two
        # alert replays must list the same matching identifiers in the same
        # order.
        for event_id, timestamp in (
            ("evt-a", 20),
            ("evt-b", 20),
            ("evt-c", 10),
        ):
            self.assertEqual(
                self.post_event(
                    make_event(
                        eventId=event_id,
                        occurredAt=timestamp,
                        payload={"region": "north"},
                    )
                )[0],
                201,
            )
        status, org_body = self.replay(self.query(threshold=1))
        self.assertEqual(status, 200)
        region_query = (
            f"organizationId={ORG1}&region=north&type={EVENT_TYPE}"
            "&windowSize=60&threshold=1&suppressionWindow=60"
        )
        status, _, region_body = self.request_raw(
            f"/alerts/region/replay/decisions?{region_query}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in org_body["steps"]],
            [step["eventId"] for step in region_body["steps"]],
        )
        self.assertEqual(
            [
                (step["action"], step["alertId"])
                for step in org_body["steps"]
            ],
            [
                (step["action"], step["alertId"])
                for step in region_body["steps"]
            ],
        )

    def test_replay_is_read_only(self) -> None:
        self.seed_events()
        status, _ = self.replay(self.query())
        self.assertEqual(status, 200)
        self.replay(self.query(threshold=1))
        self.replay(self.query(from_to=(0, 180)))
        status, _, listing = self.request_raw(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)
        status, raw, _ = self.request_raw(f"/alerts?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertIn(b'"alerts":[]', raw)

    # ------------------------------------------------------------- attribution

    def test_unknown_type_or_foreign_organization_is_isolated(self) -> None:
        self.seed_events()
        status, body = self.replay(self.query(event_type="never.seen"))
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])
        self.assertNotIn("evt-x", json.dumps(body))
        status, body = self.replay(self.query(org=ORG2), token="w2")
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["evt-x"]
        )
        self.assertNotIn("evt-d", json.dumps(body))

    def test_no_matching_events_yields_empty_steps(self) -> None:
        status, body = self.replay(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])

    # ------------------------------------------------------------- range semantics

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.seed_events()
        status, body = self.replay(self.query(from_to=(0, 180)))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 180)
        grid = [(0, 60), (60, 120), (120, 180), (180, 240)]
        for step in body["steps"]:
            self.assertEqual(
                [(row["start"], row["end"]) for row in step["windows"]], grid
            )
        self.assertEqual(
            [row["count"] for row in body["steps"][-1]["windows"]],
            [1, 1, 0, 0],
        )
        # Out-of-range events still arrive as replay steps in order.
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["evt-d", "evt-c", "evt-a", "evt-b", "evt-e"],
        )

    def test_range_suppression_simulated_only_from_ranged_peaks(self) -> None:
        for event_id, timestamp in (("evt-a", 10), ("evt-b", 60), ("evt-c", 61)):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=timestamp)
                )[0],
                201,
            )
        status, body = self.replay(
            self.query(threshold=2, suppression_window=60, from_to=(60, 120))
        )
        self.assertEqual(status, 200)
        first, second, third = body["steps"]
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
        self.assertIsNone(first["alertId"])
        self.assertEqual(second["action"], "observe")
        self.assertEqual(
            (third["action"], third["alertId"], third["suppressedCount"]),
            ("escalate", "alert-1", 0),
        )
        self.assertEqual(third["peakStart"], 60)

    # --------------------------------------------------------- serialization

    def test_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_events()
        status, raw = self.replay_raw(self.query(threshold=2))
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        body = json.loads(raw)
        self.assertEqual(
            set(body),
            {
                "organizationId",
                "type",
                "windowSize",
                "threshold",
                "suppressionWindow",
                "from",
                "to",
                "steps",
            },
        )
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )
        self.assertIsInstance(body["windowSize"], int)
        self.assertIsInstance(body["threshold"], int)
        self.assertIsInstance(body["suppressionWindow"], int)
        self.assertIsInstance(body["steps"][0]["occurredAt"], int)

    def test_repeated_requests_are_byte_identical_and_isolated(self) -> None:
        self.seed_events()
        query = self.query(threshold=2)
        raws = [self.replay_raw(query)[1] for _ in range(3)]
        self.assertEqual(raws[0], raws[1])
        self.assertEqual(raws[1], raws[2])
        self.post_json(
            "/alerts/evaluate",
            {
                "organizationId": ORG1,
                "type": EVENT_TYPE,
                "windowSize": 60,
                "threshold": 2,
                "suppressionWindow": 10000,
            },
        )
        ranged = self.replay_raw(self.query(threshold=2, from_to=(0, 180)))[1]
        self.assertNotEqual(ranged, raws[0])
        self.assertEqual(self.replay_raw(query)[1], raws[0])

    def test_read_and_write_credentials_may_both_query(self) -> None:
        self.seed_events()
        query = self.query()
        read_status, read_body = self.replay(query, token="r1")
        write_status, write_body = self.replay(query, token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    # ----------------------------------------------------------------- 401 / 403

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.seed_events()
        path = f"/alerts/replay/decisions?{self.query()}"
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
            f"{self.base_url}/alerts/replay/decisions"
            "?windowSize=not-a-number",
            headers={"Authorization": "Bearer ghost-token"},
            method="GET",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 401)
        caught.exception.close()

    def test_query_shape_is_checked_before_organization(self) -> None:
        status, body = self.replay(
            "organizationId=&type=t&windowSize=x&threshold=1&suppressionWindow=10",
            token="w2",
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_foreign_organization_is_403_and_leaves_no_trace(self) -> None:
        self.seed_events()
        status, body = self.replay(self.query(), token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        status, _, listing = self.request_raw(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)

    # ---------------------------------------------------------------------- 422

    def test_each_required_parameter_must_appear_exactly_once(self) -> None:
        self.seed_events()
        base = self.query()
        for query in (
            "",
            "type=t&windowSize=60&threshold=3&suppressionWindow=120",
            "organizationId=org-1&windowSize=60&threshold=3&suppressionWindow=120",
            "organizationId=org-1&type=t&threshold=3&suppressionWindow=120",
            "organizationId=org-1&type=t&windowSize=60&suppressionWindow=120",
            "organizationId=org-1&type=t&windowSize=60&threshold=3",
            f"{base}&organizationId=org-1",
            f"{base}&type={EVENT_TYPE}",
            f"{base}&windowSize=60",
            f"{base}&threshold=3",
            f"{base}&suppressionWindow=120",
        ):
            with self.subTest(query=query):
                status, body = self.replay(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_values_are_422(self) -> None:
        self.seed_events()
        head = f"organizationId=org-1&type={EVENT_TYPE}"
        for query in (
            "organizationId=&type=t&windowSize=60&threshold=3&suppressionWindow=120",
            "organizationId=%20&type=t&windowSize=60&threshold=3&suppressionWindow=120",
            "organizationId=org-1&type=&windowSize=60&threshold=3&suppressionWindow=120",
            "organizationId=org-1&type=%20&windowSize=60&threshold=3&suppressionWindow=120",
            f"{head}&windowSize=&threshold=3&suppressionWindow=120",
            f"{head}&windowSize=%20&threshold=3&suppressionWindow=120",
            f"{head}&windowSize=60&threshold=&suppressionWindow=120",
            f"{head}&windowSize=60&threshold=%20&suppressionWindow=120",
            f"{head}&windowSize=60&threshold=3&suppressionWindow=",
            f"{head}&windowSize=60&threshold=3&suppressionWindow=%20",
        ):
            with self.subTest(query=query):
                status, body = self.replay(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_positive_integer_fields_must_be_positive_integers(self) -> None:
        self.seed_events()
        for bad in ("0", "-1", "1.5", "abc", "1e3", "+5", "true"):
            for name, others in (
                ("windowSize", "threshold=3&suppressionWindow=120"),
                ("threshold", "windowSize=60&suppressionWindow=120"),
                ("suppressionWindow", "windowSize=60&threshold=3"),
            ):
                query = (
                    "organizationId=org-1"
                    f"&type={EVENT_TYPE}&{name}={bad}&{others}"
                )
                with self.subTest(name=name, bad=bad):
                    status, body = self.replay(query)
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_range_must_be_paired_non_negative_and_ordered(self) -> None:
        self.seed_events()
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
                status, body = self.replay(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_from_equal_to_to_is_legal(self) -> None:
        self.post_event(make_event(eventId="evt-a", occurredAt=30))
        status, body = self.replay(self.query(from_to=(30, 30)))
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(
            body["steps"][0]["windows"],
            [{"start": 0, "end": 60, "count": 1}],
        )

    def test_validation_failure_writes_nothing(self) -> None:
        self.seed_events()
        status, _ = self.replay(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=x&threshold=3&suppressionWindow=120"
        )
        self.assertEqual(status, 422)
        status, _, listing = self.request_raw(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)

    # -------------------------------------------------- main-only / no prefixes

    def test_no_snapshot_or_branch_prefixed_alert_replay_path(self) -> None:
        self.post_event(make_event())
        status, _, _ = self.request_raw(
            "/snapshots",
            method="POST",
            body=b'{"snapshotId":"snap-1"}',
        )
        self.assertEqual(status, 201)
        status, _, _ = self.request_raw(
            "/branches",
            method="POST",
            body=b'{"branchId":"br-1","snapshotId":"snap-1"}',
        )
        self.assertEqual(status, 201)
        status, _, body = self.request_raw(
            f"/snapshots/snap-1/alerts/replay/decisions?{self.query()}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, _, body = self.request_raw(
            f"/branches/br-1/alerts/replay/decisions?{self.query()}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    # ----------------------------------------------------------------- restart

    def test_replay_on_a_fresh_server_has_no_steps(self) -> None:
        self.seed_events()
        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            fresh_base = f"http://127.0.0.1:{fresh.server_port}"
            token = _support.ensure_token({}, fresh_base, ORG1)
            query = self.query()
            request = Request(
                f"{fresh_base}/alerts/replay/decisions?{query}",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(request, timeout=2) as response:
                body = json.load(response)
            self.assertEqual(body["steps"], [])
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
