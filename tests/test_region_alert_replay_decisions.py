"""Regression tests for GET /alerts/region/replay/decisions.

The baseline already had region alert evaluation, the region alert
listing, and the step-by-step region replay-decision query, but no
read-only replay of the region alert suppression rule; this locks down
the new entry point:

- only the caller's organization's events whose payload ``region`` is a
  non-empty string equal to the requested region and whose ``type``
  matches are replayed; unattributed events, other regions/types, and
  other organizations' data never enter the result;
- matching events arrive ordered by ``occurredAt`` then ``eventId``; each
  step reports the event, the window rows accumulated so far (the exact
  main-aggregate window division, ranged empty windows kept), the peak
  decision, and the action the region alert rule would take plus its
  alert identity — ``observe`` below the threshold, ``escalate`` opening
  replay-local ``alert-N`` at/after the suppression window, and
  ``suppress`` inside it with the prior id and a new running count;
- the replay never reads or writes the live alert ledger: stored alerts
  do not influence it, it never creates one, and simulated identifiers
  are numbered independently in replay order;
- the response echoes organization, region, type, window width,
  threshold, suppression window and range, is compact key-sorted JSON
  with integer values and one trailing newline, and identical requests
  are byte-for-byte identical;
- both ``read`` and ``write`` credentials may call it; the verdict order
  is fixed — 401 (credential) before 422 (query shape) before 403
  (organization) — nothing is ever written, and restarting clears the
  ledger and alerts. The replay is main-only: no branch-prefixed path
  exists.
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
REGION = "north"
EVENT_TYPE = "incident.created"

_UNSET = object()


def make_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "eventId": "evt-1",
        "organizationId": ORG1,
        "type": EVENT_TYPE,
        "occurredAt": 100,
        "payload": {"region": REGION},
    }
    event.update(overrides)
    return event


class RegionAlertReplayDecisionsTest(unittest.TestCase):
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

    def decisions(
        self, query: str, *, token: Any = _UNSET
    ) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            f"/alerts/region/replay/decisions?{query}", token=token
        )
        return status, body

    def decisions_raw(
        self, query: str, *, token: Any = _UNSET
    ) -> tuple[int, bytes]:
        status, raw, _ = self.request_raw(
            f"/alerts/region/replay/decisions?{query}", token=token
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

    def add(self, event_id: str, occurred_at: int) -> None:
        status = self.post_event(
            make_event(eventId=event_id, occurredAt=occurred_at)
        )[0]
        self.assertEqual(status, 201)

    @staticmethod
    def query(
        *,
        org: str = ORG1,
        region: str = REGION,
        event_type: str = EVENT_TYPE,
        window_size: int = 60,
        threshold: int = 3,
        suppression_window: int = 100,
        from_to: tuple[int, int] | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "organizationId": org,
            "region": region,
            "type": event_type,
            "windowSize": window_size,
            "threshold": threshold,
            "suppressionWindow": suppression_window,
        }
        if from_to is not None:
            params["from"], params["to"] = from_to
        return urlencode(params)

    def seed_suppression_events(self) -> None:
        # windowSize 60: window 0 holds evt-a/evt-b (2), window 60 holds
        # evt-c/evt-d (2), window 180 holds evt-e..evt-h (4).
        for event_id, timestamp in (
            ("evt-a", 0),
            ("evt-b", 10),
            ("evt-c", 60),
            ("evt-d", 61),
            ("evt-e", 200),
            ("evt-f", 201),
            ("evt-g", 202),
            ("evt-h", 203),
        ):
            self.add(event_id, timestamp)

    # ------------------------------------------------------------ replay order

    def test_steps_follow_replay_order_and_region_attribution(self) -> None:
        events = [
            make_event(eventId="evt-b", occurredAt=200),
            make_event(eventId="evt-a", occurredAt=200),
            make_event(eventId="evt-c", occurredAt=100),
            make_event(eventId="evt-s1", occurredAt=5, payload={"region": "south"}),
            make_event(
                eventId="evt-u1",
                type="incident.updated",
                occurredAt=10,
            ),
            make_event(eventId="evt-z1", occurredAt=11, payload={"region": ""}),
            make_event(eventId="evt-z2", occurredAt=12, payload={}),
            make_event(eventId="evt-z3", occurredAt=13, payload={"region": 7}),
            make_event(
                eventId="evt-x",
                organizationId=ORG2,
                occurredAt=1,
            ),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)
        status, body = self.decisions(self.query(threshold=5))
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("evt-c", 100), ("evt-a", 200), ("evt-b", 200)],
        )
        serialized = json.dumps(body)
        for excluded in (
            "evt-s1",
            "evt-u1",
            "evt-z1",
            "evt-z2",
            "evt-z3",
            "evt-x",
        ):
            self.assertNotIn(excluded, serialized)

    def test_unknown_region_type_or_organization_has_no_steps(self) -> None:
        self.seed_suppression_events()
        for kwargs in (
            {"region": "nowhere"},
            {"event_type": "never.seen"},
            {"org": "org-x"},
        ):
            with self.subTest(kwargs=kwargs):
                status, body = self.decisions(self.query(**kwargs))
                self.assertEqual(status, 200)
                self.assertEqual(body["steps"], [])

    def test_region_matching_is_verbatim(self) -> None:
        self.assertEqual(
            self.post_event(
                make_event(eventId="evt-1", payload={"region": " North "})
            )[0],
            201,
        )
        self.assertEqual(
            self.post_event(
                make_event(eventId="evt-2", payload={"region": "nOrth"})
            )[0],
            201,
        )
        for region, expected in (
            (" North ", ["evt-1"]),
            ("north", []),
            ("nOrth", ["evt-2"]),
            ("North", []),
        ):
            with self.subTest(region=region):
                status, body = self.decisions(self.query(region=region))
                self.assertEqual(status, 200)
                self.assertEqual(
                    [step["eventId"] for step in body["steps"]], expected
                )

    # --------------------------------------------------------- suppression rule

    def test_observe_escalate_suppress_lifecycle(self) -> None:
        self.seed_suppression_events()
        # threshold 2, suppression window 100.
        status, body = self.decisions(
            self.query(threshold=2, suppression_window=100)
        )
        self.assertEqual(status, 200)

        def summary(step: dict[str, Any]) -> tuple[str, Any, Any, int, Any]:
            return (
                step["action"],
                step["alertId"],
                step["suppressedCount"],
                step["peakCount"],
                step["peakStart"],
            )

        self.assertEqual(
            [summary(step) for step in body["steps"]],
            [
                # evt-a: window 0 count 1 — below threshold.
                ("observe", None, None, 1, 0),
                # evt-b: window 0 count 2 — first threshold hit opens alert-1.
                ("escalate", "alert-1", 0, 2, 0),
                # evt-c: peak stays on window 0; 0 - 0 < 100 suppresses.
                ("suppress", "alert-1", 1, 2, 0),
                # evt-d: windows 0 and 60 tie at 2; ties keep the earlier
                # start, so suppression continues.
                ("suppress", "alert-1", 2, 2, 0),
                # evt-e/evt-f: window 180 reaches 1 then ties at 2; the
                # earliest peak start (0) is still within the window.
                ("suppress", "alert-1", 3, 2, 0),
                ("suppress", "alert-1", 4, 2, 0),
                # evt-g: window 180 count 3 takes the peak; 180 - 0 >= 100
                # opens alert-2 with a fresh suppression count.
                ("escalate", "alert-2", 0, 3, 180),
                # evt-h: still on the 180 peak, now 180 - 180 < 100.
                ("suppress", "alert-2", 1, 4, 180),
            ],
        )

    def test_suppression_boundary_is_inclusive_of_the_window_width(self) -> None:
        # Window 0 gets 2 events (peak start 0, alert-1); window 60 gets 3
        # so its peak start becomes 60. A gap of exactly 60 escalates; a
        # gap of 59 suppresses against alert-1.
        for event_id, timestamp in (
            ("evt-a", 0),
            ("evt-b", 1),
            ("evt-c", 60),
            ("evt-d", 61),
            ("evt-e", 62),
        ):
            self.add(event_id, timestamp)
        status, body = self.decisions(
            self.query(threshold=2, suppression_window=60)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [
                (step["action"], step["alertId"], step["suppressedCount"])
                for step in body["steps"]
            ],
            [
                ("observe", None, None),
                ("escalate", "alert-1", 0),
                ("suppress", "alert-1", 1),
                ("suppress", "alert-1", 2),
                ("escalate", "alert-2", 0),
            ],
        )
        status, body = self.decisions(
            self.query(threshold=2, suppression_window=61)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [
                (step["action"], step["alertId"], step["suppressedCount"])
                for step in body["steps"]
            ],
            [
                ("observe", None, None),
                ("escalate", "alert-1", 0),
                ("suppress", "alert-1", 1),
                ("suppress", "alert-1", 2),
                ("suppress", "alert-1", 3),
            ],
        )

    def test_observed_steps_establish_no_alert_history(self) -> None:
        # threshold 2: the first two steps are observe, the third step is
        # the first threshold hit and opens alert-1 regardless of the
        # earlier observed steps.
        for event_id, timestamp in (("evt-a", 0), ("evt-b", 100), ("evt-c", 101)):
            self.add(event_id, timestamp)
        status, body = self.decisions(self.query(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(
            [
                (step["action"], step["alertId"], step["suppressedCount"])
                for step in body["steps"]
            ],
            [
                ("observe", None, None),
                ("observe", None, None),
                ("escalate", "alert-1", 0),
            ],
        )

    def test_windows_match_region_replay_contract_with_range(self) -> None:
        for event_id, timestamp in (
            ("evt-a", 10),
            ("evt-b", 60),
            ("evt-c", 61),
            ("evt-d", 200),
        ):
            self.add(event_id, timestamp)
        status, body = self.decisions(
            self.query(threshold=2, from_to=(60, 180))
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 60)
        self.assertEqual(body["to"], 180)
        grid = [(60, 120), (120, 180), (180, 240)]
        for step in body["steps"]:
            self.assertEqual(
                [(row["start"], row["end"]) for row in step["windows"]], grid
            )
        # Out-of-range events still arrive as steps, in replay order.
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["evt-a", "evt-b", "evt-c", "evt-d"],
        )
        # evt-a is before the range and counts in no ranged window (null
        # peak, observe); evt-c first reaches the threshold in window 60 and
        # opens replay alert-1; evt-d is beyond the range and keeps the same
        # counted prefix, so it suppresses against that alert.
        self.assertEqual(
            [
                (step["eventId"], step["action"], step["alertId"])
                for step in body["steps"]
            ],
            [
                ("evt-a", "observe", None),
                ("evt-b", "observe", None),
                ("evt-c", "escalate", "alert-1"),
                ("evt-d", "suppress", "alert-1"),
            ],
        )
        self.assertEqual(body["steps"][0]["peakCount"], 0)
        self.assertIsNone(body["steps"][0]["peakStart"])

    # ---------------------------------------------------------- read-only / ids

    def test_replay_is_independent_of_the_live_alert_ledger(self) -> None:
        self.add("evt-a", 0)
        self.add("evt-b", 10)
        # A real evaluation first opens alert-1 on the live store.
        status, _, _ = self.request_raw(
            "/alerts/region/evaluate",
            method="POST",
            body=json.dumps(
                {
                    "organizationId": ORG1,
                    "region": REGION,
                    "type": EVENT_TYPE,
                    "windowSize": 60,
                    "threshold": 2,
                    "suppressionWindow": 100,
                }
            ).encode(),
        )
        self.assertEqual(status, 200)
        # The replay still opens its own alert-1 from an empty simulated
        # history; the stored alert neither suppresses it nor shifts the
        # simulated identifier.
        status, body = self.decisions(
            self.query(threshold=2, suppression_window=100)
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [
                (step["action"], step["alertId"], step["suppressedCount"])
                for step in body["steps"]
            ],
            [
                ("observe", None, None),
                ("escalate", "alert-1", 0),
            ],
        )
        # The replay wrote nothing: the live listing keeps one alert, and a
        # later real escalation draws the next global id, alert-2 — proving
        # the replay's simulated alert-1 never consumed the sequence.
        status, _, listing = self.request_raw(
            f"/alerts/region?organizationId={ORG1}&region={REGION}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["alerts"]), 1)
        for event_id, timestamp in (("evt-c", 200), ("evt-d", 201), ("evt-e", 202)):
            self.add(event_id, timestamp)
        status, _, evaluated = self.request_raw(
            "/alerts/region/evaluate",
            method="POST",
            body=json.dumps(
                {
                    "organizationId": ORG1,
                    "region": REGION,
                    "type": EVENT_TYPE,
                    "windowSize": 60,
                    "threshold": 3,
                    "suppressionWindow": 100,
                }
            ).encode(),
        )
        self.assertEqual(status, 200)
        self.assertEqual(evaluated["action"], "escalate")
        self.assertEqual(evaluated["alertId"], "alert-2")

    def test_repeated_requests_are_byte_identical_and_isolated(self) -> None:
        self.seed_suppression_events()
        query = self.query(threshold=2, suppression_window=100)
        raws = [self.decisions_raw(query)[1] for _ in range(3)]
        self.assertEqual(raws[0], raws[1])
        self.assertEqual(raws[1], raws[2])
        ranged = self.decisions_raw(
            self.query(threshold=2, suppression_window=100, from_to=(0, 180))
        )[1]
        self.assertNotEqual(ranged, raws[0])
        self.assertEqual(self.decisions_raw(query)[1], raws[0])

    def test_read_and_write_credentials_may_both_query(self) -> None:
        self.seed_suppression_events()
        query = self.query(threshold=2)
        read_status, read_body = self.decisions(query, token="r1")
        write_status, write_body = self.decisions(query, token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    def test_read_only_writes_nothing(self) -> None:
        self.seed_suppression_events()
        self.decisions(self.query(threshold=1))
        self.decisions(self.query(from_to=(0, 180)))
        status, _, listing = self.request_raw(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 8)
        status, raw, _ = self.request_raw(f"/alerts?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertIn(b'"alerts":[]', raw)
        status, raw, _ = self.request_raw(
            f"/alerts/region?organizationId={ORG1}&region={REGION}"
        )
        self.assertEqual(status, 200)
        self.assertIn(b'"alerts":[]', raw)

    # --------------------------------------------------------- serialization

    def test_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_suppression_events()
        status, raw = self.decisions_raw(self.query(threshold=2))
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        body = json.loads(raw)
        self.assertEqual(
            set(body),
            {
                "organizationId",
                "region",
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
            set(body["steps"][0]),
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
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )
        for name in ("windowSize", "threshold", "suppressionWindow"):
            self.assertIsInstance(body[name], int)
        self.assertIsInstance(body["steps"][0]["occurredAt"], int)

    # ----------------------------------------------------------------- 401/403

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.seed_suppression_events()
        path = f"/alerts/region/replay/decisions?{self.query()}"
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
            f"{self.base_url}/alerts/region/replay/decisions"
            "?windowSize=not-a-number",
            headers={"Authorization": "Bearer ghost-token"},
            method="GET",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 401)
        caught.exception.close()

    def test_query_shape_is_checked_before_organization(self) -> None:
        status, body = self.decisions(
            "organizationId=&region=north&windowSize=x&threshold=1"
            "&suppressionWindow=10",
            token="w2",
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_foreign_organization_is_403_and_leaves_no_trace(self) -> None:
        self.seed_suppression_events()
        status, body = self.decisions(self.query(), token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        status, _, listing = self.request_raw(
            f"/alerts/region?organizationId={ORG1}&region={REGION}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(listing["alerts"], [])

    # ---------------------------------------------------------------------- 422

    def test_each_required_parameter_must_appear_exactly_once(self) -> None:
        self.seed_suppression_events()
        base = self.query()
        for query in (
            "",
            "region=north&type=t&windowSize=60&threshold=2&suppressionWindow=100",
            "organizationId=org-1&type=t&windowSize=60&threshold=2"
            "&suppressionWindow=100",
            "organizationId=org-1&region=north&windowSize=60&threshold=2"
            "&suppressionWindow=100",
            "organizationId=org-1&region=north&type=t&threshold=2"
            "&suppressionWindow=100",
            "organizationId=org-1&region=north&type=t&windowSize=60"
            "&suppressionWindow=100",
            "organizationId=org-1&region=north&type=t&windowSize=60&threshold=2",
            f"{base}&organizationId=org-1",
            f"{base}&region=north",
            f"{base}&type={EVENT_TYPE}",
            f"{base}&windowSize=60",
            f"{base}&threshold=2",
            f"{base}&suppressionWindow=100",
        ):
            with self.subTest(query=query):
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_values_are_422(self) -> None:
        for query in (
            "organizationId=&region=r&type=t&windowSize=60&threshold=2"
            "&suppressionWindow=100",
            "organizationId=org-1&region=&type=t&windowSize=60&threshold=2"
            "&suppressionWindow=100",
            "organizationId=org-1&region=r&type=&windowSize=60&threshold=2"
            "&suppressionWindow=100",
            "organizationId=org-1&region=r&type=t&windowSize=&threshold=2"
            "&suppressionWindow=100",
            "organizationId=org-1&region=r&type=t&windowSize=60&threshold="
            "&suppressionWindow=100",
            "organizationId=org-1&region=r&type=t&windowSize=60&threshold=2"
            "&suppressionWindow=",
        ):
            with self.subTest(query=query):
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_positive_integer_parameters_must_be_positive_integers(self) -> None:
        for bad in ("0", "-1", "1.5", "abc", "1e3", "+5", "true"):
            for name, others in (
                ("windowSize", "threshold=2&suppressionWindow=100"),
                ("threshold", "windowSize=60&suppressionWindow=100"),
                ("suppressionWindow", "windowSize=60&threshold=2"),
            ):
                query = (
                    "organizationId=org-1&region=north"
                    f"&type={EVENT_TYPE}&{name}={bad}&{others}"
                )
                with self.subTest(name=name, bad=bad):
                    status, body = self.decisions(query)
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_range_must_be_paired_non_negative_and_ordered(self) -> None:
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

    def test_validation_failure_writes_nothing(self) -> None:
        self.seed_suppression_events()
        status, _ = self.decisions(
            f"organizationId=org-1&region=north&type={EVENT_TYPE}"
            "&windowSize=60&threshold=2&suppressionWindow=x"
        )
        self.assertEqual(status, 422)
        status, _, listing = self.request_raw(
            f"/alerts/region?organizationId={ORG1}&region={REGION}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(listing["alerts"], [])

    # ------------------------------------------------------------- branch prefix

    def test_no_branch_prefixed_region_alert_replay_path(self) -> None:
        self.add("evt-a", 0)
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
            f"/branches/br-1/alerts/region/replay/decisions?{self.query()}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
