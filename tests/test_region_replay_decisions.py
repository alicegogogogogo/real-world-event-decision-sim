"""Regression tests for GET /events/region/replay/decisions.

The baseline already had the region list and region aggregate queries and
the step-by-step replay-decision query, but no region-dimension
step-by-step replay; this locks down the new read-only entry point that
applies the verbatim region attribution rule before the replay-decision
contract:

- only the caller's organization's events whose payload ``region`` is a
  non-empty string equal to the requested region and whose ``type``
  matches are replayed; unattributed events, other regions/types, and
  other organizations' data never enter the result;
- matching events arrive ordered by ``occurredAt`` then ``eventId``; each
  step reports the event, the window rows accumulated so far (the exact
  main-aggregate window division, ranged empty windows kept), and the
  peak decision (largest count, ties to the earliest start, ``escalate``
  at the threshold, ``observe`` otherwise);
- the response echoes organization, region, type, window width,
  threshold and range, is compact key-sorted JSON with integer values and
  one trailing newline, and identical requests are byte-for-byte identical
  without polluting each other;
- both ``read`` and ``write`` credentials may call it; the verdict order
  is fixed — 401 (credential) before 422 (query shape) before 403
  (organization) — nothing is ever written, and restarting clears the
  ledger. The region replay is main-only: no branch-prefixed path exists.
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

# Sentinel: a request with token=_UNSET uses the shared auto-credential
# helper, token=None sends no Authorization header at all.
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


class RegionReplayDecisionsTest(unittest.TestCase):
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
            f"/events/region/replay/decisions?{query}", token=token
        )
        return status, body

    def decisions_raw(
        self, query: str, *, token: Any = _UNSET
    ) -> tuple[int, bytes]:
        status, raw, _ = self.request_raw(
            f"/events/region/replay/decisions?{query}", token=token
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

    def add_event(
        self,
        event_id: str,
        *,
        occurred_at: int,
        token: str = "w1",
        event_type: str = EVENT_TYPE,
        payload: dict[str, Any] | None = None,
    ) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        event = make_event(
            eventId=event_id,
            organizationId=organization_id,
            occurredAt=occurred_at,
            type=event_type,
            payload={"region": REGION} if payload is None else payload,
        )
        status, _ = self.request_raw(
            "/events",
            method="POST",
            body=json.dumps(event).encode(),
            token=token,
        )
        self.assertEqual(status, 201)

    @staticmethod
    def query(
        *,
        org: str = ORG1,
        region: str = REGION,
        event_type: str = EVENT_TYPE,
        window_size: int = 60,
        threshold: int = 3,
        from_to: tuple[int, int] | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "organizationId": org,
            "region": region,
            "type": event_type,
            "windowSize": window_size,
            "threshold": threshold,
        }
        if from_to is not None:
            params["from"], params["to"] = from_to
        return urlencode(params)

    def seed_events(self) -> None:
        events = [
            make_event(eventId="evt-b", occurredAt=200),
            make_event(eventId="evt-a", occurredAt=200),
            make_event(eventId="evt-c", occurredAt=100),
            make_event(eventId="evt-d", occurredAt=0),
            make_event(eventId="evt-e", occurredAt=300),
            # Another region never enters this replay.
            make_event(
                eventId="evt-s1",
                occurredAt=5,
                payload={"region": "south"},
            ),
            # Region-bearing but a different type never opens a step.
            make_event(
                eventId="evt-u1",
                type="incident.updated",
                occurredAt=10,
                payload={"region": REGION},
            ),
            # No region attribution: empty string, missing key, non-string.
            make_event(eventId="evt-z1", occurredAt=11, payload={"region": ""}),
            make_event(eventId="evt-z2", occurredAt=12, payload={}),
            make_event(eventId="evt-z3", occurredAt=13, payload={"region": 7}),
            # Another organization's same-named region never enters.
            make_event(
                eventId="evt-x",
                organizationId=ORG2,
                occurredAt=1,
                payload={"region": REGION},
            ),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

    # ----------------------------------------------------------------- 200 shape

    def test_steps_follow_replay_order_and_region_attribution(self) -> None:
        self.seed_events()
        status, body = self.decisions(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["region"], REGION)
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
        # No other-region, other-type, unattributed, or other-org event.
        serialized = json.dumps(body)
        for excluded in ("evt-s1", "evt-u1", "evt-z1", "evt-z2", "evt-z3", "evt-x"):
            self.assertNotIn(excluded, serialized)

    def test_each_step_accumulates_window_counts(self) -> None:
        self.seed_events()
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
        self.seed_events()
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
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=occurred_at)
                )[0],
                201,
            )
        status, body = self.decisions(self.query(threshold=5))
        self.assertEqual(status, 200)
        for step in body["steps"][1:]:
            self.assertEqual(step["peakCount"], 1)
            self.assertEqual(step["peakStart"], 0)
            self.assertEqual(step["action"], "observe")

    def test_unknown_region_or_type_or_organization_has_no_steps(self) -> None:
        self.seed_events()
        for kwargs in (
            {"region": "nowhere"},
            {"event_type": "never.seen"},
            {"org": "org-x"},
        ):
            with self.subTest(kwargs=kwargs):
                status, body = self.decisions(self.query(**kwargs))
                self.assertEqual(status, 200)
                self.assertEqual(body["steps"], [])
                self.assertNotIn("evt-x", json.dumps(body))

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

    def test_only_this_organizations_events_are_replayed(self) -> None:
        self.seed_events()
        status, body = self.decisions(self.query(org=ORG2))
        self.assertEqual(status, 200)
        # org-2 only has evt-x in region north; the org-1 event ids never
        # appear even under the same region name.
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["evt-x"]
        )
        self.assertNotIn("evt-d", json.dumps(body))

    # ------------------------------------------------------------- range semantics

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.seed_events()
        status, body = self.decisions(self.query(from_to=(0, 180)))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 180)
        grid = [(0, 60), (60, 120), (120, 180), (180, 240)]
        for step in body["steps"]:
            self.assertEqual(
                [(row["start"], row["end"]) for row in step["windows"]], grid
            )
        # evt-a/evt-b at 200 and evt-e at 300 are outside [0, 180], so the
        # 180 window never gains a count here.
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
        self.post_event(make_event(eventId="evt-a", occurredAt=10))
        self.post_event(make_event(eventId="evt-b", occurredAt=70))
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

    # --------------------------------------------- consistency with region aggregate

    def test_final_step_matches_region_aggregate(self) -> None:
        self.seed_events()
        status, body = self.decisions(self.query(threshold=2))
        self.assertEqual(status, 200)
        status, _, aggregate = self.request_raw(
            "/events/region/aggregate?"
            + urlencode(
                {
                    "organizationId": ORG1,
                    "region": REGION,
                    "type": EVENT_TYPE,
                    "windowSize": 60,
                }
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"][-1]["windows"], aggregate["windows"])

    # --------------------------------------------------------- serialization / roles

    def test_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_events()
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
                "region",
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
        self.seed_events()
        query = self.query(threshold=2)
        raws = [self.decisions_raw(query)[1] for _ in range(3)]
        self.assertEqual(raws[0], raws[1])
        self.assertEqual(raws[1], raws[2])
        # A ranged read between them does not change any later result.
        ranged = self.decisions_raw(self.query(threshold=2, from_to=(0, 180)))[1]
        self.assertNotEqual(ranged, raws[0])
        self.assertEqual(self.decisions_raw(query)[1], raws[0])

    def test_read_and_write_credentials_may_both_query(self) -> None:
        self.seed_events()
        query = self.query()
        read_status, read_body = self.decisions(query, token="r1")
        write_status, write_body = self.decisions(query, token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    def test_query_is_read_only(self) -> None:
        self.seed_events()
        self.decisions(self.query(from_to=(0, 180)))
        self.decisions(self.query(threshold=1))
        status, _, listing = self.request_raw(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 10)
        status, raw, _ = self.request_raw(f"/alerts?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertIn(b'"alerts":[]', raw)

    # ----------------------------------------------------------------- 401 / 403

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.seed_events()
        path = f"/events/region/replay/decisions?{self.query()}"
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
            f"{self.base_url}/events/region/replay/decisions"
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
            "organizationId=&region=north&windowSize=x&threshold=1",
            token="w2",
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_foreign_organization_is_403_and_leaves_no_trace(self) -> None:
        self.seed_events()
        status, body = self.decisions(self.query(), token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        status, _, listing = self.request_raw(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 10)

    # ---------------------------------------------------------------------- 422

    def test_each_required_parameter_must_appear_exactly_once(self) -> None:
        self.seed_events()
        base = self.query()
        for query in (
            "",
            "region=north&type=t&windowSize=60&threshold=3",
            "organizationId=org-1&type=t&windowSize=60&threshold=3",
            "organizationId=org-1&region=north&windowSize=60&threshold=3",
            "organizationId=org-1&region=north&type=t&threshold=3",
            "organizationId=org-1&region=north&type=t&windowSize=60",
            f"{base}&organizationId=org-1",
            f"{base}&region=north",
            f"{base}&type={EVENT_TYPE}",
            f"{base}&windowSize=60",
            f"{base}&threshold=3",
        ):
            with self.subTest(query=query):
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_values_are_422(self) -> None:
        self.seed_events()
        for query in (
            f"organizationId=&region=r&type=t&windowSize=60&threshold=3",
            f"organizationId=%20&region=r&type=t&windowSize=60&threshold=3",
            "organizationId=org-1&region=&type=t&windowSize=60&threshold=3",
            "organizationId=org-1&region=%20&type=t&windowSize=60&threshold=3",
            "organizationId=org-1&region=r&type=&windowSize=60&threshold=3",
            "organizationId=org-1&region=r&type=%20&windowSize=60&threshold=3",
            "organizationId=org-1&region=r&type=t&windowSize=&threshold=3",
            "organizationId=org-1&region=r&type=t&windowSize=%20&threshold=3",
            "organizationId=org-1&region=r&type=t&windowSize=60&threshold=",
            "organizationId=org-1&region=r&type=t&windowSize=60&threshold=%20",
        ):
            with self.subTest(query=query):
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_window_size_and_threshold_must_be_positive_integers(self) -> None:
        self.seed_events()
        for bad in ("0", "-1", "1.5", "abc", "1e3", "+5", "true"):
            for name, other in (
                ("windowSize", "threshold=3"),
                ("threshold", "windowSize=60"),
            ):
                query = (
                    "organizationId=org-1&region=north"
                    f"&type={EVENT_TYPE}&{name}={bad}&{other}"
                )
                with self.subTest(name=name, bad=bad):
                    status, body = self.decisions(query)
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
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_from_equal_to_to_is_legal(self) -> None:
        self.post_event(make_event(eventId="evt-a", occurredAt=30))
        status, body = self.decisions(self.query(from_to=(30, 30)))
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(
            body["steps"][0]["windows"],
            [{"start": 0, "end": 60, "count": 1}],
        )

    def test_validation_failure_writes_nothing(self) -> None:
        self.seed_events()
        status, _ = self.decisions(
            f"organizationId=org-1&region=north&type={EVENT_TYPE}"
            "&windowSize=x&threshold=3"
        )
        self.assertEqual(status, 422)
        status, _, listing = self.request_raw(f"/events?organizationId={ORG1}")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 10)

    # ------------------------------------------------------------- branch prefix

    def test_no_branch_prefixed_region_replay_path(self) -> None:
        # The region replay is main/snapshot-only; branches expose the
        # non-region replay but no region sub-path.
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
            f"/branches/br-1/events/region/replay/decisions?{self.query()}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
