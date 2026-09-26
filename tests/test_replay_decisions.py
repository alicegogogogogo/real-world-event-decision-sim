from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server
from tests import _support


def make_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "eventId": "evt-1",
        "organizationId": "org-1",
        "type": "incident.created",
        "occurredAt": 100,
        "payload": {"severity": "low"},
    }
    event.update(overrides)
    return event


class ReplayDecisionsEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.token_cache: dict[str, str] = {}

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request_raw(
        self, path: str, *, method: str = "GET", body: bytes | None = None
    ) -> tuple[int, bytes, Any]:
        headers: dict[str, str] = {"Content-Type": "application/json"} if body is not None else {}
        headers.update(
            _support.authorization_header(
                self.token_cache, self.base_url, path, body
            )
        )
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

    def replay_decisions(self, query: str) -> tuple[int, Any]:
        status, _, body = self.request_raw(f"/events/replay/decisions?{query}")
        return status, body

    def post_event(self, event: dict[str, Any]) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            "/events", method="POST", body=json.dumps(event).encode()
        )
        return status, body

    def seed_events(self) -> None:
        events = [
            make_event(eventId="evt-b", occurredAt=200),
            make_event(eventId="evt-a", occurredAt=200),
            make_event(eventId="evt-c", occurredAt=100),
            make_event(eventId="evt-d", occurredAt=0),
            make_event(eventId="evt-e", occurredAt=300),
            # A different type never enters the replay.
            make_event(eventId="evt-f", type="incident.updated", occurredAt=150),
            # Another organization's events are never replayed for org-1.
            make_event(eventId="evt-x", organizationId="org-2", occurredAt=50),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

    BASE_QUERY = "organizationId=org-1&type=incident.created&windowSize=60&threshold=2"

    # ------------------------------------------- GET /events/replay/decisions

    def test_steps_follow_replay_order_with_cumulative_windows(self) -> None:
        self.seed_events()
        status, body = self.replay_decisions(self.BASE_QUERY)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(body["type"], "incident.created")
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 2)
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])

        steps = body["steps"]
        # occurredAt ascending, eventId code-point order within one instant.
        self.assertEqual(
            [step["eventId"] for step in steps],
            ["evt-d", "evt-c", "evt-a", "evt-b", "evt-e"],
        )
        self.assertEqual(
            [step["occurredAt"] for step in steps], [0, 100, 200, 200, 300]
        )

        self.assertEqual(
            steps[0]["windows"], [{"start": 0, "end": 60, "count": 1}]
        )
        self.assertEqual(
            steps[0]["decision"],
            {"peakStart": 0, "peakCount": 1, "action": "observe"},
        )

        self.assertEqual(
            steps[1]["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
            ],
        )
        # Peak tie resolves to the earliest window start.
        self.assertEqual(
            steps[1]["decision"],
            {"peakStart": 0, "peakCount": 1, "action": "observe"},
        )

        self.assertEqual(
            steps[2]["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
                {"start": 180, "end": 240, "count": 1},
            ],
        )
        self.assertEqual(
            steps[3]["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
                {"start": 180, "end": 240, "count": 2},
            ],
        )
        # The peak reaches the threshold once evt-b lands.
        self.assertEqual(
            steps[3]["decision"],
            {"peakStart": 180, "peakCount": 2, "action": "escalate"},
        )

        self.assertEqual(
            steps[4]["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
                {"start": 180, "end": 240, "count": 2},
                {"start": 300, "end": 360, "count": 1},
            ],
        )
        self.assertEqual(
            steps[4]["decision"],
            {"peakStart": 180, "peakCount": 2, "action": "escalate"},
        )

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.seed_events()
        status, body = self.replay_decisions(self.BASE_QUERY + "&from=60&to=180")
        self.assertEqual(status, 200)
        # Every matching event is still a replay step; the range only
        # filters which accumulated events count toward the windows.
        self.assertEqual(len(body["steps"]), 5)
        self.assertEqual(body["from"], 60)
        self.assertEqual(body["to"], 180)
        # Windows intersecting [60, 180]: [60,120), [120,180), [180,240).
        # Step 0 is evt-d at occurredAt=0, outside the range: all zero.
        self.assertEqual(
            body["steps"][0]["windows"],
            [
                {"start": 60, "end": 120, "count": 0},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 0},
            ],
        )
        self.assertEqual(
            body["steps"][0]["decision"],
            {"peakStart": None, "peakCount": 0, "action": "observe"},
        )
        # Step 1 accumulates evt-c at occurredAt=100, inside the range.
        self.assertEqual(
            body["steps"][1]["windows"],
            [
                {"start": 60, "end": 120, "count": 1},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 0},
            ],
        )
        self.assertEqual(
            body["steps"][1]["decision"],
            {"peakStart": 60, "peakCount": 1, "action": "observe"},
        )
        # The last step still counts only evt-c (100): evt-a/evt-b (200) and
        # evt-d (0) and evt-e (300) all fall outside [60, 180].
        self.assertEqual(
            body["steps"][4]["windows"],
            [
                {"start": 60, "end": 120, "count": 1},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 0},
            ],
        )
        self.assertEqual(
            body["steps"][4]["decision"],
            {"peakStart": 60, "peakCount": 1, "action": "observe"},
        )

    def test_range_without_matching_events_keeps_zero_windows(self) -> None:
        self.seed_events()
        status, body = self.replay_decisions(
            self.BASE_QUERY + "&from=600&to=660"
        )
        self.assertEqual(status, 200)
        for step in body["steps"]:
            self.assertEqual(
                step["windows"],
                [
                    {"start": 600, "end": 660, "count": 0},
                    {"start": 660, "end": 720, "count": 0},
                ],
            )
            self.assertEqual(
                step["decision"],
                {"peakStart": None, "peakCount": 0, "action": "observe"},
            )

    def test_unknown_organization_replays_zero_events(self) -> None:
        self.seed_events()
        status, body = self.replay_decisions(
            "organizationId=org-x&type=incident.created&windowSize=60&threshold=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-x")
        self.assertEqual(body["steps"], [])

    def test_unknown_type_replays_zero_events(self) -> None:
        self.seed_events()
        status, body = self.replay_decisions(
            "organizationId=org-1&type=nope&windowSize=60&threshold=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])

    def test_final_step_matches_aggregate_and_decision(self) -> None:
        self.seed_events()
        query = "organizationId=org-1&type=incident.created&windowSize=60"
        status, replay = self.replay_decisions(query + "&threshold=2")
        self.assertEqual(status, 200)

        status, _, aggregate = self.request_raw(f"/events/aggregate?{query}")
        self.assertEqual(status, 200)
        self.assertEqual(replay["steps"][-1]["windows"], aggregate["windows"])

        status, _, decision = self.request_raw(
            "/decisions/evaluate",
            method="POST",
            body=json.dumps(
                {
                    "organizationId": "org-1",
                    "type": "incident.created",
                    "windowSize": 60,
                    "threshold": 2,
                }
            ).encode(),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            replay["steps"][-1]["decision"],
            {
                "peakStart": decision["peakStart"],
                "peakCount": decision["peakCount"],
                "action": decision["action"],
            },
        )

    def test_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_events()
        status, raw, _ = self.request_raw(
            f"/events/replay/decisions?{self.BASE_QUERY}"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw, raw.decode().strip().encode() + b"\n")
        self.assertNotIn(b" ", raw)
        parsed = json.loads(raw)
        self.assertEqual(
            raw,
            (json.dumps(parsed, separators=(",", ":"), sort_keys=True) + "\n").encode(),
        )

    def test_read_only_and_deterministic(self) -> None:
        self.seed_events()
        first_status, first_raw, _ = self.request_raw(
            f"/events/replay/decisions?{self.BASE_QUERY}"
        )
        second_status, second_raw, _ = self.request_raw(
            f"/events/replay/decisions?{self.BASE_QUERY}"
        )
        self.assertEqual(first_status, 200)
        self.assertEqual(second_status, 200)
        self.assertEqual(first_raw, second_raw)
        # The replay never wrote anything: all seven events are still listed.
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)
        status, _, other = self.request_raw("/events?organizationId=org-2")
        self.assertEqual(status, 200)
        self.assertEqual(len(other["events"]), 1)

    def test_read_credential_may_replay_decisions(self) -> None:
        self.seed_events()
        payload = json.dumps(
            {"token": "reader-1", "organizationId": "org-1", "role": "read"}
        ).encode()
        request = Request(
            f"{self.base_url}/auth/tokens",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 201)
        request = Request(
            f"{self.base_url}/events/replay/decisions?{self.BASE_QUERY}",
            headers={"Authorization": "Bearer reader-1"},
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)

    # ------------------------------------------------------------- validation

    def test_missing_or_duplicated_parameters_are_422(self) -> None:
        queries = [
            "type=incident.created&windowSize=60&threshold=2",
            "organizationId=org-1&windowSize=60&threshold=2",
            "organizationId=org-1&type=incident.created&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=60",
            "organizationId=org-1&organizationId=org-1&type=incident.created&windowSize=60&threshold=2",
            "organizationId=org-1&type=incident.created&type=incident.created&windowSize=60&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=60&windowSize=60&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=2&threshold=2",
        ]
        for query in queries:
            with self.subTest(query=query):
                status, body = self.replay_decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_and_invalid_values_are_422(self) -> None:
        queries = [
            "organizationId=&type=incident.created&windowSize=60&threshold=2",
            "organizationId=%20&type=incident.created&windowSize=60&threshold=2",
            "organizationId=org-1&type=&windowSize=60&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=0&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=-5&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=6.5&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=abc&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=0",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=-1",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=x",
        ]
        for query in queries:
            with self.subTest(query=query):
                status, body = self.replay_decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_from_to_pairing_and_ordering_are_422(self) -> None:
        queries = [
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=2&from=0",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=2&to=10",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=2&from=10&to=5",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=2&from=-1&to=5",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=2&from=0&to=x",
            "organizationId=org-1&type=incident.created&windowSize=60&threshold=2&from=0&from=1&to=5",
        ]
        for query in queries:
            with self.subTest(query=query):
                status, body = self.replay_decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_unauthenticated_requests_are_401(self) -> None:
        for headers in ({}, {"Authorization": "Bearer nope"}):
            with self.subTest(headers=headers):
                request = Request(
                    f"{self.base_url}/events/replay/decisions?{self.BASE_QUERY}",
                    headers=headers,
                )
                try:
                    with urlopen(request, timeout=5):
                        self.fail("expected 401")
                except HTTPError as error:
                    self.assertEqual(error.code, 401)
                    self.assertEqual(json.loads(error.read()), {"error": "unauthorized"})
                    error.close()

    def test_other_organization_is_403(self) -> None:
        self.seed_events()
        status, body = self.replay_decisions(
            "organizationId=org-2&type=incident.created&windowSize=60&threshold=2"
        )
        # The helper authenticates as org-2 (the requested organization), so
        # register a second org-1-scoped request explicitly instead.
        self.assertEqual(status, 200)  # org-2 credential sees only org-2 data
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["evt-x"]
        )
        # An org-1 credential asking for org-2 is forbidden.
        org1_token = _support.ensure_token(self.token_cache, self.base_url, "org-1")
        request = Request(
            f"{self.base_url}/events/replay/decisions"
            "?organizationId=org-2&type=incident.created&windowSize=60&threshold=2",
            headers={"Authorization": f"Bearer {org1_token}"},
        )
        try:
            with urlopen(request, timeout=5):
                self.fail("expected 403")
        except HTTPError as error:
            self.assertEqual(error.code, 403)
            self.assertEqual(json.loads(error.read()), {"error": "forbidden"})
            error.close()


if __name__ == "__main__":
    unittest.main()
