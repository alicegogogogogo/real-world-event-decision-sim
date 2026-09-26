from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server
from tests import _support

EVENT_TYPE = "incident.created"

# Sentinel: a request with token=_UNSET uses the shared auto-credential
# helper, token=None sends no Authorization header at all.
_UNSET = object()


def make_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "eventId": "evt-1",
        "organizationId": "org-1",
        "type": EVENT_TYPE,
        "occurredAt": 100,
        "payload": {"severity": "low"},
    }
    event.update(overrides)
    return event


class ReplayDecisionEndpointsTest(unittest.TestCase):
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
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        token: str | None = _UNSET,
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

    def decisions(self, query: str, *, token: str | None = _UNSET) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            f"/events/replay/decisions?{query}", token=token
        )
        return status, body

    def post_event(self, event: dict[str, Any]) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            "/events", method="POST", body=json.dumps(event).encode()
        )
        return status, body

    def register_token(self, token: str, organization_id: str, role: str) -> None:
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
        self.assertEqual(status, 201)

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
            make_event(eventId="evt-x", organizationId="org-2", occurredAt=10),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

    # ----------------------------------------------------------------- 200 shape

    def test_steps_follow_replay_order_with_event_id_and_time(self) -> None:
        self.seed_events()
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=3"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
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
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=3"
        )
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
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=2"
        )
        self.assertEqual(status, 200)
        actions = [step["action"] for step in body["steps"]]
        # The fourth step (evt-b) is the first to put two events in one window.
        self.assertEqual(
            actions, ["observe", "observe", "observe", "escalate", "escalate"]
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
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=5"
        )
        self.assertEqual(status, 200)
        for step in body["steps"][1:]:
            self.assertEqual(step["peakCount"], 1)
            self.assertEqual(step["peakStart"], 0)
            self.assertEqual(step["action"], "observe")

    def test_no_matching_events_has_zero_count_empty_start_and_observe(self) -> None:
        self.seed_events()
        status, body = self.decisions(
            "organizationId=org-1&type=never.seen&windowSize=60&threshold=3"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "type": "never.seen",
                "windowSize": 60,
                "threshold": 3,
                "from": None,
                "to": None,
                "steps": [],
            },
        )

    def test_unknown_organization_returns_empty_steps(self) -> None:
        self.seed_events()
        self.register_token("tok-x", "org-x", "write")
        status, body = self.decisions(
            f"organizationId=org-x&type={EVENT_TYPE}&windowSize=60&threshold=3",
            token="tok-x",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-x")
        self.assertEqual(body["steps"], [])
        self.assertNotIn("org-2", json.dumps(body))

    # ------------------------------------------------------------- range semantics

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.seed_events()
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=60&threshold=3&from=0&to=180"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 180)
        grid = [(0, 60), (60, 120), (120, 180), (180, 240)]
        for step in body["steps"]:
            self.assertEqual(
                [(row["start"], row["end"]) for row in step["windows"]], grid
            )
        # Counts follow the closed interval: the events at 200 and 300 are
        # outside [0, 180], so the 180 window never gains a count here.
        self.assertEqual(
            [row["count"] for row in body["steps"][-1]["windows"]],
            [1, 1, 0, 0],
        )
        # Out-of-range events still arrive as replay steps in order.
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["evt-d", "evt-c", "evt-a", "evt-b", "evt-e"],
        )

    def test_range_boundaries_are_inclusive(self) -> None:
        for event_id, occurred_at in (
            ("evt-a", 59),
            ("evt-b", 60),
            ("evt-c", 119),
            ("evt-d", 120),
        ):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=occurred_at)
                )[0],
                201,
            )
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=60&threshold=2&from=60&to=120"
        )
        self.assertEqual(status, 200)
        last = body["steps"][-1]
        # The closed interval includes both 60 and 120: [60,120) holds the
        # events at 60 and 119, and the event exactly at 120 opens the next
        # intersecting window [120,180). The event at 59 never counts.
        self.assertEqual(
            [(row["start"], row["count"]) for row in last["windows"]],
            [(60, 2), (120, 1)],
        )
        self.assertEqual(last["peakStart"], 60)
        self.assertEqual(last["peakCount"], 2)
        self.assertEqual(last["action"], "escalate")

    # --------------------------------------------- consistency with aggregate/decision

    def test_final_step_matches_aggregate_and_decision_without_range(self) -> None:
        self.seed_events()
        query = (
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=2"
        )
        status, body = self.decisions(query)
        self.assertEqual(status, 200)

        status, _, aggregate = self.request_raw(
            f"/events/aggregate?organizationId=org-1&type={EVENT_TYPE}&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"][-1]["windows"], aggregate["windows"])

        decision_body = {
            "organizationId": "org-1",
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 2,
        }
        status, _, decision = self.request_raw(
            "/decisions/evaluate",
            method="POST",
            body=json.dumps(decision_body).encode(),
        )
        self.assertEqual(status, 200)
        last = body["steps"][-1]
        self.assertEqual(last["peakStart"], decision["peakStart"])
        self.assertEqual(last["peakCount"], decision["peakCount"])
        self.assertEqual(last["action"], decision["action"])

    def test_final_step_matches_aggregate_and_decision_with_range(self) -> None:
        self.seed_events()
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=60&threshold=2&from=0&to=180"
        )
        self.assertEqual(status, 200)
        status, _, aggregate = self.request_raw(
            f"/events/aggregate?organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=60&from=0&to=180"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"][-1]["windows"], aggregate["windows"])

        decision_body = {
            "organizationId": "org-1",
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 2,
            "from": 0,
            "to": 180,
        }
        status, _, decision = self.request_raw(
            "/decisions/evaluate",
            method="POST",
            body=json.dumps(decision_body).encode(),
        )
        self.assertEqual(status, 200)
        last = body["steps"][-1]
        self.assertEqual(last["peakStart"], decision["peakStart"])
        self.assertEqual(last["peakCount"], decision["peakCount"])
        self.assertEqual(last["action"], decision["action"])

    # --------------------------------------------------------- serialization / roles

    def test_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_events()
        status, raw, body = self.request_raw(
            f"/events/replay/decisions?organizationId=org-1"
            f"&type={EVENT_TYPE}&windowSize=60&threshold=3"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(
            set(body),
            {
                "organizationId",
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
        query = f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=2"
        raws = [
            self.request_raw(f"/events/replay/decisions?{query}")[1]
            for _ in range(3)
        ]
        self.assertEqual(raws[0], raws[1])
        self.assertEqual(raws[1], raws[2])
        # A ranged read between them does not change any later result.
        ranged = self.request_raw(
            f"/events/replay/decisions?{query}&from=0&to=180"
        )[1]
        self.assertNotEqual(ranged, raws[0])
        self.assertEqual(
            self.request_raw(f"/events/replay/decisions?{query}")[1], raws[0]
        )

    def test_read_and_write_credentials_may_both_query(self) -> None:
        self.seed_events()
        self.register_token("tok-read", "org-1", "read")
        self.register_token("tok-write", "org-1", "write")
        query = f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=3"
        read_status, read_body = self.decisions(query, token="tok-read")
        write_status, write_body = self.decisions(query, token="tok-write")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    def test_query_is_read_only(self) -> None:
        self.seed_events()
        self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=60&threshold=2&from=0&to=180"
        )
        self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=1"
        )
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)
        status, _, alerts = self.request_raw("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    def test_fresh_server_starts_with_empty_steps(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            fresh_base = f"http://127.0.0.1:{fresh.server_port}"
            token = _support.ensure_token({}, fresh_base, "org-1")
            request = Request(
                f"{fresh_base}/events/replay/decisions"
                f"?organizationId=org-1&type={EVENT_TYPE}"
                "&windowSize=60&threshold=3",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
                body = json.load(response)
                self.assertEqual(body["steps"], [])
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # ----------------------------------------------------------------- 401 / 403

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.seed_events()
        valid_query = (
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=3"
        )
        for headers in (
            {},
            {"Authorization": "Bearer"},
            {"Authorization": "Bearer "},
            {"Authorization": "Basic tok-w"},
            {"Authorization": "Bearer ghost-token"},
        ):
            with self.subTest(headers=headers):
                request = Request(
                    f"{self.base_url}/events/replay/decisions?{valid_query}",
                    headers=headers,
                    method="GET",
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
            f"{self.base_url}/events/replay/decisions?windowSize=not-a-number",
            headers={"Authorization": "Bearer ghost-token"},
            method="GET",
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 401)
        caught.exception.close()

    def test_foreign_organization_is_403_and_leaves_no_trace(self) -> None:
        self.seed_events()
        self.register_token("tok-2", "org-2", "write")
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=3",
            token="tok-2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        # The rejected read changed nothing.
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)

    # ---------------------------------------------------------------------- 422

    def test_each_required_parameter_must_appear_exactly_once(self) -> None:
        base = f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=3"
        for query in (
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
            with self.subTest(query=query):
                status, body = self.decisions(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_values_are_422(self) -> None:
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

    def test_leading_zero_integer_text_is_accepted(self) -> None:
        # isdigit text parses as a decimal integer regardless of leading zeros.
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=060&threshold=003"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 3)

    def test_range_must_be_paired_non_negative_and_ordered(self) -> None:
        base = f"organizationId=org-1&type={EVENT_TYPE}&windowSize=60&threshold=3"
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
        self.assertEqual(
            self.post_event(make_event(eventId="evt-a", occurredAt=30))[0],
            201,
        )
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}"
            "&windowSize=60&threshold=3&from=30&to=30"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(
            body["steps"][0]["windows"],
            [{"start": 0, "end": 60, "count": 1}],
        )

    def test_validation_failure_writes_nothing(self) -> None:
        self.seed_events()
        status, body = self.decisions(
            f"organizationId=org-1&type={EVENT_TYPE}&windowSize=x&threshold=3"
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 6)


if __name__ == "__main__":
    unittest.main()
