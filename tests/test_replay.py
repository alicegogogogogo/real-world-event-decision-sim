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


class ReplayEndpointsTest(unittest.TestCase):
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

    def replay(self, query: str) -> tuple[int, Any]:
        status, _, body = self.request_raw(f"/events/replay?{query}")
        return status, body

    def compare(self, query: str) -> tuple[int, Any]:
        status, _, body = self.request_raw(f"/events/replay/compare?{query}")
        return status, body

    def post_event(self, event: dict[str, Any]) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            "/events", method="POST", body=json.dumps(event).encode()
        )
        return status, body

    def seed_replay_events(self) -> None:
        events = [
            make_event(eventId="evt-b", occurredAt=200),
            make_event(eventId="evt-a", occurredAt=200),
            make_event(eventId="evt-c", occurredAt=100),
            make_event(eventId="evt-d", occurredAt=0),
            make_event(eventId="evt-e", occurredAt=300),
            # Another organization's events are never replayed for org-1.
            make_event(eventId="evt-x", organizationId="org-2", occurredAt=50),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

    # ------------------------------------------------------------- GET /events/replay

    def test_replay_lists_events_not_after_as_of_in_order(self) -> None:
        self.seed_replay_events()
        status, body = self.replay("organizationId=org-1&asOf=200")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["evt-d", "evt-c", "evt-a", "evt-b"],
        )

    def test_replay_as_of_boundary_is_inclusive(self) -> None:
        self.seed_replay_events()
        status, body = self.replay("organizationId=org-1&asOf=100")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]], ["evt-d", "evt-c"]
        )
        status, body = self.replay("organizationId=org-1&asOf=99")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]], ["evt-d"]
        )

    def test_replay_as_of_zero_and_large_values(self) -> None:
        self.seed_replay_events()
        status, body = self.replay("organizationId=org-1&asOf=0")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]], ["evt-d"]
        )
        status, body = self.replay("organizationId=org-1&asOf=999999999999")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 5)

    def test_replay_events_carry_the_five_fields_verbatim(self) -> None:
        event = make_event(payload={"nested": {"a": [1, 2]}, "region": "north"})
        self.assertEqual(self.post_event(event)[0], 201)
        status, body = self.replay("organizationId=org-1&asOf=100")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [event])
        self.assertEqual(set(body["events"][0]), set(event))

    def test_replay_unknown_organization_returns_empty(self) -> None:
        self.seed_replay_events()
        status, body = self.replay("organizationId=org-x&asOf=500")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"organizationId": "org-x", "events": []})
        self.assertNotIn("org-2", json.dumps(body))

    def test_replay_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_replay_events()
        status, raw, body = self.request_raw(
            "/events/replay?organizationId=org-1&asOf=200"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(set(body), {"organizationId", "events"})
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )

    def test_replay_requires_organization_id_and_as_of_each_once(self) -> None:
        for query in (
            "",
            "organizationId=org-1",
            "asOf=100",
            "organizationId=org-1&asOf=100&asOf=200",
            "organizationId=a&organizationId=b&asOf=100",
        ):
            with self.subTest(query=query):
                status, body = self.replay(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_replay_rejects_blank_and_invalid_values(self) -> None:
        for query in (
            "organizationId=&asOf=100",
            "organizationId=%20&asOf=100",
            "organizationId=org-1&asOf=",
            "organizationId=org-1&asOf=%20",
            "organizationId=org-1&asOf=-1",
            "organizationId=org-1&asOf=1.5",
            "organizationId=org-1&asOf=abc",
            "organizationId=org-1&asOf=+5",
            "organizationId=org-1&asOf=1e3",
        ):
            with self.subTest(query=query):
                status, body = self.replay(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_replay_is_read_only_and_deterministic(self) -> None:
        self.seed_replay_events()
        query = "organizationId=org-1&asOf=200"
        first_status, first_raw, first = self.request_raw(f"/events/replay?{query}")
        second_status, second_raw, second = self.request_raw(f"/events/replay?{query}")
        self.assertEqual((first_status, second_status), (200, 200))
        self.assertEqual(first_raw, second_raw)
        self.assertEqual(first, second)

        # The replay never wrote anything: all six events are still listed.
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 5)
        status, _, listing = self.request_raw("/events?organizationId=org-2")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 1)

    def test_replay_fresh_server_is_empty(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            fresh_base = f"http://127.0.0.1:{fresh.server_port}"
            token = _support.ensure_token({}, fresh_base, "org-1")
            request = Request(
                f"{fresh_base}"
                "/events/replay?organizationId=org-1&asOf=100",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    json.load(response),
                    {"organizationId": "org-1", "events": []},
                )
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # --------------------------------------------------- GET /events/replay/compare

    def test_compare_groups_added_removed_changed_and_unchanged(self) -> None:
        self.seed_replay_events()
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=100&toAsOf=200"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(body["fromAsOf"], 100)
        self.assertEqual(body["toAsOf"], 200)
        # evt-a and evt-b appear only in the later replay.
        self.assertEqual(body["added"], ["evt-a", "evt-b"])
        self.assertEqual(body["removed"], [])
        self.assertEqual(body["changed"], [])
        # evt-c and evt-d are present in both, unchanged.
        self.assertEqual(body["unchangedCount"], 2)

    def test_compare_time_points_may_run_in_either_direction(self) -> None:
        self.seed_replay_events()
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=200&toAsOf=100"
        )
        self.assertEqual(status, 200)
        # From the later point back to the earlier one: the events that
        # leave the replay are reported as removed.
        self.assertEqual(body["added"], [])
        self.assertEqual(body["removed"], ["evt-a", "evt-b"])
        self.assertEqual(body["changed"], [])
        self.assertEqual(body["unchangedCount"], 2)

    def test_compare_equal_time_points_reports_all_unchanged(self) -> None:
        self.seed_replay_events()
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=200&toAsOf=200"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["added"], [])
        self.assertEqual(body["removed"], [])
        self.assertEqual(body["changed"], [])
        self.assertEqual(body["unchangedCount"], 4)

    def test_compare_unknown_organization_is_an_empty_diff(self) -> None:
        self.seed_replay_events()
        status, body = self.compare(
            "organizationId=org-x&fromAsOf=0&toAsOf=999"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-x",
                "fromAsOf": 0,
                "toAsOf": 999,
                "added": [],
                "removed": [],
                "changed": [],
                "unchangedCount": 0,
            },
        )

    def test_compare_identifier_groups_are_code_point_sorted(self) -> None:
        for event_id, occurred_at in (
            ("evt-b", 10),
            ("evt-A", 10),
            ("evt-a", 10),
            ("evt-ä", 10),
        ):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=occurred_at)
                )[0],
                201,
            )
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=0&toAsOf=10"
        )
        self.assertEqual(status, 200)
        # Unicode code-point order: A < a < b < ä (U+00E4).
        self.assertEqual(body["added"], ["evt-A", "evt-a", "evt-b", "evt-ä"])

    def test_compare_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_replay_events()
        status, raw, body = self.request_raw(
            "/events/replay/compare?organizationId=org-1&fromAsOf=100&toAsOf=200"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(
            set(body),
            {
                "organizationId",
                "fromAsOf",
                "toAsOf",
                "added",
                "removed",
                "changed",
                "unchangedCount",
            },
        )
        self.assertIsInstance(body["unchangedCount"], int)
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )

    def test_compare_repeated_requests_are_byte_identical(self) -> None:
        self.seed_replay_events()
        query = "organizationId=org-1&fromAsOf=100&toAsOf=200"
        raws = [self.request_raw(f"/events/replay/compare?{query}")[1] for _ in range(3)]
        self.assertEqual(raws[0], raws[1])
        self.assertEqual(raws[1], raws[2])

    def test_compare_is_read_only(self) -> None:
        self.seed_replay_events()
        self.compare("organizationId=org-1&fromAsOf=100&toAsOf=200")
        self.compare("organizationId=org-1&fromAsOf=200&toAsOf=100")
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 5)

    def test_compare_requires_each_parameter_exactly_once(self) -> None:
        base = "organizationId=org-1&fromAsOf=100&toAsOf=200"
        for query in (
            "",
            "fromAsOf=100&toAsOf=200",
            "organizationId=org-1&toAsOf=200",
            "organizationId=org-1&fromAsOf=100",
            f"{base}&organizationId=org-1",
            f"{base}&fromAsOf=100",
            f"{base}&toAsOf=200",
        ):
            with self.subTest(query=query):
                status, body = self.compare(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_compare_rejects_blank_and_invalid_values(self) -> None:
        for query in (
            "organizationId=&fromAsOf=1&toAsOf=2",
            "organizationId=%20&fromAsOf=1&toAsOf=2",
            "organizationId=org-1&fromAsOf=&toAsOf=2",
            "organizationId=org-1&fromAsOf=1&toAsOf=",
            "organizationId=org-1&fromAsOf=%20&toAsOf=2",
            "organizationId=org-1&fromAsOf=1&toAsOf=%20",
            "organizationId=org-1&fromAsOf=-1&toAsOf=2",
            "organizationId=org-1&fromAsOf=1&toAsOf=-2",
            "organizationId=org-1&fromAsOf=1.5&toAsOf=2",
            "organizationId=org-1&fromAsOf=1&toAsOf=abc",
        ):
            with self.subTest(query=query):
                status, body = self.compare(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_compare_validation_failure_writes_nothing(self) -> None:
        self.seed_replay_events()
        status, body = self.compare("organizationId=org-1&fromAsOf=x&toAsOf=2")
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 5)

    # ------------------------------------------------------------- branch prefix

    def test_branch_prefixes_do_not_expose_replay_queries(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        status, _, _ = self.request_raw(
            "/snapshots", method="POST", body=b'{"snapshotId":"snap-1"}'
        )
        self.assertEqual(status, 201)
        status, _, _ = self.request_raw(
            "/branches",
            method="POST",
            body=b'{"branchId":"br-1","snapshotId":"snap-1"}',
        )
        self.assertEqual(status, 201)

        # Known branch: standard not_found for the unknown sub-paths.
        status, _, body = self.request_raw(
            "/branches/br-1/events/replay?organizationId=org-1&asOf=100"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, _, body = self.request_raw(
            "/branches/br-1/events/replay/compare"
            "?organizationId=org-1&fromAsOf=0&toAsOf=100"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

        # Unknown branch keeps the branch_not_found precedence.
        status, _, body = self.request_raw(
            "/branches/ghost/events/replay?organizationId=org-1&asOf=100"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")


if __name__ == "__main__":
    unittest.main()
