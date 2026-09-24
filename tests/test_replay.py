from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import compare_replays, create_server


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

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request_raw(
        self, path: str, *, method: str = "GET", body: bytes | None = None
    ) -> tuple[int, bytes, Any]:
        headers = {"Content-Type": "application/json"} if body is not None else {}
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

    def seed_events(self) -> None:
        events = [
            make_event(eventId="evt-b", occurredAt=100),
            make_event(eventId="evt-a", occurredAt=100),
            make_event(eventId="evt-c", occurredAt=50),
            make_event(eventId="evt-d", occurredAt=0),
            make_event(eventId="evt-e", occurredAt=200),
            make_event(eventId="evt-o", organizationId="org-2", occurredAt=10),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

    # --------------------------------------------------------- GET /events/replay

    def test_replay_lists_events_at_or_before_as_of_sorted(self) -> None:
        self.seed_events()
        status, body = self.replay("organizationId=org-1&asOf=100")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(body["asOf"], 100)
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["evt-d", "evt-c", "evt-a", "evt-b"],
        )

    def test_replay_events_are_the_five_field_records_verbatim(self) -> None:
        event = make_event(payload={"region": "north", "k": [1, 2]})
        self.assertEqual(self.post_event(event)[0], 201)
        status, body = self.replay("organizationId=org-1&asOf=100")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [event])

    def test_replay_as_of_zero_and_boundary_inclusion(self) -> None:
        self.seed_events()
        status, body = self.replay("organizationId=org-1&asOf=0")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]], ["evt-d"]
        )
        # occurredAt equal to asOf is included; later events are not.
        status, body = self.replay("organizationId=org-1&asOf=199")
        self.assertEqual(status, 200)
        self.assertNotIn(
            "evt-e", [event["eventId"] for event in body["events"]]
        )
        status, body = self.replay("organizationId=org-1&asOf=200")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 5)

    def test_replay_unknown_organization_returns_empty_events(self) -> None:
        self.seed_events()
        status, body = self.replay("organizationId=org-x&asOf=1000")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-x")
        self.assertEqual(body["events"], [])
        self.assertNotIn("org-2", json.dumps(body))

    def test_replay_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_events()
        status, raw, body = self.request_raw(
            "/events/replay?organizationId=org-1&asOf=100"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(set(body), {"organizationId", "asOf", "events"})
        self.assertIsInstance(body["asOf"], int)
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )

    def test_replay_rejects_missing_duplicate_blank_and_non_integer_as_of(self) -> None:
        for query in (
            "asOf=10",
            "organizationId=org-1",
            "organizationId=&asOf=10",
            "organizationId=%20&asOf=10",
            "organizationId=org-1&asOf=",
            "organizationId=org-1&asOf=%20",
            "organizationId=org-1&asOf=-1",
            "organizationId=org-1&asOf=1.5",
            "organizationId=org-1&asOf=abc",
            "organizationId=org-1&asOf=10&asOf=20",
            "organizationId=a&organizationId=b&asOf=10",
        ):
            with self.subTest(query=query):
                status, body = self.replay(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_replay_is_read_only_and_deterministic(self) -> None:
        self.seed_events()
        first_status, first = self.replay("organizationId=org-1&asOf=100")
        second_status, second = self.replay("organizationId=org-1&asOf=100")
        self.assertEqual((first_status, second_status), (200, 200))
        self.assertEqual(first, second)
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 5)

    # ------------------------------------------------- GET /events/replay/compare

    def test_compare_groups_added_removed_changed_and_unchanged(self) -> None:
        # Baseline replay at t=100: evt-a, evt-b, evt-c.
        for event in (
            make_event(eventId="evt-a", occurredAt=10),
            make_event(eventId="evt-b", occurredAt=20),
            make_event(eventId="evt-c", occurredAt=100),
            make_event(eventId="evt-d", occurredAt=150),
            make_event(eventId="evt-e", occurredAt=200),
        ):
            self.assertEqual(self.post_event(event)[0], 201)
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=100&toAsOf=200"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(body["fromAsOf"], 100)
        self.assertEqual(body["toAsOf"], 200)
        self.assertEqual(body["added"], ["evt-d", "evt-e"])
        self.assertEqual(body["removed"], [])
        self.assertEqual(body["changed"], [])
        self.assertEqual(body["unchangedCount"], 3)

    def test_compare_time_points_may_appear_in_either_order(self) -> None:
        for event in (
            make_event(eventId="evt-a", occurredAt=10),
            make_event(eventId="evt-b", occurredAt=200),
        ):
            self.assertEqual(self.post_event(event)[0], 201)
        # from later than to: "added" is still relative to the to replay.
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=200&toAsOf=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["added"], [])
        self.assertEqual(body["removed"], ["evt-b"])
        self.assertEqual(body["unchangedCount"], 1)

    def test_compare_detects_changed_fields_and_counts_identical(self) -> None:
        # The ledger keys events by eventId and rejects conflicting
        # resubmissions, so two HTTP replays of one ledger always agree on
        # shared ids; the changed-group semantics live in compare_replays.
        earlier = make_event(eventId="evt-same", occurredAt=10)
        later = make_event(eventId="evt-same", occurredAt=10, payload={"n": 2})
        diff = compare_replays([earlier], [later])
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["changed"], ["evt-same"])
        self.assertEqual(diff["unchangedCount"], 0)
        # eventId itself is the identity and never counts as a field change.
        diff = compare_replays([earlier], [dict(earlier)])
        self.assertEqual(diff["changed"], [])
        self.assertEqual(diff["unchangedCount"], 1)

        self.assertEqual(self.post_event(make_event(eventId="evt-a"))[0], 201)
        self.assertEqual(
            self.post_event(
                make_event(eventId="evt-b", occurredAt=50, payload={"n": 1})
            )[0],
            201,
        )
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=100&toAsOf=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["added"], [])
        self.assertEqual(body["removed"], [])
        self.assertEqual(body["changed"], [])
        self.assertEqual(body["unchangedCount"], 2)

    def test_compare_identifiers_sorted_by_code_point(self) -> None:
        for event_id in ("evt-10", "evt-2", "Evt-1", "evt-a"):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=500))[0],
                201,
            )
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=0&toAsOf=1000"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["added"], ["Evt-1", "evt-10", "evt-2", "evt-a"])
        status, body = self.compare(
            "organizationId=org-1&fromAsOf=1000&toAsOf=0"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["removed"], ["Evt-1", "evt-10", "evt-2", "evt-a"])

    def test_compare_unknown_organization_is_zero_events(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        status, body = self.compare(
            "organizationId=org-x&fromAsOf=0&toAsOf=1000"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["added"], [])
        self.assertEqual(body["removed"], [])
        self.assertEqual(body["changed"], [])
        self.assertEqual(body["unchangedCount"], 0)
        self.assertNotIn("org-1", json.dumps(body["added"]))

    def test_compare_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        status, raw, body = self.request_raw(
            "/events/replay/compare?organizationId=org-1&fromAsOf=0&toAsOf=100"
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
        # Repeated identical requests are byte-for-byte identical.
        status, raw2, _ = self.request_raw(
            "/events/replay/compare?organizationId=org-1&fromAsOf=0&toAsOf=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw, raw2)

    def test_compare_rejects_bad_parameters(self) -> None:
        base = "organizationId=org-1&fromAsOf=0&toAsOf=10"
        for query in (
            "fromAsOf=0&toAsOf=10",
            "organizationId=org-1&toAsOf=10",
            "organizationId=org-1&fromAsOf=0",
            "organizationId=&fromAsOf=0&toAsOf=10",
            "organizationId=org-1&fromAsOf=&toAsOf=10",
            "organizationId=org-1&fromAsOf=0&toAsOf=%20",
            "organizationId=org-1&fromAsOf=-1&toAsOf=10",
            "organizationId=org-1&fromAsOf=0&toAsOf=1.5",
            "organizationId=org-1&fromAsOf=x&toAsOf=10",
            f"{base}&fromAsOf=1",
            f"{base}&toAsOf=2",
            f"{base}&organizationId=org-2",
        ):
            with self.subTest(query=query):
                status, body = self.compare(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_compare_is_read_only(self) -> None:
        self.seed_events()
        status, _ = self.compare("organizationId=org-1&fromAsOf=0&toAsOf=200")
        self.assertEqual(status, 200)
        status, _, listing = self.request_raw("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 5)
        status, _, alerts = self.request_raw("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

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

        status, _, body = self.request_raw(
            "/branches/ghost/events/replay?organizationId=org-1&asOf=100"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")


if __name__ == "__main__":
    unittest.main()
