from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server


def make_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "eventId": "evt-1",
        "organizationId": "org-1",
        "type": "incident.created",
        "occurredAt": 100,
        "payload": {"severity": "low", "region": "us-east"},
    }
    event.update(overrides)
    return event


class RegionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, Any]:
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            try:
                return error.code, json.load(error)
            finally:
                error.close()

    def raw_request(self, path: str) -> tuple[int, bytes]:
        with urlopen(f"{self.base_url}{path}", timeout=5) as response:
            return response.status, response.read()

    def post_event(self, event: Any) -> tuple[int, Any]:
        return self.request(
            "/events",
            method="POST",
            body=json.dumps(event).encode(),
        )

    def seed_region_events(self) -> None:
        events = [
            make_event(eventId="evt-1", occurredAt=0),
            make_event(eventId="evt-2", occurredAt=59),
            make_event(eventId="evt-3", occurredAt=60),
            make_event(eventId="evt-4", occurredAt=60),
            make_event(eventId="evt-5", occurredAt=180),
            # Same organization, different region.
            make_event(
                eventId="evt-6", occurredAt=10, payload={"region": "eu-west"}
            ),
            # Same organization, no region attribution.
            make_event(eventId="evt-7", occurredAt=20, payload={}),
            make_event(eventId="evt-8", occurredAt=30, payload={"region": ""}),
            make_event(eventId="evt-9", occurredAt=40, payload={"region": 7}),
            make_event(eventId="evt-10", occurredAt=50, payload={"region": None}),
            # Same region, different organization.
            make_event(eventId="evt-11", organizationId="org-2", occurredAt=10),
            # Same organization and region, different type.
            make_event(eventId="evt-12", type="incident.updated", occurredAt=10),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

    # --- GET /events/region ----------------------------------------------------

    def region_list(self, query: str) -> tuple[int, Any]:
        return self.request(f"/events/region?{query}")

    def test_region_list_filters_and_sorts_like_plain_listing(self) -> None:
        self.seed_region_events()
        status, body = self.region_list("organizationId=org-1&region=us-east")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(body["region"], "us-east")
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["evt-1", "evt-12", "evt-2", "evt-3", "evt-4", "evt-5"],
        )
        # Full five-field events, exactly as stored.
        self.assertEqual(body["events"][0], make_event(eventId="evt-1", occurredAt=0))

    def test_region_list_tie_breaks_by_event_id(self) -> None:
        for event_id in ("evt-b", "evt-a"):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=100))[0], 201
            )
        status, body = self.region_list("organizationId=org-1&region=us-east")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]], ["evt-a", "evt-b"]
        )

    def test_region_list_unknown_region_returns_empty(self) -> None:
        self.seed_region_events()
        status, body = self.region_list("organizationId=org-1&region=ap-south")
        self.assertEqual(status, 200)
        self.assertEqual(
            body, {"organizationId": "org-1", "region": "ap-south", "events": []}
        )

    def test_region_list_unknown_organization_returns_empty(self) -> None:
        self.seed_region_events()
        status, body = self.region_list("organizationId=org-x&region=us-east")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])

    def test_region_list_never_leaks_other_organizations(self) -> None:
        self.seed_region_events()
        status, body = self.region_list("organizationId=org-2&region=us-east")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]], ["evt-11"]
        )

    def test_region_list_unattributed_events_are_excluded(self) -> None:
        # Missing key, empty string, and non-string payload regions never
        # match, even though they belong to the same organization.
        self.seed_region_events()
        status, body = self.region_list("organizationId=org-1&region=us-east")
        self.assertEqual(status, 200)
        returned = {event["eventId"] for event in body["events"]}
        self.assertNotIn("evt-7", returned)  # no region key
        self.assertNotIn("evt-8", returned)  # empty string
        self.assertNotIn("evt-9", returned)  # non-string
        self.assertNotIn("evt-10", returned)  # null

    def test_region_matching_is_exact(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        for region in ("US-EAST", "us-east ", " us-east", "us-east-1"):
            with self.subTest(region=region):
                status, body = self.region_list(
                    f"organizationId=org-1&region={region.replace(' ', '%20')}"
                )
                self.assertEqual(status, 200)
                self.assertEqual(body["events"], [])

    def test_region_list_response_is_compact_sorted_and_newline_terminated(
        self,
    ) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        status, raw = self.raw_request("/events/region?organizationId=org-1&region=us-east")
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(
            raw[:-1],
            json.dumps(
                json.loads(raw), separators=(",", ":"), sort_keys=True
            ).encode(),
        )
        self.assertEqual(
            json.loads(raw),
            {
                "organizationId": "org-1",
                "region": "us-east",
                "events": [make_event()],
            },
        )

    def test_region_list_requires_each_parameter_exactly_once(self) -> None:
        for query in (
            "",
            "organizationId=org-1",
            "region=us-east",
            "organizationId=org-1&region=us-east&region=us-east",
            "organizationId=a&organizationId=b&region=us-east",
        ):
            with self.subTest(query=query):
                status, body = self.region_list(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_region_list_rejects_blank_values(self) -> None:
        for query in (
            "organizationId=&region=us-east",
            "organizationId=%20%20&region=us-east",
            "organizationId=org-1&region=",
            "organizationId=org-1&region=%20%20",
        ):
            with self.subTest(query=query):
                status, body = self.region_list(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_region_list_ignores_unrelated_query_parameters(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        status, body = self.region_list(
            "organizationId=org-1&region=us-east&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)

    def test_region_list_is_read_only(self) -> None:
        self.seed_region_events()
        self.region_list("organizationId=org-1&region=us-east")
        self.region_list("organizationId=org-1&region=no-such")
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 11)

    # --- GET /events/region/aggregate ------------------------------------------

    def region_aggregate(self, query: str) -> tuple[int, Any]:
        return self.request(f"/events/region/aggregate?{query}")

    def test_region_aggregate_without_range_returns_only_covered_windows(
        self,
    ) -> None:
        self.seed_region_events()
        status, body = self.region_aggregate(
            "organizationId=org-1&region=us-east&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "region": "us-east",
                "type": "incident.created",
                "windowSize": 60,
                "from": None,
                "to": None,
                "windows": [
                    {"start": 0, "end": 60, "count": 2},
                    {"start": 60, "end": 120, "count": 2},
                    {"start": 180, "end": 240, "count": 1},
                ],
            },
        )

    def test_region_aggregate_with_range_keeps_empty_windows(self) -> None:
        self.seed_region_events()
        status, body = self.region_aggregate(
            "organizationId=org-1&region=us-east&type=incident.created"
            "&windowSize=60&from=59&to=180"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 59)
        self.assertEqual(body["to"], 180)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 2},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 1},
            ],
        )

    def test_region_aggregate_counts_only_matching_region_and_type(self) -> None:
        self.seed_region_events()
        # eu-west has a single event at 10; the us-east events and the
        # unattributed ones must not leak in.
        status, body = self.region_aggregate(
            "organizationId=org-1&region=eu-west&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [{"start": 0, "end": 60, "count": 1}])

    def test_region_aggregate_unknown_region_behaves_like_no_match(self) -> None:
        self.seed_region_events()
        status, body = self.region_aggregate(
            "organizationId=org-1&region=no-such&type=incident.created&windowSize=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

        status, body = self.region_aggregate(
            "organizationId=org-1&region=no-such&type=incident.created"
            "&windowSize=10&from=5&to=25"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 10, "count": 0},
                {"start": 10, "end": 20, "count": 0},
                {"start": 20, "end": 30, "count": 0},
            ],
        )

    def test_region_aggregate_never_counts_other_organizations(self) -> None:
        self.seed_region_events()
        status, body = self.region_aggregate(
            "organizationId=org-2&region=us-east&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [{"start": 0, "end": 60, "count": 1}])

    def test_region_aggregate_boundary_event_belongs_to_upper_window(self) -> None:
        self.seed_region_events()
        status, body = self.region_aggregate(
            "organizationId=org-1&region=us-east&type=incident.created"
            "&windowSize=60&from=60&to=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 60, "end": 120, "count": 2}]
        )

    def test_region_aggregate_response_is_compact_sorted_and_newline_terminated(
        self,
    ) -> None:
        self.seed_region_events()
        status, raw = self.raw_request(
            "/events/region/aggregate?organizationId=org-1&region=us-east"
            "&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(
            raw[:-1],
            json.dumps(
                json.loads(raw), separators=(",", ":"), sort_keys=True
            ).encode(),
        )

    def test_region_aggregate_is_deterministic_across_insertion_order(self) -> None:
        for event_id, occurred_at in (("evt-b", 60), ("evt-a", 60), ("evt-c", 0)):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))[0],
                201,
            )
        query = (
            "organizationId=org-1&region=us-east&type=incident.created&windowSize=60"
        )
        first = self.region_aggregate(query)
        second = self.region_aggregate(query)
        self.assertEqual(first, second)
        self.assertEqual(
            first[1]["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 2},
            ],
        )

    def test_region_aggregate_is_read_only(self) -> None:
        self.seed_region_events()
        self.region_aggregate(
            "organizationId=org-1&region=us-east&type=incident.created&windowSize=60"
        )
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 11)

    def test_region_aggregate_requires_each_parameter_exactly_once(self) -> None:
        base = (
            "organizationId=org-1&region=us-east&type=incident.created&windowSize=60"
        )
        for query in (
            "",
            "organizationId=org-1&region=us-east&type=incident.created",
            "organizationId=org-1&region=us-east&windowSize=60",
            "organizationId=org-1&type=incident.created&windowSize=60",
            "region=us-east&type=incident.created&windowSize=60",
            f"{base}&windowSize=60",
            f"{base}&type=incident.created",
            f"{base}&region=us-east",
            f"{base}&organizationId=org-1",
        ):
            with self.subTest(query=query):
                status, body = self.region_aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_region_aggregate_rejects_blank_and_invalid_values(self) -> None:
        for query in (
            "organizationId=&region=r&type=t&windowSize=1",
            "organizationId=%20&region=r&type=t&windowSize=1",
            "organizationId=o&region=&type=t&windowSize=1",
            "organizationId=o&region=%20%20&type=t&windowSize=1",
            "organizationId=o&region=r&type=&windowSize=1",
            "organizationId=o&region=r&type=t&windowSize=",
            "organizationId=o&region=r&type=t&windowSize=0",
            "organizationId=o&region=r&type=t&windowSize=-5",
            "organizationId=o&region=r&type=t&windowSize=1.5",
            "organizationId=o&region=r&type=t&windowSize=abc",
            "organizationId=o&region=r&type=t&windowSize=+5",
        ):
            with self.subTest(query=query):
                status, body = self.region_aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_region_aggregate_from_to_must_be_paired_valid_and_ordered(self) -> None:
        base = "organizationId=o&region=r&type=t&windowSize=10"
        for query in (
            f"{base}&from=0",
            f"{base}&to=0",
            f"{base}&from=0&from=1&to=2",
            f"{base}&from=0&to=1&to=2",
            f"{base}&from=-1&to=2",
            f"{base}&from=0&to=x",
            f"{base}&from=&to=2",
            f"{base}&from=10&to=5",
        ):
            with self.subTest(query=query):
                status, body = self.region_aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_region_aggregate_accepts_equal_from_and_to(self) -> None:
        status, body = self.region_aggregate(
            "organizationId=o&region=r&type=t&windowSize=10&from=0&to=0"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [{"start": 0, "end": 10, "count": 0}])

    # --- isolation and lifecycle -------------------------------------------------

    def test_region_queries_write_nothing_anywhere(self) -> None:
        self.seed_region_events()
        self.region_list("organizationId=org-1&region=us-east")
        self.region_aggregate(
            "organizationId=org-1&region=us-east&type=incident.created&windowSize=60"
        )
        # Failed queries write nothing either.
        self.region_list("organizationId=org-1")
        self.region_aggregate("organizationId=org-1&region=us-east")
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(len(body["events"]), 11)
        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(body["reservations"], [])
        status, body = self.request("/alerts?organizationId=org-1")
        self.assertEqual(body["alerts"], [])

    def test_new_server_instance_has_no_region_attribution(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(
                f"http://127.0.0.1:{fresh.server_port}"
                "/events/region?organizationId=org-1&region=us-east",
                timeout=2,
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    json.load(response),
                    {"organizationId": "org-1", "region": "us-east", "events": []},
                )
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    def test_branch_prefix_has_no_region_routes(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        status, _ = self.request(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": "snap-1"}).encode(),
        )
        self.assertEqual(status, 201)
        status, _ = self.request(
            "/branches",
            method="POST",
            body=json.dumps({"branchId": "br-1", "snapshotId": "snap-1"}).encode(),
        )
        self.assertEqual(status, 201)
        for path in (
            "/branches/br-1/events/region?organizationId=org-1&region=us-east",
            "/branches/br-1/events/region/aggregate?organizationId=org-1"
            "&region=us-east&type=incident.created&windowSize=60",
        ):
            with self.subTest(path=path):
                status, body = self.request(path)
                self.assertEqual(status, 404)
                self.assertEqual(body["error"], "not_found")

    # --- write-endpoint success-body regression -----------------------------------

    def test_write_endpoint_success_bodies_are_unchanged(self) -> None:
        # POST /events: 201 then identical-replay 200 carry the same body.
        event = make_event()
        status, body = self.post_event(event)
        self.assertEqual(status, 201)
        self.assertEqual(body, event)
        status, body = self.post_event(event)
        self.assertEqual(status, 200)
        self.assertEqual(body, event)

        # POST /reservations: 201 carries the balance view.
        reservation = {
            "organizationId": "org-1",
            "reservationId": "res-1",
            "resourceId": "r-a",
            "quantity": 2,
            "capacity": 5,
        }
        status, body = self.request(
            "/reservations", method="POST", body=json.dumps(reservation).encode()
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "reservationId": "res-1",
                "resourceId": "r-a",
                "quantity": 2,
                "capacity": 5,
                "occupied": 2,
                "remaining": 3,
            },
        )

        # POST /decisions/evaluate: 200 carries the fixed decision shape.
        status, body = self.request(
            "/decisions/evaluate",
            method="POST",
            body=json.dumps(
                {
                    "organizationId": "org-1",
                    "type": "incident.created",
                    "windowSize": 60,
                    "threshold": 3,
                }
            ).encode(),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "from": None,
                "to": None,
                "peakStart": 60,
                "peakCount": 1,
                "action": "observe",
            },
        )

        # POST /decisions/allocate: 200 carries the fixed plan shape.
        status, body = self.request(
            "/decisions/allocate",
            method="POST",
            body=json.dumps(
                {
                    "organizationId": "org-1",
                    "demands": [{"demandId": "d", "units": 1, "priority": 0}],
                    "resources": [{"resourceId": "r", "capacity": 1}],
                }
            ).encode(),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "assignments": [{"demandId": "d", "resourceId": "r", "units": 1}],
                "unassigned": [],
                "totalUnits": 1,
            },
        )

        # POST /alerts/evaluate: 200 carries the fixed alert shape.
        status, body = self.request(
            "/alerts/evaluate",
            method="POST",
            body=json.dumps(
                {
                    "organizationId": "org-1",
                    "type": "incident.created",
                    "windowSize": 60,
                    "threshold": 1,
                    "suppressionWindow": 120,
                }
            ).encode(),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "threshold": 1,
                "suppressionWindow": 120,
                "from": None,
                "to": None,
                "peakStart": 60,
                "peakCount": 1,
                "action": "escalate",
                "alertId": "alert-1",
                "suppressedCount": 0,
            },
        )

        # POST /snapshots and POST /branches: 201 summaries unchanged, and
        # snapshots capture only events, capacities, and reservations.
        status, body = self.request(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": "snap-1"}).encode(),
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "snapshotId": "snap-1",
                "events": 1,
                "resources": 1,
                "reservations": 1,
            },
        )
        status, body = self.request(
            "/branches",
            method="POST",
            body=json.dumps({"branchId": "br-1", "snapshotId": "snap-1"}).encode(),
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "branchId": "br-1",
                "snapshotId": "snap-1",
                "events": 1,
                "resources": 1,
                "reservations": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
