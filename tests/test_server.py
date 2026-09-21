from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import SERVICE_NAME, create_server


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


class ServerTest(unittest.TestCase):
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

    def post_event(
        self, event: Any, *, raw: bool = False, content_type: str | None = "application/json"
    ) -> tuple[int, Any]:
        body = event if raw else json.dumps(event).encode()
        return self.request(
            "/events", method="POST", body=body, content_type=content_type
        )

    # --- pre-existing contract ------------------------------------------------

    def test_health(self) -> None:
        with urlopen(f"{self.base_url}/health", timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                json.load(response),
                {"service": SERVICE_NAME, "status": "ok"},
            )

    def test_unknown_path_is_json_404(self) -> None:
        with self.assertRaises(HTTPError) as raised:
            urlopen(f"{self.base_url}/missing", timeout=2)
        error = raised.exception
        try:
            self.assertEqual(error.code, 404)
            self.assertEqual(
                json.load(error),
                {"error": "not_found", "path": "/missing"},
            )
        finally:
            error.close()

    def test_post_unknown_path_is_json_404(self) -> None:
        status, body = self.request(
            "/missing", method="POST", body=b"{}", content_type="application/json"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    # --- POST /events ----------------------------------------------------------

    def test_create_event_returns_201_with_exact_fields(self) -> None:
        event = make_event(payload={"nested": {"a": [1, 2]}})
        status, body = self.post_event(event)
        self.assertEqual(status, 201)
        self.assertEqual(body, event)
        self.assertEqual(set(body), set(event))

    def test_content_type_with_charset_is_accepted(self) -> None:
        event = make_event()
        status, body = self.post_event(event, content_type="application/json; charset=utf-8")
        self.assertEqual(status, 201)
        self.assertEqual(body, event)

    def test_identical_replay_returns_200_without_duplicate(self) -> None:
        event = make_event()
        first_status, first_body = self.post_event(event)
        second_status, second_body = self.post_event(event)
        self.assertEqual(first_status, 201)
        self.assertEqual(second_status, 200)
        self.assertEqual(first_body, second_body)

        status, listing = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 1)

    def test_same_event_id_different_fields_is_conflict(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        for override in (
            {"organizationId": "org-2"},
            {"type": "incident.updated"},
            {"occurredAt": 101},
            {"payload": {"severity": "high"}},
        ):
            status, body = self.post_event(make_event(**override))
            self.assertEqual(status, 409)
            self.assertEqual(body["error"], "event_id_conflict")

        # The original event is unchanged.
        status, body = self.post_event(make_event())
        self.assertEqual(status, 200)
        self.assertEqual(body, make_event())

    # --- validation: content type / JSON --------------------------------------

    def test_missing_content_type_is_415(self) -> None:
        status, body = self.post_event(make_event(), content_type=None)
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_unsupported_content_type_is_415(self) -> None:
        status, body = self.post_event(make_event(), content_type="text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_malformed_json_is_400(self) -> None:
        status, body = self.post_event(b'{"eventId": ', raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    # --- validation: fields ----------------------------------------------------

    def test_array_body_is_422(self) -> None:
        status, body = self.post_event([make_event()])
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_missing_field_is_422(self) -> None:
        event = make_event()
        del event["payload"]
        status, body = self.post_event(event)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_extra_field_is_422(self) -> None:
        status, body = self.post_event(make_event(extra="nope"))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_blank_or_non_string_identifiers_are_422(self) -> None:
        for field in ("eventId", "organizationId", "type"):
            for bad_value in ("", "   ", 123, None, ["x"]):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_event(make_event(**{field: bad_value}))
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_bad_occurred_at_is_422(self) -> None:
        for bad_value in (-1, 1.5, "100", True, None):
            with self.subTest(bad_value=bad_value):
                status, body = self.post_event(make_event(occurredAt=bad_value))
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_zero_occurred_at_is_accepted(self) -> None:
        status, _ = self.post_event(make_event(occurredAt=0))
        self.assertEqual(status, 201)

    def test_non_object_payload_is_422(self) -> None:
        for bad_value in ([], "x", 1, None):
            with self.subTest(bad_value=bad_value):
                status, body = self.post_event(make_event(payload=bad_value))
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_empty_payload_object_is_accepted(self) -> None:
        status, body = self.post_event(make_event(payload={}))
        self.assertEqual(status, 201)
        self.assertEqual(body["payload"], {})

    # --- GET /events -----------------------------------------------------------

    def test_list_filters_by_organization_and_sorts_deterministically(self) -> None:
        events = [
            make_event(eventId="evt-b", organizationId="org-a", occurredAt=200),
            make_event(eventId="evt-a", organizationId="org-a", occurredAt=200),
            make_event(eventId="evt-c", organizationId="org-a", occurredAt=100),
            make_event(eventId="evt-x", organizationId="org-b", occurredAt=50),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

        status, body = self.request("/events?organizationId=org-a")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-a")
        self.assertEqual(
            body["events"],
            [
                make_event(eventId="evt-c", organizationId="org-a", occurredAt=100),
                make_event(eventId="evt-a", organizationId="org-a", occurredAt=200),
                make_event(eventId="evt-b", organizationId="org-a", occurredAt=200),
            ],
        )

    def test_list_unknown_organization_returns_empty(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        status, body = self.request("/events?organizationId=other")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"organizationId": "other", "events": []})

    def test_list_requires_organization_id(self) -> None:
        status, body = self.request("/events")
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_list_rejects_blank_organization_id(self) -> None:
        for query in ("/events?organizationId=", "/events?organizationId=%20%20"):
            with self.subTest(query=query):
                status, body = self.request(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_list_rejects_duplicate_organization_id(self) -> None:
        status, body = self.request("/events?organizationId=a&organizationId=b")
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_list_ignores_unrelated_query_parameters(self) -> None:
        # organizationId is still present exactly once; other params are fine.
        status, body = self.request("/events?organizationId=org-1&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")

    # --- GET /events/aggregate -------------------------------------------------

    def test_aggregate_without_range_returns_only_covered_windows(self) -> None:
        events = [
            make_event(eventId="evt-1", occurredAt=0),
            make_event(eventId="evt-2", occurredAt=99),
            make_event(eventId="evt-3", occurredAt=100),
            make_event(eventId="evt-4", occurredAt=350),
            make_event(eventId="evt-5", type="incident.updated", occurredAt=50),
            make_event(eventId="evt-6", organizationId="org-2", occurredAt=50),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 100,
                "from": None,
                "to": None,
                "windows": [
                    {"start": 0, "end": 100, "count": 2},
                    {"start": 100, "end": 200, "count": 1},
                    {"start": 300, "end": 400, "count": 1},
                ],
            },
        )

    def test_aggregate_window_boundaries_are_start_inclusive_end_exclusive(self) -> None:
        for occurred_at in (0, 99, 100, 199, 200):
            self.assertEqual(
                self.post_event(make_event(eventId=f"evt-{occurred_at}", occurredAt=occurred_at))[0],
                201,
            )

        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 100, "count": 2},
                {"start": 100, "end": 200, "count": 2},
                {"start": 200, "end": 300, "count": 1},
            ],
        )

    def test_aggregate_with_range_returns_all_intersecting_windows(self) -> None:
        for occurred_at in (0, 50, 250):
            self.assertEqual(
                self.post_event(make_event(eventId=f"evt-{occurred_at}", occurredAt=occurred_at))[0],
                201,
            )

        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created"
            "&windowSize=100&from=150&to=450"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 150)
        self.assertEqual(body["to"], 450)
        self.assertEqual(
            body["windows"],
            [
                {"start": 100, "end": 200, "count": 0},
                {"start": 200, "end": 300, "count": 1},
                {"start": 300, "end": 400, "count": 0},
                {"start": 400, "end": 500, "count": 0},
            ],
        )

    def test_aggregate_range_endpoints_are_inclusive(self) -> None:
        for occurred_at in (49, 50, 150, 151):
            self.assertEqual(
                self.post_event(make_event(eventId=f"evt-{occurred_at}", occurredAt=occurred_at))[0],
                201,
            )

        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created"
            "&windowSize=100&from=50&to=150"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 100, "count": 1},
                {"start": 100, "end": 200, "count": 1},
            ],
        )

    def test_aggregate_range_within_single_window(self) -> None:
        self.assertEqual(self.post_event(make_event(occurredAt=10))[0], 201)
        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created"
            "&windowSize=100&from=20&to=30"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [{"start": 0, "end": 100, "count": 0}])

    def test_aggregate_no_matches_without_range_is_empty(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)
        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=other&windowSize=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

        status, body = self.request(
            "/events/aggregate?organizationId=other&type=incident.created&windowSize=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

    def test_aggregate_no_matches_with_range_returns_empty_windows(self) -> None:
        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created"
            "&windowSize=100&from=0&to=50"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 0, "end": 100, "count": 0}],
        )

    def test_aggregate_does_not_leak_other_organizations_or_types(self) -> None:
        self.assertEqual(
            self.post_event(make_event(eventId="evt-1", organizationId="org-a", type="t1", occurredAt=10))[0],
            201,
        )
        self.assertEqual(
            self.post_event(make_event(eventId="evt-2", organizationId="org-b", type="t1", occurredAt=10))[0],
            201,
        )
        self.assertEqual(
            self.post_event(make_event(eventId="evt-3", organizationId="org-a", type="t2", occurredAt=10))[0],
            201,
        )
        status, body = self.request(
            "/events/aggregate?organizationId=org-a&type=t1&windowSize=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [{"start": 0, "end": 100, "count": 1}])

    def test_aggregate_is_deterministic_for_replays_and_same_timestamps(self) -> None:
        event = make_event(occurredAt=10)
        self.assertEqual(self.post_event(event)[0], 201)
        self.assertEqual(self.post_event(event)[0], 200)

        for _ in range(3):
            status, body = self.request(
                "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=100"
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["windows"], [{"start": 0, "end": 100, "count": 1}])

        # Events posted out of timestamp order still come back window-sorted.
        for occurred_at, event_id in ((300, "late"), (5, "early"), (150, "middle")):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))[0],
                201,
            )
        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=100"
        )
        self.assertEqual(
            [window["start"] for window in body["windows"]],
            [0, 100, 300],
        )
        self.assertEqual(body["windows"][0]["count"], 2)

    def test_aggregate_requires_all_three_core_parameters(self) -> None:
        valid = "organizationId=org-1&type=incident.created&windowSize=100"
        for query in (
            "/events/aggregate",
            "/events/aggregate?type=incident.created&windowSize=100",
            "/events/aggregate?organizationId=org-1&windowSize=100",
            "/events/aggregate?organizationId=org-1&type=incident.created",
        ):
            with self.subTest(query=query):
                status, body = self.request(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")
        # sanity: the fully valid query succeeds
        status, _ = self.request(f"/events/aggregate?{valid}")
        self.assertEqual(status, 200)

    def test_aggregate_rejects_blank_core_parameters(self) -> None:
        for query in (
            "/events/aggregate?organizationId=&type=t&windowSize=100",
            "/events/aggregate?organizationId=%20&type=t&windowSize=100",
            "/events/aggregate?organizationId=o&type=&windowSize=100",
            "/events/aggregate?organizationId=o&type=%20&windowSize=100",
        ):
            with self.subTest(query=query):
                status, body = self.request(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_rejects_duplicate_parameters(self) -> None:
        base = "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=100"
        for suffix in (
            "&organizationId=org-2",
            "&type=other",
            "&windowSize=200",
            "&from=0&from=1&to=5",
            "&from=0&to=5&to=6",
        ):
            with self.subTest(suffix=suffix):
                status, body = self.request(base + suffix)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_rejects_invalid_window_size(self) -> None:
        for raw in ("0", "-1", "1.5", "01", "1e2", "abc", "", "%205"):
            query = (
                "/events/aggregate?organizationId=org-1&type=incident.created"
                f"&windowSize={raw}"
            )
            with self.subTest(windowSize=raw):
                status, body = self.request(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_requires_from_and_to_together(self) -> None:
        base = "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=100"
        for suffix in ("&from=0", "&to=10", "&from=", "&to="):
            with self.subTest(suffix=suffix):
                status, body = self.request(base + suffix)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_validates_range_values(self) -> None:
        base = "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=100"
        for suffix in (
            "&from=-1&to=10",
            "&from=0&to=-1",
            "&from=5&to=4",
            "&from=1.5&to=10",
            "&from=0&to=1.5",
            "&from=01&to=10",
            "&from=abc&to=10",
        ):
            with self.subTest(suffix=suffix):
                status, body = self.request(base + suffix)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_from_equal_to_to_is_valid(self) -> None:
        self.assertEqual(self.post_event(make_event(occurredAt=100))[0], 201)
        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created"
            "&windowSize=100&from=100&to=100"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [{"start": 100, "end": 200, "count": 1}])

    def test_aggregate_ignores_unrelated_query_parameters(self) -> None:
        # Matches GET /events: unrelated params are ignored as long as the
        # documented parameters are present and valid.
        status, body = self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created"
            "&windowSize=100&extra=1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

    def test_aggregate_is_read_only(self) -> None:
        self.assertEqual(
            self.post_event(make_event(eventId="evt-1", occurredAt=10))[0], 201
        )
        self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=100"
        )
        self.request(
            "/events/aggregate?organizationId=org-1&type=incident.created"
            "&windowSize=100&from=0&to=1000"
        )
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)

    # --- concurrency and isolation ---------------------------------------------

    def test_concurrent_identical_posts_create_once(self) -> None:
        event = make_event()
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda _: self.post_event(event), range(24)))

        statuses = sorted(status for status, _ in results)
        self.assertEqual(statuses.count(201), 1)
        self.assertEqual(statuses.count(200), 23)

        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(len(body["events"]), 1)
        self.assertEqual(body["events"][0], event)

    def test_new_server_instance_does_not_inherit_data(self) -> None:
        self.assertEqual(self.post_event(make_event())[0], 201)

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(
                f"http://127.0.0.1:{fresh.server_port}/events?organizationId=org-1",
                timeout=2,
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    json.load(response),
                    {"organizationId": "org-1", "events": []},
                )
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
