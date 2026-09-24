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

    def post_decision(
        self, payload: Any, *, raw: bool = False, content_type: str | None = "application/json"
    ) -> tuple[int, Any]:
        body = payload if raw else json.dumps(payload).encode()
        return self.request(
            "/decisions/evaluate", method="POST", body=body, content_type=content_type
        )

    def post_allocation(
        self, payload: Any, *, raw: bool = False, content_type: str | None = "application/json"
    ) -> tuple[int, Any]:
        body = payload if raw else json.dumps(payload).encode()
        return self.request(
            "/decisions/allocate", method="POST", body=body, content_type=content_type
        )

    def seed_aggregate_events(self) -> None:
        events = [
            make_event(eventId="evt-1", occurredAt=0),
            make_event(eventId="evt-2", occurredAt=59),
            make_event(eventId="evt-3", occurredAt=60),
            make_event(eventId="evt-4", occurredAt=60),
            make_event(eventId="evt-5", occurredAt=180),
            make_event(eventId="evt-6", type="incident.updated", occurredAt=10),
            make_event(eventId="evt-7", organizationId="org-2", occurredAt=10),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

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

    def aggregate(self, query: str) -> tuple[int, Any]:
        return self.request(f"/events/aggregate?{query}")

    def test_aggregate_without_range_returns_only_covered_windows(self) -> None:
        self.seed_aggregate_events()
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60"
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
                "windows": [
                    {"start": 0, "end": 60, "count": 2},
                    {"start": 60, "end": 120, "count": 2},
                    {"start": 180, "end": 240, "count": 1},
                ],
            },
        )

    def test_aggregate_with_range_returns_empty_windows_and_filters(self) -> None:
        self.seed_aggregate_events()
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60&from=59&to=180"
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

    def test_aggregate_boundary_event_belongs_to_upper_window(self) -> None:
        self.seed_aggregate_events()
        status, body = self.aggregate(
            "organizationId=org-1&type=incident.created&windowSize=60&from=60&to=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 60, "end": 120, "count": 2}]
        )

    def test_aggregate_no_matching_events_without_range_is_empty(self) -> None:
        self.seed_aggregate_events()
        status, body = self.aggregate(
            "organizationId=org-1&type=no.such.type&windowSize=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

    def test_aggregate_no_matching_events_with_range_returns_zero_windows(self) -> None:
        self.seed_aggregate_events()
        status, body = self.aggregate(
            "organizationId=org-1&type=no.such.type&windowSize=10&from=5&to=25"
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

    def test_aggregate_does_not_modify_ledger(self) -> None:
        self.seed_aggregate_events()
        self.aggregate("organizationId=org-1&type=incident.created&windowSize=60")
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 6)

    def test_aggregate_is_deterministic_across_insertion_order(self) -> None:
        for event_id, occurred_at in (("evt-b", 60), ("evt-a", 60), ("evt-c", 0)):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))[0],
                201,
            )
        first = self.aggregate("organizationId=org-1&type=incident.created&windowSize=60")
        second = self.aggregate("organizationId=org-1&type=incident.created&windowSize=60")
        self.assertEqual(first, second)
        self.assertEqual(
            first[1]["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 2},
            ],
        )

    def test_aggregate_requires_each_parameter_exactly_once(self) -> None:
        base = "organizationId=org-1&type=incident.created&windowSize=60"
        for query in (
            "",
            "organizationId=org-1&type=incident.created",
            "organizationId=org-1&windowSize=60",
            "type=incident.created&windowSize=60",
            f"{base}&windowSize=60",
            f"{base}&type=incident.created",
            f"{base}&organizationId=org-1",
        ):
            with self.subTest(query=query):
                status, body = self.aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_rejects_blank_and_invalid_values(self) -> None:
        for query in (
            "organizationId=&type=t&windowSize=1",
            "organizationId=%20&type=t&windowSize=1",
            "organizationId=o&type=&windowSize=1",
            "organizationId=o&type=t&windowSize=",
            "organizationId=o&type=t&windowSize=0",
            "organizationId=o&type=t&windowSize=-5",
            "organizationId=o&type=t&windowSize=1.5",
            "organizationId=o&type=t&windowSize=abc",
            "organizationId=o&type=t&windowSize=+5",
        ):
            with self.subTest(query=query):
                status, body = self.aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_from_to_must_be_paired_valid_and_ordered(self) -> None:
        base = "organizationId=o&type=t&windowSize=10"
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
                status, body = self.aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_accepts_equal_from_and_to(self) -> None:
        status, body = self.aggregate(
            "organizationId=o&type=t&windowSize=10&from=0&to=0"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [{"start": 0, "end": 10, "count": 0}])

    # --- POST /decisions/evaluate ---------------------------------------------

    def decision_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "type": "incident.created",
            "windowSize": 60,
            "threshold": 3,
        }
        payload.update(overrides)
        return payload

    def test_decision_observe_when_peak_below_threshold(self) -> None:
        self.seed_aggregate_events()
        status, body = self.post_decision(self.decision_payload(threshold=3))
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "from": None,
                "to": None,
                "peakStart": 0,
                "peakCount": 2,
                "action": "observe",
            },
        )

    def test_decision_escalate_when_peak_reaches_threshold(self) -> None:
        self.seed_aggregate_events()
        status, body = self.post_decision(self.decision_payload(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakStart"], 0)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "escalate")

    def test_decision_tie_chooses_earliest_window_start(self) -> None:
        # Two events at 10 and two at 130: windows 0 and 120 tie at count 2.
        for event_id, occurred_at in (
            ("evt-a", 10),
            ("evt-b", 11),
            ("evt-c", 130),
            ("evt-d", 131),
        ):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=occurred_at)
                )[0],
                201,
            )
        status, body = self.post_decision(self.decision_payload(threshold=5))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakStart"], 0)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "observe")

    def test_decision_no_matching_events_is_zero_observe(self) -> None:
        self.seed_aggregate_events()
        status, body = self.post_decision(self.decision_payload(type="no.such.type"))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")

    def test_decision_unknown_organization_is_isolated_zero_result(self) -> None:
        self.seed_aggregate_events()
        status, body = self.post_decision(self.decision_payload(organizationId="org-x"))
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-x")
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")

    def test_decision_with_range_filters_and_keeps_empty_windows(self) -> None:
        self.seed_aggregate_events()
        payload = self.decision_payload(
            windowSize=60, threshold=3, **{"from": 59, "to": 180}
        )
        status, body = self.post_decision(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 59)
        self.assertEqual(body["to"], 180)
        # Counts over [59,180]: window 0 -> 1 (59), 60 -> 2, 120 -> 0, 180 -> 1.
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "observe")

    def test_decision_range_with_only_empty_windows_is_zero_observe(self) -> None:
        self.seed_aggregate_events()
        payload = self.decision_payload(
            type="no.such.type", windowSize=10, threshold=1, **{"from": 5, "to": 25}
        )
        status, body = self.post_decision(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")

    def test_decision_boundary_event_belongs_to_upper_window(self) -> None:
        self.seed_aggregate_events()
        payload = self.decision_payload(
            windowSize=60, threshold=2, **{"from": 60, "to": 60}
        )
        status, body = self.post_decision(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "escalate")

    def test_decision_equal_from_and_to_accepted(self) -> None:
        status, body = self.post_decision(
            self.decision_payload(
                windowSize=10, threshold=1, **{"from": 0, "to": 0}
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 0)
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])

    def test_decision_deterministic_across_insertion_order(self) -> None:
        for event_id, occurred_at in (("evt-b", 60), ("evt-a", 60), ("evt-c", 0)):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))[0],
                201,
            )
        payload = self.decision_payload(threshold=5)
        first = self.post_decision(payload)
        second = self.post_decision(payload)
        self.assertEqual(first, second)
        self.assertEqual(first[1]["peakStart"], 60)
        self.assertEqual(first[1]["peakCount"], 2)

    def test_decision_does_not_modify_ledger(self) -> None:
        self.seed_aggregate_events()
        self.post_decision(self.decision_payload())
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 6)

    def test_decision_missing_content_type_is_415(self) -> None:
        status, body = self.post_decision(self.decision_payload(), content_type=None)
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_decision_unsupported_content_type_is_415(self) -> None:
        status, body = self.post_decision(
            self.decision_payload(), content_type="text/plain"
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_decision_malformed_json_is_400(self) -> None:
        status, body = self.post_decision(b'{"organizationId": ', raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_decision_array_body_is_422(self) -> None:
        status, body = self.post_decision([self.decision_payload()])
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_decision_missing_required_field_is_422(self) -> None:
        for field in ("organizationId", "type", "windowSize", "threshold"):
            payload = self.decision_payload()
            del payload[field]
            with self.subTest(field=field):
                status, body = self.post_decision(payload)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_decision_extra_field_is_422(self) -> None:
        status, body = self.post_decision(self.decision_payload(extra="nope"))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_decision_blank_or_non_string_identifiers_are_422(self) -> None:
        for field in ("organizationId", "type"):
            for bad_value in ("", "   ", 123, None, ["x"], True):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_decision(
                        self.decision_payload(**{field: bad_value})
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_decision_bad_positive_integer_fields_are_422(self) -> None:
        for field in ("windowSize", "threshold"):
            for bad_value in (0, -1, 1.5, "60", True, None, [3]):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_decision(
                        self.decision_payload(**{field: bad_value})
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_decision_from_to_must_be_paired(self) -> None:
        for overrides in ({"from": 0}, {"to": 0}):
            with self.subTest(overrides=overrides):
                status, body = self.post_decision(
                    self.decision_payload(**overrides)
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_decision_bad_from_to_values_are_422(self) -> None:
        for overrides in (
            {"from": -1, "to": 2},
            {"from": 0, "to": -2},
            {"from": 1.5, "to": 2},
            {"from": 0, "to": 2.5},
            {"from": True, "to": 2},
            {"from": 0, "to": False},
            {"from": "0", "to": 2},
            {"from": 10, "to": 5},
            {"from": None, "to": 2},
        ):
            with self.subTest(overrides=overrides):
                status, body = self.post_decision(
                    self.decision_payload(**overrides)
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_decision_repeated_requests_return_same_result(self) -> None:
        self.seed_aggregate_events()
        payload = self.decision_payload(threshold=2)
        results = [self.post_decision(payload) for _ in range(5)]
        self.assertTrue(all(status == 200 for status, _ in results))
        self.assertEqual(len({json.dumps(body, sort_keys=True) for _, body in results}), 1)

    # --- POST /decisions/allocate ---------------------------------------------

    def allocation_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "demands": [
                {"demandId": "d-low", "units": 3, "priority": 1},
                {"demandId": "d-b", "units": 4, "priority": 5},
                {"demandId": "d-a", "units": 2, "priority": 5},
                {"demandId": "d-big", "units": 10, "priority": 9},
            ],
            "resources": [
                {"resourceId": "r-b", "capacity": 6},
                {"resourceId": "r-a", "capacity": 5},
            ],
        }
        payload.update(overrides)
        return payload

    def test_allocation_plans_by_priority_then_id_and_deducts_capacity(self) -> None:
        # Processing order: d-big(9), d-a(5), d-b(5), d-low(1).
        # d-big fits nowhere -> unassigned. d-a takes smallest fitting r-a
        # (5 -> 3). d-b needs 4, r-a has 3, takes r-b (6 -> 2). d-low takes
        # remaining r-a (3 -> 0).
        status, body = self.post_allocation(self.allocation_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "assignments": [
                    {"demandId": "d-a", "resourceId": "r-a", "units": 2},
                    {"demandId": "d-b", "resourceId": "r-b", "units": 4},
                    {"demandId": "d-low", "resourceId": "r-a", "units": 3},
                ],
                "unassigned": ["d-big"],
                "totalUnits": 9,
            },
        )

    def test_allocation_response_has_exactly_fixed_fields(self) -> None:
        status, body = self.post_allocation(self.allocation_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            set(body), {"organizationId", "assignments", "unassigned", "totalUnits"}
        )
        for assignment in body["assignments"]:
            self.assertEqual(set(assignment), {"demandId", "resourceId", "units"})

    def test_allocation_empty_arrays_succeed(self) -> None:
        cases = [
            (
                {"organizationId": "o", "demands": [], "resources": []},
                [],
            ),
            (
                {"organizationId": "o", "demands": [], "resources": [
                    {"resourceId": "r", "capacity": 5}
                ]},
                [],
            ),
            (
                {"organizationId": "o", "demands": [
                    {"demandId": "d", "units": 1, "priority": 0}
                ], "resources": []},
                ["d"],
            ),
        ]
        for payload, expected_unassigned in cases:
            with self.subTest(payload=payload):
                status, body = self.post_allocation(payload)
                self.assertEqual(status, 200)
                self.assertEqual(body["organizationId"], "o")
                self.assertEqual(body["assignments"], [])
                self.assertEqual(body["unassigned"], expected_unassigned)
                self.assertEqual(body["totalUnits"], 0)

    def test_allocation_demand_never_split_or_oversold(self) -> None:
        # Two resources of 3 each cannot take a demand of 5 even though total
        # capacity is 6; the demand stays unassigned.
        payload = {
            "organizationId": "o",
            "demands": [
                {"demandId": "d1", "units": 5, "priority": 0},
                {"demandId": "d2", "units": 3, "priority": 0},
                {"demandId": "d3", "units": 3, "priority": 0},
                {"demandId": "d4", "units": 1, "priority": 0},
            ],
            "resources": [
                {"resourceId": "r1", "capacity": 3},
                {"resourceId": "r2", "capacity": 3},
            ],
        }
        status, body = self.post_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["assignments"],
            [
                {"demandId": "d2", "resourceId": "r1", "units": 3},
                {"demandId": "d3", "resourceId": "r2", "units": 3},
            ],
        )
        self.assertEqual(body["unassigned"], ["d1", "d4"])
        self.assertEqual(body["totalUnits"], 6)

    def test_allocation_unassigned_sorted_by_demand_id(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [
                {"demandId": "zeta", "units": 9, "priority": 10},
                {"demandId": "alpha", "units": 9, "priority": 10},
                {"demandId": "mid", "units": 9, "priority": 5},
            ],
            "resources": [{"resourceId": "r", "capacity": 1}],
        }
        status, body = self.post_allocation(payload)
        self.assertEqual(status, 200)
        # assignments follow processing order; unassigned follows demandId order
        self.assertEqual(body["assignments"], [])
        self.assertEqual(body["unassigned"], ["alpha", "mid", "zeta"])

    def test_allocation_same_priority_orders_by_unicode_demand_id(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [
                {"demandId": "b", "units": 2, "priority": 0},
                {"demandId": "ä", "units": 2, "priority": 0},
                {"demandId": "a", "units": 2, "priority": 0},
            ],
            "resources": [{"resourceId": "r", "capacity": 6}],
        }
        status, body = self.post_allocation(payload)
        self.assertEqual(status, 200)
        # Python's default string order is Unicode code-point order:
        # a < b < ä (U+00E4).
        self.assertEqual(
            [assignment["demandId"] for assignment in body["assignments"]],
            ["a", "b", "ä"],
        )

    def test_allocation_zero_priority_accepted(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [{"demandId": "d", "units": 1, "priority": 0}],
            "resources": [{"resourceId": "r", "capacity": 1}],
        }
        status, body = self.post_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["assignments"],
            [{"demandId": "d", "resourceId": "r", "units": 1}],
        )
        self.assertEqual(body["totalUnits"], 1)

    def test_allocation_deterministic_independent_of_input_order(self) -> None:
        first_payload = self.allocation_payload()
        second_payload = self.allocation_payload()
        second_payload["demands"] = list(reversed(second_payload["demands"]))
        second_payload["resources"] = list(reversed(second_payload["resources"]))
        first = self.post_allocation(first_payload)
        second = self.post_allocation(second_payload)
        self.assertEqual(first, second)

    def test_allocation_repeated_requests_return_identical_json(self) -> None:
        payload = self.allocation_payload()
        results = [self.post_allocation(payload) for _ in range(5)]
        self.assertTrue(all(status == 200 for status, _ in results))
        encoded = {json.dumps(body, sort_keys=True) for _, body in results}
        self.assertEqual(len(encoded), 1)
        # Each request plans from its own body: prior plans are not inherited.
        self.assertEqual(results[0][1], results[-1][1])

    def test_allocation_is_read_only(self) -> None:
        self.seed_aggregate_events()
        self.post_allocation(self.allocation_payload())
        self.post_allocation(self.allocation_payload())
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 6)

    def test_allocation_content_type_with_charset_is_accepted(self) -> None:
        status, _ = self.post_allocation(
            self.allocation_payload(), content_type="application/json; charset=utf-8"
        )
        self.assertEqual(status, 200)

    def test_allocation_missing_content_type_is_415(self) -> None:
        status, body = self.post_allocation(self.allocation_payload(), content_type=None)
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_allocation_unsupported_content_type_is_415(self) -> None:
        status, body = self.post_allocation(
            self.allocation_payload(), content_type="text/plain"
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_allocation_malformed_json_is_400(self) -> None:
        status, body = self.post_allocation(b'{"organizationId": ', raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_allocation_non_object_body_is_422(self) -> None:
        for bad_body in ([], "text", 42, None, True):
            with self.subTest(bad_body=bad_body):
                status, body = self.post_allocation(bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_allocation_missing_required_field_is_422(self) -> None:
        for field in ("organizationId", "demands", "resources"):
            payload = self.allocation_payload()
            del payload[field]
            with self.subTest(field=field):
                status, body = self.post_allocation(payload)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_allocation_extra_top_level_field_is_422(self) -> None:
        status, body = self.post_allocation(self.allocation_payload(extra="nope"))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_allocation_blank_or_non_string_organization_id_is_422(self) -> None:
        for bad_value in ("", "   ", 123, None, ["x"], True):
            with self.subTest(bad_value=bad_value):
                status, body = self.post_allocation(
                    self.allocation_payload(organizationId=bad_value)
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_allocation_arrays_must_be_arrays(self) -> None:
        for field in ("demands", "resources"):
            for bad_value in ({}, "x", 1, None, True):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_allocation(
                        self.allocation_payload(**{field: bad_value})
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_allocation_element_wrong_type_is_422(self) -> None:
        for field, good in (
            ("demands", {"demandId": "d", "units": 1, "priority": 0}),
            ("resources", {"resourceId": "r", "capacity": 1}),
        ):
            for bad_element in ("x", 1, None, True, ["x"]):
                payload = self.allocation_payload(**{field: [bad_element, good]})
                with self.subTest(field=field, bad_element=bad_element):
                    status, body = self.post_allocation(payload)
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_allocation_element_missing_or_extra_field_is_422(self) -> None:
        demand = {"demandId": "d", "units": 1, "priority": 0}
        resource = {"resourceId": "r", "capacity": 1}
        for bad_demand in (
            {"units": 1, "priority": 0},
            {"demandId": "d", "priority": 0},
            {"demandId": "d", "units": 1},
            {**demand, "extra": 1},
        ):
            with self.subTest(bad_demand=bad_demand):
                status, body = self.post_allocation(
                    self.allocation_payload(demands=[bad_demand])
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")
        for bad_resource in (
            {"capacity": 1},
            {"resourceId": "r"},
            {**resource, "extra": 1},
        ):
            with self.subTest(bad_resource=bad_resource):
                status, body = self.post_allocation(
                    self.allocation_payload(resources=[bad_resource])
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_allocation_blank_or_non_string_element_ids_are_422(self) -> None:
        for bad_value in ("", "   ", 123, None, ["x"], True):
            demand = {"demandId": bad_value, "units": 1, "priority": 0}
            with self.subTest(kind="demand", bad_value=bad_value):
                status, body = self.post_allocation(
                    self.allocation_payload(demands=[demand])
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")
            resource = {"resourceId": bad_value, "capacity": 1}
            with self.subTest(kind="resource", bad_value=bad_value):
                status, body = self.post_allocation(
                    self.allocation_payload(resources=[resource])
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_allocation_bad_units_is_422(self) -> None:
        for bad_value in (0, -1, 1.5, "2", True, None, [2], 1.0):
            demand = {"demandId": "d", "units": bad_value, "priority": 0}
            with self.subTest(bad_value=bad_value):
                status, body = self.post_allocation(
                    self.allocation_payload(demands=[demand])
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_allocation_bad_priority_is_422(self) -> None:
        for bad_value in (-1, 1.5, "0", True, None, [0], 0.0):
            demand = {"demandId": "d", "units": 1, "priority": bad_value}
            with self.subTest(bad_value=bad_value):
                status, body = self.post_allocation(
                    self.allocation_payload(demands=[demand])
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_allocation_bad_capacity_is_422(self) -> None:
        for bad_value in (0, -1, 1.5, "1", True, None, [1], 1.0):
            resource = {"resourceId": "r", "capacity": bad_value}
            with self.subTest(bad_value=bad_value):
                status, body = self.post_allocation(
                    self.allocation_payload(resources=[resource])
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_allocation_duplicate_identifiers_are_422(self) -> None:
        duplicate_demand = self.allocation_payload()
        duplicate_demand["demands"].append(
            {"demandId": "d-a", "units": 1, "priority": 0}
        )
        status, body = self.post_allocation(duplicate_demand)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

        duplicate_resource = self.allocation_payload()
        duplicate_resource["resources"].append(
            {"resourceId": "r-a", "capacity": 1}
        )
        status, body = self.post_allocation(duplicate_resource)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_allocation_concurrent_requests_do_not_pollute_each_other(self) -> None:
        payload = self.allocation_payload()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.post_allocation(payload), range(16)))
        self.assertTrue(all(status == 200 for status, _ in results))
        bodies = {json.dumps(body, sort_keys=True) for _, body in results}
        self.assertEqual(len(bodies), 1)

    # --- concurrency and isolation ---------------------------------------------

    def test_decision_reads_are_consistent_under_concurrent_writes(self) -> None:
        # Seed some initial events so reads are taken against a populated
        # ledger while many writes land concurrently.
        for index in range(20):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=f"seed-{index}", occurredAt=index * 3)
                )[0],
                201,
            )

        def write_event(index: int) -> None:
            self.post_event(
                make_event(eventId=f"live-{index}", occurredAt=index * 7)
            )

        payload = self.decision_payload(windowSize=10, threshold=50)

        def read_decision(_: int) -> tuple[int, Any]:
            return self.post_decision(payload)

        with ThreadPoolExecutor(max_workers=8) as pool:
            writes = [pool.submit(write_event, i) for i in range(24)]
            reads = list(pool.map(read_decision, range(16)))
            for future in writes:
                future.result()

        for status, body in reads:
            self.assertEqual(status, 200)
            # Every response must be internally coherent: peakCount is the
            # declared maximum and observe/escalate agrees with the threshold.
            self.assertGreaterEqual(body["peakCount"], 0)
            if body["peakCount"] == 0:
                self.assertIsNone(body["peakStart"])
            else:
                self.assertIsNotNone(body["peakStart"])
            expected_action = (
                "escalate" if body["peakCount"] >= payload["threshold"] else "observe"
            )
            self.assertEqual(body["action"], expected_action)

        # The ledger still contains exactly the seeded plus live events;
        # decision reads never wrote anything.
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 44)

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


def make_reservation(**overrides: Any) -> dict[str, Any]:
    reservation: dict[str, Any] = {
        "organizationId": "org-1",
        "reservationId": "rsv-1",
        "resourceId": "res-1",
        "quantity": 2,
        "capacity": 5,
    }
    reservation.update(overrides)
    return reservation


class ReservationTest(ServerTest):
    def post_reservation(
        self, payload: Any, *, raw: bool = False, content_type: str | None = "application/json"
    ) -> tuple[int, Any]:
        body = payload if raw else json.dumps(payload).encode()
        return self.request(
            "/reservations", method="POST", body=body, content_type=content_type
        )

    # --- POST /reservations ----------------------------------------------------

    def test_create_reservation_returns_201_with_inventory_view(self) -> None:
        status, body = self.post_reservation(make_reservation())
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "reservationId": "rsv-1",
                "resourceId": "res-1",
                "quantity": 2,
                "capacity": 5,
                "occupied": 2,
                "remaining": 3,
            },
        )

    def test_reservation_response_is_compact_and_newline_terminated(self) -> None:
        request = Request(
            f"{self.base_url}/reservations",
            data=json.dumps(make_reservation()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            raw = response.read()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        self.assertEqual(json.loads(raw.decode())["remaining"], 3)

    def test_occupied_and_remaining_accumulate_across_reservations(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)
        status, body = self.post_reservation(
            make_reservation(reservationId="rsv-2", quantity=2)
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["occupied"], 4)
        self.assertEqual(body["remaining"], 1)

    def test_identical_replay_returns_200_without_double_counting(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)
        status, body = self.post_reservation(make_reservation())
        self.assertEqual(status, 200)
        self.assertEqual(body["occupied"], 2)
        self.assertEqual(body["remaining"], 3)

    def test_replay_with_different_fields_is_reservation_conflict(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)
        for changed in (
            make_reservation(quantity=3),
            make_reservation(resourceId="res-2"),
            make_reservation(organizationId="org-2"),
            make_reservation(capacity=5, quantity=1),
        ):
            status, body = self.post_reservation(changed)
            self.assertEqual(status, 409)
            self.assertEqual(body["error"], "reservation_conflict")
        # The stored reservation is untouched.
        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(len(body["reservations"]), 1)
        self.assertEqual(body["reservations"][0]["occupied"], 2)

    def test_mismatched_capacity_for_known_resource_is_capacity_conflict(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)
        status, body = self.post_reservation(
            make_reservation(reservationId="rsv-2", capacity=10)
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_conflict")
        # Historical capacity is not rewritten.
        status, body = self.post_reservation(make_reservation(reservationId="rsv-3"))
        self.assertEqual(status, 201)
        self.assertEqual(body["capacity"], 5)

    def test_over_capacity_is_capacity_exceeded_and_changes_nothing(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)
        status, body = self.post_reservation(
            make_reservation(reservationId="rsv-2", quantity=4)
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_exceeded")
        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(len(body["reservations"]), 1)
        self.assertEqual(body["reservations"][0]["remaining"], 3)

    def test_exact_capacity_is_allowed(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)
        status, body = self.post_reservation(
            make_reservation(reservationId="rsv-2", quantity=3)
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["occupied"], 5)
        self.assertEqual(body["remaining"], 0)

    def test_capacity_is_tracked_per_resource(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)
        status, body = self.post_reservation(
            make_reservation(
                reservationId="rsv-2", resourceId="res-2", quantity=1, capacity=1
            )
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["capacity"], 1)
        self.assertEqual(body["occupied"], 1)

    def test_reservation_requires_json_content_type(self) -> None:
        for content_type in (None, "text/plain"):
            status, body = self.post_reservation(
                make_reservation(), content_type=content_type
            )
            self.assertEqual(status, 415)
            self.assertEqual(body["error"], "unsupported_media_type")

    def test_reservation_invalid_json_is_400_and_creates_nothing(self) -> None:
        status, body = self.post_reservation(b"{not json", raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")
        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(body["reservations"], [])

    def test_reservation_validation_errors_are_422(self) -> None:
        bad_payloads = [
            [1, 2],
            "text",
            make_reservation(quantity=0),
            make_reservation(quantity=-1),
            make_reservation(quantity=1.5),
            make_reservation(quantity=True),
            make_reservation(quantity="2"),
            make_reservation(capacity=0),
            make_reservation(capacity=False),
            make_reservation(organizationId=""),
            make_reservation(organizationId="   "),
            make_reservation(reservationId=""),
            make_reservation(resourceId=""),
            make_reservation(resourceId=7),
            {k: v for k, v in make_reservation().items() if k != "quantity"},
            {**make_reservation(), "extra": 1},
        ]
        for payload in bad_payloads:
            status, body = self.post_reservation(payload)
            self.assertEqual(status, 422, payload)
            self.assertEqual(body["error"], "validation_error", payload)

    # --- GET /reservations -----------------------------------------------------

    def test_list_reservations_filters_and_sorts(self) -> None:
        self.assertEqual(
            self.post_reservation(
                make_reservation(reservationId="rsv-b", resourceId="res-2", capacity=9)
            )[0],
            201,
        )
        self.assertEqual(self.post_reservation(make_reservation(reservationId="rsv-a"))[0], 201)
        self.assertEqual(
            self.post_reservation(
                make_reservation(
                    reservationId="rsv-c",
                    resourceId="res-2",
                    capacity=9,
                    organizationId="org-2",
                )
            )[0],
            201,
        )

        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        keys = [
            (entry["resourceId"], entry["reservationId"])
            for entry in body["reservations"]
        ]
        self.assertEqual(keys, [("res-1", "rsv-a"), ("res-2", "rsv-b")])
        entry = body["reservations"][0]
        self.assertEqual(entry["quantity"], 2)
        self.assertEqual(entry["capacity"], 5)
        self.assertEqual(entry["occupied"], 2)
        self.assertEqual(entry["remaining"], 3)

    def test_list_reservations_unknown_organization_is_empty(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)
        status, body = self.request("/reservations?organizationId=org-unknown")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"organizationId": "org-unknown", "reservations": []})

    def test_list_reservations_query_validation(self) -> None:
        for path in (
            "/reservations",
            "/reservations?organizationId=",
            "/reservations?organizationId=%20",
            "/reservations?organizationId=org-1&organizationId=org-1",
        ):
            status, body = self.request(path)
            self.assertEqual(status, 422, path)
            self.assertEqual(body["error"], "validation_error", path)

    # --- concurrency and lifecycle ----------------------------------------------

    def test_concurrent_reservations_never_oversell(self) -> None:
        self.assertEqual(
            self.post_reservation(make_reservation(quantity=1, capacity=10))[0], 201
        )
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(
                pool.map(
                    lambda index: self.post_reservation(
                        make_reservation(
                            reservationId=f"rsv-{index}", quantity=1, capacity=10
                        )
                    ),
                    range(2, 30),
                )
            )
        statuses = [status for status, _ in results]
        self.assertEqual(statuses.count(201), 9)
        self.assertEqual(statuses.count(409), 19)
        for status, body in results:
            if status == 409:
                self.assertEqual(body["error"], "capacity_exceeded")

        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(len(body["reservations"]), 10)
        for entry in body["reservations"]:
            self.assertEqual(entry["occupied"], 10)
            self.assertEqual(entry["remaining"], 0)

    def test_concurrent_identical_reservations_count_once(self) -> None:
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(
                pool.map(lambda _: self.post_reservation(make_reservation()), range(24))
            )
        statuses = sorted(status for status, _ in results)
        self.assertEqual(statuses.count(201), 1)
        self.assertEqual(statuses.count(200), 23)
        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(len(body["reservations"]), 1)
        self.assertEqual(body["reservations"][0]["occupied"], 2)

    def test_new_server_instance_has_empty_inventory(self) -> None:
        self.assertEqual(self.post_reservation(make_reservation())[0], 201)

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(
                f"http://127.0.0.1:{fresh.server_port}/reservations?organizationId=org-1",
                timeout=2,
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    json.load(response),
                    {"organizationId": "org-1", "reservations": []},
                )
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
