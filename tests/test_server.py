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

    def aggregate(self, query: str) -> tuple[int, Any]:
        return self.request(f"/events/aggregate?{query}")

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

    def evaluate(
        self, body: Any, *, raw: bool = False, content_type: str | None = "application/json"
    ) -> tuple[int, Any]:
        data = body if raw else json.dumps(body).encode()
        return self.request(
            "/decisions/evaluate",
            method="POST",
            body=data,
            content_type=content_type,
        )

    def decision_request(self, **overrides: Any) -> dict[str, Any]:
        request = {
            "organizationId": "org-1",
            "type": "incident.created",
            "windowSize": 60,
            "threshold": 3,
        }
        request.update(overrides)
        return request

    def test_decision_escalates_when_peak_reaches_threshold(self) -> None:
        for event_id, occurred_at in (
            ("evt-1", 0),
            ("evt-2", 60),
            ("evt-3", 60),
            ("evt-4", 61),
        ):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))[0],
                201,
            )
        status, body = self.evaluate(self.decision_request(threshold=3))
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
                "peakCount": 3,
                "action": "escalate",
            },
        )
        self.assertEqual(
            set(body),
            {
                "organizationId",
                "type",
                "windowSize",
                "from",
                "to",
                "peakStart",
                "peakCount",
                "action",
            },
        )

    def test_decision_threshold_boundary_equality_escalates(self) -> None:
        for event_id, occurred_at in (("evt-a", 0), ("evt-b", 1)):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))[0],
                201,
            )
        status, body = self.evaluate(self.decision_request(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["peakStart"], 0)
        self.assertEqual(body["action"], "escalate")

        status, body = self.evaluate(self.decision_request(threshold=3))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["peakStart"], 0)
        self.assertEqual(body["action"], "observe")

    def test_decision_tied_peaks_choose_earliest_start(self) -> None:
        self.seed_aggregate_events()
        # Window [0, 60) holds t=0 and t=59; window [60, 120) holds two t=60.
        status, body = self.evaluate(self.decision_request(threshold=5))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["peakStart"], 0)
        self.assertEqual(body["action"], "observe")

    def test_decision_without_range_no_match_is_null_peak(self) -> None:
        self.seed_aggregate_events()
        status, body = self.evaluate(self.decision_request(type="no.such.type"))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")

    def test_decision_unknown_organization_is_zero_events(self) -> None:
        self.seed_aggregate_events()
        status, body = self.evaluate(self.decision_request(organizationId="org-other"))
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-other")
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")

    def test_decision_with_range_filters_and_keeps_empty_windows(self) -> None:
        self.seed_aggregate_events()
        request = self.decision_request(threshold=2, **{"from": 59, "to": 180})
        status, body = self.evaluate(request)
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 59)
        self.assertEqual(body["to"], 180)
        # [0,60): t=59 -> 1; [60,120): t=60,t=60 -> 2; [120,180): 0;
        # [180,240): t=180 -> 1.
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "escalate")

    def test_decision_range_over_only_empty_space_reports_null_peak(self) -> None:
        self.seed_aggregate_events()
        request = self.decision_request(**{"from": 120, "to": 179})
        status, body = self.evaluate(request)
        self.assertEqual(status, 200)
        # The intersecting window [120, 180) contains no matching events, so
        # the no-match rule applies even though a range was supplied.
        self.assertEqual(body["from"], 120)
        self.assertEqual(body["to"], 179)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["peakCount"], 0)
        self.assertEqual(body["action"], "observe")

    def test_decision_range_closed_interval_includes_both_endpoints(self) -> None:
        self.seed_aggregate_events()
        request = self.decision_request(threshold=2, **{"from": 60, "to": 60})
        status, body = self.evaluate(request)
        self.assertEqual(status, 200)
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "escalate")

    def test_decision_does_not_include_other_organizations_or_types(self) -> None:
        self.seed_aggregate_events()
        # org-2 has one event and incident.updated has one event; neither may
        # contribute to an org-1 / incident.created evaluation.
        status, body = self.evaluate(self.decision_request(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 2)

    def test_decision_does_not_modify_ledger(self) -> None:
        self.seed_aggregate_events()
        self.evaluate(self.decision_request())
        self.evaluate(self.decision_request(**{"from": 0, "to": 200, "threshold": 1}))
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 6)
        # Replays still reconcile as "exists": evaluation stored nothing.
        self.assertEqual(self.post_event(make_event(eventId="evt-1", occurredAt=0))[0], 200)

    def test_decision_is_reproducible_with_out_of_order_submission(self) -> None:
        for event_id, occurred_at in (("evt-c", 60), ("evt-a", 0), ("evt-d", 0), ("evt-b", 60)):
            self.assertEqual(
                self.post_event(make_event(eventId=event_id, occurredAt=occurred_at))[0],
                201,
            )
        request = self.decision_request(threshold=3)
        first = self.evaluate(request)
        second = self.evaluate(request)
        self.assertEqual(first, second)
        self.assertEqual(first[1]["peakStart"], 0)
        self.assertEqual(first[1]["peakCount"], 2)
        self.assertEqual(first[1]["action"], "observe")

    def test_decision_get_is_unknown_path(self) -> None:
        status, body = self.request("/decisions/evaluate")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    # --- decision validation: content type / JSON -----------------------------

    def test_decision_missing_content_type_is_415(self) -> None:
        status, body = self.evaluate(self.decision_request(), content_type=None)
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_decision_unsupported_content_type_is_415(self) -> None:
        status, body = self.evaluate(
            self.decision_request(), content_type="text/plain"
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_decision_malformed_json_is_400(self) -> None:
        status, body = self.evaluate(b'{"organizationId": ', raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    # --- decision validation: fields ------------------------------------------

    def test_decision_array_body_is_422(self) -> None:
        status, body = self.evaluate([self.decision_request()])
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_decision_missing_required_field_is_422(self) -> None:
        for field in ("organizationId", "type", "windowSize", "threshold"):
            with self.subTest(field=field):
                request = self.decision_request()
                del request[field]
                status, body = self.evaluate(request)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_decision_extra_field_is_422(self) -> None:
        request = self.decision_request(extra="nope")
        status, body = self.evaluate(request)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_decision_blank_or_non_string_identifiers_are_422(self) -> None:
        for field in ("organizationId", "type"):
            for bad_value in ("", "   ", 123, None, ["x"], True):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.evaluate(self.decision_request(**{field: bad_value}))
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_decision_positive_integer_fields_reject_bad_values(self) -> None:
        for field in ("windowSize", "threshold"):
            for bad_value in (0, -1, 1.5, "10", True, None, [10]):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.evaluate(self.decision_request(**{field: bad_value}))
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_decision_from_to_must_be_paired(self) -> None:
        for overrides in ({"from": 0}, {"to": 0}):
            with self.subTest(overrides=overrides):
                status, body = self.evaluate(self.decision_request(**overrides))
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_decision_from_to_reject_bad_values_and_order(self) -> None:
        for overrides in (
            {"from": -1, "to": 2},
            {"from": 0, "to": -1},
            {"from": 1.5, "to": 2},
            {"from": 0, "to": 1.5},
            {"from": True, "to": 2},
            {"from": 0, "to": False},
            {"from": "0", "to": 2},
            {"from": 5, "to": 2},
        ):
            with self.subTest(overrides=overrides):
                status, body = self.evaluate(self.decision_request(**overrides))
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_decision_accepts_equal_from_and_to(self) -> None:
        status, body = self.evaluate(
            self.decision_request(**{"from": 30, "to": 30})
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 30)
        self.assertEqual(body["to"], 30)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["peakCount"], 0)
        self.assertEqual(body["action"], "observe")

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
