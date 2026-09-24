from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
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
        "payload": {},
    }
    event.update(overrides)
    return event


class AlertTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self._event_seq = 0

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

    def post_event(self, event: Any) -> tuple[int, Any]:
        return self.request(
            "/events", method="POST", body=json.dumps(event).encode()
        )

    def alert_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "type": "incident.created",
            "windowSize": 60,
            "threshold": 2,
            "suppressionWindow": 120,
        }
        payload.update(overrides)
        return payload

    def post_alert(
        self,
        payload: Any,
        *,
        raw: bool = False,
        content_type: str | None = "application/json",
    ) -> tuple[int, Any]:
        body = payload if raw else json.dumps(payload).encode()
        return self.request(
            "/alerts/evaluate", method="POST", body=body, content_type=content_type
        )

    def seed_events(
        self,
        *occurred_at: int,
        organization: str = "org-1",
        event_type: str = "incident.created",
    ) -> None:
        for timestamp in occurred_at:
            self._event_seq += 1
            status, _ = self.post_event(
                make_event(
                    eventId=f"evt-{self._event_seq}",
                    organizationId=organization,
                    type=event_type,
                    occurredAt=timestamp,
                )
            )
            self.assertEqual(status, 201)

    # --- escalation / suppression ---------------------------------------------

    def test_first_threshold_reach_escalates_with_sequential_id(self) -> None:
        self.seed_events(0, 10)
        status, body = self.post_alert(self.alert_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "threshold": 2,
                "suppressionWindow": 120,
                "from": None,
                "to": None,
                "peakStart": 0,
                "peakCount": 2,
                "action": "escalate",
                "alertId": "alert-1",
                "suppressedCount": 0,
            },
        )

    def test_below_threshold_observes_without_alert(self) -> None:
        self.seed_events(0)
        status, body = self.post_alert(self.alert_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["action"], "observe")
        self.assertIsNone(body["alertId"])
        self.assertEqual(body["suppressedCount"], 0)
        self.assertEqual(body["peakCount"], 1)
        self.assertEqual(body["peakStart"], 0)

        status, listing = self.request("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(listing["alerts"], [])

    def test_repeat_within_suppression_window_increments_count(self) -> None:
        self.seed_events(0, 10, 70, 80)
        first = self.post_alert(self.alert_payload())
        self.assertEqual(first[1]["action"], "escalate")
        self.assertEqual(first[1]["alertId"], "alert-1")
        self.assertEqual(first[1]["peakStart"], 0)

        # Restrict the range so the peak moves to window 60: 60 - 0 < 120.
        ranged = self.alert_payload(**{"from": 60, "to": 120})
        second = self.post_alert(ranged)
        self.assertEqual(second[0], 200)
        self.assertEqual(second[1]["action"], "suppress")
        self.assertEqual(second[1]["alertId"], "alert-1")
        self.assertEqual(second[1]["suppressedCount"], 1)
        self.assertEqual(second[1]["peakStart"], 60)

        third = self.post_alert(ranged)
        self.assertEqual(third[1]["action"], "suppress")
        self.assertEqual(third[1]["alertId"], "alert-1")
        self.assertEqual(third[1]["suppressedCount"], 2)

        # Still exactly one stored alert.
        status, listing = self.request("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["alerts"]), 1)
        self.assertEqual(listing["alerts"][0]["suppressedCount"], 2)

    def test_peak_beyond_suppression_window_creates_new_alert(self) -> None:
        self.seed_events(0, 10, 200, 210)
        first = self.post_alert(self.alert_payload())
        self.assertEqual(first[1]["alertId"], "alert-1")
        self.assertEqual(first[1]["peakStart"], 0)

        # Restrict the range so the peak is window 180: 180 - 0 >= 120.
        second = self.post_alert(self.alert_payload(**{"from": 150, "to": 250}))
        self.assertEqual(second[1]["action"], "escalate")
        self.assertEqual(second[1]["alertId"], "alert-2")
        self.assertEqual(second[1]["suppressedCount"], 0)
        self.assertEqual(second[1]["peakStart"], 180)

    def test_exact_suppression_window_distance_escalates(self) -> None:
        self.seed_events(0, 5, 120, 125)
        first = self.post_alert(self.alert_payload())
        self.assertEqual(first[1]["action"], "escalate")
        # Peak window 120: distance exactly 120 -> new alert.
        second = self.post_alert(self.alert_payload(**{"from": 60, "to": 180}))
        self.assertEqual(second[1]["action"], "escalate")
        self.assertEqual(second[1]["alertId"], "alert-2")
        self.assertEqual(second[1]["peakStart"], 120)

    def test_earlier_peak_is_suppressed(self) -> None:
        self.seed_events(120, 125, 130)
        first = self.post_alert(self.alert_payload())
        self.assertEqual(first[1]["action"], "escalate")
        self.assertEqual(first[1]["peakStart"], 120)

        # A range-limited evaluation whose peak (window 0) is earlier than
        # the stored alert's peak start is suppressed, not a new alert.
        self.seed_events(10)
        payload = self.alert_payload(threshold=1, **{"from": 0, "to": 60})
        second = self.post_alert(payload)
        self.assertEqual(second[1]["action"], "suppress")
        self.assertEqual(second[1]["alertId"], "alert-1")
        self.assertEqual(second[1]["suppressedCount"], 1)

    def test_alert_state_is_isolated_per_organization_and_type(self) -> None:
        self.seed_events(0, 10)
        self.seed_events(0, 10, organization="org-2")
        self.seed_events(0, 10, event_type="incident.updated")

        first = self.post_alert(self.alert_payload())
        self.assertEqual(first[1]["alertId"], "alert-1")
        # Same peak, different organization: independent alert stream.
        second = self.post_alert(self.alert_payload(organizationId="org-2"))
        self.assertEqual(second[1]["action"], "escalate")
        self.assertEqual(second[1]["alertId"], "alert-2")
        # Same organization, different type: independent alert stream.
        third = self.post_alert(self.alert_payload(type="incident.updated"))
        self.assertEqual(third[1]["action"], "escalate")
        self.assertEqual(third[1]["alertId"], "alert-3")
        # Repeating the original evaluation suppresses against alert-1.
        fourth = self.post_alert(self.alert_payload())
        self.assertEqual(fourth[1]["action"], "suppress")
        self.assertEqual(fourth[1]["alertId"], "alert-1")

    def test_alert_evaluation_does_not_modify_ledger(self) -> None:
        self.seed_events(0, 10)
        self.post_alert(self.alert_payload())
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 2)

    def test_alert_response_is_compact_json_with_trailing_newline(self) -> None:
        self.seed_events(0, 10)
        request = Request(
            f"{self.base_url}/alerts/evaluate",
            data=json.dumps(self.alert_payload()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            raw = response.read()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        self.assertEqual(json.loads(raw)["alertId"], "alert-1")

    # --- GET /alerts -----------------------------------------------------------

    def test_list_alerts_sorts_by_peak_start_then_alert_id(self) -> None:
        # org-1 / incident.created: alerts at peak starts 0 and 180.
        self.seed_events(0, 10, 200, 210)
        self.post_alert(self.alert_payload())
        self.post_alert(self.alert_payload(**{"from": 150, "to": 250}))
        # Another type peaks at 60, between the two alert peak starts.
        self.seed_events(70, 80, event_type="incident.updated")
        self.post_alert(self.alert_payload(type="incident.updated"))

        status, body = self.request("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(
            [
                (alert["alertId"], alert["type"], alert["peakStart"])
                for alert in body["alerts"]
            ],
            [
                ("alert-1", "incident.created", 0),
                ("alert-3", "incident.updated", 60),
                ("alert-2", "incident.created", 180),
            ],
        )
        for alert in body["alerts"]:
            self.assertEqual(
                set(alert),
                {"alertId", "type", "peakStart", "threshold", "suppressedCount"},
            )
            self.assertEqual(alert["threshold"], 2)

    def test_list_alerts_unknown_organization_returns_empty(self) -> None:
        self.seed_events(0, 10)
        self.post_alert(self.alert_payload())
        status, body = self.request("/alerts?organizationId=other")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"organizationId": "other", "alerts": []})

    def test_list_alerts_requires_single_non_empty_organization_id(self) -> None:
        for query in (
            "/alerts",
            "/alerts?organizationId=",
            "/alerts?organizationId=%20%20",
            "/alerts?organizationId=a&organizationId=b",
        ):
            with self.subTest(query=query):
                status, body = self.request(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    # --- validation ------------------------------------------------------------

    def test_missing_content_type_is_415(self) -> None:
        status, body = self.post_alert(self.alert_payload(), content_type=None)
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_unsupported_content_type_is_415(self) -> None:
        status, body = self.post_alert(
            self.alert_payload(), content_type="text/plain"
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_malformed_json_is_400_without_alert(self) -> None:
        status, body = self.post_alert(b'{"organizationId": ', raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")
        _, listing = self.request("/alerts?organizationId=org-1")
        self.assertEqual(listing["alerts"], [])

    def test_non_object_body_is_422(self) -> None:
        for bad_body in ([], "text", 42, None, True):
            with self.subTest(bad_body=bad_body):
                status, body = self.post_alert(bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_missing_or_extra_field_is_422(self) -> None:
        for field in (
            "organizationId",
            "type",
            "windowSize",
            "threshold",
            "suppressionWindow",
        ):
            payload = self.alert_payload()
            del payload[field]
            with self.subTest(missing=field):
                status, body = self.post_alert(payload)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")
        status, body = self.post_alert(self.alert_payload(extra="nope"))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_blank_or_non_string_identifiers_are_422(self) -> None:
        for field in ("organizationId", "type"):
            for bad_value in ("", "   ", 123, None, ["x"], True):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_alert(
                        self.alert_payload(**{field: bad_value})
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_bad_positive_integer_fields_are_422(self) -> None:
        for field in ("windowSize", "threshold", "suppressionWindow"):
            for bad_value in (0, -1, 1.5, "60", True, None, [3], 1.0):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_alert(
                        self.alert_payload(**{field: bad_value})
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_from_to_must_be_paired_valid_and_ordered(self) -> None:
        for overrides in (
            {"from": 0},
            {"to": 0},
            {"from": -1, "to": 2},
            {"from": 0, "to": -2},
            {"from": 1.5, "to": 2},
            {"from": 0, "to": 2.5},
            {"from": True, "to": 2},
            {"from": "0", "to": 2},
            {"from": 10, "to": 5},
            {"from": None, "to": 2},
        ):
            with self.subTest(overrides=overrides):
                status, body = self.post_alert(self.alert_payload(**overrides))
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_failed_evaluation_writes_no_alert(self) -> None:
        self.seed_events(0, 10)
        status, _ = self.post_alert(self.alert_payload(threshold=0))
        self.assertEqual(status, 422)
        status, _ = self.post_alert(self.alert_payload(extra="x"))
        self.assertEqual(status, 422)
        _, listing = self.request("/alerts?organizationId=org-1")
        self.assertEqual(listing["alerts"], [])

    def test_range_is_echoed_and_filters_events(self) -> None:
        self.seed_events(0, 10, 500, 510)
        payload = self.alert_payload(**{"from": 400, "to": 600})
        status, body = self.post_alert(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 400)
        self.assertEqual(body["to"], 600)
        self.assertEqual(body["peakStart"], 480)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "escalate")

    # --- concurrency / lifecycle ------------------------------------------------

    def test_concurrent_identical_evaluations_escalate_once(self) -> None:
        self.seed_events(0, 10)
        payload = self.alert_payload()
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda _: self.post_alert(payload), range(24)))

        actions = [body["action"] for _, body in results]
        self.assertEqual(actions.count("escalate"), 1)
        self.assertEqual(actions.count("suppress"), 23)
        for _, body in results:
            self.assertEqual(body["alertId"], "alert-1")

        _, listing = self.request("/alerts?organizationId=org-1")
        self.assertEqual(len(listing["alerts"]), 1)
        self.assertEqual(listing["alerts"][0]["suppressedCount"], 23)

    def test_new_server_instance_has_no_alerts(self) -> None:
        self.seed_events(0, 10)
        self.post_alert(self.alert_payload())
        _, body = self.request("/alerts?organizationId=org-1")
        self.assertEqual(len(body["alerts"]), 1)

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(
                f"http://127.0.0.1:{fresh.server_port}/alerts?organizationId=org-1",
                timeout=2,
            ) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    json.load(response),
                    {"organizationId": "org-1", "alerts": []},
                )
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # --- main-only surface -------------------------------------------------------

    def test_alerts_not_exposed_under_branches(self) -> None:
        self.seed_events(0, 10)
        self.request(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": "snap-1"}).encode(),
        )
        self.request(
            "/branches",
            method="POST",
            body=json.dumps(
                {"branchId": "br-1", "snapshotId": "snap-1"}
            ).encode(),
        )
        status, _ = self.request("/branches/br-1/alerts?organizationId=org-1")
        self.assertEqual(status, 404)
        status, body = self.request(
            "/branches/br-1/alerts/evaluate",
            method="POST",
            body=json.dumps(self.alert_payload()).encode(),
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    def test_snapshots_do_not_capture_alerts(self) -> None:
        self.seed_events(0, 10)
        self.post_alert(self.alert_payload())
        status, body = self.request(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": "snap-1"}).encode(),
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            set(body), {"snapshotId", "events", "resources", "reservations"}
        )

    def test_unknown_branch_error_body_has_only_error_key(self) -> None:
        status, body = self.request("/branches/ghost/events?organizationId=org-1")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "branch_not_found"})


if __name__ == "__main__":
    unittest.main()
