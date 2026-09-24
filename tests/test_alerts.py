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
        content_type: str | None = "application/json",
    ) -> tuple[int, bytes, Any]:
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
                raw = response.read()
                return response.status, raw, json.loads(raw)
        except HTTPError as error:
            raw = error.read()
            try:
                return error.code, raw, json.loads(raw)
            finally:
                error.close()

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, Any]:
        status, _, parsed = self.request_raw(
            path, method=method, body=body, content_type=content_type
        )
        return status, parsed

    def post_json(self, path: str, payload: Any) -> tuple[int, Any]:
        return self.request(path, method="POST", body=json.dumps(payload).encode())

    def post_event(self, event: dict[str, Any]) -> tuple[int, Any]:
        return self.post_json("/events", event)

    def alert_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "type": "incident.created",
            "windowSize": 60,
            "threshold": 2,
            "suppressionWindow": 100,
        }
        payload.update(overrides)
        return payload

    def evaluate_alert(self, **overrides: Any) -> tuple[int, Any]:
        return self.post_json("/alerts/evaluate", self.alert_payload(**overrides))

    def list_alerts(self, query: str = "organizationId=org-1") -> tuple[int, Any]:
        suffix = f"?{query}" if query else ""
        return self.request(f"/alerts{suffix}")

    def seed_events(self, *timestamps: int) -> None:
        for index, timestamp in enumerate(timestamps):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=f"evt-{index}", occurredAt=timestamp)
                )[0],
                201,
            )

    # ------------------------------------------------------------- escalation

    def test_below_threshold_is_observe_with_null_identity(self) -> None:
        self.seed_events(10)
        status, body = self.evaluate_alert(threshold=2)
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "threshold": 2,
                "suppressionWindow": 100,
                "from": None,
                "to": None,
                "peakStart": 0,
                "peakCount": 1,
                "action": "observe",
                "alertId": None,
                "suppressedCount": None,
            },
        )

    def test_threshold_reached_creates_alert_one(self) -> None:
        self.seed_events(10, 20)
        status, body = self.evaluate_alert(threshold=2)
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "threshold": 2,
                "suppressionWindow": 100,
                "from": None,
                "to": None,
                "peakStart": 0,
                "peakCount": 2,
                "action": "escalate",
                "alertId": "alert-1",
                "suppressedCount": 0,
            },
        )

    def test_zero_events_unknown_organization_is_observe(self) -> None:
        self.seed_events(10, 20)
        status, body = self.evaluate_alert(organizationId="org-x")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-x")
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")
        self.assertIsNone(body["alertId"])
        self.assertIsNone(body["suppressedCount"])

    def test_zero_events_unknown_type_is_observe(self) -> None:
        self.seed_events(10, 20)
        status, body = self.evaluate_alert(type="no.such.type")
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 0)
        self.assertEqual(body["action"], "observe")
        self.assertIsNone(body["alertId"])

    def test_observe_writes_no_alert(self) -> None:
        self.seed_events(10)
        self.assertEqual(self.evaluate_alert(threshold=5)[1]["action"], "observe")
        status, body = self.list_alerts()
        self.assertEqual(status, 200)
        self.assertEqual(body["alerts"], [])

    # ------------------------------------------------------------- suppression

    def test_repeat_peak_inside_window_is_suppressed_and_accumulates(self) -> None:
        self.seed_events(10, 20)
        status, first = self.evaluate_alert(suppressionWindow=100)
        self.assertEqual((status, first["action"]), (200, "escalate"))

        status, second = self.evaluate_alert(suppressionWindow=100)
        self.assertEqual(status, 200)
        self.assertEqual(
            second,
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "threshold": 2,
                "suppressionWindow": 100,
                "from": None,
                "to": None,
                "peakStart": 0,
                "peakCount": 2,
                "action": "suppress",
                "alertId": "alert-1",
                "suppressedCount": 1,
            },
        )

        status, third = self.evaluate_alert(suppressionWindow=100)
        self.assertEqual(third["action"], "suppress")
        self.assertEqual(third["alertId"], "alert-1")
        self.assertEqual(third["suppressedCount"], 2)

        status, body = self.list_alerts()
        self.assertEqual(len(body["alerts"]), 1)
        self.assertEqual(body["alerts"][0]["suppressedCount"], 2)

    def test_peak_at_exact_suppression_distance_creates_new_alert(self) -> None:
        # First peak starts at window 0.
        self.seed_events(10, 20)
        status, first = self.evaluate_alert(suppressionWindow=60)
        self.assertEqual(first["action"], "escalate")
        self.assertEqual(first["alertId"], "alert-1")

        # Two more events land in window 60; the peak start distance to the
        # prior alert is exactly 60, which is not "less than" the window.
        for event_id, timestamp in (("evt-a", 60), ("evt-b", 61)):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=timestamp)
                )[0],
                201,
            )
        # Window 60 ties with window 0 at count 2; the tie resolves to the
        # earliest start, so push window 60 strictly ahead.
        self.assertEqual(
            self.post_event(make_event(eventId="evt-c", occurredAt=62))[0],
            201,
        )
        status, second = self.evaluate_alert(suppressionWindow=60)
        self.assertEqual(second["action"], "escalate")
        self.assertEqual(second["alertId"], "alert-2")
        self.assertEqual(second["peakStart"], 60)
        self.assertEqual(second["suppressedCount"], 0)

    def test_peak_one_inside_suppression_distance_is_suppressed(self) -> None:
        self.seed_events(0, 1)
        self.assertEqual(self.evaluate_alert(suppressionWindow=100)[1]["action"], "escalate")
        # Three events in window [60, 120) make its count (3) beat window 0.
        for event_id, timestamp in (("evt-a", 100), ("evt-b", 101), ("evt-c", 102)):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=timestamp)
                )[0],
                201,
            )
        status, body = self.evaluate_alert(suppressionWindow=100)
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["action"], "suppress")
        self.assertEqual(body["alertId"], "alert-1")
        self.assertEqual(body["suppressedCount"], 1)

    def test_alert_ids_increment_globally(self) -> None:
        # org-a raises alert-1, org-b raises alert-2, org-a then opens
        # alert-3 after its suppression window elapses.
        for event_id, timestamp in (("a1", 0), ("a2", 1)):
            self.post_event(
                make_event(
                    eventId=event_id,
                    organizationId="org-a",
                    occurredAt=timestamp,
                )
            )
        status, body = self.evaluate_alert(
            organizationId="org-a", suppressionWindow=10
        )
        self.assertEqual(body["alertId"], "alert-1")

        for event_id, timestamp in (("b1", 0), ("b2", 1)):
            self.post_event(
                make_event(
                    eventId=event_id,
                    organizationId="org-b",
                    occurredAt=timestamp,
                )
            )
        status, body = self.evaluate_alert(
            organizationId="org-b", suppressionWindow=10
        )
        self.assertEqual(body["alertId"], "alert-2")

        for event_id, timestamp in (("a3", 100), ("a4", 101), ("a5", 102)):
            self.post_event(
                make_event(
                    eventId=event_id,
                    organizationId="org-a",
                    occurredAt=timestamp,
                )
            )
        status, body = self.evaluate_alert(
            organizationId="org-a", suppressionWindow=10
        )
        self.assertEqual(body["action"], "escalate")
        self.assertEqual(body["alertId"], "alert-3")
        self.assertEqual(body["peakStart"], 60)

    def test_suppression_is_scoped_per_organization_and_type(self) -> None:
        self.seed_events(0, 1)
        self.assertEqual(self.evaluate_alert()[1]["alertId"], "alert-1")

        # Same window start, different organization: independent alert.
        for event_id, timestamp in (("o2-1", 0), ("o2-2", 1)):
            self.post_event(
                make_event(
                    eventId=event_id,
                    organizationId="org-2",
                    occurredAt=timestamp,
                )
            )
        status, body = self.evaluate_alert(organizationId="org-2")
        self.assertEqual(body["action"], "escalate")

        # Same organization, different type: also independent.
        for event_id, timestamp in (("t2-1", 0), ("t2-2", 1)):
            self.post_event(
                make_event(
                    eventId=event_id,
                    type="incident.updated",
                    occurredAt=timestamp,
                )
            )
        status, body = self.evaluate_alert(type="incident.updated")
        self.assertEqual(body["action"], "escalate")

    def test_range_filter_uses_decision_peak_semantics(self) -> None:
        # Window 0 holds two events, window 60 holds two; the range [60, 120]
        # contains only the latter pair.
        self.seed_events(0, 1, 60, 61)
        self.assertEqual(
            self.evaluate_alert(suppressionWindow=1000)[1]["alertId"], "alert-1"
        )
        status, body = self.evaluate_alert(suppressionWindow=1000, **{"from": 60, "to": 120})
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["peakCount"], 2)
        # Distance from prior peak start 0 is 60 < 1000 -> suppressed.
        self.assertEqual(body["action"], "suppress")
        self.assertEqual(body["alertId"], "alert-1")

    # ------------------------------------------------------- response encoding

    def test_response_is_compact_stable_order_and_newline_terminated(self) -> None:
        self.seed_events(10, 20)
        status, raw, body = self.request_raw(
            "/alerts/evaluate",
            method="POST",
            body=json.dumps(self.alert_payload()).encode(),
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Keys are serialized in sorted order.
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )

    def test_strings_are_echoed_verbatim(self) -> None:
        payload = self.alert_payload(organizationId=" Org-A ", type=" Type.X ")
        status, body = self.post_json("/alerts/evaluate", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], " Org-A ")
        self.assertEqual(body["type"], " Type.X ")

    # ------------------------------------------------------------- validation

    def test_missing_content_type_is_415(self) -> None:
        status, body = self.request(
            "/alerts/evaluate",
            method="POST",
            body=json.dumps(self.alert_payload()).encode(),
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_unsupported_content_type_is_415(self) -> None:
        status, body = self.request(
            "/alerts/evaluate",
            method="POST",
            body=json.dumps(self.alert_payload()).encode(),
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_malformed_json_is_400_without_alert(self) -> None:
        status, body = self.request(
            "/alerts/evaluate", method="POST", body=b'{"organizationId": '
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")
        self.assertEqual(self.list_alerts()[1]["alerts"], [])

    def test_non_object_body_is_422(self) -> None:
        for bad_body in ([], "text", 42, None, True):
            with self.subTest(bad_body=bad_body):
                status, body = self.post_json("/alerts/evaluate", bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_missing_required_field_is_422(self) -> None:
        for field in (
            "organizationId",
            "type",
            "windowSize",
            "threshold",
            "suppressionWindow",
        ):
            payload = self.alert_payload()
            del payload[field]
            with self.subTest(field=field):
                status, body = self.post_json("/alerts/evaluate", payload)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_extra_field_is_422(self) -> None:
        status, body = self.post_json(
            "/alerts/evaluate", self.alert_payload(extra="nope")
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_blank_or_non_string_identifiers_are_422(self) -> None:
        for field in ("organizationId", "type"):
            for bad_value in ("", "   ", 123, None, ["x"], True):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_json(
                        "/alerts/evaluate",
                        self.alert_payload(**{field: bad_value}),
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_bad_positive_integer_fields_are_422(self) -> None:
        for field in ("windowSize", "threshold", "suppressionWindow"):
            for bad_value in (0, -1, 1.5, 1.0, "60", True, None, [3]):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_json(
                        "/alerts/evaluate",
                        self.alert_payload(**{field: bad_value}),
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
            {"from": True, "to": 2},
            {"from": "0", "to": 2},
            {"from": 10, "to": 5},
            {"from": None, "to": 2},
        ):
            with self.subTest(overrides=overrides):
                status, body = self.post_json(
                    "/alerts/evaluate", self.alert_payload(**overrides)
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_failed_requests_never_write_alerts(self) -> None:
        self.seed_events(10, 20)
        # Validation failure despite a threshold that would escalate.
        bad = self.alert_payload(suppressionWindow=0)
        self.assertEqual(self.post_json("/alerts/evaluate", bad)[0], 422)
        self.assertEqual(
            self.request(
                "/alerts/evaluate",
                method="POST",
                body=b"{",
            )[0],
            400,
        )
        self.assertEqual(self.list_alerts()[1]["alerts"], [])

    # --------------------------------------------------------- GET /alerts

    def test_list_alerts_filters_by_organization_and_sorts(self) -> None:
        for event_id, org, event_type, timestamp in (
            ("a1", "org-a", "t-b", 200),
            ("a2", "org-a", "t-b", 201),
            ("a3", "org-a", "t-a", 0),
            ("a4", "org-a", "t-a", 1),
            ("a5", "org-a", "t-c", 60),
            ("a6", "org-a", "t-c", 61),
            ("b1", "org-b", "t-a", 0),
            ("b2", "org-b", "t-a", 1),
        ):
            self.post_event(
                make_event(
                    eventId=event_id,
                    organizationId=org,
                    type=event_type,
                    occurredAt=timestamp,
                )
            )
        for event_type in ("t-a", "t-b", "t-c"):
            status, body = self.evaluate_alert(
                organizationId="org-a", type=event_type, suppressionWindow=10000
            )
            self.assertEqual(status, 200)
        self.evaluate_alert(organizationId="org-b", type="t-a")

        status, body = self.list_alerts("organizationId=org-a")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-a")
        self.assertEqual(
            [(a["type"], a["peakStart"], a["alertId"]) for a in body["alerts"]],
            [("t-a", 0, "alert-1"), ("t-c", 60, "alert-3"), ("t-b", 180, "alert-2")],
        )
        for alert in body["alerts"]:
            self.assertEqual(
                set(alert),
                {"alertId", "type", "peakStart", "threshold", "suppressedCount"},
            )
            self.assertEqual(alert["threshold"], 2)

    def test_list_alerts_tie_breaks_by_alert_id_code_point_order(self) -> None:
        # Two types of the same organization both peak in window 0. The list
        # tie-breaks by alertId in Unicode code-point order.
        for event_id, event_type in (("x1", "t-a"), ("x2", "t-a")):
            self.post_event(
                make_event(eventId=event_id, type=event_type, occurredAt=5)
            )
        for event_id, event_type in (("y1", "t-b"), ("y2", "t-b")):
            self.post_event(
                make_event(eventId=event_id, type=event_type, occurredAt=7)
            )
        self.assertEqual(self.evaluate_alert(type="t-a")[1]["alertId"], "alert-1")
        self.assertEqual(self.evaluate_alert(type="t-b")[1]["alertId"], "alert-2")
        status, body = self.list_alerts()
        self.assertEqual(
            [a["alertId"] for a in body["alerts"]], ["alert-1", "alert-2"]
        )

    def test_list_alerts_includes_accumulated_suppression_counts(self) -> None:
        self.seed_events(10, 20)
        for _ in range(3):
            self.evaluate_alert()
        status, body = self.list_alerts()
        self.assertEqual(len(body["alerts"]), 1)
        self.assertEqual(body["alerts"][0]["suppressedCount"], 2)

    def test_list_alerts_unknown_organization_is_empty(self) -> None:
        self.seed_events(10, 20)
        self.evaluate_alert()
        status, body = self.list_alerts("organizationId=ghost")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"organizationId": "ghost", "alerts": []})

    def test_list_alerts_requires_single_non_empty_organization_id(self) -> None:
        for query in (
            "",
            "organizationId=",
            "organizationId=%20%20",
            "organizationId=a&organizationId=b",
        ):
            with self.subTest(query=query):
                status, body = self.list_alerts(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_list_alerts_is_isolated_from_other_organizations(self) -> None:
        self.seed_events(10, 20)
        self.evaluate_alert()
        status, body = self.list_alerts("organizationId=org-2")
        self.assertEqual(body["alerts"], [])

    # ----------------------------------------------------------- ledger / state

    def test_evaluation_does_not_modify_ledger(self) -> None:
        self.seed_events(10, 20, 30)
        self.evaluate_alert(threshold=2)
        self.evaluate_alert(threshold=2)
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 3)

    def test_concurrent_threshold_hits_create_one_alert(self) -> None:
        self.seed_events(10, 20)
        payload = self.alert_payload(suppressionWindow=10000)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: self.post_json("/alerts/evaluate", payload),
                    range(24),
                )
            )
        statuses = sorted(status for status, _ in results)
        self.assertEqual(statuses.count(200), 24)
        actions = sorted(body["action"] for _, body in results)
        self.assertEqual(actions.count("escalate"), 1)
        self.assertEqual(actions.count("suppress"), 23)
        ids = {body["alertId"] for _, body in results}
        self.assertEqual(ids, {"alert-1"})
        final = [body["suppressedCount"] for _, body in results]
        self.assertEqual(max(final), 23)
        status, listing = self.list_alerts()
        self.assertEqual(len(listing["alerts"]), 1)
        self.assertEqual(listing["alerts"][0]["suppressedCount"], 23)

    def test_alerts_cleared_on_restart(self) -> None:
        self.seed_events(10, 20)
        self.evaluate_alert()
        status, body = self.list_alerts()
        self.assertEqual(len(body["alerts"]), 1)

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(
                f"http://127.0.0.1:{fresh.server_port}/alerts?organizationId=org-1",
                timeout=2,
            ) as response:
                self.assertEqual(
                    json.load(response),
                    {"organizationId": "org-1", "alerts": []},
                )
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # -------------------------------------------------- branches / snapshots

    def test_alerts_endpoints_are_main_only(self) -> None:
        self.seed_events(10, 20)
        self.post_json("/snapshots", {"snapshotId": "snap-1"})
        self.post_json(
            "/branches", {"branchId": "br-1", "snapshotId": "snap-1"}
        )

        status, body = self.post_json(
            "/branches/br-1/alerts/evaluate", self.alert_payload()
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

        status, body = self.request("/branches/br-1/alerts?organizationId=org-1")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    def test_snapshots_do_not_capture_alerts(self) -> None:
        self.seed_events(10, 20)
        self.evaluate_alert()
        status, body = self.post_json("/snapshots", {"snapshotId": "snap-1"})
        self.assertEqual(status, 201)
        self.assertEqual(
            set(body), {"snapshotId", "events", "resources", "reservations"}
        )
        status, body = self.request("/snapshots")
        self.assertEqual(
            set(body["snapshots"][0]),
            {"snapshotId", "events", "resources", "reservations"},
        )

    def test_unknown_branch_error_body_contains_only_error_key(self) -> None:
        for path in (
            "/branches/ghost",
            "/branches/ghost/events?organizationId=org-1",
            "/branches/ghost/reservations?organizationId=org-1",
            "/branches/ghost/events/aggregate?organizationId=o&type=t&windowSize=1",
            "/branches/ghost/alerts",
            "/branches/ghost/alerts/evaluate",
        ):
            with self.subTest(method="GET", path=path):
                status, body = self.request(path)
                self.assertEqual(status, 404)
                self.assertEqual(body, {"error": "branch_not_found"})
        for path in (
            "/branches/ghost/events",
            "/branches/ghost/reservations",
            "/branches/ghost/decisions/evaluate",
            "/branches/ghost/alerts/evaluate",
        ):
            with self.subTest(method="POST", path=path):
                status, body = self.post_json(path, {})
                self.assertEqual(status, 404)
                self.assertEqual(body, {"error": "branch_not_found"})

    # ------------------------------------------------------ routing / health

    def test_health_and_unknown_paths_unchanged(self) -> None:
        with urlopen(f"{self.base_url}/health", timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                json.load(response),
                {"service": "real-world-event-decision-sim", "status": "ok"},
            )
        status, body = self.request("/missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, body = self.post_json("/alerts", {})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, body = self.request("/alerts/evaluate")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
