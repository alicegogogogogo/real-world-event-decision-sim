from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
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
        "payload": {"region": "north"},
    }
    event.update(overrides)
    return event


class RegionAlertTest(unittest.TestCase):
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
        content_type: str | None = "application/json",
        auth: bool = True,
        token: str | None = None,
    ) -> tuple[int, bytes, Any]:
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        elif auth:
            headers.update(
                _support.authorization_header(
                    self.token_cache, self.base_url, path, body
                )
            )
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
            "region": "north",
            "type": "incident.created",
            "windowSize": 60,
            "threshold": 2,
            "suppressionWindow": 100,
        }
        payload.update(overrides)
        return payload

    def evaluate_alert(self, **overrides: Any) -> tuple[int, Any]:
        return self.post_json(
            "/alerts/region/evaluate", self.alert_payload(**overrides)
        )

    def list_alerts(
        self, query: str = "organizationId=org-1&region=north"
    ) -> tuple[int, Any]:
        suffix = f"?{query}" if query else ""
        return self.request(f"/alerts/region{suffix}")

    def seed_events(self, *timestamps: int, **overrides: Any) -> None:
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
                "region": "north",
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
                "region": "north",
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

    def test_unknown_region_is_observe_and_creates_nothing(self) -> None:
        self.seed_events(10, 20)
        status, body = self.evaluate_alert(region="south")
        self.assertEqual(status, 200)
        self.assertEqual(body["region"], "south")
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")
        self.assertIsNone(body["alertId"])
        self.assertIsNone(body["suppressedCount"])
        status, listing = self.list_alerts("organizationId=org-1&region=south")
        self.assertEqual(listing["alerts"], [])

    def test_only_verbatim_region_events_count(self) -> None:
        # Events without a region, with an empty region, with a non-string
        # region, or with a different region never count toward "north".
        self.post_event(make_event(eventId="e1", occurredAt=10, payload={}))
        self.post_event(
            make_event(eventId="e2", occurredAt=11, payload={"region": ""})
        )
        self.post_event(
            make_event(eventId="e3", occurredAt=12, payload={"region": 7})
        )
        self.post_event(
            make_event(eventId="e4", occurredAt=13, payload={"region": "North"})
        )
        self.post_event(
            make_event(eventId="e5", occurredAt=14, payload={"region": "north "})
        )
        self.seed_events(15)
        status, body = self.evaluate_alert(threshold=2)
        self.assertEqual(body["peakCount"], 1)
        self.assertEqual(body["action"], "observe")

    def test_other_organization_events_never_count(self) -> None:
        for event_id, timestamp in (("x1", 10), ("x2", 20)):
            self.post_event(
                make_event(
                    eventId=event_id,
                    organizationId="org-2",
                    occurredAt=timestamp,
                )
            )
        status, body = self.evaluate_alert(threshold=2)
        self.assertEqual(body["peakCount"], 0)
        self.assertEqual(body["action"], "observe")

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
        self.assertEqual(second["action"], "suppress")
        self.assertEqual(second["alertId"], "alert-1")
        self.assertEqual(second["suppressedCount"], 1)

        status, third = self.evaluate_alert(suppressionWindow=100)
        self.assertEqual(third["action"], "suppress")
        self.assertEqual(third["alertId"], "alert-1")
        self.assertEqual(third["suppressedCount"], 2)

        status, body = self.list_alerts()
        self.assertEqual(len(body["alerts"]), 1)
        self.assertEqual(body["alerts"][0]["suppressedCount"], 2)

    def test_peak_at_exact_suppression_distance_creates_new_alert(self) -> None:
        self.seed_events(10, 20)
        status, first = self.evaluate_alert(suppressionWindow=60)
        self.assertEqual(first["action"], "escalate")
        self.assertEqual(first["alertId"], "alert-1")

        # Three events in window [60, 120) push its count strictly ahead of
        # window 0; the distance to the prior peak start is exactly 60.
        for event_id, timestamp in (("evt-a", 60), ("evt-b", 61), ("evt-c", 62)):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, occurredAt=timestamp)
                )[0],
                201,
            )
        status, second = self.evaluate_alert(suppressionWindow=60)
        self.assertEqual(second["action"], "escalate")
        self.assertEqual(second["alertId"], "alert-2")
        self.assertEqual(second["peakStart"], 60)
        self.assertEqual(second["suppressedCount"], 0)

    def test_alert_ids_increment_globally_across_both_alert_kinds(self) -> None:
        # An organization-dimension alert and region-dimension alerts draw
        # from one global sequence.
        self.seed_events(10, 20)
        status, org_alert = self.post_json(
            "/alerts/evaluate",
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "threshold": 2,
                "suppressionWindow": 100,
            },
        )
        self.assertEqual(org_alert["alertId"], "alert-1")

        status, region_alert = self.evaluate_alert()
        self.assertEqual(region_alert["alertId"], "alert-2")

        # A different region's first threshold hit takes the next id.
        for event_id, timestamp in (("s1", 10), ("s2", 20)):
            self.post_event(
                make_event(
                    eventId=event_id,
                    occurredAt=timestamp,
                    payload={"region": "south"},
                )
            )
        status, south = self.evaluate_alert(region="south")
        self.assertEqual(south["alertId"], "alert-3")

    def test_alert_kinds_never_suppress_each_other(self) -> None:
        # The same events reach the threshold in both dimensions; each kind
        # opens its own alert instead of suppressing the other.
        self.seed_events(10, 20)
        status, org_alert = self.post_json(
            "/alerts/evaluate",
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "threshold": 2,
                "suppressionWindow": 10000,
            },
        )
        self.assertEqual(org_alert["action"], "escalate")
        status, region_alert = self.evaluate_alert(suppressionWindow=10000)
        self.assertEqual(region_alert["action"], "escalate")
        self.assertEqual(region_alert["alertId"], "alert-2")

        # The organization list holds only the organization-dimension alert.
        status, org_listing = self.request("/alerts?organizationId=org-1")
        self.assertEqual([a["alertId"] for a in org_listing["alerts"]], ["alert-1"])

    def test_suppression_is_scoped_per_region_and_type(self) -> None:
        self.seed_events(10, 20)
        self.assertEqual(self.evaluate_alert()[1]["alertId"], "alert-1")

        # Same window start, different region: independent alert.
        for event_id, timestamp in (("s1", 10), ("s2", 20)):
            self.post_event(
                make_event(
                    eventId=event_id,
                    occurredAt=timestamp,
                    payload={"region": "south"},
                )
            )
        status, body = self.evaluate_alert(region="south")
        self.assertEqual(body["action"], "escalate")

        # Same region, different type: also independent.
        for event_id, timestamp in (("t1", 10), ("t2", 20)):
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
        status, body = self.evaluate_alert(
            suppressionWindow=1000, **{"from": 60, "to": 120}
        )
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["peakCount"], 2)
        # Distance from prior peak start 0 is 60 < 1000 -> suppressed.
        self.assertEqual(body["action"], "suppress")
        self.assertEqual(body["alertId"], "alert-1")

    # ------------------------------------------------------- response encoding

    def test_response_is_compact_stable_order_and_newline_terminated(self) -> None:
        self.seed_events(10, 20)
        status, raw, body = self.request_raw(
            "/alerts/region/evaluate",
            method="POST",
            body=json.dumps(self.alert_payload()).encode(),
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )

    def test_repeat_identical_requests_are_byte_identical(self) -> None:
        self.seed_events(10, 20)
        body = json.dumps(self.alert_payload(threshold=5)).encode()
        _, first, _ = self.request_raw(
            "/alerts/region/evaluate", method="POST", body=body
        )
        _, second, _ = self.request_raw(
            "/alerts/region/evaluate", method="POST", body=body
        )
        self.assertEqual(first, second)

    def test_strings_are_echoed_verbatim(self) -> None:
        payload = self.alert_payload(region=" North ")
        status, body = self.post_json("/alerts/region/evaluate", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["region"], " North ")

    # ------------------------------------------------------------- validation

    def test_missing_content_type_is_415(self) -> None:
        status, body = self.request(
            "/alerts/region/evaluate",
            method="POST",
            body=json.dumps(self.alert_payload()).encode(),
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_malformed_json_is_400_without_alert(self) -> None:
        status, body = self.request(
            "/alerts/region/evaluate", method="POST", body=b'{"organizationId": '
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")
        self.assertEqual(self.list_alerts()[1]["alerts"], [])

    def test_non_object_body_is_422(self) -> None:
        for bad_body in ([], "text", 42, None, True):
            with self.subTest(bad_body=bad_body):
                status, body = self.post_json("/alerts/region/evaluate", bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_missing_required_field_is_422(self) -> None:
        for field in (
            "organizationId",
            "region",
            "type",
            "windowSize",
            "threshold",
            "suppressionWindow",
        ):
            payload = self.alert_payload()
            del payload[field]
            with self.subTest(field=field):
                status, body = self.post_json("/alerts/region/evaluate", payload)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_extra_field_is_422(self) -> None:
        status, body = self.post_json(
            "/alerts/region/evaluate", self.alert_payload(extra="nope")
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_duplicate_json_key_is_422(self) -> None:
        status, body = self.request(
            "/alerts/region/evaluate",
            method="POST",
            body=b'{"organizationId":"org-1","organizationId":"org-1",'
            b'"region":"north","type":"incident.created","windowSize":60,'
            b'"threshold":2,"suppressionWindow":100}',
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_blank_or_non_string_identifiers_are_422(self) -> None:
        for field in ("organizationId", "region", "type"):
            for bad_value in ("", "   ", 123, None, ["x"], True):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_json(
                        "/alerts/region/evaluate",
                        self.alert_payload(**{field: bad_value}),
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_bad_positive_integer_fields_are_422(self) -> None:
        for field in ("windowSize", "threshold", "suppressionWindow"):
            for bad_value in (0, -1, 1.5, 1.0, "60", True, None, [3]):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_json(
                        "/alerts/region/evaluate",
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
                    "/alerts/region/evaluate", self.alert_payload(**overrides)
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_failed_requests_never_write_alerts(self) -> None:
        self.seed_events(10, 20)
        bad = self.alert_payload(suppressionWindow=0)
        self.assertEqual(self.post_json("/alerts/region/evaluate", bad)[0], 422)
        self.assertEqual(
            self.request(
                "/alerts/region/evaluate",
                method="POST",
                body=b"{",
            )[0],
            400,
        )
        self.assertEqual(self.list_alerts()[1]["alerts"], [])

    # --------------------------------------------------------- GET /alerts/region

    def test_list_filters_by_region_and_sorts(self) -> None:
        for event_id, region, event_type, timestamp in (
            ("a1", "north", "t-b", 200),
            ("a2", "north", "t-b", 201),
            ("a3", "north", "t-a", 0),
            ("a4", "north", "t-a", 1),
            ("a5", "south", "t-a", 0),
            ("a6", "south", "t-a", 1),
        ):
            self.post_event(
                make_event(
                    eventId=event_id,
                    type=event_type,
                    occurredAt=timestamp,
                    payload={"region": region},
                )
            )
        self.evaluate_alert(region="north", type="t-b", suppressionWindow=10000)
        self.evaluate_alert(region="north", type="t-a", suppressionWindow=10000)
        self.evaluate_alert(region="south", type="t-a", suppressionWindow=10000)

        status, body = self.list_alerts("organizationId=org-1&region=north")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(body["region"], "north")
        self.assertEqual(
            [(a["type"], a["peakStart"]) for a in body["alerts"]],
            [("t-a", 0), ("t-b", 180)],
        )
        for alert in body["alerts"]:
            self.assertEqual(
                set(alert), {"type", "peakStart", "threshold", "suppressedCount"}
            )
            self.assertEqual(alert["threshold"], 2)

        status, body = self.list_alerts("organizationId=org-1&region=south")
        self.assertEqual(len(body["alerts"]), 1)

    def test_list_unknown_region_is_empty_array(self) -> None:
        self.seed_events(10, 20)
        self.evaluate_alert()
        status, body = self.list_alerts("organizationId=org-1&region=ghost")
        self.assertEqual(status, 200)
        self.assertEqual(
            body, {"organizationId": "org-1", "region": "ghost", "alerts": []}
        )

    def test_list_requires_single_non_empty_parameters(self) -> None:
        for query in (
            "",
            "organizationId=org-1",
            "region=north",
            "organizationId=&region=north",
            "organizationId=org-1&region=",
            "organizationId=org-1&region=%20%20",
            "organizationId=org-1&organizationId=org-2&region=north",
            "organizationId=org-1&region=north&region=south",
        ):
            with self.subTest(query=query):
                status, body = self.list_alerts(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_list_is_isolated_from_other_organizations(self) -> None:
        self.seed_events(10, 20)
        self.evaluate_alert()
        status, body = self.list_alerts("organizationId=org-2&region=north")
        self.assertEqual(body["alerts"], [])

    # ----------------------------------------------------------- authorization

    def register_token(self, token: str, organization_id: str, role: str) -> None:
        status, _ = self.post_json(
            "/auth/tokens",
            {"token": token, "organizationId": organization_id, "role": role},
        )
        self.assertIn(status, (200, 201))

    def test_missing_or_unregistered_credential_is_401(self) -> None:
        for path, method, body in (
            ("/alerts/region/evaluate", "POST", json.dumps(self.alert_payload()).encode()),
            ("/alerts/region?organizationId=org-1&region=north", "GET", None),
        ):
            with self.subTest(path=path):
                status, _, parsed = self.request_raw(
                    path, method=method, body=body, auth=False
                )
                self.assertEqual(status, 401)
                self.assertEqual(parsed, {"error": "unauthorized"})
            with self.subTest(path=path, token="ghost"):
                status, _, parsed = self.request_raw(
                    path, method=method, body=body, token="ghost"
                )
                self.assertEqual(status, 401)
                self.assertEqual(parsed, {"error": "unauthorized"})

    def test_read_credential_cannot_evaluate_but_can_list(self) -> None:
        self.register_token("reader", "org-1", "read")
        self.seed_events(10, 20)
        status, _, body = self.request_raw(
            "/alerts/region/evaluate",
            method="POST",
            body=json.dumps(self.alert_payload()).encode(),
            token="reader",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        self.assertEqual(self.list_alerts()[1]["alerts"], [])

        # The same read credential may list.
        self.evaluate_alert()
        status, _, body = self.request_raw(
            "/alerts/region?organizationId=org-1&region=north", token="reader"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["alerts"]), 1)

    def test_cross_organization_is_403(self) -> None:
        self.register_token("org2-writer", "org-2", "write")
        self.seed_events(10, 20)
        status, _, body = self.request_raw(
            "/alerts/region/evaluate",
            method="POST",
            body=json.dumps(self.alert_payload()).encode(),
            token="org2-writer",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        self.assertEqual(self.list_alerts()[1]["alerts"], [])

        status, _, body = self.request_raw(
            "/alerts/region?organizationId=org-1&region=north", token="org2-writer"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

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
                    lambda _: self.post_json("/alerts/region/evaluate", payload),
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
        status, listing = self.list_alerts()
        self.assertEqual(len(listing["alerts"]), 1)
        self.assertEqual(listing["alerts"][0]["suppressedCount"], 23)

    def test_alerts_cleared_on_restart(self) -> None:
        self.seed_events(10, 20)
        self.evaluate_alert()
        self.assertEqual(len(self.list_alerts()[1]["alerts"]), 1)

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            fresh_base = f"http://127.0.0.1:{fresh.server_port}"
            token = _support.ensure_token({}, fresh_base, "org-1")
            request = Request(
                f"{fresh_base}/alerts/region?organizationId=org-1&region=north",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(request, timeout=2) as response:
                self.assertEqual(
                    json.load(response),
                    {"organizationId": "org-1", "region": "north", "alerts": []},
                )
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
