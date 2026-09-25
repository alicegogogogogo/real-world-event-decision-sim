from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlsplit
from urllib.request import Request, urlopen

from event_sim.server import create_server

# Sentinel for the request helpers: pick a registered token matching the
# request's organization (defaulting to org-1) instead of an explicit one.
_AUTO_TOKEN = object()


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


class RegionEndpointsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self._tokens: dict[tuple[str, str], str] = {}

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def register_token(self, organization_id: str = "org-1", role: str = "write") -> str:
        key = (organization_id, role)
        token = self._tokens.get(key)
        if token is None:
            token = "test-token-" + quote(f"{organization_id}-{role}", safe="")
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
            self._tokens[key] = token
        return token

    def _auto_token(self, path: str, body: bytes | None) -> str:
        organization_id = None
        if body is not None:
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, ValueError):
                payload = None
            if isinstance(payload, dict) and isinstance(
                payload.get("organizationId"), str
            ):
                organization_id = payload["organizationId"]
        if organization_id is None or not organization_id.strip():
            values = parse_qs(urlsplit(path).query).get("organizationId")
            if values and len(values) == 1 and values[0].strip():
                organization_id = values[0]
            else:
                organization_id = "org-1"
        return self.register_token(organization_id)

    def request_raw(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        token: Any = _AUTO_TOKEN,
    ) -> tuple[int, bytes, Any]:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if token is _AUTO_TOKEN:
            token = self._auto_token(path, body)
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                return response.status, raw, json.loads(raw)
        except HTTPError as error:
            raw = error.read()
            try:
                return error.code, raw, json.loads(raw)
            finally:
                error.close()

    def get(self, query: str) -> tuple[int, Any]:
        status, _, body = self.request_raw(f"/events/region?{query}")
        return status, body

    def aggregate(self, query: str) -> tuple[int, Any]:
        status, _, body = self.request_raw(f"/events/region/aggregate?{query}")
        return status, body

    def post_event(self, event: dict[str, Any]) -> tuple[int, Any]:
        status, _, body = self.request_raw(
            "/events", method="POST", body=json.dumps(event).encode()
        )
        return status, body

    def seed_region_events(self) -> None:
        events = [
            make_event(eventId="evt-n1", occurredAt=10),
            make_event(eventId="evt-n2", occurredAt=59),
            make_event(eventId="evt-n3", occurredAt=60),
            make_event(eventId="evt-n4", occurredAt=60),
            make_event(eventId="evt-s1", occurredAt=5, payload={"region": "south"}),
            # Region-bearing but a different type.
            make_event(
                eventId="evt-u1",
                type="incident.updated",
                occurredAt=10,
                payload={"region": "north"},
            ),
            # No region attribution: empty string, missing key, non-string.
            make_event(eventId="evt-z1", occurredAt=11, payload={"region": ""}),
            make_event(eventId="evt-z2", occurredAt=12, payload={}),
            make_event(eventId="evt-z3", occurredAt=13, payload={"region": 7}),
            make_event(eventId="evt-z4", occurredAt=14, payload={"region": None}),
            # Same region in another organization.
            make_event(
                eventId="evt-o2",
                organizationId="org-2",
                occurredAt=1,
                payload={"region": "north"},
            ),
        ]
        for event in events:
            self.assertEqual(self.post_event(event)[0], 201)

    # ------------------------------------------------------------- attribution

    def test_only_non_empty_string_payload_region_attributes(self) -> None:
        self.seed_region_events()
        status, body = self.get("organizationId=org-1&region=north")
        self.assertEqual(status, 200)
        # The list endpoint has no type filter: evt-u1 (another type in the
        # same region) is included; sorting is occurredAt then eventId.
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["evt-n1", "evt-u1", "evt-n2", "evt-n3", "evt-n4"],
        )

    def test_events_without_region_never_match_any_region(self) -> None:
        self.seed_region_events()
        # The four unattributed events (empty string, missing key, number,
        # null) appear under neither north nor south.
        status, body = self.get("organizationId=org-1&region=north")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 5)
        status, body = self.get("organizationId=org-1&region=south")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]], ["evt-s1"]
        )
        # A blank region query is a validation error, not an "unattributed"
        # lookup.
        status, body = self.get("organizationId=org-1&region=")
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_region_matching_is_verbatim_exact(self) -> None:
        for event_id, region in (
            ("evt-1", " North "),
            ("evt-2", "nOrth"),
            ("evt-3", "北"),
        ):
            self.assertEqual(
                self.post_event(
                    make_event(eventId=event_id, payload={"region": region})
                )[0],
                201,
            )
        for region, expected in (
            (" North ", ["evt-1"]),
            ("%20North%20", ["evt-1"]),
            ("north", []),
            ("nOrth", ["evt-2"]),
            ("North", []),
            ("北", ["evt-3"]),
        ):
            with self.subTest(region=region):
                status, body = self.get(
                    f"organizationId=org-1&region={quote(region, safe='%')}"
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    [event["eventId"] for event in body["events"]], expected
                )

    # ------------------------------------------------------------- GET /events/region

    def test_list_filters_by_organization_and_region_with_list_sorting(self) -> None:
        for event in (
            make_event(eventId="evt-b", occurredAt=200),
            make_event(eventId="evt-a", occurredAt=200),
            make_event(eventId="evt-c", occurredAt=100),
            make_event(
                eventId="evt-x",
                organizationId="org-2",
                occurredAt=50,
                payload={"region": "north"},
            ),
            make_event(
                eventId="evt-s",
                occurredAt=50,
                payload={"region": "south"},
            ),
        ):
            self.assertEqual(self.post_event(event)[0], 201)

        status, body = self.get("organizationId=org-1&region=north")
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(body["region"], "north")
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["evt-c", "evt-a", "evt-b"],
        )

    def test_list_response_shape_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.assertEqual(
            self.post_event(make_event(payload={"region": "north", "k": 1}))[0],
            201,
        )
        status, raw, body = self.request_raw(
            "/events/region?organizationId=org-1&region=north"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(set(body), {"organizationId", "region", "events"})
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )

    def test_list_unknown_region_or_organization_is_empty(self) -> None:
        self.seed_region_events()
        for query in (
            "organizationId=org-1&region=nowhere",
            "organizationId=org-x&region=north",
            "organizationId=org-x&region=nowhere",
        ):
            with self.subTest(query=query):
                status, body = self.get(query)
                self.assertEqual(status, 200)
                self.assertEqual(body["events"], [])
                self.assertNotIn("org-2", json.dumps(body))

    def test_list_echoes_region_verbatim_and_ignores_unrelated_params(self) -> None:
        status, body = self.get(
            f"organizationId=org-1&region={quote(' R-9 ')}&limit=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], "org-1")
        self.assertEqual(body["region"], " R-9 ")
        self.assertEqual(body["events"], [])

    def test_list_requires_organization_id_and_region_each_once_non_empty(self) -> None:
        for query in (
            "region=north",
            "organizationId=org-1",
            "organizationId=&region=north",
            "organizationId=%20&region=north",
            "organizationId=org-1&region=",
            "organizationId=org-1&region=%20%20",
            "organizationId=org-1&region=north&region=south",
            "organizationId=a&organizationId=b&region=north",
        ):
            with self.subTest(query=query):
                status, body = self.get(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    # ----------------------------------------------------- GET /events/region/aggregate

    def test_aggregate_without_range_returns_only_covered_windows(self) -> None:
        self.seed_region_events()
        status, body = self.aggregate(
            "organizationId=org-1&region=north&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "region": "north",
                "type": "incident.created",
                "windowSize": 60,
                "from": None,
                "to": None,
                "windows": [
                    {"start": 0, "end": 60, "count": 2},
                    {"start": 60, "end": 120, "count": 2},
                ],
            },
        )

    def test_aggregate_with_range_keeps_empty_windows_and_filters(self) -> None:
        self.seed_region_events()
        status, body = self.aggregate(
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&from=59&to=180"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 2},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 0},
            ],
        )

    def test_aggregate_boundaries_and_type_region_org_isolation(self) -> None:
        self.seed_region_events()
        # Event at exactly 60 belongs to the upper window.
        status, body = self.aggregate(
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&from=60&to=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 60, "end": 120, "count": 2}]
        )
        # The other type in the same region is not counted.
        status, body = self.aggregate(
            "organizationId=org-1&region=north&type=incident.updated&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 0, "end": 60, "count": 1}]
        )
        # Another organization's same-named region is never counted.
        status, body = self.aggregate(
            "organizationId=org-2&region=north&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"], [{"start": 0, "end": 60, "count": 1}]
        )

    def test_aggregate_unknown_region_matches_zero(self) -> None:
        self.seed_region_events()
        base = "organizationId=org-1&region=nowhere&type=incident.created"
        status, body = self.aggregate(f"{base}&windowSize=60")
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])
        status, body = self.aggregate(f"{base}&windowSize=10&from=5&to=25")
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 10, "count": 0},
                {"start": 10, "end": 20, "count": 0},
                {"start": 20, "end": 30, "count": 0},
            ],
        )

    def test_aggregate_is_deterministic_and_read_only(self) -> None:
        self.seed_region_events()
        query = (
            "organizationId=org-1&region=north&type=incident.created&windowSize=60"
        )
        first_status, first = self.aggregate(query)
        second_status, second = self.aggregate(query)
        self.assertEqual((first_status, second_status), (200, 200))
        self.assertEqual(first, second)

        # Reads never write the ledger, reservations, or alerts.
        status, _, listing = self.request_raw(
            "/events?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 10)
        status, raw, _ = self.request_raw("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertIn(b'"alerts":[]', raw)

    def test_aggregate_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_region_events()
        status, raw, body = self.request_raw(
            "/events/region/aggregate"
            "?organizationId=org-1&region=north&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(
            set(body),
            {
                "organizationId",
                "region",
                "type",
                "windowSize",
                "from",
                "to",
                "windows",
            },
        )
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )

    def test_aggregate_requires_each_parameter_exactly_once(self) -> None:
        base = (
            "organizationId=org-1&region=north&type=incident.created&windowSize=60"
        )
        for query in (
            "",
            "region=north&type=incident.created&windowSize=60",
            "organizationId=org-1&type=incident.created&windowSize=60",
            "organizationId=org-1&region=north&windowSize=60",
            "organizationId=org-1&region=north&type=incident.created",
            f"{base}&organizationId=org-1",
            f"{base}&region=north",
            f"{base}&type=incident.created",
            f"{base}&windowSize=60",
        ):
            with self.subTest(query=query):
                status, body = self.aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_rejects_blank_and_invalid_values(self) -> None:
        for query in (
            "organizationId=&region=r&type=t&windowSize=1",
            "organizationId=o&region=&type=t&windowSize=1",
            "organizationId=o&region=r&type=&windowSize=1",
            "organizationId=o&region=r&type=t&windowSize=",
            "organizationId=o&region=r&type=t&windowSize=0",
            "organizationId=o&region=r&type=t&windowSize=-5",
            "organizationId=o&region=r&type=t&windowSize=1.5",
            "organizationId=o&region=r&type=t&windowSize=abc",
            "organizationId=o&region=r&type=t&windowSize=1&extra=ignored&from=x&to=2",
        ):
            with self.subTest(query=query):
                status, body = self.aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_from_to_must_be_paired_non_negative_and_ordered(self) -> None:
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
                status, body = self.aggregate(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_aggregate_accepts_equal_from_and_to(self) -> None:
        status, body = self.aggregate(
            "organizationId=o&region=r&type=t&windowSize=10&from=0&to=0"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [{"start": 0, "end": 10, "count": 0}])

    # ------------------------------------------------------- POST /events unchanged

    def test_post_events_accepts_and_stores_region_payload_unchanged(self) -> None:
        # The five-field validation, replay and conflict behavior is untouched:
        # region is just an ordinary payload key.
        event = make_event(payload={"region": "north", "severity": "low"})
        status, body = self.post_event(event)
        self.assertEqual(status, 201)
        self.assertEqual(body, event)
        status, body = self.post_event(event)
        self.assertEqual(status, 200)
        self.assertEqual(body, event)

    # ------------------------------------------------------------- branch prefix

    def test_branch_prefixes_do_not_expose_region_queries(self) -> None:
        # Snapshot/branch capture still only sees events; no region sub-paths
        # are added under branches.
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

        # Known branch: standard not_found for the unknown sub-path.
        status, _, body = self.request_raw(
            "/branches/br-1/events/region?organizationId=org-1&region=north"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, _, body = self.request_raw(
            "/branches/br-1/events/region/aggregate"
            "?organizationId=org-1&region=north&type=t&windowSize=60"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

        # Unknown branch keeps the branch_not_found precedence.
        status, _, body = self.request_raw(
            "/branches/ghost/events/region?organizationId=org-1&region=north"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")


if __name__ == "__main__":
    unittest.main()
