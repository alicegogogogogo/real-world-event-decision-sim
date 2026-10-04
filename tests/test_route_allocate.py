"""Tests for POST /decisions/route-allocate.

The endpoint is a read-only, stateless planner: it routes each demand
(indivisible, processed by priority descending then demandId in Unicode
code-point order) from the chosen resource's location to the demand's
location over the directed edge set, picking the sufficient-capacity
reachable resource with the cheapest shortest path, breaking cost ties by
resourceId ascending and equal-cost path ties by the code-point-smallest
node sequence. Capacity is deducted only within the single computation.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server
from tests import _support


class RouteAllocateTest(unittest.TestCase):
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

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        content_type: str | None = "application/json",
        auth: bool = True,
    ) -> tuple[int, Any]:
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if auth:
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
                return response.status, json.load(response)
        except HTTPError as error:
            try:
                return error.code, json.load(error)
            finally:
                error.close()

    def post_route_allocation(
        self, payload: Any, *, raw: bool = False, content_type: str | None = "application/json"
    ) -> tuple[int, Any]:
        body = payload if raw else json.dumps(payload).encode()
        return self.request(
            "/decisions/route-allocate",
            method="POST",
            body=body,
            content_type=content_type,
        )

    def route_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "edges": [
                {"from": "A", "to": "B", "cost": 2},
                {"from": "B", "to": "C", "cost": 1},
                {"from": "A", "to": "C", "cost": 9},
            ],
            "demands": [
                {"demandId": "d-low", "location": "C", "units": 3, "priority": 1},
                {"demandId": "d-b", "location": "C", "units": 4, "priority": 5},
                {"demandId": "d-a", "location": "B", "units": 2, "priority": 5},
            ],
            "resources": [
                {"resourceId": "r-b", "location": "B", "capacity": 6},
                {"resourceId": "r-a", "location": "A", "capacity": 5},
            ],
        }
        payload.update(overrides)
        return payload

    # ------------------------------------------------------------ happy path

    def test_routes_by_priority_then_id_and_deducts_capacity(self) -> None:
        # Processing order: d-a(5), d-b(5), d-low(1).
        # d-a -> B: r-b is on B (cost 0, cap 6 -> 4); r-a costs 2. Picks r-b.
        # d-b -> C: r-b has 4 left, B->C cost 1; r-a A->B->C cost 3. Picks r-b
        # (cap 4 -> 0). d-low -> C: r-b full, r-a cost 3, cap 5 -> 2.
        status, body = self.post_route_allocation(self.route_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "assignments": [
                    {
                        "demandId": "d-a",
                        "resourceId": "r-b",
                        "units": 2,
                        "path": ["B"],
                        "travelCost": 0,
                    },
                    {
                        "demandId": "d-b",
                        "resourceId": "r-b",
                        "units": 4,
                        "path": ["B", "C"],
                        "travelCost": 1,
                    },
                    {
                        "demandId": "d-low",
                        "resourceId": "r-a",
                        "units": 3,
                        "path": ["A", "B", "C"],
                        "travelCost": 3,
                    },
                ],
                "unassigned": [],
                "totalUnits": 9,
                "totalTravelCost": 4,
            },
        )

    def test_response_has_exactly_fixed_fields(self) -> None:
        status, body = self.post_route_allocation(self.route_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            set(body),
            {
                "organizationId",
                "assignments",
                "unassigned",
                "totalUnits",
                "totalTravelCost",
            },
        )
        for assignment in body["assignments"]:
            self.assertEqual(
                set(assignment),
                {"demandId", "resourceId", "units", "path", "travelCost"},
            )

    def test_shortest_path_prefers_multi_hop_over_expensive_direct(self) -> None:
        # Direct A->C costs 9; A->B->C costs 3.
        status, body = self.post_route_allocation(
            self.route_payload(
                demands=[{"demandId": "d", "location": "C", "units": 1, "priority": 0}],
                resources=[{"resourceId": "r", "location": "A", "capacity": 1}],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"][0]["path"], ["A", "B", "C"])
        self.assertEqual(body["assignments"][0]["travelCost"], 3)

    def test_same_location_path_is_single_node_zero_cost(self) -> None:
        status, body = self.post_route_allocation(
            self.route_payload(
                edges=[],
                demands=[{"demandId": "d", "location": "L", "units": 2, "priority": 0}],
                resources=[{"resourceId": "r", "location": "L", "capacity": 2}],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["assignments"],
            [
                {
                    "demandId": "d",
                    "resourceId": "r",
                    "units": 2,
                    "path": ["L"],
                    "travelCost": 0,
                }
            ],
        )
        self.assertEqual(body["totalTravelCost"], 0)

    def test_cost_tie_breaks_by_resource_id(self) -> None:
        status, body = self.post_route_allocation(
            self.route_payload(
                edges=[
                    {"from": "A", "to": "D", "cost": 5},
                    {"from": "B", "to": "D", "cost": 5},
                ],
                demands=[{"demandId": "d", "location": "D", "units": 1, "priority": 0}],
                resources=[
                    {"resourceId": "r-b", "location": "B", "capacity": 1},
                    {"resourceId": "r-a", "location": "A", "capacity": 1},
                ],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"][0]["resourceId"], "r-a")

    def test_equal_cost_paths_pick_code_point_smallest_sequence(self) -> None:
        # A->C->D and A->B->D both cost 2; [A, B, D] < [A, C, D].
        status, body = self.post_route_allocation(
            self.route_payload(
                edges=[
                    {"from": "A", "to": "C", "cost": 1},
                    {"from": "C", "to": "D", "cost": 1},
                    {"from": "A", "to": "B", "cost": 1},
                    {"from": "B", "to": "D", "cost": 1},
                ],
                demands=[{"demandId": "d", "location": "D", "units": 1, "priority": 0}],
                resources=[{"resourceId": "r", "location": "A", "capacity": 1}],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"][0]["path"], ["A", "B", "D"])

    def test_edges_are_directed(self) -> None:
        # Only C->A exists, so a resource at A cannot reach C.
        status, body = self.post_route_allocation(
            self.route_payload(
                edges=[{"from": "C", "to": "A", "cost": 1}],
                demands=[{"demandId": "d", "location": "C", "units": 1, "priority": 0}],
                resources=[{"resourceId": "r", "location": "A", "capacity": 1}],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"], [])
        self.assertEqual(
            body["unassigned"], [{"demandId": "d", "reason": "unreachable"}]
        )

    def test_unassigned_reasons(self) -> None:
        # d-big: no resource has capacity 10 -> insufficient_capacity even
        # though r-a could reach C. d-lost: r-b has capacity but cannot
        # reach Z -> unreachable.
        status, body = self.post_route_allocation(
            self.route_payload(
                demands=[
                    {"demandId": "d-big", "location": "C", "units": 10, "priority": 9},
                    {"demandId": "d-lost", "location": "Z", "units": 1, "priority": 1},
                ],
                resources=[
                    {"resourceId": "r-a", "location": "A", "capacity": 5},
                    {"resourceId": "r-b", "location": "B", "capacity": 1},
                ],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"], [])
        self.assertEqual(
            body["unassigned"],
            [
                {"demandId": "d-big", "reason": "insufficient_capacity"},
                {"demandId": "d-lost", "reason": "unreachable"},
            ],
        )
        self.assertEqual(body["totalUnits"], 0)
        self.assertEqual(body["totalTravelCost"], 0)

    def test_unassigned_follow_processing_order(self) -> None:
        status, body = self.post_route_allocation(
            self.route_payload(
                edges=[],
                demands=[
                    {"demandId": "zeta", "location": "L", "units": 1, "priority": 1},
                    {"demandId": "alpha", "location": "L", "units": 1, "priority": 9},
                    {"demandId": "mid", "location": "L", "units": 1, "priority": 5},
                ],
                resources=[],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [entry["demandId"] for entry in body["unassigned"]],
            ["alpha", "mid", "zeta"],
        )

    def test_demand_never_split_or_oversold(self) -> None:
        status, body = self.post_route_allocation(
            self.route_payload(
                edges=[],
                demands=[
                    {"demandId": "d1", "location": "L", "units": 5, "priority": 0},
                    {"demandId": "d2", "location": "L", "units": 3, "priority": 0},
                    {"demandId": "d3", "location": "L", "units": 3, "priority": 0},
                ],
                resources=[
                    {"resourceId": "r1", "location": "L", "capacity": 3},
                    {"resourceId": "r2", "location": "L", "capacity": 3},
                ],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [a["demandId"] for a in body["assignments"]], ["d2", "d3"]
        )
        self.assertEqual(
            body["unassigned"], [{"demandId": "d1", "reason": "insufficient_capacity"}]
        )
        self.assertEqual(body["totalUnits"], 6)

    def test_empty_arrays_succeed(self) -> None:
        for payload in (
            {"organizationId": "o", "edges": [], "demands": [], "resources": []},
            {
                "organizationId": "o",
                "edges": [],
                "demands": [],
                "resources": [{"resourceId": "r", "location": "L", "capacity": 5}],
            },
            {
                "organizationId": "o",
                "edges": [],
                "demands": [{"demandId": "d", "location": "L", "units": 1, "priority": 0}],
                "resources": [],
            },
        ):
            with self.subTest(payload=payload):
                status, body = self.post_route_allocation(payload)
                self.assertEqual(status, 200)
                self.assertEqual(body["organizationId"], "o")
                self.assertEqual(body["assignments"], [])
                self.assertEqual(body["totalUnits"], 0)
                self.assertEqual(body["totalTravelCost"], 0)

    def test_zero_priority_accepted(self) -> None:
        status, body = self.post_route_allocation(
            self.route_payload(
                demands=[{"demandId": "d", "location": "B", "units": 1, "priority": 0}],
                resources=[{"resourceId": "r", "location": "A", "capacity": 1}],
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"][0]["demandId"], "d")

    def test_same_priority_orders_by_unicode_demand_id(self) -> None:
        status, body = self.post_route_allocation(
            self.route_payload(
                edges=[],
                demands=[
                    {"demandId": "b", "location": "L", "units": 2, "priority": 0},
                    {"demandId": "ä", "location": "L", "units": 2, "priority": 0},
                    {"demandId": "a", "location": "L", "units": 2, "priority": 0},
                ],
                resources=[{"resourceId": "r", "location": "L", "capacity": 6}],
            )
        )
        self.assertEqual(status, 200)
        # Python's default string order is Unicode code-point order:
        # a < b < ä (U+00E4).
        self.assertEqual(
            [a["demandId"] for a in body["assignments"]], ["a", "b", "ä"]
        )

    def test_deterministic_independent_of_input_order(self) -> None:
        first_payload = self.route_payload()
        second_payload = self.route_payload()
        second_payload["edges"] = list(reversed(second_payload["edges"]))
        second_payload["demands"] = list(reversed(second_payload["demands"]))
        second_payload["resources"] = list(reversed(second_payload["resources"]))
        first = self.post_route_allocation(first_payload)
        second = self.post_route_allocation(second_payload)
        self.assertEqual(first, second)

    def test_repeated_requests_return_identical_json(self) -> None:
        payload = self.route_payload()
        results = [self.post_route_allocation(payload) for _ in range(5)]
        self.assertTrue(all(status == 200 for status, _ in results))
        encoded = {json.dumps(body, sort_keys=True) for _, body in results}
        self.assertEqual(len(encoded), 1)

    def test_is_read_only(self) -> None:
        event = {
            "eventId": "evt-1",
            "organizationId": "org-1",
            "type": "incident.created",
            "occurredAt": 0,
            "payload": {"severity": "low"},
        }
        status, _ = self.request("/events", method="POST", body=json.dumps(event).encode())
        self.assertEqual(status, 201)
        self.post_route_allocation(self.route_payload())
        self.post_route_allocation(self.route_payload())
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)
        status, body = self.request("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(body["alerts"], [])
        status, body = self.request("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(body["snapshots"], [])

    # ------------------------------------------------------------- auth

    def test_missing_or_bad_token_is_401(self) -> None:
        payload = json.dumps(self.route_payload()).encode()
        status, body = self.request(
            "/decisions/route-allocate", method="POST", body=payload, auth=False
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")
        status, body = self.request(
            "/decisions/route-allocate",
            method="POST",
            body=payload,
            auth=False,
        )
        self.assertEqual(status, 401)

    def test_unregistered_token_is_401(self) -> None:
        request = Request(
            f"{self.base_url}/decisions/route-allocate",
            data=json.dumps(self.route_payload()).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer nope",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                self.fail(f"expected 401, got {response.status}")
        except HTTPError as error:
            self.assertEqual(error.code, 401)
            self.assertEqual(json.load(error)["error"], "unauthorized")
            error.close()

    def test_cross_organization_is_403(self) -> None:
        # Register a write token for org-2, then ask it to plan for org-1.
        other = json.dumps(
            {"token": "org2-token", "organizationId": "org-2", "role": "write"}
        ).encode()
        request = Request(
            f"{self.base_url}/auth/tokens",
            data=other,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5):
            pass
        request = Request(
            f"{self.base_url}/decisions/route-allocate",
            data=json.dumps(self.route_payload()).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer org2-token",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                self.fail(f"expected 403, got {response.status}")
        except HTTPError as error:
            self.assertEqual(error.code, 403)
            self.assertEqual(json.load(error)["error"], "forbidden")
            error.close()

    def test_read_token_may_call(self) -> None:
        reader = json.dumps(
            {"token": "reader-token", "organizationId": "org-1", "role": "read"}
        ).encode()
        request = Request(
            f"{self.base_url}/auth/tokens",
            data=reader,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5):
            pass
        request = Request(
            f"{self.base_url}/decisions/route-allocate",
            data=json.dumps(self.route_payload()).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer reader-token",
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)

    # ------------------------------------------------------------- media/JSON

    def test_content_type_with_charset_is_accepted(self) -> None:
        status, _ = self.post_route_allocation(
            self.route_payload(), content_type="application/json; charset=utf-8"
        )
        self.assertEqual(status, 200)

    def test_missing_content_type_is_415(self) -> None:
        status, body = self.post_route_allocation(
            self.route_payload(), content_type=None
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_unsupported_content_type_is_415(self) -> None:
        status, body = self.post_route_allocation(
            self.route_payload(), content_type="text/plain"
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_malformed_json_is_400(self) -> None:
        status, body = self.post_route_allocation(b'{"organizationId": ', raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_duplicate_json_keys_are_422(self) -> None:
        status, body = self.post_route_allocation(
            b'{"organizationId": "org-1", "organizationId": "org-1",'
            b' "edges": [], "demands": [], "resources": []}',
            raw=True,
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    # ------------------------------------------------------------- validation

    def test_non_object_body_is_422(self) -> None:
        for bad_body in ([], "text", 42, None, True):
            with self.subTest(bad_body=bad_body):
                status, body = self.post_route_allocation(bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_missing_required_field_is_422(self) -> None:
        for field in ("organizationId", "edges", "demands", "resources"):
            payload = self.route_payload()
            del payload[field]
            with self.subTest(field=field):
                status, body = self.post_route_allocation(payload)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_extra_top_level_field_is_422(self) -> None:
        status, body = self.post_route_allocation(self.route_payload(extra="nope"))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_blank_or_non_string_organization_id_is_422(self) -> None:
        for bad_value in ("", "   ", 123, None, ["x"], True):
            with self.subTest(bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(organizationId=bad_value)
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_arrays_must_be_arrays(self) -> None:
        for field in ("edges", "demands", "resources"):
            for bad_value in ({}, "x", 1, None, True):
                with self.subTest(field=field, bad_value=bad_value):
                    status, body = self.post_route_allocation(
                        self.route_payload(**{field: bad_value})
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_element_wrong_type_is_422(self) -> None:
        good = {
            "edges": {"from": "A", "to": "B", "cost": 1},
            "demands": {"demandId": "d", "location": "L", "units": 1, "priority": 0},
            "resources": {"resourceId": "r", "location": "L", "capacity": 1},
        }
        for field in ("edges", "demands", "resources"):
            for bad_element in ("x", 1, None, True, ["x"]):
                payload = self.route_payload(**{field: [bad_element, good[field]]})
                with self.subTest(field=field, bad_element=bad_element):
                    status, body = self.post_route_allocation(payload)
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_element_missing_or_extra_field_is_422(self) -> None:
        edge = {"from": "A", "to": "B", "cost": 1}
        demand = {"demandId": "d", "location": "L", "units": 1, "priority": 0}
        resource = {"resourceId": "r", "location": "L", "capacity": 1}
        cases = []
        for bad_edge in (
            {"to": "B", "cost": 1},
            {"from": "A", "cost": 1},
            {"from": "A", "to": "B"},
            {**edge, "extra": 1},
        ):
            cases.append(("edges", [bad_edge]))
        for bad_demand in (
            {"location": "L", "units": 1, "priority": 0},
            {"demandId": "d", "units": 1, "priority": 0},
            {"demandId": "d", "location": "L", "priority": 0},
            {"demandId": "d", "location": "L", "units": 1},
            {**demand, "extra": 1},
        ):
            cases.append(("demands", [bad_demand]))
        for bad_resource in (
            {"location": "L", "capacity": 1},
            {"resourceId": "r", "capacity": 1},
            {"resourceId": "r", "location": "L"},
            {**resource, "extra": 1},
        ):
            cases.append(("resources", [bad_resource]))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                status, body = self.post_route_allocation(
                    self.route_payload(**{field: value})
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_or_non_string_locations_are_422(self) -> None:
        for bad_value in ("", "   ", 123, None, ["x"], True):
            with self.subTest(kind="edge-from", bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(edges=[{"from": bad_value, "to": "B", "cost": 1}])
                )
                self.assertEqual(status, 422)
            with self.subTest(kind="edge-to", bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(edges=[{"from": "A", "to": bad_value, "cost": 1}])
                )
                self.assertEqual(status, 422)
            with self.subTest(kind="demand-location", bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(
                        demands=[
                            {
                                "demandId": "d",
                                "location": bad_value,
                                "units": 1,
                                "priority": 0,
                            }
                        ]
                    )
                )
                self.assertEqual(status, 422)
            with self.subTest(kind="resource-location", bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(
                        resources=[
                            {"resourceId": "r", "location": bad_value, "capacity": 1}
                        ]
                    )
                )
                self.assertEqual(status, 422)

    def test_bad_cost_is_422(self) -> None:
        for bad_value in (0, -1, 1.5, "2", True, None, [2], 1.0):
            with self.subTest(bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(edges=[{"from": "A", "to": "B", "cost": bad_value}])
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_bad_units_priority_capacity_are_422(self) -> None:
        for bad_value in (0, -1, 1.5, "2", True, None, [2], 1.0):
            with self.subTest(field="units", bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(
                        demands=[
                            {
                                "demandId": "d",
                                "location": "L",
                                "units": bad_value,
                                "priority": 0,
                            }
                        ]
                    )
                )
                self.assertEqual(status, 422)
            with self.subTest(field="capacity", bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(
                        resources=[
                            {"resourceId": "r", "location": "L", "capacity": bad_value}
                        ]
                    )
                )
                self.assertEqual(status, 422)
        for bad_value in (-1, 1.5, "0", True, None, [0], 0.0):
            with self.subTest(field="priority", bad_value=bad_value):
                status, body = self.post_route_allocation(
                    self.route_payload(
                        demands=[
                            {
                                "demandId": "d",
                                "location": "L",
                                "units": 1,
                                "priority": bad_value,
                            }
                        ]
                    )
                )
                self.assertEqual(status, 422)

    def test_duplicate_identifiers_are_422(self) -> None:
        payload = self.route_payload()
        payload["demands"].append(
            {"demandId": "d-a", "location": "B", "units": 1, "priority": 0}
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

        payload = self.route_payload()
        payload["resources"].append(
            {"resourceId": "r-a", "location": "C", "capacity": 1}
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_duplicate_edge_endpoint_pair_is_422(self) -> None:
        payload = self.route_payload()
        payload["edges"].append({"from": "A", "to": "B", "cost": 7})
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_reverse_edge_is_not_a_duplicate(self) -> None:
        payload = self.route_payload()
        payload["edges"].append({"from": "B", "to": "A", "cost": 7})
        status, _ = self.post_route_allocation(payload)
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
