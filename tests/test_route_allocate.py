"""Tests for POST /decisions/route-allocate.

The endpoint is a read-only, stateless planner: it routes whole demands from
resource locations to demand destinations over a directed weighted graph
supplied in the request body. These tests cover the deterministic planning
rules, the fixed response shape, the error contract, and the guarantee that
no server-side state is touched.
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
        self,
        payload: Any,
        *,
        raw: bool = False,
        content_type: str | None = "application/json",
        auth: bool = True,
    ) -> tuple[int, Any]:
        body = payload if raw else json.dumps(payload).encode()
        return self.request(
            "/decisions/route-allocate",
            method="POST",
            body=body,
            content_type=content_type,
            auth=auth,
        )

    def register_token(self, token: str, organization_id: str, role: str) -> None:
        payload = json.dumps(
            {"token": token, "organizationId": organization_id, "role": role}
        ).encode()
        request = Request(
            f"{self.base_url}/auth/tokens",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            self.assertIn(response.status, (200, 201))

    def route_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "edges": [
                {"from": "A", "to": "B", "cost": 2},
                {"from": "B", "to": "C", "cost": 2},
                {"from": "A", "to": "C", "cost": 10},
                {"from": "C", "to": "D", "cost": 1},
                {"from": "B", "to": "D", "cost": 5},
            ],
            "demands": [
                {"demandId": "d-1", "location": "D", "units": 2, "priority": 1},
            ],
            "resources": [
                {"resourceId": "r-1", "location": "A", "capacity": 3},
                {"resourceId": "r-2", "location": "C", "capacity": 2},
            ],
        }
        payload.update(overrides)
        return payload

    # --- planning rules ------------------------------------------------------

    def test_assigns_lowest_cost_reachable_resource(self) -> None:
        # r-1 routes A->B->C->D at cost 5 (A->C->D is 11, A->B->D is 7);
        # r-2 routes C->D at cost 1 and wins.
        status, body = self.post_route_allocation(self.route_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "assignments": [
                    {
                        "demandId": "d-1",
                        "resourceId": "r-2",
                        "units": 2,
                        "path": ["C", "D"],
                        "travelCost": 1,
                    }
                ],
                "unassigned": [],
                "totalUnits": 2,
                "totalTravelCost": 1,
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

    def test_equal_cost_tie_breaks_by_resource_id(self) -> None:
        payload = self.route_payload(
            edges=[{"from": "A", "to": "D", "cost": 4}],
            demands=[
                {"demandId": "d-1", "location": "D", "units": 1, "priority": 0}
            ],
            resources=[
                {"resourceId": "r-b", "location": "A", "capacity": 1},
                {"resourceId": "r-a", "location": "A", "capacity": 1},
            ],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"][0]["resourceId"], "r-a")

    def test_equal_cost_paths_take_smallest_node_sequence(self) -> None:
        # A->B->D and A->C->D both cost 2; B < C in code-point order.
        payload = self.route_payload(
            edges=[
                {"from": "A", "to": "C", "cost": 1},
                {"from": "A", "to": "B", "cost": 1},
                {"from": "C", "to": "D", "cost": 1},
                {"from": "B", "to": "D", "cost": 1},
            ],
            demands=[
                {"demandId": "d-1", "location": "D", "units": 1, "priority": 0}
            ],
            resources=[{"resourceId": "r-1", "location": "A", "capacity": 1}],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"][0]["path"], ["A", "B", "D"])
        self.assertEqual(body["assignments"][0]["travelCost"], 2)

    def test_same_location_routes_through_itself_at_zero_cost(self) -> None:
        payload = self.route_payload(
            edges=[],
            demands=[
                {"demandId": "d-1", "location": "X", "units": 2, "priority": 0}
            ],
            resources=[{"resourceId": "r-1", "location": "X", "capacity": 5}],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["assignments"],
            [
                {
                    "demandId": "d-1",
                    "resourceId": "r-1",
                    "units": 2,
                    "path": ["X"],
                    "travelCost": 0,
                }
            ],
        )
        self.assertEqual(body["totalTravelCost"], 0)

    def test_directed_edges_are_not_traversed_backwards(self) -> None:
        payload = self.route_payload(
            edges=[{"from": "D", "to": "A", "cost": 1}],
            demands=[
                {"demandId": "d-1", "location": "D", "units": 1, "priority": 0}
            ],
            resources=[{"resourceId": "r-1", "location": "A", "capacity": 1}],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"], [])
        self.assertEqual(
            body["unassigned"], [{"demandId": "d-1", "reason": "unreachable"}]
        )

    def test_priority_order_and_capacity_deduction(self) -> None:
        # Processing order: d-big(9), d-a(5), d-b(5), d-low(1).
        payload = self.route_payload(
            edges=[{"from": "A", "to": "L", "cost": 3}],
            demands=[
                {"demandId": "d-low", "location": "L", "units": 3, "priority": 1},
                {"demandId": "d-b", "location": "L", "units": 4, "priority": 5},
                {"demandId": "d-a", "location": "L", "units": 2, "priority": 5},
                {"demandId": "d-big", "location": "L", "units": 10, "priority": 9},
            ],
            resources=[
                {"resourceId": "r-b", "location": "A", "capacity": 6},
                {"resourceId": "r-a", "location": "A", "capacity": 5},
            ],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["assignments"],
            [
                {
                    "demandId": "d-a",
                    "resourceId": "r-a",
                    "units": 2,
                    "path": ["A", "L"],
                    "travelCost": 3,
                },
                {
                    "demandId": "d-b",
                    "resourceId": "r-b",
                    "units": 4,
                    "path": ["A", "L"],
                    "travelCost": 3,
                },
                {
                    "demandId": "d-low",
                    "resourceId": "r-a",
                    "units": 3,
                    "path": ["A", "L"],
                    "travelCost": 3,
                },
            ],
        )
        self.assertEqual(
            body["unassigned"],
            [{"demandId": "d-big", "reason": "insufficient_capacity"}],
        )
        self.assertEqual(body["totalUnits"], 9)
        self.assertEqual(body["totalTravelCost"], 9)

    def test_unassigned_follow_processing_order(self) -> None:
        payload = self.route_payload(
            edges=[],
            demands=[
                {"demandId": "zeta", "location": "L", "units": 9, "priority": 10},
                {"demandId": "alpha", "location": "L", "units": 9, "priority": 10},
                {"demandId": "mid", "location": "L", "units": 9, "priority": 5},
            ],
            resources=[{"resourceId": "r", "location": "L", "capacity": 1}],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["unassigned"],
            [
                {"demandId": "alpha", "reason": "insufficient_capacity"},
                {"demandId": "zeta", "reason": "insufficient_capacity"},
                {"demandId": "mid", "reason": "insufficient_capacity"},
            ],
        )

    def test_unreachable_only_when_capacity_would_suffice(self) -> None:
        # d-too-big fits no resource (insufficient even though also
        # unreachable); d-fit would fit r-small/r-mid but no route exists.
        payload = self.route_payload(
            edges=[],
            demands=[
                {"demandId": "d-too-big", "location": "Z", "units": 5, "priority": 2},
                {"demandId": "d-fit", "location": "Z", "units": 2, "priority": 1},
            ],
            resources=[
                {"resourceId": "r-small", "location": "A", "capacity": 3},
                {"resourceId": "r-mid", "location": "B", "capacity": 4},
            ],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["unassigned"],
            [
                {"demandId": "d-too-big", "reason": "insufficient_capacity"},
                {"demandId": "d-fit", "reason": "unreachable"},
            ],
        )
        self.assertEqual(body["totalUnits"], 0)
        self.assertEqual(body["totalTravelCost"], 0)

    def test_capacity_consumed_earlier_changes_later_reason(self) -> None:
        # d-first drains r-1; d-second then has no capacity-sufficient
        # resource at all, so it reports insufficient_capacity.
        payload = self.route_payload(
            edges=[{"from": "A", "to": "L", "cost": 1}],
            demands=[
                {"demandId": "d-first", "location": "L", "units": 3, "priority": 2},
                {"demandId": "d-second", "location": "L", "units": 2, "priority": 1},
            ],
            resources=[{"resourceId": "r-1", "location": "A", "capacity": 3}],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(len(body["assignments"]), 1)
        self.assertEqual(
            body["unassigned"],
            [{"demandId": "d-second", "reason": "insufficient_capacity"}],
        )

    def test_empty_arrays_are_valid(self) -> None:
        cases = [
            {"organizationId": "o", "edges": [], "demands": [], "resources": []},
            {
                "organizationId": "o",
                "edges": [{"from": "A", "to": "B", "cost": 1}],
                "demands": [],
                "resources": [{"resourceId": "r", "location": "A", "capacity": 5}],
            },
            {
                "organizationId": "o",
                "edges": [],
                "demands": [
                    {"demandId": "d", "location": "L", "units": 1, "priority": 0}
                ],
                "resources": [],
            },
        ]
        expected_unassigned = [[], [], [{"demandId": "d", "reason": "insufficient_capacity"}]]
        for payload, unassigned in zip(cases, expected_unassigned):
            with self.subTest(payload=payload):
                status, body = self.post_route_allocation(payload)
                self.assertEqual(status, 200)
                self.assertEqual(body["organizationId"], "o")
                self.assertEqual(body["assignments"], [])
                self.assertEqual(body["unassigned"], unassigned)
                self.assertEqual(body["totalUnits"], 0)
                self.assertEqual(body["totalTravelCost"], 0)

    def test_zero_priority_accepted(self) -> None:
        payload = self.route_payload(
            edges=[],
            demands=[
                {"demandId": "d", "location": "L", "units": 1, "priority": 0}
            ],
            resources=[{"resourceId": "r", "location": "L", "capacity": 1}],
        )
        status, body = self.post_route_allocation(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["totalUnits"], 1)

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
            "occurredAt": 100,
            "payload": {},
        }
        status, _ = self.request(
            "/events", method="POST", body=json.dumps(event).encode()
        )
        self.assertEqual(status, 201)
        self.post_route_allocation(self.route_payload())
        self.post_route_allocation(self.route_payload())
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)
        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(body["reservations"], [])

    # --- authentication and authorization ------------------------------------

    def test_missing_authorization_is_401(self) -> None:
        status, body = self.post_route_allocation(self.route_payload(), auth=False)
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_unregistered_token_is_401(self) -> None:
        body = json.dumps(self.route_payload()).encode()
        request = Request(
            f"{self.base_url}/decisions/route-allocate",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer forged",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 401)
        raised.exception.close()

    def test_read_role_may_call(self) -> None:
        self.register_token("reader-1", "org-1", "read")
        body = json.dumps(self.route_payload()).encode()
        request = Request(
            f"{self.base_url}/decisions/route-allocate",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer reader-1",
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)

    def test_cross_organization_is_403(self) -> None:
        self.register_token("other-org", "org-2", "write")
        body = json.dumps(self.route_payload()).encode()
        request = Request(
            f"{self.base_url}/decisions/route-allocate",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer other-org",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request, timeout=5)
        try:
            self.assertEqual(raised.exception.code, 403)
            self.assertEqual(json.load(raised.exception)["error"], "forbidden")
        finally:
            raised.exception.close()

    # --- media type and JSON syntax ------------------------------------------

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

    # --- validation ----------------------------------------------------------

    def assert_validation_error(self, payload: Any, *, raw: bool = False) -> None:
        status, body = self.post_route_allocation(payload, raw=raw)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_non_object_body_is_422(self) -> None:
        for payload in ([], "text", 3, None):
            with self.subTest(payload=payload):
                self.assert_validation_error(payload)

    def test_missing_and_unknown_top_level_fields_are_422(self) -> None:
        payload = self.route_payload()
        del payload["edges"]
        self.assert_validation_error(payload)
        self.assert_validation_error(self.route_payload(extra=1))

    def test_duplicate_json_fields_are_422(self) -> None:
        raw = (
            b'{"organizationId":"org-1","organizationId":"org-1",'
            b'"edges":[],"demands":[],"resources":[]}'
        )
        self.assert_validation_error(raw, raw=True)

    def test_non_array_sections_are_422(self) -> None:
        self.assert_validation_error(self.route_payload(edges={}))
        self.assert_validation_error(self.route_payload(demands="x"))
        self.assert_validation_error(self.route_payload(resources=None))

    def test_edge_validation(self) -> None:
        valid_edge = {"from": "A", "to": "B", "cost": 1}
        bad_edges = [
            {},  # missing everything
            {"from": "A", "to": "B", "cost": 1, "x": 1},  # unknown field
            {"from": "", "to": "B", "cost": 1},  # blank from
            {"from": "  ", "to": "B", "cost": 1},  # whitespace-only from
            {"from": "A", "to": "", "cost": 1},  # blank to
            {"from": "A", "to": "B", "cost": 0},  # zero cost
            {"from": "A", "to": "B", "cost": -1},  # negative cost
            {"from": "A", "to": "B", "cost": 1.5},  # float cost
            {"from": "A", "to": "B", "cost": True},  # boolean cost
            {"from": 1, "to": "B", "cost": 1},  # non-string from
        ]
        for edge in bad_edges:
            with self.subTest(edge=edge):
                self.assert_validation_error(self.route_payload(edges=[edge]))
        # A duplicate directed endpoint pair is invalid even in reverse input
        # order; the reversed pair (B->A) is a different edge and is legal.
        self.assert_validation_error(
            self.route_payload(edges=[valid_edge, dict(valid_edge)])
        )
        status, _ = self.post_route_allocation(
            self.route_payload(
                edges=[valid_edge, {"from": "B", "to": "A", "cost": 1}]
            )
        )
        self.assertEqual(status, 200)

    def test_demand_validation(self) -> None:
        valid = {"demandId": "d", "location": "L", "units": 1, "priority": 0}
        bad_demands = [
            {"demandId": "d", "units": 1, "priority": 0},  # missing location
            dict(valid, extra=1),  # unknown field
            dict(valid, demandId=""),  # blank id
            dict(valid, location=""),  # blank location
            dict(valid, units=0),  # zero units
            dict(valid, units=-1),  # negative units
            dict(valid, units=1.5),  # float units
            dict(valid, units=True),  # boolean units
            dict(valid, priority=-1),  # negative priority
            dict(valid, priority=1.5),  # float priority
        ]
        for demand in bad_demands:
            with self.subTest(demand=demand):
                self.assert_validation_error(self.route_payload(demands=[demand]))
        self.assert_validation_error(
            self.route_payload(demands=[valid, dict(valid)])
        )

    def test_resource_validation(self) -> None:
        valid = {"resourceId": "r", "location": "L", "capacity": 1}
        bad_resources = [
            {"resourceId": "r", "capacity": 1},  # missing location
            dict(valid, extra=1),  # unknown field
            dict(valid, resourceId=""),  # blank id
            dict(valid, location=""),  # blank location
            dict(valid, capacity=0),  # zero capacity
            dict(valid, capacity=-1),  # negative capacity
            dict(valid, capacity=1.5),  # float capacity
            dict(valid, capacity=False),  # boolean capacity
        ]
        for resource in bad_resources:
            with self.subTest(resource=resource):
                self.assert_validation_error(
                    self.route_payload(resources=[resource])
                )
        self.assert_validation_error(
            self.route_payload(resources=[valid, dict(valid)])
        )

    def test_failed_requests_change_no_state(self) -> None:
        self.post_route_allocation(self.route_payload(edges=[{"from": "A"}]))
        self.post_route_allocation(b"{broken", raw=True)
        self.post_route_allocation(self.route_payload(), auth=False)
        status, body = self.post_route_allocation(self.route_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["totalUnits"], 2)


if __name__ == "__main__":
    unittest.main()
