"""Tests for POST /decisions/dispatch.

The dispatch endpoint is a read-only, stateless planner like
/decisions/allocate, but it also routes along a directed road network: a
demand is placed wholly on a capacity-sufficient resource that can reach the
demand node, preferring the shortest total travel time and then the smallest
resourceId.
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


class DispatchTest(unittest.TestCase):
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

    def post_dispatch(
        self, payload: Any, *, raw: bool = False, content_type: str | None = "application/json"
    ) -> tuple[int, Any]:
        body = payload if raw else json.dumps(payload).encode()
        return self.request(
            "/decisions/dispatch", method="POST", body=body, content_type=content_type
        )

    def dispatch_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "demands": [
                {"demandId": "d-low", "nodeId": "n2", "units": 1, "priority": 1},
                {"demandId": "d-b", "nodeId": "n3", "units": 2, "priority": 5},
                {"demandId": "d-a", "nodeId": "n2", "units": 2, "priority": 5},
            ],
            "resources": [
                {"resourceId": "r-b", "nodeId": "n1", "capacity": 2},
                {"resourceId": "r-a", "nodeId": "n2", "capacity": 3},
            ],
            "roads": [
                {"from": "n1", "to": "n2", "travelTime": 4},
                {"from": "n1", "to": "n3", "travelTime": 10},
                {"from": "n2", "to": "n3", "travelTime": 3},
            ],
        }
        payload.update(overrides)
        return payload

    # --- planning rules ------------------------------------------------------

    def test_dispatch_prefers_shortest_travel_then_resource_id(self) -> None:
        # Processing order: d-a(5), d-b(5), d-low(1).
        # d-a at n2: r-a is on n2 (0) vs r-b via n1->n2 (4) -> r-a (3 -> 1).
        # d-b at n3: r-a n2->n3 (3) vs r-b n1->n2->n3 (7) -> r-a, but r-a has
        # only 1 left, so r-b takes it (2 -> 0). d-low at n2: r-a (1 -> 0).
        status, body = self.post_dispatch(self.dispatch_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "assignments": [
                    {
                        "demandId": "d-a",
                        "resourceId": "r-a",
                        "units": 2,
                        "travelTime": 0,
                        "path": ["n2"],
                    },
                    {
                        "demandId": "d-b",
                        "resourceId": "r-b",
                        "units": 2,
                        "travelTime": 7,
                        "path": ["n1", "n2", "n3"],
                    },
                    {
                        "demandId": "d-low",
                        "resourceId": "r-a",
                        "units": 1,
                        "travelTime": 0,
                        "path": ["n2"],
                    },
                ],
                "unassigned": [],
                "totalUnits": 5,
            },
        )

    def test_dispatch_equal_travel_ties_break_by_resource_id(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [{"demandId": "d", "nodeId": "n", "units": 1, "priority": 0}],
            "resources": [
                {"resourceId": "r-z", "nodeId": "a", "capacity": 1},
                {"resourceId": "r-a", "nodeId": "b", "capacity": 1},
            ],
            "roads": [
                {"from": "a", "to": "n", "travelTime": 5},
                {"from": "b", "to": "n", "travelTime": 5},
            ],
        }
        status, body = self.post_dispatch(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"][0]["resourceId"], "r-a")

    def test_dispatch_equal_length_routes_pick_smallest_node_sequence(self) -> None:
        # Two shortest routes from s to t, both travel time 4:
        # s->b->t (["s","b","t"]) and s->a->t (["s","a","t"]); the latter is
        # smaller by Unicode code-point comparison of the node sequences.
        payload = {
            "organizationId": "o",
            "demands": [{"demandId": "d", "nodeId": "t", "units": 1, "priority": 0}],
            "resources": [{"resourceId": "r", "nodeId": "s", "capacity": 1}],
            "roads": [
                {"from": "s", "to": "b", "travelTime": 2},
                {"from": "b", "to": "t", "travelTime": 2},
                {"from": "s", "to": "a", "travelTime": 1},
                {"from": "a", "to": "t", "travelTime": 3},
            ],
        }
        status, body = self.post_dispatch(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"][0]["travelTime"], 4)
        self.assertEqual(body["assignments"][0]["path"], ["s", "a", "t"])

    def test_dispatch_same_node_route_is_zero_length(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [{"demandId": "d", "nodeId": "n", "units": 2, "priority": 0}],
            "resources": [{"resourceId": "r", "nodeId": "n", "capacity": 5}],
            "roads": [{"from": "n", "to": "m", "travelTime": 1}],
        }
        status, body = self.post_dispatch(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["assignments"],
            [
                {
                    "demandId": "d",
                    "resourceId": "r",
                    "units": 2,
                    "travelTime": 0,
                    "path": ["n"],
                }
            ],
        )
        self.assertEqual(body["totalUnits"], 2)

    def test_dispatch_roads_are_directed(self) -> None:
        # n2 -> n1 exists but n1 -> n2 does not, so the resource on n1 cannot
        # reach the demand on n2.
        payload = {
            "organizationId": "o",
            "demands": [{"demandId": "d", "nodeId": "n2", "units": 1, "priority": 0}],
            "resources": [{"resourceId": "r", "nodeId": "n1", "capacity": 1}],
            "roads": [{"from": "n2", "to": "n1", "travelTime": 1}],
        }
        status, body = self.post_dispatch(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["assignments"], [])
        self.assertEqual(body["unassigned"], ["d"])
        self.assertEqual(body["totalUnits"], 0)

    def test_dispatch_unreachable_or_insufficient_capacity_unassigned(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [
                {"demandId": "d-fit", "nodeId": "n", "units": 2, "priority": 0},
                {"demandId": "d-big", "nodeId": "n", "units": 9, "priority": 0},
                {"demandId": "d-lost", "nodeId": "elsewhere", "units": 1, "priority": 0},
            ],
            "resources": [{"resourceId": "r", "nodeId": "n", "capacity": 2}],
            "roads": [],
        }
        status, body = self.post_dispatch(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["assignments"],
            [
                {
                    "demandId": "d-fit",
                    "resourceId": "r",
                    "units": 2,
                    "travelTime": 0,
                    "path": ["n"],
                }
            ],
        )
        self.assertEqual(body["unassigned"], ["d-big", "d-lost"])
        self.assertEqual(body["totalUnits"], 2)

    def test_dispatch_capacity_deducted_immediately_no_split_no_oversell(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [
                {"demandId": "d1", "nodeId": "n", "units": 3, "priority": 0},
                {"demandId": "d2", "nodeId": "n", "units": 3, "priority": 0},
                {"demandId": "d3", "nodeId": "n", "units": 5, "priority": 0},
            ],
            "resources": [
                {"resourceId": "r1", "nodeId": "n", "capacity": 3},
                {"resourceId": "r2", "nodeId": "n", "capacity": 3},
            ],
            "roads": [],
        }
        status, body = self.post_dispatch(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            [(a["demandId"], a["resourceId"]) for a in body["assignments"]],
            [("d1", "r1"), ("d2", "r2")],
        )
        self.assertEqual(body["unassigned"], ["d3"])
        self.assertEqual(body["totalUnits"], 6)

    def test_dispatch_priority_then_unicode_demand_id_order(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [
                {"demandId": "b", "nodeId": "n", "units": 1, "priority": 0},
                {"demandId": "ä", "nodeId": "n", "units": 1, "priority": 0},
                {"demandId": "a", "nodeId": "n", "units": 1, "priority": 1},
            ],
            "resources": [{"resourceId": "r", "nodeId": "n", "capacity": 3}],
            "roads": [],
        }
        status, body = self.post_dispatch(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            [a["demandId"] for a in body["assignments"]], ["a", "b", "ä"]
        )

    def test_dispatch_response_has_exactly_fixed_fields(self) -> None:
        status, body = self.post_dispatch(self.dispatch_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            set(body), {"organizationId", "assignments", "unassigned", "totalUnits"}
        )
        for assignment in body["assignments"]:
            self.assertEqual(
                set(assignment),
                {"demandId", "resourceId", "units", "travelTime", "path"},
            )

    def test_dispatch_empty_arrays_succeed(self) -> None:
        cases = [
            {"organizationId": "o", "demands": [], "resources": [], "roads": []},
            {
                "organizationId": "o",
                "demands": [],
                "resources": [{"resourceId": "r", "nodeId": "n", "capacity": 5}],
                "roads": [],
            },
            {
                "organizationId": "o",
                "demands": [{"demandId": "d", "nodeId": "n", "units": 1, "priority": 0}],
                "resources": [],
                "roads": [],
            },
        ]
        expected_unassigned = [[], [], ["d"]]
        for payload, unassigned in zip(cases, expected_unassigned):
            with self.subTest(payload=payload):
                status, body = self.post_dispatch(payload)
                self.assertEqual(status, 200)
                self.assertEqual(body["organizationId"], "o")
                self.assertEqual(body["assignments"], [])
                self.assertEqual(body["unassigned"], unassigned)
                self.assertEqual(body["totalUnits"], 0)

    def test_dispatch_deterministic_independent_of_input_order(self) -> None:
        first_payload = self.dispatch_payload()
        second_payload = self.dispatch_payload()
        second_payload["demands"] = list(reversed(second_payload["demands"]))
        second_payload["resources"] = list(reversed(second_payload["resources"]))
        second_payload["roads"] = list(reversed(second_payload["roads"]))
        first = self.post_dispatch(first_payload)
        second = self.post_dispatch(second_payload)
        self.assertEqual(first, second)

    def test_dispatch_repeated_requests_return_identical_json(self) -> None:
        payload = self.dispatch_payload()
        results = [self.post_dispatch(payload) for _ in range(5)]
        self.assertTrue(all(status == 200 for status, _ in results))
        encoded = {json.dumps(body, sort_keys=True) for _, body in results}
        self.assertEqual(len(encoded), 1)

    # --- validation ----------------------------------------------------------

    def test_dispatch_validation_errors_are_422(self) -> None:
        valid = self.dispatch_payload()
        cases: list[Any] = [
            "not-an-object",
            {k: v for k, v in valid.items() if k != "roads"},  # missing field
            {**valid, "extra": 1},  # unexpected field
            {**valid, "organizationId": "  "},  # blank organization
            {**valid, "demands": "nope"},  # demands not an array
            {**valid, "roads": {}},  # roads not an array
            {**valid, "demands": ["nope"]},  # demand not an object
            {**valid, "resources": ["nope"]},  # resource not an object
            {**valid, "roads": ["nope"]},  # road not an object
            # demand element field/type violations
            {**valid, "demands": [{"demandId": "d", "nodeId": "n", "units": 1}]},
            {**valid, "demands": [
                {"demandId": "d", "nodeId": "n", "units": 1, "priority": 0, "x": 1}
            ]},
            {**valid, "demands": [
                {"demandId": "", "nodeId": "n", "units": 1, "priority": 0}
            ]},
            {**valid, "demands": [
                {"demandId": "d", "nodeId": " ", "units": 1, "priority": 0}
            ]},
            {**valid, "demands": [
                {"demandId": "d", "nodeId": "n", "units": 0, "priority": 0}
            ]},
            {**valid, "demands": [
                {"demandId": "d", "nodeId": "n", "units": 1.5, "priority": 0}
            ]},
            {**valid, "demands": [
                {"demandId": "d", "nodeId": "n", "units": True, "priority": 0}
            ]},
            {**valid, "demands": [
                {"demandId": "d", "nodeId": "n", "units": 1, "priority": -1}
            ]},
            {**valid, "demands": [
                {"demandId": "d", "nodeId": "n", "units": 1, "priority": 0},
                {"demandId": "d", "nodeId": "n", "units": 1, "priority": 0},
            ]},
            # resource element field/type violations
            {**valid, "resources": [{"resourceId": "r", "nodeId": "n"}]},
            {**valid, "resources": [
                {"resourceId": "r", "nodeId": "n", "capacity": 1, "x": 1}
            ]},
            {**valid, "resources": [
                {"resourceId": "", "nodeId": "n", "capacity": 1}
            ]},
            {**valid, "resources": [
                {"resourceId": "r", "nodeId": "", "capacity": 1}
            ]},
            {**valid, "resources": [
                {"resourceId": "r", "nodeId": "n", "capacity": 0}
            ]},
            {**valid, "resources": [
                {"resourceId": "r", "nodeId": "n", "capacity": 1},
                {"resourceId": "r", "nodeId": "n", "capacity": 1},
            ]},
            # road element field/type violations
            {**valid, "roads": [{"from": "a", "to": "b"}]},
            {**valid, "roads": [
                {"from": "a", "to": "b", "travelTime": 1, "x": 1}
            ]},
            {**valid, "roads": [{"from": "", "to": "b", "travelTime": 1}]},
            {**valid, "roads": [{"from": "a", "to": "", "travelTime": 1}]},
            {**valid, "roads": [{"from": "a", "to": "b", "travelTime": 0}]},
            {**valid, "roads": [{"from": "a", "to": "b", "travelTime": -2}]},
            {**valid, "roads": [{"from": "a", "to": "b", "travelTime": 1.5}]},
            {**valid, "roads": [{"from": "a", "to": "b", "travelTime": False}]},
            # duplicate ordered endpoints (reversed pair is a different road)
            {**valid, "roads": [
                {"from": "a", "to": "b", "travelTime": 1},
                {"from": "a", "to": "b", "travelTime": 2},
            ]},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                status, body = self.post_dispatch(payload)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_dispatch_reversed_road_pair_is_not_a_duplicate(self) -> None:
        payload = {
            "organizationId": "o",
            "demands": [],
            "resources": [],
            "roads": [
                {"from": "a", "to": "b", "travelTime": 1},
                {"from": "b", "to": "a", "travelTime": 1},
            ],
        }
        status, _ = self.post_dispatch(payload)
        self.assertEqual(status, 200)

    def test_dispatch_duplicate_json_keys_are_422(self) -> None:
        raw = (
            b'{"organizationId": "org-1", "organizationId": "org-1",'
            b' "demands": [], "resources": [], "roads": []}'
        )
        status, body = self.post_dispatch(raw, raw=True)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_dispatch_media_type_and_json_errors(self) -> None:
        status, body = self.post_dispatch(
            self.dispatch_payload(), content_type="text/plain"
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.post_dispatch(b'{"organizationId": ', raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    # --- credentials ---------------------------------------------------------

    def test_dispatch_requires_authentication(self) -> None:
        payload = json.dumps(self.dispatch_payload()).encode()
        request = Request(
            f"{self.base_url}/decisions/dispatch",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5):
                self.fail("expected 401")
        except HTTPError as error:
            self.assertEqual(error.code, 401)
            self.assertEqual(json.load(error)["error"], "unauthorized")
            error.close()

    def test_dispatch_read_token_allowed(self) -> None:
        _support.ensure_token(self.token_cache, self.base_url, "org-1")
        payload = json.dumps(
            {"token": "read-dispatch", "organizationId": "org-1", "role": "read"}
        ).encode()
        request = Request(
            f"{self.base_url}/auth/tokens",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            self.assertIn(response.status, (200, 201))

        body = json.dumps(self.dispatch_payload()).encode()
        request = Request(
            f"{self.base_url}/decisions/dispatch",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer read-dispatch",
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)

    def test_dispatch_cross_organization_is_403(self) -> None:
        _support.ensure_token(self.token_cache, self.base_url, "org-1")
        payload = json.dumps(self.dispatch_payload(organizationId="org-2")).encode()
        request = Request(
            f"{self.base_url}/decisions/dispatch",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token_cache['org-1']}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5):
                self.fail("expected 403")
        except HTTPError as error:
            self.assertEqual(error.code, 403)
            self.assertEqual(json.load(error)["error"], "forbidden")
            error.close()

    def test_dispatch_does_not_change_service_state(self) -> None:
        payload = self.dispatch_payload()
        status, _ = self.post_dispatch(payload)
        self.assertEqual(status, 200)
        # No events, reservations, snapshots, or branches are created.
        status, events = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(events["events"], [])
        status, reservations = self.request("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])


if __name__ == "__main__":
    unittest.main()
