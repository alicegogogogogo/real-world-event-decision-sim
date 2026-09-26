"""Regression tests for GET /snapshots/{snapshotId}/resources.

The snapshot resource comparisons already aligned two snapshots' balances;
this locks down the new read-only entry point that lists one snapshot's
captured resource balances for the caller's organization:

- the ``resources`` array carries one row per resource the organization
  had reserved when the snapshot was captured, code-point sorted, each
  with the ``capacity``, ``occupied`` and ``remaining`` balances;
  ``occupied`` sums every captured reservation against the resource and
  ``remaining`` is ``capacity - occupied``;
- the response echoes ``organizationId`` and ``snapshotId``, is compact,
  key-sorted JSON with integer balances and one trailing newline, and
  identical requests are byte-for-byte identical; a snapshot with no
  resources yields an empty row array;
- the balances are the captured point in time and never track later
  main-service writes; they share the comparison's exact contract, so a
  snapshot compared with itself via POST /snapshots/compare/resources
  reports every row equal, and other organizations' reservations never
  contribute;
- the verdict order is fixed — 401 (credential) before 422 (query shape)
  before 403 (organization, then foreign snapshot) before 404
  (snapshot_not_found) — and nothing is ever written or implicitly
  created, including the snapshot's own content and the main service.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"


def reservation_body(
    reservation_id: str,
    *,
    resource_id: str = "res-rc",
    quantity: int = 1,
    capacity: int = 100,
    organization_id: str = ORG1,
) -> dict[str, Any]:
    return {
        "organizationId": organization_id,
        "reservationId": reservation_id,
        "resourceId": resource_id,
        "quantity": quantity,
        "capacity": capacity,
    }


class SnapshotResourceListTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.register("w1", ORG1, "write")
        self.register("w2", ORG2, "write")
        self.register("r1", ORG1, "read")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    # -------------------------------------------------------------- low level

    def raw(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        token: str | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, bytes]:
        headers: dict[str, str] = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except HTTPError as error:
            try:
                return error.code, error.read()
            finally:
                error.close()

    def call(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Any = None,
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        status, raw = self.raw(path, method=method, body=body, token=token)
        return status, json.loads(raw)

    def register(self, token: str, organization_id: str, role: str) -> None:
        status, _ = self.call(
            "/auth/tokens",
            method="POST",
            token=None,
            payload={
                "token": token,
                "organizationId": organization_id,
                "role": role,
            },
        )
        self.assertIn(status, (200, 201))

    # ------------------------------------------------------------- scaffolding

    def add_reservation(
        self,
        reservation_id: str,
        token: str = "w1",
        **kwargs: Any,
    ) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        status, _ = self.call(
            "/reservations",
            method="POST",
            token=token,
            payload=reservation_body(
                reservation_id, organization_id=organization_id, **kwargs
            ),
        )
        self.assertEqual(status, 201)

    def capture_snapshot(
        self, snapshot_id: str = "s1", *, token: str = "w1"
    ) -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def list_resources(
        self, snapshot: str = "s1", *, token: str | None = "w1", org: str = ORG1
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/resources?organizationId={org}", token=token
        )

    # ------------------------------------------------------------- happy paths

    def test_rows_echo_identifiers_and_report_three_balances(self) -> None:
        self.add_reservation(
            "res-1", resource_id="res-a", quantity=3, capacity=10
        )
        self.add_reservation(
            "res-2", resource_id="res-b", quantity=1, capacity=4
        )
        self.capture_snapshot()

        status, body = self.list_resources()
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "res-a",
                    "capacity": 10,
                    "occupied": 3,
                    "remaining": 7,
                },
                {
                    "resourceId": "res-b",
                    "capacity": 4,
                    "occupied": 1,
                    "remaining": 3,
                },
            ],
        )

    def test_occupied_sums_every_reservation_against_the_resource(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.add_reservation("res-2", resource_id="pool", quantity=3, capacity=10)
        self.add_reservation("res-3", resource_id="pool", quantity=1, capacity=10)
        self.capture_snapshot()

        status, body = self.list_resources()
        self.assertEqual(status, 200)
        (row,) = body["resources"]
        self.assertEqual(
            row,
            {
                "resourceId": "pool",
                "capacity": 10,
                "occupied": 6,
                "remaining": 4,
            },
        )

    def test_rows_sort_by_resource_id_in_code_point_order(self) -> None:
        for resource_id in ("res-b", "res-A", "res-a", "res-1"):
            self.add_reservation(
                f"book-{resource_id}",
                resource_id=resource_id,
                quantity=1,
                capacity=7,
            )
        self.capture_snapshot()

        status, body = self.list_resources()
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]],
            ["res-1", "res-A", "res-a", "res-b"],
        )

    def test_snapshot_without_resources_returns_empty_array(self) -> None:
        self.capture_snapshot()
        status, body = self.list_resources()
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {"organizationId": ORG1, "snapshotId": "s1", "resources": []},
        )

    def test_balances_are_the_captured_point_in_time(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture_snapshot()
        # Later main-service commits (including to the same resource) never
        # reach the immutable snapshot's balances.
        self.add_reservation("res-2", resource_id="pool", quantity=5, capacity=10)
        self.add_reservation("res-3", resource_id="other", quantity=1, capacity=3)

        _, body = self.list_resources()
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "pool",
                    "capacity": 10,
                    "occupied": 2,
                    "remaining": 8,
                }
            ],
        )

        # A fresh snapshot captures the later point in time.
        self.capture_snapshot("s2")
        _, body = self.list_resources("s2")
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "other",
                    "capacity": 3,
                    "occupied": 1,
                    "remaining": 2,
                },
                {
                    "resourceId": "pool",
                    "capacity": 10,
                    "occupied": 7,
                    "remaining": 3,
                },
            ],
        )

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_reservation("res-1", resource_id="res-1", quantity=2, capacity=9)
        self.capture_snapshot()

        status, raw = self.raw(
            "/snapshots/s1/resources?organizationId=org-1", token="w1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Balances stay integers.
        self.assertIn(b'"capacity":9', raw)
        self.assertIn(b'"occupied":2', raw)
        self.assertIn(b'"remaining":7', raw)
        # Keys are sorted by code point at every level.
        body = json.loads(raw)
        self.assertEqual(list(body), ["organizationId", "resources", "snapshotId"])
        self.assertEqual(
            list(body["resources"][0]),
            ["capacity", "occupied", "remaining", "resourceId"],
        )

    def test_repeated_requests_are_byte_identical(self) -> None:
        self.add_reservation("res-1", resource_id="res-1", quantity=2, capacity=10)
        self.add_reservation("res-2", resource_id="res-2", quantity=1, capacity=4)
        self.capture_snapshot()
        raws = [
            self.raw("/snapshots/s1/resources?organizationId=org-1", token="w1")[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    def test_two_reads_do_not_pollute_each_other(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture_snapshot("s1")
        self.add_reservation("res-2", resource_id="pool", quantity=3, capacity=10)
        self.capture_snapshot("s2")

        first_a = self.raw(
            "/snapshots/s1/resources?organizationId=org-1", token="r1"
        )[1]
        first_b = self.raw(
            "/snapshots/s2/resources?organizationId=org-1", token="r1"
        )[1]
        # Interleaving the two reads in both directions keeps each response
        # byte-stable; neither read shares or mutates the other's view.
        for _ in range(3):
            self.assertEqual(
                self.raw(
                    "/snapshots/s2/resources?organizationId=org-1", token="r1"
                )[1],
                first_b,
            )
            self.assertEqual(
                self.raw(
                    "/snapshots/s1/resources?organizationId=org-1", token="r1"
                )[1],
                first_a,
            )

    # ------------------------------------------- consistency with comparison

    def test_self_comparison_agrees_row_by_row(self) -> None:
        self.add_reservation("res-1", resource_id="res-a", quantity=3, capacity=10)
        self.add_reservation("res-2", resource_id="res-a", quantity=2, capacity=10)
        self.add_reservation("res-3", resource_id="res-b", quantity=1, capacity=4)
        self.capture_snapshot()

        status, listing = self.list_resources()
        self.assertEqual(status, 200)
        status, comparison = self.call(
            "/snapshots/compare/resources",
            method="POST",
            payload={"organizationId": ORG1, "left": "s1", "right": "s1"},
        )
        self.assertEqual(status, 200)

        # Every listed resource is present on both sides of the self
        # comparison with identical balances and a true equal marker.
        self.assertEqual(comparison["same"], ["res-a", "res-b"])
        self.assertEqual(comparison["sameCount"], 2)
        self.assertEqual(comparison["leftOnly"], [])
        self.assertEqual(comparison["rightOnly"], [])
        self.assertEqual(comparison["diff"], [])
        self.assertEqual(comparison["diffCount"], 0)
        listed = {row["resourceId"]: row for row in listing["resources"]}
        self.assertEqual(len(comparison["resources"]), len(listed))
        for row in comparison["resources"]:
            resource_id = row["resourceId"]
            self.assertTrue(row["equal"])
            self.assertEqual(row["left"], row["right"])
            self.assertEqual(
                row["left"],
                {
                    "capacity": listed[resource_id]["capacity"],
                    "occupied": listed[resource_id]["occupied"],
                    "remaining": listed[resource_id]["remaining"],
                },
            )

    def test_other_organizations_never_contribute(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture_snapshot("s1")
        # ORG2 reserves the same resourceId on the main service and captures
        # its own snapshot; neither leaks into ORG1's snapshot listing.
        self.add_reservation(
            "res-2", token="w2", resource_id="pool", quantity=5, capacity=10
        )
        self.capture_snapshot("s2", token="w2")

        status, body = self.list_resources("s1")
        self.assertEqual(status, 200)
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "pool",
                    "capacity": 10,
                    "occupied": 2,
                    "remaining": 8,
                }
            ],
        )

        # ORG2's own snapshot listing sees only ORG2 balances.
        status, body = self.list_resources("s2", token="w2", org=ORG2)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG2)
        self.assertEqual(body["snapshotId"], "s2")
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "pool",
                    "capacity": 10,
                    "occupied": 5,
                    "remaining": 5,
                }
            ],
        )

    # -------------------------------------------------------------- read-only

    def test_listing_is_read_only(self) -> None:
        self.add_reservation("res-1", resource_id="res-1", quantity=1, capacity=5)
        self.capture_snapshot()

        before = self.call("/snapshots")[1]
        for _ in range(3):
            status, _ = self.list_resources()
            self.assertEqual(status, 200)
        after = self.call("/snapshots")[1]
        self.assertEqual(before, after)

        # Main-service state and alerts are untouched as well.
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(reservations["reservations"]), 1)
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    def test_concurrent_reads_stay_consistent_under_writes(self) -> None:
        self.add_reservation(
            "res-0", resource_id="pool", quantity=1, capacity=40
        )
        self.capture_snapshot("s1")
        expected = [
            {
                "resourceId": "pool",
                "capacity": 40,
                "occupied": 1,
                "remaining": 39,
            }
        ]
        errors: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                status, body = self.list_resources()
                if status != 200:
                    errors.append(f"status {status}")
                    return
                if body["resources"] != expected:
                    errors.append(f"snapshot drifted: {body['resources']}")
                    return

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for thread in readers:
            thread.start()
        # Main-service writes and newer snapshots must not perturb the read.
        for index in range(1, 8):
            self.add_reservation(
                f"res-{index}",
                resource_id="pool",
                quantity=1,
                capacity=40,
            )
            if index % 3 == 0:
                self.capture_snapshot(f"s-live-{index}")
        stop.set()
        for thread in readers:
            thread.join(timeout=5)
        self.assertEqual(errors, [])

        # The original snapshot still reports the captured balances.
        _, body = self.list_resources()
        self.assertEqual(body["resources"], expected)

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_list(self) -> None:
        self.add_reservation("res-1", resource_id="res-1", quantity=1, capacity=5)
        self.capture_snapshot()
        status, body = self.list_resources(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["resources"]), 1)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture_snapshot()
        path = "/snapshots/s1/resources?organizationId=org-1"
        # Missing header.
        status, body = self.raw(path, token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Unregistered token.
        status, body = self.raw(path, token="forged")
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Malformed Authorization header (non-Bearer scheme).
        request = Request(
            f"{self.base_url}{path}",
            headers={"Authorization": "Basic abc"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=5) as response:
                status, body = response.status, response.read()
        except HTTPError as error:
            status, body = error.code, error.read()
            error.close()
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    def test_credential_outranks_query_validation(self) -> None:
        self.capture_snapshot()
        # A missing credential is 401 even when the query is also invalid.
        status, body = self.raw("/snapshots/s1/resources", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- parameter 422

    def test_query_shape_errors_are_422(self) -> None:
        self.capture_snapshot()
        for path in (
            "/snapshots/s1/resources",
            "/snapshots/s1/resources?organizationId=",
            "/snapshots/s1/resources?organizationId=%20%20",
            "/snapshots/s1/resources?organizationId=org-1&organizationId=org-1",
            "/snapshots/s1/resources?organizationId=org-1&organizationId=org-2",
        ):
            with self.subTest(path=path):
                status, body = self.raw(path, token="w1")
                self.assertEqual(status, 422)
                self.assertEqual(json.loads(body)["error"], "validation_error")

        # No failed validation created or altered anything.
        self.assertEqual(len(self.call("/snapshots")[1]["snapshots"]), 1)

    def test_query_validation_outranks_organization_and_snapshot(self) -> None:
        self.capture_snapshot()
        # A malformed query is 422 even when the organization would also
        # mismatch and the snapshot name is unknown.
        status, body = self.raw(
            "/snapshots/ghost/resources?organizationId=org-2&organizationId=org-2",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.capture_snapshot()
        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the snapshot name does not exist anywhere.
        status, body = self.call(
            "/snapshots/ghost/resources?organizationId=org-1", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture_snapshot("s1")
        self.capture_snapshot("foreign", token="w2")
        # The organization matches the credential, but the snapshot belongs
        # to another organization.
        status, body = self.list_resources("foreign")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.list_resources("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        # The failed lookup did not implicitly create the snapshot.
        self.assertEqual(
            self.call(
                "/snapshots/compare/resources",
                method="POST",
                payload={
                    "organizationId": ORG1,
                    "left": "ghost",
                    "right": "ghost",
                },
            )[0],
            404,
        )

    def test_failed_requests_leave_no_trace(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=1, capacity=5)
        self.capture_snapshot("s1")
        self.capture_snapshot("s2", token="w2")

        for path, token in (
            ("/snapshots/s1/resources", None),  # 401
            ("/snapshots/s1/resources?organizationId=", "w1"),  # 422
            ("/snapshots/s2/resources?organizationId=org-1", "w1"),  # 403
            ("/snapshots/ghost/resources?organizationId=org-1", "w1"),  # 404
        ):
            status, _ = self.raw(path, token=token)
            self.assertIn(status, (401, 403, 404, 422))

        # Only the two explicitly created snapshots exist, with the original
        # content intact.
        snapshots = self.call("/snapshots")[1]["snapshots"]
        self.assertEqual([s["snapshotId"] for s in snapshots], ["s1"])
        self.assertEqual(snapshots[0]["resources"], 1)
        self.assertEqual(snapshots[0]["reservations"], 1)
        status, body = self.list_resources("s1")
        self.assertEqual(status, 200)
        self.assertEqual(body["resources"][0]["occupied"], 1)

    # ---------------------------------------------------------------- restart

    def test_new_server_instance_has_no_snapshots_or_balances(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture_snapshot()

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{fresh.server_port}"
            payload = json.dumps(
                {"token": "tok-fresh", "organizationId": ORG1, "role": "read"}
            ).encode()
            request = Request(
                f"{base_url}/auth/tokens",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 201)
            request = Request(
                f"{base_url}/snapshots/s1/resources?organizationId=org-1",
                headers={"Authorization": "Bearer tok-fresh"},
                method="GET",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    status = response.status
            except HTTPError as error:
                status = error.code
                self.assertEqual(
                    json.loads(error.read())["error"], "snapshot_not_found"
                )
                error.close()
            self.assertEqual(status, 404)
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
