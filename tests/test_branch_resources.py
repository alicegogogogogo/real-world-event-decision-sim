"""Regression tests for GET /branches/{branchId}/resources.

The branch resource comparisons already aligned two branches' balances;
this locks down the new read-only entry point that lists one branch's
current resource balances for the caller's organization:

- the ``resources`` array carries one row per resource the organization
  reserves in the branch, code-point sorted, each with the ``capacity``,
  ``occupied`` and ``remaining`` balances; ``occupied`` sums every
  reservation the organization holds against the resource in that branch
  and ``remaining`` is ``capacity - occupied``;
- the response echoes ``organizationId`` and ``branchId``, is compact,
  key-sorted JSON with integer balances and one trailing newline, and
  identical requests are byte-for-byte identical; a branch with no
  resources yields an empty row array;
- the balances share the comparison's exact contract: a branch compared
  with itself via POST /branches/compare/resources reports every row
  equal, and other organizations' reservations never contribute;
- the verdict order is fixed — 401 (credential) before 422 (query shape)
  before 403 (organization, then foreign branch) before 404
  (branch_not_found) — and nothing is ever written or implicitly
  created, including across concurrent reads and writes.
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


class BranchResourceListTest(unittest.TestCase):
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

    def fork_branch(
        self,
        *,
        snapshot_id: str = "s1",
        branch_id: str = "br-1",
        token: str = "w1",
    ) -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)
        status, _ = self.call(
            "/branches",
            method="POST",
            token=token,
            payload={"branchId": branch_id, "snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def add_branch_reservation(
        self,
        branch: str,
        reservation_id: str,
        token: str = "w1",
        **kwargs: Any,
    ) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        status, _ = self.call(
            f"/branches/{branch}/reservations",
            method="POST",
            token=token,
            payload=reservation_body(
                reservation_id, organization_id=organization_id, **kwargs
            ),
        )
        self.assertEqual(status, 201)

    def list_resources(
        self, branch: str = "br-1", *, token: str | None = "w1", org: str = ORG1
    ) -> tuple[int, Any]:
        return self.call(
            f"/branches/{branch}/resources?organizationId={org}", token=token
        )

    # ------------------------------------------------------------- happy paths

    def test_rows_echo_identifiers_and_report_three_balances(self) -> None:
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="res-a", quantity=3, capacity=10
        )
        self.add_branch_reservation(
            "br-1", "res-2", resource_id="res-b", quantity=1, capacity=4
        )

        status, body = self.list_resources()
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["branchId"], "br-1")
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
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="pool", quantity=2, capacity=10
        )
        self.add_branch_reservation(
            "br-1", "res-2", resource_id="pool", quantity=3, capacity=10
        )
        self.add_branch_reservation(
            "br-1", "res-3", resource_id="pool", quantity=1, capacity=10
        )

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
        self.fork_branch()
        for resource_id in ("res-b", "res-A", "res-a", "res-1"):
            self.add_branch_reservation(
                "br-1",
                f"book-{resource_id}",
                resource_id=resource_id,
                quantity=1,
                capacity=7,
            )
        status, body = self.list_resources()
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]],
            ["res-1", "res-A", "res-a", "res-b"],
        )

    def test_branch_without_resources_returns_empty_array(self) -> None:
        self.fork_branch()
        status, body = self.list_resources()
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {"organizationId": ORG1, "branchId": "br-1", "resources": []},
        )

    def test_balances_track_branch_writes(self) -> None:
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="pool", quantity=2, capacity=5
        )
        _, body = self.list_resources()
        self.assertEqual(body["resources"][0]["occupied"], 2)
        self.add_branch_reservation(
            "br-1", "res-2", resource_id="pool", quantity=3, capacity=5
        )
        _, body = self.list_resources()
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "pool",
                    "capacity": 5,
                    "occupied": 5,
                    "remaining": 0,
                }
            ],
        )

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="res-1", quantity=2, capacity=9
        )

        status, raw = self.raw(
            "/branches/br-1/resources?organizationId=org-1", token="w1"
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
        self.assertEqual(list(body), ["branchId", "organizationId", "resources"])
        self.assertEqual(
            list(body["resources"][0]),
            ["capacity", "occupied", "remaining", "resourceId"],
        )

    def test_repeated_requests_are_byte_identical(self) -> None:
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="res-1", quantity=2, capacity=10
        )
        self.add_branch_reservation(
            "br-1", "res-2", resource_id="res-2", quantity=1, capacity=4
        )
        raws = [
            self.raw("/branches/br-1/resources?organizationId=org-1", token="w1")[
                1
            ]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # ------------------------------------------- consistency with comparison

    def test_self_comparison_agrees_row_by_row(self) -> None:
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="res-a", quantity=3, capacity=10
        )
        self.add_branch_reservation(
            "br-1", "res-2", resource_id="res-a", quantity=2, capacity=10
        )
        self.add_branch_reservation(
            "br-1", "res-3", resource_id="res-b", quantity=1, capacity=4
        )

        status, listing = self.list_resources()
        self.assertEqual(status, 200)
        status, comparison = self.call(
            "/branches/compare/resources",
            method="POST",
            payload={"organizationId": ORG1, "left": "br-1", "right": "br-1"},
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
        self.fork_branch()
        self.fork_branch(snapshot_id="s2", branch_id="foreign", token="w2")
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="pool", quantity=2, capacity=10
        )
        # ORG2 reserves the same resourceId in its own branch and on the
        # main service; neither leaks into ORG1's branch listing.
        self.add_branch_reservation(
            "foreign",
            "res-2",
            token="w2",
            resource_id="pool",
            quantity=5,
            capacity=10,
        )
        status, _ = self.call(
            "/reservations",
            method="POST",
            token="w2",
            payload=reservation_body(
                "res-3", resource_id="pool", quantity=4, capacity=10,
                organization_id=ORG2,
            ),
        )
        self.assertEqual(status, 201)

        status, body = self.list_resources()
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

        # ORG2's own branch listing sees only ORG2 balances.
        status, body = self.list_resources("foreign", token="w2", org=ORG2)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG2)
        self.assertEqual(body["branchId"], "foreign")
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
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="res-1", quantity=1, capacity=5
        )

        before = self.call("/branches/br-1")[1]
        for _ in range(3):
            status, _ = self.list_resources()
            self.assertEqual(status, 200)
        after = self.call("/branches/br-1")[1]
        self.assertEqual(before, after)

        # Main-service state and alerts are untouched as well.
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    def test_concurrent_reads_and_writes_stay_consistent(self) -> None:
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-0", resource_id="pool", quantity=1, capacity=40
        )
        errors: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                status, body = self.list_resources()
                if status != 200:
                    errors.append(f"status {status}")
                    return
                rows = body["resources"]
                if [row["resourceId"] for row in rows] != sorted(
                    row["resourceId"] for row in rows
                ):
                    errors.append("rows out of order")
                    return
                for row in rows:
                    if row["remaining"] != row["capacity"] - row["occupied"]:
                        errors.append(f"inconsistent row: {row}")
                        return
                    if not (0 <= row["occupied"] <= row["capacity"]):
                        errors.append(f"impossible balances: {row}")
                        return

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for thread in readers:
            thread.start()
        for index in range(1, 8):
            self.add_branch_reservation(
                "br-1",
                f"res-{index}",
                resource_id="pool",
                quantity=1,
                capacity=40,
            )
        stop.set()
        for thread in readers:
            thread.join(timeout=5)
        self.assertEqual(errors, [])

        # After the dust settles the listing reflects every commit.
        _, body = self.list_resources()
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "pool",
                    "capacity": 40,
                    "occupied": 8,
                    "remaining": 32,
                }
            ],
        )

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_list(self) -> None:
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="res-1", quantity=1, capacity=5
        )
        status, body = self.list_resources(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["resources"]), 1)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.fork_branch()
        path = "/branches/br-1/resources?organizationId=org-1"
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
        self.fork_branch()
        # A missing credential is 401 even when the query is also invalid.
        status, body = self.raw("/branches/br-1/resources", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- parameter 422

    def test_query_shape_errors_are_422(self) -> None:
        self.fork_branch()
        for path in (
            "/branches/br-1/resources",
            "/branches/br-1/resources?organizationId=",
            "/branches/br-1/resources?organizationId=%20%20",
            "/branches/br-1/resources?organizationId=org-1&organizationId=org-1",
            "/branches/br-1/resources?organizationId=org-1&organizationId=org-2",
        ):
            with self.subTest(path=path):
                status, body = self.raw(path, token="w1")
                self.assertEqual(status, 422)
                self.assertEqual(json.loads(body)["error"], "validation_error")

        # No failed validation created or altered anything.
        self.assertEqual(self.call("/branches/br-1")[1]["reservations"], 0)

    def test_query_validation_outranks_organization_and_branch(self) -> None:
        self.fork_branch()
        # A malformed query is 422 even when the organization would also
        # mismatch and the branch name is unknown.
        status, body = self.raw(
            "/branches/ghost/resources?organizationId=org-2&organizationId=org-2",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_branch_lookup(self) -> None:
        self.fork_branch()
        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the branch name does not exist anywhere.
        status, body = self.call(
            "/branches/ghost/resources?organizationId=org-1", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_branch_is_403_not_404(self) -> None:
        self.fork_branch()
        self.fork_branch(snapshot_id="s2", branch_id="foreign", token="w2")
        # The organization matches the credential, but the branch belongs
        # to another organization.
        status, body = self.list_resources("foreign")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_branch_is_404_and_never_created(self) -> None:
        status, body = self.list_resources("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")
        # The failed lookup did not implicitly create the branch.
        self.assertEqual(self.call("/branches/ghost")[0], 404)

    def test_branch_lookup_outranks_branch_ownership(self) -> None:
        self.fork_branch(snapshot_id="s2", branch_id="foreign", token="w2")
        # A missing branch is 404 even though a foreign branch exists.
        status, body = self.list_resources("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    # ---------------------------------------------------------------- restart

    def test_new_server_instance_has_no_branches_or_balances(self) -> None:
        self.fork_branch()
        self.add_branch_reservation(
            "br-1", "res-1", resource_id="pool", quantity=2, capacity=10
        )

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
                f"{base_url}/branches/br-1/resources?organizationId=org-1",
                headers={"Authorization": "Bearer tok-fresh"},
                method="GET",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    status = response.status
            except HTTPError as error:
                status = error.code
                self.assertEqual(json.loads(error.read())["error"], "branch_not_found")
                error.close()
            self.assertEqual(status, 404)
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
