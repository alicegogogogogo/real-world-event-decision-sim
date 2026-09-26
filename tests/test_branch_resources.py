"""Regression tests for GET /branches/{branchId}/resources.

The branch resource comparison already diffs two branches' balances; this
locks down the new single-branch, read-only balance view:

- only resources the caller's organization reserves inside the branch are
  listed, one code-point-sorted row each, carrying ``resourceId`` plus the
  three balances ``capacity``, ``occupied`` (the sum of every one of the
  organization's reservations against that resource) and ``remaining``
  (``capacity - occupied``); a branch with no reservations returns ``[]``;
- the response echoes ``organizationId`` and ``branchId`` as compact,
  key-sorted JSON with one trailing newline, and identical requests are
  byte-for-byte stable;
- the balances share their per-organization snapshot with
  POST /branches/compare/resources: a branch compared with itself reports
  these same rows on both sides with every ``equal`` marker true;
- the verdict order is fixed as credential (401), parameter shape (422),
  organization match (403), then branch existence/ownership (404/403); the
  query never implicitly creates a branch and writes nothing, in the branch,
  the main service, other branches, or alerts;
- both read and write credentials may call it, and reads interleaved with
  branch reservation commits always observe a consistent snapshot.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import branch_resource_rows, create_server

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


class BranchResourcesTest(unittest.TestCase):
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
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        merged: dict[str, str] = dict(headers or {})
        if content_type is not None:
            merged["Content-Type"] = content_type
        if token is not None:
            merged["Authorization"] = f"Bearer {token}"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=merged, method=method
        )
        try:
            with urlopen(request, timeout=10) as response:
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

    def get_resources(
        self, branch: str = "br-1", *, query: str = "organizationId=org-1",
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        suffix = f"?{query}" if query else ""
        # GET requests carry no Content-Type of their own.
        status, raw = self.raw(
            f"/branches/{branch}/resources{suffix}",
            content_type=None,
            token=token,
        )
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

    def fork(
        self,
        branch: str = "br-1",
        *,
        snapshot: str = "s1",
        token: str = "w1",
    ) -> None:
        status, _ = self.call(
            "/snapshots", method="POST", token=token, payload={"snapshotId": snapshot}
        )
        self.assertEqual(status, 201)
        status, _ = self.call(
            "/branches",
            method="POST",
            token=token,
            payload={"branchId": branch, "snapshotId": snapshot},
        )
        self.assertEqual(status, 201)

    def reserve(
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

    # ------------------------------------------------------------- happy paths

    def test_rows_report_sums_capacity_and_remaining(self) -> None:
        self.fork()
        self.reserve("br-1", "a-1", resource_id="pool", quantity=2, capacity=10)
        self.reserve("br-1", "a-2", resource_id="pool", quantity=3, capacity=10)
        self.reserve("br-1", "b-1", resource_id="other", quantity=4, capacity=7)

        status, body = self.get_resources()
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["branchId"], "br-1")
        self.assertEqual(
            body["resources"],
            [
                # "other" sorts after "pool".
                {"capacity": 7, "occupied": 4, "remaining": 3,
                 "resourceId": "other"},
                {"capacity": 10, "occupied": 5, "remaining": 5,
                 "resourceId": "pool"},
            ],
        )

    def test_every_reservation_contributes_to_occupied(self) -> None:
        self.fork()
        for index in range(5):
            self.reserve(
                "br-1", f"r-{index}", resource_id="pool", quantity=1, capacity=9
            )
        status, body = self.get_resources()
        self.assertEqual(status, 200)
        (row,) = body["resources"]
        self.assertEqual(row["resourceId"], "pool")
        self.assertEqual(row["capacity"], 9)
        self.assertEqual(row["occupied"], 5)
        self.assertEqual(row["remaining"], 4)

    def test_rows_sort_by_resource_id_code_point(self) -> None:
        self.fork()
        for resource_id in ("res-b", "res-A", "res-a", "res-1"):
            self.reserve(
                "br-1",
                f"book-{resource_id}",
                resource_id=resource_id,
                quantity=1,
                capacity=7,
            )
        status, body = self.get_resources()
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]],
            ["res-1", "res-A", "res-a", "res-b"],
        )

    def test_branch_without_resources_returns_empty_array(self) -> None:
        self.fork()
        status, body = self.get_resources()
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {"branchId": "br-1", "organizationId": ORG1, "resources": []},
        )

    # ------------------------------------------------------------- byte shape

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.fork()
        self.reserve("br-1", "a-1", resource_id="res-1", quantity=2, capacity=9)
        status, raw = self.raw(
            "/branches/br-1/resources?organizationId=org-1",
            content_type=None,
            token="w1",
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertEqual(
            raw,
            b'{"branchId":"br-1","organizationId":"org-1","resources":'
            b'[{"capacity":9,"occupied":2,"remaining":7,"resourceId":"res-1"}]}\n',
        )
        body = json.loads(raw)
        self.assertEqual(list(body), ["branchId", "organizationId", "resources"])
        self.assertEqual(
            list(body["resources"][0]),
            ["capacity", "occupied", "remaining", "resourceId"],
        )

    def test_repeated_requests_are_byte_identical(self) -> None:
        self.fork()
        self.reserve("br-1", "a-1", resource_id="res-1", quantity=2, capacity=10)
        self.reserve("br-1", "a-2", resource_id="res-2", quantity=1, capacity=4)
        raws = [
            self.raw(
                "/branches/br-1/resources?organizationId=org-1",
                content_type=None,
                token="w1",
            )[1]
            for _ in range(4)
        ]
        self.assertEqual(len(set(raws)), 1)

    # ------------------------------------------------- consistency with compare

    def test_self_comparison_matches_the_single_branch_view_row_by_row(
        self,
    ) -> None:
        self.fork()
        self.reserve("br-1", "a-1", resource_id="res-a", quantity=3, capacity=10)
        self.reserve("br-1", "a-2", resource_id="res-a", quantity=2, capacity=10)
        self.reserve("br-1", "b-1", resource_id="res-b", quantity=1, capacity=4)

        status, single = self.get_resources()
        self.assertEqual(status, 200)

        status, comparison = self.call(
            "/branches/compare/resources",
            method="POST",
            payload={
                "organizationId": ORG1,
                "left": "br-1",
                "right": "br-1",
            },
        )
        self.assertEqual(status, 200)

        # Same resources in the same order.
        self.assertEqual(
            [row["resourceId"] for row in single["resources"]],
            [row["resourceId"] for row in comparison["resources"]],
        )
        for single_row, compare_row in zip(
            single["resources"], comparison["resources"]
        ):
            balances = {
                "capacity": single_row["capacity"],
                "occupied": single_row["occupied"],
                "remaining": single_row["remaining"],
            }
            self.assertEqual(compare_row["left"], balances)
            self.assertEqual(compare_row["right"], balances)
            self.assertTrue(compare_row["equal"])
        # The grouping arrays agree with a self comparison as well.
        self.assertEqual(
            comparison["same"],
            [row["resourceId"] for row in single["resources"]],
        )
        self.assertEqual(comparison["sameCount"], len(single["resources"]))
        self.assertEqual(comparison["leftOnly"], [])
        self.assertEqual(comparison["rightOnly"], [])
        self.assertEqual(comparison["diff"], [])
        self.assertEqual(comparison["diffCount"], 0)

    def test_rows_function_orders_and_copies(self) -> None:
        balances = {
            "res-b": {"capacity": 4, "occupied": 1, "remaining": 3},
            "res-A": {"capacity": 9, "occupied": 2, "remaining": 7},
        }
        rows = branch_resource_rows(balances)
        self.assertEqual(
            rows,
            [
                {"capacity": 9, "occupied": 2, "remaining": 7,
                 "resourceId": "res-A"},
                {"capacity": 4, "occupied": 1, "remaining": 3,
                 "resourceId": "res-b"},
            ],
        )
        self.assertEqual(branch_resource_rows({}), [])
        # The rows do not share mutable balance dicts with the input.
        rows[0]["capacity"] = 999
        self.assertEqual(balances["res-A"]["capacity"], 9)

    # -------------------------------------------------------------- isolation

    def test_other_organization_reservations_never_appear(self) -> None:
        self.fork("br-1")
        self.fork("br-2", snapshot="s2", token="w2")
        self.reserve("br-1", "a-1", resource_id="pool", quantity=2, capacity=10)
        self.reserve(
            "br-2", "b-1", token="w2", resource_id="pool", quantity=6, capacity=10
        )

        status, body = self.get_resources("br-1")
        self.assertEqual(status, 200)
        (row,) = body["resources"]
        self.assertEqual(
            row,
            {"capacity": 10, "occupied": 2, "remaining": 8, "resourceId": "pool"},
        )

        status, body = self.get_resources(
            "br-2", query="organizationId=org-2", token="w2"
        )
        self.assertEqual(status, 200)
        (row,) = body["resources"]
        self.assertEqual(
            row,
            {"capacity": 10, "occupied": 6, "remaining": 4, "resourceId": "pool"},
        )

    def test_query_is_read_only_everywhere(self) -> None:
        self.fork()
        self.reserve("br-1", "a-1", resource_id="res-1", quantity=1, capacity=5)

        def branch_summary() -> Any:
            return self.call("/branches/br-1")[1]

        before = branch_summary()
        for _ in range(4):
            status, _ = self.get_resources()
            self.assertEqual(status, 200)
        self.assertEqual(branch_summary(), before)

        # Main-service inventory and alerts are untouched.
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

        # A failed lookup never creates the branch.
        self.assertEqual(self.get_resources("ghost")[0], 404)
        status, body = self.call("/branches/ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    def test_reads_interleaved_with_writes_obey_consistent_snapshots(self) -> None:
        self.fork()
        stop = threading.Event()
        violations: list[str] = []
        reservations_per_writer = 20
        writer_count = 4

        def writer(worker: int) -> None:
            for index in range(reservations_per_writer):
                self.reserve(
                    "br-1",
                    f"w{worker}-{index}",
                    resource_id="pool",
                    quantity=1,
                    capacity=1000,
                )

        def reader() -> None:
            # Pace the reads: a hot loop only overflows the small default
            # accept backlog under WSL without exercising anything extra.
            while not stop.wait(0.002):
                status, body = self.get_resources()
                if status != 200:
                    violations.append(f"status {status}")
                    return
                rows = body["resources"]
                if [r["resourceId"] for r in rows] != sorted(
                    r["resourceId"] for r in rows
                ):
                    violations.append("unsorted rows")
                for row in rows:
                    if (
                        row["occupied"] + row["remaining"] != row["capacity"]
                        or not (0 <= row["occupied"] <= row["capacity"])
                    ):
                        violations.append(f"torn balances: {row}")

        readers = [threading.Thread(target=reader) for _ in range(2)]
        for thread in readers:
            thread.start()
        writers = [
            threading.Thread(target=writer, args=(worker,))
            for worker in range(writer_count)
        ]
        for thread in writers:
            thread.start()
        for thread in writers:
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive(), "writer timed out")
        stop.set()
        for thread in readers:
            thread.join(timeout=5)

        self.assertEqual(violations, [])
        status, body = self.get_resources()
        self.assertEqual(status, 200)
        (row,) = body["resources"]
        self.assertEqual(row["occupied"], reservations_per_writer * writer_count)
        self.assertEqual(row["remaining"], 1000 - row["occupied"])

    # ------------------------------------------------------------ roles / auth

    def test_read_and_write_credentials_may_query(self) -> None:
        self.fork()
        self.reserve("br-1", "a-1", resource_id="res-1", quantity=1, capacity=5)
        for token in ("w1", "r1"):
            with self.subTest(token=token):
                status, body = self.get_resources(token=token)
                self.assertEqual(status, 200)
                self.assertEqual(
                    [row["resourceId"] for row in body["resources"]], ["res-1"]
                )

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.fork()
        # A malformed/missing credential is 401 even when the parameter shape
        # is also invalid: the credential is checked first.
        status, body = self.raw(
            "/branches/br-1/resources",
            content_type=None,
            token=None,
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

        status, body = self.raw(
            "/branches/br-1/resources?organizationId=org-1",
            content_type=None,
            token="forged",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

        status, body = self.raw(
            "/branches/ghost/resources",
            content_type=None,
            headers={"Authorization": "Basic abc"},
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

        status, body = self.raw(
            "/branches/br-1/resources?organizationId=org-1",
            content_type=None,
            headers={"Authorization": "Bearer"},
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # ------------------------------------------------------------- 422 / 403/404

    def test_parameter_shape_is_422_before_org_and_branch(self) -> None:
        self.fork()
        for query, reason in (
            ("", "missing"),
            ("organizationId=", "blank"),
            ("organizationId=%20%20", "whitespace only"),
            ("organizationId=org-1&organizationId=org-1", "duplicated"),
        ):
            with self.subTest(reason):
                status, body = self.get_resources(
                    "ghost", query=query, token="w1"
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")
        # The branch still does not exist after the failed lookups.
        self.assertEqual(self.call("/branches/ghost")[0], 404)

    def test_organization_mismatch_is_403_before_branch_lookup(self) -> None:
        self.fork()
        # An org-2 credential naming org-1 against a branch name that exists
        # nowhere: the organization decision outranks the 404.
        status, body = self.get_resources(
            "ghost", query="organizationId=org-1", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # An org-1 credential naming org-2 likewise cannot probe names.
        status, body = self.get_resources(
            "ghost", query="organizationId=org-2", token="w1"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_owned_branch_is_403(self) -> None:
        self.fork("br-1")
        self.fork("foreign", snapshot="s2", token="w2")
        status, body = self.get_resources(
            "foreign", query="organizationId=org-1", token="w1"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # The owner reads it normally.
        status, body = self.get_resources(
            "foreign", query="organizationId=org-2", token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["resources"], [])

    def test_unknown_branch_with_matching_org_is_404(self) -> None:
        status, body = self.get_resources("never-existed")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    # ------------------------------------------------------- routing unchanged

    def test_unknown_subpaths_keep_standard_404(self) -> None:
        self.fork()
        # A deeper, unknown sub-path under a known branch stays not_found.
        status, body = self.raw(
            "/branches/br-1/resources/extra?organizationId=org-1",
            content_type=None,
            token="w1",
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"], "not_found")
        # No write entry point was introduced for the resource path.
        status, body = self.raw(
            "/branches/br-1/resources",
            method="POST",
            body=b'{"organizationId":"org-1"}',
            token="w1",
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"], "not_found")
        # Unknown branch keeps branch_not_found precedence on that POST.
        status, body = self.raw(
            "/branches/ghost/resources",
            method="POST",
            body=b'{"organizationId":"org-1"}',
            token="w1",
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"], "branch_not_found")


if __name__ == "__main__":
    unittest.main()
