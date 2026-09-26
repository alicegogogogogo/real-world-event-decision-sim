"""Regression tests for GET /snapshots/{snapshotId}/reservations.

The snapshot reservation comparisons already aligned two snapshots'
reservations; this locks down the new read-only entry point that lists one
snapshot's captured reservations for the caller's organization:

- the ``reservations`` array carries one row per reservation captured in
  the snapshot, sorted by ``resourceId`` then ``reservationId`` in Unicode
  code-point order, each row reporting exactly the five reservation fields
  (``organizationId``, ``reservationId``, ``resourceId``, ``quantity``,
  ``capacity``) — the same field names the reservation comparison uses;
- the response echoes ``organizationId`` and ``snapshotId``, is compact,
  key-sorted JSON with integer quantities and capacities and one trailing
  newline, and identical requests are byte-for-byte identical; a snapshot
  with no reservations yields an empty row array;
- the rows share the comparison's exact contract: a snapshot compared with
  itself via POST /snapshots/compare/reservations reports every listed
  reservation in ``same`` with zero diffs, and other organizations'
  reservations never contribute;
- snapshots are immutable: later main-service writes never change the
  listing; the verdict order is fixed — 401 (credential) before 422
  (query shape) before 403 (organization, then foreign snapshot) before
  404 (snapshot_not_found) — and nothing is ever written or implicitly
  created, including the main-service ledger, reservation inventory, and
  alert state; restarting clears snapshots and reservations.
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


class SnapshotReservationListTest(unittest.TestCase):
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
        self, reservation_id: str, token: str = "w1", **kwargs: Any
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

    def capture(
        self, snapshot_id: str = "s1", *, token: str = "w1"
    ) -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def list_reservations(
        self, snapshot: str = "s1", *, token: str | None = "w1", org: str = ORG1
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/reservations?organizationId={org}", token=token
        )

    # ------------------------------------------------------------- happy paths

    def test_rows_echo_identifiers_and_report_five_fields(self) -> None:
        self.add_reservation(
            "res-1", resource_id="res-a", quantity=3, capacity=10
        )
        self.add_reservation(
            "res-2", resource_id="res-b", quantity=1, capacity=4
        )
        self.capture()

        status, body = self.list_reservations()
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(
            body["reservations"],
            [
                {
                    "organizationId": ORG1,
                    "reservationId": "res-1",
                    "resourceId": "res-a",
                    "quantity": 3,
                    "capacity": 10,
                },
                {
                    "organizationId": ORG1,
                    "reservationId": "res-2",
                    "resourceId": "res-b",
                    "quantity": 1,
                    "capacity": 4,
                },
            ],
        )

    def test_rows_sort_by_resource_then_reservation_in_code_point_order(
        self,
    ) -> None:
        self.add_reservation(
            "res-2", resource_id="res-a", quantity=1, capacity=10
        )
        self.add_reservation(
            "res-10", resource_id="res-a", quantity=2, capacity=10
        )
        self.add_reservation(
            "res-1", resource_id="res-a", quantity=3, capacity=10
        )
        self.add_reservation(
            "res-9", resource_id="res-A", quantity=1, capacity=4
        )
        self.capture()
        status, body = self.list_reservations()
        self.assertEqual(status, 200)
        self.assertEqual(
            [
                (row["resourceId"], row["reservationId"])
                for row in body["reservations"]
            ],
            [
                ("res-A", "res-9"),
                ("res-a", "res-1"),
                ("res-a", "res-10"),
                ("res-a", "res-2"),
            ],
        )

    def test_snapshot_without_reservations_returns_empty_array(self) -> None:
        self.capture()
        status, body = self.list_reservations()
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {"organizationId": ORG1, "snapshotId": "s1", "reservations": []},
        )

    def test_listing_reflects_capture_time_only(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=5)
        self.capture()

        # Reservations committed after capture never enter the snapshot.
        self.add_reservation("res-2", resource_id="pool", quantity=3, capacity=5)
        _, body = self.list_reservations()
        self.assertEqual(
            body["reservations"],
            [
                {
                    "organizationId": ORG1,
                    "reservationId": "res-1",
                    "resourceId": "pool",
                    "quantity": 2,
                    "capacity": 5,
                }
            ],
        )

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_reservation("res-1", resource_id="res-1", quantity=2, capacity=9)
        self.capture()

        status, raw = self.raw(
            "/snapshots/s1/reservations?organizationId=org-1", token="w1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Quantities and capacities stay integers.
        self.assertIn(b'"quantity":2', raw)
        self.assertIn(b'"capacity":9', raw)
        # Keys are sorted by code point at every level.
        body = json.loads(raw)
        self.assertEqual(
            list(body), ["organizationId", "reservations", "snapshotId"]
        )
        self.assertEqual(
            list(body["reservations"][0]),
            [
                "capacity",
                "organizationId",
                "quantity",
                "reservationId",
                "resourceId",
            ],
        )

    def test_repeated_requests_are_byte_identical(self) -> None:
        self.add_reservation("res-1", resource_id="res-1", quantity=2, capacity=10)
        self.add_reservation("res-2", resource_id="res-2", quantity=1, capacity=4)
        self.capture()
        # Main-service state changes between reads must not perturb bytes.
        self.add_reservation("res-3", resource_id="res-3", quantity=1, capacity=8)
        raws = [
            self.raw(
                "/snapshots/s1/reservations?organizationId=org-1", token="w1"
            )[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # ------------------------------------------- consistency with comparison

    def test_self_comparison_agrees_item_by_item(self) -> None:
        self.add_reservation("res-1", resource_id="res-a", quantity=3, capacity=10)
        self.add_reservation("res-2", resource_id="res-a", quantity=2, capacity=10)
        self.add_reservation("res-3", resource_id="res-b", quantity=1, capacity=4)
        self.capture()

        status, listing = self.list_reservations()
        self.assertEqual(status, 200)
        status, comparison = self.call(
            "/snapshots/compare/reservations",
            method="POST",
            payload={"organizationId": ORG1, "left": "s1", "right": "s1"},
        )
        self.assertEqual(status, 200)

        # Every listed reservation lands in the self comparison's same
        # group; nothing is left-only, right-only, or different.
        self.assertEqual(comparison["same"], ["res-1", "res-2", "res-3"])
        self.assertEqual(comparison["sameCount"], 3)
        self.assertEqual(comparison["leftOnly"], [])
        self.assertEqual(comparison["leftOnlyCount"], 0)
        self.assertEqual(comparison["rightOnly"], [])
        self.assertEqual(comparison["rightOnlyCount"], 0)
        self.assertEqual(comparison["diff"], [])
        self.assertEqual(comparison["diffCount"], 0)
        self.assertEqual(
            [row["reservationId"] for row in listing["reservations"]],
            ["res-1", "res-2", "res-3"],
        )

    def test_other_organizations_never_contribute(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture("s1")
        # ORG2 reserves the same resourceId, on the main service and inside
        # its own snapshot; neither leaks into ORG1's snapshot listing.
        self.add_reservation(
            "res-2", token="w2", resource_id="pool", quantity=5, capacity=10
        )
        self.capture("s2", token="w2")

        status, body = self.list_reservations()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["reservations"],
            [
                {
                    "organizationId": ORG1,
                    "reservationId": "res-1",
                    "resourceId": "pool",
                    "quantity": 2,
                    "capacity": 10,
                }
            ],
        )

        # ORG2's own snapshot listing sees only ORG2 captured reservations.
        status, body = self.list_reservations("s2", token="w2", org=ORG2)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG2)
        self.assertEqual(body["snapshotId"], "s2")
        self.assertEqual(
            body["reservations"],
            [
                {
                    "organizationId": ORG2,
                    "reservationId": "res-2",
                    "resourceId": "pool",
                    "quantity": 5,
                    "capacity": 10,
                }
            ],
        )

    # -------------------------------------------------------------- read-only

    def test_listing_is_read_only(self) -> None:
        self.add_reservation("res-1", resource_id="res-1", quantity=1, capacity=5)
        self.capture()

        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, _ = self.list_reservations()
            self.assertEqual(status, 200)
        status, after = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

        # Main-service state and alerts are untouched as well.
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(reservations["reservations"]), 1)
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_list(self) -> None:
        self.add_reservation("res-1", resource_id="res-1", quantity=1, capacity=5)
        self.capture()
        status, body = self.list_reservations(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["reservations"]), 1)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture()
        path = "/snapshots/s1/reservations?organizationId=org-1"
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
        self.capture()
        # A missing credential is 401 even when the query is also invalid.
        status, body = self.raw("/snapshots/s1/reservations", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- parameter 422

    def test_query_shape_errors_are_422(self) -> None:
        self.capture()
        for path in (
            "/snapshots/s1/reservations",
            "/snapshots/s1/reservations?organizationId=",
            "/snapshots/s1/reservations?organizationId=%20%20",
            "/snapshots/s1/reservations?organizationId=org-1&organizationId=org-1",
            "/snapshots/s1/reservations?organizationId=org-1&organizationId=org-2",
        ):
            with self.subTest(path=path):
                status, body = self.raw(path, token="w1")
                self.assertEqual(status, 422)
                self.assertEqual(json.loads(body)["error"], "validation_error")

        # No failed validation created or altered anything.
        self.assertEqual(
            self.call("/snapshots")[1]["snapshots"][0]["reservations"], 0
        )

    def test_query_validation_outranks_organization_and_snapshot(self) -> None:
        self.capture()
        # A malformed query is 422 even when the organization would also
        # mismatch and the snapshot name is unknown.
        status, body = self.raw(
            "/snapshots/ghost/reservations?organizationId=org-2&organizationId=org-2",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.capture()
        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the snapshot name does not exist anywhere.
        status, body = self.call(
            "/snapshots/ghost/reservations?organizationId=org-1", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        # The organization matches the credential, but the snapshot belongs
        # to another organization.
        status, body = self.list_reservations("s2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.list_reservations("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        # The failed lookup did not implicitly create the snapshot.
        self.assertEqual(self.call("/snapshots")[1]["snapshots"], [])

    def test_snapshot_lookup_outranks_snapshot_ownership(self) -> None:
        self.capture("s2", token="w2")
        # A missing snapshot is 404 even though a foreign snapshot exists.
        status, body = self.list_reservations("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    # ---------------------------------------------------------------- restart

    def test_new_server_instance_has_no_snapshots_or_reservations(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture()

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
                f"{base_url}/snapshots/s1/reservations?organizationId=org-1",
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
