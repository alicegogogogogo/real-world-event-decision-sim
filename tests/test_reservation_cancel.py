"""Regression tests for POST /reservations/cancel and
POST /branches/{branchId}/reservations/cancel.

Reservation submission, organization authorization, snapshots, and branch
isolation already existed; this locks down cancellation and capacity
release:

- a valid cancel returns 200 with exactly ``organizationId``,
  ``reservationId``, ``resourceId``, ``quantity`` and ``status:
  "cancelled"`` as compact, key-sorted JSON with one trailing newline, and
  a repeated cancel returns the identical body without releasing capacity
  a second time;
- cancel and capacity release are atomic: the record leaves the
  reservation listing and the resource's occupied balance at once, other
  reservationIds can immediately reuse the released quantity, and the
  resource's recorded capacity is never rewritten;
- a cancelled reservationId is permanently unusable: the reservation
  submission entry point answers 409 ``reservation_cancelled`` for it no
  matter which fields the resubmission carries;
- snapshots taken before the cancel still hold the reservation, snapshots
  taken after it count neither the reservation nor its balance, and a
  branch derived from a snapshot inherits the then-valid or then-cancelled
  state; cancels in the main service, a branch, or a sibling branch never
  propagate, and existing snapshots stay immutable;
- 401 (credential) / 403 (read role, organization mismatch, another
  organization's reservation) / 415 / 400 / 422 / 404
  (reservation_not_found) follow the fixed ordering, branch misses and
  foreign branches keep the existing branch_not_found and ownership
  verdict order, and no failed request ever changes reservations,
  capacity, events, alerts, snapshots, or other branches.
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


def cancel_body(
    reservation_id: str, *, organization_id: str = ORG1
) -> dict[str, Any]:
    return {"organizationId": organization_id, "reservationId": reservation_id}


class ReservationCancelTest(unittest.TestCase):
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
        raw_body: bytes | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, Any]:
        body = (
            raw_body
            if raw_body is not None
            else json.dumps(payload).encode() if payload is not None
            else None
        )
        status, raw = self.raw(
            path,
            method=method,
            body=body,
            token=token,
            content_type=content_type,
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

    def cancel(
        self,
        reservation_id: str,
        *,
        token: str | None = "w1",
        organization_id: str = ORG1,
        prefix: str = "",
    ) -> tuple[int, Any]:
        return self.call(
            f"{prefix}/reservations/cancel",
            method="POST",
            token=token,
            payload=cancel_body(reservation_id, organization_id=organization_id),
        )

    def capture(self, snapshot_id: str = "s1", *, token: str = "w1") -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def fork(self, branch_id: str, snapshot_id: str, *, token: str = "w1") -> None:
        status, _ = self.call(
            "/branches",
            method="POST",
            token=token,
            payload={"branchId": branch_id, "snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def list_main(self, *, token: str | None = "w1", org: str = ORG1) -> Any:
        status, body = self.call(f"/reservations?organizationId={org}", token=token)
        self.assertEqual(status, 200)
        return body

    # ------------------------------------------------------------- happy paths

    def test_cancel_returns_five_field_cancelled_body(self) -> None:
        self.add_reservation("res-1", resource_id="res-a", quantity=3, capacity=10)
        status, body = self.cancel("res-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": ORG1,
                "reservationId": "res-1",
                "resourceId": "res-a",
                "quantity": 3,
                "status": "cancelled",
            },
        )

    def test_response_is_compact_sorted_and_newline_terminated(self) -> None:
        self.add_reservation("res-1", resource_id="res-a", quantity=2, capacity=9)
        status, raw = self.raw(
            "/reservations/cancel",
            method="POST",
            body=json.dumps(cancel_body("res-1")).encode(),
            token="w1",
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertIn(b'"quantity":2', raw)
        body = json.loads(raw)
        self.assertEqual(
            list(body),
            ["organizationId", "quantity", "reservationId", "resourceId", "status"],
        )

    def test_cancel_removes_record_from_listing_and_releases_capacity(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=4, capacity=10)
        self.add_reservation("res-2", resource_id="pool", quantity=3, capacity=10)
        status, body = self.cancel("res-1")
        self.assertEqual(status, 200)

        listing = self.list_main()
        self.assertEqual(
            [row["reservationId"] for row in listing["reservations"]], ["res-2"]
        )
        # The released quantity no longer occupies the resource.
        row = listing["reservations"][0]
        self.assertEqual(row["capacity"], 10)
        self.assertEqual(row["occupied"], 3)
        self.assertEqual(row["remaining"], 7)

        # A new reservationId can immediately use the released quantity.
        status, _ = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body(
                "res-3", resource_id="pool", quantity=7, capacity=10
            ),
        )
        self.assertEqual(status, 201)

    def test_capacity_stays_fixed_after_full_release(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        status, _ = self.cancel("res-1")
        self.assertEqual(status, 200)
        # The first reservation fixed the capacity at 10; a different
        # declared capacity still conflicts after the release.
        status, body = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body(
                "res-2", resource_id="pool", quantity=1, capacity=11
            ),
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_conflict")
        # The recorded capacity itself still works.
        status, _ = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body(
                "res-2", resource_id="pool", quantity=10, capacity=10
            ),
        )
        self.assertEqual(status, 201)

    def test_repeated_cancel_returns_identical_body_and_releases_once(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=6, capacity=10)
        status, first = self.cancel("res-1")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, again = self.cancel("res-1")
            self.assertEqual(status, 200)
            self.assertEqual(again, first)

        # Capacity was released exactly once: the pool still cannot take
        # more than its capacity, and exactly the full capacity fits.
        status, body = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body(
                "res-2", resource_id="pool", quantity=11, capacity=10
            ),
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_exceeded")
        status, _ = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body(
                "res-2", resource_id="pool", quantity=10, capacity=10
            ),
        )
        self.assertEqual(status, 201)

    def test_cancelled_id_is_never_reusable(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        status, _ = self.cancel("res-1")
        self.assertEqual(status, 200)

        # Identical resubmission.
        status, body = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body("res-1", resource_id="pool", quantity=2),
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "reservation_cancelled")
        # Different fields.
        status, body = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body(
                "res-1", resource_id="other", quantity=1, capacity=5
            ),
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "reservation_cancelled")

    # --------------------------------------------------------------- branches

    def test_branch_cancel_releases_only_inside_the_branch(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=4, capacity=10)
        self.capture("s1")
        self.fork("b1", "s1")
        self.fork("b2", "s1")

        status, body = self.cancel("res-1", prefix="/branches/b1")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "cancelled")

        # The branch listing no longer holds the reservation...
        status, listing = self.call(
            "/branches/b1/reservations?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(listing["reservations"], [])
        # ...and the released quantity is reusable inside the branch.
        status, _ = self.call(
            "/branches/b1/reservations",
            method="POST",
            payload=reservation_body(
                "res-9", resource_id="pool", quantity=10, capacity=10
            ),
        )
        self.assertEqual(status, 201)

        # The main service and the sibling branch are untouched.
        self.assertEqual(
            [row["reservationId"] for row in self.list_main()["reservations"]],
            ["res-1"],
        )
        status, sibling = self.call("/branches/b2/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["reservationId"] for row in sibling["reservations"]], ["res-1"]
        )

    def test_branch_cancel_does_not_leak_back_to_main(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=4, capacity=10)
        self.capture("s1")
        self.fork("b1", "s1")
        status, _ = self.cancel("res-1", prefix="/branches/b1")
        self.assertEqual(status, 200)
        # The cancelled id is reusable nowhere else, but the main service
        # still holds its own active reservation.
        status, body = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body(
                "res-1", resource_id="pool", quantity=4, capacity=10
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["reservationId"], "res-1")

    def test_main_cancel_does_not_reach_existing_branch(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=4, capacity=10)
        self.capture("s1")
        self.fork("b1", "s1")
        status, _ = self.cancel("res-1")
        self.assertEqual(status, 200)
        status, branch = self.call("/branches/b1/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["reservationId"] for row in branch["reservations"]], ["res-1"]
        )

    # --------------------------------------------------------------- snapshots

    def test_snapshot_before_cancel_keeps_the_reservation(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture("s1")
        status, _ = self.cancel("res-1")
        self.assertEqual(status, 200)

        status, body = self.call("/snapshots/s1/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["reservationId"] for row in body["reservations"]], ["res-1"]
        )
        status, summary = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(summary["snapshots"][0]["reservations"], 1)

    def test_snapshot_after_cancel_counts_nothing(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        status, _ = self.cancel("res-1")
        self.assertEqual(status, 200)
        self.capture("s1")

        status, body = self.call("/snapshots/s1/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(body["reservations"], [])
        status, summary = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(summary["snapshots"][0]["reservations"], 0)
        self.assertEqual(summary["snapshots"][0]["resources"], 0)

    def test_branch_inherits_cancelled_state_from_snapshot(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        status, _ = self.cancel("res-1")
        self.assertEqual(status, 200)
        self.capture("s1")
        self.fork("b1", "s1")

        # The derived branch treats the cancelled id as cancelled: the
        # submission entry point rejects it and a cancel replays the same
        # cancelled body.
        status, body = self.call(
            "/branches/b1/reservations",
            method="POST",
            payload=reservation_body("res-1", resource_id="pool", quantity=2),
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "reservation_cancelled")
        status, body = self.cancel("res-1", prefix="/branches/b1")
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": ORG1,
                "reservationId": "res-1",
                "resourceId": "pool",
                "quantity": 2,
                "status": "cancelled",
            },
        )

    def test_branch_inherits_valid_state_from_snapshot(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture("s1")
        self.fork("b1", "s1")
        # Cancelled on the main service only after the snapshot; the branch
        # still holds the valid reservation and can cancel it locally.
        status, _ = self.cancel("res-1")
        self.assertEqual(status, 200)
        status, body = self.cancel("res-1", prefix="/branches/b1")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "cancelled")

    # ------------------------------------------------------------------- auth

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.add_reservation("res-1")
        payload = json.dumps(cancel_body("res-1")).encode()
        # Missing header.
        status, body = self.raw(
            "/reservations/cancel", method="POST", body=payload, token=None
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Unregistered token.
        status, body = self.raw(
            "/reservations/cancel", method="POST", body=payload, token="forged"
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Malformed Authorization header (non-Bearer scheme).
        request = Request(
            f"{self.base_url}/reservations/cancel",
            data=payload,
            headers={
                "Authorization": "Basic abc",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                status, body = response.status, response.read()
        except HTTPError as error:
            status, body = error.code, error.read()
            error.close()
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Nothing was cancelled.
        self.assertEqual(len(self.list_main()["reservations"]), 1)

    def test_read_role_is_403(self) -> None:
        self.add_reservation("res-1")
        status, body = self.cancel("res-1", token="r1")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        self.assertEqual(len(self.list_main()["reservations"]), 1)

    def test_organization_mismatch_is_403(self) -> None:
        self.add_reservation("res-1")
        # A credential for ORG2 naming ORG1 in the body.
        status, body = self.cancel("res-1", token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        self.assertEqual(len(self.list_main()["reservations"]), 1)

    def test_cancelling_another_organizations_reservation_is_403(self) -> None:
        self.add_reservation("res-1", token="w2")
        status, body = self.cancel("res-1")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # The reservation survives on its owner's listing.
        status, listing = self.call("/reservations?organizationId=org-2", token="w2")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["reservations"]), 1)

    # ------------------------------------------------------------ media / body

    def test_unsupported_media_type_is_415(self) -> None:
        self.add_reservation("res-1")
        status, body = self.raw(
            "/reservations/cancel",
            method="POST",
            body=json.dumps(cancel_body("res-1")).encode(),
            token="w1",
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(json.loads(body)["error"], "unsupported_media_type")
        self.assertEqual(len(self.list_main()["reservations"]), 1)

    def test_invalid_json_is_400(self) -> None:
        self.add_reservation("res-1")
        status, body = self.raw(
            "/reservations/cancel",
            method="POST",
            body=b'{"organizationId": ',
            token="w1",
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "invalid_json")
        self.assertEqual(len(self.list_main()["reservations"]), 1)

    def test_body_shape_errors_are_422(self) -> None:
        self.add_reservation("res-1")
        cases = [
            b"[1,2]",
            b"{}",
            json.dumps({"organizationId": ORG1}).encode(),
            json.dumps({"reservationId": "res-1"}).encode(),
            json.dumps(
                {
                    "organizationId": ORG1,
                    "reservationId": "res-1",
                    "resourceId": "res-rc",
                }
            ).encode(),
            json.dumps({"organizationId": "", "reservationId": "res-1"}).encode(),
            json.dumps({"organizationId": "  ", "reservationId": "res-1"}).encode(),
            json.dumps({"organizationId": ORG1, "reservationId": ""}).encode(),
            json.dumps({"organizationId": ORG1, "reservationId": 7}).encode(),
            json.dumps({"organizationId": 7, "reservationId": "res-1"}).encode(),
        ]
        for raw_body in cases:
            with self.subTest(raw_body=raw_body):
                status, body = self.raw(
                    "/reservations/cancel", method="POST", body=raw_body, token="w1"
                )
                self.assertEqual(status, 422)
                self.assertEqual(json.loads(body)["error"], "validation_error")
        self.assertEqual(len(self.list_main()["reservations"]), 1)

    def test_credential_and_role_outrank_body_errors(self) -> None:
        self.add_reservation("res-1")
        # Missing credential beats a broken body.
        status, _ = self.raw(
            "/reservations/cancel", method="POST", body=b"not json", token=None
        )
        self.assertEqual(status, 401)
        # Read role beats a broken body.
        status, _ = self.raw(
            "/reservations/cancel", method="POST", body=b"not json", token="r1"
        )
        self.assertEqual(status, 403)
        # Media type beats a validation-shaped body.
        status, _ = self.raw(
            "/reservations/cancel",
            method="POST",
            body=b"{}",
            token="w1",
            content_type="text/plain",
        )
        self.assertEqual(status, 415)

    # ------------------------------------------------------------------- 404

    def test_never_existed_id_is_404(self) -> None:
        status, body = self.cancel("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "reservation_not_found")
        # The failed lookup created nothing.
        self.assertEqual(self.list_main()["reservations"], [])

    # ----------------------------------------------------- branch error order

    def test_unknown_branch_is_404_branch_not_found(self) -> None:
        status, body = self.cancel("res-1", prefix="/branches/ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    def test_foreign_branch_is_403(self) -> None:
        self.add_reservation("res-1", token="w2")
        self.capture("s2", token="w2")
        self.fork("b2", "s2", token="w2")
        status, body = self.cancel(
            "res-1", token="w1", prefix="/branches/b2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_branch_cancel_validates_body(self) -> None:
        self.add_reservation("res-1")
        self.capture("s1")
        self.fork("b1", "s1")
        status, body = self.raw(
            "/branches/b1/reservations/cancel",
            method="POST",
            body=b"{}",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")
        status, body = self.cancel("ghost", prefix="/branches/b1")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "reservation_not_found")

    # ------------------------------------------------------------- side effects

    def test_failures_change_nothing(self) -> None:
        self.add_reservation("res-1", resource_id="pool", quantity=2, capacity=10)
        self.capture("s1")
        before_listing = self.list_main()
        before_snapshots = self.call("/snapshots")[1]
        before_alerts = self.call("/alerts?organizationId=org-1")[1]

        # One of each failing verdict.
        self.cancel("res-1", token="r1")
        self.cancel("res-1", token="w2")
        self.raw(
            "/reservations/cancel",
            method="POST",
            body=b"{}",
            token="w1",
        )
        self.cancel("ghost")

        self.assertEqual(self.list_main(), before_listing)
        self.assertEqual(self.call("/snapshots")[1], before_snapshots)
        self.assertEqual(
            self.call("/alerts?organizationId=org-1")[1], before_alerts
        )
        # The reservation is still active and cancellable afterwards.
        status, body = self.cancel("res-1")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
