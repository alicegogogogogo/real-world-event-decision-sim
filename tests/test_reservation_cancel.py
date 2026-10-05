"""Reservation cancellation and capacity release contract tests.

Covers ``POST /reservations/cancel`` on the main service and
``POST /branches/{branchId}/reservations/cancel`` inside a branch: the
fixed response body, the atomic capacity release, the permanent retirement
of a cancelled reservationId, snapshot/branch inheritance of the cancelled
state, and the full error taxonomy.
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


class ReservationCancelTest(unittest.TestCase):
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
        token: str | None = None,
    ) -> tuple[int, Any]:
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
                return response.status, json.load(response)
        except HTTPError as error:
            try:
                return error.code, json.load(error)
            finally:
                error.close()

    # --- helpers ---------------------------------------------------------------

    def reservation_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "reservationId": "res-1",
            "resourceId": "r-a",
            "quantity": 2,
            "capacity": 5,
        }
        payload.update(overrides)
        return payload

    def cancel_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": "org-1",
            "reservationId": "res-1",
        }
        payload.update(overrides)
        return payload

    def post_reservation(
        self, payload: Any, path: str = "/reservations"
    ) -> tuple[int, Any]:
        return self.request(path, method="POST", body=json.dumps(payload).encode())

    def post_cancel(
        self,
        payload: Any,
        path: str = "/reservations/cancel",
        *,
        raw: bool = False,
        content_type: str | None = "application/json",
        auth: bool = True,
        token: str | None = None,
    ) -> tuple[int, Any]:
        body = payload if raw else json.dumps(payload).encode()
        return self.request(
            path,
            method="POST",
            body=body,
            content_type=content_type,
            auth=auth,
            token=token,
        )

    def list_reservations(
        self, organization_id: str = "org-1", path: str | None = None
    ) -> Any:
        target = path or f"/reservations?organizationId={organization_id}"
        status, body = self.request(target)
        assert status == 200
        return body

    def create_snapshot(self, snapshot_id: str) -> None:
        status, _ = self.request(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": snapshot_id}).encode(),
        )
        assert status == 201

    def create_branch(self, branch_id: str, snapshot_id: str) -> None:
        status, _ = self.request(
            "/branches",
            method="POST",
            body=json.dumps(
                {"branchId": branch_id, "snapshotId": snapshot_id}
            ).encode(),
        )
        assert status == 201

    # --- happy path ------------------------------------------------------------

    def test_cancel_returns_200_with_fixed_cancelled_body(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        status, body = self.post_cancel(self.cancel_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "reservationId": "res-1",
                "resourceId": "r-a",
                "quantity": 2,
                "status": "cancelled",
            },
        )

    def test_cancel_response_is_compact_sorted_json_with_newline(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        body = json.dumps(self.cancel_payload()).encode()
        request = Request(
            f"{self.base_url}/reservations/cancel",
            data=body,
            headers={
                "Content-Type": "application/json",
                **_support.authorization_header(
                    self.token_cache, self.base_url, "/reservations/cancel", body
                ),
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            raw = response.read()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(
            raw[:-1].decode(),
            '{"organizationId":"org-1","quantity":2,"reservationId":"res-1",'
            '"resourceId":"r-a","status":"cancelled"}',
        )

    def test_cancel_removes_reservation_from_listing(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.assertEqual(self.post_cancel(self.cancel_payload())[0], 200)
        listing = self.list_reservations()
        self.assertEqual(listing, {"organizationId": "org-1", "reservations": []})

    def test_cancel_releases_capacity_for_other_reservation_ids(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.assertEqual(self.post_cancel(self.cancel_payload())[0], 200)
        # The freed quantity is immediately reservable by a new identifier.
        status, body = self.post_reservation(
            self.reservation_payload(reservationId="res-2", quantity=5)
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["occupied"], 5)
        self.assertEqual(body["remaining"], 0)

    def test_cancel_keeps_the_recorded_capacity(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.assertEqual(self.post_cancel(self.cancel_payload())[0], 200)
        # The capacity fixed by the first reservation is never rewritten.
        status, body = self.post_reservation(
            self.reservation_payload(reservationId="res-2", capacity=9)
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_conflict")
        status, body = self.post_reservation(
            self.reservation_payload(reservationId="res-3", quantity=1)
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["capacity"], 5)

    def test_repeated_cancel_returns_identical_body_without_releasing_twice(
        self,
    ) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        first_status, first_body = self.post_cancel(self.cancel_payload())
        second_status, second_body = self.post_cancel(self.cancel_payload())
        self.assertEqual(first_status, 200)
        self.assertEqual(second_status, 200)
        self.assertEqual(first_body, second_body)
        # The quantity was released exactly once: capacity 5 fits only one
        # further reservation of 4 after the single release of 2.
        status, body = self.post_reservation(
            self.reservation_payload(reservationId="res-2", quantity=7)
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_exceeded")
        status, body = self.post_reservation(
            self.reservation_payload(reservationId="res-2", quantity=5)
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["remaining"], 0)

    def test_cancelled_reservation_id_is_never_reusable(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.assertEqual(self.post_cancel(self.cancel_payload())[0], 200)
        # Identical replay and any field variation are equally rejected.
        for override in (
            {},
            {"quantity": 1},
            {"resourceId": "r-b", "capacity": 9},
            {"capacity": 5, "quantity": 2, "resourceId": "r-a"},
        ):
            with self.subTest(override=override):
                status, body = self.post_reservation(
                    self.reservation_payload(**override)
                )
                self.assertEqual(status, 409)
                self.assertEqual(body["error"], "reservation_cancelled")

    def test_failed_cancel_changes_nothing(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        status, _ = self.post_cancel(self.cancel_payload(reservationId="res-x"))
        self.assertEqual(status, 404)
        listing = self.list_reservations()
        self.assertEqual(len(listing["reservations"]), 1)
        self.assertEqual(listing["reservations"][0]["occupied"], 2)
        self.assertEqual(listing["reservations"][0]["remaining"], 3)

    # --- snapshot and branch inheritance ----------------------------------------

    def test_snapshot_before_cancel_keeps_the_reservation(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.create_snapshot("snap-before")
        self.assertEqual(self.post_cancel(self.cancel_payload())[0], 200)
        status, body = self.request(
            "/snapshots/snap-before/reservations?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["reservations"]), 1)
        self.assertEqual(body["reservations"][0]["reservationId"], "res-1")

    def test_snapshot_after_cancel_drops_reservation_and_balance(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.assertEqual(self.post_cancel(self.cancel_payload())[0], 200)
        self.create_snapshot("snap-after")
        status, body = self.request(
            "/snapshots/snap-after/reservations?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["reservations"], [])
        status, body = self.request(
            "/snapshots/snap-after/resources?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["resources"], [])

    def test_branch_inherits_cancelled_state_from_snapshot(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.assertEqual(self.post_cancel(self.cancel_payload())[0], 200)
        self.create_snapshot("snap-1")
        self.create_branch("branch-1", "snap-1")
        # The cancelled identifier stays retired inside the branch too.
        status, body = self.post_reservation(
            self.reservation_payload(), path="/branches/branch-1/reservations"
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "reservation_cancelled")
        # A repeated cancel inside the branch reconciles identically.
        status, body = self.post_cancel(
            self.cancel_payload(), path="/branches/branch-1/reservations/cancel"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "cancelled")
        self.assertEqual(body["reservationId"], "res-1")

    def test_branch_inherits_active_reservation_and_cancels_it_alone(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.create_snapshot("snap-1")
        self.create_branch("branch-1", "snap-1")
        self.create_branch("branch-2", "snap-1")
        # The branch inherited the active reservation; cancelling it there
        # releases only the branch's copy.
        status, body = self.post_cancel(
            self.cancel_payload(), path="/branches/branch-1/reservations/cancel"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "cancelled")
        status, listing = self.request(
            "/branches/branch-1/reservations?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(listing["reservations"], [])
        # The sibling branch and the main service are untouched.
        status, listing = self.request(
            "/branches/branch-2/reservations?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["reservations"]), 1)
        main_listing = self.list_reservations()
        self.assertEqual(len(main_listing["reservations"]), 1)
        # The main service still accepts an identical replay of res-1.
        status, _ = self.post_reservation(self.reservation_payload())
        self.assertEqual(status, 200)

    def test_main_cancel_does_not_propagate_into_existing_branch(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.create_snapshot("snap-1")
        self.create_branch("branch-1", "snap-1")
        self.assertEqual(self.post_cancel(self.cancel_payload())[0], 200)
        status, listing = self.request(
            "/branches/branch-1/reservations?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["reservations"]), 1)
        # Inside the branch the identifier is still live, so a branch-local
        # replay reconciles instead of reporting the main-service cancel.
        status, _ = self.post_reservation(
            self.reservation_payload(), path="/branches/branch-1/reservations"
        )
        self.assertEqual(status, 200)

    # --- error taxonomy ---------------------------------------------------------

    def test_cancel_requires_authentication(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        for headers in ({}, {"Authorization": "Bearer"}, {"Authorization": "Basic x"}):
            with self.subTest(headers=headers):
                request = Request(
                    f"{self.base_url}/reservations/cancel",
                    data=json.dumps(self.cancel_payload()).encode(),
                    headers={"Content-Type": "application/json", **headers},
                    method="POST",
                )
                try:
                    with urlopen(request, timeout=5) as response:
                        status = response.status
                except HTTPError as error:
                    status = error.code
                    error.close()
                self.assertEqual(status, 401)
        status, body = self.post_cancel(
            self.cancel_payload(), token="never-registered"
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_cancel_read_role_is_forbidden(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        payload = json.dumps(
            {"token": "reader-1", "organizationId": "org-1", "role": "read"}
        ).encode()
        request = Request(
            f"{self.base_url}/auth/tokens",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 201)
        status, body = self.post_cancel(self.cancel_payload(), token="reader-1")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_cancel_with_mismatched_organization_is_forbidden(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        # The credential's organization differs from the body's.
        status, body = self.post_cancel(
            self.cancel_payload(organizationId="org-2")
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # The credential matches the body but the record belongs to org-1.
        status, body = self.post_cancel(
            {"organizationId": "org-2", "reservationId": "res-1"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # Nothing was released.
        listing = self.list_reservations()
        self.assertEqual(len(listing["reservations"]), 1)

    def test_cancel_unknown_reservation_is_404(self) -> None:
        status, body = self.post_cancel(self.cancel_payload(reservationId="x"))
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "reservation_not_found")

    def test_cancel_media_type_and_json_errors(self) -> None:
        status, body = self.post_cancel(self.cancel_payload(), content_type=None)
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")
        status, body = self.post_cancel(
            self.cancel_payload(), content_type="text/plain"
        )
        self.assertEqual(status, 415)
        status, body = self.post_cancel(b'{"organizationId": ', raw=True)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_cancel_validation_errors_are_422(self) -> None:
        for bad_body in ([], "text", 42, None, True):
            with self.subTest(bad_body=bad_body):
                status, body = self.post_cancel(bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")
        # Missing, extra, blank, and non-string fields.
        for payload in (
            {"organizationId": "org-1"},
            {"reservationId": "res-1"},
            {"organizationId": "org-1", "reservationId": "res-1", "extra": 1},
            {"organizationId": "", "reservationId": "res-1"},
            {"organizationId": "   ", "reservationId": "res-1"},
            {"organizationId": "org-1", "reservationId": ""},
            {"organizationId": "org-1", "reservationId": "  "},
            {"organizationId": 7, "reservationId": "res-1"},
            {"organizationId": "org-1", "reservationId": None},
        ):
            with self.subTest(payload=payload):
                status, body = self.post_cancel(payload)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    # --- branch endpoint errors --------------------------------------------------

    def test_branch_cancel_unknown_branch_is_branch_not_found(self) -> None:
        status, body = self.post_cancel(
            self.cancel_payload(), path="/branches/branch-x/reservations/cancel"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    def test_branch_cancel_foreign_branch_is_forbidden(self) -> None:
        self.assertEqual(
            self.post_reservation(self.reservation_payload())[0], 201
        )
        self.create_snapshot("snap-1")
        self.create_branch("branch-1", "snap-1")
        # A credential for another organization never reaches the inventory.
        status, body = self.post_cancel(
            {"organizationId": "org-2", "reservationId": "res-1"},
            path="/branches/branch-1/reservations/cancel",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_branch_cancel_unknown_reservation_is_404(self) -> None:
        self.create_snapshot("snap-1")
        self.create_branch("branch-1", "snap-1")
        status, body = self.post_cancel(
            self.cancel_payload(), path="/branches/branch-1/reservations/cancel"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "reservation_not_found")


if __name__ == "__main__":
    unittest.main()
