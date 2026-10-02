"""POST /reservations/release regression tests.

Covers the main-service reservation release: a confirmed reservation leaves
the active inventory (its quantity stops counting against the resource) while
its record is retained as audit history, so a retry conflicts instead of
releasing twice. The resource's recorded capacity survives the release and
later reservations reuse the freed balance. Snapshots and branches are
isolated copies: a release never rewrites them, and a snapshot taken after a
release holds only the still-active reservations.
"""

from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"


def reservation_body(
    reservation_id: str = "res-1", organization_id: str = ORG1, **more: Any
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "organizationId": organization_id,
        "reservationId": reservation_id,
        "resourceId": "r-a",
        "quantity": 2,
        "capacity": 5,
    }
    body.update(more)
    return body


def release_body(
    reservation_id: str = "res-1", organization_id: str = ORG1, **more: Any
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "organizationId": organization_id,
        "reservationId": reservation_id,
    }
    body.update(more)
    return body


class ReservationReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.register("tok-write-1", ORG1, "write")
        self.register("tok-read-1", ORG1, "read")
        self.register("tok-write-2", ORG2, "write")

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
        content_type: str | None = "application/json",
        authorization: str | None = None,
    ) -> tuple[int, bytes]:
        headers: dict[str, str] = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if authorization is not None:
            headers["Authorization"] = authorization
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
        token: str | None = "tok-write-1",
        raw_body: bytes | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, Any]:
        if raw_body is not None:
            body = raw_body
        elif payload is not None:
            body = json.dumps(payload).encode()
        else:
            body = None
        authorization = f"Bearer {token}" if token is not None else None
        status, raw = self.raw(
            path,
            method=method,
            body=body,
            content_type=content_type,
            authorization=authorization,
        )
        return status, json.loads(raw)

    def register(self, token: str, organization_id: str, role: str) -> None:
        status, _ = self.call(
            "/auth/tokens",
            method="POST",
            payload={"token": token, "organizationId": organization_id, "role": role},
            token=None,
        )
        assert status in (200, 201)

    def reserve(self, **overrides: Any) -> tuple[int, Any]:
        return self.call("/reservations", method="POST", payload=reservation_body(**overrides))

    def release(self, **overrides: Any) -> tuple[int, Any]:
        return self.call(
            "/reservations/release", method="POST", payload=release_body(**overrides)
        )

    def list_reservations(self, organization_id: str = ORG1) -> list[dict[str, Any]]:
        status, body = self.call(f"/reservations?organizationId={organization_id}")
        self.assertEqual(status, 200)
        return body["reservations"]

    # ------------------------------------------------------------- happy path

    def test_release_returns_200_with_post_release_balances(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        self.assertEqual(self.reserve(reservationId="res-2", quantity=1)[0], 201)

        status, body = self.release()
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": ORG1,
                "reservationId": "res-1",
                "resourceId": "r-a",
                "quantity": 2,
                "capacity": 5,
                "occupied": 1,
                "remaining": 4,
            },
        )

    def test_release_response_is_compact_sorted_json_with_newline(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        status, raw = self.raw(
            "/reservations/release",
            method="POST",
            body=json.dumps(release_body()).encode(),
            authorization="Bearer tok-write-1",
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(
            raw[:-1].decode(),
            '{"capacity":5,"occupied":0,"organizationId":"org-1","quantity":2,'
            '"remaining":5,"reservationId":"res-1","resourceId":"r-a"}',
        )

    def test_released_reservation_leaves_the_listing(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        self.assertEqual(self.reserve(reservationId="res-2", quantity=1)[0], 201)
        self.assertEqual(self.release()[0], 200)

        reservations = self.list_reservations()
        self.assertEqual([r["reservationId"] for r in reservations], ["res-2"])
        # The remaining reservation now sees the freed balance.
        self.assertEqual(reservations[0]["occupied"], 1)
        self.assertEqual(reservations[0]["remaining"], 4)

    def test_release_frees_capacity_for_later_reservations(self) -> None:
        self.assertEqual(self.reserve(quantity=5, capacity=5)[0], 201)
        # The resource is fully booked: a further reservation cannot commit.
        status, body = self.reserve(reservationId="res-2", quantity=1, capacity=5)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_exceeded")

        self.assertEqual(self.release()[0], 200)

        # The recorded capacity is reused verbatim and the freed balance is
        # available again.
        status, body = self.reserve(reservationId="res-2", quantity=4, capacity=5)
        self.assertEqual(status, 201)
        self.assertEqual(body["capacity"], 5)
        self.assertEqual(body["occupied"], 4)
        self.assertEqual(body["remaining"], 1)
        # ...but overselling the freed balance is still rejected.
        status, body = self.reserve(reservationId="res-3", quantity=2, capacity=5)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_exceeded")

    def test_release_keeps_the_recorded_capacity(self) -> None:
        self.assertEqual(self.reserve(quantity=5, capacity=5)[0], 201)
        self.assertEqual(self.release()[0], 200)
        # A different declared capacity still conflicts with the recorded one.
        status, body = self.reserve(reservationId="res-2", quantity=1, capacity=9)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_conflict")

    # --------------------------------------------------------------- conflicts

    def test_releasing_twice_conflicts(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        self.assertEqual(self.release()[0], 200)

        status, body = self.release()
        self.assertEqual(status, 409)
        self.assertEqual(body, {"error": "reservation_released"})
        # The failed retry changed nothing: the listing stays empty.
        self.assertEqual(self.list_reservations(), [])

    def test_unknown_reservation_is_404(self) -> None:
        status, body = self.release(reservationId="res-missing")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "reservation_not_found"})

    def test_other_organizations_reservation_is_forbidden(self) -> None:
        self.assertEqual(self.reserve()[0], 201)

        # org-2's credential with org-2's body: the id belongs to org-1.
        status, body = self.call(
            "/reservations/release",
            method="POST",
            payload=release_body(organization_id=ORG2),
            token="tok-write-2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        # org-1's credential naming org-2 in the body: organization mismatch.
        status, body = self.release(organization_id=ORG2)
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        # Neither attempt touched the reservation.
        self.assertEqual(len(self.list_reservations()), 1)

    def test_concurrent_releases_commit_exactly_once(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.release(), range(16)))
        statuses = sorted(status for status, _ in results)
        self.assertEqual(statuses.count(200), 1)
        self.assertEqual(statuses.count(409), 15)
        for status, body in results:
            if status == 409:
                self.assertEqual(body["error"], "reservation_released")
        self.assertEqual(self.list_reservations(), [])

    # ------------------------------------------------------------------- auth

    def test_missing_credential_is_401(self) -> None:
        status, body = self.call(
            "/reservations/release", method="POST", payload=release_body(), token=None
        )
        self.assertEqual(status, 401)
        self.assertEqual(body, {"error": "unauthorized"})

    def test_malformed_credential_is_401(self) -> None:
        status, raw = self.raw(
            "/reservations/release",
            method="POST",
            body=json.dumps(release_body()).encode(),
            authorization="tok-write-1",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

    def test_unregistered_credential_is_401(self) -> None:
        status, body = self.call(
            "/reservations/release",
            method="POST",
            payload=release_body(),
            token="tok-unknown",
        )
        self.assertEqual(status, 401)
        self.assertEqual(body, {"error": "unauthorized"})

    def test_read_role_is_403_and_changes_nothing(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        status, body = self.call(
            "/reservations/release",
            method="POST",
            payload=release_body(),
            token="tok-read-1",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        self.assertEqual(len(self.list_reservations()), 1)

    # -------------------------------------------------------------- validation

    def test_non_json_media_type_is_415(self) -> None:
        status, body = self.call(
            "/reservations/release",
            method="POST",
            raw_body=json.dumps(release_body()).encode(),
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body, {"error": "unsupported_media_type"})

    def test_invalid_json_is_400(self) -> None:
        status, body = self.call(
            "/reservations/release", method="POST", raw_body=b"{not json"
        )
        self.assertEqual(status, 400)
        self.assertEqual(body, {"error": "invalid_json"})

    def test_validation_failures_are_422(self) -> None:
        cases = [
            ["not", "an", "object"],  # non-object body
            {"organizationId": ORG1},  # missing reservationId
            {"reservationId": "res-1"},  # missing organizationId
            release_body(extra="nope"),  # unexpected field
            release_body(organizationId="   "),  # blank organizationId
            release_body(reservationId=""),  # empty reservationId
            release_body(reservationId=7),  # non-string reservationId
            release_body(organizationId=None),  # non-string organizationId
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                status, body = self.call(
                    "/reservations/release", method="POST", payload=payload
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_duplicate_field_is_422(self) -> None:
        raw = b'{"organizationId":"org-1","reservationId":"res-1","reservationId":"res-1"}'
        status, body = self.call(
            "/reservations/release", method="POST", raw_body=raw
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_failed_release_changes_nothing(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        before = self.list_reservations()

        # A 404 attempt and a 422 attempt leave the inventory untouched.
        self.assertEqual(self.release(reservationId="res-missing")[0], 404)
        self.assertEqual(
            self.call("/reservations/release", method="POST", payload={})[0], 422
        )

        self.assertEqual(self.list_reservations(), before)
        # The reservation is still live: an identical replay returns 200.
        status, body = self.reserve()
        self.assertEqual(status, 200)
        self.assertEqual(body["occupied"], 2)
        self.assertEqual(body["remaining"], 3)

    # ---------------------------------------------- snapshots and branches

    def test_snapshot_taken_after_release_holds_only_active_reservations(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        self.assertEqual(self.reserve(reservationId="res-2", quantity=1)[0], 201)
        self.assertEqual(self.release()[0], 200)

        status, body = self.call(
            "/snapshots", method="POST", payload={"snapshotId": "snap-after"}
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["reservations"], 1)
        self.assertEqual(body["resources"], 1)

        status, body = self.call(
            f"/snapshots/snap-after/reservations?organizationId={ORG1}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [r["reservationId"] for r in body["reservations"]], ["res-2"]
        )

    def test_release_does_not_rewrite_existing_snapshots(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        status, body = self.call(
            "/snapshots", method="POST", payload={"snapshotId": "snap-before"}
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["reservations"], 1)

        self.assertEqual(self.release()[0], 200)

        # The pre-release snapshot still holds the released reservation.
        status, body = self.call(
            f"/snapshots/snap-before/reservations?organizationId={ORG1}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [r["reservationId"] for r in body["reservations"]], ["res-1"]
        )

    def test_restart_clears_the_release_audit(self) -> None:
        self.assertEqual(self.reserve()[0], 201)
        self.assertEqual(self.release()[0], 200)

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{fresh.server_port}"
            payload = json.dumps(release_body()).encode()
            # Register a credential on the fresh instance, then release: the
            # id is unknown there, not "already released".
            register = Request(
                f"{base}/auth/tokens",
                data=json.dumps(
                    {"token": "tok-fresh", "organizationId": ORG1, "role": "write"}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(register, timeout=5) as response:
                self.assertEqual(response.status, 201)
            request = Request(
                f"{base}/reservations/release",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer tok-fresh",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    status, body = response.status, json.load(response)
            except HTTPError as error:
                try:
                    status, body = error.code, json.load(error)
                finally:
                    error.close()
            self.assertEqual(status, 404)
            self.assertEqual(body, {"error": "reservation_not_found"})
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
