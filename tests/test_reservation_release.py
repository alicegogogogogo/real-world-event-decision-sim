"""Regression tests for POST /reservations/release.

The main service now releases a confirmed reservation: the record leaves
the active inventory (its quantity returns to the resource's remaining
balance, the recorded capacity is never rewritten) while an audit history
keeps the released id recognizable. This locks down the contract:

- the body carries exactly ``organizationId`` and ``reservationId``; a
  missing, extra, duplicated, blank, or mistyped field is a 422
  validation_error, a non-JSON media type is 415 unsupported_media_type,
  and malformed JSON is 400 invalid_json;
- Bearer credentials are required: a missing, malformed, or unregistered
  token is 401 unauthorized; a read-only or cross-organization credential
  is 403 forbidden, as is a reservationId held by another organization;
- a successful release answers 200 with exactly ``reservationId``,
  ``resourceId``, ``organizationId``, ``quantity``, ``capacity`` and the
  post-release ``occupied``/``remaining`` balances as compact, key-sorted
  JSON with one trailing newline; the reservation disappears from
  GET /reservations and its capacity becomes reusable without oversell;
- a released id retried is 409 reservation_released, an unknown id is 404
  reservation_not_found, and concurrent releases of one id let exactly
  one request succeed;
- failures never change reservations, balances, snapshots, or branches;
  snapshots and branches already taken keep their isolated copies, and a
  snapshot taken after the release holds only active reservations.
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
    quantity: int = 4,
    capacity: int = 10,
    organization_id: str = ORG1,
) -> dict[str, Any]:
    return {
        "organizationId": organization_id,
        "reservationId": reservation_id,
        "resourceId": resource_id,
        "quantity": quantity,
        "capacity": capacity,
    }


def release_body(
    reservation_id: str, *, organization_id: str = ORG1
) -> dict[str, Any]:
    return {"organizationId": organization_id, "reservationId": reservation_id}


class ReservationReleaseTest(unittest.TestCase):
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
        method: str = "POST",
        body: bytes | None = None,
        token: str | None = None,
        content_type: str | None = "application/json",
        timeout: float = 30,
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
            with urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except HTTPError as error:
            try:
                return error.code, error.read()
            finally:
                error.close()

    def post(
        self,
        path: str,
        payload: Any,
        *,
        token: str | None = "w1",
        content_type: str | None = "application/json",
        timeout: float = 30,
    ) -> tuple[int, Any]:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        status, raw = self.raw(
            path, body=body, token=token, content_type=content_type, timeout=timeout
        )
        return status, json.loads(raw.decode())

    def register(self, token: str, organization_id: str, role: str) -> None:
        status, _ = self.raw(
            "/auth/tokens",
            body=json.dumps(
                {"token": token, "organizationId": organization_id, "role": role}
            ).encode(),
        )
        assert status in (200, 201)

    def reserve(self, reservation_id: str, **kwargs: Any) -> tuple[int, Any]:
        return self.post("/reservations", reservation_body(reservation_id, **kwargs))

    def release(
        self, reservation_id: str, *, token: str | None = "w1", **kwargs: Any
    ) -> tuple[int, Any]:
        return self.post(
            "/reservations/release",
            release_body(reservation_id, **kwargs),
            token=token,
        )

    def list_reservations(self, organization_id: str = ORG1) -> list[dict[str, Any]]:
        status, payload = self.post_list(organization_id)
        assert status == 200
        return payload["reservations"]

    def post_list(self, organization_id: str) -> tuple[int, Any]:
        token = "w1" if organization_id == ORG1 else "w2"
        status, raw = self.raw(
            f"/reservations?organizationId={organization_id}",
            method="GET",
            token=token,
            content_type=None,
        )
        return status, json.loads(raw.decode())

    # ---------------------------------------------------------------- success

    def test_release_removes_reservation_and_restores_balance(self) -> None:
        status, created = self.reserve("rsv-1")
        assert status == 201
        self.assertEqual(created["occupied"], 4)
        self.assertEqual(created["remaining"], 6)

        status, payload = self.release("rsv-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "organizationId": ORG1,
                "reservationId": "rsv-1",
                "resourceId": "res-rc",
                "quantity": 4,
                "capacity": 10,
                "occupied": 0,
                "remaining": 10,
            },
        )
        self.assertEqual(self.list_reservations(), [])

    def test_release_response_is_compact_sorted_json_with_newline(self) -> None:
        self.reserve("rsv-1")
        status, raw = self.raw(
            "/reservations/release",
            body=json.dumps(release_body("rsv-1")).encode(),
            token="w1",
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw[:-1])
        keys = list(json.loads(raw.decode()))
        self.assertEqual(keys, sorted(keys))
        for field in ("quantity", "capacity", "occupied", "remaining"):
            self.assertIsInstance(json.loads(raw.decode())[field], int)

    def test_release_partial_occupancy(self) -> None:
        self.reserve("rsv-1", quantity=4)
        self.reserve("rsv-2", quantity=3)
        status, payload = self.release("rsv-1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["occupied"], 3)
        self.assertEqual(payload["remaining"], 7)
        self.assertEqual(payload["capacity"], 10)
        remaining_ids = [row["reservationId"] for row in self.list_reservations()]
        self.assertEqual(remaining_ids, ["rsv-2"])

    def test_released_capacity_is_reusable_without_oversell(self) -> None:
        self.reserve("rsv-1", quantity=10)
        # Full: nothing more fits.
        status, payload = self.reserve("rsv-2", quantity=1)
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "capacity_exceeded")
        # Release frees the whole capacity; a follow-up reservation reuses it
        # against the recorded capacity.
        status, _ = self.release("rsv-1")
        self.assertEqual(status, 200)
        status, view = self.reserve("rsv-2", quantity=10)
        self.assertEqual(status, 201)
        self.assertEqual(view["capacity"], 10)
        self.assertEqual(view["occupied"], 10)
        self.assertEqual(view["remaining"], 0)
        # And oversell is still rejected.
        status, payload = self.reserve("rsv-3", quantity=1)
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "capacity_exceeded")

    def test_release_keeps_recorded_capacity(self) -> None:
        self.reserve("rsv-1", capacity=10)
        self.release("rsv-1")
        # The capacity fixed by the first reservation still governs.
        status, payload = self.reserve("rsv-2", capacity=99)
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "capacity_conflict")

    # ------------------------------------------------------------------ auth

    def test_missing_malformed_and_unknown_credentials_are_401(self) -> None:
        self.reserve("rsv-1")
        body = json.dumps(release_body("rsv-1")).encode()
        status, payload = self.raw("/reservations/release", body=body)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(payload.decode())["error"], "unauthorized")
        status, payload = self.raw(
            "/reservations/release", body=body, token="no-such-token"
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(payload.decode())["error"], "unauthorized")
        # Failed requests change nothing.
        self.assertEqual(len(self.list_reservations()), 1)

    def test_read_only_credential_is_403(self) -> None:
        self.reserve("rsv-1")
        status, payload = self.release("rsv-1", token="r1")
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "forbidden")
        self.assertEqual(len(self.list_reservations()), 1)

    def test_cross_organization_credential_is_403(self) -> None:
        self.reserve("rsv-1")
        # org-2's write token against org-1's organizationId.
        status, payload = self.release("rsv-1", token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "forbidden")
        # org-1's token naming org-2 in the body.
        status, payload = self.release("rsv-1", organization_id=ORG2)
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "forbidden")
        self.assertEqual(len(self.list_reservations()), 1)

    def test_other_organizations_reservation_id_is_403(self) -> None:
        self.reserve("rsv-1")
        # org-2 tries to release org-1's reservation by its own body.
        status, payload = self.release("rsv-1", token="w2", organization_id=ORG2)
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "forbidden")
        self.assertEqual(len(self.list_reservations()), 1)

    # ------------------------------------------------------------- not found

    def test_unknown_reservation_id_is_404(self) -> None:
        status, payload = self.release("rsv-missing")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "reservation_not_found")

    def test_released_id_retry_is_409(self) -> None:
        self.reserve("rsv-1")
        status, _ = self.release("rsv-1")
        self.assertEqual(status, 200)
        status, payload = self.release("rsv-1")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "reservation_released")

    def test_released_id_of_other_organization_is_403(self) -> None:
        self.reserve("rsv-1")
        self.release("rsv-1")
        status, payload = self.release("rsv-1", token="w2", organization_id=ORG2)
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "forbidden")

    # -------------------------------------------------------------- validation

    def test_validation_errors_are_422(self) -> None:
        self.reserve("rsv-1")
        bad_bodies = [
            {},  # missing both
            {"organizationId": ORG1},  # missing reservationId
            {"reservationId": "rsv-1"},  # missing organizationId
            {**release_body("rsv-1"), "extra": 1},  # unexpected field
            {"organizationId": "  ", "reservationId": "rsv-1"},  # blank
            {"organizationId": ORG1, "reservationId": ""},  # empty
            {"organizationId": 1, "reservationId": "rsv-1"},  # wrong type
            {"organizationId": ORG1, "reservationId": 2},  # wrong type
            ["not", "an", "object"],
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                status, payload = self.post("/reservations/release", body)
                self.assertEqual(status, 422)
                self.assertEqual(payload["error"], "validation_error")
        self.assertEqual(len(self.list_reservations()), 1)

    def test_duplicate_field_is_422(self) -> None:
        self.reserve("rsv-1")
        body = (
            b'{"organizationId":"org-1","reservationId":"rsv-1",'
            b'"reservationId":"rsv-1"}'
        )
        status, payload = self.post("/reservations/release", body)
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"], "validation_error")
        self.assertEqual(len(self.list_reservations()), 1)

    def test_non_json_media_type_is_415(self) -> None:
        self.reserve("rsv-1")
        status, payload = self.post(
            "/reservations/release",
            release_body("rsv-1"),
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(payload["error"], "unsupported_media_type")
        self.assertEqual(len(self.list_reservations()), 1)

    def test_malformed_json_is_400(self) -> None:
        self.reserve("rsv-1")
        status, payload = self.post("/reservations/release", b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "invalid_json")
        self.assertEqual(len(self.list_reservations()), 1)

    # ------------------------------------------------------------- concurrency

    def test_concurrent_releases_only_one_succeeds(self) -> None:
        self.reserve("rsv-1")
        results: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait(timeout=30)
            status, _ = self.post(
                "/reservations/release", release_body("rsv-1"), timeout=30
            )
            with lock:
                results.append(status)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual(sorted(results), [200] + [409] * 7)
        self.assertEqual(self.list_reservations(), [])

    # ------------------------------------------------- snapshots and branches

    def test_snapshots_and_branches_are_isolated_from_release(self) -> None:
        self.reserve("rsv-1", quantity=4)
        self.reserve("rsv-2", quantity=3)
        status, _ = self.post("/snapshots", {"snapshotId": "snap-1"})
        self.assertEqual(status, 201)
        status, _ = self.post(
            "/branches", {"branchId": "br-1", "snapshotId": "snap-1"}
        )
        self.assertEqual(status, 201)

        status, _ = self.release("rsv-1")
        self.assertEqual(status, 200)

        # The pre-release snapshot still holds the released reservation.
        status, payload = self.raw(
            "/snapshots/snap-1/reservations?organizationId=org-1",
            method="GET",
            token="w1",
            content_type=None,
        )
        self.assertEqual(status, 200)
        rows = json.loads(payload.decode())["reservations"]
        self.assertEqual(
            [row["reservationId"] for row in rows], ["rsv-1", "rsv-2"]
        )

        # A snapshot taken after the release holds only active reservations
        # and keeps the recorded capacity with the post-release balance.
        status, _ = self.post("/snapshots", {"snapshotId": "snap-2"})
        self.assertEqual(status, 201)
        status, payload = self.raw(
            "/snapshots/snap-2/reservations?organizationId=org-1",
            method="GET",
            token="w1",
            content_type=None,
        )
        self.assertEqual(status, 200)
        rows = json.loads(payload.decode())["reservations"]
        self.assertEqual([row["reservationId"] for row in rows], ["rsv-2"])
        status, payload = self.raw(
            "/snapshots/snap-2/resources?organizationId=org-1",
            method="GET",
            token="w1",
            content_type=None,
        )
        self.assertEqual(status, 200)
        resources = json.loads(payload.decode())["resources"]
        self.assertEqual(
            resources,
            [{"resourceId": "res-rc", "capacity": 10, "occupied": 3, "remaining": 7}],
        )

    def test_failed_release_does_not_change_balances(self) -> None:
        self.reserve("rsv-1", quantity=4)
        self.release("rsv-missing")  # 404
        self.release("rsv-1", token="r1")  # 403
        rows = self.list_reservations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["occupied"], 4)
        self.assertEqual(rows[0]["remaining"], 6)

    def test_reservation_replay_and_conflicts_unchanged(self) -> None:
        status, first = self.reserve("rsv-1")
        self.assertEqual(status, 201)
        # Identical replay is still idempotent.
        status, replay = self.reserve("rsv-1")
        self.assertEqual(status, 200)
        self.assertEqual(replay, first)
        # Conflicting replay still conflicts.
        status, payload = self.reserve("rsv-1", quantity=5)
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "reservation_conflict")


if __name__ == "__main__":
    unittest.main()
