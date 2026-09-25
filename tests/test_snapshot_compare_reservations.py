"""Regression tests for POST /snapshots/compare/reservations.

Branch-dimension comparisons already existed; this locks down the new
read-only snapshot-dimension entry point that aligns two snapshots'
captured reservations by reservationId:

- identifiers only on one side land in leftOnly/rightOnly, identical shared
  reservations in same, and disagreeing shared reservations in diff with
  the mismatched field names (drawn only from organizationId, resourceId,
  quantity, capacity);
- each group is code-point sorted and paired with a ``<group>Count`` key,
  and the response is compact, key-sorted JSON with one trailing newline;
- both-snapshots-empty yields four empty groups and zero counts, the same
  snapshot name on both sides is fully same, and identical submissions are
  byte-for-byte stable;
- 401 / 403 / 404 (snapshot_not_found) / 422 / 415 / 400 follow the fixed
  ordering, left is always resolved before right, and nothing is ever
  written or implicitly created, including across organizations.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import compare_reservation_sets, create_server

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


class SnapshotReservationCompareTest(unittest.TestCase):
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
        self,
        reservation_id: str,
        token: str = "w1",
        **kwargs: Any,
    ) -> None:
        organization_id = ORG1 if token in ("w1", "r1") else ORG2
        status, _ = self.call(
            "/reservations",
            method="POST",
            token=token,
            payload=reservation_body(
                reservation_id, organization_id=organization_id, **kwargs
            ),
        )
        self.assertEqual(status, 201)

    def take_snapshot(self, snapshot_id: str, token: str = "w1") -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def compare_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": ORG1,
            "left": "left",
            "right": "right",
        }
        payload.update(overrides)
        return payload

    def compare(
        self, payload: dict[str, Any] | None = None, *, token: str | None = "w1"
    ) -> tuple[int, Any]:
        return self.call(
            "/snapshots/compare/reservations",
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    # ------------------------------------------------------------- happy paths

    def test_reservations_align_by_identifier_into_groups(self) -> None:
        # Snapshots capture the main state at creation time, so an earlier
        # snapshot holds a subset of a later one's reservations.
        self.add_reservation("res-a")
        self.take_snapshot("right")
        self.add_reservation("res-b")
        self.add_reservation("res-c", resource_id="pool-c", quantity=2)
        self.take_snapshot("left")

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-b", "res-c"])
        self.assertEqual(body["leftOnlyCount"], 2)
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["rightOnlyCount"], 0)
        self.assertEqual(body["same"], ["res-a"])
        self.assertEqual(body["sameCount"], 1)
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["diffCount"], 0)

    def test_diff_groups_and_field_names_in_the_compare_function(self) -> None:
        # Two snapshots of the same service can never hold the same
        # reservationId with different fields (the inventory rejects it), so
        # the diff arm is locked down directly against the shared compare
        # function the entry point uses.
        left = {
            "res-qc": reservation_body(
                "res-qc", resource_id="pool", quantity=2, capacity=10
            ),
            "res-r": reservation_body("res-r", resource_id="pool-a"),
            "res-same": reservation_body("res-same"),
        }
        right = {
            "res-qc": reservation_body(
                "res-qc", resource_id="pool", quantity=5, capacity=20
            ),
            "res-r": reservation_body("res-r", resource_id="pool-b"),
            "res-same": reservation_body("res-same"),
        }
        result = compare_reservation_sets(left, right)
        self.assertEqual(result["same"], ["res-same"])
        self.assertEqual(
            result["diff"],
            [
                {"reservationId": "res-qc", "fields": ["capacity", "quantity"]},
                {"reservationId": "res-r", "fields": ["resourceId"]},
            ],
        )
        self.assertEqual(result["diffCount"], 2)

    def test_same_identifiers_sort_in_code_point_order(self) -> None:
        for reservation_id in ("res-b", "res-A", "res-a", "res-1"):
            self.add_reservation(reservation_id)
        self.take_snapshot("left")
        self.take_snapshot("right")
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["same"], ["res-1", "res-A", "res-a", "res-b"])

    def test_left_only_identifiers_sort_in_code_point_order(self) -> None:
        self.take_snapshot("right")
        for reservation_id in ("res-b", "res-A", "res-a", "res-1"):
            self.add_reservation(reservation_id)
        self.take_snapshot("left")
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["leftOnly"], ["res-1", "res-A", "res-a", "res-b"]
        )
        self.assertEqual(body["leftOnlyCount"], 4)
        self.assertEqual(body["rightOnly"], [])

    def test_same_snapshot_on_both_sides_is_legal_and_fully_same(self) -> None:
        self.add_reservation("res-1")
        self.add_reservation("res-2", resource_id="other")
        self.take_snapshot("left")
        payload = self.compare_payload(right="left")
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-1", "res-2"])
        self.assertEqual(body["sameCount"], 2)
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["diffCount"], 0)

    def test_empty_snapshots_yield_empty_groups_and_zero_counts(self) -> None:
        self.take_snapshot("left")
        self.take_snapshot("right")
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "leftOnly": [],
                "leftOnlyCount": 0,
                "rightOnly": [],
                "rightOnlyCount": 0,
                "same": [],
                "sameCount": 0,
                "diff": [],
                "diffCount": 0,
            },
        )

    # ----------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.take_snapshot("right")
        self.add_reservation("res-1")
        self.take_snapshot("left")

        raw_body = json.dumps(self.compare_payload()).encode()
        status, raw = self.raw(
            "/snapshots/compare/reservations",
            method="POST",
            body=raw_body,
            token="w1",
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Counts stay integers.
        self.assertIn(b'"leftOnlyCount":1', raw)
        self.assertIn(b'"diffCount":0', raw)
        # Keys are sorted by code point at every level.
        body = json.loads(raw)
        self.assertEqual(
            list(body),
            [
                "diff",
                "diffCount",
                "leftOnly",
                "leftOnlyCount",
                "rightOnly",
                "rightOnlyCount",
                "same",
                "sameCount",
            ],
        )

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.add_reservation("res-1")
        self.take_snapshot("right")
        self.add_reservation("res-2", quantity=2)
        self.add_reservation("res-3")
        self.take_snapshot("left")
        raw_body = json.dumps(self.compare_payload()).encode()
        raws = [
            self.raw(
                "/snapshots/compare/reservations",
                method="POST",
                body=raw_body,
                token="w1",
            )[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # -------------------------------------------------------------- read-only

    def test_compare_is_read_only(self) -> None:
        self.add_reservation("res-1")
        self.take_snapshot("left")
        self.add_reservation("res-2")
        self.take_snapshot("right")

        def state() -> tuple[Any, Any, Any]:
            snapshots = self.call("/snapshots")[1]
            reservations = self.call("/reservations?organizationId=org-1")[1]
            alerts = self.call("/alerts?organizationId=org-1")[1]
            return snapshots, reservations, alerts

        before = state()
        for _ in range(3):
            status, _ = self.compare()
            self.assertEqual(status, 200)
        after = state()
        self.assertEqual(before, after)

    def test_failed_compare_creates_no_snapshot(self) -> None:
        self.take_snapshot("left")
        self.take_snapshot("right")
        for payload in (
            self.compare_payload(left="ghost"),
            self.compare_payload(right="ghost"),
            self.compare_payload(left=""),
        ):
            status, _ = self.compare(payload)
            self.assertIn(status, (404, 422))
        snapshots = self.call("/snapshots")[1]["snapshots"]
        self.assertEqual(
            [entry["snapshotId"] for entry in snapshots], ["left", "right"]
        )

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_compare(self) -> None:
        self.take_snapshot("right")
        self.add_reservation("res-1")
        self.take_snapshot("left")
        status, body = self.compare(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-1"])

    def test_missing_or_unregistered_credential_is_401(self) -> None:
        self.take_snapshot("left")
        self.take_snapshot("right")
        raw_body = json.dumps(self.compare_payload()).encode()
        status, body = self.raw(
            "/snapshots/compare/reservations",
            method="POST",
            body=raw_body,
            token=None,
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        status, body = self.raw(
            "/snapshots/compare/reservations",
            method="POST",
            body=raw_body,
            token="forged",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.take_snapshot("left")
        self.take_snapshot("right")
        self.take_snapshot("foreign", token="w2")

        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the snapshot names do not exist anywhere.
        status, body = self.compare(
            self.compare_payload(left="nope", right="also-nope"), token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

        # An ORG1 credential naming ORG2's snapshot cannot read it.
        status, body = self.compare(
            self.compare_payload(left="foreign", right="left"), token="w1"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        status, body = self.compare(
            self.compare_payload(left="left", right="foreign"), token="w1"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_snapshot_existence_checked_left_then_right(self) -> None:
        self.take_snapshot("left")
        self.take_snapshot("right")
        self.take_snapshot("foreign", token="w2")

        # Unknown names are 404 snapshot_not_found; no snapshot is created.
        status, body = self.compare(self.compare_payload(left="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        status, body = self.compare(self.compare_payload(right="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

        # Left is decided first: a missing left outranks a foreign right.
        status, body = self.compare(
            self.compare_payload(left="ghost", right="foreign")
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

        # ...and a foreign left outranks a missing right.
        status, body = self.compare(
            self.compare_payload(left="foreign", right="ghost")
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_multi_organization_snapshots_compare_within_their_owner_only(
        self,
    ) -> None:
        # ORG1: one reservation before "right", one more before "left".
        self.add_reservation("res-1")
        self.take_snapshot("right")
        self.add_reservation("res-2", resource_id="pool-2", quantity=2)
        self.take_snapshot("left")

        # ORG2: its own reservations and snapshots under different names.
        self.add_reservation("res-9", token="w2", resource_id="pool-9")
        self.take_snapshot("foreign-a", token="w2")
        self.add_reservation("res-8", token="w2", resource_id="pool-8")
        self.take_snapshot("foreign-b", token="w2")

        # ORG2 compares its own two snapshots and sees only ORG2 captures.
        payload = {
            "organizationId": ORG2,
            "left": "foreign-b",
            "right": "foreign-a",
        }
        status, body = self.call(
            "/snapshots/compare/reservations",
            method="POST",
            payload=payload,
            token="w2",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-8"])
        self.assertEqual(body["same"], ["res-9"])
        self.assertEqual(body["diff"], [])

        # ORG1's comparison is unchanged and never sees ORG2 reservations.
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-2"])
        self.assertEqual(body["same"], ["res-1"])
        self.assertEqual(body["diff"], [])

    # ----------------------------------------------------------- 422 / 415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        self.take_snapshot("left")
        self.take_snapshot("right")
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "organizationId"},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {**valid, "extra": 1},
            {**valid, "windowSize": 60},  # window-compare fields not allowed
            {**valid, "snapshotId": "x"},  # snapshot-creation field not allowed
            {**valid, "left": ""},
            {**valid, "right": "   "},
            {**valid, "organizationId": ""},
            {**valid, "left": 7},
            {**valid, "right": None},
            {**valid, "organizationId": ["org-1"]},
            [],
            "x",
            42,
            True,
            None,
        ]
        for bad_payload in bad_payloads:
            with self.subTest(bad_payload=bad_payload):
                status, parsed = self.call(
                    "/snapshots/compare/reservations",
                    method="POST",
                    raw_body=json.dumps(bad_payload).encode(),
                    token="w1",
                )
                self.assertEqual(status, 422)
                self.assertEqual(parsed["error"], "validation_error")

        # No failed validation created or altered a snapshot.
        snapshots = self.call("/snapshots")[1]["snapshots"]
        self.assertEqual(
            [entry["snapshotId"] for entry in snapshots], ["left", "right"]
        )
        self.assertEqual(
            [entry["reservations"] for entry in snapshots], [0, 0]
        )

    def test_media_type_and_json_errors(self) -> None:
        self.take_snapshot("left")
        self.take_snapshot("right")
        raw_body = json.dumps(self.compare_payload()).encode()

        status, body = self.call(
            "/snapshots/compare/reservations",
            method="POST",
            raw_body=raw_body,
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/snapshots/compare/reservations",
            method="POST",
            raw_body=raw_body,
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/snapshots/compare/reservations",
            method="POST",
            raw_body=b'{"left": ',
            content_type="application/json",
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

        # Nothing failed open into the store.
        snapshots = self.call("/snapshots")[1]["snapshots"]
        self.assertEqual(
            [entry["snapshotId"] for entry in snapshots], ["left", "right"]
        )


if __name__ == "__main__":
    unittest.main()
