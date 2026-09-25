"""Regression tests for POST /branches/compare/reservations.

The window-level and event-level branch comparisons already existed; this
locks down the new read-only entry point that aligns two branches'
reservations by reservationId:

- identifiers only on one side land in leftOnly/rightOnly, shared
  reservations whose organizationId/resourceId/quantity/capacity all match
  land in same, and disagreeing shared reservations land in diff with the
  mismatched field names drawn only from those four keys;
- each group is code-point sorted and paired with a ``<group>Count`` key,
  and the response is compact, key-sorted JSON with one trailing newline;
- both-branches-empty yields four empty groups and zero counts, and
  identical submissions are byte-for-byte stable;
- 401 / 403 / 404 (branch_not_found) / 422 / 415 / 400 follow the fixed
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

from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"


def reservation_body(
    reservation_id: str,
    organization_id: str = ORG1,
    resource_id: str = "r-a",
    quantity: int = 1,
    capacity: int = 10,
) -> dict[str, Any]:
    return {
        "organizationId": organization_id,
        "reservationId": reservation_id,
        "resourceId": resource_id,
        "quantity": quantity,
        "capacity": capacity,
    }


class BranchReservationCompareTest(unittest.TestCase):
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

    def fork_branches(
        self,
        *,
        snapshot_id: str = "s1",
        branches: tuple[str, ...] = ("left", "right"),
        token: str = "w1",
    ) -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)
        for branch_id in branches:
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
            payload=reservation_body(reservation_id, organization_id, **kwargs),
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
            "/branches/compare/reservations",
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    # ------------------------------------------------------------- happy paths

    def test_reservations_align_by_identifier_into_four_groups(self) -> None:
        self.fork_branches()
        # Shared and identical.
        self.add_branch_reservation("left", "res-same", quantity=2)
        self.add_branch_reservation("right", "res-same", quantity=2)
        # Shared but quantity and capacity differ (own resource on each
        # side, so each branch fixes that resource's capacity itself).
        self.add_branch_reservation(
            "left", "res-diff", resource_id="r-d", quantity=2, capacity=10
        )
        self.add_branch_reservation(
            "right", "res-diff", resource_id="r-d", quantity=3, capacity=12
        )
        # Shared but only the resource differs (distinct resource ids, so no
        # capacity coupling between the two branches).
        self.add_branch_reservation("left", "res-moved", resource_id="r-a")
        self.add_branch_reservation("right", "res-moved", resource_id="r-b")
        # One side each.
        self.add_branch_reservation("left", "res-left", resource_id="r-l")
        self.add_branch_reservation("right", "res-right", resource_id="r-r")

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-left"])
        self.assertEqual(body["leftOnlyCount"], 1)
        self.assertEqual(body["rightOnly"], ["res-right"])
        self.assertEqual(body["rightOnlyCount"], 1)
        self.assertEqual(body["same"], ["res-same"])
        self.assertEqual(body["sameCount"], 1)
        self.assertEqual(
            body["diff"],
            [
                {"reservationId": "res-diff", "fields": ["capacity", "quantity"]},
                {"reservationId": "res-moved", "fields": ["resourceId"]},
            ],
        )
        self.assertEqual(body["diffCount"], 2)

    def test_diff_fields_come_only_from_the_four_compared_keys(self) -> None:
        self.fork_branches()
        self.add_branch_reservation("left", "res-1", quantity=1, capacity=5)
        self.add_branch_reservation("right", "res-1", quantity=2, capacity=5)
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["diff"], [{"reservationId": "res-1", "fields": ["quantity"]}]
        )
        # The compared keys are exactly the four documented English names.
        self.assertEqual(
            set(body["diff"][0]["fields"]) <=
            {"organizationId", "resourceId", "quantity", "capacity"},
            True,
        )

    def test_identifiers_sort_in_code_point_order(self) -> None:
        self.fork_branches()
        for index, reservation_id in enumerate(("res-b", "res-A", "res-a", "res-1")):
            self.add_branch_reservation(
                "left", reservation_id, resource_id=f"r-{index}"
            )
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-1", "res-A", "res-a", "res-b"])
        self.assertEqual(body["leftOnlyCount"], 4)
        self.assertEqual(body["rightOnly"], [])

    def test_same_branch_on_both_sides_is_legal_and_fully_same(self) -> None:
        self.fork_branches(branches=("left",))
        self.add_branch_reservation("left", "res-1", quantity=1)
        self.add_branch_reservation("left", "res-2", quantity=2)
        payload = self.compare_payload(right="left")
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-1", "res-2"])
        self.assertEqual(body["sameCount"], 2)
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["diffCount"], 0)

    def test_empty_branches_yield_empty_groups_and_zero_counts(self) -> None:
        self.fork_branches()
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
        self.fork_branches()
        self.add_branch_reservation("left", "res-1")

        raw_body = json.dumps(self.compare_payload()).encode()
        status, raw = self.raw(
            "/branches/compare/reservations",
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

    def test_diff_entry_keys_are_sorted(self) -> None:
        self.fork_branches()
        self.add_branch_reservation("left", "res-1", quantity=1)
        self.add_branch_reservation("right", "res-1", quantity=2)
        raw_body = json.dumps(self.compare_payload()).encode()
        status, raw = self.raw(
            "/branches/compare/reservations",
            method="POST",
            body=raw_body,
            token="w1",
        )
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual(list(body["diff"][0]), ["fields", "reservationId"])

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.fork_branches()
        self.add_branch_reservation("left", "res-1", quantity=1)
        self.add_branch_reservation("left", "res-2", quantity=2)
        self.add_branch_reservation("right", "res-2", quantity=3)
        self.add_branch_reservation("right", "res-3", resource_id="r-b")
        raw_body = json.dumps(self.compare_payload()).encode()
        raws = [
            self.raw(
                "/branches/compare/reservations",
                method="POST",
                body=raw_body,
                token="w1",
            )[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # -------------------------------------------------------------- read-only

    def test_compare_is_read_only(self) -> None:
        self.fork_branches()
        self.add_branch_reservation("left", "res-1")
        self.add_branch_reservation("right", "res-2")

        def summaries() -> tuple[Any, Any]:
            left = self.call("/branches/left")[1]
            right = self.call("/branches/right")[1]
            return left, right

        before = summaries()
        for _ in range(3):
            status, _ = self.compare()
            self.assertEqual(status, 200)
        after = summaries()
        self.assertEqual(before, after)

        # Branch balances, main-service state, and alerts are untouched.
        for branch in ("left", "right"):
            status, reservations = self.call(
                f"/branches/{branch}/reservations?organizationId=org-1"
            )
            self.assertEqual(status, 200)
            self.assertEqual(len(reservations["reservations"]), 1)
        status, main_reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(main_reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    def test_failed_compare_creates_no_branch(self) -> None:
        self.fork_branches()
        for payload in (
            self.compare_payload(left="ghost"),
            self.compare_payload(right="ghost"),
            self.compare_payload(left=""),
        ):
            status, _ = self.compare(payload)
            self.assertIn(status, (404, 422))
        self.assertEqual(self.call("/branches/ghost")[0], 404)

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_compare(self) -> None:
        self.fork_branches()
        self.add_branch_reservation("left", "res-1")
        status, body = self.compare(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-1"])

    def test_missing_or_unregistered_credential_is_401(self) -> None:
        self.fork_branches()
        raw_body = json.dumps(self.compare_payload()).encode()
        status, body = self.raw(
            "/branches/compare/reservations",
            method="POST",
            body=raw_body,
            token=None,
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        status, body = self.raw(
            "/branches/compare/reservations",
            method="POST",
            body=raw_body,
            token="forged",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_branch_lookup(self) -> None:
        self.fork_branches()
        self.fork_branches(snapshot_id="s2", branches=("foreign",), token="w2")

        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the branch names do not exist anywhere.
        status, body = self.compare(
            self.compare_payload(left="nope", right="also-nope"), token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

        # An ORG1 credential naming ORG2's branch cannot read it.
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

    def test_branch_existence_checked_left_then_right(self) -> None:
        self.fork_branches()
        self.fork_branches(snapshot_id="s2", branches=("foreign",), token="w2")

        # Unknown names are 404 branch_not_found; no branch is created.
        status, body = self.compare(self.compare_payload(left="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")
        status, body = self.compare(self.compare_payload(right="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

        # Left is decided first: a missing left outranks a foreign right.
        status, body = self.compare(
            self.compare_payload(left="ghost", right="foreign")
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

        # ...and a foreign left outranks a missing right.
        status, body = self.compare(
            self.compare_payload(left="foreign", right="ghost")
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_multi_organization_branches_compare_within_their_owner_only(
        self,
    ) -> None:
        self.fork_branches()
        self.fork_branches(
            snapshot_id="s2", branches=("foreign-a", "foreign-b"), token="w2"
        )
        self.add_branch_reservation("left", "res-1", quantity=1)
        self.add_branch_reservation("foreign-a", "res-1", quantity=1, token="w2")
        self.add_branch_reservation("foreign-a", "res-2", quantity=2, token="w2")
        self.add_branch_reservation("foreign-b", "res-2", quantity=3, token="w2")

        # ORG2 compares its own two branches and sees only ORG2 reservations.
        payload = {
            "organizationId": ORG2,
            "left": "foreign-a",
            "right": "foreign-b",
        }
        status, body = self.call(
            "/branches/compare/reservations",
            method="POST",
            payload=payload,
            token="w2",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-1"])
        self.assertEqual(body["same"], [])
        self.assertEqual(
            body["diff"], [{"reservationId": "res-2", "fields": ["quantity"]}]
        )

        # ORG1's comparison is unchanged and never sees ORG2 reservations.
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-1"])
        self.assertEqual(body["same"], [])
        self.assertEqual(body["diff"], [])

    def test_cross_organization_reservation_ids_never_leak(self) -> None:
        # The same reservationId committed under ORG1 and ORG2 branches stays
        # invisible across the organization boundary.
        self.fork_branches()
        self.fork_branches(snapshot_id="s2", branches=("foreign",), token="w2")
        self.add_branch_reservation("left", "res-shared", quantity=1)
        self.add_branch_reservation("foreign", "res-shared", quantity=9, token="w2")

        status, body = self.compare(self.compare_payload(right="left"))
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-shared"])
        self.assertEqual(body["diff"], [])

        status, body = self.compare(
            {"organizationId": ORG2, "left": "foreign", "right": "foreign"},
            token="w2",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-shared"])
        self.assertEqual(body["diff"], [])

    # ----------------------------------------------------------- 422 / 415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        self.fork_branches()
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "organizationId"},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {**valid, "extra": 1},
            {**valid, "type": "incident.created"},  # window-compare fields
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
                    "/branches/compare/reservations",
                    method="POST",
                    raw_body=json.dumps(bad_payload).encode(),
                    token="w1",
                )
                self.assertEqual(status, 422)
                self.assertEqual(parsed["error"], "validation_error")

        # No failed validation created a branch or altered the real ones.
        self.assertEqual(self.call("/branches/left")[1]["reservations"], 0)
        self.assertEqual(self.call("/branches/right")[1]["reservations"], 0)

    def test_media_type_and_json_errors(self) -> None:
        self.fork_branches()
        raw_body = json.dumps(self.compare_payload()).encode()

        status, body = self.call(
            "/branches/compare/reservations",
            method="POST",
            raw_body=raw_body,
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/branches/compare/reservations",
            method="POST",
            raw_body=raw_body,
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/branches/compare/reservations",
            method="POST",
            raw_body=b'{"left": ',
            content_type="application/json",
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

        # Nothing failed open into the stores.
        self.assertEqual(self.call("/branches/left")[0], 200)


if __name__ == "__main__":
    unittest.main()
