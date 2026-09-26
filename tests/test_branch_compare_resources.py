"""Regression tests for POST /branches/compare/resources.

The branch comparisons already covered window decisions, events, and
reservations, and the snapshot comparisons covered resource balances; this
locks down the new read-only entry point that extends the resource-balance
dimension to branches and aligns two branches' resource capacities and
occupancy by resourceId:

- the ``resources`` array carries one row per resource in the union of the
  two sides, code-point sorted, each with the two sides' ``capacity``,
  ``occupied`` and ``remaining`` balances and an ``equal`` marker; a missing
  side reads as three zeroes;
- leftOnly/rightOnly hold one-sided identifiers, same holds shared
  resources whose three balances all agree, and diff names shared resources
  that disagree plus the mismatched field names (capacity, occupied,
  remaining only);
- each group is code-point sorted and paired with a ``<group>Count`` key,
  and the response is compact, key-sorted JSON with one trailing newline;
- both-branches-empty yields an empty row array and four empty groups with
  zero counts, using one branch on both sides is legal (every row's
  ``equal`` is true), and identical submissions are byte-for-byte stable;
- 401 / 403 / 404 (branch_not_found) / 422 / 415 / 400 follow the fixed
  ordering (organization first, then left before right), and nothing is
  ever written or implicitly created, including across organizations.

A within-organization ``capacity`` mismatch for a resource present on both
sides cannot be produced over HTTP: a resource's capacity is fixed by its
first reservation in the forked inventory and never rewritten, so two
branches that both hold the resource always share its capacity. The
``capacity`` leg of the ``diff`` classification is therefore also exercised
directly against :func:`compare_branch_resources`.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import compare_branch_resources, create_server

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


def balances(
    capacity: int, occupied: int, remaining: int
) -> dict[str, int]:
    return {
        "capacity": capacity,
        "occupied": occupied,
        "remaining": remaining,
    }


class BranchResourceCompareTest(unittest.TestCase):
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
            payload=reservation_body(
                reservation_id, organization_id=organization_id, **kwargs
            ),
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
            "/branches/compare/resources",
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    # ------------------------------------------------------------- happy paths

    def test_rows_align_resources_and_report_three_balances(self) -> None:
        self.fork_branches()
        # Both branches first establish the same shared resource with the
        # same capacity, then the right branch reserves more against it, so
        # occupied/remaining diverge while capacity stays equal.
        self.add_branch_reservation(
            "left", "res-shared-1", resource_id="res-shared", quantity=4, capacity=10
        )
        self.add_branch_reservation(
            "right", "res-shared-1", resource_id="res-shared", quantity=4, capacity=10
        )
        self.add_branch_reservation(
            "right", "res-shared-2", resource_id="res-shared", quantity=3, capacity=10
        )
        # A resource committed only in the right branch is right-only and
        # reads as three zeroes on the left.
        self.add_branch_reservation(
            "right", "res-later-1", resource_id="res-later", quantity=2, capacity=5
        )

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "res-later",
                    "left": balances(0, 0, 0),
                    "right": balances(5, 2, 3),
                    "equal": False,
                },
                {
                    "resourceId": "res-shared",
                    "left": balances(10, 4, 6),
                    "right": balances(10, 7, 3),
                    "equal": False,
                },
            ],
        )
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["leftOnlyCount"], 0)
        self.assertEqual(body["rightOnly"], ["res-later"])
        self.assertEqual(body["rightOnlyCount"], 1)
        self.assertEqual(body["same"], [])
        self.assertEqual(body["sameCount"], 0)
        self.assertEqual(
            body["diff"],
            [{"resourceId": "res-shared", "fields": ["occupied", "remaining"]}],
        )
        self.assertEqual(body["diffCount"], 1)

        # Swapping the named sides swaps leftOnly/rightOnly and the balances.
        swapped = self.compare_payload(left="right", right="left")
        status, body = self.compare(swapped)
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-later"])
        self.assertEqual(body["rightOnly"], [])
        later_row = body["resources"][0]
        self.assertEqual(later_row["resourceId"], "res-later")
        self.assertEqual(later_row["left"], balances(5, 2, 3))
        self.assertEqual(later_row["right"], balances(0, 0, 0))

    def test_resource_with_unchanged_balances_is_same(self) -> None:
        self.fork_branches()
        self.add_branch_reservation(
            "left", "res-a-1", resource_id="res-a", quantity=2, capacity=8
        )
        self.add_branch_reservation(
            "right", "res-a-1", resource_id="res-a", quantity=2, capacity=8
        )
        # A reservation on a different resource in the right branch only does
        # not move res-a's balances, so res-a stays same and the new resource
        # is right-only.
        self.add_branch_reservation(
            "right", "res-b-1", resource_id="res-b", quantity=1, capacity=4
        )
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-a"])
        self.assertEqual(body["sameCount"], 1)
        self.assertEqual(body["rightOnly"], ["res-b"])
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["diffCount"], 0)
        same_row = next(row for row in body["resources"] if row["resourceId"] == "res-a")
        self.assertTrue(same_row["equal"])
        self.assertEqual(same_row["left"], balances(8, 2, 6))
        self.assertEqual(same_row["right"], balances(8, 2, 6))

    def test_rows_and_identifiers_sort_in_code_point_order(self) -> None:
        self.fork_branches()
        for resource_id in ("res-b", "res-A", "res-a", "res-1"):
            self.add_branch_reservation(
                "left", f"book-{resource_id}", resource_id=resource_id, quantity=1, capacity=7
            )
            self.add_branch_reservation(
                "right", f"book-{resource_id}", resource_id=resource_id, quantity=1, capacity=7
            )
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]],
            ["res-1", "res-A", "res-a", "res-b"],
        )
        self.assertEqual(body["same"], ["res-1", "res-A", "res-a", "res-b"])
        self.assertEqual(body["sameCount"], 4)
        self.assertTrue(all(row["equal"] for row in body["resources"]))

    def test_same_branch_on_both_sides_is_legal_and_fully_equal(self) -> None:
        self.fork_branches(branches=("left",))
        self.add_branch_reservation(
            "left", "res-a-1", resource_id="res-a", quantity=3, capacity=10
        )
        self.add_branch_reservation(
            "left", "res-b-1", resource_id="res-b", quantity=1, capacity=4
        )
        payload = self.compare_payload(right="left")
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-a", "res-b"])
        self.assertEqual(body["sameCount"], 2)
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["diffCount"], 0)
        self.assertEqual(len(body["resources"]), 2)
        self.assertTrue(all(row["equal"] for row in body["resources"]))

    def test_empty_branches_yield_empty_rows_groups_and_zero_counts(self) -> None:
        self.fork_branches()
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "resources": [],
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

    # ------------------------------------------------- diff classification

    def test_diff_classification_directly_covers_every_balance_leg(self) -> None:
        # A resource shared by both branches cannot disagree on capacity over
        # HTTP (the first reservation fixes it and forked inventories keep
        # that rule), so exercise the classifier directly.
        left_balances = {
            "res-same": balances(10, 2, 8),
            "res-cap": balances(10, 2, 8),
            "res-occ": balances(10, 4, 6),
            "res-left": balances(5, 1, 4),
        }
        right_balances = {
            "res-same": balances(10, 2, 8),
            # Capacity 10 vs 20 with equal occupied moves capacity and
            # remaining (remaining 8 vs 18), but not occupied.
            "res-cap": balances(20, 2, 18),
            # Occupied 4 vs 7 moves occupied and remaining, not capacity.
            "res-occ": balances(10, 7, 3),
            "res-right": balances(8, 1, 7),
        }
        result = compare_branch_resources(left_balances, right_balances)
        self.assertEqual(result["leftOnly"], ["res-left"])
        self.assertEqual(result["leftOnlyCount"], 1)
        self.assertEqual(result["rightOnly"], ["res-right"])
        self.assertEqual(result["rightOnlyCount"], 1)
        self.assertEqual(result["same"], ["res-same"])
        self.assertEqual(result["sameCount"], 1)
        self.assertEqual(
            result["diff"],
            [
                {"resourceId": "res-cap", "fields": ["capacity", "remaining"]},
                {"resourceId": "res-occ", "fields": ["occupied", "remaining"]},
            ],
        )
        self.assertEqual(result["diffCount"], 2)
        # The one-sided rows carry a zeroed opposite side and never land in
        # same/diff even though several balances "differ" against zero.
        left_only_row = next(
            row for row in result["resources"] if row["resourceId"] == "res-left"
        )
        self.assertEqual(left_only_row["left"], balances(5, 1, 4))
        self.assertEqual(left_only_row["right"], balances(0, 0, 0))
        self.assertFalse(left_only_row["equal"])

    # ----------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.fork_branches()
        self.add_branch_reservation(
            "left", "res-1", resource_id="res-1", quantity=2, capacity=9
        )
        self.add_branch_reservation(
            "right", "res-1", resource_id="res-1", quantity=2, capacity=9
        )

        raw_body = json.dumps(self.compare_payload()).encode()
        status, raw = self.raw(
            "/branches/compare/resources",
            method="POST",
            body=raw_body,
            token="w1",
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Balances and counts stay integers.
        self.assertIn(b'"capacity":9', raw)
        self.assertIn(b'"occupied":2', raw)
        self.assertIn(b'"remaining":7', raw)
        self.assertIn(b'"sameCount":1', raw)
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
                "resources",
                "rightOnly",
                "rightOnlyCount",
                "same",
                "sameCount",
            ],
        )
        self.assertEqual(list(body["resources"][0]), ["equal", "left", "resourceId", "right"])
        self.assertEqual(
            list(body["resources"][0]["left"]),
            ["capacity", "occupied", "remaining"],
        )

    def test_diff_entry_keys_are_sorted(self) -> None:
        self.fork_branches()
        self.add_branch_reservation(
            "left", "res-1", resource_id="pool", quantity=1, capacity=10
        )
        self.add_branch_reservation(
            "right", "res-1", resource_id="pool", quantity=2, capacity=10
        )
        raw_body = json.dumps(self.compare_payload()).encode()
        status, raw = self.raw(
            "/branches/compare/resources",
            method="POST",
            body=raw_body,
            token="w1",
        )
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual(list(body["diff"][0]), ["fields", "resourceId"])

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.fork_branches()
        self.add_branch_reservation("left", "res-a-1", resource_id="res-a", quantity=1, capacity=5)
        self.add_branch_reservation("left", "res-b-1", resource_id="res-b", quantity=2, capacity=6)
        self.add_branch_reservation("right", "res-b-1", resource_id="res-b", quantity=3, capacity=6)
        self.add_branch_reservation("right", "res-c-1", resource_id="res-c", quantity=1)
        raw_body = json.dumps(self.compare_payload()).encode()
        raws = [
            self.raw(
                "/branches/compare/resources",
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

        # Main-service state and alerts are untouched as well.
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
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
        self.assertEqual(body["leftOnly"], ["res-rc"])

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.fork_branches()
        raw_body = json.dumps(self.compare_payload()).encode()
        # Missing header.
        status, body = self.raw(
            "/branches/compare/resources",
            method="POST",
            body=raw_body,
            token=None,
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Unregistered token.
        status, body = self.raw(
            "/branches/compare/resources",
            method="POST",
            body=raw_body,
            token="forged",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Malformed Authorization header (no Bearer scheme) is likewise 401.
        request = Request(
            f"{self.base_url}/branches/compare/resources",
            data=raw_body,
            headers={"Authorization": "tok-1"},
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

    def test_other_organization_resources_are_never_visible(self) -> None:
        self.fork_branches(
            snapshot_id="s1", branches=("left", "right"), token="w1"
        )
        self.fork_branches(
            snapshot_id="s2", branches=("foreign-a", "foreign-b"), token="w2"
        )
        self.add_branch_reservation(
            "left", "res-mine-1", resource_id="pool-a", quantity=2, capacity=7
        )
        self.add_branch_reservation(
            "foreign-a",
            "res-foreign-1",
            token="w2",
            resource_id="pool-b",
            quantity=4,
            capacity=9,
        )

        # ORG1's comparison never sees ORG2's pool-b.
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]], ["pool-a"]
        )
        self.assertEqual(body["leftOnly"], ["pool-a"])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["same"], [])
        self.assertEqual(body["diff"], [])

        # ORG2 comparing its own branches succeeds on its own data only.
        status, body = self.call(
            "/branches/compare/resources",
            method="POST",
            token="w2",
            payload={
                "organizationId": ORG2,
                "left": "foreign-a",
                "right": "foreign-b",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]], ["pool-b"]
        )
        self.assertEqual(body["leftOnly"], ["pool-b"])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["same"], [])

    # ----------------------------------------------------------- 422/415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        self.fork_branches()
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "organizationId"},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {**valid, "extra": 1},
            {**valid, "windowSize": 60},  # window-compare fields not allowed
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
                    "/branches/compare/resources",
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
            "/branches/compare/resources",
            method="POST",
            raw_body=raw_body,
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/branches/compare/resources",
            method="POST",
            raw_body=raw_body,
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/branches/compare/resources",
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
