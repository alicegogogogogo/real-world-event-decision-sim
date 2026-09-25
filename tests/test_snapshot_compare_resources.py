"""Regression tests for POST /snapshots/compare/resources.

The snapshot comparisons already covered window decisions, events, and
reservations; this locks down the new read-only entry point that aligns two
snapshots' captured resource balances by resourceId:

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
- both-snapshots-empty yields an empty row array and four empty groups with
  zero counts, using one snapshot on both sides is legal (every row's
  ``equal`` is true), and identical submissions are byte-for-byte stable;
- 401 / 403 / 404 (snapshot_not_found) / 422 / 415 / 400 follow the fixed
  ordering (organization first, then left before right), and nothing is
  ever written or implicitly created, including across organizations.

A single running service cannot produce a within-organization ``capacity``
mismatch over HTTP: a resource's capacity is fixed by its first reservation
in the main inventory, so later snapshots only change occupied/remaining.
The ``capacity`` leg of the ``diff`` classification is therefore also
exercised directly against captured :class:`Snapshot` objects.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import Snapshot, compare_snapshot_resources, create_server

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


def reservation_record(
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


class SnapshotResourceCompareTest(unittest.TestCase):
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

    def add_main_reservation(
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
        self,
        snapshot_id: str,
        *,
        token: str = "w1",
        expect: int = 201,
    ) -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, expect)

    def compare_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": ORG1,
            "left": "snap-left",
            "right": "snap-right",
        }
        payload.update(overrides)
        return payload

    def compare(
        self,
        payload: dict[str, Any] | None = None,
        *,
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        return self.call(
            "/snapshots/compare/resources",
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    # ------------------------------------------------------------- happy paths

    def test_rows_align_resources_and_report_three_balances(self) -> None:
        # res-shared exists at both captures; a second reservation against it
        # moves occupied/remaining between the captures. res-later is a new
        # resource committed only after the left capture, so it is right-only
        # and reads as three zeroes on the left.
        self.add_main_reservation(
            "res-shared-1", resource_id="res-shared", quantity=4, capacity=10
        )
        self.capture("snap-left")
        self.add_main_reservation(
            "res-shared-2", resource_id="res-shared", quantity=3, capacity=10
        )
        self.add_main_reservation(
            "res-later-1", resource_id="res-later", quantity=2, capacity=5
        )
        self.capture("snap-right")

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["resources"],
            [
                {
                    "resourceId": "res-later",
                    "left": {"capacity": 0, "occupied": 0, "remaining": 0},
                    "right": {"capacity": 5, "occupied": 2, "remaining": 3},
                    "equal": False,
                },
                {
                    "resourceId": "res-shared",
                    "left": {"capacity": 10, "occupied": 4, "remaining": 6},
                    "right": {"capacity": 10, "occupied": 7, "remaining": 3},
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
        swapped = self.compare_payload(left="snap-right", right="snap-left")
        status, body = self.compare(swapped)
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-later"])
        self.assertEqual(body["rightOnly"], [])
        later_row = body["resources"][0]
        self.assertEqual(later_row["resourceId"], "res-later")
        self.assertEqual(
            later_row["left"], {"capacity": 5, "occupied": 2, "remaining": 3}
        )
        self.assertEqual(
            later_row["right"], {"capacity": 0, "occupied": 0, "remaining": 0}
        )

    def test_resource_with_unchanged_balances_is_same(self) -> None:
        self.add_main_reservation(
            "res-a-1", resource_id="res-a", quantity=2, capacity=8
        )
        self.capture("snap-left")
        # A reservation on a different resource in between does not move
        # res-a's balances, so res-a stays same and the new resource is
        # right-only.
        self.add_main_reservation(
            "res-b-1", resource_id="res-b", quantity=1, capacity=4
        )
        self.capture("snap-right")
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-a"])
        self.assertEqual(body["sameCount"], 1)
        self.assertEqual(body["rightOnly"], ["res-b"])
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["diffCount"], 0)
        same_row = next(row for row in body["resources"] if row["resourceId"] == "res-a")
        self.assertTrue(same_row["equal"])
        self.assertEqual(
            same_row["left"], {"capacity": 8, "occupied": 2, "remaining": 6}
        )
        self.assertEqual(
            same_row["right"], {"capacity": 8, "occupied": 2, "remaining": 6}
        )

    def test_rows_and_identifiers_sort_in_code_point_order(self) -> None:
        for resource_id in ("res-b", "res-A", "res-a", "res-1"):
            self.add_main_reservation(
                f"book-{resource_id}", resource_id=resource_id, quantity=1, capacity=7
            )
        self.capture("snap-left")
        self.capture("snap-right")
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]],
            ["res-1", "res-A", "res-a", "res-b"],
        )
        self.assertEqual(body["same"], ["res-1", "res-A", "res-a", "res-b"])
        self.assertEqual(body["sameCount"], 4)
        self.assertTrue(all(row["equal"] for row in body["resources"]))

    def test_same_snapshot_on_both_sides_is_legal_and_fully_equal(self) -> None:
        self.add_main_reservation(
            "res-a-1", resource_id="res-a", quantity=3, capacity=10
        )
        self.add_main_reservation(
            "res-b-1", resource_id="res-b", quantity=1, capacity=4
        )
        self.capture("snap-solo")
        payload = self.compare_payload(left="snap-solo", right="snap-solo")
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

    def test_empty_snapshots_yield_empty_rows_groups_and_zero_counts(self) -> None:
        # Captured before the organization holds any reservations (and thus
        # any resource with a recorded capacity).
        self.capture("snap-left")
        self.capture("snap-right")
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

    def test_diff_classification_against_captured_snapshots(self) -> None:
        # Main-service capacities are immutable per organization, so build the
        # captured snapshots directly to exercise the capacity leg and an
        # occupied/remaining leg independently.
        left = Snapshot(
            "snap-left",
            ORG1,
            {},
            {"res-same": 10, "res-cap": 10, "res-occ": 10, "res-left": 5},
            {
                "l-same": reservation_record(
                    "l-same", resource_id="res-same", quantity=2, capacity=10
                ),
                "l-cap": reservation_record(
                    "l-cap", resource_id="res-cap", quantity=2, capacity=10
                ),
                "l-occ-a": reservation_record(
                    "l-occ-a", resource_id="res-occ", quantity=4, capacity=10
                ),
                "l-left": reservation_record(
                    "l-left", resource_id="res-left", quantity=1, capacity=5
                ),
            },
        )
        right = Snapshot(
            "snap-right",
            ORG1,
            {},
            {"res-same": 10, "res-cap": 20, "res-occ": 10, "res-right": 8},
            {
                "r-same": reservation_record(
                    "r-same", resource_id="res-same", quantity=2, capacity=10
                ),
                "r-cap": reservation_record(
                    "r-cap", resource_id="res-cap", quantity=2, capacity=20
                ),
                "r-occ-a": reservation_record(
                    "r-occ-a", resource_id="res-occ", quantity=4, capacity=10
                ),
                "r-occ-b": reservation_record(
                    "r-occ-b", resource_id="res-occ", quantity=3, capacity=10
                ),
                "r-right": reservation_record(
                    "r-right", resource_id="res-right", quantity=1, capacity=8
                ),
            },
        )
        result = compare_snapshot_resources(
            left.resource_balances(), right.resource_balances()
        )
        self.assertEqual(result["leftOnly"], ["res-left"])
        self.assertEqual(result["leftOnlyCount"], 1)
        self.assertEqual(result["rightOnly"], ["res-right"])
        self.assertEqual(result["rightOnlyCount"], 1)
        self.assertEqual(result["same"], ["res-same"])
        self.assertEqual(result["sameCount"], 1)
        self.assertEqual(
            result["diff"],
            [
                # Capacity 10 vs 20 with equal occupied moves capacity and
                # remaining (remaining 8 vs 18), but not occupied.
                {"resourceId": "res-cap", "fields": ["capacity", "remaining"]},
                # Occupied 4 vs 7 moves occupied and remaining, not capacity.
                {"resourceId": "res-occ", "fields": ["occupied", "remaining"]},
            ],
        )
        self.assertEqual(result["diffCount"], 2)

    def test_balances_sum_all_captured_reservations_for_a_resource(self) -> None:
        snapshot = Snapshot(
            "snap-solo",
            ORG1,
            {},
            {"pool": 10},
            {
                "r1": reservation_record("r1", resource_id="pool", quantity=2, capacity=10),
                "r2": reservation_record("r2", resource_id="pool", quantity=3, capacity=10),
            },
        )
        self.assertEqual(
            snapshot.resource_balances(),
            {"pool": {"capacity": 10, "occupied": 5, "remaining": 5}},
        )

    # ----------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_main_reservation(
            "res-1", resource_id="res-1", quantity=2, capacity=9
        )
        self.capture("snap-left")
        self.capture("snap-right")

        raw_body = json.dumps(self.compare_payload()).encode()
        status, raw = self.raw(
            "/snapshots/compare/resources",
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

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.add_main_reservation(
            "res-a-1", resource_id="res-a", quantity=1, capacity=5
        )
        self.capture("snap-left")
        self.add_main_reservation(
            "res-b-1", resource_id="res-b", quantity=2, capacity=6
        )
        self.capture("snap-right")
        raw_body = json.dumps(self.compare_payload()).encode()
        raws = [
            self.raw(
                "/snapshots/compare/resources",
                method="POST",
                body=raw_body,
                token="w1",
            )[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # -------------------------------------------------------------- read-only

    def test_compare_is_read_only(self) -> None:
        self.add_main_reservation(
            "res-a-1", resource_id="res-a", quantity=1, capacity=5
        )
        self.capture("snap-left")
        self.add_main_reservation(
            "res-b-1", resource_id="res-b", quantity=1, capacity=5
        )
        self.capture("snap-right")

        def snapshot_summaries() -> Any:
            return self.call("/snapshots")[1]

        before = snapshot_summaries()
        for _ in range(3):
            status, _ = self.compare()
            self.assertEqual(status, 200)
        self.assertEqual(snapshot_summaries(), before)

        # Main-service inventory and alerts are untouched as well.
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            sorted(r["reservationId"] for r in reservations["reservations"]),
            ["res-a-1", "res-b-1"],
        )
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    def test_failed_compare_creates_no_snapshot(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        for payload in (
            self.compare_payload(left="ghost"),
            self.compare_payload(right="ghost"),
            self.compare_payload(left=""),
        ):
            status, _ = self.compare(payload)
            self.assertIn(status, (404, 422))
        snapshot_ids = [
            entry["snapshotId"] for entry in self.call("/snapshots")[1]["snapshots"]
        ]
        self.assertNotIn("ghost", snapshot_ids)
        # The never-seen name is still 404 and never materialized.
        status, _ = self.compare(self.compare_payload(left="ghost"))
        self.assertEqual(status, 404)

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_compare(self) -> None:
        self.add_main_reservation(
            "res-1", resource_id="res-1", quantity=1, capacity=5
        )
        self.capture("snap-left")
        self.capture("snap-right")
        status, body = self.compare(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-1"])

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        raw_body = json.dumps(self.compare_payload()).encode()
        # Missing header.
        status, body = self.raw(
            "/snapshots/compare/resources",
            method="POST",
            body=raw_body,
            token=None,
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Unregistered token.
        status, body = self.raw(
            "/snapshots/compare/resources",
            method="POST",
            body=raw_body,
            token="forged",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Malformed Authorization header (no Bearer scheme) is likewise 401.
        request = Request(
            f"{self.base_url}/snapshots/compare/resources",
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

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        self.capture("foreign", token="w2")

        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the snapshot names do not exist anywhere.
        status, body = self.compare(
            self.compare_payload(left="nope", right="also-nope"), token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

        # An ORG1 credential naming ORG2's snapshot cannot read it.
        status, body = self.compare(
            self.compare_payload(left="foreign", right="snap-right")
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        status, body = self.compare(
            self.compare_payload(left="snap-left", right="foreign")
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_snapshot_existence_checked_left_then_right(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        self.capture("foreign", token="w2")

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

    def test_other_organization_resources_are_never_visible(self) -> None:
        # Each organization reserves its own resource; each capture sees only
        # its own organization's capacities, and the other organization's
        # resource never appears in any row or group.
        self.add_main_reservation(
            "res-mine-1", resource_id="pool-a", quantity=2, capacity=7
        )
        self.capture("snap-mine")
        self.add_main_reservation(
            "res-foreign-1",
            token="w2",
            resource_id="pool-b",
            quantity=4,
            capacity=9,
        )
        self.capture("snap-foreign", token="w2")

        # ORG1 compares its own capture against itself and never sees ORG2's
        # pool-b; the two captures hold disjoint per-organization data.
        status, body = self.compare(
            self.compare_payload(left="snap-mine", right="snap-mine")
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]], ["pool-a"]
        )
        self.assertEqual(body["same"], ["pool-a"])
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["diff"], [])

        # ORG2 comparing its own snapshot succeeds on its own data only.
        status, body = self.call(
            "/snapshots/compare/resources",
            method="POST",
            token="w2",
            payload={
                "organizationId": ORG2,
                "left": "snap-foreign",
                "right": "snap-foreign",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["resourceId"] for row in body["resources"]], ["pool-b"]
        )
        self.assertEqual(body["same"], ["pool-b"])
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])

    # ----------------------------------------------------------- 422/415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "organizationId"},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {**valid, "extra": 1},
            {**valid, "windowSize": 60},  # window-decision fields not allowed
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
                    "/snapshots/compare/resources",
                    method="POST",
                    raw_body=json.dumps(bad_payload).encode(),
                    token="w1",
                )
                self.assertEqual(status, 422)
                self.assertEqual(parsed["error"], "validation_error")

        # No failed validation created a snapshot or altered the captures.
        summaries = {
            entry["snapshotId"]: entry
            for entry in self.call("/snapshots")[1]["snapshots"]
        }
        self.assertEqual(set(summaries), {"snap-left", "snap-right"})

    def test_media_type_and_json_errors(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
        raw_body = json.dumps(self.compare_payload()).encode()

        status, body = self.call(
            "/snapshots/compare/resources",
            method="POST",
            raw_body=raw_body,
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/snapshots/compare/resources",
            method="POST",
            raw_body=raw_body,
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/snapshots/compare/resources",
            method="POST",
            raw_body=b'{"left": ',
            content_type="application/json",
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

        # Nothing failed open into the stores.
        self.assertEqual(self.call("/snapshots")[0], 200)


if __name__ == "__main__":
    unittest.main()
