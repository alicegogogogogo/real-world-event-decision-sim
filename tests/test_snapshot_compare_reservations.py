"""Regression tests for POST /snapshots/compare/reservations.

The branch-level reservation comparison already existed; this locks down
the new read-only entry point that aligns two snapshots' captured
reservations by reservationId:

- identifiers only on one side land in leftOnly/rightOnly, identical shared
  reservations in same, and disagreeing shared reservations in diff with the
  mismatched field names (drawn only from organizationId, resourceId,
  quantity, capacity);
- each group is code-point sorted and paired with a ``<group>Count`` key,
  and the response is compact, key-sorted JSON with one trailing newline;
- both-snapshots-empty yields four empty groups and zero counts, using one
  snapshot on both sides is legal, and identical submissions are
  byte-for-byte stable;
- 401 / 403 / 404 (snapshot_not_found) / 422 / 415 / 400 follow the fixed
  ordering (organization first, then left before right), and nothing is
  ever written or implicitly created, including across organizations.

A single running service cannot produce a within-organization ``diff`` over
HTTP: a committed reservation's four compared fields are immutable in the
main inventory, so later snapshots only gain identifiers. The ``diff``
classification is therefore also exercised directly against captured
:class:`Snapshot` objects, where the same reservationId may legitimately
carry different fields.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import Snapshot, compare_snapshot_reservations, create_server

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
            "/snapshots/compare/reservations",
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    # ------------------------------------------------------------- happy paths

    def test_snapshots_align_by_identifier_into_reachable_groups(self) -> None:
        # A reservation present before the first capture is shared by both
        # snapshots; one committed only between the captures is right-only.
        self.add_main_reservation("res-shared", resource_id="pool", quantity=3)
        self.capture("snap-left")
        self.add_main_reservation("res-later")
        self.capture("snap-right")

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["leftOnlyCount"], 0)
        self.assertEqual(body["rightOnly"], ["res-later"])
        self.assertEqual(body["rightOnlyCount"], 1)
        self.assertEqual(body["same"], ["res-shared"])
        self.assertEqual(body["sameCount"], 1)
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["diffCount"], 0)

        # Swapping the named sides swaps leftOnly/rightOnly deterministically.
        swapped = self.compare_payload(left="snap-right", right="snap-left")
        status, body = self.compare(swapped)
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["res-later"])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["same"], ["res-shared"])

    def test_only_four_fields_participate_in_comparison(self) -> None:
        # res-a and res-b share the same resource; at capture time occupied
        # balances differ between the two captures, but balances are not
        # compared: the four tracked fields agree, so both land in same.
        self.add_main_reservation("res-a", resource_id="pool", quantity=1, capacity=10)
        self.capture("snap-left")
        self.add_main_reservation("res-b", resource_id="pool", quantity=4, capacity=10)
        self.capture("snap-right")
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-a"])
        self.assertEqual(body["rightOnly"], ["res-b"])
        self.assertEqual(body["diff"], [])

    def test_identifiers_sort_in_code_point_order(self) -> None:
        for reservation_id in ("res-b", "res-A", "res-a", "res-1"):
            self.add_main_reservation(reservation_id)
        self.capture("snap-left")
        self.capture("snap-right")
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-1", "res-A", "res-a", "res-b"])
        self.assertEqual(body["sameCount"], 4)
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])

    def test_same_snapshot_on_both_sides_is_legal_and_fully_same(self) -> None:
        self.add_main_reservation("res-1")
        self.add_main_reservation("res-2", resource_id="other")
        self.capture("snap-solo")
        payload = self.compare_payload(left="snap-solo", right="snap-solo")
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-1", "res-2"])
        self.assertEqual(body["sameCount"], 2)
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["diffCount"], 0)

    def test_empty_snapshots_yield_empty_groups_and_zero_counts(self) -> None:
        # Captured before the organization holds any reservations.
        self.capture("snap-left")
        self.capture("snap-right")
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

    # ------------------------------------------------- diff classification

    def test_diff_classification_names_mismatched_fields(self) -> None:
        # Main-service records are immutable per organization, so build the
        # captured snapshots directly: the same reservationId carries
        # different quantity/capacity on the two sides.
        left = Snapshot(
            "snap-left",
            ORG1,
            {},
            {"pool": 10},
            {
                "res-same": reservation_record("res-same", resource_id="pool"),
                "res-qc": reservation_record(
                    "res-qc", resource_id="pool", quantity=2, capacity=10
                ),
                "res-r": reservation_record("res-r", resource_id="pool-a"),
                "res-left": reservation_record("res-left"),
            },
        )
        right = Snapshot(
            "snap-right",
            ORG1,
            {},
            {"pool": 20},
            {
                "res-same": reservation_record("res-same", resource_id="pool"),
                "res-qc": reservation_record(
                    "res-qc", resource_id="pool", quantity=5, capacity=20
                ),
                "res-r": reservation_record("res-r", resource_id="pool-b"),
                "res-right": reservation_record("res-right"),
            },
        )
        result = compare_snapshot_reservations(
            left.reservations_snapshot(), right.reservations_snapshot()
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
                {"reservationId": "res-qc", "fields": ["capacity", "quantity"]},
                {"reservationId": "res-r", "fields": ["resourceId"]},
            ],
        )
        self.assertEqual(result["diffCount"], 2)

    def test_organization_field_participates_in_diff(self) -> None:
        left = Snapshot(
            "snap-left",
            ORG1,
            {},
            {},
            {"res-1": reservation_record("res-1", organization_id=ORG1)},
        )
        right = Snapshot(
            "snap-right",
            ORG1,
            {},
            {},
            {"res-1": reservation_record("res-1", organization_id="org-other")},
        )
        result = compare_snapshot_reservations(
            left.reservations_snapshot(), right.reservations_snapshot()
        )
        self.assertEqual(
            result["diff"],
            [{"fields": ["organizationId"], "reservationId": "res-1"}],
        )

    # ----------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_main_reservation("res-1")
        self.capture("snap-left")
        self.capture("snap-right")

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
                "rightOnly",
                "rightOnlyCount",
                "same",
                "sameCount",
            ],
        )

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.add_main_reservation("res-a")
        self.capture("snap-left")
        self.add_main_reservation("res-b")
        self.capture("snap-right")
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
        self.add_main_reservation("res-a")
        self.capture("snap-left")
        self.add_main_reservation("res-b")
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
            ["res-a", "res-b"],
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
        self.add_main_reservation("res-1")
        self.capture("snap-left")
        self.capture("snap-right")
        status, body = self.compare(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-1"])

    def test_missing_or_unregistered_credential_is_401(self) -> None:
        self.capture("snap-left")
        self.capture("snap-right")
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

    def test_other_organization_snapshot_is_never_visible(self) -> None:
        # Each organization commits its own reservation; each capture sees
        # only its own organization's record, and the other organization's
        # identifier never appears in any group.
        self.add_main_reservation("res-1", resource_id="pool-a")
        self.capture("snap-mine")
        self.add_main_reservation(
            "res-2", token="w2", resource_id="pool-b", quantity=4, capacity=9
        )
        self.capture("snap-foreign", token="w2")

        # ORG1 compares its own capture against itself and never sees ORG2's
        # res-2; the two captures hold disjoint per-organization data.
        status, body = self.compare(
            self.compare_payload(left="snap-mine", right="snap-mine")
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-1"])
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["diff"], [])

        # ORG2 comparing its own snapshot succeeds on its own data only.
        status, body = self.call(
            "/snapshots/compare/reservations",
            method="POST",
            token="w2",
            payload={
                "organizationId": ORG2,
                "left": "snap-foreign",
                "right": "snap-foreign",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["res-2"])
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
            {**valid, "windowSize": 60},  # branch-window fields not allowed
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

        # Nothing failed open into the stores.
        self.assertEqual(self.call("/snapshots")[0], 200)


if __name__ == "__main__":
    unittest.main()
