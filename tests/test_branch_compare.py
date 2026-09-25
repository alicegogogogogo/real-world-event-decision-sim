"""Regression tests for POST /branches/compare.

The branch simulator already ran each branch in isolation; this locks down
the new read-only entry point that compares two branches' window counts and
peak decisions:

- each side is recomputed from that branch's own events with the aggregate
  window division and the decision peak/action rules;
- without a range the rows are the union of hit windows, aligned by start;
  with a range every intersecting window gets a row, empty ones kept;
- both-branches-empty yields ``windows: []`` and identical submissions are
  byte-for-byte stable;
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
EVENT_TYPE = "incident.created"


def event_body(event_id: str, occurred_at: int, organization_id: str = ORG1) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": EVENT_TYPE,
        "occurredAt": occurred_at,
        "payload": {},
    }


class BranchCompareTest(unittest.TestCase):
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

    def add_branch_event(
        self, branch: str, event_id: str, occurred_at: int, token: str = "w1"
    ) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        status, _ = self.call(
            f"/branches/{branch}/events",
            method="POST",
            token=token,
            payload=event_body(event_id, occurred_at, organization_id),
        )
        self.assertEqual(status, 201)

    def compare_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": ORG1,
            "left": "left",
            "right": "right",
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 3,
        }
        payload.update(overrides)
        return payload

    def compare(
        self, payload: dict[str, Any] | None = None, *, token: str | None = "w1"
    ) -> tuple[int, Any]:
        return self.call(
            "/branches/compare",
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    # ------------------------------------------------------------- happy paths

    def test_compares_union_of_hit_windows_aligned_by_start(self) -> None:
        self.fork_branches()
        # left: two in window [0,60), one in [60,120).
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("left", "l2", 10)
        self.add_branch_event("left", "l3", 65)
        # right: one in [0,60), two in [120,180).
        self.add_branch_event("right", "r1", 10)
        self.add_branch_event("right", "r2", 120)
        self.add_branch_event("right", "r3", 125)

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "leftCount": 2, "rightCount": 1, "equal": False},
                {"start": 60, "leftCount": 1, "rightCount": 0, "equal": False},
                {"start": 120, "leftCount": 0, "rightCount": 2, "equal": False},
            ],
        )
        self.assertEqual(
            body["decision"],
            {
                "left": {"peakStart": 0, "peakCount": 2, "action": "observe"},
                "right": {"peakStart": 120, "peakCount": 2, "action": "observe"},
                "equal": False,
            },
        )
        # The request parameters are echoed back.
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["left"], "left")
        self.assertEqual(body["right"], "right")
        self.assertEqual(body["type"], EVENT_TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 3)
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])

    def test_range_keeps_empty_windows_and_filters_events(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("left", "l2", 10)
        self.add_branch_event("left", "l3", 65)  # outside [0, 60]
        self.add_branch_event("right", "r1", 10)
        self.add_branch_event("right", "r2", 120)  # outside [0, 60]

        payload = self.compare_payload(**{"from": 0, "to": 60})
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        # Window [60,120) intersects the closed interval at t=60 and is kept
        # even though neither side has an in-range event there.
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "leftCount": 2, "rightCount": 1, "equal": False},
                {"start": 60, "leftCount": 0, "rightCount": 0, "equal": True},
            ],
        )
        # Peaks are computed over the same filtered events: left peaks at 0
        # with 2, right at 0 with 1.
        self.assertEqual(body["decision"]["left"]["peakStart"], 0)
        self.assertEqual(body["decision"]["left"]["peakCount"], 2)
        self.assertEqual(body["decision"]["right"]["peakStart"], 0)
        self.assertEqual(body["decision"]["right"]["peakCount"], 1)

    def test_range_includes_trailing_window_hit_by_neither_side(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("right", "r1", 10)
        payload = self.compare_payload(**{"from": 0, "to": 200})
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["start"] for row in body["windows"]], [0, 60, 120, 180]
        )
        # The last two windows are empty on both sides and therefore equal.
        self.assertTrue(all(row["equal"] for row in body["windows"][1:]))
        self.assertEqual(
            body["windows"][0],
            {"start": 0, "leftCount": 1, "rightCount": 1, "equal": True},
        )

    def test_empty_window_difference_is_reported_row_by_row(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("left", "l2", 70)
        # The right branch forks from an empty snapshot and stays empty.
        payload = self.compare_payload(**{"from": 0, "to": 120})
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "leftCount": 1, "rightCount": 0, "equal": False},
                {"start": 60, "leftCount": 1, "rightCount": 0, "equal": False},
                {"start": 120, "leftCount": 0, "rightCount": 0, "equal": True},
            ],
        )

    def test_no_matching_events_on_either_side_is_empty_window_set(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("right", "r1", 10)
        # An unknown type matches no events on either side.
        payload = self.compare_payload(type="other.type")
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])
        self.assertEqual(
            body["decision"],
            {
                "left": {"peakStart": None, "peakCount": 0, "action": "observe"},
                "right": {"peakStart": None, "peakCount": 0, "action": "observe"},
                "equal": True,
            },
        )

    def test_same_branch_on_both_sides_is_legal_and_fully_equal(self) -> None:
        self.fork_branches(branches=("left",))
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("left", "l2", 70)
        payload = self.compare_payload(right="left")
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "leftCount": 1, "rightCount": 1, "equal": True},
                {"start": 60, "leftCount": 1, "rightCount": 1, "equal": True},
            ],
        )
        self.assertEqual(body["decision"]["equal"], True)
        self.assertEqual(
            body["decision"]["left"], body["decision"]["right"]
        )

    def test_threshold_decides_escalate_vs_observe_per_side(self) -> None:
        self.fork_branches()
        # Left reaches count 2; right stays at 1.
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("left", "l2", 10)
        self.add_branch_event("right", "r1", 10)

        status, body = self.compare(self.compare_payload(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(body["decision"]["left"]["action"], "escalate")
        self.assertEqual(body["decision"]["left"]["peakCount"], 2)
        self.assertEqual(body["decision"]["right"]["action"], "observe")
        self.assertEqual(body["decision"]["right"]["peakCount"], 1)
        self.assertFalse(body["decision"]["equal"])

    def test_peak_tie_resolves_to_earliest_window_start(self) -> None:
        self.fork_branches(branches=("left",))
        self.add_branch_event("left", "l1", 65)
        self.add_branch_event("left", "l2", 125)
        payload = self.compare_payload(right="left")
        status, body = self.compare(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["decision"]["left"]["peakStart"], 60)
        self.assertEqual(body["decision"]["left"]["peakCount"], 1)

    # ----------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_boolean_integer_and_newline_terminated(
        self,
    ) -> None:
        self.fork_branches()
        self.add_branch_event("left", "l1", 5)

        raw_body = json.dumps(self.compare_payload()).encode()
        status, raw = self.raw(
            "/branches/compare", method="POST", body=raw_body, token="w1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Integers stay integers, booleans stay booleans.
        text = raw.decode()
        self.assertIn('"leftCount":1', text)
        self.assertIn('"rightCount":0', text)
        self.assertIn('"equal":false', text)
        self.assertNotIn(b"true", raw)  # nothing is equal in this diff
        # Keys are sorted by code point at every level.
        body = json.loads(raw)
        self.assertEqual(
            list(body),
            [
                "decision",
                "from",
                "left",
                "organizationId",
                "right",
                "threshold",
                "to",
                "type",
                "windowSize",
                "windows",
            ],
        )
        self.assertEqual(list(body["decision"]), ["equal", "left", "right"])
        self.assertEqual(
            list(body["decision"]["left"]), ["action", "peakCount", "peakStart"]
        )
        self.assertEqual(
            list(body["windows"][0]),
            ["equal", "leftCount", "rightCount", "start"],
        )

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("left", "l2", 65)
        self.add_branch_event("right", "r1", 120)
        raw_body = json.dumps(
            self.compare_payload(**{"from": 0, "to": 180})
        ).encode()
        raws = [
            self.raw(
                "/branches/compare", method="POST", body=raw_body, token="w1"
            )[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # -------------------------------------------------------------- read-only

    def test_compare_is_read_only(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("right", "r1", 10)

        def summaries() -> tuple[Any, Any]:
            left = self.call("/branches/left")[1]
            right = self.call("/branches/right")[1]
            return left, right

        before = summaries()
        for _ in range(3):
            status, _ = self.compare(self.compare_payload(**{"from": 0, "to": 60}))
            self.assertEqual(status, 200)
        after = summaries()
        self.assertEqual(before, after)

        # Main-service state and other branches are untouched as well.
        status, events = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(events["events"], [])

    def test_failed_compare_creates_no_branch(self) -> None:
        self.fork_branches()
        for payload in (
            self.compare_payload(left="ghost"),
            self.compare_payload(right="ghost"),
            self.compare_payload(**{"windowSize": 0}),
        ):
            status, _ = self.compare(payload)
            self.assertIn(status, (404, 422))
        self.assertEqual(self.call("/branches/ghost")[0], 404)

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_compare(self) -> None:
        self.fork_branches()
        status, body = self.compare(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])

    def test_missing_or_unregistered_credential_is_401(self) -> None:
        self.fork_branches()
        raw_body = json.dumps(self.compare_payload()).encode()
        status, body = self.raw(
            "/branches/compare", method="POST", body=raw_body, token=None
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        status, body = self.raw(
            "/branches/compare", method="POST", body=raw_body, token="forged"
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
        self.fork_branches(snapshot_id="s2", branches=("foreign-a", "foreign-b"),
                           token="w2")
        self.add_branch_event("left", "l1", 5)
        self.add_branch_event("foreign-a", "f1", 65, token="w2")
        self.add_branch_event("foreign-b", "f2", 125, token="w2")

        # ORG2 compares its own two branches and sees only ORG2 events.
        payload = {
            "organizationId": ORG2,
            "left": "foreign-a",
            "right": "foreign-b",
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 1,
        }
        status, body = self.call(
            "/branches/compare", method="POST", payload=payload, token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 60, "leftCount": 1, "rightCount": 0, "equal": False},
                {"start": 120, "leftCount": 0, "rightCount": 1, "equal": False},
            ],
        )
        self.assertEqual(body["decision"]["left"]["action"], "escalate")
        self.assertEqual(body["decision"]["right"]["action"], "escalate")

        # ORG1's comparison is unchanged and never sees ORG2 events.
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 0, "leftCount": 1, "rightCount": 0, "equal": False}],
        )

    # ----------------------------------------------------------- 422 / 415/400

    def test_validation_errors_are_422_and_write_nothing(self) -> None:
        self.fork_branches()
        valid = self.compare_payload()
        bad_payloads: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {k: v for k, v in valid.items() if k != "type"},
            {k: v for k, v in valid.items() if k != "windowSize"},
            {k: v for k, v in valid.items() if k != "threshold"},
            {**valid, "extra": 1},
            {**valid, "left": ""},
            {**valid, "right": "   "},
            {**valid, "left": 7},
            {**valid, "right": None},
            {**valid, "type": ""},
            {**valid, "organizationId": ""},
            {**valid, "windowSize": 0},
            {**valid, "windowSize": -3},
            {**valid, "windowSize": 1.5},
            {**valid, "windowSize": True},
            {**valid, "windowSize": "60"},
            {**valid, "threshold": 0},
            {**valid, "threshold": False},
            {**valid, "threshold": 2.0},
            {**valid, "from": 10},  # unpaired
            {**valid, "to": 10},  # unpaired
            {**valid, "from": -1, "to": 10},
            {**valid, "from": 10, "to": 9},
            {**valid, "from": "0", "to": 9},
            {**valid, "from": 0, "to": False},
            [],
            "x",
            42,
            True,
            None,
        ]
        for bad_payload in bad_payloads:
            with self.subTest(bad_payload=bad_payload):
                status, parsed = self.call(
                    "/branches/compare",
                    method="POST",
                    raw_body=json.dumps(bad_payload).encode(),
                    token="w1",
                )
                self.assertEqual(status, 422)
                self.assertEqual(parsed["error"], "validation_error")

        # No failed validation created a branch or altered the real ones.
        self.assertEqual(
            self.call("/branches/left")[1]["events"], 0
        )
        self.assertEqual(self.call("/branches/right")[1]["events"], 0)

    def test_media_type_and_json_errors(self) -> None:
        self.fork_branches()
        raw_body = json.dumps(self.compare_payload()).encode()

        status, body = self.call(
            "/branches/compare",
            method="POST",
            raw_body=raw_body,
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/branches/compare",
            method="POST",
            raw_body=raw_body,
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/branches/compare",
            method="POST",
            raw_body=b'{"left": ',
            content_type="application/json",
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

        # Nothing failed open into the stores.
        self.assertEqual(self.call("/branches/left")[0], 200)

    def test_compare_is_not_available_on_get_or_under_a_branch(self) -> None:
        self.fork_branches()
        # GET resolves "compare" as a branch name under the existing route.
        status, body = self.call("/branches/compare")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")
        # Branch-prefixed compare paths stay 404 (no new branch sub-paths).
        status, body = self.call(
            "/branches/left/compare", method="POST", payload=self.compare_payload()
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
