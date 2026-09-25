"""Regression tests for POST /branches/compare.

Covers the finalized comparison contract:

- each side recomputes window counts and the peak decision from its own
  branch ledger, with the exact same windowing as the aggregate endpoint;
- without a range the rows are the union of hit windows aligned by start,
  with a range every intersecting window is a row (empty windows kept);
- the decision carries both peak triples (largest count, earliest start on
  a tie, escalate at/above threshold) plus an equal marker;
- both sides empty yields ``windows: []`` and identical observe peaks;
- the same branch name on both sides is a legal self-comparison;
- output is compact, code-point-sorted JSON with one trailing newline and is
  byte-for-byte repeatable, and the endpoint never writes;
- 401/403/404/422/415/400 precedence across multiple organizations.
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
TYPE = "incident.created"
OTHER_TYPE = "other.type"

COMPARE_PATH = "/branches/compare"


class BranchCompareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.register("w1", ORG1, "write")
        self.register("r1", ORG1, "read")
        self.register("w2", ORG2, "write")
        # Two empty org-1 branches forked from the same empty snapshot.
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_branch("a", "s1", "w1")[0], 201)
        self.assertEqual(self.make_branch("b", "s1", "w1")[0], 201)

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
        token: str | None = None,
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
            payload={
                "token": token,
                "organizationId": organization_id,
                "role": role,
            },
        )
        self.assertIn(status, (200, 201))

    def make_snapshot(self, snapshot_id: str, token: str) -> tuple[int, Any]:
        return self.call(
            "/snapshots",
            method="POST",
            payload={"snapshotId": snapshot_id},
            token=token,
        )

    def make_branch(
        self, branch_id: str, snapshot_id: str, token: str
    ) -> tuple[int, Any]:
        return self.call(
            "/branches",
            method="POST",
            payload={"branchId": branch_id, "snapshotId": snapshot_id},
            token=token,
        )

    def branch_event(
        self,
        branch: str,
        event_id: str,
        occurred_at: int,
        *,
        token: str = "w1",
        organization_id: str = ORG1,
        event_type: str = TYPE,
    ) -> tuple[int, Any]:
        return self.call(
            f"/branches/{branch}/events",
            method="POST",
            payload={
                "eventId": event_id,
                "organizationId": organization_id,
                "type": event_type,
                "occurredAt": occurred_at,
                "payload": {},
            },
            token=token,
        )

    def compare(
        self,
        payload: dict[str, Any],
        *,
        token: str = "w1",
        content_type: str | None = "application/json",
        raw_body: bytes | None = None,
    ) -> tuple[int, Any]:
        # Encode unconditionally so a None payload ships the literal JSON
        # "null" (a 422 non-object body), not an empty request body (400).
        body = (
            raw_body
            if raw_body is not None
            else json.dumps(payload).encode()
        )
        status, raw = self.raw(
            COMPARE_PATH,
            method="POST",
            body=body,
            token=token,
            content_type=content_type,
        )
        return status, json.loads(raw)

    def compare_raw(
        self, payload: dict[str, Any], *, token: str = "w1"
    ) -> tuple[int, bytes]:
        return self.raw(
            COMPARE_PATH,
            method="POST",
            body=json.dumps(payload).encode(),
            token=token,
        )

    def base_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organizationId": ORG1,
            "left": "a",
            "right": "b",
            "type": TYPE,
            "windowSize": 60,
            "threshold": 3,
        }
        payload.update(overrides)
        return payload

    # ---------------------------------------------------------- happy paths

    def test_windows_cover_union_aligned_by_start_and_decision_diff(self) -> None:
        # Left: window 0 gets two, window 60 one. Right: window 0 one,
        # window 60 two, and an extra window 120 the left never hits.
        for event_id, ts in (("a1", 10), ("a2", 20), ("a3", 70)):
            self.assertEqual(self.branch_event("a", event_id, ts)[0], 201)
        for event_id, ts in (("b1", 5), ("b2", 65), ("b3", 80), ("b4", 130)):
            self.assertEqual(self.branch_event("b", event_id, ts)[0], 201)

        status, body = self.compare(self.base_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["left"], "a")
        self.assertEqual(body["right"], "b")
        self.assertEqual(body["type"], TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 3)
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "leftCount": 2, "rightCount": 1, "equal": False},
                {"start": 60, "leftCount": 1, "rightCount": 2, "equal": False},
                {"start": 120, "leftCount": 0, "rightCount": 1, "equal": False},
            ],
        )
        self.assertEqual(
            body["decision"],
            {
                "left": {"peakStart": 0, "peakCount": 2, "action": "observe"},
                "right": {"peakStart": 60, "peakCount": 2, "action": "observe"},
                "equal": False,
            },
        )

    def test_response_is_compact_sorted_json_with_one_newline(self) -> None:
        for event_id, ts in (("a1", 10), ("a2", 20), ("a3", 70)):
            self.assertEqual(self.branch_event("a", event_id, ts)[0], 201)
        for event_id, ts in (("b1", 5), ("b2", 65), ("b3", 80), ("b4", 130)):
            self.assertEqual(self.branch_event("b", event_id, ts)[0], 201)

        status, raw = self.compare_raw(self.base_payload())
        self.assertEqual(status, 200)
        expected = {
            "organizationId": ORG1,
            "left": "a",
            "right": "b",
            "type": TYPE,
            "windowSize": 60,
            "threshold": 3,
            "from": None,
            "to": None,
            "windows": [
                {"start": 0, "leftCount": 2, "rightCount": 1, "equal": False},
                {"start": 60, "leftCount": 1, "rightCount": 2, "equal": False},
                {"start": 120, "leftCount": 0, "rightCount": 1, "equal": False},
            ],
            "decision": {
                "left": {"peakStart": 0, "peakCount": 2, "action": "observe"},
                "right": {"peakStart": 60, "peakCount": 2, "action": "observe"},
                "equal": False,
            },
        }
        encoded = json.dumps(
            expected, separators=(",", ":"), sort_keys=True
        ).encode() + b"\n"
        self.assertEqual(raw, encoded)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Integers and booleans keep their JSON types (no 2.0 / "true" text).
        self.assertIn(b'"peakCount":2', raw)
        self.assertIn(b'"equal":false', raw)

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.assertEqual(self.branch_event("a", "a1", 10)[0], 201)
        self.assertEqual(self.branch_event("b", "b1", 90)[0], 201)
        raws = [self.compare_raw(self.base_payload())[1] for _ in range(3)]
        self.assertEqual(len(set(raws)), 1)

    def test_threshold_escalation_and_peak_tie_earliest_start(self) -> None:
        # Left ties windows 0 and 60 at one each; the earlier start wins and
        # threshold 1 escalates. The right has no events.
        self.assertEqual(self.branch_event("a", "a1", 10)[0], 201)
        self.assertEqual(self.branch_event("a", "a2", 70)[0], 201)
        status, body = self.compare(self.base_payload(threshold=1))
        self.assertEqual(status, 200)
        self.assertEqual(
            body["decision"]["left"],
            {"peakStart": 0, "peakCount": 1, "action": "escalate"},
        )
        self.assertEqual(
            body["decision"]["right"],
            {"peakStart": None, "peakCount": 0, "action": "observe"},
        )
        self.assertFalse(body["decision"]["equal"])

    def test_one_side_escalates_the_other_observes(self) -> None:
        for event_id, ts in (("a1", 1), ("a2", 2), ("a3", 3)):
            self.assertEqual(self.branch_event("a", event_id, ts)[0], 201)
        self.assertEqual(self.branch_event("b", "b1", 1)[0], 201)
        status, body = self.compare(self.base_payload(threshold=3))
        self.assertEqual(status, 200)
        self.assertEqual(body["decision"]["left"]["action"], "escalate")
        self.assertEqual(body["decision"]["left"]["peakCount"], 3)
        self.assertEqual(body["decision"]["right"]["action"], "observe")
        self.assertEqual(body["decision"]["right"]["peakCount"], 1)
        self.assertFalse(body["decision"]["equal"])

    def test_self_comparison_is_equal_on_every_row_and_decision(self) -> None:
        for event_id, ts in (("a1", 10), ("a2", 70), ("a3", 130)):
            self.assertEqual(self.branch_event("a", event_id, ts)[0], 201)
        status, body = self.compare(self.base_payload(right="a"))
        self.assertEqual(status, 200)
        self.assertTrue(all(row["equal"] for row in body["windows"]))
        for row in body["windows"]:
            self.assertEqual(row["leftCount"], row["rightCount"])
        self.assertTrue(body["decision"]["equal"])
        self.assertEqual(
            body["decision"]["left"], body["decision"]["right"]
        )

    def test_both_sides_empty_without_range_has_empty_window_set(self) -> None:
        status, body = self.compare(self.base_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])
        observe = {"peakStart": None, "peakCount": 0, "action": "observe"}
        self.assertEqual(body["decision"]["left"], observe)
        self.assertEqual(body["decision"]["right"], observe)
        self.assertTrue(body["decision"]["equal"])

    # ---------------------------------------------------------- range cases

    def test_range_counts_closed_interval_and_keeps_all_grid_windows(self) -> None:
        # Left hits 70 only inside [60,120]; right hits 65 and 80. Event 130
        # and the sub-60 events fall outside the closed interval.
        for event_id, ts in (("a1", 10), ("a2", 20), ("a3", 70)):
            self.assertEqual(self.branch_event("a", event_id, ts)[0], 201)
        for event_id, ts in (("b1", 5), ("b2", 65), ("b3", 80), ("b4", 130)):
            self.assertEqual(self.branch_event("b", event_id, ts)[0], 201)

        status, body = self.compare(self.base_payload(**{"from": 60, "to": 120}))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 60)
        self.assertEqual(body["to"], 120)
        # Window 120 intersects the interval but neither side has an event in
        # it: the empty row is kept and equal.
        self.assertEqual(
            body["windows"],
            [
                {"start": 60, "leftCount": 1, "rightCount": 2, "equal": False},
                {"start": 120, "leftCount": 0, "rightCount": 0, "equal": True},
            ],
        )

    def test_range_endpoints_are_inclusive(self) -> None:
        self.assertEqual(self.branch_event("a", "a1", 60)[0], 201)
        self.assertEqual(self.branch_event("b", "b1", 119)[0], 201)
        status, body = self.compare(self.base_payload(**{"from": 60, "to": 119}))
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 60, "leftCount": 1, "rightCount": 1, "equal": True}],
        )

    def test_range_with_empty_window_difference(self) -> None:
        # Only the left has an event (at 100 -> window 60). Across [0,119]
        # window 0 is empty on both sides, window 60 differs.
        self.assertEqual(self.branch_event("a", "a1", 100)[0], 201)
        status, body = self.compare(
            self.base_payload(threshold=1, **{"from": 0, "to": 119})
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "leftCount": 0, "rightCount": 0, "equal": True},
                {"start": 60, "leftCount": 1, "rightCount": 0, "equal": False},
            ],
        )
        self.assertEqual(
            body["decision"]["left"],
            {"peakStart": 60, "peakCount": 1, "action": "escalate"},
        )
        self.assertEqual(
            body["decision"]["right"],
            {"peakStart": None, "peakCount": 0, "action": "observe"},
        )
        self.assertFalse(body["decision"]["equal"])

    def test_range_matching_no_events_keeps_empty_grid_rows(self) -> None:
        self.assertEqual(self.branch_event("a", "a1", 10)[0], 201)
        self.assertEqual(self.branch_event("b", "b1", 20)[0], 201)
        status, body = self.compare(self.base_payload(**{"from": 200, "to": 260}))
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 180, "leftCount": 0, "rightCount": 0, "equal": True},
                {"start": 240, "leftCount": 0, "rightCount": 0, "equal": True},
            ],
        )
        observe = {"peakStart": None, "peakCount": 0, "action": "observe"}
        self.assertEqual(body["decision"]["left"], observe)
        self.assertEqual(body["decision"]["right"], observe)
        self.assertTrue(body["decision"]["equal"])

    # ------------------------------------------------------- type filtering

    def test_only_matching_type_is_counted(self) -> None:
        self.assertEqual(self.branch_event("a", "a1", 10)[0], 201)
        self.assertEqual(
            self.branch_event("a", "a2", 11, event_type=OTHER_TYPE)[0], 201
        )
        self.assertEqual(self.branch_event("b", "b1", 10)[0], 201)
        status, body = self.compare(self.base_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 0, "leftCount": 1, "rightCount": 1, "equal": True}],
        )
        self.assertTrue(body["decision"]["equal"])

    # --------------------------------------------------- multi-org / 403/404

    def test_foreign_branch_on_either_side_is_403(self) -> None:
        self.assertEqual(self.make_snapshot("s2", "w2")[0], 201)
        self.assertEqual(self.make_branch("c", "s2", "w2")[0], 201)

        # Right branch belongs to org-2.
        status, body = self.compare(self.base_payload(right="c"))
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        # Left branch belongs to org-2, caller is org-2; the org-1 branch on
        # the right is then foreign to it.
        status, body = self.compare(
            self.base_payload(organizationId=ORG2, left="c", right="a"),
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_judgment_order_is_left_existence_then_right(self) -> None:
        self.assertEqual(self.make_snapshot("s2", "w2")[0], 201)
        self.assertEqual(self.make_branch("c", "s2", "w2")[0], 201)

        # Left name never appeared: 404 even though the right is foreign.
        status, body = self.compare(self.base_payload(left="ghost", right="c"))
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "branch_not_found"})

        # Left exists and belongs to the (org-2) caller; the right name has
        # never appeared, so the fixed left-then-right order reaches the
        # right side's existence check and reports 404.
        status, body = self.compare(
            self.base_payload(organizationId=ORG2, left="c", right="ghost"),
            token="w2",
        )
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "branch_not_found"})

        # A foreign (existing) left still outranks a missing right when the
        # caller cannot use the left: an org-1 credential naming org-2's
        # branch on the left is 403 before the missing right is inspected.
        status, body = self.compare(self.base_payload(left="c", right="ghost"))
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        # Both names unknown: 404.
        status, body = self.compare(
            self.base_payload(left="ghost-1", right="ghost-2")
        )
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "branch_not_found"})

        # The failed comparisons never created a branch.
        status, _ = self.call("/branches/ghost", token="w1")
        self.assertEqual(status, 404)
        status, _ = self.call("/branches/ghost-1", token="w1")
        self.assertEqual(status, 404)

    def test_organization_mismatch_is_403_before_branch_lookup(self) -> None:
        status, body = self.compare(
            self.base_payload(organizationId=ORG2, left="a", right="b")
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_foreign_same_name_on_both_sides_is_403(self) -> None:
        self.assertEqual(self.make_snapshot("s2", "w2")[0], 201)
        self.assertEqual(self.make_branch("c", "s2", "w2")[0], 201)
        status, body = self.compare(
            self.base_payload(organizationId=ORG2, left="c", right="c"),
            token="w2",
        )
        self.assertEqual(status, 200)
        # An org-1 credential cannot self-compare org-2's branch.
        status, body = self.compare(self.base_payload(left="c", right="c"))
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    # ------------------------------------------------------------- auth 401

    def test_missing_or_unregistered_credential_is_401(self) -> None:
        status, raw = self.raw(
            COMPARE_PATH,
            method="POST",
            body=json.dumps(self.base_payload()).encode(),
            token=None,
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

        status, raw = self.raw(
            COMPARE_PATH,
            method="POST",
            body=json.dumps(self.base_payload()).encode(),
            token="bogus-token",
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})

        # A non-Bearer scheme is equally unauthenticated.
        request = Request(
            f"{self.base_url}{COMPARE_PATH}",
            data=json.dumps(self.base_payload()).encode(),
            headers={"Authorization": "Basic abc", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=5)
            self.fail("expected 401")
        except HTTPError as error:
            self.assertEqual(error.code, 401)
            self.assertEqual(json.loads(error.read()), {"error": "unauthorized"})
            error.close()

    # ------------------------------------------------------- 415 / 400 / 422

    def test_read_credential_may_compare(self) -> None:
        self.assertEqual(self.branch_event("a", "a1", 10)[0], 201)
        status, _ = self.compare(self.base_payload(), token="r1")
        self.assertEqual(status, 200)

    def test_missing_or_bad_content_type_is_415(self) -> None:
        body = json.dumps(self.base_payload()).encode()
        status, parsed = self.call(
            COMPARE_PATH,
            method="POST",
            raw_body=body,
            token="w1",
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(parsed["error"], "unsupported_media_type")

        status, parsed = self.call(
            COMPARE_PATH,
            method="POST",
            raw_body=body,
            token="w1",
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(parsed["error"], "unsupported_media_type")

    def test_charset_content_type_is_accepted(self) -> None:
        status, raw = self.raw(
            COMPARE_PATH,
            method="POST",
            body=json.dumps(self.base_payload()).encode(),
            token="w1",
            content_type="application/json; charset=utf-8",
        )
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))

    def test_malformed_json_is_400(self) -> None:
        status, body = self.compare(self.base_payload(), raw_body=b'{"left": ')
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_validation_errors_are_422(self) -> None:
        valid = self.base_payload()
        bad_bodies: list[Any] = [
            {},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "right"},
            {k: v for k, v in valid.items() if k != "threshold"},
            {**valid, "extra": 1},
            {**valid, "left": ""},
            {**valid, "right": "   "},
            {**valid, "organizationId": ""},
            {**valid, "type": ""},
            {**valid, "left": 5},
            {**valid, "right": None},
            {**valid, "windowSize": 0},
            {**valid, "windowSize": -10},
            {**valid, "windowSize": "60"},
            {**valid, "windowSize": 60.0},
            {**valid, "windowSize": True},
            {**valid, "threshold": 0},
            {**valid, "threshold": False},
            {**valid, "threshold": 2.5},
            {**valid, "from": 0},
            {**valid, "to": 10},
            {**valid, "from": -1, "to": 10},
            {**valid, "from": 0, "to": "10"},
            {**valid, "from": 100, "to": 10},
            [],
            "nope",
            42,
            True,
            None,
        ]
        for bad_body in bad_bodies:
            with self.subTest(bad_body=bad_body):
                status, body = self.compare(bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_invalid_body_is_422_even_when_branch_name_unknown(self) -> None:
        # Validation precedes the existence check: a malformed body can never
        # be reported as a missing branch, and no branch is created.
        bad = self.base_payload(left="ghost")
        del bad["threshold"]
        status, body = self.compare(bad)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")
        status, _ = self.call("/branches/ghost", token="w1")
        self.assertEqual(status, 404)

    def test_failed_requests_write_nothing(self) -> None:
        self.assertEqual(self.branch_event("a", "a1", 10)[0], 201)

        # A round of rejected comparisons.
        self.compare(self.base_payload(left="ghost"))  # 404
        self.compare(self.base_payload(right="ghost"))  # 404
        self.compare(self.base_payload(threshold=0))  # 422
        self.compare(
            self.base_payload(organizationId=ORG2), token="r1"
        )  # 403 (org mismatch)
        self.raw(
            COMPARE_PATH,
            method="POST",
            body=b'{"left": ',
            token="w1",
        )  # 400
        self.raw(
            COMPARE_PATH,
            method="POST",
            body=json.dumps(self.base_payload()).encode(),
            token="w1",
            content_type=None,
        )  # 415

        # Both branches keep exactly their prior events; a valid comparison
        # still sees the same single event on the left and nothing right.
        status, body = self.compare(self.base_payload(threshold=1))
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [{"start": 0, "leftCount": 1, "rightCount": 0, "equal": False}],
        )
        status, events = self.call(
            "/branches/a/events?organizationId=org-1", token="w1"
        )
        self.assertEqual(len(events["events"]), 1)
        status, events = self.call(
            "/branches/b/events?organizationId=org-1", token="w1"
        )
        self.assertEqual(len(events["events"]), 0)

    def test_comparison_never_reads_main_service_events(self) -> None:
        # Main-service events posted after the fork never reach a branch.
        status, _ = self.call(
            "/events",
            method="POST",
            payload={
                "eventId": "main-1",
                "organizationId": ORG1,
                "type": TYPE,
                "occurredAt": 10,
                "payload": {},
            },
            token="w1",
        )
        self.assertEqual(status, 201)
        status, body = self.compare(self.base_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["windows"], [])


if __name__ == "__main__":
    unittest.main()
