"""Regression tests for POST /branches/compare/events.

The window-level branch comparison already existed; this locks down the new
read-only entry point that aligns two branches' events by eventId:

- identifiers only on one side land in ``leftOnly``/``rightOnly``; shared
  identifiers land in ``same`` when every other field matches (payload
  compared by content) and in ``diff`` otherwise, with the differing field
  names listed per entry;
- every group is sorted by code point, counts sit next to their groups,
  both-branches-empty yields four empty groups, and identical submissions
  are byte-for-byte stable;
- 401 / 403 / 404 (branch_not_found) / 422 / 415 / 400 follow the fixed
  ordering (organization first, then left before right), both roles may
  call it, and nothing is ever written or implicitly created, including
  across organizations.
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


def event_body(
    event_id: str,
    occurred_at: int,
    organization_id: str = ORG1,
    *,
    event_type: str = EVENT_TYPE,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": event_type,
        "occurredAt": occurred_at,
        "payload": {} if payload is None else payload,
    }


class BranchCompareEventsTest(unittest.TestCase):
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
        self,
        branch: str,
        event_id: str,
        occurred_at: int,
        token: str = "w1",
        **kwargs: Any,
    ) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        status, _ = self.call(
            f"/branches/{branch}/events",
            method="POST",
            token=token,
            payload=event_body(event_id, occurred_at, organization_id, **kwargs),
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
            "/branches/compare/events",
            method="POST",
            payload=self.compare_payload() if payload is None else payload,
            token=token,
        )

    # ------------------------------------------------------------- happy paths

    def test_events_align_by_identifier_into_four_groups(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "shared-ok", 5)
        self.add_branch_event("left", "shared-diff", 10)
        self.add_branch_event("left", "only-l", 15)
        self.add_branch_event("right", "shared-ok", 5)
        self.add_branch_event("right", "shared-diff", 20)  # occurredAt differs
        self.add_branch_event("right", "only-r", 25)

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["only-l"])
        self.assertEqual(body["leftOnlyCount"], 1)
        self.assertEqual(body["rightOnly"], ["only-r"])
        self.assertEqual(body["rightOnlyCount"], 1)
        self.assertEqual(body["same"], ["shared-ok"])
        self.assertEqual(body["sameCount"], 1)
        self.assertEqual(
            body["diff"],
            [{"eventId": "shared-diff", "fields": ["occurredAt"]}],
        )
        self.assertEqual(body["diffCount"], 1)

    def test_diff_lists_every_differing_field_sorted_by_code_point(self) -> None:
        self.fork_branches()
        self.add_branch_event(
            "left", "e1", 10, event_type="a.type", payload={"severity": "low"}
        )
        self.add_branch_event(
            "right", "e1", 20, event_type="b.type", payload={"severity": "high"}
        )

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            body["diff"],
            [
                {
                    "eventId": "e1",
                    "fields": ["occurredAt", "payload", "type"],
                }
            ],
        )
        self.assertEqual(body["same"], [])
        self.assertEqual(body["diffCount"], 1)

    def test_payload_is_compared_by_content_not_serialization(self) -> None:
        self.fork_branches()
        self.add_branch_event(
            "left", "e1", 10, payload={"a": 1, "b": {"c": [1, 2]}}
        )
        # Same content, different key order in the submitted JSON object.
        status, _ = self.call(
            "/branches/right/events",
            method="POST",
            payload={
                "eventId": "e1",
                "organizationId": ORG1,
                "type": EVENT_TYPE,
                "occurredAt": 10,
                "payload": {"b": {"c": [1, 2]}, "a": 1},
            },
        )
        self.assertEqual(status, 201)

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["e1"])
        self.assertEqual(body["diff"], [])

    def test_groups_and_identifiers_are_sorted_by_code_point(self) -> None:
        self.fork_branches()
        for event_id in ("b-2", "A-1", "a-3"):
            self.add_branch_event("left", event_id, 5)
        for event_id in ("z-1", "B-2", "y-3"):
            self.add_branch_event("right", event_id, 5)

        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["A-1", "a-3", "b-2"])
        self.assertEqual(body["rightOnly"], ["B-2", "y-3", "z-1"])
        self.assertEqual(body["leftOnlyCount"], 3)
        self.assertEqual(body["rightOnlyCount"], 3)

    def test_same_branch_on_both_sides_is_legal_and_fully_same(self) -> None:
        self.fork_branches(branches=("left",))
        self.add_branch_event("left", "e1", 5)
        self.add_branch_event("left", "e2", 70, payload={"k": "v"})

        status, body = self.compare(self.compare_payload(right="left"))
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["e1", "e2"])
        self.assertEqual(body["sameCount"], 2)
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], [])
        self.assertEqual(body["diff"], [])
        self.assertEqual(body["leftOnlyCount"], 0)
        self.assertEqual(body["rightOnlyCount"], 0)
        self.assertEqual(body["diffCount"], 0)

    def test_both_branches_empty_yields_four_empty_groups(self) -> None:
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
        self.add_branch_event("left", "e1", 5)
        self.add_branch_event("right", "e1", 9)

        raw_body = json.dumps(self.compare_payload()).encode()
        status, raw = self.raw(
            "/branches/compare/events", method="POST", body=raw_body, token="w1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Integers stay integers.
        self.assertIn(b'"diffCount":1', raw)
        self.assertIn(b'"sameCount":0', raw)
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
        self.assertEqual(list(body["diff"][0]), ["eventId", "fields"])

    def test_repeated_submissions_are_byte_identical(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "e1", 5)
        self.add_branch_event("left", "e2", 65)
        self.add_branch_event("right", "e2", 65)
        self.add_branch_event("right", "e3", 120)
        raw_body = json.dumps(self.compare_payload()).encode()
        raws = [
            self.raw(
                "/branches/compare/events", method="POST", body=raw_body, token="w1"
            )[1]
            for _ in range(3)
        ]
        self.assertEqual(len(set(raws)), 1)

    # -------------------------------------------------------------- read-only

    def test_compare_is_read_only(self) -> None:
        self.fork_branches()
        self.add_branch_event("left", "e1", 5)
        self.add_branch_event("right", "e2", 10)

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
        status, events = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(events["events"], [])
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
        self.add_branch_event("left", "e1", 5)
        status, body = self.compare(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["e1"])

    def test_missing_or_unregistered_credential_is_401(self) -> None:
        self.fork_branches()
        raw_body = json.dumps(self.compare_payload()).encode()
        status, body = self.raw(
            "/branches/compare/events", method="POST", body=raw_body, token=None
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        status, body = self.raw(
            "/branches/compare/events",
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
        self.add_branch_event("left", "e1", 5)
        self.add_branch_event("foreign-a", "f1", 65, token="w2")
        self.add_branch_event("foreign-b", "f1", 65, token="w2")
        self.add_branch_event("foreign-b", "f2", 125, token="w2")

        # ORG2 compares its own two branches and sees only ORG2 events.
        payload = {
            "organizationId": ORG2,
            "left": "foreign-a",
            "right": "foreign-b",
        }
        status, body = self.call(
            "/branches/compare/events", method="POST", payload=payload, token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["same"], ["f1"])
        self.assertEqual(body["leftOnly"], [])
        self.assertEqual(body["rightOnly"], ["f2"])

        # ORG1's comparison is unchanged and never sees ORG2 events.
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["leftOnly"], ["e1"])
        self.assertEqual(body["same"], [])

        # Neither organization can reach across the boundary in either
        # direction, even with a read credential.
        status, body = self.compare(
            self.compare_payload(left="left", right="foreign-a"), token="r1"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        status, body = self.call(
            "/branches/compare/events",
            method="POST",
            payload={
                "organizationId": ORG2,
                "left": "foreign-a",
                "right": "left",
            },
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

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
            {**valid, "type": EVENT_TYPE},  # window-compare fields not allowed
            {**valid, "windowSize": 60},
            {**valid, "left": ""},
            {**valid, "right": "   "},
            {**valid, "left": 7},
            {**valid, "right": None},
            {**valid, "organizationId": ""},
            {**valid, "organizationId": 3},
            [],
            "x",
            42,
            True,
            None,
        ]
        for bad_payload in bad_payloads:
            with self.subTest(bad_payload=bad_payload):
                status, parsed = self.call(
                    "/branches/compare/events",
                    method="POST",
                    raw_body=json.dumps(bad_payload).encode(),
                    token="w1",
                )
                self.assertEqual(status, 422)
                self.assertEqual(parsed["error"], "validation_error")

        # No failed validation created a branch or altered the real ones.
        self.assertEqual(self.call("/branches/left")[1]["events"], 0)
        self.assertEqual(self.call("/branches/right")[1]["events"], 0)

    def test_media_type_and_json_errors(self) -> None:
        self.fork_branches()
        raw_body = json.dumps(self.compare_payload()).encode()

        status, body = self.call(
            "/branches/compare/events",
            method="POST",
            raw_body=raw_body,
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/branches/compare/events",
            method="POST",
            raw_body=raw_body,
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.call(
            "/branches/compare/events",
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
