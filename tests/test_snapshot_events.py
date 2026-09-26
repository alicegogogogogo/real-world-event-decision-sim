"""Regression tests for GET /snapshots/{snapshotId}/events.

The snapshot event comparisons already aligned two snapshots' captured
events; this locks down the new read-only entry point that lists one
snapshot's captured events for the caller's organization:

- the ``events`` array carries one row per event captured in the snapshot,
  sorted by ``occurredAt`` ascending and then ``eventId`` in Unicode
  code-point order (exactly like ``GET /events``), each row reporting the
  five event fields (``eventId``, ``organizationId``, ``type``,
  ``occurredAt``, ``payload``) — the same field names the event comparison
  uses;
- the response echoes ``organizationId`` and ``snapshotId``, is compact,
  key-sorted JSON with integer timestamps and one trailing newline, and
  identical requests are byte-for-byte identical even with main-service
  writes landing between reads; a snapshot with no events yields an empty
  row array;
- the rows share the comparison's exact contract: a snapshot compared with
  itself via POST /snapshots/compare/events reports every listed event in
  ``same`` with zero diffs, and other organizations' events never
  contribute;
- snapshots are immutable: later main-service writes never change the
  listing; the verdict order is fixed — 401 (credential) before 422
  (query shape) before 403 (organization, then foreign snapshot) before
  404 (snapshot_not_found) — and nothing is ever written or implicitly
  created, including the main-service ledger, reservation inventory, and
  alert state; restarting clears snapshots and events.
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


def event_body(
    event_id: str,
    *,
    occurred_at: int,
    event_type: str = "incident.created",
    payload: dict[str, Any] | None = None,
    organization_id: str = ORG1,
) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": event_type,
        "occurredAt": occurred_at,
        "payload": {} if payload is None else payload,
    }


class SnapshotEventListTest(unittest.TestCase):
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
    ) -> tuple[int, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        status, raw = self.raw(path, method=method, body=body, token=token)
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

    def add_event(self, event_id: str, token: str = "w1", **kwargs: Any) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        status, _ = self.call(
            "/events",
            method="POST",
            token=token,
            payload=event_body(
                event_id, organization_id=organization_id, **kwargs
            ),
        )
        self.assertEqual(status, 201)

    def capture(
        self, snapshot_id: str = "s1", *, token: str = "w1"
    ) -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def list_events(
        self, snapshot: str = "s1", *, token: str | None = "w1", org: str = ORG1
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/events?organizationId={org}", token=token
        )

    def main_events(self, org: str = ORG1, *, token: str | None = "w1") -> list:
        status, body = self.call(f"/events?organizationId={org}", token=token)
        self.assertEqual(status, 200)
        return body["events"]

    # ------------------------------------------------------------- happy paths

    def test_rows_echo_identifiers_and_report_five_fields(self) -> None:
        self.add_event(
            "evt-1", occurred_at=100, payload={"severity": "low"}
        )
        self.add_event(
            "evt-2",
            occurred_at=200,
            event_type="incident.updated",
            payload={"region": "north"},
        )
        self.capture()

        status, body = self.list_events()
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(
            body["events"],
            [
                {
                    "eventId": "evt-1",
                    "organizationId": ORG1,
                    "type": "incident.created",
                    "occurredAt": 100,
                    "payload": {"severity": "low"},
                },
                {
                    "eventId": "evt-2",
                    "organizationId": ORG1,
                    "type": "incident.updated",
                    "occurredAt": 200,
                    "payload": {"region": "north"},
                },
            ],
        )

    def test_rows_sort_by_occurred_at_then_event_id(self) -> None:
        # Insert deliberately out of order, including equal timestamps.
        self.add_event("evt-c", occurred_at=200)
        self.add_event("evt-a", occurred_at=100)
        self.add_event("evt-b2", occurred_at=100)
        self.add_event("evt-b1", occurred_at=100)
        self.add_event("evt-B", occurred_at=100)
        self.add_event("evt-1", occurred_at=50)
        self.capture()

        status, body = self.list_events()
        self.assertEqual(status, 200)
        self.assertEqual(
            [(row["occurredAt"], row["eventId"]) for row in body["events"]],
            [
                (50, "evt-1"),
                (100, "evt-B"),
                (100, "evt-a"),
                (100, "evt-b1"),
                (100, "evt-b2"),
                (200, "evt-c"),
            ],
        )

    def test_snapshot_without_events_returns_empty_array(self) -> None:
        self.capture()
        status, body = self.list_events()
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {"organizationId": ORG1, "snapshotId": "s1", "events": []},
        )

    def test_listing_reflects_capture_time_only(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()

        # Events committed after capture never enter the snapshot, and they
        # do not move the ordering of the captured rows.
        self.add_event("evt-0", occurred_at=10)
        self.add_event("evt-2", occurred_at=200)
        status, body = self.list_events()
        self.assertEqual(status, 200)
        self.assertEqual([row["eventId"] for row in body["events"]], ["evt-1"])

    def test_payload_compares_by_content_not_key_order(self) -> None:
        # Unusual payload key order on the wire must not affect the rows.
        raw_payload = b'{"eventId":"evt-1","organizationId":"org-1","type":"incident.created","occurredAt":100,"payload":{"z":1,"a":2}}'
        status, _ = self.raw("/events", method="POST", body=raw_payload, token="w1")
        self.assertEqual(status, 201)
        self.capture()
        status, body = self.list_events()
        self.assertEqual(status, 200)
        self.assertEqual(body["events"][0]["payload"], {"z": 1, "a": 2})

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()

        status, raw = self.raw(
            "/snapshots/s1/events?organizationId=org-1", token="w1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        # Timestamps stay integers.
        self.assertIn(b'"occurredAt":100', raw)
        # Keys are sorted by code point at every level.
        body = json.loads(raw)
        self.assertEqual(list(body), ["events", "organizationId", "snapshotId"])
        self.assertEqual(
            list(body["events"][0]),
            ["eventId", "occurredAt", "organizationId", "payload", "type"],
        )

    def test_repeated_requests_are_byte_identical_and_do_not_pollute(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.add_event("evt-2", occurred_at=50)
        self.capture()
        # Main-service writes between reads must not perturb bytes or leak
        # into the snapshot; neither read affects the other.
        first = self.raw("/snapshots/s1/events?organizationId=org-1", token="w1")[1]
        self.add_event("evt-3", occurred_at=10)
        self.add_event("evt-4", occurred_at=200)
        rest = [
            self.raw("/snapshots/s1/events?organizationId=org-1", token="w1")[1]
            for _ in range(3)
        ]
        self.assertTrue(all(chunk == first for chunk in rest))

    # ------------------------------------------- consistency with comparison

    def test_self_comparison_agrees_item_by_item_with_zero_diffs(self) -> None:
        self.add_event("evt-1", occurred_at=200, payload={"severity": "high"})
        self.add_event("evt-2", occurred_at=100, event_type="x.raised")
        self.add_event("evt-3", occurred_at=100, payload={"region": "north"})
        captured = self.main_events()
        self.capture()

        status, listing = self.list_events()
        self.assertEqual(status, 200)
        status, comparison = self.call(
            "/snapshots/compare/events",
            method="POST",
            payload={"organizationId": ORG1, "left": "s1", "right": "s1"},
        )
        self.assertEqual(status, 200)

        # Every captured event lands in ``same`` and nothing in the other
        # groups; the diff count is zero. ``same`` is ordered by eventId code
        # point while the listing is ordered by occurredAt then eventId, so
        # the id sets are compared order-independently.
        self.assertEqual(
            set(comparison["same"]),
            {row["eventId"] for row in captured},
        )
        self.assertEqual(comparison["sameCount"], len(captured))
        self.assertEqual(comparison["leftOnly"], [])
        self.assertEqual(comparison["leftOnlyCount"], 0)
        self.assertEqual(comparison["rightOnly"], [])
        self.assertEqual(comparison["rightOnlyCount"], 0)
        self.assertEqual(comparison["diff"], [])
        self.assertEqual(comparison["diffCount"], 0)

        # The listing rows match the captured events item by item, in the
        # same order and with every field identical.
        self.assertEqual(listing["events"], captured)

    def test_other_organizations_never_contribute(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.add_event("evt-2", occurred_at=150)
        self.capture("s1")
        # ORG2 commits its own events (the main ledger keys eventIds
        # globally, so they use distinct ids) and captures its own snapshot;
        # nothing leaks into ORG1's snapshot listing.
        self.add_event("evt-a", token="w2", occurred_at=50)
        self.add_event(
            "evt-b",
            token="w2",
            occurred_at=999,
            event_type="other.kind",
            payload={"leak": True},
        )
        self.capture("s2", token="w2")

        status, body = self.list_events()
        self.assertEqual(status, 200)
        self.assertEqual(
            [row["eventId"] for row in body["events"]],
            ["evt-1", "evt-2"],
        )
        self.assertTrue(
            all(row["organizationId"] == ORG1 for row in body["events"])
        )

        # ORG2's own snapshot listing sees only ORG2 captured events.
        status, body = self.list_events("s2", token="w2", org=ORG2)
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG2)
        self.assertEqual(body["snapshotId"], "s2")
        self.assertEqual(
            [(row["eventId"], row["occurredAt"]) for row in body["events"]],
            [("evt-a", 50), ("evt-b", 999)],
        )

    # -------------------------------------------------------------- read-only

    def test_listing_is_read_only(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()

        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, _ = self.list_events()
            self.assertEqual(status, 200)
        status, after = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

        # Main-service ledger, reservations and alerts are untouched.
        self.assertEqual(len(self.main_events()), 1)
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_list(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()
        status, body = self.list_events(token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["events"]), 1)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture()
        path = "/snapshots/s1/events?organizationId=org-1"
        # Missing header.
        status, body = self.raw(path, token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Unregistered token.
        status, body = self.raw(path, token="forged")
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        # Malformed Authorization header (non-Bearer scheme).
        request = Request(
            f"{self.base_url}{path}",
            headers={"Authorization": "Basic abc"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=5) as response:
                status, body = response.status, response.read()
        except HTTPError as error:
            status, body = error.code, error.read()
            error.close()
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    def test_credential_outranks_query_validation(self) -> None:
        self.capture()
        # A missing credential is 401 even when the query is also invalid.
        status, body = self.raw("/snapshots/s1/events", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- parameter 422

    def test_query_shape_errors_are_422(self) -> None:
        self.capture()
        for path in (
            "/snapshots/s1/events",
            "/snapshots/s1/events?organizationId=",
            "/snapshots/s1/events?organizationId=%20%20",
            "/snapshots/s1/events?organizationId=org-1&organizationId=org-1",
            "/snapshots/s1/events?organizationId=org-1&organizationId=org-2",
        ):
            with self.subTest(path=path):
                status, body = self.raw(path, token="w1")
                self.assertEqual(status, 422)
                self.assertEqual(json.loads(body)["error"], "validation_error")

        # No failed validation created or altered anything.
        self.assertEqual(self.call("/snapshots")[1]["snapshots"][0]["events"], 0)

    def test_query_validation_outranks_organization_and_snapshot(self) -> None:
        self.capture()
        # A malformed query is 422 even when the organization would also
        # mismatch and the snapshot name is unknown.
        status, body = self.raw(
            "/snapshots/ghost/events?organizationId=org-2&organizationId=org-2",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(self) -> None:
        self.capture()
        # An ORG2 credential naming ORG1 is rejected on the organization
        # decision even when the snapshot name does not exist anywhere.
        status, body = self.call(
            "/snapshots/ghost/events?organizationId=org-1", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        # The organization matches the credential, but the snapshot belongs
        # to another organization.
        status, body = self.list_events("s2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.list_events("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        # The failed lookup did not implicitly create the snapshot.
        self.assertEqual(self.call("/snapshots")[1]["snapshots"], [])

    def test_snapshot_lookup_outranks_snapshot_ownership(self) -> None:
        self.capture("s2", token="w2")
        # A missing snapshot is 404 even though a foreign snapshot exists.
        status, body = self.list_events("ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    # ---------------------------------------------------------------- restart

    def test_new_server_instance_has_no_snapshots_or_events(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{fresh.server_port}"
            payload = json.dumps(
                {"token": "tok-fresh", "organizationId": ORG1, "role": "read"}
            ).encode()
            request = Request(
                f"{base_url}/auth/tokens",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 201)
            request = Request(
                f"{base_url}/snapshots/s1/events?organizationId=org-1",
                headers={"Authorization": "Bearer tok-fresh"},
                method="GET",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    status = response.status
            except HTTPError as error:
                status = error.code
                self.assertEqual(json.loads(error.read())["error"], "snapshot_not_found")
                error.close()
            self.assertEqual(status, 404)
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
