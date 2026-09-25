"""Request-scoped credentials and organization isolation regression tests.

Covers POST /auth/tokens registration, Bearer authentication (401),
organization containment (403), the read/write role matrix, snapshot and
branch ownership, restart clearing, and the byte-exact success bodies for
branch event/reservation commits and snapshot/branch creation.
"""

from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"


def event_body(event_id: str = "e1", organization_id: str = ORG1, **more: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "eventId": event_id,
        "organizationId": organization_id,
        "type": "incident.created",
        "occurredAt": 100,
        "payload": {},
    }
    body.update(more)
    return body


def reservation_body(
    reservation_id: str = "r1", organization_id: str = ORG1, **more: Any
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "organizationId": organization_id,
        "reservationId": reservation_id,
        "resourceId": "res-a",
        "quantity": 1,
        "capacity": 5,
    }
    body.update(more)
    return body


def decision_body(organization_id: str = ORG1) -> dict[str, Any]:
    return {
        "organizationId": organization_id,
        "type": "incident.created",
        "windowSize": 60,
        "threshold": 1,
    }


def allocation_body(organization_id: str = ORG1) -> dict[str, Any]:
    return {
        "organizationId": organization_id,
        "demands": [{"demandId": "d1", "units": 1, "priority": 0}],
        "resources": [{"resourceId": "r1", "capacity": 2}],
    }


def alert_body(organization_id: str = ORG1) -> dict[str, Any]:
    return {
        "organizationId": organization_id,
        "type": "incident.created",
        "windowSize": 60,
        "threshold": 1,
        "suppressionWindow": 100,
    }


class AuthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

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
        content_type: str | None = "application/json",
        authorization: str | None = "__NONE__",
    ) -> tuple[int, bytes]:
        headers: dict[str, str] = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if authorization != "__NONE__":
            if authorization is not None:
                headers["Authorization"] = authorization
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
        if raw_body is not None:
            body = raw_body
        elif payload is not None:
            body = json.dumps(payload).encode()
        else:
            body = None
        authorization = f"Bearer {token}" if token is not None else None
        status, raw = self.raw(
            path,
            method=method,
            body=body,
            content_type=content_type,
            authorization=authorization,
        )
        return status, json.loads(raw)

    def register(
        self,
        token: str,
        organization_id: str = ORG1,
        role: str = "write",
        *,
        payload: Any = None,
        raw_body: bytes | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, Any]:
        if payload is None and raw_body is None:
            payload = {
                "token": token,
                "organizationId": organization_id,
                "role": role,
            }
        return self.call(
            "/auth/tokens",
            method="POST",
            payload=payload,
            raw_body=raw_body,
            content_type=content_type,
        )

    # --------------------------------------------------------- health / 404

    def test_health_requires_no_credential(self) -> None:
        status, _ = self.raw("/health")
        self.assertEqual(status, 200)

    def test_registration_requires_no_credential(self) -> None:
        status, body = self.register("t1")
        self.assertEqual(status, 201)
        self.assertEqual(body["token"], "t1")

    def test_unknown_path_is_404_without_credential(self) -> None:
        status, body = self.raw("/missing")
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"], "not_found")

    # ------------------------------------------------------- POST /auth/tokens

    def test_register_returns_201_compact_sorted_echo_with_newline(self) -> None:
        # Keys deliberately out of order; the response must echo all three,
        # compactly, sorted by code point, with exactly one trailing newline.
        status, raw = self.raw(
            "/auth/tokens",
            method="POST",
            body=json.dumps(
                {"role": "write", "organizationId": ORG1, "token": "tok-1"}
            ).encode(),
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            raw,
            b'{"organizationId":"org-1","role":"write","token":"tok-1"}\n',
        )
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)

    def test_register_accepts_field_order_variations(self) -> None:
        ordered = [
            {"token": "t1", "organizationId": "o", "role": "read"},
            {"role": "read", "token": "t2", "organizationId": "o"},
            {"organizationId": "o", "role": "read", "token": "t3"},
        ]
        for payload in ordered:
            with self.subTest(payload=payload):
                status, body = self.call(
                    "/auth/tokens", method="POST", payload=payload
                )
                self.assertEqual(status, 201)
                self.assertEqual(body, payload)

    def test_identical_resubmission_is_200_and_not_duplicated(self) -> None:
        first_status, _ = self.register("dup", ORG1, "write")
        second_status, raw = self.raw(
            "/auth/tokens",
            method="POST",
            body=json.dumps(
                {"token": "dup", "organizationId": ORG1, "role": "write"}
            ).encode(),
        )
        self.assertEqual(first_status, 201)
        self.assertEqual(second_status, 200)
        self.assertEqual(
            raw, b'{"organizationId":"org-1","role":"write","token":"dup"}\n'
        )
        # A third identical submission is still 200 (idempotent).
        third_status, _ = self.register("dup", ORG1, "write")
        self.assertEqual(third_status, 200)

    def test_same_token_different_fields_is_auth_conflict(self) -> None:
        self.register("c1", ORG1, "write")
        status, body = self.register("c1", ORG2, "write")
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "auth_conflict")
        status, body = self.register("c1", ORG1, "read")
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "auth_conflict")
        # The original binding still authorizes ORG1 write.
        status, _ = self.call(
            "/events", method="POST", payload=event_body("ce1"), token="c1"
        )
        self.assertEqual(status, 201)

    def test_registration_validation_errors_are_422_and_store_nothing(self) -> None:
        bad_payloads: list[Any] = [
            {"organizationId": ORG1, "role": "write"},  # missing token
            {"token": "x", "role": "write"},  # missing organizationId
            {"token": "x", "organizationId": ORG1},  # missing role
            {"token": "x", "organizationId": ORG1, "role": "write", "extra": 1},
            {"token": "", "organizationId": ORG1, "role": "write"},
            {"token": "   ", "organizationId": ORG1, "role": "write"},
            {"token": 9, "organizationId": ORG1, "role": "write"},
            {"token": None, "organizationId": ORG1, "role": "write"},
            {"token": "x", "organizationId": "", "role": "write"},
            {"token": "x", "organizationId": 42, "role": "write"},
            {"token": "x", "organizationId": ORG1, "role": "admin"},
            {"token": "x", "organizationId": ORG1, "role": "Write"},
            {"token": "x", "organizationId": ORG1, "role": ""},
            [],
            "x",
            42,
            None,
            True,
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                status, body = self.call(
                    "/auth/tokens",
                    method="POST",
                    raw_body=json.dumps(payload).encode(),
                )
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

        # Nothing registered: every token above is still unauthenticated.
        status, _ = self.call("/events?organizationId=org-1", token="x")
        self.assertEqual(status, 401)

    def test_registration_media_type_and_json_errors(self) -> None:
        valid = {"token": "m1", "organizationId": ORG1, "role": "write"}
        status, body = self.register("m1", content_type=None)
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")
        status, body = self.register("m1", content_type="text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")
        status, body = self.register("m1", raw_body=b"{not json", content_type="application/json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")
        # The token from the failed media/JSON attempts was never registered.
        status, _ = self.call("/events?organizationId=org-1", token="m1")
        self.assertEqual(status, 401)

    # ------------------------------------------------------------ 401 forms

    def test_protected_endpoints_reject_missing_header(self) -> None:
        protected = [
            ("GET", "/events?organizationId=org-1", None),
            ("GET", "/events/aggregate?organizationId=org-1&type=t&windowSize=1", None),
            ("GET", "/events/replay?organizationId=org-1&asOf=1", None),
            (
                "GET",
                "/events/replay/compare?organizationId=org-1&fromAsOf=0&toAsOf=1",
                None,
            ),
            ("GET", "/events/region?organizationId=org-1&region=north", None),
            (
                "GET",
                "/events/region/aggregate"
                "?organizationId=org-1&region=north&type=t&windowSize=1",
                None,
            ),
            ("GET", "/reservations?organizationId=org-1", None),
            ("GET", "/alerts?organizationId=org-1", None),
            ("GET", "/snapshots", None),
            ("GET", "/branches/br-1", None),
            ("POST", "/events", event_body()),
            ("POST", "/reservations", reservation_body()),
            ("POST", "/decisions/evaluate", decision_body()),
            ("POST", "/decisions/allocate", allocation_body()),
            ("POST", "/alerts/evaluate", alert_body()),
            ("POST", "/snapshots", {"snapshotId": "s1"}),
            ("POST", "/branches", {"branchId": "b1", "snapshotId": "s1"}),
        ]
        for method, path, payload in protected:
            with self.subTest(method=method, path=path):
                status, body = self.call(path, method=method, payload=payload)
                self.assertEqual(status, 401)
                self.assertEqual(body["error"], "unauthorized")

    def test_bad_authorization_headers_are_401(self) -> None:
        self.register("good")
        cases: list[tuple[str, bytes | None]] = [
            ("Basic dXNlcjpwYXNz", None),
            ("Bearer", None),  # scheme without token
            ("Bearer ", None),
            ("Bearer  good", None),  # extra space -> two empty-ish tokens
            ("Bearer good extra", None),
            ("bearer good", None),  # scheme is case sensitive
            ("Token good", None),
        ]
        for header_value, _ in cases:
            with self.subTest(header=header_value):
                status, body = self.raw(
                    "/events?organizationId=org-1",
                    authorization=header_value,
                )
                self.assertEqual(status, 401)
                self.assertEqual(json.loads(body)["error"], "unauthorized")

    def test_unregistered_token_is_401(self) -> None:
        self.register("real")
        status, body = self.call(
            "/events?organizationId=org-1", token="forged"
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_restart_clears_tokens(self) -> None:
        self.register("t1")
        status, _ = self.call("/events?organizationId=org-1", token="t1")
        self.assertEqual(status, 200)

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{fresh.server_port}/events?organizationId=org-1",
                headers={"Authorization": "Bearer t1"},
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=2)
            self.assertEqual(raised.exception.code, 401)
            raised.exception.close()
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # ------------------------------------------------- organization containment

    def test_cross_organization_reads_are_403(self) -> None:
        self.register("w1", ORG1, "write")
        self.register("w2", ORG2, "write")
        # ORG1 owns an event and a reservation.
        self.assertEqual(
            self.call("/events", method="POST", payload=event_body("o1"), token="w1")[
                0
            ],
            201,
        )
        self.assertEqual(
            self.call(
                "/reservations", method="POST", payload=reservation_body("ro1"), token="w1"
            )[0],
            201,
        )

        # An ORG2 credential cannot read ORG1 data through any read entry.
        read_paths = [
            "/events?organizationId=org-1",
            "/events/replay?organizationId=org-1&asOf=999",
            "/events/replay/compare?organizationId=org-1&fromAsOf=0&toAsOf=999",
            "/events/region?organizationId=org-1&region=north",
            "/events/region/aggregate?organizationId=org-1&region=north&type=t&windowSize=1",
            "/events/aggregate?organizationId=org-1&type=t&windowSize=1",
            "/reservations?organizationId=org-1",
            "/alerts?organizationId=org-1",
        ]
        for path in read_paths:
            with self.subTest(path=path):
                status, body = self.call(path, token="w2")
                self.assertEqual(status, 403)
                self.assertEqual(body["error"], "forbidden")

        # Querying one's own organization works and stays isolated.
        status, body = self.call("/events?organizationId=org-2", token="w2")
        self.assertEqual(status, 200)
        self.assertEqual(body["events"], [])

    def test_cross_organization_body_writes_are_403_and_never_write(self) -> None:
        self.register("w1", ORG1, "write")
        self.register("w2", ORG2, "write")

        status, body = self.call(
            "/events", method="POST", payload=event_body("x1", ORG1), token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

        status, body = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body("xr1", ORG1),
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

        status, body = self.call(
            "/alerts/evaluate", method="POST", payload=alert_body(ORG1), token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

        # No ORG1 state was created by the rejected requests.
        status, body = self.call("/events?organizationId=org-1", token="w1")
        self.assertEqual(body["events"], [])
        status, body = self.call("/reservations?organizationId=org-1", token="w1")
        self.assertEqual(body["reservations"], [])
        status, body = self.call("/alerts?organizationId=org-1", token="w1")
        self.assertEqual(body["alerts"], [])

    def test_read_token_cannot_reach_write_endpoints_even_for_own_org(self) -> None:
        self.register("r1", ORG1, "read")
        self.register("w1", ORG1, "write")
        # Give the org an event so a decision/alert would do real work.
        self.assertEqual(
            self.call("/events", method="POST", payload=event_body("e1"), token="w1")[
                0
            ],
            201,
        )

        write_calls = [
            ("/events", event_body("e2")),
            ("/reservations", reservation_body("r2")),
            ("/alerts/evaluate", alert_body()),
            ("/snapshots", {"snapshotId": "s1"}),
        ]
        for path, payload in write_calls:
            with self.subTest(path=path):
                status, body = self.call(path, method="POST", payload=payload, token="r1")
                self.assertEqual(status, 403)
                self.assertEqual(body["error"], "forbidden")

        # Branches cannot be created by a read token either.
        self.assertEqual(
            self.call(
                "/snapshots", method="POST", payload={"snapshotId": "s1"}, token="w1"
            )[0],
            201,
        )
        status, body = self.call(
            "/branches",
            method="POST",
            payload={"branchId": "b1", "snapshotId": "s1"},
            token="r1",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_read_role_is_forbidden_on_write_entry_before_body_validation(self) -> None:
        self.register("r1", ORG1, "read")
        # Role is an entry-point decision, so it outranks media type, JSON
        # syntax, and body-shape validation for a read credential.
        status, body = self.call(
            "/events", method="POST", payload=event_body(), token="r1",
            content_type=None,
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        status, body = self.call(
            "/events",
            method="POST",
            raw_body=b'{"eventId": ',
            token="r1",
        )
        self.assertEqual(status, 403)
        status, body = self.call(
            "/events", method="POST", payload=[event_body()], token="r1"
        )
        self.assertEqual(status, 403)
        status, body = self.call(
            "/alerts/evaluate", method="POST", raw_body=b"{bad", token="r1"
        )
        self.assertEqual(status, 403)
        # A write credential still sees the underlying validation errors.
        self.register("w1", ORG1, "write")
        status, body = self.call(
            "/events",
            method="POST",
            raw_body=b'{"eventId": ',
            token="w1",
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_read_token_may_use_stateless_and_query_endpoints(self) -> None:
        self.register("w1", ORG1, "write")
        self.register("r1", ORG1, "read")
        self.call("/events", method="POST", payload=event_body("e1"), token="w1")
        self.call(
            "/reservations",
            method="POST",
            payload=reservation_body("rr1"),
            token="w1",
        )

        for path, payload in (
            ("/decisions/evaluate", decision_body()),
            ("/decisions/allocate", allocation_body()),
        ):
            with self.subTest(path=path):
                status, _ = self.call(path, method="POST", payload=payload, token="r1")
                self.assertEqual(status, 200)
        for path in (
            "/events?organizationId=org-1",
            "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=60",
            "/events/replay?organizationId=org-1&asOf=999",
            "/events/replay/compare?organizationId=org-1&fromAsOf=0&toAsOf=999",
            "/events/region?organizationId=org-1&region=north",
            "/reservations?organizationId=org-1",
            "/alerts?organizationId=org-1",
            "/snapshots",
        ):
            with self.subTest(path=path):
                status, _ = self.call(path, token="r1")
                self.assertEqual(status, 200)

    def test_read_token_rejected_on_write_with_cross_org_is_still_403(self) -> None:
        # Role and organization are both out of scope; the result is 403
        # regardless of which rule fires first, with no write performed.
        self.register("r2", ORG2, "read")
        status, body = self.call(
            "/events", method="POST", payload=event_body("e9", ORG1), token="r2"
        )
        self.assertEqual(status, 403)
        # The read-only token also cannot enumerate ORG1 snapshots.
        status, body = self.call("/snapshots", token="r2")
        self.assertEqual(status, 200)
        self.assertEqual(body["snapshots"], [])

    # --------------------------------------------------- snapshot / branch scope

    def _seed_org1_snapshot_and_branch(self) -> None:
        self.register("w1", ORG1, "write")
        self.call("/events", method="POST", payload=event_body("e1"), token="w1")
        self.call(
            "/reservations",
            method="POST",
            payload=reservation_body("r1"),
            token="w1",
        )
        self.assertEqual(
            self.call(
                "/snapshots", method="POST", payload={"snapshotId": "s1"}, token="w1"
            )[0],
            201,
        )
        self.assertEqual(
            self.call(
                "/branches",
                method="POST",
                payload={"branchId": "b1", "snapshotId": "s1"},
                token="w1",
            )[0],
            201,
        )

    def test_snapshot_listing_is_scoped_to_token_organization(self) -> None:
        self._seed_org1_snapshot_and_branch()
        self.register("w2", ORG2, "write")
        status, body = self.call("/snapshots", token="w2")
        self.assertEqual(status, 200)
        self.assertEqual(body["snapshots"], [])
        status, body = self.call("/snapshots", token="w1")
        self.assertEqual([s["snapshotId"] for s in body["snapshots"]], ["s1"])

    def test_branch_cannot_fork_another_organizations_snapshot(self) -> None:
        self._seed_org1_snapshot_and_branch()
        self.register("w2", ORG2, "write")
        # The snapshot name exists, but belongs to ORG1: forbidden, not the
        # 404 an unknown snapshot would yield.
        status, body = self.call(
            "/branches",
            method="POST",
            payload={"branchId": "b2", "snapshotId": "s1"},
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # No branch was created for ORG2.
        status, _ = self.call("/branches/b2", token="w2")
        self.assertEqual(status, 404)

    def test_branch_access_is_scoped_to_owner_organization(self) -> None:
        self._seed_org1_snapshot_and_branch()
        self.register("w2", ORG2, "write")
        for path in (
            "/branches/b1",
            "/branches/b1/events?organizationId=org-2",
            "/branches/b1/reservations?organizationId=org-2",
            "/branches/b1/events/aggregate?organizationId=org-2&type=t&windowSize=1",
        ):
            with self.subTest(path=path):
                status, body = self.call(path, token="w2")
                self.assertEqual(status, 403)
                self.assertEqual(body["error"], "forbidden")

    def test_branch_write_requires_write_role_and_owner_org(self) -> None:
        self._seed_org1_snapshot_and_branch()
        self.register("r1", ORG1, "read")
        self.register("w2", ORG2, "write")

        # Read role on the owning org: 403 on branch writes.
        status, body = self.call(
            "/branches/b1/events",
            method="POST",
            payload=event_body("be1"),
            token="r1",
        )
        self.assertEqual(status, 403)
        status, body = self.call(
            "/branches/b1/reservations",
            method="POST",
            payload=reservation_body("br1"),
            token="r1",
        )
        self.assertEqual(status, 403)

        # Another org's write token is also forbidden in the branch.
        status, body = self.call(
            "/branches/b1/events",
            method="POST",
            payload=event_body("be2", ORG2),
            token="w2",
        )
        self.assertEqual(status, 403)

        # The branch is still at its forked state (one seeded event).
        status, body = self.call(
            "/branches/b1/events?organizationId=org-1", token="w1"
        )
        self.assertEqual(len(body["events"]), 1)

    def test_snapshot_captures_only_creator_organization_state(self) -> None:
        self.register("w1", ORG1, "write")
        self.register("w2", ORG2, "write")
        self.call("/events", method="POST", payload=event_body("o1"), token="w1")
        self.call("/events", method="POST", payload=event_body("o2", ORG2), token="w2")
        self.call(
            "/reservations",
            method="POST",
            payload=reservation_body("ro1"),
            token="w1",
        )
        status, body = self.call(
            "/snapshots", method="POST", payload={"snapshotId": "s1"}, token="w1"
        )
        self.assertEqual(status, 201)
        # Only ORG1's one event and one reservation are captured.
        self.assertEqual(body["events"], 1)
        self.assertEqual(body["reservations"], 1)

    # ----------------------------------------------- byte-exact success bodies

    def test_branch_event_commit_success_body_is_newline_terminated(self) -> None:
        self._seed_org1_snapshot_and_branch()
        body = json.dumps(event_body("be1")).encode()
        status, raw = self.raw(
            "/branches/b1/events",
            method="POST",
            body=body,
            authorization="Bearer w1",
        )
        self.assertEqual(status, 201)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(json.loads(raw[:-1]), event_body("be1"))

    def test_branch_reservation_commit_success_body_is_newline_terminated(self) -> None:
        self._seed_org1_snapshot_and_branch()
        body = json.dumps(reservation_body("br2")).encode()
        status, raw = self.raw(
            "/branches/b1/reservations",
            method="POST",
            body=body,
            authorization="Bearer w1",
        )
        self.assertEqual(status, 201)
        self.assertTrue(raw.endswith(b"\n"))
        view = json.loads(raw[:-1])
        self.assertEqual(view["reservationId"], "br2")
        self.assertEqual(view["occupied"], 2)
        self.assertEqual(view["remaining"], 3)

    def test_snapshot_creation_success_body_has_no_newline(self) -> None:
        self.register("w1", ORG1, "write")
        status, raw = self.raw(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": "s1"}).encode(),
            authorization="Bearer w1",
        )
        self.assertEqual(status, 201)
        self.assertFalse(raw.endswith(b"\n"))
        self.assertEqual(
            raw, b'{"events":0,"reservations":0,"resources":0,"snapshotId":"s1"}'
        )

    def test_branch_creation_success_body_has_no_newline(self) -> None:
        self.register("w1", ORG1, "write")
        self.call(
            "/snapshots", method="POST", payload={"snapshotId": "s1"}, token="w1"
        )
        status, raw = self.raw(
            "/branches",
            method="POST",
            body=json.dumps({"branchId": "b1", "snapshotId": "s1"}).encode(),
            authorization="Bearer w1",
        )
        self.assertEqual(status, 201)
        self.assertFalse(raw.endswith(b"\n"))
        self.assertEqual(
            raw,
            b'{"branchId":"b1","events":0,"reservations":0,"resources":0,"snapshotId":"s1"}',
        )

    # ------------------------------------------------------------ concurrency

    def test_forbidden_writes_never_mutate_under_concurrency(self) -> None:
        self.register("w1", ORG1, "write")
        self.register("r1", ORG1, "read")
        # Seed one real event so a read can observe state.
        self.call("/events", method="POST", payload=event_body("seed"), token="w1")

        def attempt(index: int) -> int:
            # Read token attempts a write it must never win.
            status, _ = self.call(
                "/events",
                method="POST",
                payload=event_body(f"denied-{index}"),
                token="r1",
            )
            return status

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(attempt, range(40)))
        self.assertTrue(all(status == 403 for status in statuses))

        # The ledger still holds only the seeded event.
        status, body = self.call("/events?organizationId=org-1", token="w1")
        self.assertEqual(status, 200)
        self.assertEqual([e["eventId"] for e in body["events"]], ["seed"])

    def test_unicode_branch_id_is_accessible_to_owner_only(self) -> None:
        self._seed_org1_snapshot_and_branch()
        self.register("w2", ORG2, "write")
        self.assertEqual(
            self.call(
                "/branches",
                method="POST",
                payload={"branchId": "br-ä", "snapshotId": "s1"},
                token="w1",
            )[0],
            201,
        )
        status, body = self.call(f"/branches/{quote('br-ä')}", token="w1")
        self.assertEqual(status, 200)
        self.assertEqual(body["branchId"], "br-ä")
        status, body = self.call(f"/branches/{quote('br-ä')}", token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")


if __name__ == "__main__":
    unittest.main()
