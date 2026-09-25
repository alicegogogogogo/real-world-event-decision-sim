from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server


def make_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "eventId": "evt-1",
        "organizationId": "org-1",
        "type": "incident.created",
        "occurredAt": 100,
        "payload": {"severity": "low"},
    }
    event.update(overrides)
    return event


def make_credential(
    token: str = "tok-1", organization_id: str = "org-1", role: str = "write"
) -> dict[str, str]:
    return {"token": token, "organizationId": organization_id, "role": role}


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

    # ------------------------------------------------------------- helpers

    def request_raw(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        content_type: str | None = "application/json",
        token: str | None = None,
        authorization: str | None = None,
    ) -> tuple[int, bytes, Any]:
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if authorization is not None:
            headers["Authorization"] = authorization
        elif token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                return response.status, raw, json.loads(raw)
        except HTTPError as error:
            raw = error.read()
            try:
                return error.code, raw, json.loads(raw)
            finally:
                error.close()

    def request(self, path: str, **kwargs: Any) -> tuple[int, Any]:
        status, _, parsed = self.request_raw(path, **kwargs)
        return status, parsed

    def post_json(self, path: str, payload: Any, **kwargs: Any) -> tuple[int, Any]:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return self.request(path, method="POST", body=body, **kwargs)

    def register(
        self,
        token: str = "tok-1",
        organization_id: str = "org-1",
        role: str = "write",
    ) -> tuple[int, Any]:
        return self.post_json(
            "/auth/tokens", make_credential(token, organization_id, role)
        )

    def register_token(
        self, token: str, organization_id: str, role: str = "write"
    ) -> str:
        status, _ = self.register(token, organization_id, role)
        self.assertEqual(status, 201)
        return token

    def seed_org1(self) -> str:
        token = self.register_token("tok-w1", "org-1")
        status, _ = self.post_json("/events", make_event(), token=token)
        self.assertEqual(status, 201)
        status, _ = self.post_json(
            "/reservations",
            {
                "organizationId": "org-1",
                "reservationId": "res-1",
                "resourceId": "r-a",
                "quantity": 2,
                "capacity": 5,
            },
            token=token,
        )
        self.assertEqual(status, 201)
        return token

    # ------------------------------------------- POST /auth/tokens: success

    def test_register_returns_201_with_exact_echo_bytes(self) -> None:
        status, raw, body = self.request_raw(
            "/auth/tokens",
            method="POST",
            body=b'{"role":"write","token":"tok-1","organizationId":"org-1"}',
        )
        self.assertEqual(status, 201)
        # Compact JSON, keys in code-point order, exactly one trailing
        # newline — checked byte for byte.
        self.assertEqual(
            raw,
            b'{"organizationId":"org-1","role":"write","token":"tok-1"}\n',
        )
        self.assertEqual(
            body,
            {"token": "tok-1", "organizationId": "org-1", "role": "write"},
        )

    def test_register_read_role(self) -> None:
        status, body = self.register("tok-r", "org-1", "read")
        self.assertEqual(status, 201)
        self.assertEqual(body["role"], "read")

    def test_identical_replay_returns_200_without_duplicate(self) -> None:
        first_status, first_body = self.register()
        second_status, second_body = self.register()
        self.assertEqual(first_status, 201)
        self.assertEqual(second_status, 200)
        self.assertEqual(first_body, second_body)

    def test_replay_response_also_ends_with_newline(self) -> None:
        self.register()
        status, raw, _ = self.request_raw(
            "/auth/tokens",
            method="POST",
            body=json.dumps(make_credential()).encode(),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            raw, b'{"organizationId":"org-1","role":"write","token":"tok-1"}\n'
        )

    def test_same_token_different_fields_is_409_auth_conflict(self) -> None:
        self.assertEqual(self.register()[0], 201)
        for override in (
            {"organization_id": "org-2"},
            {"role": "read"},
        ):
            with self.subTest(override=override):
                kwargs = {"token": "tok-1", "organization_id": "org-1", "role": "write"}
                kwargs.update(override)
                status, body = self.register(**kwargs)
                self.assertEqual(status, 409)
                self.assertEqual(body["error"], "auth_conflict")
        # The original registration is untouched and still authenticates.
        status, _ = self.request("/events?organizationId=org-1", token="tok-1")
        self.assertEqual(status, 200)

    def test_different_tokens_same_organization_coexist(self) -> None:
        self.assertEqual(self.register("tok-a", "org-1")[0], 201)
        self.assertEqual(self.register("tok-b", "org-1", "read")[0], 201)

    # ---------------------------------------- POST /auth/tokens: validation

    def test_register_field_errors_are_422_and_nothing_is_stored(self) -> None:
        bad_bodies = [
            {},
            {"token": "tok-1"},
            {"token": "tok-1", "organizationId": "org-1"},
            {"token": "tok-1", "role": "write"},
            {"organizationId": "org-1", "role": "write"},
            {**make_credential(), "extra": 1},
            make_credential(token=""),
            make_credential(token="   "),
            make_credential(token=123),
            make_credential(token=None),
            make_credential(token=True),
            make_credential(organization_id=""),
            make_credential(organization_id="  "),
            make_credential(organization_id=7),
            make_credential(organization_id=None),
            make_credential(role="admin"),
            make_credential(role=""),
            make_credential(role="WRITE"),
            make_credential(role=1),
            make_credential(role=None),
            make_credential(role=True),
            [],
            "tok-1",
            42,
            None,
            True,
        ]
        for index, bad_body in enumerate(bad_bodies):
            with self.subTest(bad_body=bad_body):
                bad = bad_body if isinstance(bad_body, dict) else bad_body
                if isinstance(bad, dict) and bad.get("token") == "tok-1":
                    bad = {**bad, "token": f"tok-bad-{index}"}
                status, body = self.post_json("/auth/tokens", bad)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

        # No failed request registered anything: every candidate token is
        # still unregistered and cannot authenticate.
        for index in range(len(bad_bodies)):
            status, body = self.request(
                "/events?organizationId=org-1", token=f"tok-bad-{index}"
            )
            self.assertEqual(status, 401)
            self.assertEqual(body["error"], "unauthorized")
        status, _ = self.request("/events?organizationId=org-1", token="tok-1")
        self.assertEqual(status, 401)

    def test_register_missing_content_type_is_415(self) -> None:
        status, body = self.request(
            "/auth/tokens",
            method="POST",
            body=json.dumps(make_credential()).encode(),
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_register_unsupported_content_type_is_415(self) -> None:
        status, body = self.request(
            "/auth/tokens",
            method="POST",
            body=json.dumps(make_credential()).encode(),
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_register_charset_content_type_is_accepted(self) -> None:
        status, _ = self.request(
            "/auth/tokens",
            method="POST",
            body=json.dumps(make_credential()).encode(),
            content_type="application/json; charset=utf-8",
        )
        self.assertEqual(status, 201)

    def test_register_malformed_json_is_400(self) -> None:
        status, body = self.request(
            "/auth/tokens", method="POST", body=b'{"token": '
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_registration_endpoint_needs_no_credential(self) -> None:
        # No Authorization header at all: registration is the open entry.
        status, _ = self.request(
            "/auth/tokens",
            method="POST",
            body=json.dumps(make_credential("tok-open")).encode(),
        )
        self.assertEqual(status, 201)

    # ------------------------------------------------------- authentication

    def test_health_needs_no_credential(self) -> None:
        status, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_unknown_path_is_still_404_without_credential(self) -> None:
        status, body = self.request("/missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, body = self.request(
            "/missing", method="POST", body=b"{}"
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    def test_missing_authorization_header_is_401(self) -> None:
        self.register_token("tok-1", "org-1")
        for method, path in (
            ("GET", "/events?organizationId=org-1"),
            ("GET", "/events/aggregate?organizationId=org-1&type=t&windowSize=1"),
            ("GET", "/events/replay?organizationId=org-1&asOf=1"),
            ("GET", "/events/replay/compare?organizationId=org-1&fromAsOf=0&toAsOf=1"),
            ("GET", "/events/region?organizationId=org-1&region=north"),
            (
                "GET",
                "/events/region/aggregate"
                "?organizationId=org-1&region=north&type=t&windowSize=1",
            ),
            ("GET", "/reservations?organizationId=org-1"),
            ("GET", "/alerts?organizationId=org-1"),
            ("GET", "/snapshots"),
            ("GET", "/branches/br-1"),
            ("GET", "/branches/br-1/events?organizationId=org-1"),
            ("POST", "/events"),
            ("POST", "/reservations"),
            ("POST", "/decisions/evaluate"),
            ("POST", "/decisions/allocate"),
            ("POST", "/alerts/evaluate"),
            ("POST", "/snapshots"),
            ("POST", "/branches"),
            ("POST", "/branches/br-1/events"),
        ):
            with self.subTest(method=method, path=path):
                kwargs: dict[str, Any] = {}
                if method == "POST":
                    kwargs = {"method": "POST", "body": b"{}"}
                status, body = self.request(path, **kwargs)
                self.assertEqual(status, 401)
                self.assertEqual(body["error"], "unauthorized")

    def test_malformed_authorization_headers_are_401(self) -> None:
        self.register_token("tok-1", "org-1")
        for header in (
            "tok-1",
            "Token tok-1",
            "bearer tok-1",
            "BEARER tok-1",
            "Bearer",
            "Bearer ",
            "Bearer  tok-1",
            "Bearer tok-1 extra",
            "Bearer other-token",
            "Bearer ",
            "",
        ):
            with self.subTest(header=header):
                status, body = self.request(
                    "/events?organizationId=org-1", authorization=header
                )
                self.assertEqual(status, 401)
                self.assertEqual(body["error"], "unauthorized")

    def test_unregistered_token_is_401(self) -> None:
        status, body = self.request(
            "/events?organizationId=org-1", token="never-registered"
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_tokens_are_cleared_on_restart(self) -> None:
        self.register_token("tok-1", "org-1")
        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{fresh.server_port}/events?organizationId=org-1",
                headers={"Authorization": "Bearer tok-1"},
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=30)
            self.assertEqual(raised.exception.code, 401)
            self.assertEqual(json.load(raised.exception)["error"], "unauthorized")
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # ------------------------------------------------------------ read role

    def test_read_role_can_call_query_and_compute_endpoints(self) -> None:
        write_token = self.seed_org1()
        self.post_json(
            "/snapshots", {"snapshotId": "snap-1"}, token=write_token
        )
        self.post_json(
            "/branches",
            {"branchId": "br-1", "snapshotId": "snap-1"},
            token=write_token,
        )
        read = self.register_token("tok-r1", "org-1", "read")

        get_paths = (
            "/events?organizationId=org-1",
            "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=60",
            "/events/replay?organizationId=org-1&asOf=100",
            "/events/replay/compare?organizationId=org-1&fromAsOf=0&toAsOf=100",
            "/events/region?organizationId=org-1&region=north",
            "/events/region/aggregate"
            "?organizationId=org-1&region=north&type=t&windowSize=60",
            "/reservations?organizationId=org-1",
            "/alerts?organizationId=org-1",
            "/snapshots",
            "/branches/br-1",
            "/branches/br-1/events?organizationId=org-1",
            "/branches/br-1/reservations?organizationId=org-1",
            "/branches/br-1/events/aggregate"
            "?organizationId=org-1&type=t&windowSize=60",
        )
        for path in get_paths:
            with self.subTest(path=path):
                status, _ = self.request(path, token=read)
                self.assertEqual(status, 200)

        compute_bodies = (
            (
                "/decisions/evaluate",
                {
                    "organizationId": "org-1",
                    "type": "incident.created",
                    "windowSize": 60,
                    "threshold": 1,
                },
            ),
            (
                "/decisions/allocate",
                {
                    "organizationId": "org-1",
                    "demands": [{"demandId": "d", "units": 1, "priority": 0}],
                    "resources": [{"resourceId": "r", "capacity": 1}],
                },
            ),
            (
                "/branches/br-1/decisions/evaluate",
                {
                    "organizationId": "org-1",
                    "type": "incident.created",
                    "windowSize": 60,
                    "threshold": 1,
                },
            ),
        )
        for path, payload in compute_bodies:
            with self.subTest(path=path):
                status, _ = self.post_json(path, payload, token=read)
                self.assertEqual(status, 200)

    def test_read_role_cannot_call_write_endpoints(self) -> None:
        write_token = self.seed_org1()
        self.post_json("/snapshots", {"snapshotId": "snap-1"}, token=write_token)
        self.post_json(
            "/branches",
            {"branchId": "br-1", "snapshotId": "snap-1"},
            token=write_token,
        )
        read = self.register_token("tok-r1", "org-1", "read")

        writes = (
            ("/events", make_event(eventId="evt-denied")),
            (
                "/reservations",
                {
                    "organizationId": "org-1",
                    "reservationId": "res-denied",
                    "resourceId": "r-a",
                    "quantity": 1,
                    "capacity": 5,
                },
            ),
            (
                "/alerts/evaluate",
                {
                    "organizationId": "org-1",
                    "type": "incident.created",
                    "windowSize": 60,
                    "threshold": 1,
                    "suppressionWindow": 10,
                },
            ),
            ("/snapshots", {"snapshotId": "snap-denied"}),
            (
                "/branches",
                {"branchId": "br-denied", "snapshotId": "snap-1"},
            ),
            ("/branches/br-1/events", make_event(eventId="evt-denied-2")),
            (
                "/branches/br-1/reservations",
                {
                    "organizationId": "org-1",
                    "reservationId": "res-denied-2",
                    "resourceId": "r-a",
                    "quantity": 1,
                    "capacity": 5,
                },
            ),
        )
        for path, payload in writes:
            with self.subTest(path=path):
                status, body = self.post_json(path, payload, token=read)
                self.assertEqual(status, 403)
                self.assertEqual(body["error"], "forbidden")

        # The failed writes changed nothing: no event, no reservation, no
        # alert, no snapshot, no branch, and the branch state is untouched.
        status, body = self.request("/events?organizationId=org-1", token=read)
        self.assertEqual([e["eventId"] for e in body["events"]], ["evt-1"])
        status, body = self.request(
            "/reservations?organizationId=org-1", token=read
        )
        self.assertEqual(len(body["reservations"]), 1)
        self.assertEqual(body["reservations"][0]["occupied"], 2)
        status, body = self.request("/alerts?organizationId=org-1", token=read)
        self.assertEqual(body["alerts"], [])
        status, body = self.request("/snapshots", token=read)
        self.assertEqual(
            [s["snapshotId"] for s in body["snapshots"]], ["snap-1"]
        )
        status, body = self.request("/branches/br-denied", token=read)
        self.assertEqual(status, 404)
        status, body = self.request(
            "/branches/br-1/events?organizationId=org-1", token=read
        )
        self.assertEqual(len(body["events"]), 1)
        status, body = self.request(
            "/branches/br-1/reservations?organizationId=org-1", token=read
        )
        self.assertEqual(len(body["reservations"]), 1)

    # ------------------------------------------------------ org isolation

    def test_cross_organization_reads_are_403(self) -> None:
        self.seed_org1()
        self.register_token("tok-2", "org-2")
        paths = (
            "/events?organizationId=org-1",
            "/events/aggregate?organizationId=org-1&type=incident.created&windowSize=60",
            "/events/replay?organizationId=org-1&asOf=100",
            "/events/replay/compare?organizationId=org-1&fromAsOf=0&toAsOf=100",
            "/events/region?organizationId=org-1&region=north",
            "/events/region/aggregate"
            "?organizationId=org-1&region=north&type=t&windowSize=60",
            "/reservations?organizationId=org-1",
            "/alerts?organizationId=org-1",
        )
        for path in paths:
            with self.subTest(path=path):
                status, body = self.request(path, token="tok-2")
                self.assertEqual(status, 403)
                self.assertEqual(body["error"], "forbidden")

    def test_cross_organization_writes_are_403_without_mutation(self) -> None:
        writer = self.seed_org1()
        self.register_token("tok-2", "org-2")
        writes = (
            ("/events", make_event(eventId="evt-x", organizationId="org-1")),
            (
                "/reservations",
                {
                    "organizationId": "org-1",
                    "reservationId": "res-x",
                    "resourceId": "r-a",
                    "quantity": 1,
                    "capacity": 5,
                },
            ),
            (
                "/alerts/evaluate",
                {
                    "organizationId": "org-1",
                    "type": "incident.created",
                    "windowSize": 60,
                    "threshold": 1,
                    "suppressionWindow": 10,
                },
            ),
            (
                "/decisions/evaluate",
                {
                    "organizationId": "org-1",
                    "type": "incident.created",
                    "windowSize": 60,
                    "threshold": 1,
                },
            ),
            (
                "/decisions/allocate",
                {
                    "organizationId": "org-1",
                    "demands": [],
                    "resources": [],
                },
            ),
        )
        for path, payload in writes:
            with self.subTest(path=path):
                status, body = self.post_json(path, payload, token="tok-2")
                self.assertEqual(status, 403)
                self.assertEqual(body["error"], "forbidden")

        # Nothing leaked into org-1's ledger, inventory, or alerts.
        status, body = self.request("/events?organizationId=org-1", token=writer)
        self.assertEqual([e["eventId"] for e in body["events"]], ["evt-1"])
        status, body = self.request(
            "/reservations?organizationId=org-1", token=writer
        )
        self.assertEqual(len(body["reservations"]), 1)
        self.assertEqual(body["reservations"][0]["occupied"], 2)
        status, body = self.request("/alerts?organizationId=org-1", token=writer)
        self.assertEqual(body["alerts"], [])

    def test_cross_organization_write_uses_body_organization(self) -> None:
        # A token may only write its own organization; the body's
        # organizationId decides, not the caller's intent.
        self.register_token("tok-1", "org-1")
        status, body = self.post_json(
            "/events", make_event(organizationId="org-2"), token="tok-1"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # org-2's view confirms nothing was written.
        self.register_token("tok-2", "org-2")
        status, body = self.request("/events?organizationId=org-2", token="tok-2")
        self.assertEqual(body["events"], [])

    # -------------------------------------------- snapshot/branch ownership

    def test_snapshots_are_scoped_to_creator_organization(self) -> None:
        writer = self.seed_org1()
        self.post_json("/snapshots", {"snapshotId": "snap-1"}, token=writer)
        self.register_token("tok-2", "org-2")

        # org-2 sees none of org-1's snapshots.
        status, body = self.request("/snapshots", token="tok-2")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"snapshots": []})

        # org-2 cannot fork org-1's snapshot.
        status, body = self.post_json(
            "/branches",
            {"branchId": "br-x", "snapshotId": "snap-1"},
            token="tok-2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

        # Snapshot names are a global namespace: org-2 cannot take org-1's
        # name, and org-1's listing is unaffected by the failed attempts.
        status, _ = self.post_json(
            "/snapshots", {"snapshotId": "snap-1"}, token="tok-2"
        )
        self.assertEqual(status, 409)
        self.assertEqual(
            [
                s["snapshotId"]
                for s in self.request("/snapshots", token=writer)[1]["snapshots"]
            ],
            ["snap-1"],
        )

    def test_branch_access_is_scoped_to_creator_organization(self) -> None:
        writer = self.seed_org1()
        self.post_json("/snapshots", {"snapshotId": "snap-1"}, token=writer)
        self.post_json(
            "/branches",
            {"branchId": "br-1", "snapshotId": "snap-1"},
            token=writer,
        )
        self.register_token("tok-2", "org-2")

        cross_org_paths = (
            ("GET", "/branches/br-1"),
            ("GET", "/branches/br-1/events?organizationId=org-2"),
            ("GET", "/branches/br-1/reservations?organizationId=org-2"),
            (
                "GET",
                "/branches/br-1/events/aggregate"
                "?organizationId=org-2&type=t&windowSize=60",
            ),
            ("POST", "/branches/br-1/events"),
            ("POST", "/branches/br-1/reservations"),
            ("POST", "/branches/br-1/decisions/evaluate"),
        )
        for method, path in cross_org_paths:
            with self.subTest(method=method, path=path):
                kwargs: dict[str, Any] = {"token": "tok-2"}
                if method == "POST":
                    kwargs["method"] = "POST"
                    kwargs["body"] = b"{}"
                status, body = self.request(path, **kwargs)
                self.assertEqual(status, 403)
                self.assertEqual(body["error"], "forbidden")

        # The owner still reaches the branch normally.
        status, body = self.request("/branches/br-1", token=writer)
        self.assertEqual(status, 200)
        self.assertEqual(body["branchId"], "br-1")

    def test_unknown_branch_is_404_for_any_organization(self) -> None:
        self.register_token("tok-1", "org-1")
        status, body = self.request("/branches/ghost", token="tok-1")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    # ------------------------------------------------------- error priority

    def test_authentication_outranks_validation(self) -> None:
        # No credential at all: even a malformed body answers 401 first.
        status, body = self.request(
            "/events", method="POST", body=b"not json"
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_read_role_outranks_body_validation_on_write_endpoints(self) -> None:
        self.register_token("tok-r", "org-1", "read")
        status, body = self.request(
            "/events", method="POST", body=b"not json", token="tok-r"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")


if __name__ == "__main__":
    unittest.main()
