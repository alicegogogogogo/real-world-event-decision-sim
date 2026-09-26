from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from event_sim.server import create_server
from tests import _support


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


def make_reservation(**overrides: Any) -> dict[str, Any]:
    reservation: dict[str, Any] = {
        "organizationId": "org-1",
        "reservationId": "res-1",
        "resourceId": "r-a",
        "quantity": 2,
        "capacity": 5,
    }
    reservation.update(overrides)
    return reservation


class SnapshotBranchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.token_cache: dict[str, str] = {}

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        content_type: str | None = "application/json",
        auth: bool = True,
    ) -> tuple[int, Any]:
        status, _, parsed = self.request_raw(
            path,
            method=method,
            body=body,
            content_type=content_type,
            auth=auth,
        )
        return status, parsed

    def request_raw(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        content_type: str | None = "application/json",
        auth: bool = True,
    ) -> tuple[int, bytes, Any]:
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if auth:
            headers.update(
                _support.authorization_header(
                    self.token_cache, self.base_url, path, body
                )
            )
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=5) as response:
                raw = response.read()
                return response.status, raw, json.loads(raw)
        except HTTPError as error:
            raw = error.read()
            try:
                return error.code, raw, json.loads(raw)
            finally:
                error.close()

    def post_json(self, path: str, payload: Any) -> tuple[int, Any]:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return self.request(path, method="POST", body=body)

    def seed_main(self) -> None:
        for event in (
            make_event(eventId="evt-1", occurredAt=10),
            make_event(eventId="evt-2", occurredAt=70),
        ):
            self.assertEqual(self.post_json("/events", event)[0], 201)
        self.assertEqual(
            self.post_json("/reservations", make_reservation())[0], 201
        )

    def create_snapshot(self, snapshot_id: str = "snap-1") -> tuple[int, Any]:
        return self.post_json("/snapshots", {"snapshotId": snapshot_id})

    def create_branch(
        self, branch_id: str = "br-1", snapshot_id: str = "snap-1"
    ) -> tuple[int, Any]:
        return self.post_json(
            "/branches", {"branchId": branch_id, "snapshotId": snapshot_id}
        )

    # --------------------------------------------------------------- snapshots

    def test_snapshot_empty_state(self) -> None:
        status, body = self.create_snapshot()
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "snapshotId": "snap-1",
                "events": 0,
                "resources": 0,
                "reservations": 0,
            },
        )

    def test_snapshot_captures_events_capacities_and_reservations(self) -> None:
        self.seed_main()
        status, body = self.create_snapshot()
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "snapshotId": "snap-1",
                "events": 2,
                "resources": 1,
                "reservations": 1,
            },
        )

    def test_snapshot_response_is_compact_integer_json_with_newline(self) -> None:
        self.seed_main()
        status, raw, body = self.request_raw(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": "snap-1"}).encode(),
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["events"], 2)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)

    def test_list_snapshots_empty(self) -> None:
        status, body = self.request("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"snapshots": []})

    def test_list_snapshots_sorted_by_unicode_code_point(self) -> None:
        self.seed_main()
        for name in ("snap-b", "snap-a", "snap-ä"):
            self.assertEqual(self.create_snapshot(name)[0], 201)
        status, body = self.request("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(
            [snapshot["snapshotId"] for snapshot in body["snapshots"]],
            ["snap-a", "snap-b", "snap-ä"],
        )
        for snapshot in body["snapshots"]:
            self.assertEqual(snapshot["events"], 2)
            self.assertEqual(snapshot["resources"], 1)
            self.assertEqual(snapshot["reservations"], 1)

    def test_duplicate_snapshot_is_conflict_and_original_is_kept(self) -> None:
        self.seed_main()
        self.assertEqual(self.create_snapshot()[0], 201)

        # Main state moves on after the snapshot.
        self.assertEqual(
            self.post_json(
                "/events", make_event(eventId="evt-3", occurredAt=200)
            )[0],
            201,
        )
        self.assertEqual(
            self.post_json(
                "/reservations", make_reservation(reservationId="res-2")
            )[0],
            201,
        )

        status, body = self.create_snapshot()
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "snapshot_conflict")

        # The original snapshot content is unchanged: a branch forked from it
        # sees the state as it was at capture time.
        status, branch = self.create_branch()
        self.assertEqual(status, 201)
        self.assertEqual(branch["events"], 2)
        self.assertEqual(branch["reservations"], 1)

    def test_snapshot_does_not_mutate_main_state(self) -> None:
        self.seed_main()
        self.create_snapshot()
        self.create_snapshot("snap-2")
        status, events = self.request("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(events["events"]), 2)
        status, reservations = self.request("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(reservations["reservations"]), 1)

    def test_snapshot_validation_errors(self) -> None:
        bad_bodies = [
            {},
            {"snapshotId": "snap-1", "extra": 1},
            {"snapshotId": ""},
            {"snapshotId": "   "},
            {"snapshotId": 123},
            {"snapshotId": None},
            {"snapshotId": True},
            {"snapshotId": ["x"]},
            [],
            "snap-1",
            42,
            True,
            None,
        ]
        for bad_body in bad_bodies:
            with self.subTest(bad_body=bad_body):
                status, body = self.post_json("/snapshots", bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

        # No failed request created a snapshot.
        status, body = self.request("/snapshots")
        self.assertEqual(body["snapshots"], [])

    def test_snapshot_missing_content_type_is_415(self) -> None:
        status, body = self.request(
            "/snapshots",
            method="POST",
            body=json.dumps({"snapshotId": "snap-1"}).encode(),
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_snapshot_bad_content_type_is_415(self) -> None:
        status, body = self.request(
            "/snapshots",
            method="POST",
            body=b'{"snapshotId":"snap-1"}',
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_snapshot_charset_content_type_accepted(self) -> None:
        status, _ = self.request(
            "/snapshots",
            method="POST",
            body=b'{"snapshotId":"snap-1"}',
            content_type="application/json; charset=utf-8",
        )
        self.assertEqual(status, 201)

    def test_snapshot_malformed_json_is_400(self) -> None:
        status, body = self.request(
            "/snapshots", method="POST", body=b'{"snapshotId": '
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")
        self.assertEqual(self.request("/snapshots")[1]["snapshots"], [])

    # ---------------------------------------------------------------- branches

    def test_create_branch_copies_snapshot_state(self) -> None:
        self.seed_main()
        self.create_snapshot()
        status, body = self.create_branch()
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "branchId": "br-1",
                "snapshotId": "snap-1",
                "events": 2,
                "resources": 1,
                "reservations": 1,
            },
        )

    def test_get_branch_returns_summary(self) -> None:
        self.seed_main()
        self.create_snapshot()
        self.create_branch()
        status, body = self.request("/branches/br-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "branchId": "br-1",
                "snapshotId": "snap-1",
                "events": 2,
                "resources": 1,
                "reservations": 1,
            },
        )

    def test_get_unknown_branch_is_404(self) -> None:
        status, body = self.request("/branches/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    def test_branch_with_unicode_id_is_percent_decodable(self) -> None:
        self.create_snapshot()
        branch_id = "br-ä"
        status, _ = self.create_branch(branch_id)
        self.assertEqual(status, 201)
        status, body = self.request(f"/branches/{quote(branch_id)}")
        self.assertEqual(status, 200)
        self.assertEqual(body["branchId"], branch_id)

    def test_branch_missing_snapshot_is_404(self) -> None:
        status, body = self.create_branch("br-1", "missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    def test_duplicate_branch_is_conflict(self) -> None:
        self.create_snapshot()
        self.assertEqual(self.create_branch()[0], 201)
        status, body = self.create_branch()
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "branch_conflict")

    def test_branch_validation_errors(self) -> None:
        self.create_snapshot()
        bad_bodies = [
            {},
            {"branchId": "br-1"},
            {"snapshotId": "snap-1"},
            {"branchId": "br-1", "snapshotId": "snap-1", "extra": 1},
            {"branchId": "", "snapshotId": "snap-1"},
            {"branchId": "  ", "snapshotId": "snap-1"},
            {"branchId": 7, "snapshotId": "snap-1"},
            {"branchId": None, "snapshotId": "snap-1"},
            {"branchId": "br-1", "snapshotId": ""},
            {"branchId": "br-1", "snapshotId": " "},
            {"branchId": "br-1", "snapshotId": 9},
            {"branchId": "br-1", "snapshotId": None},
            [],
            "x",
            True,
            None,
        ]
        for bad_body in bad_bodies:
            with self.subTest(bad_body=bad_body):
                status, body = self.post_json("/branches", bad_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

        # A missing snapshot outranks no other rule, but validation failures
        # must never create a branch.
        self.assertEqual(self.request("/branches/br-1")[0], 404)

    def test_branch_missing_snapshot_outranks_duplicate_check(self) -> None:
        # Well-formed body naming an unknown snapshot: 404, not 409, even if
        # the branch name is reused only after a prior failure.
        status, body = self.create_branch("br-1", "missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    def test_branch_media_type_and_json_errors(self) -> None:
        self.create_snapshot()
        valid = {"branchId": "br-1", "snapshotId": "snap-1"}
        status, body = self.request(
            "/branches",
            method="POST",
            body=json.dumps(valid).encode(),
            content_type=None,
        )
        self.assertEqual(status, 415)
        status, body = self.request(
            "/branches", method="POST", body=b'{"branchId": ', content_type=None
        )
        self.assertEqual(status, 415)
        status, body = self.request(
            "/branches", method="POST", body=b'{"branchId": '
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")
        self.assertEqual(self.request("/branches/br-1")[0], 404)

    # ------------------------------------------------------------- isolation

    def _fork_with_seeded_snapshot(self) -> None:
        self.seed_main()
        self.create_snapshot()
        self.create_branch()

    def test_branch_copies_events_visible_via_list_and_aggregate(self) -> None:
        self._fork_with_seeded_snapshot()
        status, body = self.request("/branches/br-1/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["evt-1", "evt-2"],
        )
        status, body = self.request(
            "/branches/br-1/events/aggregate"
            "?organizationId=org-1&type=incident.created&windowSize=60"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
            ],
        )

    def test_branch_copies_reservation_balances(self) -> None:
        self._fork_with_seeded_snapshot()
        status, body = self.request(
            "/branches/br-1/reservations?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["reservations"]), 1)
        reservation = body["reservations"][0]
        self.assertEqual(reservation["reservationId"], "res-1")
        self.assertEqual(reservation["capacity"], 5)
        self.assertEqual(reservation["occupied"], 2)
        self.assertEqual(reservation["remaining"], 3)

    def test_branch_event_writes_stay_in_branch(self) -> None:
        self._fork_with_seeded_snapshot()
        self.create_branch("br-2")

        status, body = self.post_json(
            "/branches/br-1/events",
            make_event(eventId="evt-branch", occurredAt=300),
        )
        self.assertEqual(status, 201)
        # A successful branch commit returns the same full five-field body.
        self.assertEqual(
            body, make_event(eventId="evt-branch", occurredAt=300)
        )

        # The new event is visible in br-1 only.
        status, body = self.request("/branches/br-1/events?organizationId=org-1")
        self.assertEqual(len(body["events"]), 3)
        status, body = self.request("/branches/br-2/events?organizationId=org-1")
        self.assertEqual(len(body["events"]), 2)
        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(len(body["events"]), 2)

        # Main writes after the fork never reach the branch.
        self.assertEqual(
            self.post_json(
                "/events", make_event(eventId="evt-main", occurredAt=400)
            )[0],
            201,
        )
        status, body = self.request("/branches/br-1/events?organizationId=org-1")
        self.assertEqual(len(body["events"]), 3)
        status, body = self.request("/branches/br-2/events?organizationId=org-1")
        self.assertEqual(len(body["events"]), 2)
        status, summary = self.request("/branches/br-1")
        self.assertEqual(summary["events"], 3)

    def test_branch_reservation_writes_stay_in_branch(self) -> None:
        self._fork_with_seeded_snapshot()
        self.create_branch("br-2")

        status, body = self.post_json(
            "/branches/br-1/reservations",
            make_reservation(reservationId="res-branch", quantity=1),
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "reservationId": "res-branch",
                "resourceId": "r-a",
                "quantity": 1,
                "capacity": 5,
                "occupied": 3,
                "remaining": 2,
            },
        )

        # Balances moved only in br-1.
        status, listing = self.request(
            "/branches/br-1/reservations?organizationId=org-1"
        )
        self.assertEqual(len(listing["reservations"]), 2)
        status, listing = self.request(
            "/branches/br-2/reservations?organizationId=org-1"
        )
        self.assertEqual(len(listing["reservations"]), 1)
        self.assertEqual(listing["reservations"][0]["occupied"], 2)
        status, listing = self.request("/reservations?organizationId=org-1")
        self.assertEqual(len(listing["reservations"]), 1)
        self.assertEqual(listing["reservations"][0]["occupied"], 2)

    def test_branch_reservation_replay_and_conflicts_follow_main_rules(self) -> None:
        self._fork_with_seeded_snapshot()

        # Identical replay: 200, not double counted.
        status, body = self.post_json(
            "/branches/br-1/reservations", make_reservation()
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "reservationId": "res-1",
                "resourceId": "r-a",
                "quantity": 2,
                "capacity": 5,
                "occupied": 2,
                "remaining": 3,
            },
        )

        # Same reservationId, different fields.
        status, body = self.post_json(
            "/branches/br-1/reservations", make_reservation(quantity=3)
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "reservation_conflict")

        # Different declared capacity for a known resource.
        status, body = self.post_json(
            "/branches/br-1/reservations",
            make_reservation(reservationId="res-2", capacity=9),
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_conflict")

        # Exceeding the copied balance is rejected without mutation.
        status, body = self.post_json(
            "/branches/br-1/reservations",
            make_reservation(reservationId="res-2", quantity=4),
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_exceeded")
        status, listing = self.request(
            "/branches/br-1/reservations?organizationId=org-1"
        )
        self.assertEqual(len(listing["reservations"]), 1)
        self.assertEqual(listing["reservations"][0]["occupied"], 2)

        # The exact remaining quantity still fits.
        status, body = self.post_json(
            "/branches/br-1/reservations",
            make_reservation(reservationId="res-2", quantity=3),
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {
                "organizationId": "org-1",
                "reservationId": "res-2",
                "resourceId": "r-a",
                "quantity": 3,
                "capacity": 5,
                "occupied": 5,
                "remaining": 0,
            },
        )

    def test_branch_reservation_response_ends_with_newline(self) -> None:
        self._fork_with_seeded_snapshot()
        status, raw, _ = self.request_raw(
            "/branches/br-1/reservations",
            method="POST",
            body=json.dumps(make_reservation(reservationId="res-2")).encode(),
        )
        self.assertEqual(status, 201)
        self.assertTrue(raw.endswith(b"\n"))

    def test_branch_event_replay_and_conflict_follow_main_rules(self) -> None:
        self._fork_with_seeded_snapshot()

        seeded = make_event(eventId="evt-1", occurredAt=10)
        status, body = self.post_json("/branches/br-1/events", seeded)
        self.assertEqual(status, 200)
        # An identical replay returns the same full body, nothing duplicated.
        self.assertEqual(body, seeded)
        status, body = self.post_json(
            "/branches/br-1/events", {**seeded, "payload": {"severity": "high"}}
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "event_id_conflict")

    def test_branch_event_media_type_json_and_validation_errors(self) -> None:
        self._fork_with_seeded_snapshot()
        path = "/branches/br-1/events"

        status, body = self.request(
            path,
            method="POST",
            body=json.dumps(make_event()).encode(),
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.request(
            path, method="POST", body=b'{"eventId": '
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

        bad_event = make_event()
        del bad_event["payload"]
        status, body = self.post_json(path, bad_event)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

        status, body = self.post_json(path, [make_event()])
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

        # Nothing failed open into the branch.
        status, listing = self.request(
            "/branches/br-1/events?organizationId=org-1"
        )
        self.assertEqual(len(listing["events"]), 2)

    def test_branch_reservation_media_type_json_and_validation_errors(self) -> None:
        self._fork_with_seeded_snapshot()
        path = "/branches/br-1/reservations"

        status, body = self.request(
            path,
            method="POST",
            body=json.dumps(make_reservation(reservationId="res-2")).encode(),
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.request(
            path, method="POST", body=b'{"organizationId": '
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

        bad = make_reservation()
        del bad["capacity"]
        status, body = self.post_json(path, bad)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

        status, body = self.post_json(path, [make_reservation()])
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_branch_list_and_aggregate_query_validation(self) -> None:
        self._fork_with_seeded_snapshot()
        for query in (
            "/branches/br-1/events",
            "/branches/br-1/events?organizationId=",
            "/branches/br-1/events?organizationId=%20",
            "/branches/br-1/events?organizationId=a&organizationId=b",
            "/branches/br-1/reservations",
            "/branches/br-1/reservations?organizationId=",
            "/branches/br-1/reservations?organizationId=a&organizationId=b",
            "/branches/br-1/events/aggregate?organizationId=org-1&type=t",
            "/branches/br-1/events/aggregate"
            "?organizationId=org-1&type=t&windowSize=60&windowSize=120",
            "/branches/br-1/events/aggregate"
            "?organizationId=org-1&type=t&windowSize=0",
            "/branches/br-1/events/aggregate"
            "?organizationId=org-1&type=t&windowSize=60&from=10",
        ):
            with self.subTest(query=query):
                status, body = self.request(query)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_branch_aggregate_window_semantics_match_main(self) -> None:
        self._fork_with_seeded_snapshot()
        query = (
            "/branches/br-1/events/aggregate"
            "?organizationId=org-1&type=incident.created&windowSize=60&from=0&to=120"
        )
        status, body = self.request(query)
        self.assertEqual(status, 200)
        self.assertEqual(
            body["windows"],
            [
                {"start": 0, "end": 60, "count": 1},
                {"start": 60, "end": 120, "count": 1},
                {"start": 120, "end": 180, "count": 0},
            ],
        )

    def test_branch_evaluate_reads_only_branch_events_and_is_repeatable(self) -> None:
        self.seed_main()
        self.create_snapshot()
        self.create_branch()

        # Two extra branch events land in window 0: window counts become
        # 3/1 in the branch but stay 1/1 on the main service.
        for event_id, occurred_at in (("evt-b1", 5), ("evt-b2", 15)):
            self.assertEqual(
                self.post_json(
                    "/branches/br-1/events",
                    make_event(eventId=event_id, occurredAt=occurred_at),
                )[0],
                201,
            )

        payload = {
            "organizationId": "org-1",
            "type": "incident.created",
            "windowSize": 60,
            "threshold": 3,
        }
        responses = [
            self.post_json("/branches/br-1/decisions/evaluate", payload)
            for _ in range(3)
        ]
        self.assertTrue(all(status == 200 for status, _ in responses))
        self.assertEqual(len({json.dumps(b, sort_keys=True) for _, b in responses}), 1)
        self.assertEqual(
            responses[0][1],
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "from": None,
                "to": None,
                "peakStart": 0,
                "peakCount": 3,
                "action": "escalate",
            },
        )

        status, main_result = self.post_json("/decisions/evaluate", payload)
        self.assertEqual(status, 200)
        self.assertEqual(main_result["peakCount"], 1)
        self.assertEqual(main_result["action"], "observe")

    def test_branch_evaluate_validation_and_parse_errors(self) -> None:
        self._fork_with_seeded_snapshot()
        path = "/branches/br-1/decisions/evaluate"
        valid = {
            "organizationId": "org-1",
            "type": "incident.created",
            "windowSize": 60,
            "threshold": 3,
        }

        status, body = self.request(
            path,
            method="POST",
            body=json.dumps(valid).encode(),
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

        status, body = self.request(path, method="POST", body=b'{"organizationId": ')
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

        bad = dict(valid)
        del bad["threshold"]
        status, body = self.post_json(path, bad)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

        status, body = self.post_json(path, [valid])
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_branch_operations_never_change_main_state(self) -> None:
        self._fork_with_seeded_snapshot()

        self.post_json(
            "/branches/br-1/events",
            make_event(eventId="evt-branch", occurredAt=300),
        )
        self.post_json(
            "/branches/br-1/reservations",
            make_reservation(reservationId="res-branch", quantity=1),
        )
        self.post_json(
            "/branches/br-1/decisions/evaluate",
            {
                "organizationId": "org-1",
                "type": "incident.created",
                "windowSize": 60,
                "threshold": 1,
            },
        )
        self.request("/branches/br-1/events?organizationId=org-1")
        self.request("/branches/br-1/events/aggregate?organizationId=org-1&type=incident.created&windowSize=60")
        self.request("/branches/br-1/reservations?organizationId=org-1")

        status, body = self.request("/events?organizationId=org-1")
        self.assertEqual(len(body["events"]), 2)
        status, body = self.request("/reservations?organizationId=org-1")
        self.assertEqual(len(body["reservations"]), 1)
        self.assertEqual(body["reservations"][0]["occupied"], 2)

    def test_branch_subpaths_unknown_branch_return_branch_not_found(self) -> None:
        for method, path in (
            ("GET", "/branches/ghost/events?organizationId=org-1"),
            ("GET", "/branches/ghost/reservations?organizationId=org-1"),
            (
                "GET",
                "/branches/ghost/events/aggregate"
                "?organizationId=org-1&type=t&windowSize=60",
            ),
        ):
            with self.subTest(method=method, path=path):
                status, body = self.request(path, method=method)
                self.assertEqual(status, 404)
                self.assertEqual(body["error"], "branch_not_found")

        for path in (
            "/branches/ghost/events",
            "/branches/ghost/reservations",
            "/branches/ghost/decisions/evaluate",
        ):
            with self.subTest(path=path):
                status, body = self.post_json(path, {})
                self.assertEqual(status, 404)
                self.assertEqual(body["error"], "branch_not_found")

    def test_unknown_branch_subpaths_are_not_found(self) -> None:
        self.seed_main()
        self.create_snapshot()
        self.create_branch()
        status, body = self.request("/branches/br-1/widgets")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, body = self.post_json("/branches/br-1/widgets", {})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")
        status, body = self.request("/branches")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    def test_branch_responses_are_compact_integer_json(self) -> None:
        self._fork_with_seeded_snapshot()
        status, raw, _ = self.request_raw(
            "/branches/br-1/events?organizationId=org-1"
        )
        self.assertEqual(status, 200)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)

    # ------------------------------------------------------------- restart

    def test_new_server_instance_has_no_snapshots_or_branches(self) -> None:
        self.seed_main()
        self.create_snapshot()
        self.create_branch()

        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            fresh_base = f"http://127.0.0.1:{fresh.server_port}"
            token = _support.ensure_token({}, fresh_base, "org-1")
            snap_request = Request(
                f"{fresh_base}/snapshots",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(snap_request, timeout=2) as response:
                self.assertEqual(json.load(response), {"snapshots": []})
            branch_request = Request(
                f"{fresh_base}/branches/br-1",
                headers={"Authorization": f"Bearer {token}"},
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(branch_request, timeout=2)
            self.assertEqual(raised.exception.code, 404)
            raised.exception.close()
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # --------------------------------------------------- main contract unchanged

    def test_main_endpoints_still_work_alongside_snapshots(self) -> None:
        self.seed_main()
        self.create_snapshot()
        self.create_branch()

        status, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

        status, body = self.request("/missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

        # Allocation remains a main-only, stateless endpoint and is not exposed
        # under branches.
        allocation = {
            "organizationId": "org-1",
            "demands": [{"demandId": "d", "units": 1, "priority": 0}],
            "resources": [{"resourceId": "r", "capacity": 1}],
        }
        status, body = self.post_json("/decisions/allocate", allocation)
        self.assertEqual(status, 200)
        self.assertEqual(body["totalUnits"], 1)

        status, body = self.post_json(
            "/branches/br-1/decisions/allocate", allocation
        )
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")


if __name__ == "__main__":
    unittest.main()
