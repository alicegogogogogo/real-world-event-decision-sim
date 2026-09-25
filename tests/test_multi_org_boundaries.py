"""Multi-organization boundary regression tests for snapshots and branches.

Pins down the cross-organization contract: snapshots capture only the
creator organization's state and summary counts, snapshot/branch names are
unique across the whole service, cross-organization forking or reading is
403 with no content, only never-used names yield 404, branch commits compare
identifiers only against the organization's own captured records, and
rejected or racing requests never leave objects behind.
"""

from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"


def event_body(
    event_id: str, organization_id: str = ORG1, **more: Any
) -> dict[str, Any]:
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
    reservation_id: str,
    organization_id: str = ORG1,
    **more: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "organizationId": organization_id,
        "reservationId": reservation_id,
        "resourceId": "r-a",
        "quantity": 1,
        "capacity": 5,
    }
    body.update(more)
    return body


class MultiOrgBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        # One write token per organization, registered up front.
        for token, org in (("w1", ORG1), ("w2", ORG2)):
            status, _ = self.call(
                "/auth/tokens",
                method="POST",
                payload={"token": token, "organizationId": org, "role": "write"},
            )
            self.assertEqual(status, 201)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    # -------------------------------------------------------------- low level

    def call(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Any = None,
        token: str | None = None,
    ) -> tuple[int, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            finally:
                error.close()

    # ------------------------------------------------------------------ seeds

    def seed_org1(self) -> None:
        for event_id in ("o1-evt-1", "o1-evt-2"):
            self.assertEqual(
                self.call(
                    "/events",
                    method="POST",
                    payload=event_body(event_id),
                    token="w1",
                )[0],
                201,
            )
        self.assertEqual(
            self.call(
                "/reservations",
                method="POST",
                payload=reservation_body("o1-res-1", quantity=2, capacity=5),
                token="w1",
            )[0],
            201,
        )

    def seed_org2(self) -> None:
        for event_id in ("o2-evt-1", "o2-evt-2", "o2-evt-3"):
            self.assertEqual(
                self.call(
                    "/events",
                    method="POST",
                    payload=event_body(event_id, ORG2),
                    token="w2",
                )[0],
                201,
            )
        for reservation_id, resource_id in (
            ("o2-res-1", "r-b"),
            ("o2-res-2", "r-c"),
        ):
            self.assertEqual(
                self.call(
                    "/reservations",
                    method="POST",
                    payload=reservation_body(
                        reservation_id, ORG2, resourceId=resource_id
                    ),
                    token="w2",
                )[0],
                201,
            )

    def create_snapshot(self, snapshot_id: str, token: str) -> tuple[int, Any]:
        return self.call(
            "/snapshots",
            method="POST",
            payload={"snapshotId": snapshot_id},
            token=token,
        )

    def create_branch(
        self, branch_id: str, snapshot_id: str, token: str
    ) -> tuple[int, Any]:
        return self.call(
            "/branches",
            method="POST",
            payload={"branchId": branch_id, "snapshotId": snapshot_id},
            token=token,
        )

    # ------------------------------------------------- capture and summaries

    def test_snapshot_counts_only_creator_organization(self) -> None:
        self.seed_org1()
        self.seed_org2()
        status, body = self.create_snapshot("snap-1", "w1")
        self.assertEqual(status, 201)
        # Only ORG1's two events, one resource, and one reservation count;
        # ORG2's three events, two resources, and two reservations do not.
        self.assertEqual(
            body,
            {
                "snapshotId": "snap-1",
                "events": 2,
                "resources": 1,
                "reservations": 1,
            },
        )
        # The listing summary reports the same own-organization counts.
        status, listing = self.call("/snapshots", token="w1")
        self.assertEqual(status, 200)
        self.assertEqual(listing["snapshots"], [body])

    def test_same_state_snapshots_have_identical_counts(self) -> None:
        self.seed_org1()
        self.seed_org2()
        status, first = self.create_snapshot("snap-a", "w1")
        self.assertEqual(status, 201)
        status, second = self.create_snapshot("snap-b", "w1")
        self.assertEqual(status, 201)
        self.assertEqual(
            {key: first[key] for key in ("events", "resources", "reservations")},
            {key: second[key] for key in ("events", "resources", "reservations")},
        )

    def test_snapshot_listing_hides_other_organizations(self) -> None:
        self.seed_org1()
        self.create_snapshot("snap-1", "w1")
        self.create_snapshot("snap-2", "w2")
        status, body = self.call("/snapshots", token="w2")
        self.assertEqual(status, 200)
        self.assertEqual(
            [snapshot["snapshotId"] for snapshot in body["snapshots"]],
            ["snap-2"],
        )
        status, body = self.call("/snapshots", token="w1")
        self.assertEqual(
            [snapshot["snapshotId"] for snapshot in body["snapshots"]],
            ["snap-1"],
        )

    def test_branch_carries_only_own_organization_data(self) -> None:
        self.seed_org1()
        self.seed_org2()
        self.create_snapshot("snap-1", "w1")
        status, summary = self.create_branch("br-1", "snap-1", "w1")
        self.assertEqual(status, 201)
        self.assertEqual(
            summary,
            {
                "branchId": "br-1",
                "snapshotId": "snap-1",
                "events": 2,
                "resources": 1,
                "reservations": 1,
            },
        )
        # The branch holds only ORG1's events; ORG2's never crossed over.
        status, body = self.call(
            "/branches/br-1/events?organizationId=org-1", token="w1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [event["eventId"] for event in body["events"]],
            ["o1-evt-1", "o1-evt-2"],
        )
        status, body = self.call(
            "/branches/br-1/reservations?organizationId=org-1", token="w1"
        )
        self.assertEqual(
            [r["reservationId"] for r in body["reservations"]], ["o1-res-1"]
        )

    # ------------------------------------------------------- name uniqueness

    def test_snapshot_name_is_unique_across_organizations(self) -> None:
        self.seed_org1()
        self.assertEqual(self.create_snapshot("snap-1", "w1")[0], 201)

        # ORG2 reusing the taken name conflicts; nothing is created for it.
        status, body = self.create_snapshot("snap-1", "w2")
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "snapshot_conflict")
        status, listing = self.call("/snapshots", token="w2")
        self.assertEqual(listing["snapshots"], [])

        # The original snapshot content is not rewritten: main state moves
        # on, but a fork still sees the capture-time counts.
        self.call(
            "/events",
            method="POST",
            payload=event_body("o1-evt-3"),
            token="w1",
        )
        status, branch = self.create_branch("br-1", "snap-1", "w1")
        self.assertEqual(status, 201)
        self.assertEqual(branch["events"], 2)

    def test_branch_name_is_unique_across_organizations(self) -> None:
        self.seed_org1()
        self.create_snapshot("snap-1", "w1")
        self.create_snapshot("snap-2", "w2")
        self.assertEqual(self.create_branch("br-1", "snap-1", "w1")[0], 201)

        # ORG2 reusing the taken branch name conflicts, even forking its own
        # snapshot; the existing branch is unaffected.
        status, body = self.create_branch("br-1", "snap-2", "w2")
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "branch_conflict")
        status, summary = self.call("/branches/br-1", token="w1")
        self.assertEqual(status, 200)
        self.assertEqual(summary["snapshotId"], "snap-1")
        self.assertEqual(summary["events"], 2)

    def test_branch_creation_checks_snapshot_ownership_before_name_conflict(
        self,
    ) -> None:
        self.create_snapshot("snap-1", "w1")
        self.assertEqual(self.create_branch("br-1", "snap-1", "w1")[0], 201)

        # Both rules are violated (foreign snapshot, taken branch name);
        # ownership is judged first, so the answer is 403, not 409.
        status, body = self.create_branch("br-1", "snap-1", "w2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------- cross-org 403 and 404

    def test_forking_other_organizations_snapshot_is_403_without_content(
        self,
    ) -> None:
        self.seed_org1()
        self.create_snapshot("snap-1", "w1")
        status, body = self.create_branch("br-2", "snap-1", "w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})
        # No branch was created for ORG2.
        status, body = self.call("/branches/br-2", token="w2")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    def test_reading_other_organizations_branch_is_403_without_content(
        self,
    ) -> None:
        self.seed_org1()
        self.create_snapshot("snap-1", "w1")
        self.create_branch("br-1", "snap-1", "w1")

        for path in (
            "/branches/br-1",
            "/branches/br-1/events?organizationId=org-2",
            "/branches/br-1/reservations?organizationId=org-2",
            "/branches/br-1/events/aggregate"
            "?organizationId=org-2&type=t&windowSize=60",
        ):
            with self.subTest(path=path):
                status, body = self.call(path, token="w2")
                self.assertEqual(status, 403)
                self.assertEqual(body, {"error": "forbidden"})

        for path, payload in (
            ("/branches/br-1/events", event_body("o2-evt-9", ORG2)),
            (
                "/branches/br-1/reservations",
                reservation_body("o2-res-9", ORG2),
            ),
            (
                "/branches/br-1/decisions/evaluate",
                {
                    "organizationId": ORG2,
                    "type": "t",
                    "windowSize": 60,
                    "threshold": 1,
                },
            ),
        ):
            with self.subTest(path=path):
                status, body = self.call(
                    path, method="POST", payload=payload, token="w2"
                )
                self.assertEqual(status, 403)
                self.assertEqual(body, {"error": "forbidden"})

    def test_unknown_names_are_404_and_create_nothing(self) -> None:
        status, body = self.call("/branches/ghost", token="w1")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

        status, body = self.create_branch("br-1", "ghost-snap", "w1")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

        # Neither 404 implicitly created anything.
        status, listing = self.call("/snapshots", token="w1")
        self.assertEqual(listing["snapshots"], [])
        status, body = self.call("/branches/br-1", token="w1")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "branch_not_found")

    # ------------------------------------------- branch identifier isolation

    def test_other_organization_identifiers_do_not_conflict_in_branch(
        self,
    ) -> None:
        # ORG2 owns an event id, a reservation id, and a capacity record in
        # the main service that ORG1's snapshot never captures.
        self.assertEqual(
            self.call(
                "/events",
                method="POST",
                payload=event_body("shared-evt", ORG2),
                token="w2",
            )[0],
            201,
        )
        self.assertEqual(
            self.call(
                "/reservations",
                method="POST",
                payload=reservation_body(
                    "shared-res", ORG2, resourceId="shared-r",
                    quantity=2, capacity=5,
                ),
                token="w2",
            )[0],
            201,
        )

        self.create_snapshot("snap-1", "w1")
        self.create_branch("br-1", "snap-1", "w1")

        # The same event id is a fresh commit inside ORG1's branch.
        status, body = self.call(
            "/branches/br-1/events",
            method="POST",
            payload=event_body("shared-evt"),
            token="w1",
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["eventId"], "shared-evt")

        # The same reservation id and resource id commit with ORG1's own
        # capacity: no reservation_conflict and no capacity_conflict against
        # ORG2's records.
        status, view = self.call(
            "/branches/br-1/reservations",
            method="POST",
            payload=reservation_body(
                "shared-res", resourceId="shared-r", quantity=4, capacity=9
            ),
            token="w1",
        )
        self.assertEqual(status, 201)
        self.assertEqual(
            view,
            {
                "organizationId": ORG1,
                "reservationId": "shared-res",
                "resourceId": "shared-r",
                "quantity": 4,
                "capacity": 9,
                "occupied": 4,
                "remaining": 5,
            },
        )

        # ORG2's main-service records and balances are untouched.
        status, listing = self.call(
            "/reservations?organizationId=org-2", token="w2"
        )
        self.assertEqual(listing["reservations"][0]["occupied"], 2)
        self.assertEqual(listing["reservations"][0]["remaining"], 3)

    def test_other_organization_reservations_do_not_affect_branch_balances(
        self,
    ) -> None:
        # Both organizations reserve the same resource in the main service.
        self.assertEqual(
            self.call(
                "/reservations",
                method="POST",
                payload=reservation_body(
                    "o1-res-1", resourceId="shared-r", quantity=1, capacity=5
                ),
                token="w1",
            )[0],
            201,
        )
        self.assertEqual(
            self.call(
                "/reservations",
                method="POST",
                payload=reservation_body(
                    "o2-res-1", ORG2, resourceId="shared-r",
                    quantity=3, capacity=5,
                ),
                token="w2",
            )[0],
            201,
        )
        # Main balances count both organizations...
        status, listing = self.call(
            "/reservations?organizationId=org-1", token="w1"
        )
        self.assertEqual(listing["reservations"][0]["occupied"], 4)

        # ...but the branch forked from ORG1's snapshot counts only ORG1's
        # reservation against the shared capacity record.
        self.create_snapshot("snap-1", "w1")
        self.create_branch("br-1", "snap-1", "w1")
        status, listing = self.call(
            "/branches/br-1/reservations?organizationId=org-1", token="w1"
        )
        self.assertEqual(len(listing["reservations"]), 1)
        self.assertEqual(listing["reservations"][0]["occupied"], 1)
        self.assertEqual(listing["reservations"][0]["remaining"], 4)

    # ------------------------------------------------------- rejected writes

    def test_rejected_requests_leave_no_alerts_snapshots_or_branches(
        self,
    ) -> None:
        # A read credential for ORG1 is out of scope for every write entry
        # point, and ORG2's write credential is out of scope for ORG1's
        # objects.
        self.call(
            "/auth/tokens",
            method="POST",
            payload={"token": "r1", "organizationId": ORG1, "role": "read"},
        )
        # Seed ORG1 events so an alert evaluation would escalate if allowed.
        for event_id in ("o1-evt-1", "o1-evt-2"):
            self.call(
                "/events",
                method="POST",
                payload=event_body(event_id),
                token="w1",
            )
        alert_payload = {
            "organizationId": ORG1,
            "type": "incident.created",
            "windowSize": 60,
            "threshold": 2,
            "suppressionWindow": 100,
        }

        # Read role: alert evaluation, snapshot creation, and branch
        # creation are all rejected before any object exists.
        self.create_snapshot("snap-1", "w1")
        for path, payload in (
            ("/alerts/evaluate", alert_payload),
            ("/snapshots", {"snapshotId": "snap-denied"}),
            ("/branches", {"branchId": "br-denied", "snapshotId": "snap-1"}),
        ):
            with self.subTest(path=path):
                status, _ = self.call(
                    path, method="POST", payload=payload, token="r1"
                )
                self.assertEqual(status, 403)

        # Another organization's write token cannot evaluate ORG1's alert or
        # fork ORG1's snapshot either.
        for path, payload in (
            ("/alerts/evaluate", alert_payload),
            ("/branches", {"branchId": "br-denied", "snapshotId": "snap-1"}),
        ):
            with self.subTest(path=path):
                status, _ = self.call(
                    path, method="POST", payload=payload, token="w2"
                )
                self.assertEqual(status, 403)

        # Nothing slipped through: no alert, no extra snapshot, no branch.
        status, body = self.call("/alerts?organizationId=org-1", token="w1")
        self.assertEqual(body["alerts"], [])
        status, body = self.call("/snapshots", token="w1")
        self.assertEqual(
            [s["snapshotId"] for s in body["snapshots"]], ["snap-1"]
        )
        status, body = self.call("/branches/br-denied", token="w1")
        self.assertEqual(status, 404)

    # ------------------------------------------------------------- concurrency

    def test_concurrent_same_name_snapshot_creation_only_one_succeeds(
        self,
    ) -> None:
        self.seed_org1()

        def attempt(token: str) -> int:
            return self.create_snapshot("snap-race", token)[0]

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(attempt, ["w1", "w2"] * 4))
        self.assertEqual(statuses.count(201), 1)
        self.assertEqual(statuses.count(409), 7)

        # Exactly one snapshot exists, owned by whichever organization won.
        total = 0
        for token in ("w1", "w2"):
            status, listing = self.call("/snapshots", token=token)
            self.assertEqual(status, 200)
            names = [s["snapshotId"] for s in listing["snapshots"]]
            self.assertIn(names, (["snap-race"], []))
            total += len(names)
        self.assertEqual(total, 1)

    def test_concurrent_same_name_branch_creation_only_one_succeeds(
        self,
    ) -> None:
        self.create_snapshot("snap-1", "w1")

        def attempt(_: int) -> int:
            return self.create_branch("br-race", "snap-1", "w1")[0]

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(attempt, range(8)))
        self.assertEqual(statuses.count(201), 1)
        self.assertEqual(statuses.count(409), 7)

        status, summary = self.call("/branches/br-race", token="w1")
        self.assertEqual(status, 200)
        self.assertEqual(summary["branchId"], "br-race")


if __name__ == "__main__":
    unittest.main()
