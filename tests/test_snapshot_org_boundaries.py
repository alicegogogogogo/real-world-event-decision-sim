"""Multi-organization boundary regression tests for snapshots and branches.

Locks down the finalized cross-organization contract:

- a snapshot captures only the creator organization's events, capacities, and
  reservations, and its three counts count only that organization;
- snapshot and branch names are unique across the whole service and are owned
  by the creator organization, so a taken name conflicts (409) regardless of
  which organization holds it;
- forking another organization's snapshot or reading another organization's
  branch is 403 with no content; 404 is reserved for names that never
  appeared, and nothing is ever implicitly created;
- branch creation judges snapshot ownership before the duplicate branch name;
- inside a branch, another organization's event ids, reservation ids, and
  capacities never conflict and never affect balances;
- authorization and the write are one locked step: rejected requests leave
  nothing and concurrent same-name creations have exactly one winner.
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


def event_body(event_id: str, organization_id: str = ORG1, **more: Any) -> dict[str, Any]:
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
    resource_id: str,
    *,
    organization_id: str = ORG1,
    quantity: int = 1,
    capacity: int = 5,
) -> dict[str, Any]:
    return {
        "organizationId": organization_id,
        "reservationId": reservation_id,
        "resourceId": resource_id,
        "quantity": quantity,
        "capacity": capacity,
    }


def alert_body(organization_id: str) -> dict[str, Any]:
    return {
        "organizationId": organization_id,
        "type": "incident.created",
        "windowSize": 60,
        "threshold": 1,
        "suppressionWindow": 100,
    }


class SnapshotOrganizationBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.register("w1", ORG1, "write")
        self.register("w2", ORG2, "write")

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
    ) -> tuple[int, Any]:
        body = (
            raw_body
            if raw_body is not None
            else json.dumps(payload).encode() if payload is not None
            else None
        )
        status, raw = self.raw(path, method=method, body=body, token=token)
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

    def seed_both_organizations(self) -> None:
        # ORG1: one event, one reservation occupying 2 of res-a (cap 5).
        self.assertEqual(
            self.call("/events", method="POST", payload=event_body("e1"), token="w1")[
                0
            ],
            201,
        )
        self.assertEqual(
            self.call(
                "/reservations",
                method="POST",
                payload=reservation_body("r1", "res-a", quantity=2, capacity=5),
                token="w1",
            )[0],
            201,
        )
        # ORG2: one event, a reservation on its own resource, and 2 units on a
        # resource ORG1 also uses (the main inventory pools that capacity).
        self.assertEqual(
            self.call(
                "/events",
                method="POST",
                payload=event_body("e2", ORG2),
                token="w2",
            )[0],
            201,
        )
        self.assertEqual(
            self.call(
                "/reservations",
                method="POST",
                payload=reservation_body(
                    "r2", "res-b", organization_id=ORG2, quantity=4, capacity=10
                ),
                token="w2",
            )[0],
            201,
        )
        self.assertEqual(
            self.call(
                "/reservations",
                method="POST",
                payload=reservation_body(
                    "r2-shared", "res-c", organization_id=ORG2, quantity=2,
                    capacity=5,
                ),
                token="w2",
            )[0],
            201,
        )
        self.assertEqual(
            self.call(
                "/reservations",
                method="POST",
                payload=reservation_body(
                    "r1-shared", "res-c", organization_id=ORG1, quantity=2,
                    capacity=5,
                ),
                token="w1",
            )[0],
            201,
        )

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

    # -------------------------------------------------- capture / summaries

    def test_snapshot_counts_only_creator_organization(self) -> None:
        self.seed_both_organizations()
        status, body = self.make_snapshot("s1", "w1")
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {"snapshotId": "s1", "events": 1, "resources": 2,
             "reservations": 2},
        )

        status, body = self.make_snapshot("s2", "w2")
        self.assertEqual(status, 201)
        self.assertEqual(
            body,
            {"snapshotId": "s2", "events": 1, "resources": 2,
             "reservations": 2},
        )

    def test_snapshots_created_at_same_state_have_identical_counts(self) -> None:
        self.seed_both_organizations()
        first_status, first = self.make_snapshot("s-a", "w1")
        second_status, second = self.make_snapshot("s-b", "w1")
        self.assertEqual((first_status, second_status), (201, 201))
        self.assertEqual(first["events"], second["events"])
        self.assertEqual(first["resources"], second["resources"])
        self.assertEqual(first["reservations"], second["reservations"])

    def test_branch_carries_only_owner_organization_data(self) -> None:
        self.seed_both_organizations()
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_branch("b1", "s1", "w1")[0], 201)

        status, body = self.call("/branches/b1/events?organizationId=org-1",
                                 token="w1")
        self.assertEqual(status, 200)
        self.assertEqual([e["eventId"] for e in body["events"]], ["e1"])

        status, body = self.call(
            "/branches/b1/reservations?organizationId=org-1", token="w1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            sorted(r["reservationId"] for r in body["reservations"]),
            ["r1", "r1-shared"],
        )

    def test_foreign_identifiers_and_capacities_never_conflict_in_branch(self) -> None:
        self.seed_both_organizations()
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_branch("b1", "s1", "w1")[0], 201)

        # ORG2 owns event e2 on the main service; reusing that id in ORG1's
        # branch is a fresh create, not a conflict.
        status, body = self.call(
            "/branches/b1/events",
            method="POST",
            payload=event_body("e2", ORG1),
            token="w1",
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["eventId"], "e2")

        # res-b is an ORG2 resource with recorded capacity 10 on the main
        # service. Declaring capacity 3 for it inside the branch must not raise
        # capacity_conflict: the branch has never seen that resource.
        status, body = self.call(
            "/branches/b1/reservations",
            method="POST",
            payload=reservation_body(
                "r2", "res-b", organization_id=ORG1, quantity=1, capacity=3
            ),
            token="w1",
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["capacity"], 3)
        self.assertEqual(body["occupied"], 1)
        self.assertEqual(body["remaining"], 2)

        # Reusing ORG2's reservation id is likewise a fresh branch create.
        status, body = self.call(
            "/branches/b1/reservations",
            method="POST",
            payload=reservation_body(
                "r2-again", "res-b", organization_id=ORG1, quantity=1, capacity=3
            ),
            token="w1",
        )
        self.assertEqual(status, 201)

    def test_foreign_reservations_never_affect_branch_balance(self) -> None:
        self.seed_both_organizations()
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_branch("b1", "s1", "w1")[0], 201)

        # On the main service res-c has 4/5 occupied (2 per org), so ORG1
        # cannot take 3 more there.
        status, body = self.call(
            "/reservations",
            method="POST",
            payload=reservation_body(
                "main-x", "res-c", organization_id=ORG1, quantity=3, capacity=5
            ),
            token="w1",
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "capacity_exceeded")

        # The branch captured only ORG1's 2 units: 3 more fit exactly there.
        status, body = self.call(
            "/branches/b1/reservations",
            method="POST",
            payload=reservation_body(
                "branch-x", "res-c", organization_id=ORG1, quantity=3, capacity=5
            ),
            token="w1",
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["occupied"], 5)
        self.assertEqual(body["remaining"], 0)

        # The main service balance was not touched by the branch commit.
        status, body = self.call("/reservations?organizationId=org-1", token="w1")
        main_shared = next(
            r for r in body["reservations"] if r["resourceId"] == "res-c"
        )
        self.assertEqual(main_shared["occupied"], 4)
        self.assertEqual(main_shared["remaining"], 1)

    # ------------------------------------------------------- global naming

    def test_snapshot_name_conflicts_across_organizations(self) -> None:
        self.seed_both_organizations()
        self.assertEqual(self.make_snapshot("shared", "w1")[0], 201)

        # The same name taken by ORG2 conflicts even though ORG2 has none.
        status, body = self.make_snapshot("shared", "w2")
        self.assertEqual(status, 409)
        self.assertEqual(body, {"error": "snapshot_conflict"})

        # The loser stores nothing and sees nothing of the winner's snapshot.
        status, listing = self.call("/snapshots", token="w2")
        self.assertEqual(listing["snapshots"], [])

        # The original snapshot content is intact: a fork sees ORG1's state.
        status, branch = self.make_branch("b1", "shared", "w1")
        self.assertEqual(status, 201)
        self.assertEqual(branch["events"], 1)
        self.assertEqual(branch["reservations"], 2)

    def test_branch_name_conflicts_across_organizations(self) -> None:
        self.seed_both_organizations()
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_snapshot("s2", "w2")[0], 201)
        self.assertEqual(self.make_branch("shared-branch", "s1", "w1")[0], 201)

        status, body = self.make_branch("shared-branch", "s2", "w2")
        self.assertEqual(status, 409)
        self.assertEqual(body, {"error": "branch_conflict"})

        # The existing branch is unaffected and still owned by ORG1.
        status, summary = self.call("/branches/shared-branch", token="w1")
        self.assertEqual(status, 200)
        self.assertEqual(summary["snapshotId"], "s1")
        status, body = self.call("/branches/shared-branch", token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    # -------------------------------------------------- 403 / 404 precedence

    def test_fork_of_foreign_snapshot_is_403_with_no_content(self) -> None:
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        status, body = self.make_branch("b2", "s1", "w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        # The rejected fork created nothing.
        status, _ = self.call("/branches/b2", token="w2")
        self.assertEqual(status, 404)

    def test_foreign_branch_read_is_403_on_every_branch_entry_point(self) -> None:
        self.seed_both_organizations()
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_branch("b1", "s1", "w1")[0], 201)

        for path in (
            "/branches/b1",
            "/branches/b1/events?organizationId=org-2",
            "/branches/b1/reservations?organizationId=org-2",
            "/branches/b1/events/aggregate"
            "?organizationId=org-2&type=t&windowSize=60",
        ):
            with self.subTest(path=path):
                status, body = self.call(path, token="w2")
                self.assertEqual(status, 403)
                self.assertEqual(body, {"error": "forbidden"})

        for path, payload in (
            ("/branches/b1/events", event_body("x", ORG2)),
            (
                "/branches/b1/reservations",
                reservation_body("x", "res-x", organization_id=ORG2),
            ),
        ):
            with self.subTest(path=path):
                status, body = self.call(
                    path, method="POST", payload=payload, token="w2"
                )
                self.assertEqual(status, 403)
                self.assertEqual(body, {"error": "forbidden"})

    def test_404_only_when_name_has_never_appeared(self) -> None:
        # Names that have never existed: 404 with the fixed error codes, and
        # no implicit creation.
        status, body = self.make_branch("b1", "ghost-snapshot", "w1")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "snapshot_not_found"})

        status, body = self.call("/branches/ghost-branch", token="w1")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "branch_not_found"})

        # Once ORG1 creates a snapshot with that name, ORG2's fork flips from
        # 404 to 403: the name exists but is not theirs to use.
        self.assertEqual(self.make_snapshot("ghost-snapshot", "w1")[0], 201)
        status, body = self.make_branch("b2", "ghost-snapshot", "w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        # Still nothing created for ORG2.
        status, _ = self.call("/branches/b2", token="w2")
        self.assertEqual(status, 404)

    def test_branch_creation_checks_snapshot_ownership_before_duplicate_name(
        self,
    ) -> None:
        # ORG1 owns snapshot s1 and already has branch b1. ORG2 asks to fork
        # s1 reusing the taken branch name b1: ownership (403) must outrank
        # the duplicate name (409).
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_branch("b1", "s1", "w1")[0], 201)

        status, body = self.make_branch("b1", "s1", "w2")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        # Same ordering with a brand-new branch name: still ownership first.
        status, body = self.make_branch("brand-new", "s1", "w2")
        self.assertEqual(status, 403)

        # The owner forking its own snapshot into the taken name gets 409.
        status, body = self.make_branch("b1", "s1", "w1")
        self.assertEqual(status, 409)
        self.assertEqual(body, {"error": "branch_conflict"})

    def test_snapshot_listing_and_branch_summaries_are_org_scoped(self) -> None:
        self.seed_both_organizations()
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_branch("b1", "s1", "w1")[0], 201)

        status, body = self.call("/snapshots", token="w2")
        self.assertEqual(status, 200)
        self.assertEqual(body["snapshots"], [])
        status, body = self.call("/snapshots", token="w1")
        self.assertEqual([s["snapshotId"] for s in body["snapshots"]], ["s1"])

    # ------------------------------------------------------------ locking

    def test_concurrent_same_name_snapshots_have_one_winner(self) -> None:
        def attempt(token: str) -> int:
            status, _ = self.make_snapshot("race", token)
            return status

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(
                pool.map(attempt, ["w1", "w2"] * 4)
            )
        self.assertEqual(sorted(statuses).count(201), 1)
        self.assertEqual(sorted(statuses).count(409), 7)

        # Exactly one snapshot named "race" exists, owned by a single org;
        # the other organization's listing stays empty.
        _, org1 = self.call("/snapshots", token="w1")
        _, org2 = self.call("/snapshots", token="w2")
        winners = [
            name
            for listing in (org1["snapshots"], org2["snapshots"])
            for name in [s["snapshotId"] for s in listing]
        ]
        self.assertEqual(winners, ["race"])

    def test_concurrent_same_name_branches_have_one_winner(self) -> None:
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)
        self.assertEqual(self.make_snapshot("s2", "w2")[0], 201)

        def attempt(token_and_snapshot: tuple[str, str]) -> int:
            token, snapshot_id = token_and_snapshot
            status, _ = self.make_branch("br-race", snapshot_id, token)
            return status

        contenders = [("w1", "s1"), ("w2", "s2")] * 4
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(attempt, contenders))
        self.assertEqual(sorted(statuses).count(201), 1)
        self.assertEqual(sorted(statuses).count(409), 7)

    def test_rejected_requests_leave_no_snapshot_branch_or_alert(self) -> None:
        self.seed_both_organizations()
        self.assertEqual(self.make_snapshot("s1", "w1")[0], 201)

        # A read credential cannot create a snapshot.
        self.register("r1", ORG1, "read")
        status, body = self.make_snapshot("denied", "r1")
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

        # A foreign write credential cannot fork the snapshot.
        status, _ = self.make_branch("denied-branch", "s1", "w2")
        self.assertEqual(status, 403)

        # A forbidden alert evaluation raises no alert.
        status, _ = self.call(
            "/alerts/evaluate", method="POST", payload=alert_body(ORG1),
            token="w2",
        )
        self.assertEqual(status, 403)

        status, body = self.call("/snapshots", token="w1")
        self.assertEqual([s["snapshotId"] for s in body["snapshots"]], ["s1"])
        status, body = self.call("/alerts?organizationId=org-1", token="w1")
        self.assertEqual(body["alerts"], [])
        status, _ = self.call("/branches/denied-branch", token="w1")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
