"""Regression tests for POST /branches/compare/events/replay/decisions.

The baseline already had the single-branch step-by-step replay
(``GET /branches/{branchId}/events/replay/decisions``) and the two-branch
window difference (``POST /branches/compare``), but no two-branch replay
alignment; this locks down the new read-only entry point that replays each
branch independently and aligns the two sides' steps by ``eventId``:

- each side accumulates only that branch's events of the caller's
  organization and the requested ``type``, ordered by ``occurredAt`` then
  ``eventId``; an identifier on both sides opens one aligned step, an
  identifier reached by only one side opens its own step with the missing
  side's window counts all zero and an empty-prefix decision;
- window rows align by start ascending (union of hit windows without a
  range; every intersecting window kept with a range, empty ones counted
  zero), each carrying both counts plus an ``equal`` marker, and the
  decision block carries each side's peak and an agreement marker;
- the same branch name on both sides is legal; neither side having a
  matching event yields no steps; the response is compact key-sorted JSON
  with integer values and one trailing newline, and identical requests
  are byte-for-byte identical without polluting each other;
- both ``read`` and ``write`` credentials may call it; the verdict order
  is fixed — 401 (credential) before 415/400/422 (media type and body
  shape) before 403 (organization, then the branches left then right)
  before 404 (branch_not_found) — nothing is ever written or implicitly
  created, and restarting clears branches and events.
"""

from __future__ import annotations

import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from event_sim.server import create_server

ORG1 = "org-1"
ORG2 = "org-2"
EVENT_TYPE = "incident.created"
ENDPOINT = "/branches/compare/events/replay/decisions"


def event_body(
    event_id: str,
    *,
    occurred_at: int,
    event_type: str = EVENT_TYPE,
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


class BranchCompareReplayDecisionsTest(unittest.TestCase):
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
        method: str = "POST",
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
        method: str = "POST",
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

    def add_branch_event(
        self, branch: str, event_id: str, *, token: str = "w1", **kwargs: Any
    ) -> None:
        organization_id = ORG1 if token == "w1" else ORG2
        status, _ = self.call(
            f"/branches/{branch}/events",
            method="POST",
            token=token,
            payload=event_body(
                event_id, organization_id=organization_id, **kwargs
            ),
        )
        self.assertEqual(status, 201)

    def capture(self, snapshot_id: str = "s1", *, token: str = "w1") -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def fork(
        self,
        branch_id: str,
        snapshot_id: str = "s1",
        *,
        token: str = "w1",
    ) -> None:
        status, _ = self.call(
            "/branches",
            method="POST",
            token=token,
            payload={"branchId": branch_id, "snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    def payload(self, **overrides: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "organizationId": ORG1,
            "left": "L",
            "right": "R",
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": 3,
        }
        body.update(overrides)
        return body

    def compare(
        self,
        payload: dict[str, Any] | None = None,
        *,
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        return self.call(
            ENDPOINT,
            payload=self.payload() if payload is None else payload,
            token=token,
        )

    def compare_raw(
        self, payload: dict[str, Any] | None = None, *, token: str | None = "w1"
    ) -> tuple[int, bytes]:
        body = json.dumps(self.payload() if payload is None else payload).encode()
        return self.raw(ENDPOINT, body=body, token=token)

    def seed_pair(self) -> None:
        # Shared forked base: both branches replay evt-a@0 and evt-b@100.
        self.add_event("evt-a", occurred_at=0)
        self.add_event("evt-b", occurred_at=100)
        # A different type and another organization's event never enter.
        self.add_event("evt-other", event_type="other.kind", occurred_at=0)
        self.add_event("evt-xorg", token="w2", occurred_at=0)
        self.capture()
        self.fork("L")
        self.fork("R")
        # Left-only and right-only identifiers, interleaved in time.
        self.add_branch_event("L", "evt-l", occurred_at=50)
        self.add_branch_event("R", "evt-r", occurred_at=60)
        self.add_branch_event("L", "evt-c", occurred_at=200)

    def single_branch_replay_steps(
        self, branch: str, *, threshold: int = 3, from_to: tuple[int, int] | None = None
    ) -> dict[str, dict[str, Any]]:
        params: dict[str, Any] = {
            "organizationId": ORG1,
            "type": EVENT_TYPE,
            "windowSize": 60,
            "threshold": threshold,
        }
        if from_to is not None:
            params["from"], params["to"] = from_to
        status, body = self.call(
            f"/branches/{branch}/events/replay/decisions?{urlencode(params)}",
            method="GET",
        )
        self.assertEqual(status, 200)
        return {step["eventId"]: step for step in body["steps"]}

    # ----------------------------------------------------------------- 200 shape

    def test_echoes_request_parameters(self) -> None:
        self.seed_pair()
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["left"], "L")
        self.assertEqual(body["right"], "R")
        self.assertEqual(body["type"], EVENT_TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 3)
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])

    def test_steps_align_by_event_id_in_replay_order(self) -> None:
        self.seed_pair()
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [
                ("evt-a", 0),
                ("evt-l", 50),
                ("evt-r", 60),
                ("evt-b", 100),
                ("evt-c", 200),
            ],
        )
        self.assertNotIn("evt-other", json.dumps(body))
        self.assertNotIn("evt-xorg", json.dumps(body))

    def test_shared_step_carries_both_prefixes(self) -> None:
        self.seed_pair()
        status, body = self.compare()
        self.assertEqual(status, 200)
        first = body["steps"][0]
        self.assertEqual(first["eventId"], "evt-a")
        self.assertEqual(
            first["windows"],
            [
                {
                    "start": 0,
                    "leftCount": 1,
                    "rightCount": 1,
                    "equal": True,
                }
            ],
        )
        self.assertEqual(
            first["decision"],
            {
                "left": {"peakStart": 0, "peakCount": 1, "action": "observe"},
                "right": {"peakStart": 0, "peakCount": 1, "action": "observe"},
                "equal": True,
            },
        )
        # The shared evt-b step carries each side's accumulated prefix:
        # evt-l lives only on the left, evt-r only on the right.
        evt_b = next(step for step in body["steps"] if step["eventId"] == "evt-b")
        self.assertEqual(
            evt_b["windows"],
            [
                {"start": 0, "leftCount": 2, "rightCount": 1, "equal": False},
                {"start": 60, "leftCount": 1, "rightCount": 2, "equal": False},
            ],
        )
        self.assertFalse(evt_b["decision"]["equal"])

    def test_one_side_only_step_counts_missing_side_as_zero(self) -> None:
        self.seed_pair()
        status, body = self.compare()
        self.assertEqual(status, 200)
        left_only = next(
            step for step in body["steps"] if step["eventId"] == "evt-l"
        )
        self.assertEqual(
            left_only["windows"],
            [
                {
                    "start": 0,
                    "leftCount": 2,
                    "rightCount": 0,
                    "equal": False,
                }
            ],
        )
        # The missing side carries the empty-prefix decision.
        self.assertEqual(
            left_only["decision"],
            {
                "left": {"peakStart": 0, "peakCount": 2, "action": "observe"},
                "right": {"peakStart": None, "peakCount": 0, "action": "observe"},
                "equal": False,
            },
        )
        right_only = next(
            step for step in body["steps"] if step["eventId"] == "evt-r"
        )
        self.assertEqual(
            right_only["windows"],
            [
                {"start": 0, "leftCount": 0, "rightCount": 1, "equal": False},
                {"start": 60, "leftCount": 0, "rightCount": 1, "equal": False},
            ],
        )
        self.assertEqual(
            right_only["decision"]["left"],
            {"peakStart": None, "peakCount": 0, "action": "observe"},
        )
        self.assertEqual(
            right_only["decision"]["right"],
            {"peakStart": 0, "peakCount": 1, "action": "observe"},
        )

    def test_each_side_matches_its_single_branch_replay_prefixes(self) -> None:
        self.seed_pair()
        status, body = self.compare()
        self.assertEqual(status, 200)
        left_steps = self.single_branch_replay_steps("L")
        right_steps = self.single_branch_replay_steps("R")
        for step in body["steps"]:
            event_id = step["eventId"]
            aligned_counts = {
                row["start"]: row for row in step["windows"]
            }
            for side, single_steps, count_field in (
                ("left", left_steps, "leftCount"),
                ("right", right_steps, "rightCount"),
            ):
                if event_id in single_steps:
                    expected = single_steps[event_id]
                    # Every window the single-branch replay shows at this
                    # prefix appears in the aligned row with the same count.
                    for row in expected["windows"]:
                        self.assertIn(row["start"], aligned_counts)
                        self.assertEqual(
                            aligned_counts[row["start"]][count_field],
                            row["count"],
                        )
                    self.assertEqual(
                        step["decision"][side]["peakStart"],
                        expected["peakStart"],
                    )
                    self.assertEqual(
                        step["decision"][side]["peakCount"],
                        expected["peakCount"],
                    )
                    self.assertEqual(
                        step["decision"][side]["action"], expected["action"]
                    )
                else:
                    # The side never reaches this identifier: it is the
                    # missing side at a solo step.
                    self.assertEqual(
                        step["decision"][side],
                        {"peakStart": None, "peakCount": 0, "action": "observe"},
                    )
                    self.assertTrue(
                        all(
                            row[count_field] == 0 for row in step["windows"]
                        )
                    )

    def test_action_escalates_per_side_when_peak_reaches_threshold(self) -> None:
        self.seed_pair()
        status, body = self.compare(self.payload(threshold=2))
        self.assertEqual(status, 200)
        by_id = {step["eventId"]: step for step in body["steps"]}
        # evt-a@0: one each — still observe.
        self.assertEqual(by_id["evt-a"]["decision"]["left"]["action"], "observe")
        self.assertEqual(by_id["evt-a"]["decision"]["right"]["action"], "observe")
        # evt-l@50 puts the left side at two in window 0 — left escalates,
        # the missing right side observes.
        self.assertEqual(by_id["evt-l"]["decision"]["left"]["action"], "escalate")
        self.assertEqual(by_id["evt-l"]["decision"]["right"]["action"], "observe")
        # evt-r@60 only adds the right side's first window-60 event, so the
        # right peak is still one; the right side first escalates at evt-b,
        # when evt-r and evt-b share window 60.
        self.assertEqual(by_id["evt-r"]["decision"]["right"]["action"], "observe")
        self.assertEqual(by_id["evt-b"]["decision"]["right"]["action"], "escalate")
        self.assertEqual(by_id["evt-b"]["decision"]["right"]["peakStart"], 60)
        self.assertEqual(by_id["evt-b"]["decision"]["right"]["peakCount"], 2)

    def test_same_branch_name_on_both_sides_is_equal_by_construction(self) -> None:
        self.seed_pair()
        status, body = self.compare(self.payload(right="L", threshold=1))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["evt-a", "evt-l", "evt-b", "evt-c"],
        )
        for step in body["steps"]:
            self.assertTrue(step["decision"]["equal"])
            self.assertTrue(all(row["equal"] for row in step["windows"]))
            self.assertEqual(
                [row["leftCount"] for row in step["windows"]],
                [row["rightCount"] for row in step["windows"]],
            )

    def test_one_side_without_events_opens_only_the_other_sides_steps(self) -> None:
        # R forks from an empty snapshot captured before any event landed;
        # L forks from a snapshot carrying evt-a and evt-b and gains evt-c.
        self.capture("s-empty")
        self.add_event("evt-a", occurred_at=10)
        self.add_event("evt-b", occurred_at=70)
        self.capture("s-base")
        self.fork("L", "s-base")
        self.fork("R", "s-empty")
        self.add_branch_event("L", "evt-c", occurred_at=130)
        status, body = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(
            [(step["eventId"], step["occurredAt"]) for step in body["steps"]],
            [("evt-a", 10), ("evt-b", 70), ("evt-c", 130)],
        )
        for step in body["steps"]:
            self.assertTrue(
                all(row["rightCount"] == 0 for row in step["windows"])
            )
            self.assertEqual(
                step["decision"]["right"],
                {"peakStart": None, "peakCount": 0, "action": "observe"},
            )

    def test_neither_side_matching_the_type_produces_no_steps(self) -> None:
        self.seed_pair()
        status, body = self.compare(self.payload(type="never.seen"))
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": ORG1,
                "left": "L",
                "right": "R",
                "type": "never.seen",
                "windowSize": 60,
                "threshold": 3,
                "from": None,
                "to": None,
                "steps": [],
            },
        )

    # ------------------------------------------------------------- range semantics

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.seed_pair()
        status, body = self.compare(self.payload(**{"from": 0, "to": 120}))
        self.assertEqual(status, 200)
        grid = [0, 60, 120]
        for step in body["steps"]:
            self.assertEqual(
                [row["start"] for row in step["windows"]], grid
            )
        # evt-c@200 is outside [0,120], yet it still arrives as a step. The
        # left prefix at that point holds the in-range events evt-a@0,
        # evt-l@50 and evt-b@100; evt-c itself counts nowhere, and the
        # right side is absent from the step entirely.
        evt_c = next(step for step in body["steps"] if step["eventId"] == "evt-c")
        self.assertEqual(
            [(row["leftCount"], row["rightCount"]) for row in evt_c["windows"]],
            [(2, 0), (1, 0), (0, 0)],
        )
        self.assertEqual(evt_c["decision"]["left"]["peakCount"], 2)
        self.assertEqual(evt_c["decision"]["left"]["peakStart"], 0)
        self.assertEqual(
            evt_c["decision"]["right"],
            {"peakStart": None, "peakCount": 0, "action": "observe"},
        )

    def test_range_filters_counts_but_keeps_step_order(self) -> None:
        self.add_event("evt-a", occurred_at=10)
        self.capture()
        self.fork("L")
        self.fork("R")
        self.add_branch_event("L", "evt-l", occurred_at=200)
        self.add_branch_event("R", "evt-r", occurred_at=70)
        status, body = self.compare(self.payload(**{"from": 60, "to": 120}))
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["evt-a", "evt-r", "evt-l"],
        )
        # Every step shows the fixed grid [60,120),[120,180).
        evt_a = body["steps"][0]
        self.assertEqual(
            [(row["start"], row["leftCount"], row["rightCount"]) for row in evt_a["windows"]],
            [(60, 0, 0), (120, 0, 0)],
        )
        evt_r = body["steps"][1]
        self.assertEqual(
            [(row["start"], row["leftCount"], row["rightCount"]) for row in evt_r["windows"]],
            [(60, 0, 1), (120, 0, 0)],
        )
        evt_l = body["steps"][2]
        # evt-l is left-only and at 200 outside the range, so both sides'
        # counts on every ranged row are zero (left filtered out, right
        # absent from the step).
        self.assertEqual(
            [(row["start"], row["leftCount"], row["rightCount"]) for row in evt_l["windows"]],
            [(60, 0, 0), (120, 0, 0)],
        )
        self.assertEqual(evt_l["decision"]["left"]["peakCount"], 0)
        self.assertIsNone(evt_l["decision"]["left"]["peakStart"])
        self.assertEqual(
            evt_l["decision"]["right"],
            {"peakStart": None, "peakCount": 0, "action": "observe"},
        )

    # --------------------------------------------------------- serialization / roles

    def test_response_is_compact_sorted_keys_newline_terminated(self) -> None:
        self.seed_pair()
        status, raw = self.compare_raw()
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        body = json.loads(raw)
        self.assertEqual(
            set(body),
            {
                "organizationId",
                "left",
                "right",
                "type",
                "windowSize",
                "threshold",
                "from",
                "to",
                "steps",
            },
        )
        self.assertEqual(
            set(body["steps"][0]),
            {"eventId", "occurredAt", "windows", "decision"},
        )
        self.assertEqual(
            set(body["steps"][0]["decision"]), {"left", "right", "equal"}
        )
        self.assertEqual(
            set(body["steps"][0]["decision"]["left"]),
            {"peakStart", "peakCount", "action"},
        )
        self.assertEqual(
            set(body["steps"][0]["windows"][0]),
            {"start", "leftCount", "rightCount", "equal"},
        )
        self.assertEqual(
            raw[:-1],
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode(),
        )
        self.assertIsInstance(body["windowSize"], int)
        self.assertIsInstance(body["threshold"], int)
        self.assertIsInstance(body["steps"][0]["occurredAt"], int)
        self.assertIsInstance(body["steps"][0]["windows"][0]["leftCount"], int)

    def test_repeated_requests_are_byte_identical(self) -> None:
        self.seed_pair()
        raws = [self.compare_raw()[1] for _ in range(3)]
        self.assertEqual(raws[0], raws[1])
        self.assertEqual(raws[1], raws[2])
        ranged = self.compare_raw(self.payload(**{"from": 0, "to": 180}))[1]
        self.assertNotEqual(ranged, raws[0])
        self.assertEqual(self.compare_raw()[1], raws[0])

    def test_read_and_write_credentials_may_both_compare(self) -> None:
        self.seed_pair()
        read_status, read_body = self.compare(token="r1")
        write_status, write_body = self.compare(token="w1")
        self.assertEqual(read_status, 200)
        self.assertEqual(write_status, 200)
        self.assertEqual(read_body, write_body)

    def test_comparison_is_read_only(self) -> None:
        self.seed_pair()
        self.compare()
        self.compare(self.payload(threshold=1))
        self.compare(self.payload(**{"from": 0, "to": 180}))
        # Each branch forked the three org-1 main events (including the
        # other-type event); L additionally holds evt-l and evt-c, R evt-r.
        for branch, events in (("L", 5), ("R", 4)):
            status, summary = self.call(f"/branches/{branch}", method="GET")
            self.assertEqual(status, 200)
            self.assertEqual(summary["events"], events)
        # The main ledger (org-1 only), reservations and alerts are untouched.
        status, listing = self.call("/events?organizationId=org-1", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 3)
        status, alerts = self.call("/alerts?organizationId=org-1", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])
        status, reservations = self.call(
            "/reservations?organizationId=org-1", method="GET"
        )
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])

    def test_fresh_server_unknown_branch_is_404(self) -> None:
        fresh = create_server(port=0)
        thread = threading.Thread(target=fresh.serve_forever, daemon=True)
        thread.start()
        try:
            fresh_base = f"http://127.0.0.1:{fresh.server_port}"
            register = Request(
                f"{fresh_base}/auth/tokens",
                data=json.dumps(
                    {"token": "tok-f", "organizationId": ORG1, "role": "write"}
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(register, timeout=2) as response:
                self.assertEqual(response.status, 201)
            request = Request(
                f"{fresh_base}{ENDPOINT}",
                data=json.dumps(self.payload()).encode(),
                headers={
                    "Authorization": "Bearer tok-f",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=2)
            self.assertEqual(caught.exception.code, 404)
            self.assertEqual(
                json.loads(caught.exception.read()),
                {"error": "branch_not_found"},
            )
            caught.exception.close()
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)

    # ------------------------------------------------------------- 401 / 403 / 404

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.seed_pair()
        body = json.dumps(self.payload()).encode()
        for headers in (
            {},
            {"Authorization": "Bearer"},
            {"Authorization": "Bearer "},
            {"Authorization": "Basic w1"},
            {"Authorization": "Bearer ghost-token"},
        ):
            with self.subTest(headers=headers):
                request = Request(
                    f"{self.base_url}{ENDPOINT}",
                    data=body,
                    headers={"Content-Type": "application/json", **headers},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request, timeout=5)
                error = caught.exception
                self.assertEqual(error.code, 401)
                self.assertEqual(json.loads(error.read()),
                                 {"error": "unauthorized"})
                error.close()

    def test_credential_is_checked_before_media_type_and_body(self) -> None:
        self.seed_pair()
        # No credential plus an unsupported media type: 401 wins.
        status, raw = self.raw(
            ENDPOINT, body=b"{}", token=None, content_type="text/plain"
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(raw), {"error": "unauthorized"})
        # Unregistered credential plus malformed JSON: still 401.
        status, _ = self.raw(
            ENDPOINT, body=b"{", token="ghost", content_type="application/json"
        )
        self.assertEqual(status, 401)

    def test_body_shape_is_checked_before_organization(self) -> None:
        # A foreign-organization credential with a malformed body gets 422,
        # not 403: the body shape outranks the organization comparison.
        status, body = self.call(
            ENDPOINT,
            payload={**self.payload(), "windowSize": 0},
            token="w2",
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_organization_is_checked_before_branch_names(self) -> None:
        self.seed_pair()
        # A credential for org-2 naming org-1 in the body gets 403 even
        # when both branch names have never existed: branch names are not
        # probed across organizations.
        status, body = self.compare(
            self.payload(organizationId=ORG1, left="ghost", right="ghost"),
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_foreign_branch_is_403_not_404(self) -> None:
        self.seed_pair()
        self.add_event("org2-event", token="w2", occurred_at=10)
        self.capture("s2", token="w2")
        self.fork("br2", "s2", token="w2")
        status, body = self.compare(
            self.payload(organizationId=ORG2, left="L", right="br2"),
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_branches_are_resolved_left_before_right(self) -> None:
        self.seed_pair()
        # Missing left outranks a missing right: 404 branch_not_found.
        status, body = self.compare(
            self.payload(left="ghost", right="also-ghost")
        )
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "branch_not_found"})
        # A foreign left branch outranks a missing right branch: 403.
        self.add_event("org2-event", token="w2", occurred_at=10)
        self.capture("s2", token="w2")
        self.fork("br2", "s2", token="w2")
        status, body = self.call(
            ENDPOINT,
            payload=self.payload(
                organizationId=ORG2, left="L", right="ghost"
            ),
            token="w2",
        )
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": "forbidden"})

    def test_unknown_branch_is_404_and_never_created(self) -> None:
        self.seed_pair()
        status, body = self.compare(self.payload(right="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "branch_not_found"})
        status, again = self.compare(self.payload(right="ghost"))
        self.assertEqual(status, 404)
        self.assertEqual(again, {"error": "branch_not_found"})

    def test_failed_request_leaves_no_trace(self) -> None:
        self.seed_pair()
        status, _ = self.compare(self.payload(right="ghost"))
        self.assertEqual(status, 404)
        # Re-running a valid comparison afterwards is unchanged even when
        # other read-only parameters were used in between.
        status, before = self.compare()
        self.assertEqual(status, 200)
        status, _ = self.compare(self.payload(threshold=99))
        self.assertEqual(status, 200)
        status, after = self.compare()
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

    # ----------------------------------------------------------------- 415/400/422

    def test_unsupported_media_type_is_415(self) -> None:
        self.seed_pair()
        for content_type in (None, "text/plain", "application/x-www-form-urlencoded"):
            with self.subTest(content_type=content_type):
                status, body = self.raw(
                    ENDPOINT, body=b"{}", token="w1", content_type=content_type
                )
                self.assertEqual(status, 415)
                self.assertEqual(
                    json.loads(body), {"error": "unsupported_media_type"}
                )

    def test_malformed_json_is_400(self) -> None:
        self.seed_pair()
        for raw_body in (b"{", b"", b"[1, 2]", b'"a string"'):
            with self.subTest(raw_body=raw_body):
                status, body = self.raw(
                    ENDPOINT, body=raw_body, token="w1"
                )
                # An empty body is valid JSON nowhere; arrays/strings are
                # syntactically valid JSON and must reach 422 instead.
                if raw_body in (b"[1, 2]", b'"a string"'):
                    self.assertEqual(status, 422)
                    self.assertEqual(
                        json.loads(body)["error"], "validation_error"
                    )
                else:
                    self.assertEqual(status, 400)
                    self.assertEqual(json.loads(body), {"error": "invalid_json"})

    def test_required_and_allowed_fields(self) -> None:
        self.seed_pair()
        valid = self.payload()
        # Missing required fields and extra fields.
        for bad in (
            {},
            {k: v for k, v in valid.items() if k != "left"},
            {k: v for k, v in valid.items() if k != "threshold"},
            {k: v for k, v in valid.items() if k != "type"},
            {**valid, "extra": 1},
        ):
            with self.subTest(bad=bad):
                status, body = self.compare(bad)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")
        # Non-object JSON bodies reach validation and are rejected there.
        for raw_body in (b"[1, 2]", b'"a string"', b"5", b"null"):
            with self.subTest(raw_body=raw_body):
                status, body = self.call(ENDPOINT, raw_body=raw_body)
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_blank_or_non_string_text_fields_are_422(self) -> None:
        self.seed_pair()
        for field in ("organizationId", "left", "right", "type"):
            for value in ("", "   ", 5, True, ["L"], None):
                with self.subTest(field=field, value=value):
                    bad = self.payload(**{field: value})
                    status, body = self.compare(bad)
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_window_size_and_threshold_must_be_positive_integers(self) -> None:
        self.seed_pair()
        for field in ("windowSize", "threshold"):
            for value in (0, -1, 1.5, "60", True, None):
                with self.subTest(field=field, value=value):
                    status, body = self.compare(self.payload(**{field: value}))
                    self.assertEqual(status, 422)
                    self.assertEqual(body["error"], "validation_error")

    def test_range_must_be_paired_non_negative_and_ordered(self) -> None:
        self.seed_pair()
        for bad in (
            {"from": 0},
            {"to": 10},
            {"from": -1, "to": 10},
            {"from": 0, "to": -10},
            {"from": 1.5, "to": 10},
            {"from": "60", "to": 10},
            {"from": 10, "to": 9},
            {"from": True, "to": 10},
        ):
            with self.subTest(bad=bad):
                status, body = self.compare(self.payload(**bad))
                self.assertEqual(status, 422)
                self.assertEqual(body["error"], "validation_error")

    def test_from_equal_to_to_is_legal(self) -> None:
        self.seed_pair()
        status, body = self.compare(self.payload(**{"from": 0, "to": 0}))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 0)
        self.assertEqual(body["to"], 0)
        # Only the window containing 0 intersects the zero-width interval.
        for step in body["steps"]:
            self.assertEqual([row["start"] for row in step["windows"]], [0])

    def test_validation_failure_writes_nothing(self) -> None:
        self.seed_pair()
        status, body = self.compare(self.payload(windowSize=0))
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")
        # Each branch forked the three org-1 main events; L additionally
        # holds evt-l and evt-c, R evt-r.
        for branch, events in (("L", 5), ("R", 4)):
            status, summary = self.call(f"/branches/{branch}", method="GET")
            self.assertEqual(status, 200)
            self.assertEqual(summary["events"], events)


if __name__ == "__main__":
    unittest.main()
