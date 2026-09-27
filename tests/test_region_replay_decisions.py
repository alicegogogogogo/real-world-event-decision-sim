"""Regression tests for the region-dimension step-by-step replay decisions.

The baseline already had the region queries (``GET /events/region`` and
``GET /events/region/aggregate``) and the step-by-step replay decisions
(``GET /events/replay/decisions`` and its snapshot counterpart), but no
region-dimension replay; this locks down the two new read-only entry
points:

- ``GET /events/region/replay/decisions``
- ``GET /snapshots/{snapshotId}/events/region/replay/decisions``

Only events whose payload ``region`` is a non-empty string matching the
requested region verbatim (and matching the organization and ``type``)
enter the replay; an unknown region matches zero events and yields no
steps. Each step reports the event id and time, the window rows of the
accumulated prefix (the exact division of the region aggregate), and the
peak decision at that point. The snapshot query counts only the caller's
organization's captured events; the main query counts only the caller's
organization's events. Both are read-only, byte-stable across repeats, and
share the fixed verdict order: 401 (credential) before 422 (query shape)
before 403 (organization, then foreign snapshot) before 404
(snapshot_not_found).
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
REGION = "north"


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


class RegionReplayDecisionsTest(unittest.TestCase):
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
    ) -> tuple[int, bytes]:
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
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

    def capture(self, snapshot_id: str = "s1", *, token: str = "w1") -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    @staticmethod
    def query(
        *,
        org: str = ORG1,
        region: str = REGION,
        event_type: str = EVENT_TYPE,
        window_size: int = 60,
        threshold: int = 2,
        from_to: tuple[int, int] | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "organizationId": org,
            "region": region,
            "type": event_type,
            "windowSize": window_size,
            "threshold": threshold,
        }
        if from_to is not None:
            params["from"], params["to"] = from_to
        return urlencode(params)

    def main_replay(
        self, query: str, *, token: str | None = "w1"
    ) -> tuple[int, Any]:
        return self.call(
            f"/events/region/replay/decisions?{query}", token=token
        )

    def main_replay_raw(
        self, query: str, *, token: str | None = "w1"
    ) -> tuple[int, bytes]:
        return self.raw(
            f"/events/region/replay/decisions?{query}", token=token
        )

    def snap_replay(
        self, query: str, *, snapshot: str = "s1", token: str | None = "w1"
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/events/region/replay/decisions?{query}",
            token=token,
        )

    def snap_replay_raw(
        self, query: str, *, snapshot: str = "s1", token: str | None = "w1"
    ) -> tuple[int, bytes]:
        return self.raw(
            f"/snapshots/{snapshot}/events/region/replay/decisions?{query}",
            token=token,
        )

    def seed_main(self) -> None:
        # Region/type matches, inserted out of order; evt-c ties evt-b's time.
        self.add_event("evt-b", occurred_at=30, payload={"region": "north"})
        self.add_event("evt-a", occurred_at=0, payload={"region": "north"})
        self.add_event("evt-c", occurred_at=30, payload={"region": "north"})
        self.add_event("evt-d", occurred_at=70, payload={"region": "north"})
        # Same type, another region: never a step.
        self.add_event("evt-s", occurred_at=10, payload={"region": "south"})
        # No region attribution: never a step.
        self.add_event("evt-n", occurred_at=20, payload={})
        self.add_event("evt-e", occurred_at=40, payload={"region": ""})
        # Same region, another type: never a step.
        self.add_event(
            "evt-t",
            occurred_at=25,
            event_type="other.kind",
            payload={"region": "north"},
        )
        # Another organization's matching event: never a step.
        self.add_event(
            "evt-x", token="w2", occurred_at=15, payload={"region": "north"}
        )

    # ------------------------------------------------------------- happy paths

    def test_steps_accumulate_only_region_and_type_matches(self) -> None:
        self.seed_main()

        status, body = self.main_replay(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["organizationId"], ORG1)
        self.assertEqual(body["region"], REGION)
        self.assertEqual(body["type"], EVENT_TYPE)
        self.assertEqual(body["windowSize"], 60)
        self.assertEqual(body["threshold"], 2)
        self.assertIsNone(body["from"])
        self.assertIsNone(body["to"])
        # Steps ordered by occurredAt, ties by eventId code-point order.
        self.assertEqual(
            [step["eventId"] for step in body["steps"]],
            ["evt-a", "evt-b", "evt-c", "evt-d"],
        )
        self.assertEqual(
            [step["occurredAt"] for step in body["steps"]],
            [0, 30, 30, 70],
        )
        self.assertEqual(
            body["steps"][0],
            {
                "eventId": "evt-a",
                "occurredAt": 0,
                "windows": [{"start": 0, "end": 60, "count": 1}],
                "peakStart": 0,
                "peakCount": 1,
                "action": "observe",
            },
        )
        self.assertEqual(
            body["steps"][1]["windows"],
            [{"start": 0, "end": 60, "count": 2}],
        )
        self.assertEqual(body["steps"][1]["action"], "escalate")
        self.assertEqual(body["steps"][1]["peakCount"], 2)
        self.assertEqual(body["steps"][1]["peakStart"], 0)
        self.assertEqual(
            body["steps"][2]["windows"],
            [{"start": 0, "end": 60, "count": 3}],
        )
        self.assertEqual(
            body["steps"][3]["windows"],
            [
                {"start": 0, "end": 60, "count": 3},
                {"start": 60, "end": 120, "count": 1},
            ],
        )
        self.assertEqual(body["steps"][3]["peakCount"], 3)

    def test_peak_tie_resolves_to_earliest_start(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=70, payload={"region": "north"})

        status, body = self.main_replay(self.query(threshold=5))
        self.assertEqual(status, 200)
        last = body["steps"][-1]
        self.assertEqual(last["peakCount"], 1)
        self.assertEqual(last["peakStart"], 0)
        self.assertEqual(last["action"], "observe")

    def test_region_matches_verbatim_without_normalization(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "North"})
        self.add_event("evt-2", occurred_at=10, payload={"region": "north "})

        status, body = self.main_replay(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], [])

        status, body = self.main_replay(self.query(region="North"))
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["evt-1"])

    def test_unknown_region_is_zero_match_with_no_steps(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})

        status, body = self.main_replay(self.query(region="atlantis"))
        self.assertEqual(status, 200)
        self.assertEqual(body["region"], "atlantis")
        self.assertEqual(body["steps"], [])

    def test_range_keeps_intersecting_empty_windows_at_every_step(self) -> None:
        self.add_event("evt-1", occurred_at=65, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=181, payload={"region": "north"})

        status, body = self.main_replay(self.query(from_to=(30, 180)))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 30)
        self.assertEqual(body["to"], 180)
        # Both events arrive as steps (the one at 181 is out of range but
        # still a step); every window intersecting [30, 180] is kept.
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["evt-1", "evt-2"]
        )
        self.assertEqual(
            body["steps"][0]["windows"],
            [
                {"start": 0, "end": 60, "count": 0},
                {"start": 60, "end": 120, "count": 1},
                {"start": 120, "end": 180, "count": 0},
                {"start": 180, "end": 240, "count": 0},
            ],
        )
        self.assertEqual(body["steps"][0]["peakCount"], 1)
        self.assertEqual(body["steps"][0]["peakStart"], 60)
        # The out-of-range event adds a step but no count.
        self.assertEqual(
            body["steps"][1]["windows"],
            body["steps"][0]["windows"],
        )

    def test_step_with_no_counted_window_observes_null_peak(self) -> None:
        self.add_event("evt-1", occurred_at=1000, payload={"region": "north"})

        status, body = self.main_replay(self.query(from_to=(0, 60)))
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        step = body["steps"][0]
        self.assertEqual(
            step["windows"],
            [
                {"start": 0, "end": 60, "count": 0},
                {"start": 60, "end": 120, "count": 0},
            ],
        )
        self.assertEqual(step["peakCount"], 0)
        self.assertIsNone(step["peakStart"])
        self.assertEqual(step["action"], "observe")

    # ------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})

        status, raw = self.main_replay_raw(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        self.assertNotIn(b": ", raw)
        self.assertIn(b'"windowSize":60', raw)
        body = json.loads(raw)
        self.assertEqual(
            list(body),
            [
                "from",
                "organizationId",
                "region",
                "steps",
                "threshold",
                "to",
                "type",
                "windowSize",
            ],
        )
        self.assertEqual(
            list(body["steps"][0]),
            ["action", "eventId", "occurredAt", "peakCount", "peakStart",
             "windows"],
        )
        self.assertEqual(
            list(body["steps"][0]["windows"][0]), ["count", "end", "start"]
        )

    def test_repeated_requests_are_byte_identical_and_do_not_pollute(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=50, payload={"region": "north"})
        query = self.query(from_to=(0, 120))
        first = self.main_replay_raw(query)[1]
        rest = [self.main_replay_raw(query)[1] for _ in range(3)]
        self.assertTrue(all(chunk == first for chunk in rest))

    # -------------------------------------------------------------- read-only

    def test_replay_is_read_only(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()
        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, _ = self.main_replay(self.query())
            self.assertEqual(status, 200)
            status, _ = self.snap_replay(self.query())
            self.assertEqual(status, 200)
        status, after = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

        status, events = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(events["events"]), 1)
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_replay_both_entry_points(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()

        status, body = self.main_replay(self.query(), token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)
        status, body = self.snap_replay(self.query(), token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["steps"]), 1)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        for path in (
            f"/events/region/replay/decisions?{self.query()}",
            f"/snapshots/s1/events/region/replay/decisions?{self.query()}",
        ):
            with self.subTest(path=path):
                status, body = self.raw(path, token=None)
                self.assertEqual(status, 401)
                self.assertEqual(json.loads(body), {"error": "unauthorized"})

                status, body = self.raw(path, token="forged")
                self.assertEqual(status, 401)
                self.assertEqual(json.loads(body), {"error": "unauthorized"})

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
                self.assertEqual(json.loads(body), {"error": "unauthorized"})

    def test_credential_outranks_query_validation(self) -> None:
        # Missing credential is 401 even though the query is also invalid.
        status, body = self.raw("/events/region/replay/decisions", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")
        status, body = self.raw(
            "/snapshots/s1/events/region/replay/decisions", token=None
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "unauthorized")

    # -------------------------------------------------------- parameter 422

    def test_query_shape_errors_are_422_on_both_entry_points(self) -> None:
        self.capture()
        bases = [
            "/events/region/replay/decisions",
            "/snapshots/s1/events/region/replay/decisions",
        ]
        bad_queries = [
            "",
            "organizationId=org-1",
            "organizationId=org-1&region=north",
            "organizationId=org-1&region=north&type=incident.created",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60",
            "region=north&type=incident.created&windowSize=60&threshold=2",
            "organizationId=org-1&type=incident.created&windowSize=60"
            "&threshold=2",
            "organizationId=org-1&region=north&windowSize=60&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60",
            "organizationId=&region=north&type=incident.created&windowSize=60"
            "&threshold=2",
            "organizationId=%20%20&region=north&type=incident.created"
            "&windowSize=60&threshold=2",
            "organizationId=org-1&organizationId=org-1&region=north"
            "&type=incident.created&windowSize=60&threshold=2",
            "organizationId=org-1&region=&type=incident.created&windowSize=60"
            "&threshold=2",
            "organizationId=org-1&region=%20&type=incident.created"
            "&windowSize=60&threshold=2",
            "organizationId=org-1&region=north&region=south"
            "&type=incident.created&windowSize=60&threshold=2",
            "organizationId=org-1&region=north&type=&windowSize=60&threshold=2",
            "organizationId=org-1&region=north&type=%20&windowSize=60"
            "&threshold=2",
            "organizationId=org-1&region=north&type=incident.created&type=x"
            "&windowSize=60&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=0&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=-3&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=1.5&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=abc&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&windowSize=30&threshold=2",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=0",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=-1",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2.5",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&threshold=3",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&from=0",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&to=60",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&from=-1&to=60",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&from=0&to=-1",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&from=abc&to=60",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&from=1.5&to=60",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&from=61&to=60",
            "organizationId=org-1&region=north&type=incident.created"
            "&windowSize=60&threshold=2&from=0&to=60&from=10",
        ]
        for base in bases:
            for query in bad_queries:
                with self.subTest(base=base, query=query):
                    status, body = self.raw(f"{base}?{query}", token="w1")
                    self.assertEqual(status, 422)
                    self.assertEqual(
                        json.loads(body)["error"], "validation_error"
                    )

        # No failed validation created or altered anything.
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(len(snapshots["snapshots"]), 1)
        self.assertEqual(snapshots["snapshots"][0]["events"], 0)

    def test_query_validation_outranks_organization_and_snapshot(self) -> None:
        self.capture()
        status, body = self.raw(
            "/events/region/replay/decisions"
            "?organizationId=org-2&organizationId=org-2"
            "&region=north&type=t&windowSize=60&threshold=2",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")
        status, body = self.raw(
            "/snapshots/ghost/events/region/replay/decisions"
            "?organizationId=org-2&organizationId=org-2"
            "&region=north&type=t&windowSize=60&threshold=2",
            token="w1",
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403(self) -> None:
        self.capture()
        status, body = self.main_replay(self.query(org=ORG1), token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        status, body = self.snap_replay(self.query(org=ORG1), token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_organization_outranks_snapshot_lookup(self) -> None:
        self.capture()
        # The organization decision precedes even the snapshot lookup: an
        # unknown snapshot name is still 403 for a foreign organization.
        status, body = self.snap_replay(
            self.query(org=ORG1), snapshot="ghost", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        status, body = self.snap_replay(self.query(), snapshot="s2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.snap_replay(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(snapshots["snapshots"], [])

    def test_snapshot_lookup_outranks_snapshot_ownership(self) -> None:
        self.capture("s2", token="w2")
        # A missing snapshot is 404 even though a foreign snapshot exists.
        status, body = self.snap_replay(self.query(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    # ------------------------------------------------------ snapshot scoping

    def test_snapshot_counts_only_captured_events(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.add_event("evt-2", occurred_at=20, payload={"region": "north"})
        self.capture()
        # Committed after the capture: never enters the snapshot replay.
        self.add_event("evt-3", occurred_at=30, payload={"region": "north"})

        status, body = self.snap_replay(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(
            [step["eventId"] for step in body["steps"]], ["evt-1", "evt-2"]
        )

        status, main_body = self.main_replay(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(
            [step["eventId"] for step in main_body["steps"]],
            ["evt-1", "evt-2", "evt-3"],
        )

    def test_snapshot_holds_only_owning_organization_events(self) -> None:
        self.add_event("evt-1", occurred_at=10, payload={"region": "north"})
        self.capture("s1")
        self.add_event(
            "evt-a", token="w2", occurred_at=10, payload={"region": "north"}
        )
        self.capture("s2", token="w2")

        status, body = self.snap_replay(self.query(), snapshot="s1")
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["evt-1"])

        status, body = self.snap_replay(
            self.query(org=ORG2), snapshot="s2", token="w2"
        )
        self.assertEqual(status, 200)
        self.assertEqual([step["eventId"] for step in body["steps"]], ["evt-a"])

    def test_snapshot_response_echoes_snapshot_and_region(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()

        status, raw = self.snap_replay_raw(self.query())
        self.assertEqual(status, 200)
        self.assertEqual(raw[-1:], b"\n")
        self.assertEqual(raw.count(b"\n"), 1)
        self.assertNotIn(b", ", raw)
        body = json.loads(raw)
        self.assertEqual(
            list(body),
            [
                "from",
                "organizationId",
                "region",
                "snapshotId",
                "steps",
                "threshold",
                "to",
                "type",
                "windowSize",
            ],
        )
        self.assertEqual(body["snapshotId"], "s1")
        self.assertEqual(body["region"], REGION)

    def test_snapshot_repeated_requests_are_byte_identical(self) -> None:
        self.add_event("evt-1", occurred_at=100, payload={"region": "north"})
        self.capture()
        query = self.query(from_to=(0, 120))
        first = self.snap_replay_raw(query)[1]
        # Main-service writes between reads must not perturb the snapshot.
        self.add_event("evt-2", occurred_at=10, payload={"region": "north"})
        rest = [self.snap_replay_raw(query)[1] for _ in range(3)]
        self.assertTrue(all(chunk == first for chunk in rest))


if __name__ == "__main__":
    unittest.main()
