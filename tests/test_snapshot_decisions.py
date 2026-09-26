"""Regression tests for POST /snapshots/{snapshotId}/decisions/evaluate.

The baseline already had the single-snapshot event listing and window
aggregate plus the two-snapshot window decision comparison, but no
snapshot-dimension peak decision; this locks down the new read-only entry
point that lowers the decision contract of ``POST /decisions/evaluate``
onto the events one snapshot captured at its creation time:

- only the snapshot's captured events of the caller's organization and the
  requested ``type`` are counted; other organizations' data and events
  committed after capture never enter the result;
- the window division is exactly the main decision's: windows start at
  zero and cover ``[start, start + windowSize)``; without ``from``/``to``
  only windows hit by matching events are considered, and with a range
  every window intersecting the closed interval is considered, including
  empty windows counted as zero;
- the peak is the largest window count, ties resolve to the earliest
  start, ``action`` is ``escalate`` when the peak reaches ``threshold``
  and ``observe`` otherwise, and an empty match reports ``peakCount: 0``
  with ``peakStart: null``;
- the response echoes the organization, snapshot, type, window width,
  threshold and range, is compact key-sorted JSON with integer values and
  one trailing newline, and identical requests are byte-for-byte
  identical without polluting each other;
- a snapshot evaluated alone agrees item by item with the two-snapshot
  window comparison's decision for that snapshot compared with itself;
- both ``read`` and ``write`` credentials may call it; the verdict order
  is fixed — 401 (credential) before 415/400/422 (request body) before
  403 (organization, then foreign snapshot) before 404
  (snapshot_not_found) — nothing is ever written or implicitly created,
  and restarting clears snapshots and events.
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


class SnapshotDecisionTest(unittest.TestCase):
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

    def capture(self, snapshot_id: str = "s1", *, token: str = "w1") -> None:
        status, _ = self.call(
            "/snapshots",
            method="POST",
            token=token,
            payload={"snapshotId": snapshot_id},
        )
        self.assertEqual(status, 201)

    @staticmethod
    def decision_body(
        *,
        org: str = ORG1,
        event_type: str = EVENT_TYPE,
        window_size: Any = 60,
        threshold: Any = 3,
        from_to: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "organizationId": org,
            "type": event_type,
            "windowSize": window_size,
            "threshold": threshold,
        }
        if from_to is not None:
            body["from"], body["to"] = from_to
        return body

    def evaluate(
        self,
        payload: Any,
        *,
        snapshot: str = "s1",
        token: str | None = "w1",
    ) -> tuple[int, Any]:
        return self.call(
            f"/snapshots/{snapshot}/decisions/evaluate",
            method="POST",
            payload=payload,
            token=token,
        )

    def evaluate_raw(
        self,
        body: bytes | None,
        *,
        snapshot: str = "s1",
        token: str | None = "w1",
        content_type: str | None = "application/json",
    ) -> tuple[int, bytes]:
        return self.raw(
            f"/snapshots/{snapshot}/decisions/evaluate",
            method="POST",
            body=body,
            token=token,
            content_type=content_type,
        )

    # ------------------------------------------------------------- happy paths

    def test_peak_decision_over_captured_events(self) -> None:
        self.add_event("evt-1", occurred_at=0)
        self.add_event("evt-2", occurred_at=59)
        self.add_event("evt-3", occurred_at=60)
        self.add_event("evt-4", occurred_at=10, event_type="other.kind")
        self.capture()

        status, body = self.evaluate(self.decision_body())
        self.assertEqual(status, 200)
        self.assertEqual(
            body,
            {
                "organizationId": ORG1,
                "snapshotId": "s1",
                "type": EVENT_TYPE,
                "windowSize": 60,
                "from": None,
                "to": None,
                "peakStart": 0,
                "peakCount": 2,
                "action": "observe",
            },
        )

    def test_escalate_when_peak_reaches_threshold(self) -> None:
        for index, timestamp in enumerate((0, 1, 59)):
            self.add_event(f"evt-{index}", occurred_at=timestamp)
        self.capture()

        status, body = self.evaluate(self.decision_body(threshold=3))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakStart"], 0)
        self.assertEqual(body["peakCount"], 3)
        self.assertEqual(body["action"], "escalate")

    def test_tie_resolves_to_earliest_window_start(self) -> None:
        for index, timestamp in enumerate((0, 10, 60, 70)):
            self.add_event(f"evt-{index}", occurred_at=timestamp)
        self.capture()

        status, body = self.evaluate(self.decision_body(threshold=5))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["peakStart"], 0)
        self.assertEqual(body["action"], "observe")

    def test_window_is_left_closed_right_open_at_boundaries(self) -> None:
        # The event exactly on the boundary belongs to the upper window —
        # the same division as the main decision endpoint.
        for index, timestamp in enumerate((59, 60, 119)):
            self.add_event(f"evt-{index}", occurred_at=timestamp)
        self.capture()

        status, body = self.evaluate(self.decision_body(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "escalate")

    def test_no_matching_events_is_zero_count_and_null_start(self) -> None:
        self.add_event("evt-1", occurred_at=10, event_type="other.kind")
        self.capture()

        status, body = self.evaluate(self.decision_body())
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")

    def test_empty_snapshot_is_zero_count_and_null_start(self) -> None:
        self.capture()
        status, body = self.evaluate(self.decision_body())
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")

    def test_unknown_type_computes_as_zero_events(self) -> None:
        self.add_event("evt-1", occurred_at=10)
        self.capture()
        status, body = self.evaluate(
            self.decision_body(event_type="never.seen")
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])

    def test_range_considers_intersecting_empty_windows_as_zero(self) -> None:
        self.add_event("evt-1", occurred_at=65)
        self.add_event("evt-2", occurred_at=181)
        self.capture()

        status, body = self.evaluate(self.decision_body(from_to=(30, 180)))
        self.assertEqual(status, 200)
        self.assertEqual(body["from"], 30)
        self.assertEqual(body["to"], 180)
        # [30,180] intersects the windows starting 0, 60, 120 and 180; the
        # event at 181 falls outside the closed interval, so only the
        # window at 60 counts — the intersecting empty windows count zero.
        self.assertEqual(body["peakStart"], 60)
        self.assertEqual(body["peakCount"], 1)

    def test_range_with_zero_matches_is_null_start(self) -> None:
        self.add_event("evt-1", occurred_at=1000)
        self.capture()
        status, body = self.evaluate(self.decision_body(from_to=(0, 120)))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 0)
        self.assertIsNone(body["peakStart"])
        self.assertEqual(body["action"], "observe")

    def test_range_filters_events_by_closed_interval(self) -> None:
        for index, timestamp in enumerate((0, 10, 59, 60, 61)):
            self.add_event(f"evt-{index}", occurred_at=timestamp)
        self.capture()

        status, body = self.evaluate(self.decision_body(from_to=(10, 60)))
        self.assertEqual(status, 200)
        # 10 and 59 land in window 0, 60 is inclusive in window 60.
        self.assertEqual(body["peakStart"], 0)
        self.assertEqual(body["peakCount"], 2)

    def test_events_committed_after_capture_never_enter(self) -> None:
        self.add_event("evt-1", occurred_at=10)
        self.capture()
        self.add_event("evt-2", occurred_at=10)
        self.add_event("evt-3", occurred_at=10)

        status, body = self.evaluate(self.decision_body(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 1)
        self.assertEqual(body["action"], "observe")

    def test_other_organizations_never_contribute(self) -> None:
        self.add_event("evt-1", occurred_at=10)
        self.capture("s1")
        self.add_event("evt-a", token="w2", occurred_at=10)
        self.add_event("evt-b", token="w2", occurred_at=11)
        self.capture("s2", token="w2")

        status, body = self.evaluate(self.decision_body(threshold=2))
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 1)
        self.assertEqual(body["action"], "observe")

        status, body = self.evaluate(
            self.decision_body(org=ORG2, threshold=2),
            snapshot="s2",
            token="w2",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 2)
        self.assertEqual(body["action"], "escalate")

    # ---------------------------------------------------------- byte contract

    def test_response_is_compact_sorted_integer_and_newline_terminated(
        self,
    ) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()

        status, raw = self.evaluate_raw(
            json.dumps(self.decision_body()).encode()
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            raw,
            b'{"action":"observe","from":null,"organizationId":"org-1",'
            b'"peakCount":1,"peakStart":60,"snapshotId":"s1",'
            b'"to":null,"type":"incident.created","windowSize":60}\n',
        )

    def test_repeated_requests_are_byte_identical_and_do_not_pollute(
        self,
    ) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.add_event("evt-2", occurred_at=50)
        self.capture()
        payload = json.dumps(self.decision_body(from_to=(0, 120))).encode()
        first = self.evaluate_raw(payload)[1]
        # Main-service writes between reads must not perturb bytes or leak
        # into the snapshot; neither read affects the other.
        self.add_event("evt-3", occurred_at=10)
        self.add_event("evt-4", occurred_at=200)
        rest = [self.evaluate_raw(payload)[1] for _ in range(3)]
        self.assertTrue(all(chunk == first for chunk in rest))

    # ------------------------------------------- consistency with comparison

    def test_self_comparison_decision_matches_item_by_item(self) -> None:
        self.add_event("evt-1", occurred_at=5)
        self.add_event("evt-2", occurred_at=65)
        self.add_event("evt-3", occurred_at=66)
        self.add_event("evt-other", occurred_at=5, event_type="x.raised")
        self.capture()

        for from_to in (None, (0, 180), (30, 130)):
            with self.subTest(from_to=from_to):
                status, decision = self.evaluate(
                    self.decision_body(threshold=2, from_to=from_to)
                )
                self.assertEqual(status, 200)

                compare_payload: dict[str, Any] = {
                    "organizationId": ORG1,
                    "left": "s1",
                    "right": "s1",
                    "type": EVENT_TYPE,
                    "windowSize": 60,
                    "threshold": 2,
                }
                if from_to is not None:
                    compare_payload["from"], compare_payload["to"] = from_to
                status, comparison = self.call(
                    "/snapshots/compare",
                    method="POST",
                    payload=compare_payload,
                )
                self.assertEqual(status, 200)

                # The single-snapshot decision agrees with the comparison's
                # decision for the snapshot compared with itself, item by
                # item; both sides are equal by construction.
                for side in ("left", "right"):
                    self.assertEqual(
                        comparison["decision"][side]["peakStart"],
                        decision["peakStart"],
                    )
                    self.assertEqual(
                        comparison["decision"][side]["peakCount"],
                        decision["peakCount"],
                    )
                    self.assertEqual(
                        comparison["decision"][side]["action"],
                        decision["action"],
                    )
                self.assertTrue(comparison["decision"]["equal"])

    def test_matches_main_decision_at_capture_time_only(self) -> None:
        self.add_event("evt-1", occurred_at=10)
        self.capture()
        self.add_event("evt-2", occurred_at=20)
        self.add_event("evt-3", occurred_at=30)

        status, main_body = self.call(
            "/decisions/evaluate",
            method="POST",
            payload=self.decision_body(threshold=2),
        )
        self.assertEqual(status, 200)
        status, snapshot_body = self.evaluate(self.decision_body(threshold=2))
        self.assertEqual(status, 200)
        # The main ledger has since moved on; the snapshot stays at capture.
        self.assertEqual(main_body["peakCount"], 3)
        self.assertEqual(main_body["action"], "escalate")
        self.assertEqual(snapshot_body["peakCount"], 1)
        self.assertEqual(snapshot_body["action"], "observe")

    # -------------------------------------------------------------- read-only

    def test_decision_is_read_only_even_when_escalating(self) -> None:
        for index, timestamp in enumerate((0, 1, 2)):
            self.add_event(f"evt-{index}", occurred_at=timestamp)
        self.capture()
        status, before = self.call("/snapshots")
        self.assertEqual(status, 200)
        for _ in range(3):
            status, body = self.evaluate(self.decision_body(threshold=3))
            self.assertEqual(status, 200)
            self.assertEqual(body["action"], "escalate")
        status, after = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(before, after)

        # An escalating snapshot decision never writes the main-service
        # ledger, inventory, or alert state.
        status, events = self.call("/events?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(events["events"]), 3)
        status, reservations = self.call("/reservations?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(reservations["reservations"], [])
        status, alerts = self.call("/alerts?organizationId=org-1")
        self.assertEqual(status, 200)
        self.assertEqual(alerts["alerts"], [])

    # ------------------------------------------------------------ roles / auth

    def test_read_credential_may_evaluate(self) -> None:
        self.add_event("evt-1", occurred_at=100)
        self.capture()
        status, body = self.evaluate(self.decision_body(), token="r1")
        self.assertEqual(status, 200)
        self.assertEqual(body["peakCount"], 1)
        self.assertEqual(body["peakStart"], 60)

    def test_missing_malformed_or_unregistered_credential_is_401(self) -> None:
        self.capture()
        payload = json.dumps(self.decision_body()).encode()
        for token in (None, "forged"):
            with self.subTest(token=token):
                status, body = self.evaluate_raw(payload, token=token)
                self.assertEqual(status, 401)
                self.assertEqual(json.loads(body), {"error": "unauthorized"})

        request = Request(
            f"{self.base_url}/snapshots/s1/decisions/evaluate",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Basic abc",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                status, body = response.status, response.read()
        except HTTPError as error:
            status, body = error.code, error.read()
            error.close()
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"error": "unauthorized"})

    def test_credential_outranks_body_validation(self) -> None:
        self.capture()
        # Missing credential is 401 even though the media type is also
        # unsupported and the body is not JSON.
        status, body = self.evaluate_raw(
            b"not json", token=None, content_type="text/plain"
        )
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"error": "unauthorized"})

    # --------------------------------------------------------- media type 415

    def test_missing_or_unsupported_media_type_is_415(self) -> None:
        self.capture()
        payload = json.dumps(self.decision_body()).encode()
        for content_type in (None, "text/plain"):
            with self.subTest(content_type=content_type):
                status, body = self.evaluate_raw(
                    payload, content_type=content_type
                )
                self.assertEqual(status, 415)
                self.assertEqual(
                    json.loads(body), {"error": "unsupported_media_type"}
                )

    # ------------------------------------------------------------ bad JSON 400

    def test_invalid_json_is_400(self) -> None:
        self.capture()
        status, body = self.evaluate_raw(b"not json")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "invalid_json"})

    # ------------------------------------------------------------ body 422

    def test_body_shape_errors_are_422(self) -> None:
        self.capture()
        good = self.decision_body()
        bad_payloads: list[Any] = [
            [],
            "text",
            42,
            None,
            # Missing required fields.
            {k: v for k, v in good.items() if k != "organizationId"},
            {k: v for k, v in good.items() if k != "type"},
            {k: v for k, v in good.items() if k != "windowSize"},
            {k: v for k, v in good.items() if k != "threshold"},
            # Extra fields.
            {**good, "snapshotId": "s1"},
            {**good, "region": "north"},
            # Blank or non-string identifiers.
            {**good, "organizationId": ""},
            {**good, "organizationId": "  "},
            {**good, "organizationId": 7},
            {**good, "type": ""},
            {**good, "type": "   "},
            {**good, "type": None},
            # Non-positive-integer windowSize / threshold.
            {**good, "windowSize": 0},
            {**good, "windowSize": -3},
            {**good, "windowSize": 1.5},
            {**good, "windowSize": "60"},
            {**good, "windowSize": True},
            {**good, "windowSize": None},
            {**good, "threshold": 0},
            {**good, "threshold": -1},
            {**good, "threshold": 2.0},
            {**good, "threshold": "3"},
            {**good, "threshold": False},
            # Unpaired or invalid from/to.
            {**good, "from": 0},
            {**good, "to": 60},
            {**good, "from": 61, "to": 60},
            {**good, "from": -1, "to": 60},
            {**good, "from": 0, "to": -1},
            {**good, "from": 0.5, "to": 60},
            {**good, "from": 0, "to": True},
            {**good, "from": "0", "to": "60"},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                status, body = self.evaluate_raw(
                    json.dumps(payload).encode()
                )
                self.assertEqual(status, 422)
                self.assertEqual(
                    json.loads(body)["error"], "validation_error"
                )

        # No failed validation created or altered anything.
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(len(snapshots["snapshots"]), 1)
        self.assertEqual(snapshots["snapshots"][0]["events"], 0)

    def test_body_validation_outranks_organization_and_snapshot(self) -> None:
        self.capture()
        # A foreign organization and an unknown snapshot are both invisible
        # behind a body validation failure.
        status, body = self.evaluate(
            {"organizationId": ORG2}, snapshot="ghost"
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    # -------------------------------------------------------- organization 403

    def test_foreign_organization_request_is_403_before_snapshot_lookup(
        self,
    ) -> None:
        self.capture()
        status, body = self.evaluate(self.decision_body(), token="w2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")
        # The organization decision precedes even the snapshot lookup: an
        # unknown snapshot name is still 403 for a foreign organization.
        status, body = self.evaluate(
            self.decision_body(), snapshot="ghost", token="w2"
        )
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    def test_foreign_snapshot_is_403_not_404(self) -> None:
        self.capture("s1")
        self.capture("s2", token="w2")
        status, body = self.evaluate(self.decision_body(), snapshot="s2")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "forbidden")

    # ------------------------------------------------------------------ 404

    def test_unknown_snapshot_is_404_and_never_created(self) -> None:
        status, body = self.evaluate(self.decision_body(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")
        status, snapshots = self.call("/snapshots")
        self.assertEqual(status, 200)
        self.assertEqual(snapshots["snapshots"], [])

    def test_snapshot_lookup_outranks_snapshot_ownership(self) -> None:
        self.capture("s2", token="w2")
        # A missing snapshot is 404 even though a foreign snapshot exists.
        status, body = self.evaluate(self.decision_body(), snapshot="ghost")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "snapshot_not_found")

    def test_unknown_snapshot_subpath_is_generic_404(self) -> None:
        self.capture()
        for path in (
            "/snapshots/s1/decisions",
            "/snapshots/s1/decisions/evaluate/extra",
            "/snapshots/s1/unknown",
        ):
            with self.subTest(path=path):
                status, body = self.call(
                    path, method="POST", payload=self.decision_body()
                )
                self.assertEqual(status, 404)
                self.assertEqual(body["error"], "not_found")

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
                f"{base_url}/snapshots/s1/decisions/evaluate",
                data=json.dumps(self.decision_body()).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer tok-fresh",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=5) as response:
                    status = response.status
            except HTTPError as error:
                status = error.code
                self.assertEqual(
                    json.loads(error.read())["error"], "snapshot_not_found"
                )
                error.close()
            self.assertEqual(status, 404)
        finally:
            fresh.shutdown()
            fresh.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
