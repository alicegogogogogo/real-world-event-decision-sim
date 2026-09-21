from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from event_sim.server import SERVICE_NAME, create_server


def valid_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "eventId": "evt-1",
        "organizationId": "org-1",
        "type": "incident.created",
        "occurredAt": 100,
        "payload": {"severity": "low"},
    }
    event.update(overrides)
    return event


class ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(
        self,
        path: str,
        body: object = None,
        *,
        method: str = "GET",
        content_type: str | None = "application/json",
        raw_body: bytes | None = None,
    ) -> tuple[int, object, object]:
        data: bytes | None
        if raw_body is not None:
            data = raw_body
        elif body is not None:
            data = json.dumps(body).encode()
        else:
            data = None
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.headers.get("Content-Type"), json.load(
                    response
                )
        except HTTPError as error:
            try:
                return error.code, error.headers.get("Content-Type"), json.load(error)
            finally:
                error.close()

    def test_health(self) -> None:
        with urlopen(f"{self.base_url}/health", timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                json.load(response),
                {"service": SERVICE_NAME, "status": "ok"},
            )

    def test_unknown_path_is_json_404(self) -> None:
        with self.assertRaises(HTTPError) as raised:
            urlopen(f"{self.base_url}/missing", timeout=2)
        error = raised.exception
        try:
            self.assertEqual(error.code, 404)
            self.assertEqual(
                json.load(error),
                {"error": "not_found", "path": "/missing"},
            )
        finally:
            error.close()

    def test_create_event_returns_201_with_exact_fields(self) -> None:
        event = valid_event()
        status, content_type, response = self._request("/events", event, method="POST")
        self.assertEqual(status, 201)
        self.assertEqual(content_type, "application/json; charset=utf-8")
        self.assertEqual(response, event)

    def test_create_event_accepts_content_type_with_charset(self) -> None:
        status, _, response = self._request(
            "/events",
            valid_event(),
            method="POST",
            content_type="application/json; charset=utf-8",
        )
        self.assertEqual(status, 201)
        self.assertEqual(response["eventId"], "evt-1")

    def test_create_event_zero_timestamp(self) -> None:
        status, _, _ = self._request(
            "/events", valid_event(eventId="evt-zero", occurredAt=0), method="POST"
        )
        self.assertEqual(status, 201)

    def test_missing_content_type_is_415(self) -> None:
        status, content_type, response = self._request(
            "/events",
            valid_event(eventId="evt-ct"),
            method="POST",
            content_type=None,
        )
        self.assertEqual(status, 415)
        self.assertEqual(content_type, "application/json; charset=utf-8")
        self.assertEqual(response["error"], "unsupported_media_type")

    def test_unsupported_content_type_is_415(self) -> None:
        status, _, response = self._request(
            "/events",
            valid_event(eventId="evt-ct2"),
            method="POST",
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        self.assertEqual(response["error"], "unsupported_media_type")

    def test_malformed_json_is_400(self) -> None:
        status, _, response = self._request(
            "/events", raw_body=b"{not json", method="POST"
        )
        self.assertEqual(status, 400)
        self.assertEqual(response["error"], "invalid_json")

    def test_json_array_is_422(self) -> None:
        status, _, response = self._request(
            "/events", raw_body=b"[1, 2, 3]", method="POST"
        )
        self.assertEqual(status, 422)
        self.assertEqual(response["error"], "validation_error")

    def test_missing_field_is_422(self) -> None:
        event = valid_event()
        del event["payload"]
        status, _, response = self._request("/events", event, method="POST")
        self.assertEqual(status, 422)
        self.assertEqual(response["error"], "validation_error")

    def test_extra_field_is_422(self) -> None:
        status, _, response = self._request(
            "/events", valid_event(extra="nope"), method="POST"
        )
        self.assertEqual(status, 422)
        self.assertEqual(response["error"], "validation_error")

    def test_blank_identifiers_are_422(self) -> None:
        for field in ("eventId", "organizationId", "type"):
            with self.subTest(field=field):
                event = valid_event(**{field: "   "})
                if field != "eventId":
                    event["eventId"] = f"evt-{field}"
                status, _, response = self._request(
                    "/events",
                    event,
                    method="POST",
                )
                self.assertEqual(status, 422)
                self.assertEqual(response["error"], "validation_error")

    def test_non_string_identifiers_are_422(self) -> None:
        cases = {
            "eventId": 12,
            "organizationId": None,
            "type": True,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                event = valid_event(**{field: value})
                if field != "eventId":
                    event["eventId"] = f"evt-type-{field}"
                status, _, response = self._request(
                    "/events",
                    event,
                    method="POST",
                )
                self.assertEqual(status, 422)

    def test_float_and_negative_timestamp_are_422(self) -> None:
        for value in (1.5, -1, True, "100", None):
            with self.subTest(value=value):
                status, _, response = self._request(
                    "/events",
                    valid_event(eventId=f"evt-ts-{value!r}", occurredAt=value),
                    method="POST",
                )
                self.assertEqual(status, 422)

    def test_payload_must_be_object(self) -> None:
        for payload in ([], "x", 1, None, [{}]):
            with self.subTest(payload=payload):
                status, _, response = self._request(
                    "/events",
                    valid_event(eventId=f"evt-p-{payload!r}", payload=payload),
                    method="POST",
                )
                self.assertEqual(status, 422)

    def test_idempotent_resubmission_returns_200_stored(self) -> None:
        event = valid_event()
        status1, _, response1 = self._request("/events", event, method="POST")
        self.assertEqual(status1, 201)
        status2, _, response2 = self._request(
            "/events", valid_event(payload={"severity": "low"}), method="POST"
        )
        self.assertEqual(status2, 200)
        self.assertEqual(response2, event)
        self.assertEqual(response1, response2)

        status, _, listing = self._request(
            f"/events?{urlencode({'organizationId': 'org-1'})}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["events"]), 1)

    def test_conflicting_event_id_is_409(self) -> None:
        status1, _, _ = self._request(
            "/events",
            valid_event(eventId="evt-dup", organizationId="org-a", occurredAt=1),
            method="POST",
        )
        self.assertEqual(status1, 201)
        status2, _, response = self._request(
            "/events",
            valid_event(eventId="evt-dup", organizationId="org-b", occurredAt=2),
            method="POST",
        )
        self.assertEqual(status2, 409)
        self.assertEqual(response["error"], "event_id_conflict")

        status, _, listing = self._request(
            f"/events?{urlencode({'organizationId': 'org-b'})}"
        )
        self.assertEqual(listing["events"], [])

    def test_list_events_sorted_deterministically(self) -> None:
        submitted = [
            valid_event(eventId="evt-c", organizationId="org-x", occurredAt=30),
            valid_event(eventId="evt-a", organizationId="org-x", occurredAt=10),
            valid_event(eventId="evt-b", organizationId="org-x", occurredAt=10),
            valid_event(eventId="evt-other", organizationId="org-y", occurredAt=1),
        ]
        for event in submitted:
            status, _, _ = self._request("/events", event, method="POST")
            self.assertEqual(status, 201)

        status, _, response = self._request(
            f"/events?{urlencode({'organizationId': 'org-x'})}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(response["organizationId"], "org-x")
        self.assertEqual(
            [event["eventId"] for event in response["events"]],
            ["evt-a", "evt-b", "evt-c"],
        )

    def test_list_events_unknown_organization_is_empty(self) -> None:
        status, _, response = self._request(
            f"/events?{urlencode({'organizationId': 'nope'})}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            response, {"organizationId": "nope", "events": []}
        )

    def test_list_events_requires_non_empty_organization_id(self) -> None:
        for query in ("", "other=1", "organizationId=", "organizationId=%20%20"):
            with self.subTest(query=query):
                status, _, response = self._request(f"/events?{query}")
                self.assertEqual(status, 422)
                self.assertEqual(response["error"], "validation_error")

    def test_list_events_rejects_repeated_organization_id(self) -> None:
        status, _, response = self._request(
            "/events?organizationId=org-a&organizationId=org-b"
        )
        self.assertEqual(status, 422)
        self.assertEqual(response["error"], "validation_error")

    def test_unknown_post_path_is_json_404(self) -> None:
        status, _, response = self._request(
            "/missing", valid_event(eventId="evt-404"), method="POST"
        )
        self.assertEqual(status, 404)
        self.assertEqual(response, {"error": "not_found", "path": "/missing"})

    def test_concurrent_identical_submissions_create_once(self) -> None:
        event = valid_event(eventId="evt-concurrent")

        def submit() -> tuple[int, object]:
            status, _, response = self._request("/events", event, method="POST")
            return status, response

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda _: submit(), range(12)))

        statuses = sorted(status for status, _ in results)
        self.assertEqual(statuses.count(201), 1)
        self.assertEqual(statuses.count(200), 11)
        for _, response in results:
            self.assertEqual(response, event)

        status, _, listing = self._request(
            f"/events?{urlencode({'organizationId': 'org-1'})}"
        )
        self.assertEqual(
            [event["eventId"] for event in listing["events"]], ["evt-concurrent"]
        )

    def test_concurrent_conflicting_submissions(self) -> None:
        versions = [
            valid_event(eventId="evt-race", type="a"),
            valid_event(eventId="evt-race", type="b"),
        ]

        def submit(index: int) -> tuple[int, int]:
            status, _, _ = self._request("/events", versions[index], method="POST")
            return index, status

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(submit, [i % 2 for i in range(8)]))

        statuses = [status for _, status in results]
        self.assertEqual(statuses.count(201), 1)

        # The single stored event decides which duplicate version gets 200
        # and which gets 409; no version may be created twice.
        listing_status, _, listing = self._request(
            f"/events?{urlencode({'organizationId': 'org-1'})}"
        )
        self.assertEqual(listing_status, 200)
        self.assertEqual(len(listing["events"]), 1)
        winner_type = listing["events"][0]["type"]
        for index, status in results:
            if status == 201:
                self.assertEqual(versions[index]["type"], winner_type)
            elif versions[index]["type"] == winner_type:
                self.assertEqual(status, 200)
            else:
                self.assertEqual(status, 409)

    def test_new_server_has_no_events(self) -> None:
        status, _, _ = self._request(
            "/events", valid_event(eventId="evt-fresh"), method="POST"
        )
        self.assertEqual(status, 201)

        fresh_server = create_server(port=0)
        fresh_thread = threading.Thread(target=fresh_server.serve_forever, daemon=True)
        fresh_thread.start()
        try:
            url = f"http://127.0.0.1:{fresh_server.server_port}"
            with urlopen(
                f"{url}/events?organizationId=org-1", timeout=2
            ) as response:
                self.assertEqual(
                    json.load(response),
                    {"organizationId": "org-1", "events": []},
                )
        finally:
            fresh_server.shutdown()
            fresh_server.server_close()
            fresh_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
