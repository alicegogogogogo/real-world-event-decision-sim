# Real-world Event Decision Simulator

A small backend foundation for an emergency-response and logistics decision simulation platform. The service exposes a public HTTP entry point that future features can extend without changing the startup contract.

## Run

Requires Python 3.11 or newer and has no third-party runtime dependencies.

```bash
python -m event_sim serve --host 127.0.0.1 --port 8000
```

Check the public health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Public interface

- `GET /health` returns JSON with the service name and `ok` status.
- Unknown paths return a JSON `not_found` error with HTTP 404.
- `python -m event_sim --help` documents the command-line entry point.

## Event ledger

Events are held only in the server process: restarting (or starting a new
instance) begins with an empty ledger.

### `POST /events`

Requires `Content-Type: application/json`. The body must be a JSON object with
exactly these five top-level fields:

| Field          | Rule                                      |
| -------------- | ----------------------------------------- |
| `eventId`      | non-empty string                          |
| `organizationId` | non-empty string                        |
| `type`         | non-empty string                          |
| `occurredAt`   | non-negative integer (no floats, no booleans) |
| `payload`      | JSON object                               |

```bash
curl -X POST http://127.0.0.1:8000/events \
  -H 'Content-Type: application/json' \
  -d '{"eventId":"evt-1","organizationId":"org-1","type":"incident.created","occurredAt":100,"payload":{"severity":"low"}}'
```

- `201 Created` — new event stored; the response is the same five-field JSON.
- `200 OK` — the same `eventId` was resubmitted with identical fields; the
  stored event is returned and nothing is added.
- `409 Conflict` (`{"error": "event_id_conflict"}`) — the `eventId` exists but
  the other fields differ.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` — fields are missing, extra, or invalid
  (arrays at the top level, blank identifiers, floats, etc.).

### `GET /events?organizationId=...`

The `organizationId` query parameter must appear exactly once and be non-empty.
Returns `200` with `{"organizationId": "...", "events": [...]}`, where events
belong only to that organization and are sorted by `occurredAt`, then
`eventId`. An unknown organization returns `"events": []`. A missing, blank, or
duplicated parameter yields `422`.

### `GET /events/aggregate?organizationId=...&type=...&windowSize=...`

Counts stored events per time window, without modifying the ledger. Only
events matching both `organizationId` and `type` are counted; other
organizations' data is never included.

Query parameters:

| Parameter        | Rule                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| `organizationId` | required exactly once, non-empty string                              |
| `type`           | required exactly once, non-empty string                              |
| `windowSize`     | required exactly once, positive integer text (e.g. `60`)             |
| `from`           | optional; non-negative integer text, exactly once if present         |
| `to`             | optional; non-negative integer text, exactly once if present         |

`from` and `to` must be omitted together or supplied together, and must
satisfy `from <= to`. A missing, duplicated, or invalid parameter yields
`422` with a JSON body containing a stable `error` field.

Windows start at `0` and each covers `[start, end)` with
`end = start + windowSize`; an event belongs to a window when
`start <= occurredAt < end`. The `200` response is:

```json
{
  "organizationId": "org-1",
  "type": "incident.created",
  "windowSize": 60,
  "from": null,
  "to": null,
  "windows": [{"start": 0, "end": 60, "count": 2}]
}
```

- Without `from`/`to` (`from` and `to` echo back as `null`), only windows
  actually covered by matching events are returned; with no matching events
  `windows` is `[]`.
- With `from`/`to`, only events with `from <= occurredAt <= to` are counted,
  and every window intersecting the closed interval `[from, to]` is returned,
  including empty windows with `count: 0`.
- `windows` is sorted by `start` ascending; results are deterministic
  regardless of event insertion order, replays, or identical timestamps.

```bash
curl 'http://127.0.0.1:8000/events/aggregate?organizationId=org-1&type=incident.created&windowSize=60&from=0&to=180'
```

### `POST /decisions/evaluate`

A read-only, deterministic emergency decision over the same windows as the
aggregate endpoint. The ledger is never written to or changed by a decision
request. Requires `Content-Type: application/json`. The body must be a JSON
object containing exactly these fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string                                   |
| `type`           | required, non-empty string                                   |
| `windowSize`     | required, positive integer (no booleans, floats, or strings) |
| `threshold`      | required, positive integer (no booleans, floats, or strings) |
| `from`           | optional; non-negative integer, present only together with `to` |
| `to`             | optional; non-negative integer, present only together with `from` |

Only events matching both `organizationId` and `type` are counted. Windows
start at `0` and cover `[start, start + windowSize)`. Without `from`/`to`,
only windows actually covered by matching events are considered; with a
range, events are filtered by the closed interval `from <= occurredAt <= to`
and every window intersecting that interval is considered, including empty
ones for audit reconciliation.

The `200` response has a fixed shape:

```json
{
  "organizationId": "org-1",
  "type": "incident.created",
  "windowSize": 60,
  "from": null,
  "to": null,
  "peakStart": 60,
  "peakCount": 2,
  "action": "observe"
}
```

- `peakCount` is the largest window count and `peakStart` is that window's
  start; ties resolve to the earliest start, so results are reproducible
  regardless of event submission order.
- With no matching events (or no non-empty windows in range), `peakCount` is
  `0`, `peakStart` is `null`, and `action` is `"observe"`.
- `action` is `"escalate"` when `peakCount >= threshold`, otherwise
  `"observe"`.
- Unknown organizations or types compute as zero events and return normally;
  other organizations' data is never included.

```bash
curl -X POST http://127.0.0.1:8000/decisions/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","type":"incident.created","windowSize":60,"threshold":3}'
```

- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON; no decision is
  produced.
- `422 Unprocessable Entity` — a non-object body, missing or extra fields,
  invalid types, unpaired `from`/`to`, or out-of-range values.

