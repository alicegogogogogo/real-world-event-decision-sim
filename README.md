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
- `GET /events?organizationId=...` lists an organization's events.
- `GET /events/aggregate?...` returns event counts grouped into fixed time windows.
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

### `GET /events/aggregate`

Counts stored events for one organization and event type, grouped into
fixed-width time windows. Windows start at `0` and are `windowSize` wide; an
event belongs to window `start` when `start <= occurredAt < end` (so `end` is
exclusive). The ledger is never modified and only matching events of the
requested organization are counted.

Query parameters:

| Parameter      | Rule                                                                   |
| -------------- | ---------------------------------------------------------------------- |
| `organizationId` | required exactly once; non-empty string                              |
| `type`         | required exactly once; non-empty string                                |
| `windowSize`   | required exactly once; positive integer text (e.g. `100`, not `0`)    |
| `from`         | optional; must be paired with `to`; non-negative integer text          |
| `to`           | optional; must be paired with `from`; non-negative integer text, `from <= to` |

`from` and `to` must either both be omitted or both appear exactly once. Only
events with `from <= occurredAt <= to` are counted when the range is given.

Returns `200` with:

```json
{
  "organizationId": "org-1",
  "type": "incident.created",
  "windowSize": 100,
  "from": 0,
  "to": 250,
  "windows": [
    {"start": 0, "end": 100, "count": 1},
    {"start": 100, "end": 200, "count": 0},
    {"start": 200, "end": 300, "count": 2}
  ]
}
```

- Without `from`/`to`, only windows actually covered by matching events are
  returned; if no event matches, `"windows": []` (and `from`/`to` are `null`).
- With `from`/`to`, every window that intersects the closed interval
  `[from, to]` is returned in ascending `start` order, including empty windows
  with `"count": 0`.
- A missing, duplicated, blank, or malformed parameter (including a lone
  `from`/`to`, `from > to`, a zero `windowSize`, or a negative or non-integer
  value) yields `422` with an `error` field.

