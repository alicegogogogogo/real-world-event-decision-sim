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

### `POST /decisions/allocate`

A read-only, deterministic resource plan computed entirely from the request
body. Nothing is written to the event ledger, no earlier plan is inherited,
and concurrent requests cannot affect each other. Requires
`Content-Type: application/json`. The body must be a JSON object containing
exactly these fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string                                   |
| `demands`        | required array (may be empty) of demand objects              |
| `resources`      | required array (may be empty) of resource objects            |

A demand object has exactly `demandId` (non-empty string, unique within the
request), `units` (positive integer), and `priority` (non-negative integer).
A resource object has exactly `resourceId` (non-empty string, unique within
the request) and `capacity` (positive integer). Booleans, floats, negative or
zero values where a positive integer is required, wrong element types, blank
or duplicate identifiers, and unknown fields are all rejected.

Demands are handled by `priority` descending; ties resolve by `demandId` in
Unicode code-point order. A demand is placed wholly into the
lexicographically smallest `resourceId` whose remaining capacity covers its
units, and that capacity is deducted immediately. Demands are never split and
capacity is never oversold; a demand that fits in no remaining resource goes
to `unassigned` — priorities and units are never adjusted to place more
demands.

The `200` response has a fixed shape:

```json
{
  "organizationId": "org-1",
  "assignments": [
    {"demandId": "d-a", "resourceId": "r-a", "units": 2},
    {"demandId": "d-b", "resourceId": "r-b", "units": 4}
  ],
  "unassigned": ["d-big"],
  "totalUnits": 6
}
```

- `assignments` lists `{"demandId", "resourceId", "units"}` objects in
  processing order.
- `unassigned` lists the demandIds that could not be placed, sorted by
  `demandId`.
- `totalUnits` counts only units that were assigned.
- Empty `demands`/`resources` arrays are valid and return empty results;
  identical submissions return byte-for-byte identical JSON.

```bash
curl -X POST http://127.0.0.1:8000/decisions/allocate \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","demands":[{"demandId":"d-a","units":2,"priority":1}],"resources":[{"resourceId":"r-a","capacity":5}]}'
```

- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` — a non-object body, missing or extra fields,
  invalid types, duplicate or blank identifiers, or out-of-range values.

## Reservation inventory

Reservations are held only in the server process: restarting (or starting a
new instance) begins with empty inventory. The first claim naming a resource
fixes that resource's capacity; it can never be rewritten afterwards.

### `POST /reservations`

Requires `Content-Type: application/json`. The body must be a JSON object
with exactly these five fields:

| Field            | Rule                                      |
| ---------------- | ----------------------------------------- |
| `organizationId` | non-empty string                          |
| `reservationId`  | non-empty string                          |
| `resourceId`     | non-empty string                          |
| `quantity`       | positive integer (no floats, no booleans) |
| `capacity`       | positive integer (no floats, no booleans) |

```bash
curl -X POST http://127.0.0.1:8000/reservations \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","reservationId":"rsv-1","resourceId":"res-1","quantity":2,"capacity":5}'
```

The availability check and the deduction are performed atomically, so
concurrent requests can never oversell a resource or overwrite each other.
A resource's `remaining` is its `capacity` minus the total quantity of all
confirmed reservations against it and never goes negative.

Responses are compact JSON with a stable field order, integer values, and a
trailing newline:

- `201 Created` — new reservation confirmed; the body is
  `{"organizationId", "reservationId", "resourceId", "quantity", "capacity",
  "occupied", "remaining"}` where `occupied`/`remaining` are resource-wide
  totals after this reservation.
- `200 OK` — the same `reservationId` was resubmitted with identical fields;
  the stored view is returned and nothing is counted twice.
- `409 Conflict` (`{"error": "reservation_conflict"}`) — the `reservationId`
  exists but the other fields differ.
- `409 Conflict` (`{"error": "capacity_conflict"}`) — the resource was
  already recorded with a different `capacity`; history is not rewritten.
- `409 Conflict` (`{"error": "capacity_exceeded"}`) — `quantity` exceeds the
  resource's remaining inventory; inventory and existing reservations are
  unchanged.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON; no reservation
  is produced.
- `422 Unprocessable Entity` — a non-object body, missing or extra fields,
  blank identifiers, or non-positive-integer `quantity`/`capacity`.

### `GET /reservations?organizationId=...`

The `organizationId` query parameter must appear exactly once and be
non-empty. Returns `200` with `{"organizationId": "...", "reservations":
[...]}` containing only that organization's reservations, sorted by
`resourceId` then `reservationId` in Unicode code-point order. Each entry
carries `quantity`, the resource's `capacity`, and the resource-wide
`occupied` and `remaining` totals. An unknown organization returns
`"reservations": []`. A missing, blank, or duplicated parameter yields `422`.

