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

- `GET /health` returns JSON with the service name and `ok` status, under the keys `service` and `status`: `{"service": "real-world-event-decision-sim", "status": "ok"}`.
- Unknown paths return a JSON `not_found` error with HTTP 404.
- `python -m event_sim --help` documents the command-line entry point.

## Authentication and organization isolation

Every entry point except `GET /health` and `POST /auth/tokens` requires an
`Authorization: Bearer <token>` header naming a registered token. A missing
header, a non-Bearer value, a malformed token, or an unregistered token
returns `401` with `{"error": "unauthorized"}`. Credentials live only in the
server process: a restart clears every token, and an unregistered token
cannot reach any protected entry point.

### `POST /auth/tokens`

Registers a credential. Requires `Content-Type: application/json`. The body
must be a JSON object with exactly these three fields (in any order):

| Field            | Rule                        |
| ---------------- | --------------------------- |
| `token`          | non-empty string            |
| `organizationId` | non-empty string            |
| `role`           | `"read"` or `"write"`       |

- `201 Created` — new credential registered; the response echoes the three
  fields as compact JSON with keys in code-point order and one trailing
  newline.
- `200 OK` — the same token was resubmitted with identical fields; nothing
  is registered twice and the body matches the creation response.
- `409 Conflict` (`{"error": "auth_conflict"}`) — the token exists with
  different fields.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — fields are missing,
  extra, blank, or of the wrong type; nothing is registered.

### Authorization rules

- The `organizationId` in a request (query parameter or body field) must
  match the token's registered organization; cross-organization reads and
  writes return `403` with `{"error": "forbidden"}` and never change state.
- Snapshots and branches belong to the organization of the credential that
  created them. `GET /snapshots` lists only the caller's own snapshots;
  forking another organization's snapshot or reaching another
  organization's branch (including all branch-prefixed entry points)
  returns `403`.
- The `read` role may call every non-mutating entry point: all list,
  aggregate, replay, and region queries, plus the decision computations
  (`POST /decisions/evaluate`, `POST /decisions/allocate`, and their branch
  counterparts).
- Only the `write` role may call state-changing entry points: submitting
  events and reservations (main or branch), evaluating alerts, and creating
  snapshots and branches. A `read` credential calling a write entry point
  returns `403`; the role check and the write commit under the same lock,
  so a failed request never changes the ledger or the inventory.


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

### Region attribution

An event's region is the value of the `region` key in its `payload`. Only a
non-empty string attributes an event to a region: a missing key, an empty
string, or a non-string value means the event has no region. Region values
are matched verbatim (exact comparison, no trimming or normalization), and
regions need no registration — an unknown region simply matches zero events
and is never implicitly created. The region endpoints below are read-only:
they never write the event ledger, reservations, or alerts.

### `GET /events/region?organizationId=...&region=...`

Lists one organization's events attributed to one region. Both
`organizationId` and `region` must appear exactly once and be non-empty; a
missing, duplicated, or blank parameter yields `422`. Returns `200` with
`{"organizationId": "...", "region": "...", "events": [...]}`, where events
are sorted by `occurredAt`, then `eventId`, exactly like `GET /events`. An
unknown organization or region returns `"events": []`; other organizations'
events are never included. The body is compact JSON with keys sorted by code
point and a trailing newline.

```bash
curl 'http://127.0.0.1:8000/events/region?organizationId=org-1&region=north'
```

### `GET /events/region/aggregate?organizationId=...&region=...&type=...&windowSize=...`

Counts per time window, using the exact same parameter rules, window
boundaries, ordering, and `from`/`to` semantics as
`GET /events/aggregate`, with an added `region` parameter that must appear
exactly once and be non-empty. Only events matching the organization,
region, and type are counted; events without a non-empty string payload
`region` never match. The `200` response mirrors the aggregate response with
one added `region` key (compact JSON, keys sorted by code point, trailing
newline):

```json
{
  "organizationId": "org-1",
  "region": "north",
  "type": "incident.created",
  "windowSize": 60,
  "from": null,
  "to": null,
  "windows": [{"start": 0, "end": 60, "count": 2}]
}
```

Without a range, only windows covered by matching events are returned; an
unknown region behaves like a zero-event match. With a range, every window
intersecting `[from, to]` is returned, including empty ones. Any parameter
problem — missing, duplicated, or blank values, a non-positive-integer
`windowSize`, unpaired or non-integer `from`/`to`, or `from > to` — yields
`422` with `{"error": "validation_error", ...}`.

```bash
curl 'http://127.0.0.1:8000/events/region/aggregate?organizationId=org-1&region=north&type=incident.created&windowSize=60&from=0&to=180'
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

Reservations are held only in the server process alongside the event ledger:
restarting (or starting a new instance) begins with empty inventory. The
first reservation that names a resource fixes that resource's capacity; it
is never rewritten afterwards.

### `POST /reservations`

Requires `Content-Type: application/json`. The body must be a JSON object
with exactly these five top-level fields:

| Field           | Rule                                      |
| --------------- | ----------------------------------------- |
| `organizationId` | non-empty string                         |
| `reservationId` | non-empty string                          |
| `resourceId`    | non-empty string                          |
| `quantity`      | positive integer (no floats, no booleans) |
| `capacity`      | positive integer (no floats, no booleans) |

```bash
curl -X POST http://127.0.0.1:8000/reservations \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","reservationId":"res-1","resourceId":"r-a","quantity":2,"capacity":5}'
```

A successful response is compact JSON with stable key order, integer values,
and a trailing newline:

```json
{"capacity":5,"occupied":2,"organizationId":"org-1","quantity":2,"remaining":3,"reservationId":"res-1","resourceId":"r-a"}
```

`occupied` is the total confirmed quantity on the resource and `remaining`
is `capacity - occupied`; the balance can never go negative because the
check and the deduction commit atomically under one lock, so concurrent
requests cannot oversell a resource or lose each other's writes.

- `201 Created` — new reservation committed; the first reservation for a
  resource also establishes its capacity.
- `200 OK` — the same `reservationId` was resubmitted with identical fields;
  nothing is counted twice and the balances are unchanged.
- `409 Conflict` (`{"error": "reservation_conflict"}`) — the `reservationId`
  exists but the other fields differ.
- `409 Conflict` (`{"error": "capacity_conflict"}`) — the resource already
  has a recorded capacity and the declared `capacity` differs; the recorded
  capacity is never rewritten.
- `409 Conflict` (`{"error": "capacity_exceeded"}`) — the quantity exceeds
  the resource's remaining balance; inventory and existing reservations are
  unchanged.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON; no reservation
  is produced.
- `422 Unprocessable Entity` — a non-object body, missing or extra fields,
  blank or non-string identifiers, or non-positive/non-integer values.

### `GET /reservations?organizationId=...`

The `organizationId` query parameter must appear exactly once and be
non-empty. Returns `200` with `{"organizationId": "...", "reservations":
[...]}` containing only that organization's reservations, sorted by
`resourceId` then `reservationId` in Unicode code-point order. Each entry
carries `quantity`, the resource's `capacity`, and the resource-wide
`occupied` and `remaining` balances. An unknown organization returns
`"reservations": []`. A missing, blank, or duplicated parameter yields
`422`.

## Alert suppression and escalation

Alerts are held only in the main service process alongside the event
ledger; restarting (or starting a new instance) begins with no alerts.
Snapshots and branches never capture alerts, and the alert entry points
exist only on the main service — no `/branches/{branchId}/alerts/...`
paths are added.

### `POST /alerts/evaluate`

Requires `Content-Type: application/json`. The body must be a JSON object
containing exactly these fields:

| Field               | Rule                                                          |
| ------------------- | ------------------------------------------------------------ |
| `organizationId`    | required, non-empty string                                   |
| `type`              | required, non-empty string                                   |
| `windowSize`        | required, positive integer (no booleans, floats, or strings) |
| `threshold`         | required, positive integer (no booleans, floats, or strings) |
| `suppressionWindow` | required, positive integer (no booleans, floats, or strings) |
| `from`              | optional; non-negative integer, present only together with `to` |
| `to`                | optional; non-negative integer, present only together with `from` |

The peak uses the exact same windowing as `POST /decisions/evaluate`: only
events matching both `organizationId` and `type` are counted, windows
start at `0` and cover `[start, start + windowSize)`, ties resolve to the
earliest window start, and an unknown organization or type computes as
zero events. The event ledger is never modified.

- When the peak is below the threshold, `action` is `"observe"`, no alert
  is written, `alertId` is `null`, and `suppressedCount` is `null`.
- When the peak reaches the threshold, the peak start is compared with the
  most recent prior alert for the same organization and type:
  - `peakStart` not earlier than the prior start and at least
    `suppressionWindow` away (`peakStart - priorStart >=
    suppressionWindow`) creates a new alert; `action` is `"escalate"`,
    `alertId` is the new id (`alert-1`, `alert-2`, …, one global sequence
    across all organizations and types), and `suppressedCount` is `0`.
  - A peak start less than `suppressionWindow` after the prior alert's
    peak start increments that alert's suppression count instead; `action`
    is `"suppress"`, `alertId` is the prior alert's id, and
    `suppressedCount` is the new total. The first threshold hit for an
    organization/type always creates an alert.

The suppression decision and the alert creation commit atomically under
one lock, so concurrent threshold hits cannot both open alerts or suppress
against a stale view. The `200` response echoes every request parameter
and adds the result fields (compact JSON, stable key order, integer values,
trailing newline):

```json
{"action":"escalate","alertId":"alert-1","from":null,"organizationId":"org-1","peakCount":3,"peakStart":60,"suppressedCount":0,"suppressionWindow":120,"threshold":3,"to":null,"type":"incident.created","windowSize":60}
```

```bash
curl -X POST http://127.0.0.1:8000/alerts/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","type":"incident.created","windowSize":60,"threshold":3,"suppressionWindow":120}'
```

- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON; no alert is
  written.
- `422 Unprocessable Entity` — a non-object body, missing or extra fields,
  blank or non-string identifiers, booleans/floats where integers are
  required, non-positive values, unpaired `from`/`to`, or `from > to`.

### `GET /alerts?organizationId=...`

The `organizationId` query parameter must appear exactly once and be
non-empty. Returns `200` with `{"organizationId": "...", "alerts": [...]}`
containing only that organization's alerts. Each entry carries `alertId`,
`type`, `peakStart`, `threshold`, and `suppressedCount`, sorted by
`peakStart` ascending and then `alertId` in Unicode code-point order. An
unknown organization returns `"alerts": []`. A missing, blank, or
duplicated parameter yields `422`.

## Snapshots and branches

Snapshots capture the current main service state — every event, every
recorded resource capacity, and every reservation — under a unique name so
that decisions can later be recomputed against that exact state. Branches fork
a snapshot into an isolated copy that supports the existing event, aggregate,
decision, and reservation entry points without sharing any mutable state with
the main service or with other branches. Snapshots and branches capture only
events, resource capacities, and reservations — alerts are never captured, and
the alert entry points remain main-only with no branch-prefixed sub-paths.
Like everything else here, snapshots and branches live only in the server
process and are cleared on restart; there is no persistence or replay across
restarts.

### `POST /snapshots`

Requires `Content-Type: application/json`. The body must be a JSON object
containing exactly one field:

| Field        | Rule                     |
| ------------ | ------------------------ |
| `snapshotId` | non-empty string         |

A successful response is `201` with compact integer JSON:

```json
{"events":2,"reservations":1,"resources":1,"snapshotId":"snap-1"}
```

- `events`, `resources`, and `reservations` count the captured events,
  distinct resources with a recorded capacity, and reservations.
- `409 Conflict` (`{"error": "snapshot_conflict"}`) — the name already
  exists; the original snapshot content is never replaced or mutated.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` — a non-object body, a missing or extra field,
  a blank value, or a non-string `snapshotId`.
- Any failed request leaves both the main state and existing snapshots
  unchanged.

### `GET /snapshots`

Returns `200` with `{"snapshots": [...]}`, where each entry has the same
shape as a creation response. Entries are sorted by `snapshotId` in Unicode
code-point order. With no snapshots the collection is an empty array.

### `POST /branches`

Requires `Content-Type: application/json`. The body must be a JSON object
containing exactly these fields:

| Field        | Rule                     |
| ------------ | ------------------------ |
| `branchId`   | non-empty string         |
| `snapshotId` | non-empty string         |

The new branch receives independent copies of the snapshot's events,
resource capacities, and reservation balances.

- `201 Created` — the branch summary is returned:

```json
{"branchId":"br-1","events":2,"reservations":1,"resources":1,"snapshotId":"snap-1"}
```

- `404 Not Found` (`{"error": "snapshot_not_found"}`) — no snapshot has that
  name; no branch is created.
- `409 Conflict` (`{"error": "branch_conflict"}`) — a branch with that
  `branchId` already exists.
- `415`, `400`, and `422` follow the same rules as other JSON write
  endpoints; `422` (`validation_error`) covers missing, extra, blank, or
  non-string fields.

### `GET /branches/{branchId}`

Returns `200` with the branch summary (see above). An unknown branch returns
`404` with `{"error": "branch_not_found"}`.

### Branch-prefixed entry points

A known branch exposes the existing endpoints under
`/branches/{branchId}/...`, with identical request contracts, query
validation, ordering, and window semantics:

| Method | Path                                      | Effect                                    |
| ------ | ----------------------------------------- | ----------------------------------------- |
| GET    | `/branches/{branchId}/events`             | list the branch's events                  |
| GET    | `/branches/{branchId}/events/aggregate`   | aggregate the branch's events             |
| POST   | `/branches/{branchId}/events`             | commit an event into the branch only      |
| POST   | `/branches/{branchId}/decisions/evaluate` | evaluate against the branch's events only |
| GET    | `/branches/{branchId}/reservations`       | list the branch's reservations            |
| POST   | `/branches/{branchId}/reservations`       | reserve against branch balances only      |

- Branch event commits honor `415`, `400`, `422`, identical-replay (`200`),
  and `event_id_conflict` (`409`); writes stay in the branch.
- Branch reservation commits honor `415`, `400`, `422`, and the replay and
  capacity rules: identical replays return `200` without double counting,
  while `reservation_conflict`, `capacity_conflict`, and
  `capacity_exceeded` still return `409`; a failed commit never changes the
  branch inventory.
- Branch list and aggregate queries apply the same parameter rules as the
  main endpoints; missing, duplicated, or empty parameters return `422` with
  `validation_error`.
- Branch evaluation is read-only over branch events, and branch aggregates
  never count main-service events; repeated identical requests return
  identical results.
- No branch operation creates or rolls back main-service state, and branches
  never observe each other's writes. Unknown sub-paths under a known branch
  return the standard JSON `404`; sub-paths under an unknown branch return
  `branch_not_found`.
- The region queries (`/events/region` and `/events/region/aggregate`) are
  main-only; no branch-prefixed region paths are added.
- `/decisions/allocate` remains a main-only, stateless endpoint.

