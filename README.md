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

## Credentials and organization isolation

Every entry point except `GET /health` and the credential registration entry
point requires a request-scoped Bearer credential. Credentials are held only
in the server process: restarting (or starting a new instance) begins with an
empty credential registry, and an unregistered token reaches nothing.

### `POST /auth/tokens`

The one entry point that requires no credential. Requires
`Content-Type: application/json`. The body must be a JSON object with exactly
these three fields, in any order:

| Field            | Rule                                  |
| ---------------- | ------------------------------------- |
| `token`          | non-empty string                      |
| `organizationId` | non-empty string                      |
| `role`           | one of `read` or `write`              |

- `201 Created` — new token registered. The body echoes the three fields as
  compact JSON with keys sorted by code point and one trailing newline:
  `{"organizationId":"org-1","role":"write","token":"tok-1"}\n`.
- `200 OK` — the same token was resubmitted with the identical organization
  and role; it is not registered a second time and the same body is echoed.
- `409 Conflict` (`{"error": "auth_conflict"}`) — the token is already bound
  to a different organization or role. The binding is never rewritten.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — a non-object body, a
  missing/extra field, a blank or non-string `token`/`organizationId`, or a
  `role` other than `read`/`write`. No record is stored.

```bash
curl -X POST http://127.0.0.1:8000/auth/tokens \
  -H 'Content-Type: application/json' \
  -d '{"token":"tok-1","organizationId":"org-1","role":"write"}'
```

### Bearer authentication

All other requests carry the registered token in an
`Authorization: Bearer <token>` header. A missing header, a non-Bearer
scheme, a malformed bearer value, or an unregistered token yields
`401 Unauthorized` with `{"error": "unauthorized"}`.

A valid credential that acts outside its bound organization or role yields
`403 Forbidden` with `{"error": "forbidden"}` and performs no write. The
`organizationId` in any request parameter or body field must equal the
token's registered organization; reading or writing another organization's
data is forbidden.

- A `read` token may call only the state-free query and decision-computation
  entry points: the event list, aggregate, region, and replay (including
  replay comparison) queries, the reservation/alert listings,
  `POST /decisions/evaluate`, `POST /decisions/allocate`,
  `POST /branches/compare`, `POST /branches/compare/events`,
  `POST /branches/compare/reservations`, `POST /snapshots/compare`,
  `POST /snapshots/compare/events`,
  `POST /snapshots/compare/reservations`,
  `POST /snapshots/compare/resources`, and the other
  read-only branch entry points.
- Committing an event or reservation, evaluating an alert, and creating a
  snapshot or branch are writes; only a `write` token may call them. A
  `read` token against a write entry point receives `403`. The authorization
  decision and the ledger/inventory mutation complete under one lock, so a
  rejected request never changes the ledger or inventory.

Snapshots and branches belong to the organization that created them. A
snapshot captures only that organization's events, resource capacities, and
reservations, and the three summary counts count only that organization; a
branch forks only a snapshot owned by the same organization, and branch entry
points authenticate with the same rules as the main service. Snapshot and
branch names are unique across the whole service (not merely within one
organization), and accessing another organization's snapshot or branch by name
is `403`, never `404`.

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

Snapshots capture the current main service state under a unique name so that
decisions can later be recomputed against that exact state. Branches fork a
snapshot into an isolated copy that supports the existing event, aggregate,
decision, and reservation entry points without sharing any mutable state with
the main service or with other branches.

### Organization boundaries

Every snapshot and branch is owned by the organization of the `write`
credential that created it, and all capture, naming, and visibility rules are
scoped to that boundary:

- A snapshot captures **only the creator's own organization** — its events,
  the capacities of resources it reserves, and its reservations. Other
  organizations' events, reservation identifiers, and capacity records are
  neither captured nor summarized. The three counts in the response
  (`events`, `resources`, `reservations`) count only that organization. Two
  snapshots created from the same state return identical counts; another
  organization's data never changes them.
- A branch forked from a snapshot carries only that snapshot organization's
  data. Inside the branch, event and reservation names are compared only
  against that organization's own records. Another organization's event
  identifiers, reservation identifiers, and capacity records never cause a
  branch commit to conflict and never affect a branch balance.
- Snapshot and branch names are **unique across the whole service** and are
  owned by the creator's organization. A duplicate name conflicts regardless
  of which organization holds it: creating a snapshot with a name already in
  use returns `409` `snapshot_conflict` (the original snapshot is never
  rewritten), and creating a branch with a name already in use returns `409`
  `branch_conflict` (the existing branch is untouched). Names are not
  namespaced per organization.
- Using another organization's **snapshot name as a fork source**, or reading
  another organization's **branch by name**, returns `403 Forbidden`
  (`{"error": "forbidden"}`) with no content, even though the name exists. A
  name that has never appeared anywhere is the only case that yields `404`:
  a missing snapshot is `snapshot_not_found` and a missing branch is
  `branch_not_found`. Missing objects are never implicitly created.
- When creating a branch, snapshot ownership (`403`) is decided before the
  duplicate branch name (`409`); a foreign snapshot is forbidden even if the
  requested `branchId` is also taken. The snapshot list (`GET /snapshots`)
  and branch summaries are visible only within the owning organization.
- Authorization and the write happen under one lock for alert evaluation,
  snapshot creation, and branch creation alike. A rejected request leaves no
  alert, snapshot, or branch behind, and concurrent creations of the same
  name have exactly one winner; the loser receives the conflict response.

Snapshots and branches capture only events, resource capacities, and
reservations — alerts are never captured, and the alert entry points remain
main-only with no branch-prefixed sub-paths. Like everything else here,
snapshots and branches live only in the server process and are cleared on
restart; there is no persistence, authorization, or replay across restarts.

### `POST /snapshots`

Requires a `write` credential and `Content-Type: application/json`. The body
must be a JSON object containing exactly one field:

| Field        | Rule                     |
| ------------ | ------------------------ |
| `snapshotId` | non-empty string         |

The snapshot is owned by the caller's organization and captures only that
organization's state. A successful response is `201` with compact integer
JSON:

```json
{"events":2,"reservations":1,"resources":1,"snapshotId":"snap-1"}
```

- `events`, `resources`, and `reservations` count the creator organization's
  captured events, distinct resources with a recorded capacity, and
  reservations; other organizations' data is never counted.
- `409 Conflict` (`{"error": "snapshot_conflict"}`) — the name already exists
  anywhere in the service (held by any organization); the original snapshot
  content is never replaced or mutated.
- `403 Forbidden` — the credential is read-only or acts outside its
  organization; no snapshot is stored.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` — a non-object body, a missing or extra field,
  a blank value, or a non-string `snapshotId`.
- Any failed request leaves both the main state and existing snapshots
  unchanged.

### `GET /snapshots`

Returns `200` with `{"snapshots": [...]}`, listing **only the caller
organization's** snapshots; other organizations' snapshots are never visible.
Each entry has the same shape as a creation response, and entries are sorted
by `snapshotId` in Unicode code-point order. With no snapshots the collection
is an empty array.

### `POST /branches`

Requires a `write` credential and `Content-Type: application/json`. The body
must be a JSON object containing exactly these fields:

| Field        | Rule                     |
| ------------ | ------------------------ |
| `branchId`   | non-empty string         |
| `snapshotId` | non-empty string         |

The new branch receives independent copies of the snapshot's events,
resource capacities, and reservation balances (which hold only the snapshot
owner organization's data).

- `201 Created` — the branch summary is returned:

```json
{"branchId":"br-1","events":2,"reservations":1,"resources":1,"snapshotId":"snap-1"}
```

- `403 Forbidden` (`{"error": "forbidden"}`) — the named snapshot exists but
  belongs to another organization (ownership is checked before the branch
  name), or the credential is read-only. No branch is created and no snapshot
  content is returned.
- `404 Not Found` (`{"error": "snapshot_not_found"}`) — no snapshot with that
  name has ever existed; no branch is created.
- `409 Conflict` (`{"error": "branch_conflict"}`) — the snapshot is owned by
  the caller but a branch with that `branchId` already exists anywhere in the
  service. The existing branch is unaffected.
- `415`, `400`, and `422` follow the same rules as other JSON write
  endpoints; `422` (`validation_error`) covers missing, extra, blank, or
  non-string fields.

### `GET /branches/{branchId}`

Returns `200` with the branch summary (see above) when the branch belongs to
the caller's organization. A branch owned by another organization returns
`403` with `{"error": "forbidden"}` and no content. Only a branch name that
has never existed returns `404` with `{"error": "branch_not_found"}`.

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
  and `event_id_conflict` (`409`); writes stay in the branch. Name and
  capacity comparisons run only against the owning organization's records
  that were forked into the branch, so another organization's event
  identifiers, reservation identifiers, and capacity records never trigger a
  conflict and never affect a branch balance.
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

### `POST /branches/compare`

A read-only comparison of two existing branches' window counts and peak
decisions. Each side recomputes its window counts and peak from that
branch's own events only; nothing in either branch, in any other branch, or
in the main service is read for mutation or written, and identical
submissions return byte-for-byte identical JSON. Both `read` and `write`
credentials may call it. Requires `Content-Type: application/json` and a
Bearer credential bound to the request's `organizationId`. The body must be
a JSON object containing exactly these fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string; both branches must belong to it |
| `left`           | required, non-empty branch name (the left-hand side)         |
| `right`          | required, non-empty branch name (the right-hand side)        |
| `type`           | required, non-empty string                                   |
| `windowSize`     | required, positive integer (no booleans, floats, or strings) |
| `threshold`      | required, positive integer (no booleans, floats, or strings) |
| `from`           | optional; non-negative integer, present only together with `to` |
| `to`             | optional; non-negative integer, present only together with `from` |

Using the same branch name for `left` and `right` is legal; the two sides
are then equal by construction. Window division matches the aggregate and
decision entry points exactly: windows start at `0` and cover
`[start, start + windowSize)`, and an event counts toward the window of
`floor(occurredAt / windowSize) * windowSize`.

- Without `from`/`to`, the `windows` rows cover the **union** of the windows
  hit by matching events on either side, aligned row-by-row by `start`
  ascending. A window hit by only one side shows the other side's count as
  `0`. When neither side has a matching event, `windows` is `[]`.
- With `from`/`to`, only events in the closed interval
  `from <= occurredAt <= to` count, and **every** window intersecting that
  interval gets a row, including windows empty on both sides.
- Each row is `{"start", "leftCount", "rightCount", "equal"}`, with
  `equal` true exactly when the two counts agree.
- `decision.left` and `decision.right` each carry `peakStart`,
  `peakCount`, and `action`: the peak is the largest window count, ties
  resolve to the earliest window start, and `action` is `"escalate"` when
  the peak reaches `threshold` and `"observe"` otherwise. `decision.equal`
  is true exactly when the two sides' peak results agree. An empty side has
  `peakStart: null`, `peakCount: 0`, and `action: "observe"`.

The `200` response is compact JSON with keys sorted by code point, integer
values kept as integers, booleans kept as booleans, and one trailing
newline:

```json
{"decision":{"equal":false,"left":{"action":"observe","peakCount":2,"peakStart":0},"right":{"action":"escalate","peakCount":3,"peakStart":60}},"from":null,"left":"br-a","organizationId":"org-1","right":"br-b","threshold":3,"to":null,"type":"incident.created","windowSize":60,"windows":[{"equal":false,"leftCount":2,"rightCount":1,"start":0},{"equal":false,"leftCount":0,"rightCount":3,"start":60}]}
```

```bash
curl -X POST http://127.0.0.1:8000/branches/compare \
  -H 'Authorization: Bearer tok-1' \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","left":"br-a","right":"br-b","type":"incident.created","windowSize":60,"threshold":3}'
```

- `401 Unauthorized` — the Bearer credential is missing, malformed, or not
  registered.
- `403 Forbidden` (`{"error": "forbidden"}`) — the credential is bound to a
  different organization, or one of the named branches belongs to another
  organization. The organization decision happens before branch names are
  inspected, and branches are checked in the fixed order `left` then
  `right` (a missing left outranks any problem on the right).
- `404 Not Found` (`{"error": "branch_not_found"}`) — either `left` or
  `right` has never existed as a branch. Both participating branches must
  already exist; a comparison never implicitly creates a branch.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — a non-object body, a
  missing or extra field, a blank or non-string `organizationId`/`left`/
  `right`/`type`, non-positive-integer `windowSize`/`threshold`, unpaired or
  non-integer `from`/`to`, or `from > to`.

Every non-`200` result is read-only as well: a failed comparison creates no
branch and changes no event, reservation, or decision state.

### `POST /branches/compare/events`

A read-only, event-level comparison of two existing branches. Where
`POST /branches/compare` aligns window counts, this entry point aligns the
two branches' events by `eventId` and reports the deterministic difference
summary. Nothing in either branch, in any other branch, or in the main
service is read for mutation or written, and identical submissions return
byte-for-byte identical JSON. Both `read` and `write` credentials may call
it. Requires `Content-Type: application/json` and a Bearer credential bound
to the request's `organizationId`. The body must be a JSON object
containing exactly these fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string; both branches must belong to it |
| `left`           | required, non-empty branch name (the left-hand side)         |
| `right`          | required, non-empty branch name (the right-hand side)        |

Using the same branch name for `left` and `right` is legal; the two sides
are then equal by construction and every shared event lands in `same`.

Events are aligned by `eventId`:

- `leftOnly` — identifiers that appear only in the left branch.
- `rightOnly` — identifiers that appear only in the right branch.
- `same` — identifiers on both sides whose events are identical in every
  field except `eventId`; payloads compare by content, so key order inside
  a payload never matters.
- `diff` — identifiers on both sides whose events disagree. Each entry is
  `{"eventId", "fields"}`, where `fields` names the mismatched fields
  (drawn from `organizationId`, `type`, `occurredAt`, `payload`).

Identifiers and field names are sorted in Unicode code-point order. Each
group array is accompanied by a count key named after the group plus
`Count` (`leftOnlyCount`, `rightOnlyCount`, `sameCount`, `diffCount`). When
neither branch holds any event, all four groups are empty arrays and all
four counts are `0`.

The `200` response is compact JSON with keys sorted by code point, integer
values kept as integers, and one trailing newline:

```json
{"diff":[{"eventId":"evt-2","fields":["occurredAt","payload"]}],"diffCount":1,"leftOnly":["evt-1"],"leftOnlyCount":1,"rightOnly":["evt-4"],"rightOnlyCount":1,"same":["evt-3"],"sameCount":1}
```

```bash
curl -X POST http://127.0.0.1:8000/branches/compare/events \
  -H 'Authorization: Bearer tok-1' \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","left":"br-a","right":"br-b"}'
```

- `401 Unauthorized` — the Bearer credential is missing, malformed, or not
  registered.
- `403 Forbidden` (`{"error": "forbidden"}`) — the credential is bound to a
  different organization, or one of the named branches belongs to another
  organization. The organization decision happens before branch names are
  inspected, and branches are checked in the fixed order `left` then
  `right` (a missing left outranks any problem on the right).
- `404 Not Found` (`{"error": "branch_not_found"}`) — either `left` or
  `right` has never existed as a branch. Both participating branches must
  already exist; a comparison never implicitly creates a branch.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — a non-object body, a
  missing or extra field, or a blank or non-string
  `organizationId`/`left`/`right`.

Every non-`200` result is read-only as well: a failed comparison creates no
branch and changes no event, reservation, alert, or main-service state.

### `POST /branches/compare/reservations`

A read-only, reservation-level comparison of two existing branches. Where
`POST /branches/compare/events` aligns events by `eventId`, this entry point
aligns the two branches' reservations by `reservationId` and reports the
deterministic difference summary. Nothing in either branch, in any other
branch, or in the main service is read for mutation or written, alert state
is untouched, and identical submissions return byte-for-byte identical JSON.
Both `read` and `write` credentials may call it. Requires
`Content-Type: application/json` and a Bearer credential bound to the
request's `organizationId`. The body must be a JSON object containing
exactly these fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string; both branches must belong to it |
| `left`           | required, non-empty branch name (the left-hand side)         |
| `right`          | required, non-empty branch name (the right-hand side)        |

Using the same branch name for `left` and `right` is legal; the two sides
are then equal by construction and every shared reservation lands in
`same`.

Reservations are aligned by `reservationId`:

- `leftOnly` — identifiers that appear only in the left branch.
- `rightOnly` — identifiers that appear only in the right branch.
- `same` — identifiers on both sides whose `organizationId`, `resourceId`,
  `quantity`, and `capacity` all match.
- `diff` — identifiers on both sides that disagree in at least one of those
  four fields. Each entry is `{"reservationId", "fields"}`, where `fields`
  names the mismatched fields, drawn only from `organizationId`,
  `resourceId`, `quantity`, and `capacity`; no other content participates
  in the comparison.

Identifiers and field names are sorted in Unicode code-point order. Each
group array is accompanied by a count key named after the group plus
`Count` (`leftOnlyCount`, `rightOnlyCount`, `sameCount`, `diffCount`). When
neither branch holds any reservation for the organization, all four groups
are empty arrays and all four counts are `0`.

The `200` response is compact JSON with keys sorted by code point, integer
values kept as integers, and one trailing newline:

```json
{"diff":[{"fields":["capacity","quantity"],"reservationId":"res-2"}],"diffCount":1,"leftOnly":["res-1"],"leftOnlyCount":1,"rightOnly":["res-4"],"rightOnlyCount":1,"same":["res-3"],"sameCount":1}
```

```bash
curl -X POST http://127.0.0.1:8000/branches/compare/reservations \
  -H 'Authorization: Bearer tok-1' \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","left":"br-a","right":"br-b"}'
```

- `401 Unauthorized` — the Bearer credential is missing, malformed, or not
  registered.
- `403 Forbidden` (`{"error": "forbidden"}`) — the credential is bound to a
  different organization, or one of the named branches belongs to another
  organization. The organization decision happens before branch names are
  inspected, and branches are checked in the fixed order `left` then
  `right` (a missing left outranks any problem on the right).
- `404 Not Found` (`{"error": "branch_not_found"}`) — either `left` or
  `right` has never existed as a branch. Both participating branches must
  already exist; a comparison never implicitly creates a branch.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — a non-object body, a
  missing or extra field, or a blank or non-string
  `organizationId`/`left`/`right`.

Every non-`200` result is read-only as well: a failed comparison creates no
branch and changes no event, reservation, alert, or main-service state.

### `POST /snapshots/compare`

A read-only comparison of two existing snapshots' window counts and peak
decisions. Where `POST /branches/compare` recomputes over two branches'
live events, this entry point recomputes over the events the two snapshots
captured at their own creation times; nothing in either snapshot, in any
branch, or in the main service is read for mutation or written — snapshots
are immutable, the main-service event ledger, reservation inventory, and
alert state are untouched, and identical submissions return byte-for-byte
identical JSON. Both `read` and `write` credentials may call it. Requires
`Content-Type: application/json` and a Bearer credential bound to the
request's `organizationId`. The body must be a JSON object containing
exactly these fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string; both snapshots must belong to it |
| `left`           | required, non-empty snapshot name (the left-hand side)      |
| `right`          | required, non-empty snapshot name (the right-hand side)     |
| `type`           | required, non-empty string                                   |
| `windowSize`     | required, positive integer (no booleans, floats, or strings) |
| `threshold`      | required, positive integer (no booleans, floats, or strings) |
| `from`           | optional; non-negative integer, present only together with `to` |
| `to`             | optional; non-negative integer, present only together with `from` |

Using the same snapshot name for `left` and `right` is legal; the two sides
are then equal by construction. Window division matches the aggregate and
decision entry points exactly: windows start at `0` and cover
`[start, start + windowSize)`, and an event counts toward the window of
`floor(occurredAt / windowSize) * windowSize`.

- Without `from`/`to`, the `windows` rows cover the **union** of the windows
  hit by matching captured events on either side, aligned row-by-row by
  `start` ascending. A window hit by only one side shows the other side's
  count as `0`. When neither side has a matching event, `windows` is `[]`.
- With `from`/`to`, only events in the closed interval
  `from <= occurredAt <= to` count, and **every** window intersecting that
  interval gets a row, including windows empty on both sides.
- Each row is `{"start", "leftCount", "rightCount", "equal"}`, with
  `equal` true exactly when the two counts agree.
- `decision.left` and `decision.right` each carry `peakStart`,
  `peakCount`, and `action`: the peak is the largest window count, ties
  resolve to the earliest window start, and `action` is `"escalate"` when
  the peak reaches `threshold` and `"observe"` otherwise. `decision.equal`
  is true exactly when the two sides' peak results agree. An empty side has
  `peakStart: null`, `peakCount: 0`, and `action: "observe"`.

The `200` response is compact JSON with keys sorted by code point, integer
values kept as integers, booleans kept as booleans, and one trailing
newline:

```json
{"decision":{"equal":false,"left":{"action":"observe","peakCount":2,"peakStart":0},"right":{"action":"escalate","peakCount":3,"peakStart":60}},"from":null,"left":"snap-a","organizationId":"org-1","right":"snap-b","threshold":3,"to":null,"type":"incident.created","windowSize":60,"windows":[{"equal":false,"leftCount":2,"rightCount":1,"start":0},{"equal":false,"leftCount":0,"rightCount":3,"start":60}]}
```

```bash
curl -X POST http://127.0.0.1:8000/snapshots/compare \
  -H 'Authorization: Bearer tok-1' \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","left":"snap-a","right":"snap-b","type":"incident.created","windowSize":60,"threshold":3}'
```

- `401 Unauthorized` — the Bearer credential is missing, malformed, or not
  registered.
- `403 Forbidden` (`{"error": "forbidden"}`) — the credential is bound to a
  different organization, or one of the named snapshots belongs to another
  organization. The organization decision happens before snapshot names are
  inspected, and snapshots are checked in the fixed order `left` then
  `right` (a missing left outranks any problem on the right).
- `404 Not Found` (`{"error": "snapshot_not_found"}`) — either `left` or
  `right` has never existed as a snapshot. Both participating snapshots
  must already exist; a comparison never implicitly creates a snapshot.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — a non-object body, a
  missing or extra field, a blank or non-string `organizationId`/`left`/
  `right`/`type`, non-positive-integer `windowSize`/`threshold`, unpaired or
  non-integer `from`/`to`, or `from > to`.

Every non-`200` result is read-only as well: a failed comparison creates no
snapshot and changes no event, reservation, alert, or main-service state.

### `POST /snapshots/compare/events`

A read-only, event-level comparison of two existing snapshots. Where the
branch comparison aligns two branches' live events, this entry point aligns
the events the two snapshots captured at their own creation times, by
`eventId`, and reports the deterministic difference summary. Nothing in
either snapshot, in any branch, or in the main service is read for mutation
or written — snapshots are immutable, the main-service event ledger,
reservation inventory, and alert state are untouched, and identical
submissions return byte-for-byte identical JSON. Both `read` and `write`
credentials may call it. Requires `Content-Type: application/json` and a
Bearer credential bound to the request's `organizationId`. The body must be
a JSON object containing exactly these fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string; both snapshots must belong to it |
| `left`           | required, non-empty snapshot name (the left-hand side)      |
| `right`          | required, non-empty snapshot name (the right-hand side)     |

Using the same snapshot name for `left` and `right` is legal; the two sides
are then the same capture and every shared event lands in `same`.

Events are aligned by `eventId`:

- `leftOnly` — identifiers that appear only in the left snapshot.
- `rightOnly` — identifiers that appear only in the right snapshot.
- `same` — identifiers on both sides whose events are identical in every
  field except `eventId`; payloads compare by content, so key order inside
  a payload never matters.
- `diff` — identifiers on both sides whose events disagree. Each entry is
  `{"eventId", "fields"}`, where `fields` names the mismatched fields
  (drawn from `organizationId`, `type`, `occurredAt`, `payload`).

Identifiers and field names are sorted in Unicode code-point order. Each
group array is accompanied by a count key named after the group plus
`Count` (`leftOnlyCount`, `rightOnlyCount`, `sameCount`, `diffCount`). When
neither snapshot holds any event, all four groups are empty arrays and all
four counts are `0`.

The `200` response is compact JSON with keys sorted by code point, integer
values kept as integers, and one trailing newline:

```json
{"diff":[{"eventId":"evt-2","fields":["occurredAt","payload"]}],"diffCount":1,"leftOnly":["evt-1"],"leftOnlyCount":1,"rightOnly":["evt-4"],"rightOnlyCount":1,"same":["evt-3"],"sameCount":1}
```

```bash
curl -X POST http://127.0.0.1:8000/snapshots/compare/events \
  -H 'Authorization: Bearer tok-1' \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","left":"snap-a","right":"snap-b"}'
```

- `401 Unauthorized` — the Bearer credential is missing, malformed, or not
  registered.
- `403 Forbidden` (`{"error": "forbidden"}`) — the credential is bound to a
  different organization, or one of the named snapshots belongs to another
  organization. The organization decision happens before snapshot names are
  inspected, and snapshots are checked in the fixed order `left` then
  `right` (a missing left outranks any problem on the right).
- `404 Not Found` (`{"error": "snapshot_not_found"}`) — either `left` or
  `right` has never existed as a snapshot. Both participating snapshots
  must already exist; a comparison never implicitly creates a snapshot.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — a non-object body, a
  missing or extra field, or a blank or non-string
  `organizationId`/`left`/`right`.

Every non-`200` result is read-only as well: a failed comparison creates no
snapshot and changes no event, reservation, alert, or main-service state.

### `POST /snapshots/compare/reservations`

A read-only, reservation-level comparison of two existing snapshots. Where
the branch comparison aligns two branches' live reservations, this entry
point aligns the reservations the two snapshots captured at their own
creation times, by `reservationId`, and reports the deterministic
difference summary. Nothing in either snapshot, in any branch, or in the
main service is read for mutation or written — snapshots are immutable,
alert state is untouched, and identical submissions return byte-for-byte
identical JSON. Both `read` and `write` credentials may call it. Requires
`Content-Type: application/json` and a Bearer credential bound to the
request's `organizationId`. The body must be a JSON object containing
exactly these fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string; both snapshots must belong to it |
| `left`           | required, non-empty snapshot name (the left-hand side)      |
| `right`          | required, non-empty snapshot name (the right-hand side)     |

Using the same snapshot name for `left` and `right` is legal; the two sides
are then the same capture and every shared reservation lands in `same`.

Reservations are aligned by `reservationId`:

- `leftOnly` — identifiers that appear only in the left snapshot.
- `rightOnly` — identifiers that appear only in the right snapshot.
- `same` — identifiers on both sides whose `organizationId`, `resourceId`,
  `quantity`, and `capacity` all match.
- `diff` — identifiers on both sides that disagree in at least one of those
  four fields. Each entry is `{"reservationId", "fields"}`, where `fields`
  names the mismatched fields, drawn only from `organizationId`,
  `resourceId`, `quantity`, and `capacity`; no other content participates
  in the comparison.

Identifiers and field names are sorted in Unicode code-point order. Each
group array is accompanied by a count key named after the group plus
`Count` (`leftOnlyCount`, `rightOnlyCount`, `sameCount`, `diffCount`). When
neither snapshot holds any reservation for the organization, all four
groups are empty arrays and all four counts are `0`.

The `200` response is compact JSON with keys sorted by code point, integer
values kept as integers, and one trailing newline:

```json
{"diff":[{"fields":["capacity","quantity"],"reservationId":"res-2"}],"diffCount":1,"leftOnly":["res-1"],"leftOnlyCount":1,"rightOnly":["res-4"],"rightOnlyCount":1,"same":["res-3"],"sameCount":1}
```

```bash
curl -X POST http://127.0.0.1:8000/snapshots/compare/reservations \
  -H 'Authorization: Bearer tok-1' \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","left":"snap-a","right":"snap-b"}'
```

- `401 Unauthorized` — the Bearer credential is missing, malformed, or not
  registered.
- `403 Forbidden` (`{"error": "forbidden"}`) — the credential is bound to a
  different organization, or one of the named snapshots belongs to another
  organization. The organization decision happens before snapshot names are
  inspected, and snapshots are checked in the fixed order `left` then
  `right` (a missing left outranks any problem on the right).
- `404 Not Found` (`{"error": "snapshot_not_found"}`) — either `left` or
  `right` has never existed as a snapshot. Both participating snapshots
  must already exist; a comparison never implicitly creates a snapshot.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — a non-object body, a
  missing or extra field, or a blank or non-string
  `organizationId`/`left`/`right`.

Every non-`200` result is read-only as well: a failed comparison creates no
snapshot and changes no event, reservation, alert, or main-service state.

### `POST /snapshots/compare/resources`

A read-only, resource-capacity comparison of two existing snapshots. Where
`POST /snapshots/compare/reservations` aligns captured reservations by
`reservationId`, this entry point aligns the resources the two snapshots
captured at their own creation times, by `resourceId`, and compares the two
sides' three resource balances — `capacity`, `occupied`, and `remaining`.
Nothing in either snapshot, in any branch, or in the main service is read
for mutation or written — snapshots are immutable, alert state is untouched,
and identical submissions return byte-for-byte identical JSON. Both `read`
and `write` credentials may call it. Requires `Content-Type:
application/json` and a Bearer credential bound to the request's
`organizationId`. The body must be a JSON object containing exactly these
fields:

| Field            | Rule                                                          |
| ---------------- | ------------------------------------------------------------ |
| `organizationId` | required, non-empty string; both snapshots must belong to it |
| `left`           | required, non-empty snapshot name (the left-hand side)      |
| `right`          | required, non-empty snapshot name (the right-hand side)     |

Using the same snapshot name for `left` and `right` is legal; the two sides
are then the same capture and every row's `equal` marker is true.

Resources are aligned by `resourceId`. The `resources` array carries one row
per resource in the union of the two sides, sorted by `resourceId` in
Unicode code-point order. Each row is
`{"resourceId", "left", "right", "equal"}`, where `left` and `right` each
hold the three balances `capacity`, `occupied`, and `remaining`. A resource
missing on one side is treated as absent and that side's three balances are
all `0`. `equal` is true exactly when the two sides' three balances all
agree, and false when any one of them differs.

In addition to the rows, four grouping arrays partition the resources:

- `leftOnly` — identifiers that appear only in the left snapshot.
- `rightOnly` — identifiers that appear only in the right snapshot.
- `same` — identifiers present on both sides whose `capacity`, `occupied`,
  and `remaining` all match.
- `diff` — identifiers present on both sides that disagree in at least one
  of those three balances. Each entry is `{"resourceId", "fields"}`, where
  `fields` names the mismatched balances, drawn only from `capacity`,
  `occupied`, and `remaining`.

A resource present on only one side is never placed in `same` or `diff`; it
appears only in `leftOnly`/`rightOnly` (its row still carries a zeroed
opposite side). Identifiers and field names are sorted in Unicode code-point
order. Each group array is accompanied by a count key named after the group
plus `Count` (`leftOnlyCount`, `rightOnlyCount`, `sameCount`, `diffCount`).
When neither snapshot holds any resource, the `resources` array and all four
groups are empty and all four counts are `0`.

The `200` response is compact JSON with keys sorted by code point, integer
values kept as integers, booleans kept as booleans, and one trailing
newline:

```json
{"diff":[{"fields":["occupied","remaining"],"resourceId":"res-2"}],"diffCount":1,"leftOnly":["res-1"],"leftOnlyCount":1,"resources":[{"equal":false,"left":{"capacity":5,"occupied":2,"remaining":3},"resourceId":"res-1","right":{"capacity":0,"occupied":0,"remaining":0}},{"equal":false,"left":{"capacity":10,"occupied":4,"remaining":6},"resourceId":"res-2","right":{"capacity":10,"occupied":7,"remaining":3}},{"equal":true,"left":{"capacity":8,"occupied":2,"remaining":6},"resourceId":"res-3","right":{"capacity":8,"occupied":2,"remaining":6}}],"rightOnly":[],"rightOnlyCount":0,"same":["res-3"],"sameCount":1}
```

```bash
curl -X POST http://127.0.0.1:8000/snapshots/compare/resources \
  -H 'Authorization: Bearer tok-1' \
  -H 'Content-Type: application/json' \
  -d '{"organizationId":"org-1","left":"snap-a","right":"snap-b"}'
```

- `401 Unauthorized` — the Bearer credential is missing, malformed, or not
  registered.
- `403 Forbidden` (`{"error": "forbidden"}`) — the credential is bound to a
  different organization, or one of the named snapshots belongs to another
  organization. The organization decision happens before snapshot names are
  inspected, and snapshots are checked in the fixed order `left` then
  `right` (a missing left outranks any problem on the right).
- `404 Not Found` (`{"error": "snapshot_not_found"}`) — either `left` or
  `right` has never existed as a snapshot. Both participating snapshots
  must already exist; a comparison never implicitly creates a snapshot.
- `415 Unsupported Media Type` — missing or unsupported `Content-Type`.
- `400 Bad Request` — body is not syntactically valid JSON.
- `422 Unprocessable Entity` (`validation_error`) — a non-object body, a
  missing or extra field, or a blank or non-string
  `organizationId`/`left`/`right`.

Every non-`200` result is read-only as well: a failed comparison creates no
snapshot and changes no event, reservation, alert, or main-service state.

