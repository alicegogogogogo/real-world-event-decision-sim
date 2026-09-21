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

### Events

Events live only in the server process: restarting the service (or starting a
new server instance) starts with an empty ledger.

`POST /events` accepts `Content-Type: application/json` only. The body must be
a JSON object with exactly these fields:

- `eventId`, `organizationId`, `type`: non-empty strings
- `occurredAt`: a non-negative integer
- `payload`: a JSON object

Responses:

- `201` with the stored event (the same five fields) when it is first created.
- `200` with the already stored event when the same `eventId` is submitted again
  with identical fields (no duplicate is stored).
- `409` with `{"error": "event_id_conflict"}` when the same `eventId` is
  resubmitted with different fields.
- `415` with `{"error": "unsupported_media_type"}` for a missing or
  unsupported `Content-Type`.
- `400` with `{"error": "invalid_json"}` for malformed JSON.
- `422` with `{"error": "validation_error"}` for invalid fields (arrays at the
  top level, missing/extra/blank fields, floats or negative timestamps,
  non-object payloads).

```bash
curl -X POST http://127.0.0.1:8000/events \
  -H 'Content-Type: application/json' \
  -d '{"eventId":"evt-1","organizationId":"org-1","type":"incident.created","occurredAt":0,"payload":{}}'
```

`GET /events?organizationId=...` requires the `organizationId` parameter
exactly once and non-empty. It returns `200` with
`{"organizationId": "...", "events": [...]}`, where events belong only to that
organization and are sorted by `(occurredAt, eventId)` ascending. An
organization without events gets `[]`. Missing, blank, or repeated
`organizationId` parameters return `422`.

