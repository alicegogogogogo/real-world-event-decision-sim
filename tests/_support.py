"""Shared test support for request-scoped credentials.

The service now authenticates every protected entry point with a Bearer
token bound to one organization. The pre-existing tests exercise many
organizations (org-1, org-2, org-a, unknown orgs, even verbatim whitespace
ids), so the low-level request helpers pick the credential organization from
each request itself and lazily register a write token for that exact
organization. Registration is cached per test/server instance.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any
from urllib.parse import parse_qs, urlsplit

DEFAULT_ORG = "org-1"


def select_organization(path: str, body: bytes | None) -> str | None:
    """Return the exact organization a request acts on, if determinable.

    A single, non-blank ``organizationId`` query parameter wins; otherwise a
    non-blank ``organizationId`` string in a JSON object body is used. The
    value is returned verbatim (whitespace kept) so the credential matches the
    service's exact comparison. Blank, duplicated, or absent ids return
    ``None``; those requests fail parameter/body validation (422) before the
    organization is ever compared.
    """
    query = urlsplit(path).query
    values = parse_qs(query, keep_blank_values=True).get("organizationId")
    if values and len(values) == 1 and values[0].strip():
        return values[0]

    if body is not None:
        try:
            data: Any = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            organization_id = data.get("organizationId")
            if isinstance(organization_id, str) and organization_id.strip():
                return organization_id
    return None


def ensure_token(cache: dict[str, str], base_url: str, organization_id: str) -> str:
    """Register (once) and return a write token for ``organization_id``."""
    token = cache.get(organization_id)
    if token is None:
        token = f"test-token-{len(cache) + 1}"
        payload = json.dumps(
            {
                "token": token,
                "organizationId": organization_id,
                "role": "write",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/auth/tokens",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            # 201 (new) or 200 (identical resubmission) are both fine.
            assert response.status in (200, 201)
        cache[organization_id] = token
    return token


def authorization_header(
    cache: dict[str, str], base_url: str, path: str, body: bytes | None
) -> dict[str, str]:
    """Build the Authorization header for one outgoing test request."""
    organization_id = select_organization(path, body) or DEFAULT_ORG
    token = ensure_token(cache, base_url, organization_id)
    return {"Authorization": f"Bearer {token}"}
