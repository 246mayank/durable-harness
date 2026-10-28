"""Failure taxonomy, classified by what the caller knows, not what the server did.

    NotApplied  definitely not processed (connection refused, 429,   -> always safe to retry
                408 deadline exceeded)
    Ambiguous   may have taken effect (5xx, 504, timeout, reset)     -> recover by effect class
    Rejected    processed and refused (other 4xx)                    -> don't retry; tell the planner

A 500 is Ambiguous even though this repo's simulator happens to fail before
committing: from the client's side, a 500 from a real service can follow a
commit (a crash while serializing the response, a failing post-commit hook,
a proxy in between). Only the server knows, and it didn't say.
"""

from __future__ import annotations

from typing import Any

import httpx


class ToolError(Exception):
    kind = "error"

    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class NotApplied(ToolError):
    """The request was definitely not processed."""

    kind = "not_applied"


class Ambiguous(ToolError):
    """The request may or may not have taken effect."""

    kind = "ambiguous"


class Rejected(ToolError):
    """The request was processed and refused."""

    kind = "rejected"


class Unresolvable(Exception):
    """A lookup ran but cannot tell whether the effect happened."""


# Raised before any bytes could have reached the server.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def classify_status(status: int) -> type[ToolError] | None:
    if status < 400:
        return None
    if status in (408, 429):  # 408: the platform refused it unprocessed, past its deadline
        return NotApplied
    if status >= 500:
        return Ambiguous
    return Rejected


def classify_exception(exc: httpx.HTTPError) -> ToolError:
    if isinstance(exc, _NOT_SENT):
        return NotApplied(f"not sent: {type(exc).__name__}")
    # Read/write timeouts, resets and protocol errors can all follow a commit.
    return Ambiguous(f"transport: {type(exc).__name__}")


def request(client: httpx.Client, method: str, url: str, **kwargs: Any) -> Any:
    """Send one request; return the JSON body on 2xx, raise a ToolError otherwise."""
    try:
        response = client.request(method, url, **kwargs)
    except httpx.HTTPError as exc:
        raise classify_exception(exc) from exc
    error = classify_status(response.status_code)
    if error is None:
        return response.json()
    try:
        body = response.json()
    except ValueError:
        body = response.text
    raise error(f"{method} {url} -> {response.status_code}", response.status_code, body)
