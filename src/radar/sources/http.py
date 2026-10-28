"""Shared HTTP plumbing for every source: one polite client, and retries with backoff.

Public APIs (arXiv, Semantic Scholar, Hugging Face, Hacker News) all rate-limit. They answer
429 (too many requests) or a 5xx when busy, sometimes with a Retry-After header saying how many
seconds to wait. `request_with_retry` waits and tries again, doubling the wait each time
(exponential backoff), and raises `SourceUnavailable` with a readable message if the source
still refuses.
"""

import time

import httpx

USER_AGENT = "ai-research-radar/0.1 (personal research agent; https://github.com)"
RETRY_STATUSES = (429, 500, 502, 503, 504)
RETRIES = 4
BACKOFF_S = 5.0  # waits 5, 10, 20, 40 seconds unless the server sends Retry-After


class SourceUnavailable(Exception):
    """A source kept refusing requests (rate limit, outage, network) after all retries."""


def make_client(headers: dict | None = None, timeout: float = 30) -> httpx.Client:
    # Many APIs throttle anonymous default user agents, so say who we are.
    return httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        timeout=timeout,
    )


def request_with_retry(
    client: httpx.Client,
    url: str,
    *,
    name: str,
    method: str = "GET",
    params: dict | None = None,
    json: object = None,
    retries: int = RETRIES,
    backoff_s: float = BACKOFF_S,
) -> httpx.Response:
    """Send a request, retrying rate limits, server errors and network errors.

    Other errors (404, 400, ...) raise httpx.HTTPStatusError at once: retrying won't fix them.
    """
    problem = ""
    for attempt in range(retries + 1):
        retry_after = ""
        try:
            resp = client.request(method, url, params=params, json=json)
        except httpx.TransportError as e:  # timeouts, dropped connections
            problem = f"a network error ({type(e).__name__})"
        else:
            if resp.status_code not in RETRY_STATUSES:
                resp.raise_for_status()
                return resp
            problem = f"status {resp.status_code}"
            retry_after = resp.headers.get("Retry-After", "")
        if attempt == retries:
            break
        time.sleep(float(retry_after) if retry_after.isdigit() else backoff_s * 2**attempt)
    raise SourceUnavailable(f"{name} gave {problem} after {retries + 1} tries.")
