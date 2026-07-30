"""Minimal Keenable client: rate-limited search + page fetch.

Keenable's documented limits are 10 req/s per organization when authenticated (no hourly
cap; unauthenticated is 1,000/hr + 10/s per IP). We run at a fraction of that -- the whole
corpus is a few hundred requests, so there is nothing to gain by pushing the limit, and a
shared org key means someone else may be spending it at the same time.

The API returns at most 10 results per search and exposes no pagination, so breadth comes
from issuing several query variants per topic rather than asking for more results.

Key is read from $KEENABLE_API_KEY, or from .env.local at the repo root (gitignored).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.keenable.ai"
MAX_RPS = 4.0          # documented ceiling is 10/s per org; stay well under it
MAX_RETRIES = 5


def load_key() -> str:
    key = os.environ.get("KEENABLE_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parent.parent / ".env.local"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("KEENABLE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("set KEENABLE_API_KEY (or put it in .env.local)")


class _Throttle:
    """Serialise requests to at most MAX_RPS across threads."""

    def __init__(self, rps: float):
        self._min_gap = 1.0 / rps
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next - now
            self._next = max(now, self._next) + self._min_gap
        if sleep_for > 0:
            time.sleep(sleep_for)


_throttle = _Throttle(MAX_RPS)


def _request(method: str, path: str, *, body: dict | None = None,
             params: dict | None = None) -> dict:
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {load_key()}"}
    if data:
        headers["Content-Type"] = "application/json"

    for attempt in range(MAX_RETRIES):
        _throttle.wait()
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # 429 = we were too fast anyway; 5xx = transient. Honour Retry-After.
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                delay = float(e.headers.get("Retry-After") or 0) or 2 ** attempt
                time.sleep(delay)
                continue
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{method} {path} failed: {e}") from e
    raise RuntimeError("unreachable")


def search(query: str, *, published_after: str | None = None,
           published_before: str | None = None, site: str | None = None) -> list[dict]:
    body = {"query": query}
    if published_after:
        body["published_after"] = published_after
    if published_before:
        body["published_before"] = published_before
    if site:
        body["site"] = site
    return _request("POST", "/v1/search", body=body).get("results", [])


def fetch(url: str, *, live: bool = False) -> dict:
    return _request("GET", "/v1/fetch", params={"url": url, "live": str(live).lower()})
