"""The ONE place the Snwolley Agent API Toolkit is called.

Contract (from the assignment brief):
    POST https://v1.snwolley.ai/v1/chat/completions/assignment
    headers: Content-Type: application/json, X-API-Key: <key>
    body:    {"message": "<text>", "stream": false}
    limits:  5 requests/minute. HTTP 429 carries Retry-After (seconds). HTTP 401 = bad/missing key.

The endpoint takes ONE text field, so the system rules, the facts sheet and the required JSON shape
are sent together as a single message. The reply is returned as text; core.ai.insights parses it and
the guardrails validate it (citations, numbers, wording), so a weak or odd answer falls back to the
labelled template instead of reaching the page.

The key comes from the environment only (SNWOLLEY_API_KEY via core.config). It is never logged and
never placed in an error message.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Callable, Protocol

import httpx

from core.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://v1.snwolley.ai/v1/chat/completions/assignment"
_REPLY_KEYS = ("response", "message", "content", "text", "answer", "output", "reply", "result")


class ProviderUnavailable(Exception):
    """The AI provider cannot answer right now (not configured, rate limited, network error)."""


class Provider(Protocol):
    name: str

    def generate(self, system: str, user: str, json_schema: dict) -> str: ...


def _retry_seconds(resp: httpx.Response) -> float:
    """Seconds to wait after a 429: Retry-After header first, then 'in N seconds' in the body."""
    header = resp.headers.get("Retry-After", "")
    if header.strip().isdigit():
        return float(header)
    match = re.search(r"(\d+)\s*second", resp.text or "")
    return float(match.group(1)) if match else 15.0


def _text_of(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):  # e.g. {"content": "..."} or {"message": {"content": "..."}}
        for key in _REPLY_KEYS:
            found = _text_of(value.get(key))
            if found:
                return found
    return None


def extract_reply(resp: httpx.Response) -> str:
    """Pull the answer text out of the response. Tolerant of the common shapes; raises if none match."""
    try:
        data = resp.json()
    except ValueError:
        data = None
    if data is None:  # not JSON: treat a non-empty body as the text itself
        if (resp.text or "").strip():
            return resp.text
        raise ProviderUnavailable("Snwolley returned an empty reply.")
    if isinstance(data, str) and data.strip():
        return data
    if isinstance(data, dict):
        choices = data.get("choices")  # OpenAI-style
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            found = _text_of(choices[0].get("message")) or _text_of(choices[0].get("text"))
            if found:
                return found
        found = _text_of(data)
        if found:
            return found
    raise ProviderUnavailable("Snwolley reply had an unrecognised shape (no answer text found).")


class SnwolleyAdapter:
    name = "snwolley"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        *,
        timeout: float = 60.0,
        max_wait: float = 60.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._api_key = api_key  # never log or print this
        self._url = base_url or DEFAULT_URL
        self._timeout = timeout
        self._max_wait = max_wait
        self._client = client
        self._sleep = sleep

    def _post(self, message: str) -> httpx.Response:
        headers = {"Content-Type": "application/json", "X-API-Key": self._api_key}
        body = {"message": message, "stream": False}
        if self._client is not None:
            return self._client.post(self._url, headers=headers, json=body, timeout=self._timeout)
        with httpx.Client() as client:
            return client.post(self._url, headers=headers, json=body, timeout=self._timeout)

    def generate(self, system: str, user: str, json_schema: dict) -> str:
        message = f"{system}\n\n{user}"
        for attempt in (1, 2):
            try:
                resp = self._post(message)
            except httpx.TimeoutException:
                raise ProviderUnavailable("Snwolley request timed out.") from None
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"Snwolley network error ({type(exc).__name__}).") from None

            if resp.status_code == 429:
                wait = _retry_seconds(resp)
                if attempt == 1 and wait <= self._max_wait:
                    logger.info("Snwolley rate limit hit; waiting %.0fs then retrying once", wait)
                    self._sleep(wait + 1)
                    continue
                raise ProviderUnavailable(f"Snwolley rate limit (5 requests/minute); retry in about {wait:.0f}s.")
            if resp.status_code == 401:
                raise ProviderUnavailable("Snwolley rejected the API key (401). Check SNWOLLEY_API_KEY.")
            if resp.status_code >= 400:
                raise ProviderUnavailable(f"Snwolley returned HTTP {resp.status_code}.")
            return extract_reply(resp)
        raise ProviderUnavailable("Snwolley did not answer.")  # not reachable; keeps the type checker calm

    def __repr__(self) -> str:  # so the key can never leak through a log line or traceback
        return "SnwolleyAdapter(api_key=***)"


def get_provider(settings: Settings) -> Provider | None:
    """The configured provider, or None when no key is set."""
    if not settings.snwolley_configured:
        return None
    return SnwolleyAdapter(settings.snwolley_api_key.get_secret_value(), settings.snwolley_base_url)
