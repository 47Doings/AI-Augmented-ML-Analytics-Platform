"""Passwords, login tokens, and a simple login throttle.

PASSWORDS
  Never stored. We store a salted scrypt hash (Python standard library, no extra
  package). scrypt is deliberately slow and memory-hungry, so guessing passwords
  from a stolen database is expensive. Every user gets their own random salt, so
  two people with the same password have different hashes.

TOKENS
  After login the server gives the browser a signed token (JWT, HS256) holding
  the user id and an expiry. The server signs it with APP_SECRET_KEY; anyone
  who changes the contents breaks the signature. We pin the algorithm on
  decode, so a forged token cannot choose a weaker one ("alg confusion").

THROTTLE
  After 5 failed logins for the same email + client address within 15 minutes,
  further attempts are refused until the window passes. It lives in memory, so
  it resets on restart and is per process; a real deployment would put this in
  Redis or at the reverse proxy.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import jwt

from core.config import get_settings

_N, _R, _P = 2**14, 8, 1
_ALGORITHM = "HS256"
MIN_PASSWORD, MAX_PASSWORD = 8, 128  # the upper bound stops people from sending huge strings to the hasher


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r), p=int(p), dklen=32)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, bytes.fromhex(digest_hex))  # constant-time comparison


# Used when the email does not exist, so "no such user" takes as long as "wrong password".
_DUMMY_HASH = hash_password(secrets.token_hex(8))


def burn_time() -> None:
    verify_password("not-the-password", _DUMMY_HASH)


def create_token(user_id: int) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(hours=s.session_hours)}
    return jwt.encode(payload, s.app_secret_key.get_secret_value(), algorithm=_ALGORITHM)


def read_token(token: str) -> int | None:
    """The user id inside a valid, unexpired token, otherwise None."""
    try:
        payload = jwt.decode(
            token, get_settings().app_secret_key.get_secret_value(),
            algorithms=[_ALGORITHM], options={"require": ["exp", "sub"]},
        )
        return int(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError):
        return None


class LoginThrottle:
    def __init__(self, max_failures: int = 5, window_seconds: int = 900):
        self.max_failures, self.window = max_failures, window_seconds
        self._failures: dict[tuple, deque] = defaultdict(deque)

    def _prune(self, key: tuple) -> deque:
        q = self._failures[key]
        cutoff = time.monotonic() - self.window
        while q and q[0] < cutoff:
            q.popleft()
        return q

    def blocked(self, key: tuple) -> bool:
        return len(self._prune(key)) >= self.max_failures

    def record_failure(self, key: tuple) -> None:
        self._prune(key).append(time.monotonic())

    def reset(self, key: tuple) -> None:
        self._failures.pop(key, None)


throttle = LoginThrottle()
