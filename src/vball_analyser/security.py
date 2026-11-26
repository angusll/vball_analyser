"""Security helpers: secret resolution, cookie sessions, CSRF, rate limiting, and path confinement."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import deque
from pathlib import Path

from vball_analyser.video_io import validate_video_magic_bytes

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "vball_session"
ALLOWED_HTTP_METHODS = frozenset({"GET", "HEAD", "POST"})

SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "base-uri 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def resolve_secret_key(secret_file: Path) -> bytes:
    """Resolve signing key from env, local 0600 file, or secure OS PRNG (no literal fallback)."""
    env_key = os.environ.get("VBALL_SECRET_KEY", "").strip()
    if env_key:
        return hashlib.sha256(env_key.encode("utf-8")).digest()

    if secret_file.is_file():
        stored = secret_file.read_text(encoding="utf-8").strip()
        if stored:
            return hashlib.sha256(stored.encode("utf-8")).digest()

    logger.warning("VBALL_SECRET_KEY not set; generating ephemeral local secret at %s.", secret_file)
    generated = secrets.token_hex(32)
    try:
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        secret_file.write_text(generated, encoding="utf-8")
        os.chmod(secret_file, 0o600)
    except OSError as exc:
        logger.warning("Could not persist secret key file (%s); keeping in memory.", exc)
    return hashlib.sha256(generated.encode("utf-8")).digest()


class SessionAuthManager:
    """Server-side bounded-TTL session store with HMAC-signed cookies and CSRF tokens."""

    def __init__(self, secret_key: bytes, ttl_seconds: int = 28800) -> None:
        self._secret_key = secret_key
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, dict[str, float | str]] = {}

    def _sign(self, payload: str) -> str:
        return hmac.new(self._secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _prune_expired(self) -> None:
        now = time.time()
        expired = [sid for sid, rec in self._sessions.items() if now > float(rec["expires_at"])]
        for sid in expired:
            self._sessions.pop(sid, None)

    def create_session(self) -> tuple[str, str]:
        self._prune_expired()
        sid = secrets.token_urlsafe(24)
        csrf_nonce = secrets.token_urlsafe(24)
        self._sessions[sid] = {
            "expires_at": time.time() + self._ttl_seconds,
            "csrf_nonce": csrf_nonce,
        }
        cookie_val = f"{sid}.{self._sign(sid)}"
        return sid, cookie_val

    def verify_cookie(self, cookie_val: str | None) -> str | None:
        if not cookie_val or "." not in cookie_val:
            return None
        sid, sig = cookie_val.rsplit(".", 1)
        if not hmac.compare_digest(self._sign(sid), sig):
            return None
        rec = self._sessions.get(sid)
        if rec is None:
            return None
        if time.time() > float(rec["expires_at"]):
            self._sessions.pop(sid, None)
            return None
        return sid

    def invalidate_session(self, sid: str | None) -> None:
        if sid:
            self._sessions.pop(sid, None)

    def issue_csrf_token(self, sid: str) -> str:
        rec = self._sessions.get(sid)
        if rec is None:
            raise ValueError("Unknown or expired session.")
        nonce = str(rec["csrf_nonce"])
        msg = f"csrf:{sid}:{nonce}"
        return f"{nonce}.{self._sign(msg)}"

    def verify_csrf_token(self, sid: str, token: str | None) -> bool:
        if not token or "." not in token:
            return False
        rec = self._sessions.get(sid)
        if rec is None:
            return False
        expected_nonce = str(rec["csrf_nonce"])
        nonce, sig = token.split(".", 1)
        if not hmac.compare_digest(nonce, expected_nonce):
            return False
        expected_sig = self._sign(f"csrf:{sid}:{expected_nonce}")
        return hmac.compare_digest(sig, expected_sig)


class SlidingWindowRateLimiter:
    """In-memory per-client sliding-window request limiter."""

    def __init__(self, max_requests: int = 300, window_s: float = 60.0) -> None:
        self._max_requests = max_requests
        self._window_s = window_s
        self._hits: dict[str, deque[float]] = {}

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        dq = self._hits.setdefault(client_key, deque())
        cutoff = now - self._window_s
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self._max_requests:
            return False
        dq.append(now)
        return True


def validate_local_video_path(
    raw_path: str,
    allowed_dirs: tuple[Path, ...],
    allowed_extensions: frozenset[str],
) -> Path:
    """Resolve and strictly confine a user-provided local video file path."""
    cleaned = raw_path.strip()
    if not cleaned or "\x00" in cleaned:
        raise ValueError("Invalid video path.")

    resolved = Path(cleaned).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Selected path is not a regular file.")

    ext = resolved.suffix.lower()
    if ext not in allowed_extensions:
        raise ValueError(f"Unsupported video extension: {ext}")

    allowed_resolved = [d.resolve() for d in allowed_dirs]
    if not any(resolved.is_relative_to(base) for base in allowed_resolved):
        raise PermissionError("Access denied: path is outside allowed local directories.")

    with resolved.open("rb") as fh:
        header = fh.read(32)
    if not validate_video_magic_bytes(header, ext):
        raise ValueError("File header does not match declared video container format.")

    return resolved
