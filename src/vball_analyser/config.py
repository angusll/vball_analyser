"""Application configuration and security/pipeline defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _parse_allowed_local_dirs() -> tuple[Path, ...]:
    raw = os.environ.get("VBALL_ALLOWED_LOCAL_DIRS", "")
    if raw.strip():
        parts = [Path(p).resolve() for p in raw.split(os.pathsep) if p.strip()]
        if parts:
            return tuple(parts)
    return (PROJECT_ROOT.resolve(), Path.home().resolve())


def _parse_bind_host() -> str:
    raw = os.environ.get("VBALL_HOST", "").strip()
    return raw or "127.0.0.1"


def _parse_default_port() -> int:
    raw = os.environ.get("VBALL_PORT", "").strip()
    if not raw:
        return 8081
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid VBALL_PORT value: {raw!r} (must be an integer 1..65535)") from exc
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid VBALL_PORT value: {port} (must be between 1 and 65535)")
    return port


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("VBALL_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve())
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("VBALL_MODELS_DIR", str(PROJECT_ROOT / "models"))).resolve())
    secret_file: Path = field(default_factory=lambda: Path(os.environ.get("VBALL_SECRET_FILE", str(PROJECT_ROOT / ".vball_secret"))).resolve())
    allowed_local_dirs: tuple[Path, ...] = field(default_factory=_parse_allowed_local_dirs)
    max_upload_bytes: int = int(os.environ.get("VBALL_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
    allowed_extensions: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".avi"})
    short_clip_max_sec: float = 20.0
    default_net_height_m: float = 2.43
    net_width_m: float = 9.0
    bind_host: str = field(default_factory=_parse_bind_host)
    default_port: int = field(default_factory=_parse_default_port)
    session_ttl_seconds: int = 28800
    rate_limit_requests: int = 300
    rate_limit_window_s: float = 60.0

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def serve_sessions_dir(self) -> Path:
        return self.data_dir / "serve_sessions"


DEFAULT_CONFIG = AppConfig()
