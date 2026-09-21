"""ASGI app re-export. Canonical definition lives in backend.main."""

from backend.main import app, create_app

__all__ = ["app", "create_app"]
