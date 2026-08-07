"""Thin Python client for the PPD REST API at https://ppd-api.vivekladha.com."""
from __future__ import annotations
import json
import urllib.request
import urllib.parse

PPD_API  = "https://ppd-api.vivekladha.com"
_TIMEOUT = 10
_HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "node-fetch/3.3.2",
}


def _get(path: str) -> object:
    req = urllib.request.Request(f"{PPD_API}{path}", headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode())


def search_people(q: str) -> list[dict]:
    """Search people by name/email/phone. Returns list of person objects."""
    return _get(f"/api/persons/search?q={urllib.parse.quote(q)}")


def get_upcoming_milestones(days: int = 30) -> dict:
    """Return {birthdays: [...], anniversaries: [...], windowDays: N}."""
    return _get(f"/api/reminders/upcoming?days={days}")


def list_events() -> list[dict]:
    """Return all events (event_id, name, type, start_at, end_at, …)."""
    return _get("/api/events")
