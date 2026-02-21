"""Centralized PoC V2 state — single source of truth for generation activity."""

import threading

_lock = threading.Lock()
_poc_active: bool = False


def is_poc_active() -> bool:
    """Return *True* while a PoC generation round is in progress."""
    return _poc_active


def set_poc_active(active: bool) -> None:
    """Set the PoC-active flag (thread-safe)."""
    global _poc_active
    with _lock:
        _poc_active = active
