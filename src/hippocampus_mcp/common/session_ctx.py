"""Session routing after daemonization.

**Why this must exist**: the daemon makes "one process serving N sessions"
real, so the session-level stickiness of ``library_use()``
(``server.set_library_binding``) must be routed per connection -- otherwise
when session A switches to library 甲, tool calls issued by session B
would also land in library 甲, **with no error on either side** (exactly
the "silent failure" archetype from memory).

**Why contextvars instead of globals**: the daemon runs one thread per
connection, and anyio task groups run inside each thread. contextvars
propagate within the same task and are naturally isolated across
connections; a global cannot give "one key per connection".

**Why stdio is not an issue**: with a direct stdio connection one process
serves one session, ``current_session()`` is always ``__default__``, and
behavior is identical to before the change.

**Explicitly out of scope**: this module only handles "which connection
the current context belongs to"; it does not resolve libraries and does
not bind -- that lives in ``server.py``; this module has zero dependency
on server.
"""

from __future__ import annotations

from contextvars import ContextVar

__all__ = [
    "DEFAULT_SESSION_KEY",
    "HELLO_PREFIX",
    "bind_session",
    "current_session",
    "session_scope",
]

#: Prefix of the **out-of-band handshake line** from launcher → daemon
#: (fixes cross-library crosstalk when the daemon is reused).
#:
#: Background: the daemon process's default library is a **process-level**
#: constant (computed at server module import from env
#: ``HIPPOCAMPUS_PERSIST_DIR`` + the cwd guide file). If two workspaces
#: reuse the same daemon (same key -- which is exactly the point of the
#: "N→1 model" memory saving), the second workspace's writes land in
#: **the library of whichever launcher started first**: no error, no
#: warning, just crosstalk.
#:
#: So the launcher sends one plain-text line declaring its library before
#: forwarding any MCP traffic, and the daemon remembers it per
#: **connection**. This line is not JSON-RPC; the daemon must swallow it
#: and must not leave it for MCP parsing (otherwise MCP reports a protocol
#: error).
#:
#: Placed in this module because both launcher and bridge can reference it
#: at zero cost (this module is pure stdlib, no heavy dependencies), and it
#: is the **single source of truth** -- two separately maintained copies of
#: the string would drift sooner or later.
HELLO_PREFIX = "HIPPO-HELLO "

#: Session name when no routing has happened. Must match ``server.DEFAULT_KEY``.
DEFAULT_SESSION_KEY = "__default__"

_current_session: ContextVar[str] = ContextVar(
    "hippocampus_session", default=DEFAULT_SESSION_KEY)


def bind_session(key: str) -> None:
    """Bind the **current context** to a session key.

    Called by each daemon connection before spinning up its JSON-RPC
    session. An empty key always falls to :data:`DEFAULT_SESSION_KEY` --
    "not specified" must not degrade into "bound to the previous route's
    residual value", which is exactly what session crosstalk looks like.
    """
    _current_session.set((key or "").strip() or DEFAULT_SESSION_KEY)


def current_session() -> str:
    """The current context's session key (:data:`DEFAULT_SESSION_KEY` when unbound)."""
    return _current_session.get() or DEFAULT_SESSION_KEY


class session_scope:
    """Temporarily bind a session for a block of code, restoring on exit (used by tests and the bridge).

    Usage::

        with session_scope("conn-7"):
            ...  # tool calls inside this block all use conn-7's library binding
    """

    def __init__(self, key: str) -> None:
        self.key = (key or "").strip() or DEFAULT_SESSION_KEY
        self._token = None

    def __enter__(self) -> "session_scope":
        self._token = _current_session.set(self.key)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _current_session.reset(self._token)
            self._token = None
