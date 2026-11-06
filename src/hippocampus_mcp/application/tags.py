# -*- coding: utf-8 -*-
"""``filter_tags`` key-name whitelist validation.

It needs to call ``domain.tagging``
(pure domain validation), and both server.py and the use-case services use
it -- the application layer must not import server, so it lands in this
module; server.py imports it under an alias, keeping the existing call
sites.

Returns ``(effective_tags, info)``. Caller contract:

* ``info is None``            -- no information (whitelist undeclared or
  all keys valid); use ``effective_tags`` directly (**bit-for-bit
  identical** to the passed-in ``parsed_tags``).
* ``info`` contains ``error``       -- blocked; return ``info`` directly as
  the error body.
* ``info`` contains ``degraded``    -- already soft-degraded (invalid keys
  stripped); continue with ``effective_tags`` and hand ``info`` to
  :func:`~hippocampus_mcp.common.params.filter_applied` for the echo.

Why this runs **after** parsing: a parse failure (not valid JSON) and an
invalid key name are two different errors; the former needs no schema
lookup, so checking it first is cheaper and yields more accurate error
wording.
"""
from __future__ import annotations

from typing import Any

from ..domain.tagging import validate_filter_keys


def check_tag_keys(brain: Any, parsed_tags: dict[str, Any] | None
                   ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate the key names of ``filter_tags`` (brain is used only to fetch the ``tag_extractor``)."""
    if not parsed_tags:
        return parsed_tags, None
    return validate_filter_keys(parsed_tags, brain.tag_extractor)
