"""Construction injection seams: the string-transport bridge between restricted
layers and config.

The architecture dependency matrix only allows engine/brain to import config;
the knobs of consumption layers such as store/lnn/retrieval are all registered
in ``config`` (ENV_MAP), but the way consumption points receive values is:
the **sole production constructor engine** (at manager import time) calls each
module's ``apply_config`` to pour ``load_config()``'s field values into this
module's injection slots.

Resolution order = **env first, injected value as fallback**
(:func:`resolve`):

  * env set (including the empty string) -> the env text verbatim -- the
    "read fresh every call, tests can flip it" contract preserved as-is; the
    entire existing surface of tests setting ``os.environ`` directly keeps
    zero changes;
  * env unset -> the injected value (production defaults flow through config
    at a single point);
  * neither -> each knob's original env fallback value.

A config value is itself the result of reading env, so both paths agree under
any ordering; parsing (float/int/whitelist) still lives at each consumption
point (unified contract: string transport, zero drift).
"""

from __future__ import annotations

import os

_INJECTED: dict[str, str] = {}


def apply_config(**values: str) -> None:
    """Constructor injects config string-transport values (later calls overwrite
    same-named keys)."""
    _INJECTED.update(values)


def resolve(env_name: str, key: str, env_default: str) -> str:
    """Knob value: env first (empty string counts as set, verbatim); only when
    env is unset does the injected value / original default apply."""
    injected = _INJECTED.get(key)
    return os.environ.get(
        env_name, env_default if injected is None else injected)
