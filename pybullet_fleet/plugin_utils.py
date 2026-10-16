"""Shared registry and factory utilities for Controller, AgentPlugin, and SimPlugin.

Provides:

- :class:`PluginRegistry` — reusable registry with ``type:``/``class:`` resolution.
- :func:`from_config_introspect` — unified ``config dict → __init__ kwargs`` factory,
  used by all three base classes' ``from_config()`` classmethods.

Usage in each subsystem::

    _registry = PluginRegistry[Controller]("controller")

    class Controller(ABC):
        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)
            _registry.auto_register_subclass(cls)

        @classmethod
        def from_config(cls, config):
            return from_config_introspect(cls, (), config)

    class AgentPlugin:
        @classmethod
        def from_config(cls, agent, config):
            return from_config_introspect(cls, (agent,), config)
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Generic, Tuple, Type, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Config → __init__ kwargs factory (shared by Controller, AgentPlugin, SimPlugin)
# ---------------------------------------------------------------------------


def from_config_introspect(cls: type, owner_args: Tuple[Any, ...], config: dict) -> Any:
    """Create an instance by matching *config* keys to ``__init__`` kwargs.

    This is the unified factory behind ``Controller.from_config()``,
    ``AgentPlugin.from_config()``, and ``SimPlugin.from_config()``.

    Args:
        cls: The class to instantiate.
        owner_args: Positional args passed before kwargs
            (e.g. ``()`` for Controller, ``(agent,)`` for AgentPlugin,
            ``(sim_core,)`` for SimPlugin).
        config: Dict of config keys to match against ``__init__`` params.

    Returns:
        Instance of *cls* constructed with matched kwargs and
        ``config`` attribute set to the original dict.
    """
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.values())

    # Skip 'self' + positional owner params
    skip = 1 + len(owner_args)
    remaining = params[skip:]

    # Check if __init__ has **kwargs  (pass all config through if so)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in remaining)

    if has_var_keyword:
        kwargs = dict(config)
    else:
        valid = {p.name for p in remaining}
        kwargs = {k: v for k, v in config.items() if k in valid}

    instance = cls(*owner_args, **kwargs)
    # Store raw config for introspection (overrides the {} set by base __init__)
    instance.config = dict(config)
    return instance


class PluginRegistry(Generic[T]):
    """Self-contained registry supporting ``type:`` and ``class:`` resolution.

    Each subsystem (Controller, AgentPlugin, SimPlugin) creates one instance.
    The registry owns its internal dict — no external dict is required.

    Parameters:
        label: Human-readable label for error messages (e.g. ``"controller"``).
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._registry: Dict[str, Type[T]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, cls: Type[T]) -> None:
        """Manually register *cls* under *name*."""
        self._registry[name] = cls

    def unregister(self, name: str) -> None:
        """Remove *name* from the registry. No-op if not present."""
        self._registry.pop(name, None)

    def auto_register_subclass(self, cls: type) -> None:
        """Auto-register *cls* if it has ``_registry_name`` and is concrete.

        Intended to be called from ``__init_subclass__`` in base classes.
        Abstract classes (detected via :func:`inspect.isabstract`) are skipped.
        """
        # A subclass must opt in with its own name; inheriting a built-in
        # name would silently replace that built-in in the registry.
        name = cls.__dict__.get("_registry_name")
        if name and not inspect.isabstract(cls):
            self._registry[name] = cls

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Type[T]:
        """Look up a class by registry *name*.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"Unknown {self._label}: {name!r}. Available: {list(self._registry)}")
        return self._registry[name]

    def items(self) -> Dict[str, Type[T]]:
        """Return a shallow copy of the registry dict."""
        return dict(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    # ------------------------------------------------------------------
    # YAML entry resolution (type: / class:)
    # ------------------------------------------------------------------

    def resolve_from_entry(self, entry: Dict[str, Any], base_class: Type[T]) -> Type[T]:
        """Resolve a class from a YAML-style entry dict.

        Supports two formats:

        - ``{"type": "<registry_name>", ...}`` — registry lookup.
        - ``{"class": "<dotted.path>", ...}`` — dynamic import.

        Args:
            entry: Dict with ``type`` or ``class`` key (and optional ``config``).
            base_class: Expected base class for type validation (dotted path only).

        Returns:
            Resolved class object.

        Raises:
            KeyError: If ``type`` name is not in the registry.
            TypeError: If ``class`` does not resolve to a *base_class* subclass.
            ValueError: If neither ``type`` nor ``class`` is present.
        """
        if "type" in entry:
            return self.get(entry["type"])

        if "class" in entry:
            from pybullet_fleet.config_utils import resolve_class

            cls = resolve_class(entry["class"])
            if not (isinstance(cls, type) and issubclass(cls, base_class)):
                raise TypeError(
                    f"{self._label.replace('_', ' ').title()} class must be a " f"{base_class.__name__} subclass, got {cls!r}"
                )
            return cls  # type: ignore[return-value]

        raise ValueError(f"{self._label.replace('_', ' ').title()} entry must have 'type' or 'class' key, got: {entry!r}")
