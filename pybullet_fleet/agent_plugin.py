"""AgentPlugin — per-agent lifecycle-managed plugins.

Provides a lightweight plugin system for attaching reusable behaviours
to individual :class:`~pybullet_fleet.agent.Agent` instances.

Each plugin receives lifecycle callbacks and has access to its owning
agent via ``self.agent``.

Lifecycle::

    __init__(agent, config)   # construction
    on_init()                 # after agent is fully constructed
    on_update(dt)             # each Agent.update() step
    on_reset()                # when agent state is reset
    on_destroy()              # when agent is removed / sim shuts down

All hooks are **optional no-ops** by default — override only what your
plugin needs.

Registry
--------
::

    from pybullet_fleet.agent_plugin import create_agent_plugin

    plugin = create_agent_plugin("battery", agent)
    plugin = create_agent_plugin("battery", agent, {"charge_rate": 0.01})

Subclasses with a ``_registry_name`` class attribute are auto-registered::

    class BatteryPlugin(AgentPlugin):
        _registry_name = "battery"
        ...

YAML usage::

    plugins:
      - type: "battery"
        config: {}
      - class: "my_project.sensors.LidarPlugin"
        config:
          range: 10.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from pybullet_fleet.plugin_utils import PluginRegistry, from_config_introspect

if TYPE_CHECKING:
    from pybullet_fleet.agent import Agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: PluginRegistry["AgentPlugin"] = PluginRegistry("agent_plugin")


def register_agent_plugin(name: str, cls: Type["AgentPlugin"]) -> None:
    """Register a plugin class under *name*."""
    _registry.register(name, cls)


def create_agent_plugin(
    name: str,
    agent: "Agent",
    config: Optional[Dict[str, Any]] = None,
) -> "AgentPlugin":
    """Create a plugin by registered *name*.

    Convenience wrapper around :func:`create_agent_plugin_from_entry`
    for use in Python code where building an entry dict is verbose.

    Args:
        name: Registry name (e.g. ``"battery"``).
        agent: The owning Agent instance.
        config: Optional configuration dict passed to the plugin.

    Raises:
        KeyError: If *name* is not registered.
    """
    return create_agent_plugin_from_entry({"type": name, "config": config or {}}, agent)


def list_agent_plugins() -> Dict[str, Type["AgentPlugin"]]:
    """Return a copy of the agent plugin registry for inspection."""
    return _registry.items()


def create_agent_plugin_from_entry(
    entry: Dict[str, Any],
    agent: "Agent",
) -> "AgentPlugin":
    """Create a plugin from a YAML-style config entry.

    Supports two formats:

    - **Registry shorthand**: ``{"type": "battery", "config": {...}}``
    - **Dotted path**: ``{"class": "my_pkg.MyPlugin", "config": {...}}``

    Args:
        entry: Plugin definition dict with ``type`` or ``class`` key.
        agent: The owning Agent instance.

    Returns:
        Instantiated :class:`AgentPlugin`.

    Raises:
        KeyError: If ``type`` is not in the registry.
        TypeError: If ``class`` does not resolve to an AgentPlugin subclass.
        ValueError: If neither ``type`` nor ``class`` is present.
    """
    cls = _registry.resolve_from_entry(entry, AgentPlugin)
    plugin_config = entry.get("config", {})
    return cls.from_config(agent, plugin_config)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class AgentPlugin:
    """Base class for per-agent plugins.

    Lifecycle::

        __init__(agent)           # construction (typed kwargs in subclasses)
        on_init()                 # after agent is fully constructed
        on_update(dt)             # each simulation step
        on_reset()                # when agent state is reset
        on_destroy()              # when agent is removed / sim shuts down

    Subclasses override the hooks they need.  All hooks are no-ops by
    default.

    Subclasses with a ``_registry_name`` class attribute are auto-registered
    in ``AGENT_PLUGIN_REGISTRY`` (e.g. ``_registry_name = "battery"``).

    Attributes:
        agent: The owning :class:`Agent` instance.
        config: Raw configuration dict (populated by :meth:`from_config`).
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _registry.auto_register_subclass(cls)

    def __init__(self, agent: "Agent") -> None:
        self.agent = agent
        self.config: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, agent: "Agent", config: Dict[str, Any] | None = None) -> "AgentPlugin":
        """Create from config dict, matching keys to ``__init__`` parameters.

        Works like :meth:`Controller.from_config`: inspects the subclass
        ``__init__`` signature and passes matching keys as kwargs.
        The raw *config* dict is stored as :attr:`config`.

        Args:
            agent: The owning Agent instance.
            config: Configuration dict whose keys are matched to ``__init__``
                parameters (unmatched keys are silently ignored).
        """
        return from_config_introspect(cls, (agent,), config or {})

    @classmethod
    def create_batch(cls, sim_core: Any) -> Optional["PluginBatch"]:
        """A :class:`PluginBatch` that updates every plugin of this class in one call, or ``None``.

        The simulation asks once per plugin class. With a batch, the plugins
        of the class are added to it as their agents join the simulation and
        the batch's :meth:`PluginBatch.update` runs once per step in place of
        each plugin's :meth:`on_update`. ``None`` (the default) keeps the
        plugin-by-plugin updates.

        A batched plugin is updated by the simulation's step, not by
        :meth:`Agent.update`: calling ``agent.update(dt)`` by hand runs the
        agent's per-agent plugins only.
        """
        return None

    def on_init(self) -> None:
        """Called after the agent is fully constructed.

        Override to perform one-time setup that depends on the agent
        being registered with sim_core.  No-op by default.
        """

    def on_update(self, dt: float) -> None:
        """Called each simulation step from :meth:`Agent.update`.

        Override in subclasses.  Default is a no-op.

        Args:
            dt: Simulation timestep in seconds.
        """

    def on_reset(self) -> None:
        """Called when the agent's state is reset.

        Override to reinitialise plugin state.  No-op by default.
        """

    def on_destroy(self) -> None:
        """Called when the agent is removed or the simulation shuts down.

        Override to release resources.  No-op by default.
        """

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} agent='{self.agent.name}'>"


class PluginBatch:
    """Updates every plugin of one class in one call (see :meth:`AgentPlugin.create_batch`).

    A plugin is added when its agent is in the simulation and removed when
    it leaves (or the plugin is removed from the agent); between the two the
    batch, not the agent, updates it.
    """

    def add(self, plugin: "AgentPlugin") -> bool:
        """Take ``plugin`` into the batch.

        Returns ``False`` for a plugin the batch does not take (one whose
        behaviour it cannot reproduce, say); the agent then updates that
        plugin itself. ``None`` counts as taken.
        """
        raise NotImplementedError

    def remove(self, plugin: "AgentPlugin") -> None:
        raise NotImplementedError

    def update(self, dt: float) -> None:
        raise NotImplementedError
