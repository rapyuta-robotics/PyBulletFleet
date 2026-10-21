"""SimPlugin — lifecycle-managed simulation plugins.

Gazebo World Plugin equivalent for PyBulletFleet.  Plugins receive
lifecycle callbacks and can access the full simulation API via
``self.sim_core``.

All lifecycle methods (``on_init``, ``on_step``, ``on_reset``,
``on_shutdown``) are optional no-ops by default.  Override only the
methods your plugin needs.

**Step-based plugin** (called every sim step / at a fixed frequency)::

    class TrafficMonitor(SimPlugin):
        def on_init(self) -> None:
            self.counts = {}

        def on_step(self, dt: float) -> None:
            for a in self.sim_core.agents:
                self.counts[a.name] = self.counts.get(a.name, 0) + 1

    sim.register_plugin(TrafficMonitor, config={}, frequency=2.0)

**Event-driven plugin** (reacts to EventBus events, no per-step work)::

    class CollisionLogger(SimPlugin):
        def on_init(self) -> None:
            self.sim_core.events.on(
                SimEvents.COLLISION_STARTED, self._on_collision
            )

        def _on_collision(self, obj_a, obj_b):
            print(f"Collision: {obj_a.name} <-> {obj_b.name}")

    sim.register_plugin(CollisionLogger)

**YAML-driven usage**::

    plugins:
      - class: "my_pkg.plugins.TrafficMonitor"
        frequency: 2.0
        config:
          log_interval: 5.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from pybullet_fleet._frequency import FrequencyGate
from pybullet_fleet.plugin_utils import PluginRegistry, from_config_introspect

if TYPE_CHECKING:
    from pybullet_fleet.core_simulation import MultiRobotSimulationCore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_registry: PluginRegistry["SimPlugin"] = PluginRegistry("sim_plugin")


def register_sim_plugin(name: str, cls: Type["SimPlugin"]) -> None:
    """Register a sim plugin class under *name*."""
    _registry.register(name, cls)


def create_sim_plugin(
    name: str,
    sim_core: "MultiRobotSimulationCore",
    config: Optional[Dict[str, Any]] = None,
    frequency: Optional[float] = None,
) -> "SimPlugin":
    """Create a sim plugin by registered *name*.

    Convenience wrapper around :func:`create_sim_plugin_from_entry`.

    Args:
        name: Registry name (e.g. ``"workcell"``).
        sim_core: The simulation core instance.
        config: Optional configuration dict.
        frequency: Step frequency in Hz (``None`` = every step).

    Raises:
        KeyError: If *name* is not registered.
    """
    entry: Dict[str, Any] = {"type": name, "config": config or {}}
    if frequency is not None:
        entry["frequency"] = frequency
    return create_sim_plugin_from_entry(entry, sim_core)


def create_sim_plugin_from_entry(
    entry: Dict[str, Any],
    sim_core: "MultiRobotSimulationCore",
) -> "SimPlugin":
    """Create a sim plugin from a YAML-style config entry.

    Supports two formats:

    - **Registry shorthand**: ``{"type": "workcell", "config": {...}}``
    - **Dotted path**: ``{"class": "my_pkg.MyPlugin", "config": {...}}``

    Both formats support an optional ``frequency`` field (Hz).

    Args:
        entry: Plugin definition dict with ``type`` or ``class`` key.
        sim_core: The simulation core instance.

    Returns:
        Instantiated :class:`SimPlugin` with ``frequency`` set.

    Raises:
        KeyError: If ``type`` is not in the registry.
        TypeError: If ``class`` does not resolve to a SimPlugin subclass.
        ValueError: If neither ``type`` nor ``class`` is present.
    """
    cls = _registry.resolve_from_entry(entry, SimPlugin)
    config = entry.get("config", {})
    freq = entry.get("frequency", None)

    plugin = cls.from_config(sim_core, config)
    plugin.frequency = freq
    return plugin


def list_sim_plugins() -> Dict[str, Type["SimPlugin"]]:
    """Return a copy of the sim plugin registry for inspection."""
    return _registry.items()


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------


class SimPlugin:
    """Base class for simulation plugins.

    Plugins receive lifecycle callbacks and can access the full
    simulation API via ``self.sim_core``.

    All lifecycle hooks are **optional no-ops** by default — override
    only what your plugin needs.  Step-based plugins override
    :meth:`on_step`; event-driven plugins subscribe to
    ``self.sim_core.events`` in :meth:`on_init` and leave
    ``on_step`` alone.

    Lifecycle::

        __init__(sim_core)               # construction (typed kwargs in subclasses)
        on_init()                        # after world/robots are spawned
        on_step(dt)                      # each simulation step (frequency-controlled)
        on_reset()                       # when simulation is reset
        on_shutdown()                    # before PyBullet disconnects

    Subclasses with a ``_registry_name`` class attribute are auto-registered
    via ``_registry_name`` class attribute (e.g. ``_registry_name = "workcell"``).
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _registry.auto_register_subclass(cls)

    def __init__(self, sim_core: "MultiRobotSimulationCore") -> None:
        self.sim_core = sim_core
        self.config: Dict[str, Any] = {}
        self._frequency_gate = FrequencyGate()  # schedule for ``frequency``

    @property
    def frequency(self) -> Optional[float]:
        """Step frequency in Hz; ``None`` calls :meth:`on_step` every step.

        Changing it starts a fresh schedule on the new rate.
        """
        return self._frequency_gate.frequency

    @frequency.setter
    def frequency(self, frequency: Optional[float]) -> None:
        self._frequency_gate.set_frequency(frequency)

    @classmethod
    def from_config(cls, sim_core: "MultiRobotSimulationCore", config: Dict[str, Any] | None = None) -> "SimPlugin":
        """Create from config dict, matching keys to ``__init__`` parameters.

        Works like :meth:`Controller.from_config`: inspects the subclass
        ``__init__`` signature and passes matching keys as kwargs.
        The raw *config* dict is stored as :attr:`config`.

        Args:
            sim_core: The simulation core instance.
            config: Configuration dict whose keys are matched to ``__init__``
                parameters (unmatched keys are silently ignored).
        """
        return from_config_introspect(cls, (sim_core,), config or {})

    def on_init(self) -> None:
        """Called after world and robots are spawned.

        Override to query ``self.sim_core.agents``, subscribe to
        events, or perform one-time setup.  No-op by default.
        """

    def on_step(self, dt: float) -> None:
        """Called each simulation step (subject to frequency control).

        Override for step-based logic.  Event-driven plugins can leave
        this as a no-op and react via ``self.sim_core.events`` instead.

        Args:
            dt: Simulated time since the last call (seconds): the step's
                timestep when :attr:`frequency` is ``None``, otherwise the
                time since the previous scheduled run.
        """

    def on_reset(self) -> None:
        """Called when the simulation is reset.  Override to clear plugin state."""

    def on_shutdown(self) -> None:
        """Called before PyBullet disconnects.  Override to release resources."""


# ------------------------------------------------------------------
# YAML → plugin loading
# ------------------------------------------------------------------


def _load_plugins_from_config(
    plugin_configs: List[Dict[str, Any]],
    sim_core: "MultiRobotSimulationCore",
) -> List[SimPlugin]:
    """Load and instantiate plugins from YAML ``plugins:`` section.

    Each plugin's :attr:`frequency` is set from the YAML entry's
    ``frequency`` field (``None`` means every step).

    Supports both ``type:`` (registry lookup) and ``class:`` (dotted path)
    entries.

    Args:
        plugin_configs: List of dicts, each with ``type`` or ``class``,
            optional ``frequency`` (Hz), and optional ``config`` (dict).
        sim_core: The simulation core instance passed to each plugin.

    Returns:
        List of instantiated :class:`SimPlugin` instances (with
        ``frequency`` already set).

    Raises:
        KeyError: If ``type`` is not in the registry.
        ValueError: If ``class`` is not a valid dotted path.
        ModuleNotFoundError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
        TypeError: If the class is not a :class:`SimPlugin` subclass.
    """
    plugins: List[SimPlugin] = []

    for entry in plugin_configs:
        plugin = create_sim_plugin_from_entry(entry, sim_core)
        plugins.append(plugin)
        logger.info("Loaded sim plugin %r", plugin.__class__.__name__)

    return plugins
