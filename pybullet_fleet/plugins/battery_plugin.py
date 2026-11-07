"""BatteryPlugin — per-agent battery simulation.

Drains/charges battery SOC each step based on the plugin's
rate attributes and the agent's current motion/charging state.

The plugin **owns** all battery state (rates, SOC, charging flag).
Agent exposes delegate properties for convenience::

    agent.battery_soc       # → plugin.soc
    agent.battery_plugin    # → the BatteryPlugin instance (or None)
    agent.is_charging       # → plugin.is_charging
    agent.set_charging(b)   # → plugin.set_charging(b)

Created automatically by :meth:`Agent.from_params` when listed in
``AgentSpawnParams.plugins``, or attached manually.

Example (via plugins list in YAML)::

    plugins:
      - type: battery
        config:
          initial_soc: 0.8
          discharge_rate: 0.002

Example (manual)::

    from pybullet_fleet.plugins.battery_plugin import BatteryPlugin
    agent.add_plugin(BatteryPlugin(agent, discharge_rate=0.002))

Example (custom subclass)::

    class TemperatureBattery(BatteryPlugin):
        def on_update(self, dt: float) -> None:
            temp_factor = 1.0 + 0.02 * (self.agent.user_data.get("temp", 25) - 25)
            if self.is_charging:
                self.soc = min(1.0, self.soc + self.charge_rate * dt)
            elif self.agent.is_moving:
                self.soc = max(0.0, self.soc - self.discharge_rate * dt * temp_factor)

    agent.remove_plugin(BatteryPlugin)
    agent.add_plugin(TemperatureBattery(agent, initial_soc=0.9, discharge_rate=0.005))
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from pybullet_fleet.agent_plugin import AgentPlugin, PluginBatch

if TYPE_CHECKING:
    from pybullet_fleet.agent import Agent

logger = logging.getLogger(__name__)

# Columns of a battery's state, in a plugin's own array and in the batch's.
_DISCHARGE, _CHARGE, _IDLE, _SOC, _CHARGING = range(5)


class _BatteryBatch(PluginBatch):
    """Every :class:`BatteryPlugin` of one simulation, updated as arrays.

    ``values`` holds one row per battery (the plugin's state columns); a
    plugin reads and writes its row through its properties. Whether each
    agent moves comes from the simulation's ``is_moving`` column, through
    the agents' rows, which are cached until the fleet changes.
    """

    def __init__(self, sim_core: Any) -> None:
        self.sim_core = sim_core
        self.plugins: List["BatteryPlugin"] = []
        self.values = np.zeros((16, 5))
        self._agent_rows: Optional[np.ndarray] = None
        self._rows_key: Any = None

    def add(self, plugin: "BatteryPlugin") -> None:
        n = len(self.plugins)
        if n == len(self.values):
            grown = np.zeros((2 * n, 5))
            grown[:n] = self.values
            self.values = grown
        self.values[n] = plugin._own
        plugin._batch = self
        plugin._row = n
        self.plugins.append(plugin)
        self._rows_key = None

    def remove(self, plugin: "BatteryPlugin") -> None:
        row = plugin._row
        if plugin._batch is not self or row < 0:
            return
        plugin._own[:] = self.values[row]
        plugin._batch = None
        plugin._row = -1
        last = len(self.plugins) - 1
        if row != last:
            moved = self.plugins[last]
            self.values[row] = self.values[last]
            moved._row = row
            self.plugins[row] = moved
        self.plugins.pop()
        self._rows_key = None

    def _moving(self) -> np.ndarray:
        store = self.sim_core._store
        key = (self.sim_core._fleet_revision, len(self.plugins))
        if self._rows_key != key or self._agent_rows is None:
            self._agent_rows = store.rows([plugin.agent.object_id for plugin in self.plugins])
            self._rows_key = key
        return store.is_moving[self._agent_rows]

    def update(self, dt: float) -> None:
        n = len(self.plugins)
        if n == 0:
            return
        values = self.values[:n]
        charging = values[:, _CHARGING] > 0.5
        moving = self._moving()
        delta = np.where(
            charging,
            values[:, _CHARGE] * dt,
            np.where(moving, -values[:, _DISCHARGE] * dt, -values[:, _IDLE] * dt),
        )
        values[:, _SOC] = np.clip(values[:, _SOC] + delta, 0.0, 1.0)


class BatteryPlugin(AgentPlugin):
    """Default linear battery drain/charge plugin.

    Owns all battery state as plain attributes:

    - :attr:`discharge_rate` — SOC drain per second while moving
    - :attr:`charge_rate` — SOC gain per second while charging
    - :attr:`idle_rate` — SOC drain per second while idle
    - :attr:`soc` — current state of charge [0.0, 1.0]
    - :attr:`is_charging` — whether the agent is charging

    Update logic:

    - **charging** → ``soc += charge_rate * dt``
    - **moving** → ``soc -= discharge_rate * dt``
    - **idle** → ``soc -= idle_rate * dt``

    SOC is clamped to ``[0.0, 1.0]``.
    """

    _registry_name = "battery"

    def __init__(
        self,
        agent: "Agent",
        initial_soc: float = 1.0,
        discharge_rate: float = 0.001,
        charge_rate: float = 0.005,
        idle_rate: float = 0.0,
    ) -> None:
        super().__init__(agent)
        # The state lives here until the plugin joins a simulation's battery
        # batch, which keeps it in columns and updates every battery at once.
        self._batch: Optional["_BatteryBatch"] = None
        self._row: int = -1
        self._own = np.array([discharge_rate, charge_rate, idle_rate, initial_soc, 0.0], dtype=float)

    def _value(self, column: int) -> float:
        batch = self._batch
        return float(batch.values[self._row, column]) if batch is not None else float(self._own[column])

    def _set_value(self, column: int, value: float) -> None:
        batch = self._batch
        if batch is not None:
            batch.values[self._row, column] = value
        else:
            self._own[column] = value

    @property
    def discharge_rate(self) -> float:
        """SOC drain per second while moving."""
        return self._value(_DISCHARGE)

    @discharge_rate.setter
    def discharge_rate(self, value: float) -> None:
        self._set_value(_DISCHARGE, value)

    @property
    def charge_rate(self) -> float:
        """SOC gain per second while charging."""
        return self._value(_CHARGE)

    @charge_rate.setter
    def charge_rate(self, value: float) -> None:
        self._set_value(_CHARGE, value)

    @property
    def idle_rate(self) -> float:
        """SOC drain per second while idle."""
        return self._value(_IDLE)

    @idle_rate.setter
    def idle_rate(self, value: float) -> None:
        self._set_value(_IDLE, value)

    @property
    def soc(self) -> float:
        """Current state of charge [0.0, 1.0]."""
        return self._value(_SOC)

    @soc.setter
    def soc(self, value: float) -> None:
        self._set_value(_SOC, value)

    @property
    def is_charging(self) -> bool:
        """Whether the agent is charging."""
        return bool(self._value(_CHARGING))

    @is_charging.setter
    def is_charging(self, value: bool) -> None:
        self._set_value(_CHARGING, 1.0 if value else 0.0)

    @classmethod
    def create_batch(cls, sim_core: Any) -> Optional["_BatteryBatch"]:
        """One batch per simulation, for the plain :class:`BatteryPlugin` (a subclass with its own
        ``on_update`` keeps its per-agent update)."""
        if cls.on_update is not BatteryPlugin.on_update or not hasattr(sim_core, "_store"):
            return None
        return _BatteryBatch(sim_core)

    def set_charging(self, charging: bool) -> None:
        """Start or stop charging.

        Args:
            charging: ``True`` to start, ``False`` to stop.
        """
        if self.is_charging != charging:
            logger.info("'%s': charging %s", self.agent.name, "started" if charging else "stopped")
        self.is_charging = charging

    def on_update(self, dt: float) -> None:
        """Update battery SOC based on agent's current state."""
        if self.is_charging:
            self.soc = min(1.0, self.soc + self.charge_rate * dt)
        elif self.agent.is_moving:
            self.soc = max(0.0, self.soc - self.discharge_rate * dt)
        else:
            self.soc = max(0.0, self.soc - self.idle_rate * dt)
