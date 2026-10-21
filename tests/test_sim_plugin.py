"""Tests for SimPlugin ABC and core_simulation integration.

Covers lifecycle, frequency control, error isolation, YAML loading,
and programmatic registration.
"""

import logging

import pybullet as p
import pytest

from pybullet_fleet.core_simulation import MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.sim_plugin import SimPlugin, _load_plugins_from_config


# ---------------------------------------------------------------------------
# Test plugin implementations
# ---------------------------------------------------------------------------


class LifecyclePlugin(SimPlugin):
    """Records every lifecycle call for assertion."""

    def __init__(self, sim_core):
        super().__init__(sim_core)
        self.calls: list = []

    def on_init(self) -> None:
        self.calls.append("on_init")

    def on_step(self, dt: float) -> None:
        self.calls.append(("on_step", round(dt, 6)))

    def on_reset(self) -> None:
        self.calls.append("on_reset")

    def on_shutdown(self) -> None:
        self.calls.append("on_shutdown")


class ErrorPlugin(SimPlugin):
    """Raises an exception in on_step to test error isolation."""

    def on_init(self) -> None:
        pass

    def on_step(self, dt: float) -> None:
        raise RuntimeError("boom")


class ConfigPlugin(SimPlugin):
    """Stores config for assertion."""

    def on_init(self) -> None:
        self.inited = True

    def on_step(self, dt: float) -> None:
        pass


class CounterPlugin(SimPlugin):
    """Counts on_step invocations."""

    def __init__(self, sim_core):
        super().__init__(sim_core)
        self.step_count = 0

    def on_init(self) -> None:
        pass

    def on_step(self, dt: float) -> None:
        self.step_count += 1


class _NotAPlugin:
    """Plain class — not a SimPlugin subclass."""

    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim():
    """Create a minimal headless sim for plugin tests."""
    params = SimulationParams(gui=False, monitor=False, physics=False)
    return MultiRobotSimulationCore(params)


# ---------------------------------------------------------------------------
# Tests: SimPlugin ABC
# ---------------------------------------------------------------------------


class TestSimPluginABC:
    """Tests for the SimPlugin base class itself."""

    def test_base_class_instantiates_with_noop_defaults(self):
        """SimPlugin can be instantiated directly — all hooks are no-op."""
        plugin = SimPlugin(None)  # type: ignore[arg-type]
        assert plugin.sim_core is None
        assert plugin.config == {}
        # All lifecycle methods are callable no-ops
        plugin.on_init()
        plugin.on_step(0.1)
        plugin.on_reset()
        plugin.on_shutdown()

    def test_concrete_subclass_instantiates(self):
        """A concrete subclass can be instantiated."""
        plugin = LifecyclePlugin.from_config(None, {"key": "val"})
        assert plugin.sim_core is None
        assert plugin.config == {"key": "val"}

    def test_on_reset_default_noop(self):
        """on_reset default does nothing (no error)."""
        plugin = LifecyclePlugin(None)
        plugin.on_reset()  # should not raise

    def test_on_shutdown_default_noop(self):
        """on_shutdown default does nothing (no error)."""
        plugin = LifecyclePlugin(None)
        plugin.on_shutdown()  # should not raise


# ---------------------------------------------------------------------------
# Tests: Plugin lifecycle via core_simulation integration
# ---------------------------------------------------------------------------


class TestPluginLifecycle:
    """Tests for plugin lifecycle through MultiRobotSimulationCore."""

    def test_register_plugin_instance(self):
        """register_plugin(instance) adds plugin to sim.plugins."""
        sim = _make_sim()
        plugin = LifecyclePlugin(sim)
        result = sim.register_plugin(plugin)
        assert result is plugin
        assert plugin in sim.plugins
        p.disconnect(sim.client)

    def test_register_plugin_class(self):
        """register_plugin(PluginClass, config) instantiates and adds."""
        sim = _make_sim()
        plugin = sim.register_plugin(LifecyclePlugin, config={"x": 1})
        assert isinstance(plugin, LifecyclePlugin)
        assert plugin.config == {"x": 1}
        assert plugin.sim_core is sim
        assert plugin in sim.plugins
        p.disconnect(sim.client)

    def test_register_plugin_rejects_non_subclass(self):
        """register_plugin with a non-SimPlugin class raises TypeError."""
        sim = _make_sim()
        with pytest.raises(TypeError, match="SimPlugin"):
            sim.register_plugin(_NotAPlugin)  # type: ignore[arg-type]
        p.disconnect(sim.client)

    def test_lifecycle_order(self):
        """on_init → on_step × N → on_shutdown called in order."""
        sim = _make_sim()
        plugin = LifecyclePlugin(sim)
        sim.register_plugin(plugin)

        # on_init is called explicitly
        sim._init_plugins()
        assert plugin.calls == ["on_init"]

        # Step 3 times
        dt = sim._params.timestep
        for _ in range(3):
            sim.step_once()

        assert plugin.calls[0] == "on_init"
        assert len([c for c in plugin.calls if isinstance(c, tuple) and c[0] == "on_step"]) == 3

        # Shutdown
        sim._shutdown_plugins()
        assert plugin.calls[-1] == "on_shutdown"
        p.disconnect(sim.client)

    def test_on_reset_called(self):
        """on_reset is called during sim.reset()."""
        sim = _make_sim()
        plugin = LifecyclePlugin(sim)
        sim.register_plugin(plugin)
        sim._init_plugins()

        sim.reset()
        assert "on_reset" in plugin.calls
        p.disconnect(sim.client)

    def test_plugins_property_returns_copy(self):
        """sim.plugins returns a copy — mutation doesn't affect internals."""
        sim = _make_sim()
        plugin = LifecyclePlugin(sim)
        sim.register_plugin(plugin)
        plugins_copy = sim.plugins
        plugins_copy.clear()
        assert len(sim.plugins) == 1  # original unchanged
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Frequency control
# ---------------------------------------------------------------------------


class TestPluginFrequency:
    """Tests for plugin step frequency control."""

    def test_no_frequency_calls_every_step(self):
        """Plugin with frequency=None is called every step."""
        sim = _make_sim()
        plugin = CounterPlugin(sim)
        sim.register_plugin(plugin, frequency=None)

        for _ in range(10):
            sim.step_once()

        assert plugin.step_count == 10
        p.disconnect(sim.client)

    def test_frequency_throttles_calls(self):
        """Plugin with frequency=Hz is called approximately at that rate."""
        sim = _make_sim()
        plugin = CounterPlugin(sim)
        # Default timestep = 0.1s, frequency = 2 Hz → interval = 0.5s → every 5 steps
        sim.register_plugin(plugin, frequency=2.0)

        for _ in range(50):
            sim.step_once()

        # 50 steps × 0.1s = 5.0s at 2 Hz → expect ~10 calls
        assert 9 <= plugin.step_count <= 11
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Error isolation
# ---------------------------------------------------------------------------


class TestPluginErrorIsolation:
    """on_step errors are caught; simulation continues."""

    def test_step_error_logged_and_isolated(self, caplog):
        """Exception in on_step is logged, sim continues stepping."""
        sim = _make_sim()
        error_plugin = ErrorPlugin(sim)
        counter_plugin = CounterPlugin(sim)
        sim.register_plugin(error_plugin)
        sim.register_plugin(counter_plugin)

        with caplog.at_level(logging.ERROR, logger="pybullet_fleet.core_simulation"):
            for _ in range(5):
                sim.step_once()

        # Counter plugin still ran despite error plugin crashing
        assert counter_plugin.step_count == 5
        # Error was logged (check message text and exc_info)
        assert any("on_step() failed" in r.message for r in caplog.records)
        assert any(r.exc_info is not None and r.exc_info[1] is not None for r in caplog.records)
        p.disconnect(sim.client)

    def test_shutdown_error_isolated(self, caplog):
        """Exception in on_shutdown is caught; other plugins still shut down."""

        class ShutdownErrorPlugin(SimPlugin):
            def on_init(self):
                pass

            def on_step(self, dt):
                pass

            def on_shutdown(self):
                raise RuntimeError("shutdown boom")

        sim = _make_sim()
        error_plugin = ShutdownErrorPlugin(sim)
        lifecycle_plugin = LifecyclePlugin(sim)
        sim.register_plugin(error_plugin)
        sim.register_plugin(lifecycle_plugin)

        with caplog.at_level(logging.ERROR, logger="pybullet_fleet.core_simulation"):
            sim._shutdown_plugins()

        assert "on_shutdown" in lifecycle_plugin.calls
        assert any("on_shutdown() failed" in r.message for r in caplog.records)
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: YAML / config loading
# ---------------------------------------------------------------------------


class TestPluginConfigLoading:
    """Tests for _load_plugins_from_config and from_dict integration."""

    def test_load_valid_plugin(self):
        """_load_plugins_from_config loads a plugin from dotted path."""
        sim = _make_sim()
        configs = [
            {
                "class": "tests.test_sim_plugin.LifecyclePlugin",
                "config": {"key": "value"},
            }
        ]
        plugins = _load_plugins_from_config(configs, sim)
        assert len(plugins) == 1
        assert isinstance(plugins[0], LifecyclePlugin)
        assert plugins[0].config == {"key": "value"}
        assert plugins[0].frequency is None
        assert plugins[0]._frequency_gate.frequency is None
        p.disconnect(sim.client)

    def test_load_with_frequency(self):
        """Frequency from config is set on plugin."""
        sim = _make_sim()
        configs = [
            {
                "class": "tests.test_sim_plugin.CounterPlugin",
                "frequency": 5.0,
            }
        ]
        plugins = _load_plugins_from_config(configs, sim)
        assert plugins[0].frequency == 5.0
        p.disconnect(sim.client)

    def test_load_bad_dotted_path(self):
        """Invalid dotted path raises ValueError."""
        sim = _make_sim()
        with pytest.raises(ValueError, match="dotted Python path"):
            _load_plugins_from_config([{"class": "NoDotsHere"}], sim)
        p.disconnect(sim.client)

    def test_load_missing_module(self):
        """Non-existent module raises ModuleNotFoundError."""
        sim = _make_sim()
        with pytest.raises(ModuleNotFoundError, match="package installed"):
            _load_plugins_from_config([{"class": "nonexistent.module.Plugin"}], sim)
        p.disconnect(sim.client)

    def test_load_missing_class(self):
        """Non-existent class in valid module raises AttributeError."""
        sim = _make_sim()
        with pytest.raises(AttributeError, match="not found"):
            _load_plugins_from_config([{"class": "tests.test_sim_plugin.NoSuchClass"}], sim)
        p.disconnect(sim.client)

    def test_load_not_subclass(self):
        """Class that isn't a SimPlugin subclass raises TypeError."""
        sim = _make_sim()
        with pytest.raises(TypeError, match="SimPlugin subclass"):
            _load_plugins_from_config([{"class": "tests.test_sim_plugin._NotAPlugin"}], sim)
        p.disconnect(sim.client)

    def test_from_dict_with_plugins(self):
        """from_dict processes plugins: section and calls on_init."""
        config = {
            "simulation": {"gui": False, "monitor": False, "physics": False},
            "plugins": [
                {
                    "class": "tests.test_sim_plugin.LifecyclePlugin",
                    "config": {"x": 42},
                }
            ],
        }
        sim = MultiRobotSimulationCore.from_dict(config)
        assert len(sim.plugins) == 1
        plugin = sim.plugins[0]
        assert isinstance(plugin, LifecyclePlugin)
        assert plugin.config == {"x": 42}
        # on_init should have been called by from_dict
        assert "on_init" in plugin.calls
        p.disconnect(sim.client)

    def test_config_passthrough(self):
        """Config dict from YAML arrives in plugin.config intact."""
        sim = _make_sim()
        plugin = sim.register_plugin(ConfigPlugin, config={"nested": {"a": 1}})
        assert plugin.config == {"nested": {"a": 1}}
        p.disconnect(sim.client)


# ---------------------------------------------------------------------------
# Tests: Plugin access to sim_core
# ---------------------------------------------------------------------------


class TestPluginSimCoreAccess:
    """Plugins can access sim_core state."""

    def test_access_agents_in_on_init(self):
        """plugin.sim_core.agents is accessible (and populated) in on_init."""

        class AgentCheckPlugin(SimPlugin):
            def on_init(self):
                self.agent_count = len(self.sim_core.agents)

            def on_step(self, dt):
                pass

        sim = _make_sim()
        plugin = AgentCheckPlugin(sim)
        sim.register_plugin(plugin)
        sim._init_plugins()
        assert plugin.agent_count == 0  # no agents spawned
        p.disconnect(sim.client)

    def test_access_sim_time(self):
        """plugin.sim_core.sim_time updates during on_step."""

        class TimePlugin(SimPlugin):
            def __init__(self, sim_core):
                super().__init__(sim_core)
                self.times: list = []

            def on_init(self):
                pass

            def on_step(self, dt):
                self.times.append(self.sim_core.sim_time)

        sim = _make_sim()
        plugin = TimePlugin(sim)
        sim.register_plugin(plugin)

        for _ in range(5):
            sim.step_once()

        assert len(plugin.times) == 5
        # sim_time should be monotonically increasing
        for i in range(1, len(plugin.times)):
            assert plugin.times[i] > plugin.times[i - 1]
        p.disconnect(sim.client)
