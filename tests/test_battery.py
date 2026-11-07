"""Tests for Agent battery simulation (BatteryPlugin).

Tests cover:
- BatteryPlugin defaults and custom values
- Battery drain while moving
- Battery charge via set_charging()
- Idle drain
- No battery when plugin is absent
- BatteryPlugin from AgentSpawnParams.from_dict()
- Battery SOC clamping [0.0, 1.0]
"""

import pybullet as p
import pybullet_data
import pytest

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.plugins.battery_plugin import BatteryPlugin
from pybullet_fleet.geometry import Pose
from tests.conftest import MockSimCore

MOBILE_URDF = "robots/mobile_robot.urdf"


@pytest.fixture
def pybullet_env():
    """Headless PyBullet DIRECT session."""
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf")
    yield client
    p.disconnect()


@pytest.fixture
def sim_core(pybullet_env):
    """MockSimCore with physics disabled."""
    sc = MockSimCore(physics=False)
    sc._client = pybullet_env
    return sc


def _make_agent(sim_core, **battery_kwargs):
    """Create a mobile agent with optional battery plugin kwargs."""
    params = AgentSpawnParams(
        urdf_path=MOBILE_URDF,
        initial_pose=Pose.from_xyz(0, 0, 0.1),
        controller={"type": "omni", "max_linear_vel": 2.0},
    )
    agent = Agent.from_params(params, sim_core)
    if battery_kwargs:
        agent.add_plugin(BatteryPlugin(agent, **battery_kwargs))
    return agent


# -------------------------------------------------------------------
# BatteryPlugin defaults
# -------------------------------------------------------------------


class TestBatteryPluginDefaults:
    """Tests for BatteryPlugin default attribute values."""

    def test_defaults(self, sim_core):
        agent = _make_agent(sim_core, initial_soc=1.0)
        bp = agent.battery_plugin
        assert bp.soc == 1.0
        assert bp.discharge_rate == 0.001
        assert bp.charge_rate == 0.005
        assert bp.idle_rate == 0.0

    def test_custom_values(self, sim_core):
        agent = _make_agent(
            sim_core,
            initial_soc=0.5,
            discharge_rate=0.01,
            charge_rate=0.02,
            idle_rate=0.001,
        )
        bp = agent.battery_plugin
        assert bp.soc == 0.5
        assert bp.discharge_rate == 0.01
        assert bp.charge_rate == 0.02
        assert bp.idle_rate == 0.001


# -------------------------------------------------------------------
# Agent battery properties
# -------------------------------------------------------------------


class TestAgentBattery:
    """Tests for battery properties on Agent."""

    def test_no_battery_defaults_to_1(self, sim_core):
        """Agent without battery plugin always reports SOC=1.0."""
        agent = _make_agent(sim_core)
        assert agent.battery_soc == 1.0
        assert agent.battery_plugin is None
        assert agent.is_charging is False

    def test_initial_soc(self, sim_core):
        """Agent's initial SOC matches plugin's initial_soc."""
        agent = _make_agent(sim_core, initial_soc=0.75)
        assert agent.battery_soc == pytest.approx(0.75)
        assert agent.battery_plugin is not None

    def test_charging_flag(self, sim_core):
        """set_charging toggles is_charging."""
        agent = _make_agent(sim_core, initial_soc=1.0)
        assert agent.is_charging is False
        agent.set_charging(True)
        assert agent.is_charging is True
        agent.set_charging(False)
        assert agent.is_charging is False

    def test_set_charging_noop_without_plugin(self, sim_core):
        """set_charging is a no-op when no battery plugin."""
        agent = _make_agent(sim_core)
        agent.set_charging(True)
        assert agent.is_charging is False  # unchanged


# -------------------------------------------------------------------
# Battery drain/charge during update
# -------------------------------------------------------------------


class TestBatteryUpdate:
    """Tests for battery drain and charge during Agent.update()."""

    def test_drain_while_moving(self, sim_core):
        """Battery drains when agent is moving."""
        agent = _make_agent(sim_core, initial_soc=1.0, discharge_rate=0.1)

        # Set a goal so the agent starts moving
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0.1))
        sim_core.tick(10)  # 10 steps

        assert agent.battery_soc < 1.0

    def test_no_drain_when_idle_rate_zero(self, sim_core):
        """Battery does not drain when idle and idle_rate=0."""
        agent = _make_agent(sim_core, initial_soc=0.8, discharge_rate=0.1, idle_rate=0.0)
        # Don't set any goal — agent is idle
        sim_core.tick(100)
        assert agent.battery_soc == pytest.approx(0.8)

    def test_idle_drain(self, sim_core):
        """Battery drains when idle with non-zero idle_rate."""
        agent = _make_agent(sim_core, initial_soc=1.0, idle_rate=0.01)
        # Agent is idle — no goal set
        sim_core.tick(10)
        assert agent.battery_soc < 1.0

    def test_charge_increases_soc(self, sim_core):
        """Battery charges when set_charging(True) is active."""
        agent = _make_agent(sim_core, initial_soc=0.5, charge_rate=0.1)
        agent.set_charging(True)
        sim_core.tick(10)
        assert agent.battery_soc > 0.5

    def test_soc_clamp_at_zero(self, sim_core):
        """SOC never goes below 0.0."""
        agent = _make_agent(sim_core, initial_soc=0.001, discharge_rate=1.0)
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0.1))
        sim_core.tick(100)
        assert agent.battery_soc == pytest.approx(0.0)

    def test_soc_clamp_at_one(self, sim_core):
        """SOC never goes above 1.0."""
        agent = _make_agent(sim_core, initial_soc=0.99, charge_rate=1.0)
        agent.set_charging(True)
        sim_core.tick(100)
        assert agent.battery_soc == pytest.approx(1.0)

    def test_no_battery_update_without_plugin(self, sim_core):
        """Agent without battery plugin always stays at 1.0 regardless of movement."""
        agent = _make_agent(sim_core)
        agent.set_goal_pose(Pose.from_xyz(10, 0, 0.1))
        sim_core.tick(100)
        assert agent.battery_soc == 1.0

    def test_charge_rate_accuracy(self, sim_core):
        """Verify linear charge rate over known time period."""
        dt = sim_core._dt  # timestep
        charge_rate = 0.1  # SOC/s
        agent = _make_agent(sim_core, initial_soc=0.5, charge_rate=charge_rate)
        agent.set_charging(True)

        n_steps = 10
        sim_core.tick(n_steps)
        expected_soc = 0.5 + charge_rate * dt * n_steps
        assert agent.battery_soc == pytest.approx(expected_soc, abs=1e-6)

    def test_discharge_rate_accuracy(self, sim_core):
        """Verify linear discharge rate over known time period."""
        dt = sim_core._dt  # timestep
        discharge_rate = 0.1  # SOC/s
        agent = _make_agent(sim_core, initial_soc=1.0, discharge_rate=discharge_rate)
        # Set unreachable goal so agent stays moving for all steps
        agent.set_goal_pose(Pose.from_xyz(1000, 0, 0.1))

        n_steps = 10
        sim_core.tick(n_steps)
        expected_soc = 1.0 - discharge_rate * dt * n_steps
        assert agent.battery_soc == pytest.approx(expected_soc, abs=1e-6)


# -------------------------------------------------------------------
# AgentSpawnParams.from_dict with battery plugin
# -------------------------------------------------------------------


class TestBatteryFromDict:
    """Tests for parsing plugins from YAML-like dicts."""

    def test_from_dict_with_battery_plugin(self, sim_core):
        """Battery plugin configured via plugins list in dict."""
        config = {
            "name": "test_bot",
            "urdf_path": MOBILE_URDF,
            "plugins": [
                {
                    "type": "battery",
                    "config": {
                        "initial_soc": 0.6,
                        "discharge_rate": 0.002,
                        "charge_rate": 0.01,
                        "idle_rate": 0.0001,
                    },
                }
            ],
        }
        params = AgentSpawnParams.from_dict(config)
        assert params.plugins is not None
        agent = Agent.from_params(params, sim_core)
        assert agent.battery_soc == pytest.approx(0.6)
        bp = agent.battery_plugin
        assert bp is not None
        assert bp.discharge_rate == 0.002
        assert bp.charge_rate == 0.01
        assert bp.idle_rate == 0.0001

    def test_from_dict_without_battery(self):
        config = {"name": "test_bot", "urdf_path": MOBILE_URDF}
        params = AgentSpawnParams.from_dict(config)
        assert not params.plugins


# --- the simulation's battery batch -------------------------------------------


def _fleet_with_batteries(n, **battery):
    import pybullet as p

    from pybullet_fleet import Agent, AgentSpawnParams, MultiRobotSimulationCore, Pose, SimulationParams
    from pybullet_fleet.agent_manager import AgentManager

    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, monitor=False, timestep=0.1, log_level="warning")
    )
    manager = AgentManager(sim_core=sim, fleet_controller={"type": "batch_differential"})
    agents = []
    for k in range(n):
        agent = Agent.from_params(
            AgentSpawnParams(
                urdf_path="robots/simple_cube.urdf",
                initial_pose=Pose.from_xyz(float(2 * k), 0.0, 0.0),
                plugins=[{"type": "battery", "config": dict(initial_soc=0.5 + 0.01 * (k % 10), **battery)}],
            ),
            sim_core=sim,
        )
        manager.add_object(agent)
        agents.append(agent)
    return sim, agents, lambda: p.disconnect(sim._client)


def test_batteries_in_a_simulation_are_one_batch_and_their_agents_take_the_fast_path():
    sim, agents, close = _fleet_with_batteries(5)
    try:
        batch = sim._plugin_batches[BatteryPlugin]
        assert batch is not None and len(batch.plugins) == 5
        for agent in agents:
            assert agent._per_agent_plugins == [] and agent.battery_plugin._batch is batch
            assert agent.update(0.1) is False  # nothing of its own to run: the batch updates the battery
    finally:
        close()


def test_the_batch_gives_the_same_soc_as_the_per_agent_update():
    sim, agents, close = _fleet_with_batteries(40, discharge_rate=0.02, charge_rate=0.05, idle_rate=0.004)
    try:
        from pybullet_fleet import Pose

        for k, agent in enumerate(agents):
            if k % 3 == 0:
                agent.set_goal_pose(Pose.from_xyz(agent.get_pose().x + 50.0, 0.0, 0.0))  # moving
            elif k % 3 == 1:
                agent.set_charging(True)
        expected = [agent.battery_soc for agent in agents]
        for _ in range(20):
            sim.step_once()
            for k, agent in enumerate(agents):  # the per-agent rule, applied by hand with the same inputs
                plugin = agent.battery_plugin
                if plugin.is_charging:
                    expected[k] = min(1.0, expected[k] + plugin.charge_rate * 0.1)
                elif agent.is_moving:
                    expected[k] = max(0.0, expected[k] - plugin.discharge_rate * 0.1)
                else:
                    expected[k] = max(0.0, expected[k] - plugin.idle_rate * 0.1)
        assert [agent.battery_soc for agent in agents] == pytest.approx(expected, abs=1e-12)
        assert any(agents[k].is_moving for k in range(0, 40, 3))
    finally:
        close()


def test_a_plugin_write_reaches_the_batch_and_a_removed_plugin_keeps_its_state():
    sim, agents, close = _fleet_with_batteries(4)
    try:
        batch = sim._plugin_batches[BatteryPlugin]
        first = agents[0].battery_plugin
        first.soc = 0.25
        first.set_charging(True)
        assert batch.values[first._row, 3] == 0.25 and batch.values[first._row, 4] == 1.0
        agents[0].remove_plugin(BatteryPlugin)  # the last row moves into row 0
        assert first._batch is None and first.soc == 0.25 and first.is_charging
        assert len(batch.plugins) == 3
        for plugin in batch.plugins:
            assert batch.plugins[plugin._row] is plugin
        sim.remove_object(agents[1])  # its plugin leaves the batch and keeps its state
        plugin = agents[1].battery_plugin
        assert plugin._batch is None and plugin.soc == pytest.approx(0.51)
        assert len(batch.plugins) == 2
        sim.step_once()  # the remaining two still update
    finally:
        close()


def test_a_subclass_with_its_own_update_stays_per_agent():
    class TemperatureBattery(BatteryPlugin):
        def on_update(self, dt):
            self.soc = max(0.0, self.soc - 0.1 * dt)

    import pybullet as p

    from pybullet_fleet import Agent, AgentSpawnParams, MultiRobotSimulationCore, Pose, SimulationParams

    sim = MultiRobotSimulationCore(
        SimulationParams(gui=False, physics=False, monitor=False, timestep=0.1, log_level="warning")
    )
    try:
        agent = Agent.from_params(
            AgentSpawnParams(urdf_path="robots/simple_cube.urdf", initial_pose=Pose.from_xyz(0, 0, 0)), sim_core=sim
        )
        agent.add_plugin(TemperatureBattery(agent, initial_soc=1.0))
        assert sim._plugin_batches[TemperatureBattery] is None
        assert agent._per_agent_plugins == [agent.battery_plugin]
        sim.step_once()
        assert agent.battery_soc == pytest.approx(0.99)
    finally:
        p.disconnect(sim._client)


def test_a_battery_subclass_and_an_agent_that_answers_is_moving_itself_keep_the_per_agent_update():
    sim, agents, close = _fleet_with_batteries(2, discharge_rate=0.1)
    try:
        from pybullet_fleet.agent import Agent

        class Tagged(BatteryPlugin):
            pass  # no override at all, still per-agent: the batch stands in for the plain class only

        tagged = Tagged(agents[0], initial_soc=1.0, discharge_rate=0.1)
        agents[0].remove_plugin(BatteryPlugin)
        agents[0].add_plugin(tagged)
        assert agents[0]._per_agent_plugins == [tagged] and tagged._batch is None

        class Restless(Agent):
            @property
            def is_moving(self) -> bool:
                return True

        restless = Restless.from_params(
            AgentSpawnParams(
                urdf_path="robots/simple_cube.urdf",
                initial_pose=Pose.from_xyz(10.0, 0.0, 0.0),
                plugins=[{"type": "battery", "config": dict(initial_soc=1.0, discharge_rate=0.1)}],
            ),
            sim_core=sim,
        )
        battery = restless.battery_plugin
        assert restless._per_agent_plugins == [battery] and battery._batch is None
        for _ in range(10):
            sim.step_once()
        assert battery.soc == pytest.approx(0.9)  # drained at the discharge rate: its is_moving was asked
        assert agents[1].battery_plugin.soc == pytest.approx(0.51)  # idle (idle_rate 0), batched
    finally:
        close()


def test_a_battery_added_to_a_removed_agent_is_the_agents_own_and_the_batch_still_steps():
    sim, agents, close = _fleet_with_batteries(3)
    try:
        gone = agents[0]
        sim.remove_object(gone)
        late = BatteryPlugin(gone, initial_soc=0.5)
        gone.add_plugin(late)
        assert late._batch is None and late in gone._per_agent_plugins
        sim.step_once()  # the batch of the two remaining batteries steps
        assert len(sim._plugin_batches[BatteryPlugin].plugins) == 2
    finally:
        close()
