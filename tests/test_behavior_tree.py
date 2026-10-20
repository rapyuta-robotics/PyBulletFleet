from pathlib import Path
from types import SimpleNamespace

from xml.etree import ElementTree

from pybullet_fleet.behavior_tree import (
    AgentActionId,
    AgentBehaviorTree,
    BehaviorTree,
    BehaviorTreeNodeType,
    NavigationStatus,
    TickResult,
    WorkerBehaviorTree,
    WorkerActionId,
)
from pybullet_fleet import Agent, MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams
from pybullet_fleet.types import CollisionMode


class _Agent:
    def __init__(self):
        self.is_moving = False
        self.goal_pose = None

    def set_goal_pose(self, goal):
        self.goal_pose = goal
        self.is_moving = True


class _TickOnlyTree:
    def tick(self, _sim_core, _dt):
        pass


class _NavigationAdapter:
    def __init__(self):
        self.goal = None
        self.status = NavigationStatus.RUNNING

    def start_navigation(self, goal):
        self.goal = goal

    def navigation_status(self):
        return self.status


def test_worker_tree_selects_seeded_goal_navigates_and_waits():
    path = Path(__file__).parent / "fixtures" / "worker_wander.xml"
    agent = _Agent()
    tree = WorkerBehaviorTree.from_file(
        path,
        agent=agent,
        waypoint_sets={"aisle": [(1.0, 2.0, 0.0)]},
        blackboard={"worker_waypoint_set": "aisle"},
        seed=7,
    )
    assert isinstance(tree, BehaviorTree)
    assert isinstance(tree, AgentBehaviorTree)
    sim = SimpleNamespace(sim_time=0.0)

    tree.tick(sim, 0.1)
    assert agent.goal_pose.position == [1.0, 2.0, 0.0]
    assert tree.blackboard["goal"] == (1.0, 2.0, 0.0)

    agent.is_moving = False
    agent.goal_pose = None
    sim.sim_time = 0.5
    tree.tick(sim, 0.1)
    sim.sim_time = 1.5
    tree.tick(sim, 0.1)
    assert agent.goal_pose is None


def test_worker_tree_rejects_non_btcpp_document(tmp_path):
    path = tmp_path / "invalid.xml"
    path.write_text("<root><BehaviorTree ID='x' /></root>", encoding="utf-8")

    try:
        WorkerBehaviorTree.from_file(path, agent=_Agent(), waypoint_sets={})
    except ValueError as exc:
        assert "BehaviorTree.CPP v4" in str(exc)
    else:
        raise AssertionError("Expected malformed XML profile to be rejected")


def test_base_tree_loads_without_an_execution_target():
    path = Path(__file__).parent / "fixtures" / "worker_wander.xml"

    tree = BehaviorTree.from_file(path)

    assert tree.tree_id == "WorkerWander"
    assert not hasattr(tree, "agent")


def test_behavior_tree_node_type_is_the_portable_xml_vocabulary():
    assert BehaviorTreeNodeType.SEQUENCE.value == "Sequence"
    assert BehaviorTreeNodeType.ACTION.value == "Action"
    assert TickResult.RUNNING.value == "running"
    assert AgentActionId.NAVIGATE_TO.value == "NavigateTo"
    assert WorkerActionId.SELECT_GOAL_FROM_WAYPOINT_SET.value == "SelectGoalFromWaypointSet"


def test_agent_tree_uses_navigation_adapter_lifecycle():
    adapter = _NavigationAdapter()
    node = ElementTree.fromstring('<Action ID="NavigateTo" goal="{goal}" />')
    tree = AgentBehaviorTree(
        root=node,
        agent=_Agent(),
        blackboard={"goal": (1.0, 2.0, 0.0)},
        navigation_adapter=adapter,
    )

    assert tree._navigate_to(node) is TickResult.RUNNING
    assert adapter.goal == (1.0, 2.0, 0.0)
    adapter.status = NavigationStatus.SUCCEEDED
    assert tree._navigate_to(node) is TickResult.SUCCESS


def test_agent_tree_retries_a_goal_the_agent_refused():
    """start_navigation() returning False (the agent refused the goal while
    moving) keeps the action running and asks again on the next tick."""

    class _RefusingOnce(_NavigationAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def start_navigation(self, goal):
            self.calls += 1
            if self.calls == 1:
                return False
            self.goal = goal
            return True

    adapter = _RefusingOnce()
    node = ElementTree.fromstring('<Action ID="NavigateTo" goal="{goal}" />')
    tree = AgentBehaviorTree(root=node, agent=_Agent(), blackboard={"goal": (1.0, 2.0, 0.0)}, navigation_adapter=adapter)

    assert tree._navigate_to(node) is TickResult.RUNNING
    assert adapter.goal is None
    assert tree._navigate_to(node) is TickResult.RUNNING
    assert adapter.calls == 2 and adapter.goal == (1.0, 2.0, 0.0)
    adapter.status = NavigationStatus.SUCCEEDED
    assert tree._navigate_to(node) is TickResult.SUCCESS


def test_behavior_tree_registration_keeps_runtime_tick_contract():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False))
    tree = _TickOnlyTree()
    try:
        sim.initialize_simulation()
        assert sim.register_behavior_tree(tree) is tree  # type: ignore[arg-type]
    finally:
        import pybullet as p

        p.disconnect(sim.client)


def test_worker_tree_drives_an_agent_on_the_simulation_thread():
    sim = MultiRobotSimulationCore(SimulationParams(gui=False, monitor=False, enable_floor=False, timestep=0.05))
    sim.initialize_simulation()
    try:
        worker = Agent.from_mesh(
            visual_shape=ShapeParams(shape_type="box", half_extents=[0.15, 0.15, 0.4], rgba_color=[1, 0.8, 0, 1]),
            collision_shape=ShapeParams(shape_type="box", half_extents=[0.15, 0.15, 0.4]),
            pose=Pose.from_xyz(0.0, 0.0, 0.4),
            controller={"type": "omni", "max_linear_vel": 1.0},
            collision_mode=CollisionMode.NORMAL_2D,
            sim_core=sim,
        )
        tree = WorkerBehaviorTree.from_file(
            Path(__file__).parent / "fixtures" / "worker_wander.xml",
            agent=worker,
            waypoint_sets={"aisle": [(1.0, 0.0, 0.4)]},
            blackboard={"worker_waypoint_set": "aisle"},
        )
        assert sim.register_behavior_tree(tree) is tree
        assert sim.register_behavior_tree(tree) is tree
        assert sim.behavior_trees == (tree,)
        for _ in range(50):
            sim.step_once()
        assert worker.get_pose().x > 0.8
        sim.remove_object(worker)
        assert sim.behavior_trees == ()
        assert not sim.unregister_behavior_tree(tree)
    finally:
        import pybullet as p

        p.disconnect(sim.client)
