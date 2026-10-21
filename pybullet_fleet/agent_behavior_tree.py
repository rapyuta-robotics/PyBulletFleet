"""PyBulletFleet mobile-Agent behavior-tree actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any  # noqa: F401 - Sphinx resolves inherited annotations in this module.
from typing import Protocol, cast
from xml.etree import ElementTree

from pybullet_fleet.behavior_tree_core import (
    ActionHandler,
    BehaviorTree,
    NavigationAdapter,
    NavigationStatus,
    Position3,
    TickResult,
)
from pybullet_fleet._replan import RefusesPaths, refused_until_stopped
from pybullet_fleet.geometry import Pose


class GoalPoseAgent(RefusesPaths, Protocol):
    """PyBulletFleet Agent surface used by the default navigation adapter.

    ``path_rejection`` (from :class:`~pybullet_fleet._replan.RefusesPaths`)
    tells a refusal that lasts only while the agent moves, which is retried.
    """

    is_moving: bool
    goal_pose: Pose | None

    def set_goal_pose(self, goal: Pose) -> bool:
        """Begin moving toward a goal pose; ``False`` if the agent refused it."""


class AgentActionId(str, Enum):
    """Portable Action IDs implemented for mobile agents."""

    NAVIGATE_TO = "NavigateTo"


@dataclass
# Intentionally structural: adapters implement NavigationAdapter without
# inheriting it, so backend packages depend only on the portable contract.
class PyBulletFleetNavigationAdapter:
    """Map portable navigation requests to the existing Agent goal API."""

    agent: GoalPoseAgent

    def start_navigation(self, goal: Position3) -> bool:
        pose = Pose.from_xyz(*goal)
        # False only for a refusal that ends with the current motion; a
        # permanent one counts as started, as before.
        return self.agent.set_goal_pose(pose) is not False or not refused_until_stopped(self.agent, [pose])

    def navigation_status(self) -> NavigationStatus:
        if self.agent.is_moving or self.agent.goal_pose is not None:
            return NavigationStatus.RUNNING
        return NavigationStatus.SUCCEEDED


@dataclass
class AgentBehaviorTree(BehaviorTree):
    """Common behavior-tree actions for mobile PyBulletFleet agents."""

    agent: GoalPoseAgent = field(kw_only=True)
    navigation_adapter: NavigationAdapter | None = field(default=None, kw_only=True)
    _navigation: NavigationAdapter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._navigation = self.navigation_adapter or PyBulletFleetNavigationAdapter(self.agent)

    def _action_handlers(self) -> dict[str | None, ActionHandler]:
        return {AgentActionId.NAVIGATE_TO.value: lambda node, _sim_time: self._navigate_to(node)}

    def _navigate_to(self, node: ElementTree.Element) -> TickResult:
        state = self._state.setdefault(id(node), {})
        if not state.get("started"):
            goal = self._port(node.attrib.get("goal", "{goal}"))
            if not isinstance(goal, (tuple, list)) or len(goal) != 3:
                return TickResult.FAILURE
            if self._navigation.start_navigation(cast(Position3, tuple(goal))) is False:
                # The agent refused the goal while still moving (see
                # Agent.set_path); try again on the next tick.
                return TickResult.RUNNING
            state["started"] = True
            return TickResult.RUNNING
        navigation_status = self._navigation.navigation_status()
        if navigation_status is NavigationStatus.RUNNING:
            return TickResult.RUNNING
        self._reset(node)
        if navigation_status is NavigationStatus.SUCCEEDED:
            return TickResult.SUCCESS
        return TickResult.FAILURE
