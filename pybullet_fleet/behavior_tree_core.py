"""Backend-neutral BehaviorTree.CPP XML execution primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from random import Random
from typing import Any, Callable, Protocol, TypeAlias
from xml.etree import ElementTree


class BehaviorTreeNodeType(str, Enum):
    """Control and leaf node tags supported by the portable XML profile."""

    SEQUENCE = "Sequence"
    FALLBACK = "Fallback"
    REPEAT = "Repeat"
    ACTION = "Action"


class TickResult(str, Enum):
    """Outcomes returned by a behavior-tree node tick."""

    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class NavigationStatus(str, Enum):
    """Portable lifecycle outcomes for one navigation request."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


Position3: TypeAlias = tuple[float, float, float]


class NavigationAdapter(Protocol):
    """Backend-neutral navigation operations used by agent behavior trees."""

    def start_navigation(self, goal: Position3) -> bool | None:
        """Start navigation toward ``goal`` for this adapter's execution target.

        Return ``False`` when the target refuses the goal for now (for
        example, it is moving and cannot turn toward it within its limits);
        the action then tries again on its next tick. ``None`` or ``True``
        means started.
        """

    def navigation_status(self) -> NavigationStatus:
        """Return the lifecycle state of the currently active navigation."""


ActionHandler = Callable[[ElementTree.Element, float], TickResult]


@dataclass
class BehaviorTree:
    """Execute the common strict BehaviorTree.CPP XML profile.

    This class has no PyBulletFleet dependency. Subclasses provide only their
    supported leaf actions; it remains smaller than the full C++ runtime.
    """

    root: ElementTree.Element
    blackboard: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    tree_id: str = "MainTree"
    _state: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        blackboard: dict[str, Any] | None = None,
        seed: int = 0,
        **subclass_kwargs: Any,
    ) -> "BehaviorTree":
        document = ElementTree.parse(path).getroot()
        if document.tag != "root" or document.attrib.get("BTCPP_format") != "4":
            raise ValueError("Expected a BehaviorTree.CPP v4 <root BTCPP_format='4'> document")
        tree_id = document.attrib.get("main_tree_to_execute")
        trees = {tree.attrib.get("ID"): tree for tree in document.findall("BehaviorTree")}
        if not tree_id:
            tree_id = next(iter(trees), None)
        if not tree_id or tree_id not in trees:
            raise ValueError("Behavior tree document has no executable BehaviorTree")
        children = list(trees[tree_id])
        if len(children) != 1:
            raise ValueError(f"BehaviorTree {tree_id!r} must contain exactly one root node")
        return cls(
            root=children[0],
            blackboard=dict(blackboard or {}),
            seed=seed,
            tree_id=tree_id,
            **subclass_kwargs,
        )

    def tick(self, sim_core: Any, _dt: float) -> None:
        """Simulation callback compatible with ``sim.register_callback``."""
        self._tick(self.root, float(sim_core.sim_time))

    def _tick(self, node: ElementTree.Element, sim_time: float) -> TickResult:
        try:
            node_type = BehaviorTreeNodeType(node.tag)
        except ValueError as error:
            raise ValueError(f"Unsupported BehaviorTree.CPP profile node <{node.tag}>") from error
        if node_type is BehaviorTreeNodeType.SEQUENCE:
            return self._tick_sequence(node, sim_time, fallback=False)
        if node_type is BehaviorTreeNodeType.FALLBACK:
            return self._tick_sequence(node, sim_time, fallback=True)
        if node_type is BehaviorTreeNodeType.REPEAT:
            return self._tick_repeat(node, sim_time)
        if node_type is BehaviorTreeNodeType.ACTION:
            return self._tick_action(node, sim_time)
        raise AssertionError(f"Unhandled behavior tree node type {node_type!r}")

    def _tick_sequence(self, node: ElementTree.Element, sim_time: float, *, fallback: bool) -> TickResult:
        children = list(node)
        if not children:
            raise ValueError(f"<{node.tag}> requires at least one child")
        state = self._state.setdefault(id(node), {"index": 0})
        while state["index"] < len(children):
            child = children[state["index"]]
            result = self._tick(child, sim_time)
            if result is TickResult.RUNNING:
                return result
            if (fallback and result is TickResult.SUCCESS) or (not fallback and result is TickResult.FAILURE):
                self._reset(node)
                return result
            state["index"] += 1
        self._reset(node)
        return TickResult.FAILURE if fallback else TickResult.SUCCESS

    def _tick_repeat(self, node: ElementTree.Element, sim_time: float) -> TickResult:
        children = list(node)
        if len(children) != 1:
            raise ValueError("<Repeat> requires exactly one child")
        cycles = int(node.attrib.get("num_cycles", "-1"))
        state = self._state.setdefault(id(node), {"completed": 0})
        result = self._tick(children[0], sim_time)
        if result is TickResult.RUNNING:
            return result
        if result is TickResult.FAILURE:
            self._reset(node)
            return result
        state["completed"] += 1
        self._reset(children[0])
        if cycles >= 0 and state["completed"] >= cycles:
            self._reset(node)
            return TickResult.SUCCESS
        return TickResult.RUNNING

    def _tick_action(self, node: ElementTree.Element, sim_time: float) -> TickResult:
        action_id = node.attrib.get("ID")
        handler = self._action_handlers().get(action_id)
        if handler is not None:
            return handler(node, sim_time)
        raise ValueError(f"Unsupported BehaviorTree.CPP Action ID {action_id!r}")

    def _action_handlers(self) -> dict[str | None, ActionHandler]:
        """Return leaf-node handlers supported by this tree profile."""
        return {}

    def _port(self, value: str) -> Any:
        if value.startswith("{") and value.endswith("}"):
            return self.blackboard.get(value[1:-1])
        return value

    def _set_port(self, port: str, value: Any) -> None:
        if not (port.startswith("{") and port.endswith("}")):
            raise ValueError("BehaviorTree.CPP output ports must use {blackboard_key} syntax")
        self.blackboard[port[1:-1]] = value

    def _reset(self, node: ElementTree.Element) -> None:
        self._state.pop(id(node), None)
        for child in node:
            self._reset(child)
