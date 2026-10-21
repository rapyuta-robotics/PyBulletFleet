"""One-sided corridor traffic experiment with an external collision response.

Run ``python -m pybullet_fleet.examples.evaluation.fleet_corridor_traffic_failure``.
Omit OUTPUT_DIR to create a unique result directory under the system temporary directory.
The policies are scenario-specific, not PBF collision-response modes.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import pybullet as p

from pybullet_fleet.agent import Agent, AgentSpawnParams
from pybullet_fleet.commands import RobotGoalCommand2D
from pybullet_fleet.core_simulation import CollisionObservation, MultiRobotSimulationCore, SimulationParams
from pybullet_fleet.events import SimEvents
from pybullet_fleet.fleet_api import FleetCommandDispatcher, FleetStateProvider
from pybullet_fleet.geometry import Pose
from pybullet_fleet.sim_object import ShapeParams, SimObject, SimObjectSpawnParams
from pybullet_fleet.states import RobotState2D
from pybullet_fleet.types import CollisionDetectionMethod, CollisionMode


_CORRIDOR_HALF_LENGTH = 0.75
_CORRIDOR_INNER_HALF_WIDTH = 0.17
_WALL_HALF_THICKNESS = 0.05
_ROBOT_Z = 0.1
_FEEDER_Y = (-1.2, -0.6, 0.6, 1.2)
_ENTRANCE_WAYPOINT = (-_CORRIDOR_HALF_LENGTH - 0.45, 0.0)
_EXIT_WAYPOINT = (_CORRIDOR_HALF_LENGTH + 0.45, 0.0)


class RoutePhase(str, Enum):
    ENTRANCE = "entrance"
    EXIT = "exit"
    DESTINATION = "destination"


_NEXT_ROUTE_PHASE = {
    RoutePhase.ENTRANCE: RoutePhase.EXIT,
    RoutePhase.EXIT: RoutePhase.DESTINATION,
}


@dataclass(frozen=True)
class TrafficConfig:
    robots: int = 20
    timestep: float = 0.1
    cutoff: float = 300.0
    speed: float = 1.0
    collision_margin: float = 0.02
    min_block_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.robots < 2 or self.robots % 2:
            raise ValueError("robots must be an even number of at least two")
        for field in ("timestep", "cutoff", "speed", "min_block_seconds"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not math.isfinite(self.collision_margin) or self.collision_margin < 0:
            raise ValueError("collision_margin must be finite and nonnegative")
        if not math.isclose(self.cutoff / self.timestep, round(self.cutoff / self.timestep), abs_tol=1e-8):
            raise ValueError("cutoff must be an integer number of steps")

    @property
    def steps(self) -> int:
        return round(self.cutoff / self.timestep)

    @property
    def block_steps(self) -> int:
        return math.ceil(self.min_block_seconds / self.timestep)


def _workload(config: TrafficConfig) -> list[dict]:
    """Four A-side approaches merge before the entrance, then disperse after the exit."""
    return [
        {
            "task_id": f"r{index:02d}-a2b",
            "robot_id": f"r{index:02d}",
            "start": [-5.0 - 0.35 * (index // 4), _FEEDER_Y[index % 4]],
            "entrance": list(_ENTRANCE_WAYPOINT),
            "exit": list(_EXIT_WAYPOINT),
            "destination": [4.5, 0.5 * (index - (config.robots - 1) / 2)],
        }
        for index in range(config.robots)
    ]


def _make_sim(
    config: TrafficConfig, tasks: list[dict], *, gui: bool = False, monitor_gui: bool = False
) -> tuple[MultiRobotSimulationCore, dict[int, str]]:
    sim = MultiRobotSimulationCore(
        SimulationParams(
            gui=gui,
            monitor=monitor_gui,
            enable_monitor_gui=monitor_gui,
            enable_floor=False,
            physics=False,
            target_rtf=0,
            timestep=config.timestep,
            collision_check_frequency=None,
            collision_detection_method=CollisionDetectionMethod.CLOSEST_POINTS,
            collision_margin=config.collision_margin,
            ignore_static_collision=False,
            log_level="warning",
        )
    )
    names: dict[int, str] = {}
    model = Path(__file__).resolve().parents[2] / "robots" / "simple_cube.urdf"
    with sim.batch_spawn():
        for task in tasks:
            agent = Agent.from_params(
                AgentSpawnParams(
                    name=task["robot_id"],
                    urdf_path=str(model),
                    initial_pose=Pose.from_xyz(task["start"][0], task["start"][1], _ROBOT_Z),
                    mass=0.0,
                    pickable=False,
                    collision_mode=CollisionMode.NORMAL_2D,
                    controller={
                        "type": "omni",
                        "navigation_2d": True,
                        "max_linear_vel": config.speed,
                        "max_linear_accel": 2.0,
                        "max_angular_vel": 2.0,
                        "max_angular_accel": 4.0,
                    },
                ),
                sim,
            )
            names[agent.object_id] = task["robot_id"]
        wall_y = _CORRIDOR_INNER_HALF_WIDTH + _WALL_HALF_THICKNESS
        for side, y in (("north", wall_y), ("south", -wall_y)):
            shape = ShapeParams(shape_type="box", half_extents=[_CORRIDOR_HALF_LENGTH, _WALL_HALF_THICKNESS, 0.25])
            wall = SimObject.from_params(
                SimObjectSpawnParams(
                    name=f"wall-{side}",
                    initial_pose=Pose.from_xyz(0, y, _ROBOT_Z),
                    mass=0.0,
                    pickable=False,
                    visual_shape=shape,
                    collision_shape=shape,
                    collision_mode=CollisionMode.STATIC,
                ),
                sim,
            )
            names[wall.object_id] = f"wall-{side}"
    if gui:
        sim.setup_camera(
            camera_config={
                "camera_mode": "manual",
                "camera_distance": 8.0,
                "camera_yaw": 0,
                "camera_pitch": -89,
                "camera_target": [-1.5, 0, _ROBOT_Z],
            }
        )
    return sim, names


def _components(pairs: set[tuple[str, str]]) -> list[set[str]]:
    """Connected robot conflict groups, independent of pair iteration order."""
    neighbors: dict[str, set[str]] = {}
    for first, second in pairs:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    groups = []
    unseen = set(neighbors)
    while unseen:
        pending = [min(unseen)]
        group: set[str] = set()
        while pending:
            name = pending.pop()
            if name in group:
                continue
            group.add(name)
            pending.extend(neighbors[name] - group)
        unseen.difference_update(group)
        groups.append(group)
    return groups


def _robot_overlaps(
    check: CollisionObservation | None, step: int, names: dict[int, str], goals: dict[str, tuple[float, float]]
) -> set[tuple[str, str]]:
    if check is None or check.step != step:
        return set()
    overlaps = set()
    for record in check.pairs:
        if record.sample_step != step or record.signed_distance is None or record.signed_distance > 0:
            continue
        first, second = (names[object_id] for object_id in record.object_ids)
        if first in goals and second in goals:
            overlaps.add(tuple(sorted((first, second))))
    return overlaps


def _exit_priority(name: str, states: dict[str, RobotState2D]) -> tuple[float, str]:
    """Prioritize the robot nearest the B-side corridor exit plane."""
    return max(_CORRIDOR_HALF_LENGTH - states[name].position[0], 0.0), name


def _hold_final_gui(sim: MultiRobotSimulationCore) -> None:
    """Keep the final frame visible before the core disconnects its client."""
    print("Simulation finished. Close the GUI window or press Ctrl+C to save the report and exit.")
    try:
        while p.isConnected(sim.client):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def run_policy(
    policy: str,
    config: TrafficConfig = TrafficConfig(),
    *,
    gui: bool = False,
    monitor_gui: bool = False,
    view_rtf: float = 1.0,
    hold_gui: bool = False,
) -> dict:
    """Run one fresh world with the selected external app policy."""
    if policy not in ("pass_through", "collision_stop"):
        raise ValueError("policy must be pass_through or collision_stop")
    if gui and (not math.isfinite(view_rtf) or view_rtf <= 0):
        raise ValueError("view_rtf must be finite and positive in GUI mode")
    if hold_gui and not gui:
        raise ValueError("hold_gui requires gui=True")
    if monitor_gui and not gui:
        raise ValueError("monitor_gui requires gui=True")
    tasks = _workload(config)
    goals = {task["robot_id"]: tuple(task["destination"]) for task in tasks}
    route_targets = {
        task["robot_id"]: {
            RoutePhase.ENTRANCE: tuple(task["entrance"]),
            RoutePhase.EXIT: tuple(task["exit"]),
            RoutePhase.DESTINATION: tuple(task["destination"]),
        }
        for task in tasks
    }
    route_phase = {name: RoutePhase.ENTRANCE for name in goals}
    sim, names = _make_sim(config, tasks, gui=gui, monitor_gui=monitor_gui)
    sim.params.target_rtf = view_rtf if gui else 0
    dispatcher = FleetCommandDispatcher(sim, retain_command_events=False)
    provider = FleetStateProvider(sim)
    entered: dict[str, int] = {}
    exited: dict[str, int] = {}
    arrived: dict[str, int] = {}
    blocked: dict[str, int] = {}  # robot -> earliest release step
    block_started: dict[str, int] = {}
    block_intervals: list[dict] = []
    decisions: list[dict] = []
    last_overlap_pairs: set[tuple[str, str]] = set()
    overlap_entries = 0
    overlap_entries_by_zone = {
        "before_entrance": 0,
        "corridor_or_boundary": 0,
        "after_exit": 0,
        "near_destinations": 0,
    }
    wall_overlap_samples = 0
    peak_blocked = 0
    peak_blocked_before_entrance = 0
    peak_blocked_in_corridor = 0
    callback_error: list[Exception] = []
    wall_start = time.perf_counter()

    def issue_goal(name: str, step: int, action: str) -> None:
        goal = route_targets[name][route_phase[name]]
        ack = dispatcher.navigate(
            [RobotGoalCommand2D(name=name, position=goal, z=_ROBOT_Z, command_id=f"{name}-{action}-{step}")],
            source="corridor-traffic-example",
            command_id=f"{name}-{action}-{step}",
        )
        if name not in ack.accepted_names:
            raise RuntimeError(f"{action} rejected for {name}: {dict(ack.rejected)}")
        decisions.append(
            {
                "step": step,
                "robot_id": name,
                "action": action,
                "route_phase": route_phase[name].value,
                "command_id": ack.command_id,
            }
        )

    def advance_route(name: str, state: RobotState2D, step: int, action: str = "next_waypoint") -> bool:
        phase = route_phase[name]
        next_phase = _NEXT_ROUTE_PHASE.get(phase)
        if next_phase is None or state.is_moving or math.dist(state.position, route_targets[name][phase]) > 0.02:
            return False
        route_phase[name] = next_phase
        issue_goal(name, step, action)
        return True

    def resume_blocked(name: str, state: RobotState2D, step: int) -> None:
        if not advance_route(name, state, step, action="resume"):
            issue_goal(name, step, "resume")
        block_intervals.append({"robot_id": name, "start_step": block_started.pop(name), "end_step": step})
        del blocked[name]

    def before_step(**_: object) -> None:
        if callback_error:
            return
        try:
            step = sim.step_count
            if step == 0:
                for name in sorted(goals):
                    issue_goal(name, step, "start")
                return
            states = {state.name: state for state in provider.get_states_2d()}
            for name in sorted(goals):
                if name not in blocked:
                    advance_route(name, states[name], step)
            if policy != "collision_stop" or not blocked:
                return
            check = sim.get_collision_observation()
            if check is None or check.step != step:
                return
            name = min(blocked, key=lambda robot: _exit_priority(robot, states))
            if step >= blocked[name]:
                resume_blocked(name, states[name], step)
        except Exception as exc:
            callback_error.append(exc)

    def after_step(**_: object) -> None:
        nonlocal overlap_entries, wall_overlap_samples, peak_blocked, peak_blocked_before_entrance
        nonlocal peak_blocked_in_corridor, last_overlap_pairs
        if callback_error:
            return
        try:
            step = sim.step_count  # POST_STEP sees the counter of the step it completes.
            states = {state.name: state for state in provider.get_states_2d()}
            for name, state in states.items():
                x, y = state.position
                in_corridor = -_CORRIDOR_HALF_LENGTH <= x <= _CORRIDOR_HALF_LENGTH and abs(y) <= _CORRIDOR_INNER_HALF_WIDTH
                if name not in entered and in_corridor:
                    entered[name] = step
                if (
                    name in entered
                    and name not in exited
                    and x > _CORRIDOR_HALF_LENGTH
                    and abs(y) <= _CORRIDOR_INNER_HALF_WIDTH
                ):
                    exited[name] = step
                if (
                    name not in arrived
                    and name not in blocked
                    and route_phase[name] == RoutePhase.DESTINATION
                    and math.dist(state.position, goals[name]) <= 0.02
                    and not state.is_moving
                ):
                    arrived[name] = step

            check = sim.get_collision_observation()
            overlaps = _robot_overlaps(check, step, names, goals)
            if check is not None and check.step == step:
                for record in check.pairs:
                    if record.sample_step != step or record.signed_distance is None or record.signed_distance > 0:
                        continue
                    first, second = (names[object_id] for object_id in record.object_ids)
                    if first.startswith("wall-") or second.startswith("wall-"):
                        wall_overlap_samples += 1
            new_overlaps = overlaps - last_overlap_pairs
            overlap_entries += len(new_overlaps)
            for first, second in new_overlaps:
                first_x, second_x = states[first].position[0], states[second].position[0]
                if max(first_x, second_x) < -_CORRIDOR_HALF_LENGTH:
                    overlap_entries_by_zone["before_entrance"] += 1
                elif min(first_x, second_x) >= 4.0:
                    overlap_entries_by_zone["near_destinations"] += 1
                elif min(first_x, second_x) > _CORRIDOR_HALF_LENGTH:
                    overlap_entries_by_zone["after_exit"] += 1
                else:
                    overlap_entries_by_zone["corridor_or_boundary"] += 1
            last_overlap_pairs = overlaps
            if policy == "collision_stop":
                # Keep measuring every overlap, but a completed endpoint task
                # cannot be stopped or restarted by this response policy.
                response_pairs = {pair for pair in overlaps if pair[0] not in arrived and pair[1] not in arrived}
                for group in _components(response_pairs):
                    winner = min(group, key=lambda name: _exit_priority(name, states))
                    # A stopped winner must be able to proceed; otherwise this
                    # response can stop every moving member of the group.
                    if winner in blocked:
                        resume_blocked(winner, states[winner], step)
                    for name in sorted(group - {winner}):
                        if name in blocked:
                            continue
                        ack = dispatcher.stop([name], source="corridor-traffic-example", command_id=f"{name}-stop-{step}")
                        if name not in ack.accepted_names:
                            raise RuntimeError(f"stop rejected for {name}: {dict(ack.rejected)}")
                        blocked[name] = step + config.block_steps
                        block_started[name] = step
                        decisions.append(
                            {"step": step, "robot_id": name, "action": "stop", "winner": winner, "command_id": ack.command_id}
                        )
            peak_blocked = max(peak_blocked, len(blocked))
            peak_blocked_before_entrance = max(
                peak_blocked_before_entrance,
                sum(states[name].position[0] < -_CORRIDOR_HALF_LENGTH for name in blocked),
            )
            peak_blocked_in_corridor = max(
                peak_blocked_in_corridor,
                sum(
                    -_CORRIDOR_HALF_LENGTH <= states[name].position[0] <= _CORRIDOR_HALF_LENGTH
                    and abs(states[name].position[1]) <= _CORRIDOR_INNER_HALF_WIDTH
                    for name in blocked
                ),
            )
            if hold_gui and step == config.steps:
                sim.pause()
                _hold_final_gui(sim)
        except Exception as exc:
            callback_error.append(exc)
            sim.params.target_rtf = 0

    sim.events.on(SimEvents.PRE_STEP, before_step)
    sim.events.on(SimEvents.POST_STEP, after_step)
    try:
        sim.run_simulation(duration=config.cutoff)
        if callback_error:
            raise RuntimeError("traffic scenario callback failed") from callback_error[0]
        if sim.step_count != config.steps:
            raise RuntimeError("simulation ended before the fixed cutoff")
        for name in blocked:
            block_intervals.append(
                {"robot_id": name, "start_step": block_started[name], "end_step": sim.step_count, "censored_at_cutoff": True}
            )
        all_passed = len(exited) == len(goals)
        all_arrived = len(arrived) == len(goals)
        return {
            "schema_version": 2,
            "scenario_id": "pbf.one_sided_corridor_traffic.v1",
            "policy": policy,
            "execution": {"gui": gui, "monitor_gui": monitor_gui, "target_view_rtf": view_rtf if gui else None},
            "conditions": {
                **asdict(config),
                "corridor_x": [-_CORRIDOR_HALF_LENGTH, _CORRIDOR_HALF_LENGTH],
                "corridor_inner_y": [-_CORRIDOR_INNER_HALF_WIDTH, _CORRIDOR_INNER_HALF_WIDTH],
            },
            "measurement_contract": {
                "timebase": "simulation steps; seconds = step * timestep",
                "passage": (
                    f"robot reference point crossed x={_CORRIDOR_HALF_LENGTH:g} within corridor y bounds " "after first entry"
                ),
                "overlap": "fresh robot-robot signed closest-point distance <= 0",
                "overlap_zones": (
                    f"before_entrance: both x < {-_CORRIDOR_HALF_LENGTH:g}; "
                    "near_destinations: both x >= 4; "
                    f"after_exit: both x > {_CORRIDOR_HALF_LENGTH:g} but not near_destinations; "
                    "corridor_or_boundary: all other pairs"
                ),
                "response_scope": (
                    "stops apply until endpoint arrival, including after the corridor exit; " "completed tasks are excluded"
                ),
            },
            "tasks": tasks,
            "first_corridor_entry_step": entered,
            "first_corridor_exit_step": exited,
            "endpoint_arrival_step": arrived,
            "decisions": decisions,
            "blocked_intervals": block_intervals,
            "metrics": {
                "all_passed_at_seconds": max(exited.values()) * config.timestep if all_passed else None,
                "passed_count": len(exited),
                "unpassed_count": len(goals) - len(exited),
                "deadlock_at_cutoff": not all_passed,
                "endpoint_completed_count": len(arrived),
                "endpoint_unfinished_count": len(goals) - len(arrived),
                "all_arrived_at_seconds": max(arrived.values()) * config.timestep if all_arrived else None,
                "robot_overlap_entries": overlap_entries,
                "robot_overlap_entries_by_zone": overlap_entries_by_zone,
                "wall_overlap_samples": wall_overlap_samples,
                "stop_count": sum(item["action"] == "stop" for item in decisions),
                "resume_count": sum(item["action"] == "resume" for item in decisions),
                "peak_blocked": peak_blocked,
                "peak_blocked_before_entrance": peak_blocked_before_entrance,
                "peak_blocked_in_corridor": peak_blocked_in_corridor,
                "blocked_robot_seconds": sum(
                    (item["end_step"] - item["start_step"]) * config.timestep for item in block_intervals
                ),
                "simulated_seconds": config.cutoff,
                "wall_seconds": time.perf_counter() - wall_start,
            },
        }
    finally:
        sim.events.off(SimEvents.PRE_STEP, before_step)
        sim.events.off(SimEvents.POST_STEP, after_step)
        if sim.client is not None and p.isConnected(sim.client):
            p.disconnect(sim.client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", help="Result directory; omitted creates a unique temporary one")
    parser.add_argument("--robots", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--cutoff", type=float, default=300.0)
    parser.add_argument("--policy", choices=("both", "pass_through", "collision_stop"), default="both")
    parser.add_argument("--gui", action="store_true", help="Observe one policy in the PyBullet GUI")
    parser.add_argument("--monitor", action="store_true", help="Show the live DataMonitor alongside --gui")
    parser.add_argument("--rtf", type=float, default=1.0, help="GUI target real-time factor (default: 1)")
    args = parser.parse_args()
    if args.gui and args.policy == "both":
        parser.error("--gui requires --policy pass_through or --policy collision_stop")
    if args.monitor and not args.gui:
        parser.error("--monitor requires --gui")
    if args.gui and (not math.isfinite(args.rtf) or args.rtf <= 0):
        parser.error("--rtf must be finite and positive with --gui")
    config = TrafficConfig(robots=args.robots, timestep=args.dt, cutoff=args.cutoff)
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("output directory already exists")
    output_dir = (
        Path(tempfile.mkdtemp(prefix="pbf-traffic-", dir=tempfile.gettempdir()))
        if args.output_dir is None
        else args.output_dir
    )
    policies = ("pass_through", "collision_stop") if args.policy == "both" else (args.policy,)
    reports = [
        run_policy(policy, config, gui=args.gui, monitor_gui=args.monitor, view_rtf=args.rtf, hold_gui=args.gui)
        for policy in policies
    ]
    if args.output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    for report in reports:
        (output_dir / f"{report['policy']}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved reports in {output_dir}")
    print(json.dumps({report["policy"]: report["metrics"] for report in reports}, indent=2))


if __name__ == "__main__":
    main()
