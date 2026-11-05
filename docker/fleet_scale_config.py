#!/usr/bin/env python3
"""Generate a temporary ROS bridge config for fleet scale checks."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import yaml

try:
    from pybullet_fleet_ros.interface_config import FLEET_STATE_QOS_PRESETS
except ModuleNotFoundError:
    _BRIDGE_SOURCE = Path(__file__).resolve().parents[1] / "ros2_bridge/pybullet_fleet_ros"
    if not _BRIDGE_SOURCE.is_dir():
        raise
    sys.path.insert(0, str(_BRIDGE_SOURCE))
    from pybullet_fleet_ros.interface_config import FLEET_STATE_QOS_PRESETS

DEFAULT_TEMPLATE_CANDIDATES = [
    Path(__file__).resolve().parents[1] / "ros2_bridge/pybullet_fleet_ros/config/bridge_fleet_scale.yaml",
    Path("/rmf_demos_ws/install/pybullet_fleet_ros/share/pybullet_fleet_ros/config/bridge_fleet_scale.yaml"),
    Path("/rmf_demos_ws/src/pybullet_fleet_ros/config/bridge_fleet_scale.yaml"),
    Path("/opt/pybullet_fleet/ros2_bridge/pybullet_fleet_ros/config/bridge_fleet_scale.yaml"),
]


def _parse_groups(raw: str) -> set[str]:
    if raw == "all":
        return {"state_publishers", "tf", "command_topics", "services", "actions"}
    if raw == "default":
        return {"state_publishers", "tf", "command_topics"}
    if raw == "none":
        return set()
    valid = {"state_publishers", "tf", "command_topics", "services", "actions"}
    groups = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = groups - valid
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown per-robot group(s): {sorted(unknown)}")
    return groups


def moving_workload_direction(index: int, robot_count: int) -> float:
    """Sign of the x travel of robot ``index`` in the moving RTF workload.

    Rows of the spawn grid alternate direction, so every robot keeps moving
    for the whole measurement while neighbouring rows pass each other and the
    collision broad phase sees a changing set of candidate pairs.
    """
    side = int(math.ceil(math.sqrt(robot_count)))
    return -1.0 if (index // side) % 2 else 1.0


def _default_template() -> Path:
    for path in DEFAULT_TEMPLATE_CANDIDATES:
        if path.exists():
            return path
    candidates = ", ".join(str(path) for path in DEFAULT_TEMPLATE_CANDIDATES)
    raise FileNotFoundError(f"bridge_fleet_scale.yaml not found; checked: {candidates}")


def _state_qos_config(args: argparse.Namespace) -> dict:
    config = {"preset": args.state_qos_preset}
    for key in ("reliability", "history", "depth", "durability"):
        value = getattr(args, f"state_qos_{key}")
        if value is not None:
            config[key] = value
    return config


def generate_bridge_config(
    robot_count: int,
    output_path: Path,
    *,
    template_path: Path,
    gui: bool,
    target_rtf: float,
    interface_mode: str,
    per_robot_groups: set[str],
    robot_model: str,
    transport_probe: bool,
    state_qos: dict,
    manager_count: int,
    enable_time_profiling: bool,
    profiling_interval: int,
    state_plane: str | None = None,
) -> None:
    side = int(math.ceil(math.sqrt(robot_count)))
    fleet_enabled = interface_mode in {"fleet", "hybrid"}
    per_robot_enabled = interface_mode in {"per_robot", "hybrid"}

    with template_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    config.setdefault("simulation", {})
    config["simulation"]["gui"] = gui
    config["simulation"]["target_rtf"] = target_rtf
    config["simulation"]["enable_time_profiling"] = enable_time_profiling
    config["simulation"]["profiling_interval"] = profiling_interval
    if enable_time_profiling:
        # Core profiling reports at INFO level; scale templates otherwise use
        # the quiet production-like default (warn).
        config["simulation"]["log_level"] = "info"

    config["fleet_api"] = {
        "enabled": fleet_enabled,
        "states": fleet_enabled and manager_count == 0,
        "navigate": fleet_enabled and manager_count == 0,
        "joint_command": False,
        "transport_probe": fleet_enabled and manager_count == 0 and transport_probe,
        "state_qos": state_qos,
    }
    if state_plane:
        # The bridge writes the fleet state to shared memory; the state_publisher
        # process (started by test_fleet_scale.sh) publishes /clock and /fleet/states.
        config["state_plane"] = {"enabled": True, "name": "pbf_state", "mode": state_plane}
    config["per_robot_api"] = {
        "enabled": per_robot_enabled,
        "state_publishers": per_robot_enabled and "state_publishers" in per_robot_groups,
        "tf": per_robot_enabled and "tf" in per_robot_groups,
        "command_topics": per_robot_enabled and "command_topics" in per_robot_groups,
        "services": per_robot_enabled and "services" in per_robot_groups,
        "actions": per_robot_enabled and "actions" in per_robot_groups,
    }

    entity = config["entities"][0]
    manager = config["managers"][0]
    _apply_robot_model(entity, manager["fleet_controller"], robot_model)
    if manager_count == 0:
        entity.setdefault("grid", {})
        entity["grid"]["count"] = robot_count
        entity["grid"]["columns"] = side
    else:
        _configure_manager_scale_scenario(
            config,
            entity=entity,
            manager=manager,
            robot_count=robot_count,
            manager_count=manager_count,
            transport_probe=transport_probe,
            state_qos=state_qos,
        )
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _configure_manager_scale_scenario(
    config: dict,
    *,
    entity: dict,
    manager: dict,
    robot_count: int,
    manager_count: int,
    transport_probe: bool,
    state_qos: dict,
) -> None:
    """Generate near-even manager scopes for transport-scale checks."""
    base_count, extra = divmod(robot_count, manager_count)
    manager_side = int(math.ceil(math.sqrt(base_count + bool(extra))))
    manager_grid = int(math.ceil(math.sqrt(manager_count)))
    manager_spacing = manager_side * 2.0 + 10.0
    config["managers"] = []
    config["entities"] = []
    config["fleet_api"]["manager_interfaces"] = []
    for index in range(manager_count):
        name = f"manager_{index:02d}"
        count = base_count + (1 if index < extra else 0)
        manager_entry = copy.deepcopy(manager)
        manager_entry["name"] = name
        config["managers"].append(manager_entry)

        entity_entry = copy.deepcopy(entity)
        entity_entry["name"] = f"robot_manager_{index:02d}"
        entity_entry["manager"] = name
        entity_entry["grid"]["count"] = count
        entity_entry["grid"]["columns"] = int(math.ceil(math.sqrt(count)))
        entity_entry["grid"]["offset"] = [
            float(index % manager_grid) * manager_spacing,
            float(index // manager_grid) * manager_spacing,
            entity_entry["grid"]["offset"][2],
        ]
        config["entities"].append(entity_entry)
        config["fleet_api"]["manager_interfaces"].append(
            {
                "manager": name,
                "states": True,
                "transport_probe": transport_probe,
                "state_qos": copy.deepcopy(state_qos),
            }
        )


def _apply_robot_model(entity: dict, fleet_controller: dict, robot_model: str) -> None:
    if robot_model == "simple_cube":
        entity["urdf_path"] = "robots/simple_cube.urdf"
        fleet_controller["type"] = "batch_omni"
        entity["controller"] = {
            "type": "omni",
            "max_linear_vel": 2.0,
            "max_angular_vel": 3.0,
        }
        entity.setdefault("grid", {})["spacing"] = [2.0, 2.0, 0.0]
        entity["grid"]["offset"] = [0.0, 0.0, 0.05]
    elif robot_model == "mobile_robot":
        entity["urdf_path"] = "robots/mobile_robot.urdf"
        fleet_controller["type"] = "batch_omni"
        entity["controller"] = {
            "type": "omni",
            "max_linear_vel": 2.0,
            "max_linear_accel": 5.0,
        }
        entity.setdefault("grid", {})["spacing"] = [2.0, 2.0, 0.0]
        entity["grid"]["offset"] = [0.0, 0.0, 0.3]
    elif robot_model in {"tb3_burger", "tb3_waffle"}:
        model_name = "turtlebot3_burger" if robot_model == "tb3_burger" else "turtlebot3_waffle"
        entity["urdf_path"] = model_name
        fleet_controller["type"] = "batch_differential"
        entity["controller"] = {
            "type": "differential",
            "max_linear_vel": 0.22,
            "max_linear_accel": 2.5,
            "max_angular_vel": 2.84,
            "max_angular_accel": 10.0,
        }
        entity.setdefault("grid", {})["spacing"] = [1.0, 1.0, 0.0]
        entity["grid"]["offset"] = [0.0, 0.0, 0.01]
    else:
        raise ValueError(f"unknown robot model: {robot_model}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", type=int, default=100, help="Number of robots to spawn")
    parser.add_argument(
        "--manager-count",
        type=int,
        default=0,
        help="Create this many manager-scoped state streams; 0 keeps one global stream",
    )
    parser.add_argument(
        "--robot-model",
        choices=["simple_cube", "mobile_robot", "tb3_burger", "tb3_waffle"],
        default="simple_cube",
        help="Robot model to place in the generated grid",
    )
    parser.add_argument("--gui", action="store_true", help="Set generated bridge config simulation.gui=true")
    parser.add_argument("--target-rtf", type=float, default=0.0, help="Simulation target RTF")
    parser.add_argument(
        "--interface-mode",
        choices=["fleet", "hybrid", "per_robot"],
        default="fleet",
        help="ROS interfaces to create",
    )
    parser.add_argument("--template", type=Path, default=None, help="Base bridge_fleet_scale.yaml template")
    parser.add_argument("--config-out", type=Path, required=True, help="Write generated bridge config to this path")
    parser.add_argument(
        "--per-robot-groups",
        type=_parse_groups,
        default=_parse_groups("default"),
        help="Comma-separated per-robot groups for per_robot/hybrid configs, or default/all/none",
    )
    parser.add_argument("--transport-probe", action="store_true", help="Enable the isolated ROS transport timing probe")
    parser.add_argument(
        "--state-plane",
        choices=["every_step", "latest"],
        default=None,
        help="Publish /fleet/states and /clock from the state_publisher process through shared memory, in this mode",
    )
    parser.add_argument(
        "--enable-time-profiling",
        action="store_true",
        help="Enable core per-step profiling in the generated bridge configuration",
    )
    parser.add_argument(
        "--profiling-interval",
        type=int,
        default=1000,
        help="Steps per core profiling report when --enable-time-profiling is set",
    )
    parser.add_argument(
        "--state-qos-preset",
        choices=sorted(FLEET_STATE_QOS_PRESETS),
        default="fleet_state_reliable",
        help="Evaluated FleetState QoS profile",
    )
    parser.add_argument(
        "--state-qos-reliability",
        choices=["reliable", "best_effort"],
        default=None,
        help="Override QoS reliability for /fleet/states",
    )
    parser.add_argument(
        "--state-qos-history",
        choices=["keep_last", "keep_all"],
        default=None,
        help="Override QoS history policy for /fleet/states",
    )
    parser.add_argument("--state-qos-depth", type=int, default=None, help="Override QoS depth for /fleet/states")
    parser.add_argument(
        "--state-qos-durability",
        choices=["volatile", "transient_local"],
        default=None,
        help="Override QoS durability for /fleet/states",
    )
    args = parser.parse_args()
    if args.state_qos_depth is not None and args.state_qos_depth < 1:
        parser.error("--state-qos-depth must be at least 1")
    if args.manager_count < 0 or args.manager_count > args.robots:
        parser.error("--manager-count must be between 0 and --robots")
    if args.manager_count and args.interface_mode == "per_robot":
        parser.error("--manager-count requires --interface-mode fleet or hybrid")
    if args.profiling_interval < 1:
        parser.error("--profiling-interval must be at least 1")

    template_path = args.template if args.template is not None else _default_template()
    generate_bridge_config(
        args.robots,
        args.config_out,
        template_path=template_path,
        gui=args.gui,
        target_rtf=args.target_rtf,
        interface_mode=args.interface_mode,
        per_robot_groups=args.per_robot_groups,
        robot_model=args.robot_model,
        transport_probe=args.transport_probe,
        state_qos=_state_qos_config(args),
        manager_count=args.manager_count,
        enable_time_profiling=args.enable_time_profiling,
        profiling_interval=args.profiling_interval,
        state_plane=args.state_plane,
    )
    print(f"[config] wrote generated bridge config: {args.config_out} (template={template_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
