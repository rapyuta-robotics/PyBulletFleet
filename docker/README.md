# Docker — ROS 2 Bridge

Docker environment for the [`ros2_bridge/`](../ros2_bridge/) package (ROS 2 Jazzy).
The Dockerfile builds `pybullet_fleet`, `pybullet_fleet_ros`, and `pybullet_fleet_rmf`
inside a single colcon workspace.
Use the [ROS 2 ReadTheDocs section](https://pybulletfleet.readthedocs.io/en/latest/ros2/index.html)
for user-facing tutorials, endpoint semantics, and RMF usage.  This page is the
Docker-specific reference for image setup, GUI forwarding, operations, and
automated checks.

## Build

Docker Compose bind-mounts this checkout, so start from a cloned repository:

```bash
git clone https://github.com/yuokamoto/PyBulletFleet.git
cd PyBulletFleet/docker
```

Build the bridge image before running the `docker compose run` examples:

```bash
docker compose build bridge
```

## Quick Start

All demos use **two terminals**: Terminal 1 runs the bridge in foreground,
Terminal 2 sends commands via `docker exec`.

### Navigation Demo — TurtleBot3 (Differential Drive + RViz)

Uses `ros-jazzy-turtlebot3-description` (installed in the Docker image).

**Terminal 1** — launch the bridge with PyBullet GUI + RViz:

```bash
cd docker
xhost +local:docker   # first time only

docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros tb3_demo.launch.py gui:=true
```

**Terminal 2** — send a navigation goal through the ROS API
(`/tb3_0/goal_pose`, `geometry_msgs/PoseStamped`):

```bash
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic pub --once /tb3_0/goal_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 1.0, z: 0.01}, orientation: {w: 1.0}}}"'
```

The TurtleBot3 rotates toward the goal and drives forward. Both PyBullet GUI and
RViz (Odometry arrow, Path, Current Goal, RobotModel) update in real time.

You can also use the Waffle model: `model:=waffle`.

Stop with **Ctrl+C** in Terminal 1.

### Fleet API Navigation Demo — Batched Goals Before RMF

This is the bridge-level fleet API path without RMF. It uses the same
`fleet_demo.launch.py` installed for the native Jazzy APT preview, with Docker
providing the image and display forwarding. The launch enables only fleet-level
ROS interfaces, then accepts one batched navigation request at `/fleet/navigate`.

**Terminal 1** — launch the fleet API demo:

```bash
cd docker
xhost +local:docker

docker compose run --rm --name pbf_bridge \
  bridge ros2 launch pybullet_fleet_ros fleet_demo.launch.py \
    robots:=100 robot_model:=simple_cube gui:=true target_rtf:=1.0
```

Change `robot_model:=simple_cube` to `mobile_robot`, `tb3_burger`, or
`tb3_waffle` to try the same fleet API path with another robot model.
For a lower-overhead RViz view instead of the PyBullet GUI, use
`gui:=false rviz:=true`; the launch renders lightweight fleet markers from the
single `/fleet/states` stream.

**Terminal 2** — inspect fleet state and send batched goals through the ROS API.
The helper script is a thin ROS client: it reads `/fleet/states`, selects the
first N robots, and sends one `/fleet/navigate` service request.

```bash
# Fleet state topic: one message contains all robot states
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic echo /fleet/states --once'

# Fleet navigation service: one request contains multiple RobotGoal2D commands
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 -m pybullet_fleet_ros.fleet_nav_demo \
    --robots 10 --dx 0.5 --dy 0.5 --transport service'
```

Use `--transport topic` to publish the same command to the `/fleet/navigate`
topic instead of calling the service.

### Presentation: one direct 10-robot service call

For a live explanation of the Fleet ROS API, the following command shows the
full request without a helper script. It targets the first ten robots from the
default `simple_cube` grid, shifting each by `(+0.5, +0.5)` while retaining the
2 m spacing. Start Terminal 1 with `target_rtf:=1.0` so the movement is visible.

```bash
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 service call /fleet/navigate pybullet_fleet_msgs/srv/FleetNavigate \
    "{command_id: presentation_grid_shift, source: docker_readme, goals_2d: [
      {name: robot_0, position: [0.5, 0.5], yaw: 0.0, z: 0.05},
      {name: robot_1, position: [2.5, 0.5], yaw: 0.0, z: 0.05},
      {name: robot_2, position: [4.5, 0.5], yaw: 0.0, z: 0.05},
      {name: robot_3, position: [6.5, 0.5], yaw: 0.0, z: 0.05},
      {name: robot_4, position: [0.5, 2.5], yaw: 0.0, z: 0.05},
      {name: robot_5, position: [2.5, 2.5], yaw: 0.0, z: 0.05},
      {name: robot_6, position: [4.5, 2.5], yaw: 0.0, z: 0.05},
      {name: robot_7, position: [6.5, 2.5], yaw: 0.0, z: 0.05},
      {name: robot_8, position: [0.5, 4.5], yaw: 0.0, z: 0.05},
      {name: robot_9, position: [2.5, 4.5], yaw: 0.0, z: 0.05}
    ], goals_3d: []}"'
```

For scale/performance checks, use `test_fleet_scale.sh` in the automated test
section. It exercises the same `/fleet/states` and `/fleet/navigate` interfaces
headlessly and can switch between fleet, per-robot, and hybrid interface modes.

### Arm Demo — UR5e (Fixed-Base 6-DOF Arm + RViz)

Uses `ros-jazzy-ur-description` (installed in the Docker image).

**Terminal 1**:

```bash
cd docker
xhost +local:docker

docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros ur5e_demo.launch.py gui:=true
```

**Terminal 2** — send a joint trajectory through the ROS API
(`/ur5e_0/follow_joint_trajectory`, `control_msgs/FollowJointTrajectory`
action):

```bash
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/send_joint_goal_ur5e.py --robot ur5e_0 \
    --positions 0.0 -1.57 1.57 -1.57 -1.57 0.0'
```

RViz shows the UR5e arm model updating as all 6 joints move to target positions.

### Headless / No PyBullet GUI

Pass `gui:=false` to any demo launch to skip the PyBullet GUI window. RViz is
independent (`rviz:=true|false`), so you can keep the RViz view while PyBullet
runs headless, or disable both for a server:

```bash
cd docker
# No PyBullet GUI, RViz on (default)
docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros tb3_demo.launch.py gui:=false

# Fully headless (no PyBullet GUI, no RViz)
docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros tb3_demo.launch.py gui:=false rviz:=false
```

To run a full **RMF demo** headless instead, disable GUI + RViz on the default
`docker compose up` path:

```bash
GUI=false RVIZ=false docker compose up bridge                   # office, headless
DEMO_WORLD=hotel GUI=false RVIZ=false docker compose up bridge  # another world
```

> **Monitor window:** the bridge also opens a small tkinter "Simulation Monitor"
> window. It follows `gui` by default — `gui:=false` hides it too. Force it
> on/off independently with `enable_monitor_gui:=true` / `enable_monitor_gui:=false`
> (an explicit value, or YAML `simulation.enable_monitor_gui`, always wins).

### Launch Arguments

The `tb3_demo.launch.py` and `ur5e_demo.launch.py` accept:

| Argument | Default | Description |
|----------|---------|-------------|
| `gui` | `false` | Enable PyBullet GUI |
| `rviz` | `true` | Launch RViz + robot_state_publisher |
| `enable_monitor_gui` | _(follows `gui`)_ | Show the tkinter monitor window (`false` hides it even with `gui:=true`) |
| `target_rtf` | `1.0` | Real-time factor |
| `publish_rate` | `50.0` | odom / joint_states publish rate (Hz) |

```bash
# GUI only, no RViz
docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros tb3_demo.launch.py gui:=true rviz:=false

# Waffle variant
docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros tb3_demo.launch.py model:=waffle gui:=true
```

### Attach/Detach Demo — Pick and Place Objects

Spawns a TurtleBot3 + 3 pickable boxes. Navigate close to a box and attach it.

**Terminal 1** — launch the attach demo:

```bash
cd docker
xhost +local:docker

docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros attach_demo.launch.py gui:=true
```

**Terminal 2** — navigate to box_A and attach it. The helper scripts call the
same ROS APIs exposed by the bridge:

| Helper | ROS API |
|--------|---------|
| `send_nav_goal.py` | `/<robot>/navigate_to_pose` action |
| `attach_object.py --attach/--detach` | `/<robot>/attach_object` service |
| `attach_object.py --toggle` | `/<robot>/toggle_attach` service |

```bash
# Navigate close to box_A (at x=0.5)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/send_nav_goal.py --robot tb3_0 --x 0.3 --y 0.0'

# Wait, then attach nearest object
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --attach'

# Drive with the object attached
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/send_nav_goal.py --robot tb3_0 --x 2.0 --y 2.0'

# Detach the object at the new location
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --detach'
```

### Environment Variables (docker compose)

These control the default `docker compose up bridge` path, which launches the
RMF demo `${DEMO_WORLD}_pybullet.launch.py` (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_WORLD` | `office` | RMF demo world: `office`, `hotel`, `clinic`, `airport_terminal`, `battle_royale`, `campus` |
| `GUI` | `true` | PyBullet GUI window (`false` = headless) |
| `RVIZ` | `true` | Launch RViz (`false` = headless, saves CPU/GPU) |
| `ROS_DOMAIN_ID` | `42` | ROS 2 domain ID (keep all containers on the same id) |

> **Note:** Source code is volume-mounted, so local Python edits take effect on container restart.

### Config files

Each launch file picks up a **bridge config** (a PyBulletFleet world YAML that
defines the robots, world, and meshes). Standalone ROS demos resolve theirs
automatically; `bridge.launch.py` takes one via `config_yaml:=`.

| Launch | Bridge config |
|--------|---------------|
| `bridge.launch.py` | `config_yaml:=` argument (required) — pass any path below |
| `nav_demo.launch.py` | `pybullet_fleet_ros/config/bridge_nav.yaml` |
| `tb3_demo.launch.py` | `bridge_tb3_burger.yaml` / `bridge_tb3_waffle.yaml` (by `model`) |
| `ur5e_demo.launch.py` | `pybullet_fleet_ros/config/bridge_ur5e.yaml` |
| `arm_demo.launch.py` | `pybullet_fleet_ros/config/bridge_arm.yaml` |
| `attach_demo.launch.py` | `pybullet_fleet_ros/config/bridge_attach_demo.yaml` |
| `${DEMO_WORLD}_pybullet.launch.py` (RMF) | `pybullet_fleet_rmf/config/bridge_<world>.yaml` + an `rmf_demos` fleet config |

RMF worlds map to `bridge_<world>.yaml` as: `office`→`bridge_office`,
`hotel`→`bridge_hotel`, `clinic`→`bridge_clinic`,
`airport_terminal`→`bridge_airport`, `battle_royale`→`bridge_battle_royale`,
`campus`→`bridge_campus`.

Inside the container these live under
`/rmf_demos_ws/install/<pkg>/share/<pkg>/config/` (volume-mounted from
`ros2_bridge/<pkg>/config/`, so edits apply on restart).

## ROS Interface Reference

### Topics (per robot, e.g. `/robot0/...`)

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `cmd_vel` | Twist | Sub | Velocity command (body frame) |
| `goal_pose` | PoseStamped | Sub | Goal pose → robot navigates |
| `path` | Path | Sub | Waypoint path → robot follows |
| `joint_trajectory` | JointTrajectory | Sub | Joint targets (topic) |
| `joint_commands` | Float64MultiArray | Sub | Raw joint positions |
| `odom` | Odometry | Pub | Odometry |
| `joint_states` | JointState | Pub | Joint positions/velocities |
| `plan` | Path | Pub | Remaining path being followed |
| `current_goal` | PoseStamped | Pub | Current goal (arrow points toward target) |
| `diagnostics` | DiagnosticArray | Pub | Robot status (is_moving, action_queue, etc.) |

### Action Servers (per robot)

| Action | Type | Description |
|--------|------|-------------|
| `navigate_to_pose` | NavigateToPose | Nav2-compatible navigation |
| `follow_path` | FollowPath | Nav2-compatible path following |
| `follow_joint_trajectory` | FollowJointTrajectory | MoveIt-compatible joint control |

### Services (per robot)

| Service | Type | Description |
|---------|------|-------------|
| `toggle_attach` | SetBool | rmf_demos cart delivery compat — `true`=attach nearest, `false`=detach first |
| `attach_object` | AttachObject | Detailed attach/detach — object name, parent link, offset, search radius |

### Simulation Services

| Service | Description |
|---------|-------------|
| `/sim/get_entities` | List all entities |
| `/sim/get_entity_state` | Get entity pose |
| `/sim/spawn_entity` | Spawn a robot |
| `/sim/delete_entity` | Delete an entity |
| `/sim/step_simulation` | Step N simulation ticks |
| `/sim/set_simulation_state` | Pause / resume |
| `/sim/get_simulator_features` | Query supported features |

## Operations Guide

While the bridge is running in Terminal 1, use Terminal 2 for interaction.

> Helper scripts in `ros2_bridge/scripts/` are volume-mounted at `/opt/pybullet_fleet/scripts/`.
> They are thin ROS clients, not direct PyBulletFleet calls. Use them for
> copy/paste-safe examples while still exercising the bridge's ROS topics,
> services, and actions.

### Introspection

```bash
# List topics / services / actions
docker exec pbf_bridge bash -c "source /rmf_demos_ws/install/setup.bash && ros2 topic list"
docker exec pbf_bridge bash -c "source /rmf_demos_ws/install/setup.bash && ros2 action list"

# Check odometry
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic echo /tb3_0/odom --once'

# Check diagnostics
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic echo /tb3_0/diagnostics --once'

# List entities
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/query_entities.py'
```

### Navigation (goal_pose / path / action)

```bash
# Send goal pose (topic)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic pub --once /tb3_0/goal_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 1.0, z: 0.01}, orientation: {w: 1.0}}}"'

# Send square path (script -> /tb3_0/path topic)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/send_path.py --robot tb3_0 --size 2.0'

# NavigateToPose action (script -> /tb3_0/navigate_to_pose action, with feedback)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/send_nav_goal.py --robot tb3_0 --x 2.0 --y 1.0'

# Check remaining path
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic echo /tb3_0/plan --once'
```

> **RViz tip:** Use the "2D Goal Pose" toolbar button to click-publish goals directly.

### Arm Control (joint_trajectory / action)

```bash
# FollowJointTrajectory action (script -> /ur5e_0/follow_joint_trajectory action;
# with actual/desired/error feedback — recommended)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/send_joint_goal_ur5e.py --robot ur5e_0 \
    --positions 0.0 -1.57 1.57 -1.57 -1.57 0.0'

# Joint trajectory topic (script -> /ur5e_0/joint_trajectory topic; fire-and-forget)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/send_joint_trajectory.py --robot ur5e_0 \
    --positions 0.0 -1.57 1.57 -1.57 -1.57 0.0'

# Check joint states
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic echo /ur5e_0/joint_states --once'
```

### Velocity Control

```bash
# Script -> /tb3_0/cmd_vel topic, repeated for 5 seconds
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/teleop_cmd_vel.py --robot tb3_0 --vx 0.2 --wz 0.5 --duration 5.0'

# Raw topic pub (single)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic pub --once /tb3_0/cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.22}, angular: {z: 0.5}}"'

# Keyboard teleop
docker exec -it pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -r /cmd_vel:=/tb3_0/cmd_vel'
```

### Attach / Detach Objects

Requires pickable objects in the simulation. Use the **attach demo** config
which spawns a robot + pickable boxes:

```bash
# Terminal 1 — launch attach demo
docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros attach_demo.launch.py gui:=true
```

```bash
# Navigate close to box_A (at 0.5, 0)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/send_nav_goal.py --robot tb3_0 --x 0.3 --y 0.0'

# Attach nearest pickable object (script -> /tb3_0/attach_object service)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --attach'

# Attach specific object by name
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --attach --object box_A'

# Attach to a specific link with offset
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --attach \
    --object box_A --parent-link base_link --offset 0 0 0.2'

# Attach with larger search radius
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --attach --search-radius 3.0'

# Detach (first attached object)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --detach'

# Detach specific object by name
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --detach --object box_A'
```

Alternatively, use `toggle_attach` (rmf_demos compatible SetBool service):

```bash
# Attach nearest (toggle mode, script -> /tb3_0/toggle_attach service)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/attach_object.py --robot tb3_0 --attach --toggle'

# Raw service call (without script)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 service call /tb3_0/toggle_attach std_srvs/srv/SetBool "{data: true}"'

# Detach via raw service call
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 service call /tb3_0/toggle_attach std_srvs/srv/SetBool "{data: false}"'
```

### Spawn / Delete Entities

```bash
# Spawn a UR5e arm at position (5, 3)
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/spawn_robots.py --name ur5e_1 --urdf ur5e --x 5.0 --y 3.0'

# Delete
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 service call /sim/delete_entity simulation_interfaces/srv/DeleteEntity "{entity: ur5e_1}"'
```

### Simulation Control

```bash
# Step 100 ticks
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/step_simulation.py --steps 100'

# Pause / Resume / Status
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/step_simulation.py --pause'
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  python3 /opt/pybullet_fleet/scripts/step_simulation.py --resume'
```

## Config-Driven Launch

Use `config_yaml` to spawn heterogeneous robots:

```bash
# Omni demo: 2 mobile robots + 1 floating cube (6-DoF)
docker compose run --rm --name pbf_bridge bridge \
  ros2 run pybullet_fleet_ros bridge_node \
    --ros-args \
    -p config_yaml:=/rmf_demos_ws/src/pybullet_fleet_ros/config/bridge_omni_demo.yaml \
    -p gui:=true
```

The cube responds to full 6-DoF `cmd_vel` (including `angular.x` / `angular.y` for roll/pitch):

```bash
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic pub -r 10 /cube0/cmd_vel geometry_msgs/msg/Twist \
    "{linear: {z: 0.5}, angular: {x: 0.3, y: 0.2, z: 0.1}}"'
```

### Provided Config Files

| File | Description |
|------|-------------|
| `config/bridge_tb3_burger.yaml` / `config/bridge_tb3_waffle.yaml` | TurtleBot3 navigation (used by `tb3_demo.launch.py`, select via `model:=burger|waffle`) |
| `config/bridge_ur5e.yaml` | UR5e arm joint control (used by `ur5e_demo.launch.py`) |
| `config/bridge_nav.yaml` | Generic differential drive navigation (used by `nav_demo.launch.py`) |
| `config/bridge_arm.yaml` | Generic arm robot joint control (used by `arm_demo.launch.py`) |
| `config/bridge_omni_demo.yaml` | Omni + 6-DoF cube demo (2 mobile robots + 1 floating cube) |
| `config/bridge_attach_demo.yaml` | Attach/detach demo — robot + pickable boxes (used by `attach_demo.launch.py`) |
| `config/bridge_test.yaml` | 3 mobile robots for the bridge API check |

The bridge is **`config_yaml`-driven**: pass a config to spawn robots (there is no
`num_robots`/`robot_urdf` fallback). Without `config_yaml` it starts with no robots.

## All ROS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_yaml` | str | `""` | Path to a PyBulletFleet world YAML (defines the robots/world) |
| `gui` | bool | false | Enable PyBullet GUI |
| `enable_monitor_gui` | bool | _(follows `gui`)_ | Show the tkinter monitor window (explicit value or YAML wins) |
| `physics` | bool | false | Enable physics simulation |
| `publish_rate` | float | 50.0 | odom / joint_states publish rate (Hz) |
| `target_rtf` | float | 1.0 | Real-time factor |
| `enable_sim_services` | bool | true | Enable simulation_interfaces services |

## Automated Tests

The Docker image copies these test scripts under `/opt/pybullet_fleet/docker/`.
The examples below mount the local `docker/` directory to `/docker:ro` so script
edits can be tested without rebuilding the image. Omit the mount and run the
`/opt/pybullet_fleet/docker/...` scripts when you want to test only the image
contents.

```bash
cd docker

# Bridge API check (headless) — spawns 3 robots from bridge_test.yaml and
# checks topics, services, cmd_vel, simulation_interfaces, and fleet API calls.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_bridge_api.sh

# Optional fleet API scale check. The shell wrapper derives a temporary config
# from pybullet_fleet_ros/config/bridge_fleet_scale.yaml, updates the grid count,
# starts bridge_node, then the Python checker sends one /fleet/navigate request
# for all robots and waits until the commanded robots start moving.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 100 --interface-mode fleet \
    --command-interface fleet --publish-rate 5

# Optional per-robot comparison path. This creates per-robot command topics and
# publishes the same goal pattern to /robot_N/goal_pose, then verifies motion via
# simulation_interfaces state queries.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 100 --interface-mode per_robot \
    --command-interface per_robot --publish-rate 5

# Optional side-by-side command path check. Hybrid mode creates both fleet and
# per-robot command interfaces, then the checker exercises both on one bridge
# run. This is a measurement/debug pattern; normal demos should use matching
# interface and command paths.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 100 --interface-mode hybrid \
    --command-interface all --publish-rate 5 --target-rtf 0.0

# Optional RMF client-mode matrix. The default matrix runs office and hotel in
# per_robot_ros, fleet_ros, and python_fleet modes. Add --full for readiness
# checks across the remaining RMF demos.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_rmf_client_modes.sh

docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_rmf_client_modes.sh --full

# Fast RMF stack check. This launches office headless, verifies RMF protocol
# topics/handlers are up, confirms the expected robots are registered, and sends
# a direct NavigateToPose goal through the selected client mode.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_rmf_stack.sh --client-mode python_fleet

# RMF dispatch flow check. This launches one RMF demo and requires dispatched
# tasks to reach explicit successful terminal state by default.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_rmf_dispatch.sh --client-mode python_fleet office_pybullet \
    "patrol:lounge" \
    "delivery:pantry,coke_dispenser,hardware_2,coke_ingestor"

# Hybrid debug: enable only selected per-robot interface groups to isolate which
# ROS entities add latency. Valid groups are state_publishers, tf, command_topics,
# services, and actions. Use default (state_publishers,tf,command_topics), all,
# or none as shorthands. Here --command-interface selects the checker path, not
# which interfaces the bridge exposes.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 1000 --interface-mode hybrid \
    --command-interface fleet --per-robot-groups command_topics --publish-rate 5

# Skip motion verification when measuring only endpoint readiness and command
# send/ack overhead.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 1000 --interface-mode fleet \
    --command-interface fleet --no-verify-motion

# Enable the separate transport timing probe and measure same-host wall-clock
# timing for fleet state and navigation paths during a 10-second window. Repeat
# ten full-fleet service requests and measure observed /clock RTF in the same run.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 1000 --interface-mode fleet \
    --command-interface fleet --publish-rate 5 --measure-transport --measure-rtf \
    --fleet-service-repeats 10

# Measure the asynchronous topic command path. With transport measurement, the
# checker waits for each bridge callback completion before sending the next
# full-fleet command, so this does not intentionally fill the topic queue.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 1000 --interface-mode fleet \
    --command-interface fleet_topic --publish-rate 5 --measure-transport --measure-rtf \
    --fleet-topic-repeats 10

# Measure the maximum RTF with every robot moving instead of an idle fleet. The
# checker sends each robot a goal 500 m away (--moving-distance) right before
# the RTF window and fails if fewer than 95% of the robots are moving during it.
# Add --state-plane every_step|latest to publish the state from a second process.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 1000 --interface-mode fleet \
    --publish-rate 10 --target-rtf 0 --measure-rtf --rtf-workload moving

# The same workload without ROS, for the core-only comparison.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash -c 'python3 /docker/fleet_scale_config.py --robots 1000 --target-rtf 0 \
    --config-out /tmp/scale.yaml && python3 /docker/core_scale_rtf.py /tmp/scale.yaml \
    --duration 10 --moving-distance 500'

# Compare manager-scoped state streams. "selective" subscribes to one manager;
# "complete" subscribes to all of them. These are state-only checks, so the
# helper suppresses its normal navigation command automatically.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 1000 --interface-mode fleet \
    --manager-count 10 --manager-subscription-mode selective --publish-rate 5 \
    --target-rtf 0 --measure-transport --measure-rtf --no-verify-motion

# Use one checker process per manager to separate client-side fan-out from the
# bridge's single-process publication cost.
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  bridge bash /docker/test_fleet_scale.sh --robots 1000 --interface-mode fleet \
    --manager-count 10 --manager-subscription-mode distributed --publish-rate 5 \
    --target-rtf 0 --measure-transport --measure-rtf --no-verify-motion

# To inspect the generated bridge config:
docker compose run --rm --no-deps \
  -v "$(pwd):/docker:ro" \
  -v "$(pwd):/work" \
  bridge bash /docker/test_fleet_scale.sh --robots 10 --config-out /work/bridge_fleet_scale.yaml \
    --generate-only

# colcon launch_testing
docker compose run --rm bridge bash -c "\
  source /ros_entrypoint.sh && \
  cd /rmf_demos_ws && \
  colcon test --packages-select pybullet_fleet_ros --event-handlers console_direct+"
```

## Manual Bridge Smoke Test

**Terminal 1**:

```bash
cd docker
xhost +local:docker
docker compose build bridge

docker compose run --rm --name pbf_bridge bridge \
  ros2 launch pybullet_fleet_ros tb3_demo.launch.py gui:=true
```

**Terminal 2**:

```bash
# Verify topics and actions
docker exec pbf_bridge bash -c "source /rmf_demos_ws/install/setup.bash && ros2 topic list"
docker exec pbf_bridge bash -c "source /rmf_demos_ws/install/setup.bash && ros2 action list"

# Send goal and check diagnostics
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic pub --once /tb3_0/goal_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 1.0, z: 0.01}, orientation: {w: 1.0}}}"'
sleep 2
docker exec pbf_bridge bash -c 'source /rmf_demos_ws/install/setup.bash && \
  ros2 topic echo /tb3_0/diagnostics --once'
# Expected: is_moving=true

# Stop with Ctrl+C in Terminal 1
```

## Troubleshooting

### Login Screen / Stale Volume Data

If the bridge container shows a login screen or behaves unexpectedly after
image rebuilds, stale Docker volumes (Fuel cache, colcon build artifacts) may
be the cause.  Clean up with:

```bash
docker system prune --volumes
```

### `Error in gladLoadGLX`

If PyBullet GUI fails to open inside Docker:

1. Run `xhost +local:docker` on the host
2. Ensure `/dev/dri` is mounted (included in compose by default)
3. Ensure Mesa/libglvnd packages are installed in the Dockerfile

### Service Calls Timeout

DDS discovery issues can occur when calling from a separate container.
Use `docker exec pbf_bridge` to run commands inside the same container.

### URDF Meshes in RViz

The default URDFs use primitive shapes (box, cylinder). To use custom meshes
(`.stl` / `.dae`), reference them via `<mesh filename="..."/>` in the URDF and
place mesh files in the `robots/` directory. `robot_state_publisher` will
publish them to RViz automatically.

## Open-RMF Integration Demo

Demonstrates PyBulletFleet integrated with Open-RMF task dispatching
using the **EasyFullControl** API (the recommended Python interface).

### Prerequisites

The Docker image builds `rmf_fleet_adapter_python` from source (the Jazzy
binary is C++ only). This provides the `rmf_adapter` Python module with
`easy_full_control` API. Build dependencies (`ros-jazzy-rmf-fleet-adapter`,
`pybind11-dev`, `pybind11_json_vendor`) are installed automatically.

### Launch

```bash
# Start bridge + fleet adapter (office demo, 2 robots)
docker compose run --rm bridge \
  ros2 launch pybullet_fleet_ros office_pybullet.launch.py

# In another terminal: dispatch a patrol task
docker compose run --rm bridge \
  ros2 run rmf_demos_tasks dispatch_patrol -p pantry lounge -n 3
```

### Dispatch Test Completion

`docker/rmf_dispatch_flow_check.py` treats explicit RMF terminal task success as the
normal pass condition. `/fleet_states` is used to confirm robot assignment, but
task-id clearing alone is not success because failed or canceled tasks also
become unassigned.

`--allow-fleet-state-clear-fallback` is available for CI portability only. When
enabled, a cleared `/fleet_states` task id can pass after task underway, robot
motion, scenario-specific physical checks, and a short grace period with no
failed or canceled terminal status.

### Client Modes

The RMF launch files expose `client_mode:=...`:

| Mode | Path | Use case |
|------|------|----------|
| `python_fleet` | RMF adapter calls the in-process PyBulletFleet API through bridge plugins | Default path for lowest ROS overhead |
| `fleet_ros` | RMF adapter calls fleet-level ROS interfaces such as `/fleet/state` and `/fleet/navigate` | ROS-visible fleet API path |
| `per_robot_ros` | RMF adapter calls per-robot ROS topics/actions/services | Compatibility/debug path |

### Architecture

```
Open-RMF Traffic Schedule
    │
    ▼
FleetAdapterNode (fleet_adapter.py)
    │  EasyFullControl API → RobotCallbacks.navigate()
    ▼
RmfFleetClient facade
    │
    ├─ python_fleet: in-process PyBulletFleet API / bridge plugins
    ├─ fleet_ros: fleet-level ROS API
    └─ per_robot_ros: per-robot ROS API
    │
    ▼
PyBulletFleet (kinematic navigation)
```

The fleet adapter reports robot state to Open-RMF continuously via `RobotState`
updates. The state source depends on `client_mode`: `python_fleet` reads directly
from the simulation, `fleet_ros` uses fleet state, and `per_robot_ros` uses
per-robot odometry.
