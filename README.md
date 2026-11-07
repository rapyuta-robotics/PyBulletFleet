# PyBulletFleet

[![Documentation](https://readthedocs.org/projects/pybulletfleet/badge/?version=latest)](https://pybulletfleet.readthedocs.io/en/latest/)

<table align="center">
<tr>
<td align="center"><b>100 Robot Grid</b><br><sub>100robots_grid_demo.py</sub><br>
<img src="docs/media/100robots_grid_mixed.gif" width="380" alt="100 Robot Grid"></td>
<td align="center"><b>Cube Patrol</b><br><sub>100robots_cube_patrol_demo.py</sub><br>
<img src="docs/media/100robots_cube_patrol.gif" width="380" alt="Cube Patrol"></td>
</tr>
<tr>
<td align="center"><b>Mobile Pick & Drop</b><br><sub>pick_drop_mobile_100robots_demo.py</sub><br>
<img src="docs/media/pick_drop_mobile_100robots.gif" width="380" alt="Mobile Pick and Drop"></td>
<td align="center"><b>Arm Pick & Drop</b><br><sub>pick_drop_arm_100robots_demo.py</sub><br>
<img src="docs/media/pick_drop_arm_100robots.gif" width="380" alt="Arm Pick and Drop"></td>
</tr>
</table>

A **kinematics-first** simulation framework for large-scale multi-robot fleets, built on PyBullet and designed for **fast N× real-time** evaluation.

## What is PyBulletFleet?

Different simulation goals call for different tools.
Physics-focused simulators (Gazebo, Isaac Sim, MuJoCo, etc.) excel at accurate contact dynamics, sensor modelling, and single-robot control — but stepping a full physics engine for every robot becomes the bottleneck when you need to evaluate **fleet-level** systems at scale.

PyBulletFleet sits in a different part of the design space: it is a **kinematics-first, fleet-scale** simulation engine whose primary goal is to enable fast development and testing of the software that *orchestrates* robot fleets rather than the software that *controls* individual robots.

### Design Priorities

- **Speed over fidelity** — Fleet algorithms (task allocation, traffic control, path planning) must be tested with hundreds to thousands of robots running *much faster* than real time. Kinematics-based stepping — teleporting each robot to its next pose without calling `stepSimulation()` — removes the physics bottleneck and enables N× real-time execution.
- **System integration over low-level control** — The primary consumers are high-level systems: WMS (Warehouse Management Systems), task orchestrators, fleet managers, and monitoring dashboards. These systems issue goals, observe progress via state snapshots, and react to events — they do not need joint-level torque feedback.
- **Scale over detail** — Validating behaviour at 100+ robot scale matters more than modelling individual link dynamics or sensor noise.
- **Interoperability** — A ROS 2 Jazzy bridge exposes per-robot and fleet-level
  topics, services, and actions, while the Open-RMF adapter supports direct
  Python, fleet-level ROS, and per-robot ROS control paths. Other transports
  such as gRPC remain future work.
- **Physics as an option** — When physical interaction *is* needed (grasping, conveyor dynamics, contact verification), full PyBullet physics can be switched on per-scenario without changing the rest of the stack.

### Target Use Cases

| Use Case | Description |
|----------|-------------|
| Fleet algorithm evaluation | Test path planning, task allocation, and traffic control for large robot fleets at N× real-time speed |
| Warehouse simulation | Simulate pick-and-place, patrol, and transport operations with mobile robots and arms |
| Scalability benchmarking | Measure how fleet software scales from tens to thousands of agents |
| Rapid prototyping | Quickly iterate on multi-robot behaviors with minimal boilerplate |

## Quick Start

### Install from PyPI

```bash
pip install pybullet-fleet
```

For ROS 2 Jazzy, do not install the core only inside a virtual environment:
`ros2 run` uses the system Python interpreter. Follow the [ROS 2 bridge
quickstart](https://pybulletfleet.readthedocs.io/en/latest/ros2/quickstart.html)
instead; it installs PyBulletFleet once into the system Python user site before
adding the ROS packages.

### Or install from source (for development)

```bash
git clone https://github.com/yuokamoto/PyBulletFleet.git
cd PyBulletFleet
sudo apt install python3-tk  # required for the optional DataMonitor tkinter GUI
pip install -e ".[dev]"
```

### ROS 2 Jazzy APT preview

For Ubuntu 24.04 or WSL with ROS 2 Jazzy, the bridge packages are available
from the GitHub-hosted APT preview. The simulation core remains a separate
Python package. Follow the [APT preview installation guide](https://yuokamoto.github.io/pybullet-fleet-apt/README.md)
for the repository setup, system-Python installation, bundled example, and
bridge smoke test. The preview is intended for short-lived demo environments
until the official ROS package release is available.
See the [ROS 2 bridge quickstart](https://pybulletfleet.readthedocs.io/en/latest/ros2/quickstart.html)
and [RMF quickstart](https://pybulletfleet.readthedocs.io/en/latest/ros2/rmf-quickstart.html)
for the complete native sequence.

### Run a demo

The examples ship inside the package, so after `pip install pybullet-fleet` you
can list and run them with the `pybullet-fleet` CLI — no clone needed:

```bash
pybullet-fleet examples --list                          # all demos
pybullet-fleet examples --run 100robots_grid_demo.py     # launch one (GUI)
pybullet-fleet examples --copy ./examples                # copy them out to read/edit
pybullet-fleet examples --path                           # where they're installed
```

`--run` takes the file name as shown by `--list` (the `.py` is optional).

Many demo scripts accept a `--robot` argument to swap the robot model (forwarded
through `--run`). Pass a model name (resolved via `resolve_model()`) or a direct
URDF path:

```bash
pybullet-fleet examples --run path_following_demo.py --robot racecar
pybullet-fleet examples --run pick_drop_arm_demo.py --robot kuka_iiwa
pybullet-fleet config --copy ./config                       # copy YAML templates to edit
# (resolve_model_demo has its own --list; copy it out and run it directly)
```

From a source checkout you can also run the files directly, e.g.
`python pybullet_fleet/examples/scale/100robots_grid_demo.py`.

| Category | Demo (pass to `--run`) | `--robot` default | Alternatives |
|----------|------------------------|-------------------|-------------|
| Arm demos | `pick_drop_arm_*.py`, `rail_arm_demo.py` | `panda` | `kuka_iiwa`, `arm_robot` |
| Mobile demos | `path_following_demo.py` | `husky` | `racecar`, `mobile_robot` |
| Scale demos (mobile) | `100robots_cube_patrol_demo.py`, `pick_drop_mobile_100robots_demo.py` | `husky` | `racecar`, `mobile_robot` |
| Scale demo (grid) | `100robots_grid_demo.py` | `--robot` (`mobile_robot`) | `husky`, `racecar` |
| Scale demo (mixed) | `100robots_mixed_demo.py` | `--mobile-robot` (`husky`), `--arm-robot` (`panda`) | `racecar`, `mobile_robot`; `kuka_iiwa`, `arm_robot` |
| Scale demos (arm) | `pick_drop_arm_100robots_demo.py` | `panda` | `kuka_iiwa`, `arm_robot` |
| Model demos | `resolve_model_demo.py`, `robot_descriptions_demo.py` | `panda` / `tiago` | any registered model |

`100robots_grid_demo.py` defaults to a 100 mobile robot `entities[].grid` scene.
Use `--robot` to override the model for every mobile entity, or `--config` to
point it at another `entities[].grid` scene. `100robots_mixed_demo.py` is the
separate Husky + Panda mixed showcase; use `--mobile-robot` and `--arm-robot`
to override its two groups.

See [Tutorial 6 — Robot Models](https://pybulletfleet.readthedocs.io/en/latest/examples/robot-models.html) for the full model resolution system.

## Performance

<!-- sync with docs/benchmarking/results.md -->
> Results from a single test environment (AMD Ryzen AI 7 PRO 350, 27 GB RAM, Linux WSL2). Your numbers will vary depending on hardware.

| Agents | Real-Time Factor | Step Time |
|--------|-----------------|-----------|
| 100    | 177.4× | 0.6 ms  |
| 500    | 36.3× | 2.8 ms |
| 1000   | 17.8× | 5.6 ms |
| 2000   | 6.1× | 16.3 ms |

Kinematics mode (physics OFF), headless, `simple_cube` robots, batch controller + fleet command interface. Measured on 2026-11-07. See [Benchmark Results](docs/benchmarking/results.md) for full data, component breakdown, and methodology.

For ROS 2, use the fleet-level interface as the scalable default. Manager-scoped
endpoints support operational ownership and selective state subscriptions; they
do not increase total throughput when clients subscribe to every manager stream.
See [`ros2_bridge/PERFORMANCE.md`](ros2_bridge/PERFORMANCE.md) for measurement
conditions, current results, and deployment guidance.

## Robot Models

PyBulletFleet includes a model resolution system that loads robots **by name** from multiple sources:

```python
from pybullet_fleet import MultiRobotSimulationCore, Agent, Pose, resolve_model

sim = MultiRobotSimulationCore()

# Resolve by name — searches local robots/, pybullet_data, robot_descriptions
urdf = resolve_model("panda")

# Agent.from_urdf() calls resolve_model() internally
agent = Agent.from_urdf(urdf_path="panda", pose=Pose.from_xyz(0, 0, 0), sim_core=sim)
```

| Tier | Source | Example models |
|------|--------|---------------|
| 0 — local | `robots/` directory | arm_robot, mobile_robot, mobile_manipulator |
| 1 — pybullet_data | PyBullet bundled | panda, kuka_iiwa, r2d2 |
| 2 — ROS | ROS install paths | (future) |
| 3 — robot_descriptions | pip package | tiago, pr2 (`pip install robot_descriptions`) |

Run `pybullet-fleet examples --copy ./examples && python ./examples/models/resolve_model_demo.py --list` to see all registered models and their availability.

## Documentation

📖 **Full documentation:** [Read the Docs](https://pybulletfleet.readthedocs.io)

For the ROS 2 bridge and Open-RMF integration, start with the
[ROS 2 documentation](https://pybulletfleet.readthedocs.io/en/latest/ros2/index.html).
The repository-level [bridge README](ros2_bridge/README.md) remains the detailed
interface and release reference.

For local builds:
```bash
cd docs && sphinx-build -b html . _build/html
```

## Development Setup

A root `Makefile` provides all common dev commands. Run `make help` to list targets.

```bash
make verify        # Lint + test (CI subset, excludes docs/security)
make test          # Tests with coverage (75% threshold)
make test-fast     # Quick test (stop on first failure)
make lint          # All pre-commit hooks (black, pyright, flake8)
make format        # Auto-format with black
make typecheck     # Pyright type check
make bench-smoke   # Quick benchmark (~10s)
make docs          # Sphinx docs (warnings = errors)
make clean         # Remove caches and build artifacts
```

### Pre-commit hooks

Install pre-commit hooks for automatic formatting and linting on commit:

```bash
pip install pre-commit
pre-commit install
```
