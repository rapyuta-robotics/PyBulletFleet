# Examples

Runnable demos for PyBulletFleet, grouped by area:

| Folder | What it shows |
|--------|---------------|
| `basics/` | Spawning, the action system, the event bus |
| `mobile/` | Path following and mobile-robot navigation |
| `arm/` | Arm pick/drop, end-effector (IK) control, mobile manipulators, collision of an arm's links |
| `models/` | Bundled model names, `robot_descriptions`, SDF and OpenUSD worlds |
| `scale/` | 100–1000 robot fleets |
| `evaluation/` | Synthetic fleet scenarios and external policy/evaluator comparisons |
| `validation/` | State-discovery and checkpoint proofs used to establish supported boundaries |

The `validation/` programs are focused verification drivers. They do not
represent the eventual workflow of enabling recording and restart in an
ordinary demo. The `evaluation/` programs remain runnable examples of how an
external application can compare fleet behavior using PBF observations.

Most demos open a PyBullet GUI window and run until you close it — they are meant
to be watched, not run in CI. Some `models/` demos need extras:
`pip install 'pybullet-fleet[sdf,models,usd]'`.

The OpenUSD warehouse demo uses a bundled self-contained `.usda` scene:

```bash
python pybullet_fleet/examples/models/usd_warehouse_demo.py
python pybullet_fleet/examples/models/usd_warehouse_demo.py --headless
python pybullet_fleet/examples/models/usd_warehouse_demo.py --usd /path/to/local_warehouse.usdz
python pybullet_fleet/examples/models/usd_warehouse_demo.py --lighting-controls
```

The generic worker behavior-tree demo does not require OpenUSD:

```bash
python pybullet_fleet/examples/basics/behavior_tree_demo.py
python pybullet_fleet/examples/basics/behavior_tree_demo.py --headless --duration 5
```

The synthetic corridor evaluation runs 20 robots under two external entry
policies. Select one policy with `--gui` to watch a live simulation; the
default command runs both policies headlessly and writes comparable reports.
See the [corridor evaluation guide](../../docs/how-to/fleet-corridor-evaluation.md).

```bash
python -m pybullet_fleet.examples.evaluation.fleet_corridor_evaluation /tmp/pbf-corridor \
  --gui --policy uncontrolled --cutoff 30
```

## Scale Demo Roles

| Demo | Role |
|------|------|
| `scale/100robots_grid_demo.py` | Primary config-driven 100 mobile robot grid demo. Use it to try controller, command-interface, and movement switches in a user-facing scene. |
| `scale/100robots_mixed_demo.py` | Representative mixed fleet scene with 50 Husky robots and 50 fixed-base `arm_robot` arms, using `entities[].grid` groups. |
| `scale/100robots_cube_patrol_demo.py` | Tutorial-style 100 mobile robot patrol scene using `managers:` and `fleet_controller:` config. |
| `scale/batch_controller_scale_demo.py` | Focused batch-controller scale demo. Robot count is `--robots` (default 500); use it for quick fleet API and command setup checks, not rigorous benchmarking. |
| `scale/pick_drop_mobile_100robots_demo.py` | Mobile robots using pick/drop action sequences and object management. `--robots` controls the count, and `--no-gui` is available for headless checks. |
| `scale/pick_drop_arm_100robots_demo.py` | Fixed-base arms using synchronized joint/action pick-drop cycles. `--robots` controls the count, and `--no-gui` is available for headless checks. |

## Fleet API Verification

`scale/batch_controller_scale_demo.py` can exercise the same scale scene through
either the per-agent command interface or the transport-neutral fleet API. The
controller implementation remains the batch controller in both modes, and the
default command interface is fleet.

```bash
# Default: batch controller + fleet command interface.
python3 pybullet_fleet/examples/scale/batch_controller_scale_demo.py \
  --no-gui --duration 5 --n 100

# Per-agent command interface comparison path.
python3 pybullet_fleet/examples/scale/batch_controller_scale_demo.py \
  --no-gui --duration 5 --n 100 --command-interface per_agent
```

Both modes print command setup time and fleet state snapshot time so local runs
can compare the control path without changing the scene, robot count, or loop.

`scale/100robots_grid_demo.py` is a config-driven example for trying the same
switches in a user-facing demo. `config/100robots_config.yaml` uses
`entities[].grid` as the primary scene definition, and CLI flags choose the
controller and command path.

```bash
# Default: batch controller, fleet command interface, random movement.
python3 pybullet_fleet/examples/scale/100robots_grid_demo.py --duration 5

# Compare command APIs on a repeated goal-command run.
python3 pybullet_fleet/examples/scale/100robots_grid_demo.py \
  --movement goal --command-interface per_agent --duration 5
python3 pybullet_fleet/examples/scale/100robots_grid_demo.py \
  --movement goal --command-interface fleet --duration 5
```

For repeatable timing comparisons, use the benchmark runner:

```bash
python3 benchmark/run_benchmark.py --type mobile \
  --controller per_agent batch --command-interface per_agent fleet --agents 500 --steps 600
```

`scale/100robots_mixed_demo.py` keeps the mixed fleet showcase separate from
the controller/API comparison demo:

```bash
python3 pybullet_fleet/examples/scale/100robots_mixed_demo.py --duration 10
python3 pybullet_fleet/examples/scale/100robots_mixed_demo.py \
  --controller per_agent --duration 10
```

For ROS bridge verification, run the Docker smoke and optional scale checks
documented in `docker/README.md`; they check `/fleet/states`, `/fleet/navigate`,
and `/fleet/joint_command` using the bridge.

## Running them after `pip install` (no clone needed)

The examples ship **inside the wheel**, so a pip-installed package can list,
locate, copy, or run them via the `pybullet-fleet` CLI:

```bash
pip install pybullet-fleet

pybullet-fleet examples --list                 # what's available
pybullet-fleet examples --path                 # where they're installed
pybullet-fleet examples --run path_following_demo.py   # launch one (GUI)
pybullet-fleet examples --copy ./my-examples    # copy them out to read/edit
```

`--run` forwards trailing flags to the example
(`pybullet-fleet examples --run path_following_demo.py --duration 5`). If an example
flag clashes with a CLI option name (e.g. some demos have their own `--list`),
use `--copy` and run the copied file directly instead.

## Where are they installed?

`pybullet-fleet examples --path` prints the directory. It is alongside the
installed package:

- **pip install** → `…/site-packages/pybullet_fleet/examples/`
- **editable (`pip install -e .`)** → `…/<repo>/pybullet_fleet/examples/` (your
  working tree — edit these directly)

To read or tweak a demo from a normal install, copy it out first
(`pybullet-fleet examples --copy ./my-examples`) and edit the copy; the originals
in `site-packages` are part of the installed package.

## Testing released vs local code side by side

`scripts/setup_example_venvs.sh` builds two virtualenvs so you can compare the
released package against your working tree:

```bash
scripts/setup_example_venvs.sh            # install venv tracks latest PyPI
scripts/setup_example_venvs.sh 0.4.1      # or pin a version

# released package (what users get):
.venvs/example-install/bin/pybullet-fleet examples --run path_following_demo.py

# your working tree (editable install):
.venvs/example-mount/bin/pybullet-fleet examples --run path_following_demo.py
```

Both venvs install the package, so each runs its own bundled examples and library:
`example-install` exercises the released wheel (including its bundled `robots/`,
`config/`, and `mesh/` data), `example-mount` runs this checkout. Complementary to
`scripts/test_clean_install.sh <version>`, which smoke-tests the same packaging
non-interactively.
