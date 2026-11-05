# API Reference

The full reference is generated from source docstrings with Sphinx autodoc.
Use the high-level guides for concepts and examples; use this section to look
up exact Python signatures, parameters, and return values.

```{toctree}
:hidden:

generated/pybullet_fleet
```

## Full Module Reference

See [Full Module Reference](generated/pybullet_fleet) for documented members of
all non-private package modules. It includes advanced and developer-facing
modules as well as the stable user-facing API.

## User-Facing API

| Area | Primary modules | Examples |
|------|-----------------|----------|
| Simulation and entities | `core_simulation`, `sim_object`, `agent`, `agent_manager` | `MultiRobotSimulationCore`, `SimulationParams`, `AgentSpawnParams` |
| Controllers and actions | `controller`, `controllers`, `action` | `OmniController`, `DifferentialController`, action classes |
| Fleet commands and state | `commands`, `fleet_api`, `states`, `state_plane` | `FleetCommandDispatcher`, `FleetStateProvider`, `StatePlanePublisher`, `SharedStateReader` |
| Events and plugins | `events`, `sim_plugin`, `agent_plugin`, `plugins` | `EventBus`, `SimPlugin`, `AgentPlugin` |
| Worlds and models | `robot_models`, `sdf_loader`, `world_loader` | `resolve_model`, SDF/world loading helpers |
| Devices and recording | `devices`, `recorder` | `Door`, `Elevator`, `SimulationRecorder` |
| Configuration and common values | `config_utils`, `geometry`, `types`, `tools` | YAML loading, `Pose`, enums, coordinate helpers |

## Advanced and Developer Modules

The full reference also includes implementation-support modules such as
`controller_params`, `entity_registry`, `plugin_utils`, `data_monitor`, and
`camera_controller`, plus command-line and compatibility modules. Prefer the
user-facing API above unless you are extending the framework. In particular,
`collision_visualizer` is retained for compatibility and is not part of the
recommended public workflow.

:::{tip}
To refresh the generated reference after adding or removing a package module:
```bash
source .venv/bin/activate
sphinx-apidoc --force --no-toc \
  --output-dir docs/api/generated pybullet_fleet
cd docs && sphinx-build -W -b html . _build/html
```

Review the generated files before committing them. The API index is the only
toctree entry point, so do not add the `modules.rst` table-of-contents file
created by older `sphinx-apidoc` runs.
:::
