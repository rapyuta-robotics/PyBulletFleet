# Configuration Reference

`MultiRobotSimulationCore.from_yaml()` and `from_dict()` use this nested
layout. A flat simulation-parameter dictionary is still supported for
compatibility, but new configurations should use the nested form.

```yaml
simulation: {}      # SimulationParams fields
world: {}           # Optional SDF/world/mesh environment
entity_classes: {}  # Optional custom entity type registry
managers: []        # Optional named entity managers
entities: []        # Agents, sim objects, and devices to spawn
plugins: []         # Optional simulation-wide plugins
```

```{note}
`pybullet_fleet/config/config.yaml` is a runnable starting point, not the
exhaustive parameter reference. This page describes the supported schema.
```

## Simulation

Keys under `simulation:` map to `SimulationParams`. `None` means automatic or
disabled behaviour.

| Key | Default | Meaning |
|---|---:|---|
| `target_rtf` | `1.0` | Real-time factor. `0` runs without pacing; a positive value is the target simulation-time / wall-time ratio. |
| `timestep` | `0.1` | Simulation seconds per step. |
| `duration` | `0` | Default `run_simulation()` duration; `0` runs until interrupted or the GUI closes. |
| `gui` / `physics` | `true` / `false` | Open a GUI / call `p.stepSimulation()` and update physics bodies each step. |
| `monitor` / `enable_monitor_gui` | `true` / `true` | Enable DataMonitor and its optional tkinter window. The window provides playback controls plus entity selection and camera follow; requests are applied on the simulation step thread. |
| `monitor_update_interval` | `None` | Simulated seconds between DataMonitor refreshes while `monitor` is on. A refresh builds `last_monitor_frame` and writes the file the monitor window reads (the window reads it every 0.5 wall seconds). `None` is 0.25 s while the monitor window is open (`enable_monitor_gui`, or `gui` when that is unset) and 1 s otherwise; `0` refreshes every step. Simulated time: at a real-time factor above 1 a short interval means more writes per wall second. While paused, a selection, follow or pause change and a single step refresh at once; the final frame of a run is written, `close()` writes the last state, and `flush_monitor()` does so on demand. |
| `monitor_log_interval` | `1.0` | Wall-clock seconds between INFO `[MONITOR]` log lines (real-time factor, collision totals, step count), independent of `monitor`, `gui` and `monitor_update_interval`; `0` logs none. The line shows only when `log_level` is `info` or lower. |
| `log_level` | `warn` | Python logging level applied to the process root logger. |
| `max_steps_per_frame` | `10` | Catch-up-step limit used by the pacing loop. |
| `max_sleep_frames` | `4.0` | Maximum one pacing sleep in normal frame intervals. |
| `gui_min_fps` | `30` | GUI responsiveness target used by the pacing loop. |
| `enable_floor` | `true` | Load the default plane. `world:` disables it unless explicitly set. |
| `model_paths` | `None` | Additional model-search directories. |
| `window_width` / `window_height` | `1024` / `768` | PyBullet GUI window size. |
| `monitor_width` / `monitor_height` | `680` / `480` | DataMonitor window size. The default leaves room for its side-by-side statistics and entity inspector panes. |
| `monitor_x` / `monitor_y` | `-1` / `-1` | DataMonitor location; `-1` leaves placement to the window manager. |

### Profiling

| Key | Default | Meaning |
|---|---:|---|
| `enable_time_profiling` | `false` | Collect and periodically log per-step timing fields. |
| `enable_memory_profiling` | `false` | Collect Python allocation measurements through `tracemalloc`. |
| `profiling_interval` | `100` | Steps per periodic time and memory report. |

See [Time Profiling](../how-to/time-profiling) and
[Memory Profiling](../how-to/memory-profiling). Set `log_level: info` to show
their periodic reports.

### Collision and visualisation

| Key | Default | Meaning |
|---|---:|---|
| `collision_check_frequency` | `None` | `None`: every step; `0`: disabled; positive value: Hz. |
| `collision_detection_method` | automatic | `closest_points`, `contact_points`, or `hybrid`. Automatic selection is closest-points without physics and contact-points with physics. |
| `collision_margin` | `0.02` | Metres used by closest-points checks; ignored by contact-points checks. |
| `spatial_hash_cell_size_mode` | `auto_initial` | `auto_initial`, `auto_adaptive`, or `constant`. |
| `spatial_hash_cell_size` | `None` | Fixed cell size in metres when mode is `constant`. |
| `multi_cell_threshold` | `1.5` | Dimensionless cell-size multiplier for multi-cell registration. |
| `ignore_static_collision` | `true` | Skip every pair containing an entity with `collision_mode: static`. |
| `ignore_attached_collision` | `true` | Do not report contact between objects on one attachment tree (`attach_object()`): a carrier, the objects attached to it, and the objects attached to those, so a parent and child, two loads on one carrier, and a load on a load are all excluded. For other expected contact, see `set_collision_ignore_filter()` in the collision internals. |
| `enable_collision_color_change` | `false` | Change object colour while a collision is active. |
| `enable_collision_shapes` | `false` | Initially show collision-shape wireframes in the GUI. |
| `enable_structure_transparency` | `false` | Initially render structure bodies semi-transparently. |
| `enable_shadows` / `enable_gui_panel` | `true` / `false` | PyBullet visualiser options. |

See [Collision Configuration](../how-to/collision-config) and
[Collision Internals](../architecture/collision-internals) for behaviour and
tuning guidance.

### Camera

`camera` is accepted as an alias for `camera_config`.

```yaml
simulation:
  camera:
    camera_mode: auto
    camera_view_type: perspective
    camera_auto_scale: 0.6
```

Manual camera settings can specify `camera_distance`, `camera_yaw`,
`camera_pitch`, and `camera_target`. Recording camera selection is documented
in [Capturing Demo Videos](../how-to/capturing-demos).

### GUI lighting

`lighting` is accepted as an alias for `lighting_config`. It configures
PyBullet GUI's single main light and its shadow map; it does not import or
reproduce USD light prims.

```yaml
simulation:
  enable_shadows: true
  lighting:
    light_position: [8, -8, 15]
    shadow_map_world_size: 20
    shadow_map_resolution: 2048
    enable_controls: true
```

When `enable_controls` is true, PyBulletFleet displays the native Parameters
panel with sliders for these values. The simulation thread applies changes, so
the controls are safe for normal `run_simulation()` execution. The panel is
not available in headless mode.

## World

`world:` loads environment geometry before `entities:`. The loader selects one
source in this order: `world_file`, `sdf`, then `mesh_dir`.

```yaml
world:
  world_file: environments/office.world  # or: sdf: environments/office.sdf
  skip_models: [existing_robot_name]
  force_color: [0.7, 0.7, 0.7, 1.0]     # optional flat world-mesh colour
```

Use `mesh_dir: environments/meshes` to load a directory of OBJ/STL meshes.
Named entities are automatically added to `skip_models` for a `world_file`, so
they are not loaded twice.

## Entity classes and entities

`entities:` is a list. Each entry defaults to `type: agent`; use
`type: sim_object` for a passive object, or register a custom type through
`entity_classes:`.

```yaml
entity_classes:
  forklift: my_package.entities.ForkliftAgent

entities:
  - name: robot_0
    urdf_path: robots/mobile_robot.urdf
    pose: [0.0, 0.0, 0.05]
    mass: 0.0
    controller:
      type: differential

  - type: sim_object
    name: rack
    visual_shape: {shape_type: box, half_extents: [1.0, 0.4, 1.0]}
    collision_shape: {shape_type: box, half_extents: [1.0, 0.4, 1.0]}
    pose: [3.0, 0.0, 1.0]
    collision_mode: static
```

Common keys are `name`, `pose`, `mass`, `user_data`, `visual_shape`,
`collision_shape`, and `collision_mode`. Agents also accept `urdf_path`,
`sdf_path`, `use_fixed_base`, `controller`, and agent `plugins`.

`collision_mode` is per entity: `normal_3d`, `normal_2d`, `static`, or
`disabled`. `static` is for fixed structures; `ignore_static_collision` skips
all pairs involving that object.

### Grid entities

Add `grid:` to expand an entity entry. The count-based form is concise:

```yaml
entities:
  - name: robot
    urdf_path: robots/mobile_robot.urdf
    controller: {type: omni}
    grid:
      count: 100
      columns: 10             # optional; defaults to ceil(sqrt(count))
      spacing: [2.0, 2.0]
      offset: [0.0, 0.0, 0.05]
```

The range-based form accepts `x_min`, `x_max`, `y_min`, `y_max`, `spacing`,
and `offset`, with optional `z_min` / `z_max` for layers.

### Controller selection

Use explicit `controller.type` for new entities. `motion_mode` is a deprecated
fallback and should not be added to new YAML.

```yaml
controller:
  type: differential
  max_linear_vel: 1.5
  max_angular_vel: 2.0
```

The YAML value may be `null`, a registry-name string, a mapping, or a list of
mappings for a controller chain. Python APIs also accept `Controller` and
`ControllerParams` instances. See [Controller Configuration](../how-to/controller-config).

## Named managers and fleet controllers

Managers must be declared before entities that reference them. They own shared
controller defaults and optional batch execution.

```yaml
managers:
  - name: delivery_fleet
    fleet_controller:
      type: batch_differential
      max_linear_vel: 1.5
      max_linear_accel: 2.0

entities:
  - urdf_path: robots/mobile_robot.urdf
    manager: delivery_fleet
    grid:
      count: 50
      spacing: [2.0, 2.0]
```

Built-in batch types are `batch_omni` and `batch_differential`. A custom batch
controller may use a registered name or dotted class path. Omit
`fleet_controller.type` for per-agent execution with shared defaults.
`batch_controller` and `fleet_controller` are not valid directly under an
entity.

## Plugins

Simulation-wide plugins are root entries. Per-agent plugins are nested below
an entity. Both accept registry `type` or dotted `class`, plus `config`.

```yaml
plugins:
  - class: my_package.plugins.TrafficMonitor
    frequency: 2.0
    config:
      log_interval: 5.0

entities:
  - urdf_path: robots/mobile_robot.urdf
    plugins:
      - type: battery
        config:
          capacity: 100.0
```

See [Plugins and Events](../architecture/plugins-events) for lifecycle and
two-phase step ordering.

## Defaults and environment overrides

`pybullet_fleet/_defaults.py` holds built-in defaults. At module load time,
`PBF_{SECTION}_{KEY}` environment variables replace non-`None` defaults; for
example, `PBF_SIMULATION_TIMESTEP=0.05`, or
`PBF_CONTROLLER_GOAL_REPLACEMENT=reject` for how every agent without its own
setting handles a goal it cannot reach from its current motion. If `python-dotenv` is installed, a
`.env` file fills otherwise unset environment variables.

YAML values passed to `from_yaml()` / `from_dict()` and explicit
`SimulationParams(...)` values override the defaults for the configuration
being constructed. Use `_defaults.get("simulation", "timestep")` only in
framework or extension code that intentionally needs the global default.

## Included configurations and benchmarks

`pybullet_fleet/config/` contains runnable scenario and mode examples.
`config_physics_off.yaml`, `config_physics_on.yaml`, and `config_hybrid.yaml`
illustrate collision-method choices; select them for the required fidelity, not
a fixed performance claim.

Benchmark inputs live under `benchmark/configs/`. See
[Benchmark Suite](../benchmarking/benchmark-suite) for commands and result
interpretation.

## See Also

- [Controller Configuration](../how-to/controller-config)
- [Collision Configuration](../how-to/collision-config)
- [Plugins and Events](../architecture/plugins-events)
- [ROS 2 Configuration](../ros2/configuration)
