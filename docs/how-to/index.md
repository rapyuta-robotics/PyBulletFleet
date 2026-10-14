# How-to Guides

Practical guides for common tasks.

```{warning}
State recording, checkpoint/restore and result playback are under development.
The [supported kinematic recording guide](supported-state-recording) describes
the current validation profile; it does not provide general save/load for
ordinary simulations. Do not rely on its artifacts for production runs yet.
Video capture in [Capturing Demos](capturing-demos) is a separate feature.
```

**📘 For Simulation Users** — building and configuring your own simulation:

| Guide | What It Covers |
|-------|----------------|
| [Collision Configuration](collision-config) | Detection method, per-object modes, margin, cell size, multi-cell threshold |
| [Controller Configuration](controller-config) | `controller` argument forms, `motion_mode` vs `controller.type`, batch controllers (auto-registration, coexistence, direction, action system), custom controllers |
| [Custom Profiling](custom-profiling) | Adding your own profiling metrics via Agent/SimulationCore subclasses |
| [Capturing Demos](capturing-demos) | Recording demo videos, `demos.yaml` format, centralized defaults (`_defaults.py`) |
| [Supported Kinematic Recording](supported-state-recording) | Development profile: record, restore and play back supported kinematic state |

**🔧 For PyBulletFleet Developers** — developing and debugging the framework:

| Guide | Measures | Detects | Tool | Overhead |
|-------|----------|---------|------|----------|
| [Time Profiling](time-profiling) | Execution time (ms) | Performance bottlenecks | `time.perf_counter()` | < 0.1% CPU |
| [Memory Profiling](memory-profiling) | Memory usage (MB) | Memory leaks, high usage | `tracemalloc` (built-in) | ~5–10% memory |
| [Logging](logging) | Log messages | Configuration & runtime issues | Python `logging` | Negligible |

> **Tip:** Time and memory profiling can be enabled simultaneously for
> comprehensive analysis. Both share the `profiling_interval` setting.

```{toctree}
:maxdepth: 2
:caption: For Simulation Users

collision-config
controller-config
custom-profiling
capturing-demos
supported-state-recording
fleet-corridor-evaluation
fleet-corridor-traffic-failure
```

```{toctree}
:maxdepth: 2
:caption: For PyBulletFleet Developers

time-profiling
memory-profiling
logging
```
