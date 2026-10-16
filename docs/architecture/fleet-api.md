# Fleet API Architecture

The Fleet API is the transport-neutral boundary for whole-fleet state and
commands. It intentionally has no ROS imports, so Python callers, ROS bridge
wrappers, RMF bridge plugins, and future transports can use the same state and
command semantics.

## Components

| Component | Responsibility |
| --- | --- |
| `FleetStateProvider` | Builds deterministic 2D or 3D snapshots from a simulation core |
| `FleetCommandDispatcher` | Resolves named targets, validates commands, mutates accepted agents, and returns a `CommandAck` |
| Command/state dataclasses | Stable Python payloads for goals, joints, attachment, actions, and robot state |
| Command event | Records command id, source, simulation time, accepted names, and per-name rejection reasons |

The dispatcher refreshes its robot-name index for every command. A target is
rejected when its name is unknown, ambiguous, or duplicated in one request.
Accepted and rejected targets may coexist in one acknowledgement, so callers
must process `accepted_names` and rejection reasons rather than treating a
partially accepted command as all-or-nothing.

## Command flow

```text
transport adapter or Python caller
  -> command dataclasses
  -> FleetCommandDispatcher
       -> name validation and command event
       -> public Agent mutation / action queue
  -> CommandAck
```

An acknowledgement means that the command passed ingress validation and was
applied or queued for the listed robots. It is not a completion event. Clients
that need completion should observe state, actions, or their domain-specific
task result after the acknowledgement.

## Transport mappings

| Integration | Uses the Fleet API how |
| --- | --- |
| Python application | Calls provider and dispatcher directly |
| ROS 2 fleet API | Converts `/fleet/*` messages/services to the same command/state models |
| RMF `python_fleet` | Calls the provider and dispatcher in-process through a bridge plugin |
| RMF `fleet_ros` | Reaches the models via the fleet-level ROS bridge |

This boundary deliberately separates **how a command arrives** from **how the
simulation executes it**. Batch controllers can accelerate motion calculation,
but they are neither required by nor coupled to `FleetCommandDispatcher`.

## Developer rules

- Keep data models free of ROS types and imports.
- Add a command type only with a corresponding acknowledgement, validation, and
  transport mapping test.
- Preserve command ids and `source` values across adapters for traceability.
- Treat simulation state as a snapshot; do not retain mutable Agent objects in
  transport payloads.

For endpoint names and user configuration, see [Bridge Interfaces](../ros2/overview).
For the Python symbols, see the generated [API reference](../api/generated/pybullet_fleet).

## Opt-in replay execution boundary

The restricted navigation replay session owns a fresh core and applies ordered
effective inputs before PRE_STEP. The ordinary dispatcher remains synchronous;
`CommandAck` is still acceptance/application, not navigation completion.
`retain_command_events=False` can disable the in-memory diagnostic history for
a session whose durable journal already records commands; the default is unchanged.

Replay inputs preserve payload, source, command ID, allowed target names, and
(step, phase, order). Capture records the actual returned ack independently of
the existing pre-mutation `CommandEvent`; that event alone lacks replay payload.
Python, ROS, RMF and other producers can supply the same effective input after
translation. Existing ROS endpoints are not automatically moved to this boundary.
See the [initial-state navigation re-execution guide](https://github.com/yuokamoto/PyBulletFleet/blob/main/docs/design/snapshot-replay/navigation-reexecution-guide.md) for that restricted profile and API.
