# Snapshot / Replay — Approved v1 Scope

**Status:** Implemented and merged in PR #50 on 2026-09-29.

PR #50 is a restricted development validation profile. Its implemented
initial-state re-execution does not make the overall snapshot/replay feature
ready for general or production use.

This document fixes the approved **v1** boundary. Read the
[overall product goals](product-goals.md) before scoping later work; v1 does
not itself deliver the full recording/restart/playback/debug workflow.

The first conceptual change is re-execution and result comparison of effective
Fleet API commands from a recorded initial state, for a fixed kinematic navigation
profile. Explicit configuration overrides or changed code produce a variant from
that same initial state, not a resume from an intermediate checkpoint. Scope and
architecture were explicitly approved under workflow v0.1.

The final Snapshot/Replay goals are (1) playback of recorded simulation data
like `rosbag play`, (2) restore/resume from an arbitrary recorded snapshot, and
(3) algorithm changes after restore for failure reproduction and A/B tests.
The v1 initial-state re-execution profile is a limited step toward these goals,
not their completion. Schema, journal, execution boundary, identity, USO and
trace decisions are means to these workflows or possible secondary uses, not
independent goals.

## Contract

- Session-owned fresh simulation; physics off, fixed timestep/entities.
- Planar navigate and stop, omni/batch_omni, bundled simple_cube/static boxes.
- Stable artifact entity ID separate from API name and runtime IDs.
- Input phase `(step=k, order=n)` before controller update; observations of S_(k+1).
- PBF-owned versioned JSON/JSONL: initial state, execution journal, full
  observations, provenance, successful completion/integrity.
- Same-environment supported-profile comparison, not bitwise/cross-platform determinism.
- Fail-fast when recording completeness is lost; valid differences remain distinct.

See [initial-state re-execution contract](navigation-reexecution-guide.md), [implementation plan](plan.md),
and [evidence](evidence.md).

## Explicit non-goals

Execution checkpoint/restore/resume, general result playback, deltas, arbitrary
Python/plugin serialization, BT/device/action/joint/attachment replay, dynamic
worlds, ROS/DDS deterministic replay, rosbag integration, generic asset management,
shared USO runtime/schema package, GUI editor and cloud storage.

## Relationship to the 2026-04-05 draft

The original draft proposed USO-shaped full/delta serialization followed by
physics-off pose playback, then a USO adapter. The later roadmap also proposed
input re-execution and checkpoint/restore. Those are distinct capabilities;
neither draft supplied the sufficient execution-state contract for arbitrary
restore. The approved v1 deliberately implements initial-state re-execution.

The original draft's single-string `connected_to`, collision-oriented
`_moved_this_step` delta strategy, and assumed private lookup fields must not be
used as current contracts. USO itself now describes logical connections as lists
and leaves joint/version/extension details open. No persisted old format was
identified, so there is no legacy-reader migration. Git history retains the old
draft. Existing movie recording and ordinary Fleet API behavior remain separate.

The [2026-04-05 PBF draft](https://github.com/yuokamoto/PyBulletFleet/blob/59e4375/docs/design/snapshot-replay/spec.md)
remains available for its DataMonitor separation and proposed PBF-to-USO mapping.
It was not moved into the USO repository and is not an implementation contract
for v1.

## USO evolution

This PBF artifact validates concrete concepts for future USO refinement. It does
not permanently choose a competing schema. Findings will feed into USO, then be
validated in another backend before any shared abstraction is extracted. The USO
repository is not modified by this change.

For the USO-side design and its unsettled questions, see the
[snapshot specification](https://github.com/yuokamoto/Unified-Simulation-Orchestrator/blob/main/03_Snapshot_Specification_EN.md),
[logging/replay specification](https://github.com/yuokamoto/Unified-Simulation-Orchestrator/blob/main/06_Logging_Replay_EN.md),
[open questions](https://github.com/yuokamoto/Unified-Simulation-Orchestrator/blob/main/17_Open_Questions_EN.md),
and [bottom-up rollout approach](https://github.com/yuokamoto/Unified-Simulation-Orchestrator/blob/main/16_Current_Status_and_Rollout_Approach_EN.md).
These documents describe future integration inputs, not a PBF v1 compatibility
claim. See [implementation evidence](evidence.md) for findings to take back to USO.
