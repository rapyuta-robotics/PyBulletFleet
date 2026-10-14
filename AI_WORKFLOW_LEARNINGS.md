# AI Workflow Learnings

Reusable observations from completed development work. These are candidates for
future workflow improvement, not new requirements or an activity log.

## Distinguish the final capability from the incremental slice

**Occurrences:** 2 (#50, snapshot/replay v1; #55, manipulation state discovery)

**Evidence:** The approved v1 implemented initial-state input re-execution, but
review discussions had to repeatedly distinguish it from recorded-result
playback and intermediate checkpoint restore/resume. The final user goals and
the limited role of v1 were then made explicit in the specification, user guide,
and roadmap. In #55, the scenario only observed state, but post-implementation
review again needed an explicit explanation and checklist separating observable
values from save/load and fresh-process continuation.

**Reusable check:** For an incremental v1, describe both the final intended
capability and the capability delivered by this slice as user actions and
observable results. In replay scopes, name playback, re-execution, and
restore/resume separately before deriving required state and inputs.

**Status:** The general end-to-end goal/slice distinction is proposed for
promotion into [`docs/AI_DEVELOPMENT_WORKFLOW.md`](docs/AI_DEVELOPMENT_WORKFLOW.md)
in this PR. Keep this entry as evidence of the original replay-specific
confusion; no separate template or Skill is proposed.

## Separate simulator capabilities from replay orchestration

**Occurrences:** 1 (#50, snapshot/replay v1)

**Evidence:** The first implementation put replay-session-dependent behavior in
simulation and model paths. Human review clarified that simulator internals may
need to expose or restore required state and inputs, while recording, replay,
restore, and resume workflows should be controlled externally. The implementation
was refactored so `ReplaySession` applies inputs and advances ordinary steps.

**Reusable check:** During future scope and architecture review, distinguish
simulator-internal save/load or input-access capabilities from external
orchestration before adding hooks to core simulation paths. Define what direct,
out-of-session mutations mean for the recording contract.

**Candidate destination:** Snapshot/replay architecture guidance, if another
slice needs the same correction. **Status:** watch; no promotion yet.

## Preserve existing mode semantics during a localized correctness fix

**Occurrences:** 1 (#52, collision-margin broadphase)

**Evidence:** The margin fix went through several broader implementations before
review narrowed it to the candidate AABB envelope. Independent review then
found that the initial fix changed the 2D mode's Z-overlap behavior. A focused
regression restored that existing mode contract.

**Reusable check:** State the invariant being repaired and the behavior of each
existing mode before changing shared candidate extraction. Keep the fix at the
smallest layer that enforces the invariant, and test a counterexample plus mode
boundaries.

**Candidate destination:** Collision/performance development guidance if this
recurs. **Status:** watch; no promotion yet.

## Deliver a complete user operation before multiplying discovery slices

**Occurrences:** Snapshot/replay work across merged #50–#56. This is one
extended development theme, not seven independent confirmations of the same
lesson.

**Evidence:** #50 delivered restricted initial-state/input re-execution, while
the intended recording, result playback and intermediate restart were still
unclear to reviewers. #51 and #54 built useful fleet failure experiments, and
#52–#53 corrected/exposed collision facts, but these primarily advanced
evaluation rather than the user's save/restart/playback workflow. #55 inventoried
manipulation state without restoring it. #56 proves fresh-process continuation
only for one fixed straight-moving omni robot; ordinary examples still cannot
opt into capture and restart. The product-goal document and state checklist
were written after much of this work, so each narrow PR required repeated
explanation of what it did *not* deliver. More examples, design documents and
review rounds accumulated faster than the end-to-end capability.

**Replay-specific follow-up:** Consider one supported scenario spanning
navigation, joint motion and attachment if the next V1 user operation requires
all three; narrow profiles can remain tests within that slice. Treat corridor
evaluation as useful in its own right, but do not count it as checkpoint
progress. Assess whether #50's re-execution code serves a real user operation
before extending or removing it.

**Status:** The general workflow and verification changes were merged in #57.
This entry retains the historical evidence and replay-specific follow-up;
assess the revised workflow during the next V1 work.

## Validate the user-facing restore operation as one capability

**Occurrence:** PR #59 delivered a versioned recorder and a manipulation
validation profile, but review still required the application to resolve a
profile, read a manifest, choose a checkpoint, restore the core and arrange
continuation. It also exposed recording failures that could interrupt normal
simulation until the runtime failure path was corrected.

**Observation:** Extensible artifact internals and a passing validation
scenario do not by themselves make recording and restart usable from an
ordinary PBF example. The next acceptance test should begin with the user
operation: enable recording in a normal run, then construct and restore from
an artifact in one public call and continue with ordinary `run_simulation()`.
Prove the support boundary on a second, differently composed simulation and
reject unsupported restore state. Keep recording failures from stopping a
healthy simulation while reporting an incomplete artifact.

**Scope implication:** Measure duplicate frame/checkpoint storage, step cost
and load time, but decide on a format change separately. This is one
observation from the same snapshot/replay theme, not evidence for another
workflow rule or a reason to change `AGENTS.md` or CI now.

**Status:** Watch during the next user-facing restore slice.
