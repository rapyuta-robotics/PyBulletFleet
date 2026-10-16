"""Restricted re-execution and declared execution-state recording profiles.

The original ReplaySession artifact has no intermediate checkpoint. The state
recorder accepts versioned profiles supplied by callers.
"""

from .artifact import ReplayArtifact
from .schema import ReplayError, ReplayInput
from .session import ReplaySession
from .runner import Comparison, compare, reexecute
from .state_recording import (
    CompletedStepContext,
    DataRecord,
    RecordingProfile,
    ResultPlayback,
    StateRecorder,
    load_recording_manifest,
    load_supported_checkpoint,
    load_supported_simulation,
    register_data_restorer,
    register_recording_profile,
    restore_supported_simulation,
)

__all__ = [
    "ReplayArtifact",
    "ReplayError",
    "ReplayInput",
    "ReplaySession",
    "Comparison",
    "compare",
    "reexecute",
    "CompletedStepContext",
    "DataRecord",
    "RecordingProfile",
    "ResultPlayback",
    "StateRecorder",
    "load_recording_manifest",
    "load_supported_checkpoint",
    "load_supported_simulation",
    "register_data_restorer",
    "register_recording_profile",
    "restore_supported_simulation",
]
