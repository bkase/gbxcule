"""Backend implementations for GBxCuLE.

This module exports the shared backend contract and types. Backend implementations
are imported lazily to avoid heavy deps at import time.
"""

from gbxcule.backends.common import (
    ACTION_A,
    ACTION_B,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_SELECT,
    ACTION_START,
    ACTION_TO_BUTTON,
    ACTION_UP,
    NUM_ACTIONS,
    RESULT_SCHEMA_VERSION,
    ArraySpec,
    BackendSpec,
    CpuFlags,
    CpuState,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
    RunConfig,
    RunResult,
    Stage,
    StepOutput,
    SystemInfo,
    VecBackend,
    action_to_button,
    as_i32_actions,
    empty_obs,
    flags_from_f,
    run_artifact_to_json_dict,
    validate_actions,
    validate_step_output,
)

__all__ = [
    # Literals and type aliases
    "Device",
    "Stage",
    "NDArrayF32",
    "NDArrayI32",
    "NDArrayBool",
    # Core types
    "ArraySpec",
    "BackendSpec",
    "StepOutput",
    "CpuFlags",
    "CpuState",
    # Protocol
    "VecBackend",
    # Action constants
    "ACTION_NOOP",
    "ACTION_UP",
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_A",
    "ACTION_B",
    "ACTION_START",
    "ACTION_SELECT",
    "NUM_ACTIONS",
    "ACTION_TO_BUTTON",
    "action_to_button",
    # Run metadata
    "RESULT_SCHEMA_VERSION",
    "SystemInfo",
    "RunConfig",
    "RunResult",
    # Helpers
    "flags_from_f",
    "validate_actions",
    "as_i32_actions",
    "validate_step_output",
    "empty_obs",
    "run_artifact_to_json_dict",
]
