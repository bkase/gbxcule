"""Backend implementations for GBxCuLE.

This module exports the shared backend contract and types. Backend implementations
are imported lazily to avoid heavy deps at import time.
"""

from gbxcule.backends.common import (
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
