"""Dreamer v3 scaffolding and core components."""

from __future__ import annotations

from gbxcule.rl.dreamer_v3.config import DreamerV3Config
from gbxcule.rl.dreamer_v3.dists import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    SymlogTwoHot,
)
from gbxcule.rl.dreamer_v3.ingest_cuda import ReplayIngestorCUDA
from gbxcule.rl.dreamer_v3.math import symexp, symlog, twohot, twohot_to_value
from gbxcule.rl.dreamer_v3.replay import ReplayRing
from gbxcule.rl.dreamer_v3.replay_commit import ReplayCommitManager
from gbxcule.rl.dreamer_v3.replay_cuda import ReplayRingCUDA
from gbxcule.rl.dreamer_v3.return_ema import ReturnEMA
from gbxcule.rl.dreamer_v3.rssm import RSSM, DecoupledRSSM, build_rssm, shift_actions
from gbxcule.rl.dreamer_v3.unpack import unpack_packed2

__all__ = [
    "BernoulliSafeMode",
    "DreamerV3Config",
    "DecoupledRSSM",
    "MSEDistribution",
    "ReplayCommitManager",
    "ReplayIngestorCUDA",
    "ReplayRing",
    "ReplayRingCUDA",
    "ReturnEMA",
    "RSSM",
    "SymlogDistribution",
    "SymlogTwoHot",
    "build_rssm",
    "shift_actions",
    "symexp",
    "symlog",
    "twohot",
    "twohot_to_value",
    "unpack_packed2",
]
