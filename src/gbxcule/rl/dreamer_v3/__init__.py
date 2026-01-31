"""Dreamer v3 scaffolding and core components."""

from __future__ import annotations

from gbxcule.rl.dreamer_v3.behavior import Actor, BehaviorLosses, Critic, behavior_step
from gbxcule.rl.dreamer_v3.config import DreamerV3Config
from gbxcule.rl.dreamer_v3.dists import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    SymlogTwoHot,
    TwoHotEncodingDistribution,
)
from gbxcule.rl.dreamer_v3.imagination import ImaginationOutput, imagine_rollout
from gbxcule.rl.dreamer_v3.ingest_cuda import ReplayIngestorCUDA
from gbxcule.rl.dreamer_v3.math import symexp, symlog, twohot, twohot_to_value
from gbxcule.rl.dreamer_v3.replay import ReplayRing
from gbxcule.rl.dreamer_v3.replay_commit import ReplayCommitManager
from gbxcule.rl.dreamer_v3.replay_cuda import ReplayRingCUDA
from gbxcule.rl.dreamer_v3.return_ema import ReturnEMA
from gbxcule.rl.dreamer_v3.returns import lambda_returns
from gbxcule.rl.dreamer_v3.rssm import RSSM, DecoupledRSSM, build_rssm, shift_actions
from gbxcule.rl.dreamer_v3.targets import maybe_update_target, update_target
from gbxcule.rl.dreamer_v3.unpack import unpack_packed2

__all__ = [
    "Actor",
    "BehaviorLosses",
    "BernoulliSafeMode",
    "Critic",
    "DreamerV3Config",
    "DecoupledRSSM",
    "ImaginationOutput",
    "MSEDistribution",
    "ReplayCommitManager",
    "ReplayIngestorCUDA",
    "ReplayRing",
    "ReplayRingCUDA",
    "ReturnEMA",
    "RSSM",
    "SymlogDistribution",
    "SymlogTwoHot",
    "TwoHotEncodingDistribution",
    "behavior_step",
    "build_rssm",
    "imagine_rollout",
    "lambda_returns",
    "maybe_update_target",
    "shift_actions",
    "symexp",
    "symlog",
    "twohot",
    "twohot_to_value",
    "update_target",
    "unpack_packed2",
]
