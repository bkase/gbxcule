"""Streaming A2C (TD(0)) core utilities (torch, GPU-friendly)."""

from __future__ import annotations

from typing import Any

from gbxcule.rl.ppo import logprob_from_logits


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.a2c. Install with `uv sync --group rl`."
        ) from exc


def a2c_td0_losses(  # type: ignore[no-untyped-def]
    logits,
    actions,
    values,
    rewards,
    done,
    trunc,
    v_next,
    *,
    gamma: float,
    value_coef: float,
    entropy_coef: float,
):
    """Compute streaming A2C TD(0) losses and metrics."""
    torch = _require_torch()
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [N, num_actions]")
    if logits.dtype is not torch.float32:
        raise ValueError("logits must be float32")
    if actions.ndim != 1:
        raise ValueError("actions must be 1D [N]")
    if actions.dtype not in (torch.int64, torch.int32):
        raise ValueError("actions must be int64 or int32")
    if values.ndim != 1:
        raise ValueError("values must be 1D [N]")
    if values.dtype is not torch.float32:
        raise ValueError("values must be float32")
    if rewards.ndim != 1:
        raise ValueError("rewards must be 1D [N]")
    if rewards.dtype is not torch.float32:
        raise ValueError("rewards must be float32")
    if done.ndim != 1 or trunc.ndim != 1:
        raise ValueError("done and trunc must be 1D [N]")
    if done.dtype is not torch.bool or trunc.dtype is not torch.bool:
        raise ValueError("done and trunc must be bool")
    if v_next.ndim != 1:
        raise ValueError("v_next must be 1D [N]")
    if v_next.dtype is not torch.float32:
        raise ValueError("v_next must be float32")
    if logits.shape[0] != actions.shape[0]:
        raise ValueError("logits/actions batch mismatch")
    if values.shape != rewards.shape or values.shape != done.shape:
        raise ValueError("values/rewards/done shape mismatch")
    if trunc.shape != done.shape or v_next.shape != done.shape:
        raise ValueError("trunc/v_next shape mismatch")

    not_done = (~(done | trunc)).to(torch.float32)
    target = rewards + float(gamma) * not_done * v_next.detach()
    adv = target - values

    logp = logprob_from_logits(logits, actions)
    policy_loss = -(logp * adv.detach()).mean()
    value_loss = (target.detach() - values).pow(2).mean() * float(value_coef)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -float(entropy_coef) * entropy

    total_loss = policy_loss + value_loss + entropy_loss

    return {
        "loss_total": total_loss,
        "loss_policy": policy_loss,
        "loss_value": value_loss,
        "loss_entropy": entropy_loss,
        "entropy": entropy,
    }
