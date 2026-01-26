"""PPO-lite core utilities (torch, GPU-friendly)."""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl. Install with `uv sync --group rl`."
        ) from exc


def logprob_from_logits(logits, actions):  # type: ignore[no-untyped-def]
    """Compute log-prob of chosen actions from logits."""
    torch = _require_torch()
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [batch, num_actions]")
    if actions.ndim != 1:
        raise ValueError("actions must be 1D [batch]")
    if logits.shape[0] != actions.shape[0]:
        raise ValueError("logits and actions batch size mismatch")
    if actions.dtype not in (torch.int64, torch.int32):
        raise ValueError("actions must be int64 or int32")
    if logits.dtype is not torch.float32:
        raise ValueError("logits must be float32")
    actions_i64 = actions.to(torch.int64)
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(1, actions_i64.unsqueeze(1)).squeeze(1)


def compute_gae(  # type: ignore[no-untyped-def]
    rewards,
    values,
    dones,
    last_value,
    *,
    gamma: float,
    gae_lambda: float,
):
    """Compute GAE advantages and returns."""
    torch = _require_torch()
    if rewards.ndim != 2 or values.ndim != 2 or dones.ndim != 2:
        raise ValueError("rewards, values, dones must be 2D [T, N]")
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, dones must have matching shapes")
    if rewards.dtype is not torch.float32:
        raise ValueError("rewards must be float32")
    if values.dtype is not torch.float32:
        raise ValueError("values must be float32")
    if dones.dtype is not torch.bool:
        raise ValueError("dones must be bool")
    if last_value.ndim != 1 or last_value.shape[0] != rewards.shape[1]:
        raise ValueError("last_value must be 1D [N]")
    if last_value.dtype is not torch.float32:
        raise ValueError("last_value must be float32")

    steps, num_envs = rewards.shape
    advantages = torch.zeros(
        (steps, num_envs), device=rewards.device, dtype=torch.float32
    )
    gae = torch.zeros((num_envs,), device=rewards.device, dtype=torch.float32)

    for t in range(steps - 1, -1, -1):
        not_done = (~dones[t]).to(torch.float32)
        next_value = last_value if t == steps - 1 else values[t + 1]
        delta = rewards[t] + float(gamma) * next_value * not_done - values[t]
        gae = delta + float(gamma) * float(gae_lambda) * not_done * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_losses(  # type: ignore[no-untyped-def]
    logits,
    actions,
    old_logprobs,
    returns,
    advantages,
    values,
    *,
    clip: float,
    value_coef: float,
    entropy_coef: float,
    normalize_adv: bool = True,
    eps: float = 1e-8,
):
    """Compute PPO losses and metrics for a batch."""
    torch = _require_torch()
    if logits.ndim not in (2, 3):
        raise ValueError("logits must be 2D or 3D")
    if logits.dtype is not torch.float32:
        raise ValueError("logits must be float32")
    if actions.dtype not in (torch.int64, torch.int32):
        raise ValueError("actions must be int64 or int32")
    if old_logprobs.dtype is not torch.float32:
        raise ValueError("old_logprobs must be float32")
    if returns.dtype is not torch.float32:
        raise ValueError("returns must be float32")
    if advantages.dtype is not torch.float32:
        raise ValueError("advantages must be float32")
    if values.dtype is not torch.float32:
        raise ValueError("values must be float32")

    if logits.ndim == 3:
        steps, num_envs, num_actions = logits.shape
        batch = steps * num_envs
        logits = logits.reshape(batch, num_actions)
        actions = actions.reshape(batch)
        old_logprobs = old_logprobs.reshape(batch)
        returns = returns.reshape(batch)
        advantages = advantages.reshape(batch)
        values = values.reshape(batch)

    if logits.ndim != 2:
        raise ValueError("logits must be 2D after flattening")
    if actions.ndim != 1:
        raise ValueError("actions must be 1D after flattening")
    if logits.shape[0] != actions.shape[0]:
        raise ValueError("logits/actions batch mismatch")

    new_logprobs = logprob_from_logits(logits, actions)
    if old_logprobs.shape != new_logprobs.shape:
        raise ValueError("old_logprobs shape mismatch")
    if returns.shape != new_logprobs.shape:
        raise ValueError("returns shape mismatch")
    if advantages.shape != new_logprobs.shape:
        raise ValueError("advantages shape mismatch")
    if values.shape != new_logprobs.shape:
        raise ValueError("values shape mismatch")

    if normalize_adv:
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        advantages = (advantages - adv_mean) / (adv_std + float(eps))

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip), 1.0 + float(clip))
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()

    value_loss = ((returns - values) ** 2).mean() * float(value_coef)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -float(entropy_coef) * entropy

    total_loss = policy_loss + value_loss + entropy_loss

    approx_kl = (old_logprobs - new_logprobs).mean()
    clipfrac = (torch.abs(ratio - 1.0) > float(clip)).to(torch.float32).mean()

    return {
        "loss_total": total_loss,
        "loss_policy": policy_loss,
        "loss_value": value_loss,
        "loss_entropy": entropy_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
    }
