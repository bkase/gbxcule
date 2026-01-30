"""World model training utilities for Dreamer v3 (M4)."""

from __future__ import annotations

from typing import Any

from gbxcule.rl.dreamer_v3.world_model import WorldModel


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


def _ensure_binary(value, name: str) -> None:  # type: ignore[no-untyped-def]
    torch = _require_torch()
    if not torch.is_floating_point(value):
        value = value.to(torch.float32)
    unique = torch.unique(value)
    if not torch.all((unique == 0) | (unique == 1)):
        raise ValueError(f"{name} must be 0/1 values")


def wm_train_step(
    batch: dict[str, Any],
    model: WorldModel,
    optimizer,  # type: ignore[no-untyped-def]
    *,
    obs_format: str,
    kl_dynamic: float,
    kl_representation: float,
    kl_free_nats: float,
    kl_regularizer: float,
    action_dim: int | None = None,
):
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    is_first = batch["is_first"]
    continues = batch.get("continue")
    if continues is None:
        terminated = batch.get("terminated")
        if terminated is None:
            raise KeyError("batch must include 'continue' or 'terminated'")
        continues = 1 - terminated
    if continues.ndim == 2:
        continues = continues.unsqueeze(-1)
    if rewards.ndim == 2:
        rewards = rewards.unsqueeze(-1)
    _ensure_binary(continues, "continue targets")
    enc_obs, loss_obs = model.prepare_obs(obs, obs_format=obs_format)
    outputs = model(enc_obs, actions, is_first, action_dim=action_dim)
    metrics = model.loss(
        outputs,
        loss_obs,
        rewards,
        continues,
        kl_dynamic=kl_dynamic,
        kl_representation=kl_representation,
        kl_free_nats=kl_free_nats,
        kl_regularizer=kl_regularizer,
    )
    optimizer.zero_grad(set_to_none=True)
    metrics.loss.backward()
    optimizer.step()
    return metrics
