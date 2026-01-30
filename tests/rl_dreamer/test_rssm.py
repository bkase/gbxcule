from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.rssm import build_rssm, shift_actions  # noqa: E402


def _fixture_root() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "dreamer_v3"


def _load_meta() -> dict[str, int | float]:
    path = _fixture_root() / "rssm_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _build_from_meta(meta: dict[str, int | float]):
    return build_rssm(
        action_dim=int(meta["action_dim"]),
        embed_dim=int(meta["embed_dim"]),
        stochastic_size=int(meta["stochastic_size"]),
        discrete_size=int(meta["discrete_size"]),
        recurrent_state_size=int(meta["recurrent_state_size"]),
        dense_units=int(meta["dense_units"]),
        hidden_size=int(meta["hidden_size"]),
        unimix=float(meta["unimix"]),
        layer_norm_eps=float(meta["layer_norm_eps"]),
        activation="torch.nn.SiLU",
        learnable_initial_recurrent_state=True,
        hafner_init=True,
        rnn_dtype=torch.float32,
    )


def test_scan_shapes() -> None:
    meta = _load_meta()
    model = _build_from_meta(meta)
    T = int(meta["seq_len"])
    B = int(meta["batch_size"])
    actions = torch.randn(T, B, int(meta["action_dim"]))
    embedded_obs = torch.randn(T, B, int(meta["embed_dim"]))
    is_first = torch.zeros(T, B, 1)
    out = model.scan(actions, embedded_obs, is_first, sample_state=False)
    assert out["recurrent_state"].shape == (
        T,
        B,
        int(meta["recurrent_state_size"]),
    )
    assert out["priors"].shape == (
        T,
        B,
        int(meta["stochastic_size"]),
        int(meta["discrete_size"]),
    )


def test_is_first_reset_masking() -> None:
    meta = _load_meta()
    model = _build_from_meta(meta)
    state_dict = torch.load(_fixture_root() / "rssm_state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    B = int(meta["batch_size"])
    action = torch.randn(1, B, int(meta["action_dim"]))
    embedded_obs = torch.randn(1, B, int(meta["embed_dim"]))
    posterior_logits = torch.randn(
        1,
        B,
        int(meta["stochastic_size"]),
        int(meta["discrete_size"]),
    )
    posterior = torch.nn.functional.one_hot(
        posterior_logits.argmax(-1), num_classes=int(meta["discrete_size"])
    ).float()
    recurrent_state = torch.randn(1, B, int(meta["recurrent_state_size"]))
    is_first = torch.ones(1, B, 1)

    out_reset = model.dynamic(
        posterior,
        recurrent_state,
        action,
        embedded_obs,
        is_first,
        sample_state=False,
    )

    init_h, init_post = model.get_initial_states((1, B))
    out_expected = model.dynamic(
        init_post,
        init_h,
        torch.zeros_like(action),
        embedded_obs,
        torch.zeros_like(is_first),
        sample_state=False,
    )

    assert torch.allclose(out_reset[0], out_expected[0], atol=1e-5, rtol=1e-4)
    assert torch.allclose(out_reset[1], out_expected[1], atol=1e-5, rtol=1e-4)


def test_unimix_changes_logits() -> None:
    meta = _load_meta()
    model = _build_from_meta(meta)
    logits = torch.randn(
        2, 3, int(meta["stochastic_size"]) * int(meta["discrete_size"])
    )
    mixed = model._uniform_mix(logits)
    assert mixed.shape == logits.shape
    assert not torch.allclose(mixed, logits)


def test_shift_actions() -> None:
    actions = torch.arange(12, dtype=torch.float32).view(3, 2, 2)
    shifted = shift_actions(actions)
    assert torch.allclose(shifted[0], torch.zeros_like(actions[0]))
    assert torch.allclose(shifted[1:], actions[:-1])


def test_amp_stability() -> None:
    if not hasattr(torch, "autocast"):
        pytest.skip("torch.autocast not available")
    meta = _load_meta()
    model = _build_from_meta(meta)
    T = int(meta["seq_len"])
    B = int(meta["batch_size"])
    actions = torch.randn(T, B, int(meta["action_dim"]))
    embedded_obs = torch.randn(T, B, int(meta["embed_dim"]))
    is_first = torch.zeros(T, B, 1)
    baseline = model.scan(actions, embedded_obs, is_first, sample_state=False)

    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model.scan(actions, embedded_obs, is_first, sample_state=False)
    except (RuntimeError, TypeError):
        pytest.skip("autocast not supported on this device")

    assert torch.isfinite(out["recurrent_state"]).all()
    max_diff = (out["recurrent_state"] - baseline["recurrent_state"]).abs().max().item()
    assert max_diff < 1e-2


def test_fixture_parity() -> None:
    meta = _load_meta()
    model = _build_from_meta(meta)
    state_dict = torch.load(_fixture_root() / "rssm_state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dyn = np.load(_fixture_root() / "rssm_dynamic_step.npz")
    posterior = torch.from_numpy(dyn["posterior"]).float()
    recurrent_state = torch.from_numpy(dyn["recurrent_state"]).float()
    action = torch.from_numpy(dyn["action"]).float()
    embedded_obs = torch.from_numpy(dyn["embedded_obs"]).float()
    is_first = torch.from_numpy(dyn["is_first"]).float()

    out = model.dynamic(
        posterior,
        recurrent_state,
        action,
        embedded_obs,
        is_first,
        sample_state=False,
    )
    assert torch.allclose(
        out[0],
        torch.from_numpy(dyn["next_recurrent_state"]).float(),
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        out[1], torch.from_numpy(dyn["next_posterior"]).float(), atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        out[2], torch.from_numpy(dyn["prior"]).float(), atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        out[3], torch.from_numpy(dyn["posterior_logits"]).float(), atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        out[4], torch.from_numpy(dyn["prior_logits"]).float(), atol=1e-5, rtol=1e-4
    )

    scan = np.load(_fixture_root() / "rssm_scan.npz")
    scan_out = model.scan(
        torch.from_numpy(scan["actions"]).float(),
        torch.from_numpy(scan["embedded_obs"]).float(),
        torch.from_numpy(scan["is_first"]).float(),
        sample_state=False,
    )
    assert torch.allclose(
        scan_out["recurrent_state"],
        torch.from_numpy(scan["recurrent_states"]).float(),
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        scan_out["priors_logits"],
        torch.from_numpy(scan["priors_logits"]).float(),
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        scan_out["posteriors_logits"],
        torch.from_numpy(scan["posteriors_logits"]).float(),
        atol=1e-5,
        rtol=1e-4,
    )

    init = np.load(_fixture_root() / "rssm_initial_state.npz")
    init_h, init_post = model.get_initial_states((1, int(meta["batch_size"])))
    assert torch.allclose(
        init_h,
        torch.from_numpy(init["initial_recurrent_state"]).float(),
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        init_post,
        torch.from_numpy(init["initial_posterior"]).float(),
        atol=1e-5,
        rtol=1e-4,
    )
