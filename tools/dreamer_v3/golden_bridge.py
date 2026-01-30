"""Golden Bridge fixture generator for Dreamer v3."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Dreamer v3 parity fixtures",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/dreamer_v3"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Fixture subset selector (all|m2|rssm)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=7,
        help="Number of bins for TwoHot fixtures (default: 7)",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=-20.0,
        help="Low bound for symlog bins",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=20.0,
        help="High bound for symlog bins",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest",
    )
    return parser.parse_args(argv)


def _ensure_reference_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sheeprl_path = repo_root / "third_party" / "sheeprl"
    sys.path.insert(0, str(sheeprl_path))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_sheeprl_models() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    sheeprl_root = repo_root / "third_party" / "sheeprl" / "sheeprl"
    sheeprl_pkg = types.ModuleType("sheeprl")
    sheeprl_pkg.__path__ = [str(sheeprl_root)]
    sys.modules.setdefault("sheeprl", sheeprl_pkg)

    utils_pkg = types.ModuleType("sheeprl.utils")
    utils_pkg.__path__ = [str(sheeprl_root / "utils")]
    sys.modules.setdefault("sheeprl.utils", utils_pkg)

    models_pkg = types.ModuleType("sheeprl.models")
    models_pkg.__path__ = [str(sheeprl_root / "models")]
    sys.modules.setdefault("sheeprl.models", models_pkg)

    _load_module("sheeprl.utils.model", sheeprl_root / "utils" / "model.py")
    return _load_module("sheeprl.models.models", sheeprl_root / "models" / "models.py")


def _write_manifest(
    out_dir: Path, entries: list[dict[str, Any]], *, force: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _require_torch():  # type: ignore[no-untyped-def]
    try:
        import torch
    except Exception as exc:  # pragma: no cover - CLI only
        raise RuntimeError("torch is required to generate Dreamer v3 fixtures") from exc
    return torch


def _to_numpy(tensor):  # type: ignore[no-untyped-def]
    return tensor.detach().cpu().numpy()


def _generate_m2_fixtures(
    out_dir: Path, *, bins: int, low: float, high: float
) -> list[dict[str, Any]]:
    torch = _require_torch()
    from gbxcule.rl.dreamer_v3 import math as dv3_math
    from gbxcule.rl.dreamer_v3.dists import SymlogTwoHot
    from gbxcule.rl.dreamer_v3.return_ema import ReturnEMA

    out_dir.mkdir(parents=True, exist_ok=True)

    bins_tensor = torch.linspace(
        float(low), float(high), int(bins), dtype=torch.float32
    )
    _write_json(
        out_dir / "bins.json",
        {"bins": bins_tensor.tolist(), "low": low, "high": high},
    )

    symlog_inputs = torch.tensor(
        [-1000.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 1000.0],
        dtype=torch.float32,
    )
    symlog_values = dv3_math.symlog(symlog_inputs)
    symexp_values = dv3_math.symexp(symlog_values)
    symlog_cases = [
        {"x": float(x), "symlog": float(y), "symexp": float(z)}
        for x, y, z in zip(
            symlog_inputs.tolist(),
            symlog_values.tolist(),
            symexp_values.tolist(),
            strict=True,
        )
    ]
    _write_json(out_dir / "symlog_cases.json", {"cases": symlog_cases})

    twohot_inputs = torch.tensor([-1e9, -10.0, 0.0, 10.0, 1e9], dtype=torch.float32)
    twohot_cases = []
    for x in twohot_inputs:
        y = dv3_math.symlog(x)
        weights = dv3_math.twohot(y, bins_tensor)
        twohot_cases.append(
            {"x": float(x), "y": float(y), "weights": weights.squeeze(0).tolist()}
        )
    _write_json(out_dir / "twohot_cases.json", {"cases": twohot_cases})

    logits = torch.tensor(
        [
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([-2.0, 3.0], dtype=torch.float32).unsqueeze(-1)
    dist = SymlogTwoHot(logits, bins=bins_tensor, dims=1)
    log_prob = dist.log_prob(targets).tolist()
    mean = dist.mean.squeeze(-1).tolist()
    _write_json(
        out_dir / "symlog_twohot_dist.json",
        {
            "logits": logits.tolist(),
            "targets": targets.squeeze(-1).tolist(),
            "log_prob": log_prob,
            "mean": mean,
        },
    )

    decay = 0.99
    max_value = 1.0
    p_low = 0.05
    p_high = 0.95
    batches = [
        [i * 100.0 for i in range(11)],
        [100.0 + i * 100.0 for i in range(11)],
    ]
    ema = ReturnEMA(decay=decay, percentiles=(p_low, p_high), max_value=max_value)
    updates = []
    for batch in batches:
        values = torch.tensor(batch, dtype=torch.float32)
        offset, invscale = ema.update(values)
        assert ema.high is not None
        updates.append(
            {
                "values": [float(v) for v in batch],
                "low": float(offset.item()),
                "high": float(ema.high.item()),
                "invscale": float(invscale.item()),
            }
        )
    return_ema_cases = {
        "config": {
            "decay": decay,
            "max_value": max_value,
            "percentiles": [p_low, p_high],
        },
        "updates": updates,
    }
    _write_json(out_dir / "return_ema_cases.json", return_ema_cases)

    entries = [
        {
            "name": "bins",
            "dtype": "float32",
            "shape": [int(bins)],
            "file": "bins.json",
            "notes": "linspace in symlog space",
        },
        {
            "name": "symlog_cases",
            "dtype": "json",
            "shape": [len(symlog_cases)],
            "file": "symlog_cases.json",
            "notes": "symlog/symexp roundtrip cases",
        },
        {
            "name": "twohot_cases",
            "dtype": "json",
            "shape": [len(twohot_cases)],
            "file": "twohot_cases.json",
            "notes": "twohot weights for selected values",
        },
        {
            "name": "symlog_twohot_dist",
            "dtype": "json",
            "shape": [],
            "file": "symlog_twohot_dist.json",
            "notes": "log_prob and mean for SymlogTwoHot distribution",
        },
        {
            "name": "return_ema_cases",
            "dtype": "json",
            "shape": [],
            "file": "return_ema_cases.json",
            "notes": "Moments/ReturnEMA updates",
        },
    ]
    return entries


def _generate_rssm_fixtures(out_dir: Path) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    from gbxcule.rl.dreamer_v3.init import init_weights, uniform_init_weights

    models = _bootstrap_sheeprl_models()
    LayerNorm = models.LayerNorm
    LayerNormGRUCell = models.LayerNormGRUCell
    MLP = models.MLP

    def _compute_stochastic_state(logits, *, discrete: int, sample: bool = True):
        from torch.distributions import Independent, OneHotCategoricalStraightThrough

        logits = logits.view(*logits.shape[:-1], -1, discrete)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return dist.rsample() if sample else dist.mode

    class RefRecurrentModel(torch.nn.Module):
        def __init__(
            self,
            input_size: int,
            recurrent_state_size: int,
            dense_units: int,
            layer_norm_eps: float,
        ) -> None:
            super().__init__()
            self.mlp = MLP(
                input_dims=input_size,
                output_dim=None,
                hidden_sizes=[dense_units],
                activation=torch.nn.SiLU,
                layer_args={"bias": False},
                flatten_dim=None,
                norm_layer=[LayerNorm],
                norm_args=[{"eps": layer_norm_eps, "normalized_shape": dense_units}],
            )
            self.rnn = LayerNormGRUCell(
                dense_units,
                recurrent_state_size,
                bias=False,
                batch_first=False,
                layer_norm_cls=LayerNorm,
                layer_norm_kw={"eps": layer_norm_eps},
            )
            self.recurrent_state_size = recurrent_state_size

        def forward(self, input, recurrent_state):  # type: ignore[no-untyped-def]
            feat = self.mlp(input)
            return self.rnn(feat, recurrent_state)

    class RefRSSM(torch.nn.Module):
        def __init__(
            self,
            *,
            recurrent_model: RefRecurrentModel,
            representation_model: torch.nn.Module,
            transition_model: torch.nn.Module,
            discrete: int,
            unimix: float,
        ) -> None:
            super().__init__()
            self.recurrent_model = recurrent_model
            self.representation_model = representation_model
            self.transition_model = transition_model
            self.discrete = discrete
            self.unimix = unimix
            self.initial_recurrent_state = torch.nn.Parameter(
                torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )

        def get_initial_states(self, batch_shape):  # type: ignore[no-untyped-def]
            initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(
                *batch_shape, -1
            )
            initial_posterior = self._transition(
                initial_recurrent_state, sample_state=False
            )[1]
            return initial_recurrent_state, initial_posterior

        def _uniform_mix(self, logits):  # type: ignore[no-untyped-def]
            dim = logits.dim()
            if dim == 3:
                logits = logits.view(*logits.shape[:-1], -1, self.discrete)
            elif dim == 4:
                pass
            else:
                raise RuntimeError(
                    f"The logits expected shape is 3 or 4: received a {dim}D tensor"
                )
            if self.unimix > 0.0:
                probs = logits.softmax(dim=-1)
                uniform = torch.ones_like(probs) / self.discrete
                probs = (1 - self.unimix) * probs + self.unimix * uniform
                from torch.distributions.utils import probs_to_logits

                logits = probs_to_logits(probs, is_binary=False)
            logits = logits.view(*logits.shape[:-2], -1)
            return logits

        def _representation(
            self, recurrent_state, embedded_obs, sample_state: bool = True
        ):  # type: ignore[no-untyped-def]
            logits = self.representation_model(
                torch.cat((recurrent_state, embedded_obs), -1)
            )
            logits = self._uniform_mix(logits)
            return logits, _compute_stochastic_state(
                logits, discrete=self.discrete, sample=sample_state
            )

        def _transition(self, recurrent_out, sample_state: bool = True):  # type: ignore[no-untyped-def]
            logits = self.transition_model(recurrent_out)
            logits = self._uniform_mix(logits)
            return logits, _compute_stochastic_state(
                logits, discrete=self.discrete, sample=sample_state
            )

        def dynamic(
            self,
            posterior,
            recurrent_state,
            action,
            embedded_obs,
            is_first,
            *,
            sample_state: bool = True,
        ):  # type: ignore[no-untyped-def]
            if is_first.dtype is torch.bool:
                is_first = is_first.to(recurrent_state.dtype)
            action = (1 - is_first) * action
            initial_recurrent_state, initial_posterior = self.get_initial_states(
                recurrent_state.shape[:2]
            )
            recurrent_state = (
                1 - is_first
            ) * recurrent_state + is_first * initial_recurrent_state
            posterior = posterior.view(*posterior.shape[:-2], -1)
            posterior = (
                1 - is_first
            ) * posterior + is_first * initial_posterior.view_as(posterior)
            recurrent_state = self.recurrent_model(
                torch.cat((posterior, action), -1), recurrent_state
            )
            prior_logits, prior = self._transition(
                recurrent_state, sample_state=sample_state
            )
            posterior_logits, posterior = self._representation(
                recurrent_state, embedded_obs, sample_state=sample_state
            )
            return recurrent_state, posterior, prior, posterior_logits, prior_logits

    meta = {
        "seed": 20260130,
        "stochastic_size": 2,
        "discrete_size": 3,
        "recurrent_state_size": 5,
        "action_dim": 4,
        "embed_dim": 6,
        "dense_units": 7,
        "hidden_size": 8,
        "unimix": 0.01,
        "layer_norm_eps": 1e-3,
        "hafner_init": True,
        "batch_size": 2,
        "seq_len": 3,
    }
    torch.manual_seed(meta["seed"])

    stoch_dim = meta["stochastic_size"] * meta["discrete_size"]
    recurrent_model = RefRecurrentModel(
        input_size=meta["action_dim"] + stoch_dim,
        recurrent_state_size=meta["recurrent_state_size"],
        dense_units=meta["dense_units"],
        layer_norm_eps=meta["layer_norm_eps"],
    )
    representation_model = MLP(
        input_dims=meta["recurrent_state_size"] + meta["embed_dim"],
        output_dim=stoch_dim,
        hidden_sizes=[meta["hidden_size"]],
        activation=torch.nn.SiLU,
        layer_args={"bias": False},
        flatten_dim=None,
        norm_layer=[LayerNorm],
        norm_args=[
            {"eps": meta["layer_norm_eps"], "normalized_shape": meta["hidden_size"]}
        ],
    )
    transition_model = MLP(
        input_dims=meta["recurrent_state_size"],
        output_dim=stoch_dim,
        hidden_sizes=[meta["hidden_size"]],
        activation=torch.nn.SiLU,
        layer_args={"bias": False},
        flatten_dim=None,
        norm_layer=[LayerNorm],
        norm_args=[
            {"eps": meta["layer_norm_eps"], "normalized_shape": meta["hidden_size"]}
        ],
    )

    rssm = RefRSSM(
        recurrent_model=recurrent_model.apply(init_weights),
        representation_model=representation_model.apply(init_weights),
        transition_model=transition_model.apply(init_weights),
        discrete=meta["discrete_size"],
        unimix=meta["unimix"],
    )
    if meta["hafner_init"]:
        transition_model.model[-1].apply(uniform_init_weights(1.0))
        representation_model.model[-1].apply(uniform_init_weights(1.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rssm_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(rssm.state_dict(), out_dir / "rssm_state_dict.pt")

    init_h, init_post = rssm.get_initial_states((1, meta["batch_size"]))
    np.savez(
        out_dir / "rssm_initial_state.npz",
        initial_recurrent_state=_to_numpy(init_h),
        initial_posterior=_to_numpy(init_post),
        initial_recurrent_param=_to_numpy(rssm.initial_recurrent_state),
    )

    torch.manual_seed(meta["seed"] + 1)
    posterior_logits = torch.randn(
        1,
        meta["batch_size"],
        meta["stochastic_size"],
        meta["discrete_size"],
    )
    posterior = torch.nn.functional.one_hot(
        posterior_logits.argmax(-1), num_classes=meta["discrete_size"]
    ).float()
    recurrent_state = torch.randn(1, meta["batch_size"], meta["recurrent_state_size"])
    action = torch.randn(1, meta["batch_size"], meta["action_dim"])
    embedded_obs = torch.randn(1, meta["batch_size"], meta["embed_dim"])
    is_first = torch.tensor([[1.0], [0.0]]).view(1, meta["batch_size"], 1)
    (
        next_recurrent_state,
        next_posterior,
        prior,
        posterior_logits_out,
        prior_logits_out,
    ) = rssm.dynamic(
        posterior,
        recurrent_state,
        action,
        embedded_obs,
        is_first,
        sample_state=False,
    )
    np.savez(
        out_dir / "rssm_dynamic_step.npz",
        posterior=_to_numpy(posterior),
        recurrent_state=_to_numpy(recurrent_state),
        action=_to_numpy(action),
        embedded_obs=_to_numpy(embedded_obs),
        is_first=_to_numpy(is_first),
        next_recurrent_state=_to_numpy(next_recurrent_state),
        next_posterior=_to_numpy(next_posterior),
        prior=_to_numpy(prior),
        posterior_logits=_to_numpy(posterior_logits_out),
        prior_logits=_to_numpy(prior_logits_out),
    )

    torch.manual_seed(meta["seed"] + 2)
    T = meta["seq_len"]
    B = meta["batch_size"]
    actions = torch.randn(T, B, meta["action_dim"])
    embedded_obs = torch.randn(T, B, meta["embed_dim"])
    is_first = torch.zeros(T, B, 1)
    is_first[0, :, :] = 1.0
    is_first[2, 0, :] = 1.0

    posterior = torch.zeros(1, B, meta["stochastic_size"], meta["discrete_size"])
    recurrent_state = torch.zeros(1, B, meta["recurrent_state_size"])
    recurrent_states = torch.empty(T, B, meta["recurrent_state_size"])
    priors_logits = torch.empty(T, B, stoch_dim)
    posteriors_logits = torch.empty(T, B, stoch_dim)
    priors = torch.empty(T, B, meta["stochastic_size"], meta["discrete_size"])
    posteriors = torch.empty(T, B, meta["stochastic_size"], meta["discrete_size"])
    for t in range(T):
        (
            recurrent_state,
            posterior,
            prior,
            posterior_logits_out,
            prior_logits_out,
        ) = rssm.dynamic(
            posterior,
            recurrent_state,
            actions[t : t + 1],
            embedded_obs[t : t + 1],
            is_first[t : t + 1],
            sample_state=False,
        )
        recurrent_states[t] = recurrent_state
        priors_logits[t] = prior_logits_out
        posteriors_logits[t] = posterior_logits_out
        priors[t] = prior
        posteriors[t] = posterior

    np.savez(
        out_dir / "rssm_scan.npz",
        actions=_to_numpy(actions),
        embedded_obs=_to_numpy(embedded_obs),
        is_first=_to_numpy(is_first),
        recurrent_states=_to_numpy(recurrent_states),
        priors_logits=_to_numpy(priors_logits),
        posteriors_logits=_to_numpy(posteriors_logits),
        priors=_to_numpy(priors),
        posteriors=_to_numpy(posteriors),
    )

    entries = [
        {
            "name": "rssm_meta",
            "dtype": "json",
            "shape": [],
            "file": "rssm_meta.json",
        },
        {
            "name": "rssm_state_dict",
            "dtype": "pt",
            "shape": [],
            "file": "rssm_state_dict.pt",
        },
        {
            "name": "rssm_initial_state",
            "dtype": "npz",
            "shape": [],
            "file": "rssm_initial_state.npz",
        },
        {
            "name": "rssm_dynamic_step",
            "dtype": "npz",
            "shape": [],
            "file": "rssm_dynamic_step.npz",
        },
        {
            "name": "rssm_scan",
            "dtype": "npz",
            "shape": [],
            "file": "rssm_scan.npz",
        },
    ]
    return entries


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    _ensure_reference_on_path()
    entries: list[dict[str, Any]] = []
    subset = args.subset.lower()
    if subset in ("all", "m2", "math", "dist"):
        entries.extend(
            _generate_m2_fixtures(
                args.out, bins=args.bins, low=args.low, high=args.high
            )
        )
    if subset in ("all", "m3", "rssm"):
        entries.extend(_generate_rssm_fixtures(args.out))
    _write_manifest(args.out, entries, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
