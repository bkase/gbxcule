"""RSSM core for Dreamer v3 (M3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gbxcule.rl.dreamer_v3.init import init_weights, uniform_init_weights
from gbxcule.rl.dreamer_v3.mlp import MLP, LayerNorm


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


torch = _require_torch()


def probs_to_logits(probs):  # type: ignore[no-untyped-def]
    from torch.distributions.utils import probs_to_logits as torch_probs_to_logits

    return torch_probs_to_logits(probs, is_binary=False)


def compute_stochastic_state(logits, *, discrete: int, sample: bool = True):  # type: ignore[no-untyped-def]
    from torch.distributions import Independent, OneHotCategoricalStraightThrough

    device_type = logits.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        logits32 = logits.float().view(*logits.shape[:-1], -1, discrete)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits32), 1)
        out = dist.rsample() if sample else dist.mode
    return out


class LayerNormGRUCell(torch.nn.Module):  # type: ignore[misc]
    """GRU cell with LayerNorm, FP32 internal math (matches sheeprl)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm_cls=LayerNorm,
        layer_norm_kw: dict[str, Any] | None = None,
        output_dtype=None,
    ) -> None:
        super().__init__()
        if layer_norm_kw is None:
            layer_norm_kw = {}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = torch.nn.Linear(
            input_size + hidden_size, 3 * hidden_size, bias=bias
        )
        layer_norm_kw = dict(layer_norm_kw)
        layer_norm_kw.pop("normalized_shape", None)
        self.layer_norm = layer_norm_cls(3 * hidden_size, **layer_norm_kw)
        self.output_dtype = output_dtype

    def forward(self, input, hx=None):  # type: ignore[no-untyped-def]
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected 3-D input with sequence length 1, "
                    f"received length {input.shape[int(self.batch_first)]}"
                )
        if hx is not None and hx.dim() == 3:
            hx = hx.squeeze(0)
        if input.dim() not in (1, 2):
            raise AssertionError(
                f"LayerNormGRUCell: Expected input dim 1 or 2, received {input.dim()}"
            )

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        elif not is_batched:
            hx = hx.unsqueeze(0)

        device_type = input.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            input32 = input.float()
            hx32 = hx.float()
            x = torch.cat((hx32, input32), -1)
            x = self.linear(x)
            x = self.layer_norm(x)
            reset, cand, update = torch.chunk(x, 3, -1)
            reset = torch.sigmoid(reset)
            cand = torch.tanh(reset * cand)
            update = torch.sigmoid(update - 1)
            hx32 = update * cand + (1 - update) * hx32

        out = hx32 if self.output_dtype is None else hx32.to(self.output_dtype)
        if not is_batched:
            out = out.squeeze(0)
        elif is_3d:
            out = out.unsqueeze(0)
        return out


class RecurrentModel(torch.nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        input_size: int,
        recurrent_state_size: int,
        dense_units: int,
        activation_fn=None,
        layer_norm_cls=LayerNorm,
        layer_norm_kw: dict[str, Any] | None = None,
        rnn_output_dtype=None,
    ) -> None:
        super().__init__()
        if activation_fn is None:
            activation_fn = torch.nn.SiLU
        if layer_norm_kw is None:
            layer_norm_kw = {"eps": 1e-3}
        self.mlp = MLP(
            input_dims=input_size,
            output_dim=None,
            hidden_sizes=[dense_units],
            activation=activation_fn,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw={**layer_norm_kw, "normalized_shape": dense_units},
            bias=layer_norm_cls in (None, torch.nn.Identity),
        )
        self.rnn = LayerNormGRUCell(
            dense_units,
            recurrent_state_size,
            bias=False,
            batch_first=False,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw,
            output_dtype=rnn_output_dtype,
        )
        self.recurrent_state_size = recurrent_state_size

    def forward(self, input, recurrent_state):  # type: ignore[no-untyped-def]
        feat = self.mlp(input)
        return self.rnn(feat, recurrent_state)


class RSSM(torch.nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        recurrent_model,
        representation_model,
        transition_model,
        distribution_cfg: dict[str, Any] | None = None,
        discrete: int = 32,
        unimix: float = 0.01,
        learnable_initial_recurrent_state: bool = True,
        rnn_dtype=None,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.transition_model = transition_model
        self.distribution_cfg = distribution_cfg or {}
        self.discrete = discrete
        self.unimix = unimix
        self.rnn_dtype = rnn_dtype or torch.float32
        if learnable_initial_recurrent_state:
            self.initial_recurrent_state = torch.nn.Parameter(
                torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "initial_recurrent_state",
                torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32),
            )

    def get_initial_states(self, batch_shape: Sequence[int] | torch.Size):  # type: ignore[no-untyped-def]
        initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(
            *batch_shape, -1
        )
        initial_recurrent_state = initial_recurrent_state.to(self.rnn_dtype)
        initial_posterior = self._transition(
            initial_recurrent_state, sample_state=False
        )[1]
        return initial_recurrent_state, initial_posterior

    def _uniform_mix(self, logits):  # type: ignore[no-untyped-def]
        dim = logits.dim()
        if dim == 2:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim == 3:
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
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def _representation(self, recurrent_state, embedded_obs, sample_state: bool = True):  # type: ignore[no-untyped-def]
        logits = self.representation_model(
            torch.cat((recurrent_state, embedded_obs), -1)
        )
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(
            logits, discrete=self.discrete, sample=sample_state
        )

    def _transition(self, recurrent_out, sample_state: bool = True):  # type: ignore[no-untyped-def]
        logits = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(
            logits, discrete=self.discrete, sample=sample_state
        )

    def dynamic(  # type: ignore[no-untyped-def]
        self,
        posterior,
        recurrent_state,
        action,
        embedded_obs,
        is_first,
        *,
        sample_state: bool = True,
    ):
        if is_first.dtype is torch.bool:
            is_first = is_first.to(recurrent_state.dtype)
        while is_first.dim() < recurrent_state.dim():
            is_first = is_first.unsqueeze(-1)
        action = (1 - is_first) * action
        initial_recurrent_state, initial_posterior = self.get_initial_states(
            recurrent_state.shape[:2]
        )
        recurrent_state = (
            1 - is_first
        ) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(
            posterior
        )
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

    def imagination(
        self, prior, recurrent_state, actions, *, sample_state: bool = True
    ):  # type: ignore[no-untyped-def]
        recurrent_state = self.recurrent_model(
            torch.cat((prior, actions), -1), recurrent_state
        )
        _, imagined_prior = self._transition(recurrent_state, sample_state=sample_state)
        return imagined_prior, recurrent_state

    def scan(  # type: ignore[no-untyped-def]
        self,
        actions,
        embedded_obs,
        is_first,
        *,
        force_first_reset: bool = False,
        sample_state: bool = True,
    ):
        if actions.dim() < 2:
            raise ValueError("actions must be time-major [T, B, ...]")
        if embedded_obs.shape[:2] != actions.shape[:2]:
            raise ValueError("embedded_obs must match actions shape [T, B, ...]")
        if is_first.shape[:2] != actions.shape[:2]:
            raise ValueError("is_first must match actions shape [T, B]")

        if force_first_reset:
            is_first = is_first.clone()
            is_first[0] = torch.ones_like(is_first[0])

        T, B = actions.shape[:2]
        device = actions.device
        stoch_dim = self.transition_model.output_dim
        stochastic_size = int(stoch_dim / self.discrete)
        recurrent_state = torch.zeros(
            1,
            B,
            self.recurrent_model.recurrent_state_size,
            device=device,
            dtype=self.rnn_dtype,
        )
        posterior = torch.zeros(
            1, B, stochastic_size, self.discrete, device=device, dtype=actions.dtype
        )
        recurrent_states = torch.empty(
            T,
            B,
            self.recurrent_model.recurrent_state_size,
            device=device,
            dtype=self.rnn_dtype,
        )
        priors_logits = torch.empty(T, B, stoch_dim, device=device, dtype=actions.dtype)
        posteriors_logits = torch.empty(
            T, B, stoch_dim, device=device, dtype=actions.dtype
        )
        priors = torch.empty(
            T, B, stochastic_size, self.discrete, device=device, dtype=actions.dtype
        )
        posteriors = torch.empty(
            T, B, stochastic_size, self.discrete, device=device, dtype=actions.dtype
        )

        for t in range(T):
            recurrent_state, posterior, prior, posterior_logits, prior_logits = (
                self.dynamic(
                    posterior,
                    recurrent_state,
                    actions[t : t + 1],
                    embedded_obs[t : t + 1],
                    is_first[t : t + 1],
                    sample_state=sample_state,
                )
            )
            recurrent_states[t] = recurrent_state
            priors_logits[t] = prior_logits
            posteriors_logits[t] = posterior_logits
            priors[t] = prior
            posteriors[t] = posterior

        return {
            "recurrent_state": recurrent_states,
            "priors_logits": priors_logits,
            "posteriors_logits": posteriors_logits,
            "priors": priors,
            "posteriors": posteriors,
        }


class DecoupledRSSM(RSSM):  # type: ignore[misc]
    def dynamic(
        self,
        posterior,
        recurrent_state,
        action,
        embedded_obs,
        is_first,
        *,
        sample_state: bool = True,
    ):  # type: ignore[override,no-untyped-def]
        if is_first.dtype is torch.bool:
            is_first = is_first.to(recurrent_state.dtype)
        while is_first.dim() < recurrent_state.dim():
            is_first = is_first.unsqueeze(-1)
        action = (1 - is_first) * action
        initial_recurrent_state, initial_posterior = self.get_initial_states(
            recurrent_state.shape[:2]
        )
        recurrent_state = (
            1 - is_first
        ) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(
            posterior
        )
        recurrent_state = self.recurrent_model(
            torch.cat((posterior, action), -1), recurrent_state
        )
        prior_logits, prior = self._transition(
            recurrent_state, sample_state=sample_state
        )
        # Decoupled RSSM does not compute posterior here; reuse prior for shape parity.
        return recurrent_state, prior, prior, prior_logits, prior_logits

    def _representation(self, embedded_obs, sample_state: bool = True):  # type: ignore[override,no-untyped-def]
        logits = self.representation_model(embedded_obs)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(
            logits, discrete=self.discrete, sample=sample_state
        )


def shift_actions(actions):  # type: ignore[no-untyped-def]
    """Prepend zero action and drop last to align with Dreamer dynamics."""
    if actions.dim() < 2:
        raise ValueError("actions must be time-major [T, B, ...]")
    zero = torch.zeros_like(actions[:1])
    return torch.cat((zero, actions[:-1]), dim=0)


def build_rssm(  # type: ignore[no-untyped-def]
    *,
    action_dim: int,
    embed_dim: int,
    stochastic_size: int,
    discrete_size: int,
    recurrent_state_size: int,
    dense_units: int,
    hidden_size: int,
    unimix: float = 0.01,
    layer_norm_eps: float = 1e-3,
    activation: str | Any = "torch.nn.SiLU",
    learnable_initial_recurrent_state: bool = True,
    hafner_init: bool = True,
    rnn_dtype=None,
):
    activation_cls = _resolve_activation(activation)
    layer_norm_kw = {"eps": layer_norm_eps}
    stoch_dim = stochastic_size * discrete_size
    recurrent_model = RecurrentModel(
        input_size=int(action_dim + stoch_dim),
        recurrent_state_size=recurrent_state_size,
        dense_units=dense_units,
        activation_fn=activation_cls,
        layer_norm_cls=LayerNorm,
        layer_norm_kw=layer_norm_kw,
        rnn_output_dtype=rnn_dtype,
    )
    representation_model = MLP(
        input_dims=recurrent_state_size + embed_dim,
        output_dim=stoch_dim,
        hidden_sizes=[hidden_size],
        activation=activation_cls,
        layer_norm_cls=LayerNorm,
        layer_norm_kw={**layer_norm_kw, "normalized_shape": hidden_size},
        bias=False,
        flatten_dim=None,
    )
    transition_model = MLP(
        input_dims=recurrent_state_size,
        output_dim=stoch_dim,
        hidden_sizes=[hidden_size],
        activation=activation_cls,
        layer_norm_cls=LayerNorm,
        layer_norm_kw={**layer_norm_kw, "normalized_shape": hidden_size},
        bias=False,
        flatten_dim=None,
    )

    recurrent_model.apply(init_weights)
    representation_model.apply(init_weights)
    transition_model.apply(init_weights)
    if hafner_init:
        representation_model.model[-1].apply(uniform_init_weights(1.0))
        transition_model.model[-1].apply(uniform_init_weights(1.0))

    return RSSM(
        recurrent_model=recurrent_model,
        representation_model=representation_model,
        transition_model=transition_model,
        distribution_cfg={},
        discrete=discrete_size,
        unimix=unimix,
        learnable_initial_recurrent_state=learnable_initial_recurrent_state,
        rnn_dtype=rnn_dtype or torch.float32,
    )


def _resolve_activation(spec: Any):  # type: ignore[no-untyped-def]
    if isinstance(spec, str):
        lowered = spec.lower()
        if "silu" in lowered:
            return torch.nn.SiLU
        if "relu" in lowered:
            return torch.nn.ReLU
        if "elu" in lowered:
            return torch.nn.ELU
        raise ValueError(f"Unknown activation spec: {spec}")
    if isinstance(spec, type) and issubclass(spec, torch.nn.Module):
        return spec
    if callable(spec):
        return spec
    raise ValueError(f"Unsupported activation spec: {spec}")
