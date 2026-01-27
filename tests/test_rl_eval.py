from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.eval import run_greedy_eval  # noqa: E402


class _ToyEvalEnv:
    def __init__(self, done_at: list[int], trunc_at: list[int]) -> None:
        if len(done_at) != len(trunc_at):
            raise ValueError("done_at and trunc_at must match length")
        self.num_envs = len(done_at)
        self._done_at = torch.tensor(done_at, dtype=torch.int32)
        self._trunc_at = torch.tensor(trunc_at, dtype=torch.int32)
        self._step = torch.zeros((self.num_envs,), dtype=torch.int32)
        self._obs = torch.zeros((self.num_envs, 1), dtype=torch.float32)

    def reset(self, seed: int | None = None):  # type: ignore[no-untyped-def]
        self._step.zero_()
        return self._obs

    def step(self, actions):  # type: ignore[no-untyped-def]
        self._step.add_(1)
        done = self._step == self._done_at
        trunc = (~done) & (self._step == self._trunc_at)
        reward = torch.ones((self.num_envs,), dtype=torch.float32)
        reset_mask = done | trunc
        self._step = torch.where(reset_mask, torch.zeros_like(self._step), self._step)
        return self._obs, reward, done, trunc, {}


class _ToyModel(torch.nn.Module):
    def forward(self, obs):  # type: ignore[no-untyped-def]
        batch = obs.shape[0]
        logits = torch.zeros((batch, 2), dtype=torch.float32)
        values = torch.zeros((batch,), dtype=torch.float32)
        return logits, values


def test_eval_collects_episodes_and_metrics() -> None:
    env = _ToyEvalEnv(done_at=[2, 999], trunc_at=[999, 3])
    model = _ToyModel()
    summary = run_greedy_eval(env, model, episodes=5)
    assert summary.episodes == 5
    assert summary.successes == 3
    assert summary.success_rate == pytest.approx(0.6)
    assert summary.median_steps_to_goal == 2
    assert summary.mean_return == pytest.approx(2.4)


def test_eval_median_none_when_no_success() -> None:
    env = _ToyEvalEnv(done_at=[999], trunc_at=[2])
    model = _ToyModel()
    summary = run_greedy_eval(env, model, episodes=2)
    assert summary.successes == 0
    assert summary.median_steps_to_goal is None


def test_eval_trajectory_requires_single_env(tmp_path):  # type: ignore[no-untyped-def]
    env = _ToyEvalEnv(done_at=[1, 1], trunc_at=[999, 999])
    model = _ToyModel()
    with pytest.raises(ValueError):
        run_greedy_eval(env, model, episodes=1, trajectory_path=tmp_path / "t.jsonl")


def test_eval_trajectory_dump(tmp_path):  # type: ignore[no-untyped-def]
    env = _ToyEvalEnv(done_at=[2], trunc_at=[999])
    model = _ToyModel()
    path = tmp_path / "traj.jsonl"
    run_greedy_eval(env, model, episodes=1, trajectory_path=path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[-1])
    assert record["t"] == 2
    assert record["done"] is True
