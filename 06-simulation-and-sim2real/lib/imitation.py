"""Minimal Behavioral Cloning and DAgger for lecture demos.

Uses simple fully-connected networks (numpy only, no deep-learning
framework required) to keep the demo self-contained.  For real
applications use torch / jax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Tiny MLP (numpy) — sufficient for low-dim demos
# ---------------------------------------------------------------------------

@dataclass
class Normalizer:
    """Online mean/std normalizer for inputs."""
    mean: NDArray
    std: NDArray

    @classmethod
    def fit(cls, x: NDArray) -> "Normalizer":
        return cls(mean=x.mean(axis=0), std=x.std(axis=0).clip(min=1e-8))

    def transform(self, x: NDArray) -> NDArray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: NDArray) -> NDArray:
        return x * self.std + self.mean


@dataclass
class MLP:
    """Two-layer ReLU network with MSE training via SGD + gradient clipping."""

    weights: list[NDArray]
    biases: list[NDArray]

    @classmethod
    def create(cls, dims: list[int], rng: np.random.Generator | None = None) -> "MLP":
        rng = rng or np.random.default_rng()
        weights, biases = [], []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2 / d_in) * 0.5
            weights.append(rng.normal(0, scale, (d_in, d_out)))
            biases.append(np.zeros(d_out))
        return cls(weights, biases)

    def predict(self, x: NDArray) -> NDArray:
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)  # ReLU
        return x

    def _forward_cache(self, x: NDArray) -> list[NDArray]:
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)
            activations.append(x)
        return activations

    def train_step(self, x: NDArray, y: NDArray, lr: float = 1e-3,
                   max_grad_norm: float = 5.0) -> float:
        """One gradient-descent step (full-batch) with gradient clipping."""
        acts = self._forward_cache(x)
        n = x.shape[0]
        diff = acts[-1] - y
        loss = float(np.mean(diff ** 2))

        grad = 2 * diff / n
        for i in reversed(range(len(self.weights))):
            dw = acts[i].T @ grad
            db = grad.sum(axis=0)

            # Gradient clipping
            gn = np.sqrt(np.sum(dw**2) + np.sum(db**2))
            if gn > max_grad_norm:
                scale = max_grad_norm / (gn + 1e-8)
                dw *= scale
                db *= scale

            if i > 0:
                grad = grad @ self.weights[i].T
                grad = grad * (acts[i] > 0)  # ReLU derivative

            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
        return loss


# ---------------------------------------------------------------------------
# Normalized policy wrapper
# ---------------------------------------------------------------------------

@dataclass
class NormalizedPolicy:
    """MLP wrapped with input/output normalization."""
    net: MLP
    s_norm: Normalizer
    a_norm: Normalizer

    def predict(self, s: NDArray) -> NDArray:
        s_n = self.s_norm.transform(s)
        a_n = self.net.predict(s_n)
        return self.a_norm.inverse_transform(a_n)


# ---------------------------------------------------------------------------
# Behavioral Cloning
# ---------------------------------------------------------------------------

def behavioral_cloning(
    states: NDArray,
    actions: NDArray,
    hidden: int = 32,
    epochs: int = 200,
    lr: float = 3e-3,
    rng: np.random.Generator | None = None,
) -> tuple[NormalizedPolicy, list[float]]:
    """Train a policy via supervised learning on (state, action) pairs.

    Returns (policy, loss_history).
    """
    rng = rng or np.random.default_rng(0)
    s_norm = Normalizer.fit(states)
    a_norm = Normalizer.fit(actions)
    s_n = s_norm.transform(states)
    a_n = a_norm.transform(actions)

    s_dim, a_dim = states.shape[1], actions.shape[1]
    net = MLP.create([s_dim, hidden, a_dim], rng)
    losses = []
    for _ in range(epochs):
        loss = net.train_step(s_n, a_n, lr=lr)
        losses.append(loss)
    policy = NormalizedPolicy(net=net, s_norm=s_norm, a_norm=a_norm)
    return policy, losses


# ---------------------------------------------------------------------------
# DAgger
# ---------------------------------------------------------------------------

def dagger(
    env_step_fn: Callable[[NDArray, NDArray], NDArray],
    expert_fn: Callable[[NDArray], NDArray],
    initial_states: NDArray,
    initial_actions: NDArray,
    n_rounds: int = 10,
    rollout_len: int = 50,
    n_rollouts: int = 5,
    hidden: int = 32,
    train_epochs: int = 100,
    lr: float = 3e-3,
    rng: np.random.Generator | None = None,
    reset_sampler: Callable[[np.random.Generator], NDArray] | None = None,
) -> tuple[NormalizedPolicy, list[float], list[NDArray]]:
    """Dataset Aggregation (DAgger).

    Parameters
    ----------
    env_step_fn : (state, action) -> next_state
        Deterministic transition.
    expert_fn : state -> action
        Oracle policy.
    initial_states, initial_actions
        Seed dataset from expert demonstrations.
    n_rounds : int
        Number of DAgger iterations.

    Returns
    -------
    (policy, loss_history, rollout_states_per_round)
    """
    rng = rng or np.random.default_rng(0)
    all_states = initial_states.copy()
    all_actions = initial_actions.copy()
    s_dim, a_dim = all_states.shape[1], all_actions.shape[1]
    losses: list[float] = []
    rollout_states_history: list[NDArray] = []

    for _ in range(n_rounds):
        # Re-fit normalization on growing dataset
        s_norm = Normalizer.fit(all_states)
        a_norm = Normalizer.fit(all_actions)
        s_n = s_norm.transform(all_states)
        a_n = a_norm.transform(all_actions)

        # Fresh network each round (small enough to retrain)
        net = MLP.create([s_dim, hidden, a_dim], rng)
        for _ in range(train_epochs):
            loss = net.train_step(s_n, a_n, lr=lr)
        losses.append(loss)

        policy = NormalizedPolicy(net=net, s_norm=s_norm, a_norm=a_norm)

        # Rollout current policy, collect visited states
        new_states = []
        for _ in range(n_rollouts):
            if reset_sampler is None:
                idx = rng.integers(0, len(all_states))
                s = all_states[idx].copy()
            else:
                s = np.array(reset_sampler(rng), dtype=float)
            for _ in range(rollout_len):
                a = policy.predict(s[np.newaxis])[0]
                s = env_step_fn(s, a)
                new_states.append(s.copy())

        new_states_arr = np.array(new_states)
        rollout_states_history.append(new_states_arr)

        # Query expert for labels on visited states
        expert_actions = np.array([expert_fn(s) for s in new_states_arr])

        # Aggregate
        all_states = np.concatenate([all_states, new_states_arr])
        all_actions = np.concatenate([all_actions, expert_actions])

    return policy, losses, rollout_states_history


# ---------------------------------------------------------------------------
# Simple 2D environment for demos
# ---------------------------------------------------------------------------

def make_2d_nav_env(noise_std: float = 0.02, seed: int | None = None):
    """A 2D navigation env with nonlinear dynamics.

    The expert navigates toward y=0 while moving forward (x increases).
    Nonlinear drag term makes the dynamics harder to generalise from
    a narrow expert distribution, clearly demonstrating covariate shift.

    Returns (step_fn, expert_fn, generate_expert_data).
    """
    dt = 0.1
    noise_rng = np.random.default_rng(seed)

    def step_fn(state: NDArray, action: NDArray) -> NDArray:
        x, y = state
        ax, ay = action
        # Nonlinear drag: stronger away from y=0
        drag = 0.3 * np.tanh(y)
        nx = x + dt * ax
        ny = y + dt * (ay + drag) + noise_rng.normal(0, noise_std)
        return np.array([nx, ny])

    def expert_fn(state: NDArray) -> NDArray:
        _, y = state
        ax = 1.0
        # Compensate for drag exactly
        drag = 0.3 * np.tanh(y)
        ay = -3.0 * y - drag
        return np.array([ax, ay])

    def generate_expert_data(n: int = 300, rng=None) -> tuple[NDArray, NDArray]:
        rng = rng or np.random.default_rng(0)
        states, actions = [], []
        for _ in range(3):
            s = np.array([0.0, rng.uniform(-0.3, 0.3)])
            for _ in range(n // 3):
                a = expert_fn(s)
                states.append(s.copy())
                actions.append(a.copy())
                s = step_fn(s, a)
        return np.array(states), np.array(actions)

    return step_fn, expert_fn, generate_expert_data
