from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from .controllers import Controller
from .cartpole_env import CartPoleEnv


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, hidden: int = 32, action_bound: float = 10.0):
        super().__init__()
        self.action_bound = action_bound
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.action_bound


class REINFORCEController(Controller):
    def __init__(
        self,
        action_bound: float = 10.0,
        hidden: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        sigma_start: float = 1.0,
        sigma_end: float = 0.3,
        sigma_anneal_episodes: int = 400,
    ):
        self.action_bound = action_bound
        self.gamma = gamma
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.sigma_anneal_episodes = sigma_anneal_episodes
        self._episode_count = 0

        self.policy = PolicyNetwork(state_dim=4, hidden=hidden, action_bound=action_bound)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

    @property
    def sigma(self) -> float:
        frac = min(self._episode_count / max(self.sigma_anneal_episodes, 1), 1.0)
        return self.sigma_start + (self.sigma_end - self.sigma_start) * frac

    def __call__(self, t: float, state: NDArray) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32)
        mu = self.policy(state_t).squeeze()
        dist = torch.distributions.Normal(mu, self.sigma)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return float(action.clamp(-self.action_bound, self.action_bound).item())

    def reset(self):
        self.log_probs.clear()
        self.rewards.clear()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self) -> float:
        if len(self.rewards) == 0:
            return 0.0

        returns: list[float] = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = torch.zeros(1)
        for log_prob, G_t in zip(self.log_probs, returns_t):
            loss -= log_prob * G_t

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self._episode_count += 1
        return float(loss.item())


def train_reinforce(
    env: CartPoleEnv,
    controller: REINFORCEController,
    n_episodes: int = 500,
    T: float = 5.0,
    print_every: int = 50,
) -> list[float]:
    reward_history: list[float] = []

    for ep in range(n_episodes):
        controller.reset()
        state = env.reset()
        n_steps = int(T / env.dt)
        total_reward = 0.0

        for _ in range(n_steps):
            u = controller(env.t, state)
            state, reward, done, _ = env.step(u)
            controller.store_reward(reward)
            total_reward += reward
            if done:
                break

        loss = controller.update()
        reward_history.append(total_reward)

        if (ep + 1) % print_every == 0:
            avg = np.mean(reward_history[-print_every:])
            print(
                f"Episode {ep + 1:4d} | "
                f"reward {total_reward:8.1f} | "
                f"avg({print_every}) {avg:8.1f} | "
                f"loss {loss:8.3f} | "
                f"sigma {controller.sigma:.3f}"
            )

    return reward_history


def plot_training(reward_history: list[float], window: int = 50):
    fig, ax = plt.subplots(figsize=(10, 4))
    episodes = np.arange(1, len(reward_history) + 1)
    ax.plot(episodes, reward_history, alpha=0.3, color="steelblue", label="Episode reward")

    if len(reward_history) >= window:
        running_avg = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1:],
            running_avg,
            color="firebrick",
            linewidth=2,
            label=f"Running avg ({window})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("REINFORCE Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
