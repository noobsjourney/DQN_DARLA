"""
A simple Advantage Actor-Critic (A2C) agent for the anti‑jamming
environment defined in ``env.py``.  This implementation uses
PyTorch to build a neural network with shared convolutional
feature extraction and separate heads for the policy (actor) and
value function (critic).  The agent performs one‑step updates
with the temporal difference error as the advantage estimator.

Usage:
    from env import JammingEnv
    from a2c_agent import A2CAgent

    env = JammingEnv(n_freq_bins=20, history_length=20, n_channels=5, jam_pattern="sweep")
    agent = A2CAgent(state_shape=(env.M, env.N), n_actions=env.K)
    rewards = agent.train(env, num_steps=5000)

This will train the agent and return the list of rewards obtained
at each step.  You can adjust hyperparameters such as learning
rate, discount factor and entropy coefficient via the class
constructor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List


class ActorCritic(nn.Module):
    """A convolutional actor‑critic network.

    The network processes the 2D spectrum waterfall state and
    outputs both a policy distribution over discrete actions and
    a state value estimate.  Convolutional layers are used to
    extract spatial features from the time‑frequency matrix.
    """

    def __init__(self, state_shape: Tuple[int, int], n_actions: int) -> None:
        super().__init__()
        M, N = state_shape
        # Convolutional feature extractor: 1 input channel -> 16 -> 32
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute the size of the flattened feature vector
        dummy = torch.zeros(1, 1, M, N)
        with torch.no_grad():
            feat_dim = self.conv(dummy).shape[1]
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, 1, M, N)
        features = self.conv(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value.squeeze(-1)


class A2CAgent:
    """Advantage Actor‑Critic agent with one‑step updates."""

    def __init__(
        self,
        state_shape: Tuple[int, int],
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        entropy_coef: float = 1e-3,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        # Create the ActorCritic network
        self.model = ActorCritic(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select an action according to the current policy.

        Returns the chosen action index along with the log probability and
        state value needed for the gradient update.
        """
        state_tensor = (
            torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )  # shape (1,1,M,N)
        logits, value = self.model(state_tensor)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob, value

    def train(self, env, num_steps: int = 5000) -> List[float]:
        """Train the agent in the provided environment.

        Parameters
        ----------
        env : gym‑like environment with reset() and step() methods
        num_steps : int
            Total number of training steps

        Returns
        -------
        List[float]
            A list of rewards collected at each step.
        """
        state = env.reset()
        rewards: List[float] = []
        for _ in range(num_steps):
            # Select action
            action, log_prob, value = self.select_action(state)
            # Step in environment
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            # Compute bootstrap target: r + gamma * V(next)
            next_state_tensor = (
                torch.from_numpy(next_state)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                _, next_value = self.model(next_state_tensor)
            advantage = reward + self.gamma * next_value - value
            # Policy loss (maximize advantage * log_prob) => minimize -advantage * log_prob
            policy_loss = -(advantage.detach() * log_prob)
            # Value loss
            value_loss = advantage.pow(2)
            # Entropy loss to encourage exploration
            logits, _ = self.model(
                torch.from_numpy(state)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            probs = torch.softmax(logits, dim=1)
            entropy = torch.distributions.Categorical(probs).entropy().mean()
            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Update state
            state = next_state
            if done:
                state = env.reset()
        return rewards


if __name__ == "__main__":
    # Quick test of the A2C agent on the anti‑jamming environment
    from env import JammingEnv

    env = JammingEnv(n_freq_bins=20, history_length=20, n_channels=5, jam_pattern="sweep")
    agent = A2CAgent(state_shape=(env.M, env.N), n_actions=env.K)
    rewards = agent.train(env, num_steps=200)
    print("Average reward:", np.mean(rewards[-50:]))