"""
A simple Deep Deterministic Policy Gradient (DDPG) agent adapted to the anti‑jamming
environment defined in ``env.py``.  DDPG is an actor–critic algorithm for
continuous action spaces.  In the anti‑jamming setting, although the
environment's action is discrete (selecting one of ``n_channels`` frequency
bands), we map the continuous actor output onto the discrete action
space by rounding to the nearest integer.  This allows the agent to
learn a continuous control policy while interfacing with the existing
discrete environment.

This example demonstrates how to build an actor–critic architecture
with target networks and a replay buffer in PyTorch.  It uses
convolutional layers to process the spectrum waterfall state and
fully–connected layers to produce a one–dimensional action in the
range ``[0, n_channels - 1]``.  The critic estimates the Q‑value for
a given state–action pair.  Target networks are softly updated at
each step using the parameter ``tau``.

Usage:
    from env import JammingEnv
    from ddpg_agent import DDPGAgent

    env = JammingEnv(n_freq_bins=20, history_length=20, n_channels=5, jam_pattern="sweep")
    agent = DDPGAgent(state_shape=(env.M, env.N), n_actions=env.K)
    rewards = agent.train(env, num_steps=5000)

This will train the agent and return a list of rewards collected at
each step.  Hyperparameters such as learning rates, discount factor,
noise scale and soft update coefficient can be adjusted via the
``DDPGAgent`` constructor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List


class Actor(nn.Module):
    """Actor network for continuous action output.

    The actor takes a spectrum waterfall state and outputs a scalar
    continuous action.  The output is squashed into the range ``[0,
    max_action]`` using a ``tanh`` activation scaled appropriately.
    """

    def __init__(self, state_shape: Tuple[int, int], max_action: float) -> None:
        super().__init__()
        M, N = state_shape
        self.max_action = max_action
        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Determine flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, M, N)
            feat_dim = self.conv(dummy).shape[1]
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, M, N)
        features = self.conv(x)
        action_raw = self.fc(features)
        # Squash to [-1, 1] then rescale to [0, max_action]
        action = torch.tanh(action_raw) * (self.max_action / 2.0) + (
            self.max_action / 2.0
        )
        return action  # shape: (batch, 1)


class Critic(nn.Module):
    """Critic network estimating Q(state, action).

    The critic shares the convolutional feature extractor with the actor
    to process the 2D state.  It then concatenates the flattened
    features with the action and predicts a scalar Q‑value.
    """

    def __init__(self, state_shape: Tuple[int, int]) -> None:
        super().__init__()
        M, N = state_shape
        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, M, N)
            feat_dim = self.conv(dummy).shape[1]
        # Critic fully connected layers
        # Input dimension is feature dimension + 1 (for continuous action)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, M, N), action shape: (batch, 1)
        features = self.conv(x)
        # Concatenate action to feature vector
        x_and_a = torch.cat([features, action], dim=1)
        q_value = self.fc(x_and_a)
        return q_value.squeeze(-1)


class ReplayBuffer:
    """Simple replay buffer for off‑policy algorithms."""

    def __init__(self, max_size: int = 100000) -> None:
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def add(self, transition: Tuple[np.ndarray, float, float, np.ndarray, float]) -> None:
        """Add a transition to the buffer.

        Each transition is a tuple (state, action, reward, next_state, done).
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)  # expand buffer
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> List:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.float32).reshape(-1, 1),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent.

    This agent maintains actor and critic networks along with their
    corresponding target networks.  It stores experience in a replay
    buffer and updates the networks by sampling mini‑batches.  The
    continuous action output from the actor is rounded to the nearest
    integer to interface with the discrete anti‑jamming environment.
    """

    def __init__(
        self,
        state_shape: Tuple[int, int],
        n_actions: int,
        gamma: float = 0.99,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        tau: float = 1e-3,
        buffer_size: int = 100000,
        batch_size: int = 64,
        noise_std: float = 0.2,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        # Continuous action range: [0, max_action]
        self.max_action = float(n_actions - 1)
        # Networks
        self.actor = Actor(state_shape, self.max_action).to(self.device)
        self.target_actor = Actor(state_shape, self.max_action).to(self.device)
        self.critic = Critic(state_shape).to(self.device)
        self.target_critic = Critic(state_shape).to(self.device)
        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> float:
        """Select a continuous action and optionally add exploration noise."""
        self.actor.eval()
        state_tensor = (
            torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )  # shape (1,1,M,N)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0, 0]
        self.actor.train()
        if add_noise:
            action += np.random.normal(0.0, self.noise_std)
        # Clip to valid range [0, max_action]
        action = float(np.clip(action, 0.0, self.max_action))
        return action

    def _update_networks(self) -> None:
        """Update actor and critic networks using a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough samples
        # Sample a mini‑batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        # Convert to tensors
        states_tensor = (
            torch.from_numpy(states).float().unsqueeze(1).to(self.device)
        )  # (B,1,M,N)
        actions_tensor = torch.from_numpy(actions).float().to(self.device)  # (B,1)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)  # (B,)
        next_states_tensor = (
            torch.from_numpy(next_states).float().unsqueeze(1).to(self.device)
        )
        dones_tensor = torch.from_numpy(dones).float().to(self.device)  # (B,)
        # --- Critic update ---
        # Compute target actions with target actor
        with torch.no_grad():
            next_actions = self.target_actor(next_states_tensor)
            # Target Q = r + gamma * (1 - done) * Q'(s', a')
            target_q = self.target_critic(next_states_tensor, next_actions).detach()
            y = rewards_tensor + self.gamma * (1.0 - dones_tensor) * target_q
        # Current Q estimates
        current_q = self.critic(states_tensor, actions_tensor)
        critic_loss = nn.functional.mse_loss(current_q, y)
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # --- Actor update ---
        # Actor aims to maximize Q, so minimize -Q
        predicted_actions = self.actor(states_tensor)
        actor_loss = -self.critic(states_tensor, predicted_actions).mean()
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # --- Soft update target networks ---
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, env, num_steps: int = 5000, warmup_steps: int = 1000) -> List[float]:
        """Interact with the environment and train the agent.

        Parameters
        ----------
        env : gym‑like environment with discrete action space
            The environment must implement ``reset()`` and ``step(action)``.
        num_steps : int
            Total number of time steps for training
        warmup_steps : int
            Number of initial steps to collect data without updating networks

        Returns
        -------
        List[float]
            A list of rewards collected at each step.
        """
        state = env.reset()
        rewards_history: List[float] = []
        for t in range(num_steps):
            # Select continuous action
            cont_action = self.select_action(state, add_noise=True)
            # Round to nearest integer to interface with environment
            discrete_action = int(np.rint(cont_action))
            # Step in environment
            next_state, reward, done, _ = env.step(discrete_action)
            # Store transition in replay buffer
            self.replay_buffer.add((state, cont_action, reward, next_state, float(done)))
            rewards_history.append(reward)
            # Update networks after warmup
            if t >= warmup_steps:
                self._update_networks()
            # Move to next state
            state = next_state
            if done:
                state = env.reset()
        return rewards_history


if __name__ == "__main__":
    # Quick test of the DDPG agent on the anti‑jamming environment
    from env import JammingEnv

    env = JammingEnv(
        n_freq_bins=20, history_length=20, n_channels=5, jam_pattern="sweep"
    )
    agent = DDPGAgent(state_shape=(env.M, env.N), n_actions=env.K)
    rewards = agent.train(env, num_steps=2000)
    print("Average reward (last 50 steps):", np.mean(rewards[-50:]))