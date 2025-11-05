"""
main.py
=======

This script trains a PyTorch-based deep Q‑network (DQN) agent
to perform anti‑jamming communications.  It uses the
convolutional DQN defined in ``agent.py`` together with an
experience replay buffer and a target network.
The environment is defined in ``env.py`` and simulates
jamming on a wireless spectrum.

Three jamming patterns—sweep, comb and random—are trained
separately.  After training, the script produces two kinds of
plots per pattern: the normalised throughput over time
(smoothed rewards) and the evolution of action selection
probabilities.  Running this script in a Python environment
with PyTorch and Matplotlib installed will automatically
generate PNG files in the working directory.

The key algorithmic elements mirror those in the cited paper:
states correspond to raw spectrum waterfalls, the
reward encodes successful transmission and switching cost,
and the DQN processes the raw data directly with
convolutional layers.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from agent import DQN, ReplayMemory, update_target_network
from env import JammingEnv


def train_agent(
    env: JammingEnv,
    num_steps: int = 3000,
    batch_size: int = 32,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.1,
    epsilon_decay: int = 2000,
    memory_capacity: int = 5000,
    target_update_interval: int = 500,
) -> tuple[list[float], list[np.ndarray]]:
    """
    Train a DQN agent for a given environment.

    Parameters
    ----------
    env : JammingEnv
        The anti‑jamming environment.
    num_steps : int
        Number of training steps.
    batch_size : int
        Batch size for optimisation.
    gamma : float
        Discount factor.
    lr : float
        Learning rate for optimiser.
    epsilon_start, epsilon_final, epsilon_decay : float or int
        Parameters for the linear epsilon‑greedy schedule.
    memory_capacity : int
        Capacity of the experience replay buffer.
    target_update_interval : int
        Number of steps between updates of the target network.

    Returns
    -------
    rewards : list[float]
        Reward obtained at each training step.
    action_prob_history : list[np.ndarray]
        List of action probability vectors (softmax over Q values) recorded
        periodically during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialise networks
    state_shape = (env.M, env.N)
    policy_net = DQN(state_shape, env.K).to(device)
    target_net = DQN(state_shape, env.K).to(device)
    update_target_network(policy_net, target_net)
    target_net.eval()
    optimiser = optim.Adam(policy_net.parameters(), lr=lr)
    # Replay memory
    memory = ReplayMemory(memory_capacity)
    # Epsilon schedule
    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_final) / max(1, epsilon_decay)
    # Logging
    rewards: list[float] = []
    action_prob_history: list[np.ndarray] = []
    # Reset environment
    state = env.reset()
    # Main training loop
    for step_idx in range(num_steps):
        # Prepare state tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        # ε‑greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(env.K)
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = int(torch.argmax(q_values).item())
        # Execute action
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        # Store transition in memory (keep next_state and state as np arrays)
        memory.push(state, action, reward, next_state, done)
        # Update state
        state = next_state
        # Decay epsilon
        if epsilon > epsilon_final:
            epsilon -= epsilon_decay_rate
            if epsilon < epsilon_final:
                epsilon = epsilon_final
        # Optimisation step
        if len(memory) >= batch_size:
            transitions = memory.sample(batch_size)
            # Convert batch to tensors
            batch_states = torch.tensor(
                [t.state for t in transitions], dtype=torch.float32, device=device
            ).unsqueeze(1)
            batch_actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device).unsqueeze(1)
            batch_rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device).unsqueeze(1)
            batch_next_states = torch.tensor(
                [t.next_state for t in transitions], dtype=torch.float32, device=device
            ).unsqueeze(1)
            batch_dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=device).unsqueeze(1)
            # Current Q values
            current_q = policy_net(batch_states).gather(1, batch_actions)
            # Target Q values
            with torch.no_grad():
                max_next_q = target_net(batch_next_states).max(1)[0].unsqueeze(1)
                target_q = batch_rewards + gamma * max_next_q * (1.0 - batch_dones)
            # Compute loss and optimise
            loss = nn.functional.mse_loss(current_q, target_q)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        # Update target network
        if (step_idx + 1) % target_update_interval == 0:
            update_target_network(policy_net, target_net)
        # Record action probabilities periodically
        if (step_idx + 1) % max(1, num_steps // 50) == 0:
            with torch.no_grad():
                q = policy_net(state_tensor).cpu().numpy().flatten()
                probs = np.exp(q - np.max(q))
                probs /= probs.sum() if probs.sum() > 0 else 1.0
                action_prob_history.append(probs.copy())
    return rewards, action_prob_history


def smooth_series(values: list[float], window: int = 50) -> np.ndarray:
    """Compute a running mean of a list for smoothing reward curves."""
    if len(values) < window:
        return np.array(values, dtype=np.float32)
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def main() -> None:
    # Fix random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    patterns = ["sweep", "comb", "random"]
    num_steps = 3000
    reward_histories: dict[str, list[float]] = {}
    prob_histories: dict[str, list[np.ndarray]] = {}
    for pattern in patterns:
        print(f"Training under {pattern} jamming...")
        env = JammingEnv(n_freq_bins=20, history_length=20, n_channels=5, jam_pattern=pattern)
        rewards, probs = train_agent(
            env,
            num_steps=num_steps,
            batch_size=32,
            gamma=0.99,
            lr=1e-3,
            epsilon_start=1.0,
            epsilon_final=0.1,
            epsilon_decay=int(0.7 * num_steps),
            memory_capacity=5000,
            target_update_interval=500,
        )
        reward_histories[pattern] = rewards
        prob_histories[pattern] = probs
    # Plot throughput curves
    plt.figure(figsize=(8, 5))
    for pattern in patterns:
        smoothed = smooth_series(reward_histories[pattern], window=50)
        x = np.arange(len(smoothed))
        plt.plot(x, smoothed, label=pattern)
    plt.xlabel("Step")
    plt.ylabel("Normalised throughput")
    plt.title("Learning curves for different jamming patterns (PyTorch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("throughput_curves.png")
    plt.close()
    # Plot action probability evolution for each pattern
    for pattern in patterns:
        probs = np.array(prob_histories[pattern])
        if probs.size == 0:
            continue
        plt.figure(figsize=(8, 5))
        steps = np.linspace(0, num_steps, probs.shape[0])
        for action_idx in range(probs.shape[1]):
            plt.plot(steps, probs[:, action_idx], label=f"Action {action_idx}")
        plt.xlabel("Step")
        plt.ylabel("Selection probability")
        plt.title(f"Action probabilities over time ({pattern}) (PyTorch)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"action_probs_{pattern}.png")
        plt.close()
    print("Training complete. Figures saved in the current directory.")


if __name__ == "__main__":
    main()