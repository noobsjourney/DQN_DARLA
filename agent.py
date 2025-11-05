"""
agent.py
========

This module defines the agent components for the anti‑jamming
deep reinforcement learning demo.  It uses PyTorch to build a
deep Q‑network (DQN) with a separate target network and an
experience replay buffer, following the principles described in
the paper "Anti‑jamming Communications Using Spectrum
Waterfall: A Deep Reinforcement Learning Approach".  In
particular, the network processes raw spectrum waterfall data as
input and outputs Q values for each discrete action.

Classes
-------
ReplayMemory
    A fixed‑size circular buffer for storing state–action–reward
    transitions.  Experience replay helps break correlations in
    sequential data and stabilises DQN learning.

DQN
    A convolutional neural network that approximates the
    action‑value function.  It consists of two convolutional
    layers followed by two fully connected layers, similar to
    the architecture proposed in the paper.

update_target_network
    Utility function to copy the weights from the policy
    network to the target network.

Note
----
The environment uses a time–frequency matrix ``S_t`` as the
state.  The DQN defined here expects the state
to be provided as a PyTorch tensor of shape ``(batch_size, 1,
M, N)`` where ``M`` is the history length and ``N`` is the
number of frequency bins.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Transition:
    """Simple container for a single transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayMemory:
    """
    Experience replay buffer.

    Transitions are stored as Numpy arrays and converted to
    tensors in the training loop.  This class supports appending
    and random sampling.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: Deque[Transition] = Deque(maxlen=capacity)

    def push(self, *args: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        """Store a transition in the buffer."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Randomly sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    """
    Deep Q‑Network using convolutional layers.

    The network maps a 2‑D spectrum matrix to a vector of Q
    values.  It consists of two convolutional layers, each
    followed by ReLU activation, and two fully connected layers.
    This mirrors the architecture described in the paper for
    processing raw spectrum information.
    """

    def __init__(self, input_shape: Tuple[int, int], n_actions: int) -> None:
        super().__init__()
        _, height, width = (1,) + input_shape  # expect input as (batch, 1, M, N)
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # Determine flatten size
        def conv_out_size(size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
            return (size - kernel_size + 2 * padding) // stride + 1
        conv_h = conv_out_size(conv_out_size(height, 5, 2), 3, 2)
        conv_w = conv_out_size(conv_out_size(width, 5, 2), 3, 2)
        self.flatten_size = conv_h * conv_w * 64
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q values for each action."""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def update_target_network(source_net: nn.Module, target_net: nn.Module) -> None:
    """Copy parameters from ``source_net`` to ``target_net``."""
    target_net.load_state_dict(source_net.state_dict())