"""
env.py
=======

Environment definition for the anti‑jamming demonstration.

The environment simulates a wireless spectrum with N frequency bins
and maintains a history of the last M time steps of power
measurements (spectrum waterfall).  At each step a
jammer injects high power into certain bins following one of
three patterns: sweep, comb, or random.  The
agent selects one of K contiguous channels to transmit on.  If
the chosen channel overlaps a jammed region, the transmission
fails (reward 0).  Otherwise the transmission succeeds and the
reward is 1 minus a small cost for switching channels.  This
simple model reflects the reward structure described in the
paper, where the reward is tied to successful transmission and
a cost is incurred when the action changes.
"""

from __future__ import annotations

import random
from typing import Dict, Tuple, List

import numpy as np


class JammingEnv:
    """
    Anti‑jamming environment with configurable spectrum size and jamming patterns.

    Parameters
    ----------
    n_freq_bins : int
        Number of discrete frequency bins.
    history_length : int
        Number of time steps maintained in the state matrix.
    n_channels : int
        Number of equal‑width channels the agent may choose from.
    jam_pattern : str
        Type of jammer ('sweep', 'comb', or 'random').  Sweep moves
        a single jammer across the band, comb places fixed jammers at
        evenly spaced locations, and random hops a single jammer
        randomly at each step.
    jam_bandwidth : int
        Width of each jammer in bins.
    channel_change_cost : float
        Reward penalty incurred when the agent switches channels.
    """

    def __init__(
        self,
        n_freq_bins: int = 20,
        history_length: int = 20,
        n_channels: int = 5,
        jam_pattern: str = "sweep",
        jam_bandwidth: int = 2,
        channel_change_cost: float = 0.1,
    ) -> None:
        assert jam_pattern in {"sweep", "comb", "random"}
        assert n_freq_bins % n_channels == 0, "n_freq_bins must be divisible by n_channels"
        self.N = n_freq_bins
        self.M = history_length
        self.K = n_channels
        self.pattern = jam_pattern
        self.bandwidth = jam_bandwidth
        self.change_cost = channel_change_cost
        self.channel_width = self.N // self.K
        # state: M x N matrix representing past spectra
        self.state_matrix = np.zeros((self.M, self.N), dtype=np.float32)
        self.prev_action = 0
        self.step_count = 0
        # jamming positions
        self.jam_positions: List[int] = []
        if self.pattern == "comb":
            # three evenly spaced jammers
            self.jam_positions = [self.N // 4, self.N // 2, 3 * self.N // 4]

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state matrix."""
        self.state_matrix.fill(0.0)
        self.prev_action = 0
        self.step_count = 0
        if self.pattern == "sweep":
            self.jam_positions = [0]
        elif self.pattern == "random":
            self.jam_positions = [random.randrange(self.N)]
        return self.state_matrix.copy()

    def _update_jam_positions(self) -> None:
        """Update jammer positions based on the selected pattern."""
        if self.pattern == "sweep":
            new_pos = (self.jam_positions[0] + 1) % self.N
            self.jam_positions = [new_pos]
        elif self.pattern == "random":
            self.jam_positions = [random.randrange(self.N)]
        # comb pattern: positions remain fixed

    def _generate_spectrum_vector(self) -> np.ndarray:
        """Generate a power spectrum vector with noise and jamming spikes."""
        spectrum = np.random.normal(loc=0.0, scale=0.05, size=self.N).astype(np.float32)
        for centre in self.jam_positions:
            start = max(0, centre - self.bandwidth // 2)
            end = min(self.N, centre + self.bandwidth // 2 + 1)
            spectrum[start:end] += 1.0  # high power in jammed bins
        return spectrum

    def _is_channel_jammed(self, action: int) -> bool:
        """Determine whether the selected channel overlaps any jamming region."""
        start_bin = action * self.channel_width
        end_bin = start_bin + self.channel_width
        for centre in self.jam_positions:
            jam_start = max(0, centre - self.bandwidth // 2)
            jam_end = min(self.N, centre + self.bandwidth // 2 + 1)
            if not (jam_end <= start_bin or jam_start >= end_bin):
                return True
        return False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Returns
        -------
        next_state : np.ndarray
            Updated state matrix after the new spectrum observation.
        reward : float
            Reward received from taking the selected action.
        done : bool
            Always False; included for compatibility with Gym APIs.
        info : dict
            Additional information (empty in this environment).
        """
        assert 0 <= action < self.K
        # move jammer(s)
        self._update_jam_positions()
        # acquire new spectrum observation
        new_spectrum = self._generate_spectrum_vector()
        # update state matrix (drop oldest row, append new)
        self.state_matrix[:-1] = self.state_matrix[1:]
        self.state_matrix[-1] = new_spectrum
        # compute reward: success if channel not jammed
        jammed = self._is_channel_jammed(action)
        reward = 1.0 if not jammed else 0.0
        # penalty for switching channels
        if action != self.prev_action:
            reward -= self.change_cost
        reward = max(reward, 0.0)
        self.prev_action = action
        self.step_count += 1
        return self.state_matrix.copy(), reward, False, {}