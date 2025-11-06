# utility/plot_utils.py
import numpy as np
import matplotlib.pyplot as plt

def smooth_series(values: list[float], window: int = 50) -> np.ndarray:
    """Compute a running mean of a list for smoothing reward curves."""
    if len(values) < window:
        return np.array(values, dtype=np.float32)
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def plot_throughput_curves(reward_histories, window=50,
                           filename="throughput_curves.png",
                           title="Learning curves"):
    """根据 reward_histories 绘制多条平滑后的吞吐曲线并保存图像。"""
    plt.figure(figsize=(8,5))
    for pattern, rewards in reward_histories.items():
        smoothed = smooth_series(rewards, window)
        x = np.arange(len(smoothed))
        plt.plot(x, smoothed, label=pattern)
    plt.xlabel("Step")
    plt.ylabel("Normalised throughput")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_action_probabilities(prob_histories, num_steps,
                              prefix="action_probs",
                              title_prefix="Action probabilities over time"):
    """根据概率历史绘制每种干扰模式下的动作选择概率曲线。"""
    for pattern, probs in prob_histories.items():
        if not probs:
            continue
        probs_arr = np.array(probs)
        plt.figure(figsize=(8,5))
        steps = np.linspace(0, num_steps, probs_arr.shape[0])
        for action_idx in range(probs_arr.shape[1]):
            plt.plot(steps, probs_arr[:, action_idx], label=f"Action {action_idx}")
        plt.xlabel("Step")
        plt.ylabel("Selection probability")
        plt.title(f"{title_prefix} ({pattern})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}_{pattern}.png")
        plt.close()
