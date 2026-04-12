"""
plot_training.py — Generate training curve plots from GRPO metrics.

Reads metrics.jsonl (one JSON object per line) and produces:
  1. Mean reward vs iteration (with max reward band)
  2. Per-task reward heatmap
  3. Loss + KL divergence (if gradient training was used)

Usage:
  python plot_training.py --metrics grpo_output/metrics.jsonl --output training_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_metrics(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_rewards(records: list[dict], ax):
    iters = [r["iteration"] for r in records]
    means = [r["mean_reward"] for r in records]
    maxes = [r["max_reward"] for r in records]

    ax.fill_between(iters, means, maxes, alpha=0.2, color="steelblue", label="max reward")
    ax.plot(iters, means, "o-", color="steelblue", markersize=4, linewidth=1.5, label="mean reward")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Training: Reward vs Iteration")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)


def plot_per_task(records: list[dict], ax):
    all_tasks = sorted({t for r in records for t in r.get("task_rewards", {})})
    if not all_tasks:
        ax.text(0.5, 0.5, "No per-task data", ha="center", va="center")
        return

    data = []
    for r in records:
        tr = r.get("task_rewards", {})
        data.append([tr.get(t, 0.0) for t in all_tasks])

    iters = [r["iteration"] for r in records]
    for i, task in enumerate(all_tasks):
        vals = [d[i] for d in data]
        short_name = task.replace("T", "").replace("_", " ").title()
        ax.plot(iters, vals, "o-", markersize=3, linewidth=1, label=short_name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Per-Task Reward")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)


def plot_loss_kl(records: list[dict], ax):
    has_loss = any("loss" in r for r in records)
    if not has_loss:
        ax.text(0.5, 0.5, "API mode (no gradient updates)\nRollout-only metrics shown",
                ha="center", va="center", fontsize=10, style="italic")
        ax.set_title("Loss & KL (gradient training only)")
        return

    iters = [r["iteration"] for r in records if "loss" in r]
    losses = [r["loss"] for r in records if "loss" in r]
    kls = [r["kl"] for r in records if "kl" in r]

    ax.plot(iters, losses, "s-", color="tomato", markersize=3, label="loss")
    ax2 = ax.twinx()
    ax2.plot(iters, kls, "^-", color="seagreen", markersize=3, label="KL")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss", color="tomato")
    ax2.set_ylabel("KL Divergence", color="seagreen")
    ax.set_title("Loss & KL Divergence")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser(description="Plot GRPO training curves")
    p.add_argument("--metrics", default="grpo_output/metrics.jsonl",
                   help="Path to metrics.jsonl")
    p.add_argument("--output", default="training_curves.png",
                   help="Output image path (default: training_curves.png)")
    args = p.parse_args()

    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Generating text summary instead...\n")
        records = load_metrics(args.metrics)
        for r in records:
            print(f"Iter {r['iteration']:3d}: mean_reward={r['mean_reward']:.4f} "
                  f"max_reward={r['max_reward']:.4f} valid={r['n_valid']}/{r['n_total']}")
        return

    records = load_metrics(args.metrics)
    if not records:
        print(f"No records found in {args.metrics}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_rewards(records, axes[0])
    plot_per_task(records, axes[1])
    plot_loss_kl(records, axes[2])

    fig.suptitle("Comtrade Agent — GRPO Training Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {args.output}")

    # Also print summary
    last = records[-1]
    first = records[0]
    print(f"\nSummary ({len(records)} iterations):")
    print(f"  First iter: mean_reward={first['mean_reward']:.4f}")
    print(f"  Last iter:  mean_reward={last['mean_reward']:.4f}")
    print(f"  Best max:   {max(r['max_reward'] for r in records):.4f}")


if __name__ == "__main__":
    main()
