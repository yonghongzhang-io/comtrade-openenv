"""
plot_envelope.py — Render the GRPO operating-envelope figure.

Shows the three training configurations side-by-side:
  1.5B full-param  (under-capacity)
  3B  + LoRA       (learns then collapses)
  7B  + LoRA       (saturated)

Each panel plots mean_reward per iteration with failure-mode annotations.
Matches the dark palette of benchmark_results.png.

Output: grpo_envelope.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().strip().split("\n") if l.strip()]


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    # Palette (matches benchmark_results.png)
    bg = "#0f172a"
    surface = "#1e293b"
    text_color = "#e2e8f0"
    muted = "#94a3b8"
    color_15b = "#fbbf24"   # amber (noise-dominated / under-capacity)
    color_3b = "#4ade80"    # green (learns)
    color_3b_collapse = "#f87171"  # red (collapse)
    color_7b = "#22d3ee"    # cyan (saturated)
    color_baseline = "#818cf8"  # indigo

    # Load the three runs
    m_15b = load_metrics(root / "grpo_gradient_training.jsonl")
    m_3b  = load_metrics(root / "grpo_gradient_training_3b.jsonl")
    # 7B only has a summary (5 iters) - inline the data
    m_7b = [
        {"iteration": 1, "mean_reward": 0.987, "reward_std": 0.000},
        {"iteration": 2, "mean_reward": 0.975, "reward_std": 0.012},
        {"iteration": 3, "mean_reward": 0.955, "reward_std": 0.014},
        {"iteration": 4, "mean_reward": 0.968, "reward_std": 0.014},
        {"iteration": 5, "mean_reward": 0.950, "reward_std": 0.003},
    ]

    # Baseline reward (rule-based agent)
    baseline_reward = 0.968

    # Three-panel figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor=bg,
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.22})

    # ================== Panel 1: Qwen2.5-1.5B (under-capacity) ==================
    ax = axs[0]
    ax.set_facecolor(bg)
    if m_15b:
        iters = [m["iteration"] for m in m_15b]
        rewards = [m["mean_reward"] for m in m_15b]
        ax.plot(iters, rewards, color=color_15b, linewidth=2, alpha=0.9,
                label="mean_reward", marker="o", markersize=3)
        ax.axhline(baseline_reward, color=color_baseline, linestyle="--", linewidth=1, alpha=0.7)
        ax.text(49, baseline_reward + 0.015, "baseline 0.968",
                fontsize=8.5, color=color_baseline, ha="right", va="bottom")

    ax.set_title("🟡 Qwen2.5-1.5B (50 iter, full-param)",
                 fontsize=13, color=text_color, fontweight="bold", pad=12)
    ax.text(0.5, 0.98, "UNDER-CAPACITY",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=color_15b, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=surface,
                     edgecolor=color_15b, linewidth=1.2))
    ax.text(25, 0.10,
            "Noise-dominated oscillation.\nNo net upward trend.\nMax reward drops to 0.24\non T9/T10 — capacity ceiling.",
            fontsize=9, color=muted, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=surface,
                     edgecolor=muted, linewidth=0.5, alpha=0.85))

    # ================== Panel 2: Qwen2.5-3B + LoRA (learns then collapses) ==================
    ax = axs[1]
    ax.set_facecolor(bg)
    if m_3b:
        iters = [m["iteration"] for m in m_3b]
        rewards = [m["mean_reward"] for m in m_3b]

        # Split: iter 3-14 (learning green), iter 15-18 (collapse red)
        learn_iters = [m["iteration"] for m in m_3b if m["iteration"] <= 14]
        learn_rewards = [m["mean_reward"] for m in m_3b if m["iteration"] <= 14]
        collapse_iters = [m["iteration"] for m in m_3b if m["iteration"] >= 14]
        collapse_rewards = [m["mean_reward"] for m in m_3b if m["iteration"] >= 14]

        ax.plot(learn_iters, learn_rewards, color=color_3b, linewidth=2, alpha=0.9,
                marker="o", markersize=4, label="learning phase")
        ax.plot(collapse_iters, collapse_rewards, color=color_3b_collapse, linewidth=2.5,
                alpha=0.95, marker="x", markersize=8, label="collapse")
        ax.axhline(baseline_reward, color=color_baseline, linestyle="--", linewidth=1, alpha=0.7)
        ax.text(17.5, baseline_reward + 0.015, "baseline 0.968",
                fontsize=8.5, color=color_baseline, ha="right", va="bottom")

        # Annotate the collapse with an arrow
        ax.annotate(
            "iter 15-17\nzero valid rollouts\n(policy collapse)",
            xy=(17, 0.05), xytext=(12.5, 0.45),
            fontsize=8.5, color=color_3b_collapse, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=surface,
                     edgecolor=color_3b_collapse, linewidth=1),
            arrowprops=dict(arrowstyle="->", color=color_3b_collapse, lw=1.3,
                           connectionstyle="arc3,rad=-0.3"),
        )

    ax.set_title("🟢 Qwen2.5-3B + LoRA (15 + 3 skipped iters)",
                 fontsize=13, color=text_color, fontweight="bold", pad=12)
    ax.text(0.5, 0.98, "LEARNS → COLLAPSES",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=color_3b, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=surface,
                     edgecolor=color_3b, linewidth=1.2))
    ax.text(6, 0.78,
            "iters 3-14: learning\nkl grows 8e-6 → 5.6e-4\nreward_std 0.46 – 0.55",
            fontsize=8.5, color=color_3b, ha="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=surface,
                     edgecolor=color_3b, linewidth=0.8, alpha=0.95))

    # ================== Panel 3: Qwen2.5-7B + LoRA (saturated) ==================
    ax = axs[2]
    ax.set_facecolor(bg)
    iters = [m["iteration"] for m in m_7b]
    rewards = [m["mean_reward"] for m in m_7b]
    ax.plot(iters, rewards, color=color_7b, linewidth=2.5, alpha=0.95,
            marker="o", markersize=6)
    ax.axhline(baseline_reward, color=color_baseline, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(1.1, baseline_reward - 0.04, "baseline 0.968",
            fontsize=8.5, color=color_baseline, ha="left", va="top")

    ax.set_title("⚠️ Qwen2.5-7B + LoRA (5 iter)",
                 fontsize=13, color=text_color, fontweight="bold", pad=12)
    ax.text(0.5, 0.98, "SATURATED AT INIT",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=color_7b, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=surface,
                     edgecolor=color_7b, linewidth=1.2))
    ax.text(3, 0.38,
            "iter 1 mean = 0.987\n(already above baseline)\n\nreward_std ≈ 0\n→ GRPO advantage = 0\n→ no gradient signal",
            fontsize=9, color=muted, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=surface,
                     edgecolor=muted, linewidth=0.5, alpha=0.85))

    # Shared styling
    for ax in axs:
        ax.set_xlabel("iteration", color=text_color, fontsize=10)
        ax.set_ylabel("mean reward", color=text_color, fontsize=10)
        ax.set_ylim(-0.05, 1.08)
        ax.tick_params(axis="both", colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(muted)
            spine.set_alpha(0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=muted, alpha=0.15, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    # Global title & subtitle
    fig.suptitle("GRPO Operating Envelope on ComtradeBench — Three Failure Modes Empirically Mapped",
                 fontsize=15, color=text_color, fontweight="bold", y=0.99)
    fig.text(0.5, 0.02,
             "The useful GRPO training band exists (3B iters 3-14 prove it) but is NARROW and FRAGILE. "
             "Stable training on 3B requires adaptive KL penalty + trust-region clipping + early-stop on reward-variance collapse.",
             ha="center", fontsize=9.5, color=muted, style="italic")

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_path = root / "grpo_envelope.png"
    fig.savefig(out_path, dpi=140, facecolor=bg, bbox_inches="tight")
    print(f"Saved: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
