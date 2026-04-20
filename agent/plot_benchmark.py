"""
plot_benchmark.py — regenerate benchmark_results.png from the canonical
result JSON files (inference_results_baseline.json, llm_results_kimi.json,
llm_results_claude.json, llm_results_llama.json).

Produces a 4-bar-per-task chart (Baseline / Kimi-128k / Claude Sonnet 4.6 /
Llama-3.3-70B) with T9 and T10 highlighted as novel tasks and a visual
emphasis on the T9 cross-model gap.

Usage:
  python agent/plot_benchmark.py
  # overwrites benchmark_results.png in the current directory
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


TASK_DISPLAY = {
    "T1_single_page":           "T1\nSingle",
    "T2_multi_page":             "T2\nMulti",
    "T3_duplicates":             "T3\nDedup",
    "T4_rate_limit_429":         "T4\n429",
    "T5_server_error_500":       "T5\n500",
    "T6_page_drift":             "T6\nDrift",
    "T7_totals_trap":            "T7\nTotals",
    "T8_mixed_faults":           "T8\nMixed",
    "T9_adaptive_adversary":     "T9\nAdaptive*",
    "T10_constrained_budget":    "T10\nBudget*",
}
NOVEL_TASKS = {"T9_adaptive_adversary", "T10_constrained_budget"}


def load_scores(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    out = {}
    for r in data["results"]:
        out[r["task_id"]] = r["score"]
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    baseline = load_scores(root / "inference_results_baseline.json")
    kimi = load_scores(root / "llm_results_kimi.json")
    claude = load_scores(root / "llm_results_claude.json")
    llama = load_scores(root / "llm_results_llama.json")
    gpt5_path = root / "llm_results_gpt5.json"
    gpt5 = load_scores(gpt5_path) if gpt5_path.exists() else None
    qwen7b_path = root / "llm_results_qwen7b_zeroshot.json"
    qwen7b = load_scores(qwen7b_path) if qwen7b_path.exists() else None

    task_order = list(TASK_DISPLAY.keys())
    xlabels = [TASK_DISPLAY[t] for t in task_order]

    baseline_scores = [baseline[t] for t in task_order]
    kimi_scores = [kimi[t] for t in task_order]
    claude_scores = [claude[t] for t in task_order]
    llama_scores = [llama[t] for t in task_order]
    gpt5_scores = [gpt5[t] for t in task_order] if gpt5 else None
    qwen7b_scores = [qwen7b[t] for t in task_order] if qwen7b else None

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    kimi_avg = sum(kimi_scores) / len(kimi_scores)
    claude_avg = sum(claude_scores) / len(claude_scores)
    llama_avg = sum(llama_scores) / len(llama_scores)
    gpt5_avg = (sum(gpt5_scores) / len(gpt5_scores)) if gpt5_scores else None
    qwen7b_avg = (sum(qwen7b_scores) / len(qwen7b_scores)) if qwen7b_scores else None

    # Theme — match the dark palette used on the landing page and blog
    bg = "#0f172a"
    surface = "#1e293b"
    text_color = "#e2e8f0"
    muted = "#94a3b8"
    color_baseline = "#818cf8"   # indigo
    color_kimi = "#4ade80"        # green
    color_claude = "#fbbf24"      # amber
    color_gpt5 = "#22d3ee"        # cyan
    color_qwen7b = "#a78bfa"      # purple (open-source zero-shot)
    color_llama = "#f87171"       # red
    novel_shade = "#334155"

    fig, ax = plt.subplots(figsize=(18, 7), facecolor=bg)
    ax.set_facecolor(bg)

    x = list(range(len(task_order)))
    if qwen7b_scores and gpt5_scores:
        n_bars = 6
        width = 0.14
        offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]  # baseline / kimi / claude / qwen7b / gpt5 / llama
    elif gpt5_scores:
        n_bars = 5
        width = 0.16
        offsets = [-2, -1, 0, 1, 2]
    else:
        n_bars = 4
        width = 0.20
        offsets = [-1.5, -0.5, 0.5, 1.5]

    # Novel-task background shading (T9, T10)
    for i, t in enumerate(task_order):
        if t in NOVEL_TASKS:
            ax.axvspan(i - 0.5, i + 0.5, color=novel_shade, alpha=0.35, zorder=0)

    # Bars — laid out in leaderboard order (baseline / kimi / claude / qwen7b / gpt5 / llama)
    bars_b = ax.bar([xi + offsets[0] * width for xi in x], baseline_scores, width,
                    label=f"Rule-Based Baseline (avg {baseline_avg:.1f})",
                    color=color_baseline, edgecolor=bg, linewidth=0.5, zorder=3)
    bars_k = ax.bar([xi + offsets[1] * width for xi in x], kimi_scores, width,
                    label=f"Kimi Moonshot V1-128k (avg {kimi_avg:.1f})",
                    color=color_kimi, edgecolor=bg, linewidth=0.5, zorder=3)
    bars_c = ax.bar([xi + offsets[2] * width for xi in x], claude_scores, width,
                    label=f"Claude Sonnet 4.6 (avg {claude_avg:.1f})",
                    color=color_claude, edgecolor=bg, linewidth=0.5, zorder=3)
    bars_q = None
    if qwen7b_scores:
        bars_q = ax.bar([xi + offsets[3] * width for xi in x], qwen7b_scores, width,
                        label=f"Qwen2.5-7B (open-source, zero-shot) (avg {qwen7b_avg:.1f})",
                        color=color_qwen7b, edgecolor=bg, linewidth=0.5, zorder=3)
    bars_g = None
    if gpt5_scores:
        gpt5_offset_idx = 4 if qwen7b_scores else 3
        bars_g = ax.bar([xi + offsets[gpt5_offset_idx] * width for xi in x], gpt5_scores, width,
                        label=f"GPT-5 (avg {gpt5_avg:.1f})",
                        color=color_gpt5, edgecolor=bg, linewidth=0.5, zorder=3)
    bars_l = ax.bar([xi + offsets[-1] * width for xi in x], llama_scores, width,
                    label=f"Llama 3.3 70B (avg {llama_avg:.1f})",
                    color=color_llama, edgecolor=bg, linewidth=0.5, zorder=3)

    # Value labels on top of each bar
    def label_bars(bars, values, color, offset=1.2):
        if bars is None:
            return
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7, color=color, fontweight="bold", zorder=4)

    label_bars(bars_b, baseline_scores, color_baseline)
    label_bars(bars_k, kimi_scores, color_kimi)
    label_bars(bars_c, claude_scores, color_claude)
    label_bars(bars_q, qwen7b_scores if qwen7b_scores else [], color_qwen7b)
    label_bars(bars_g, gpt5_scores if gpt5_scores else [], color_gpt5)
    label_bars(bars_l, llama_scores, color_llama)

    # Baseline reference line
    ax.axhline(baseline_avg, color=color_baseline, linestyle="--",
               linewidth=1, alpha=0.45, zorder=1)

    # Callout for T9 cross-model discrimination
    t9_idx = task_order.index("T9_adaptive_adversary")
    if gpt5_scores:
        gap_text = (f"T9 separates execution vs reasoning\n"
                    f"Kimi/Claude {kimi_scores[t9_idx]:.1f}  "
                    f"GPT-5 {gpt5_scores[t9_idx]:.1f}  "
                    f"Llama {llama_scores[t9_idx]:.1f}\n"
                    f"→ frontier ≠ single point")
    else:
        gap_text = (f"T9 frontier-vs-sub-frontier gap\n"
                    f"Kimi / Claude {kimi_scores[t9_idx]:.1f}  vs  Llama {llama_scores[t9_idx]:.1f}\n"
                    f"→ {kimi_scores[t9_idx] - llama_scores[t9_idx]:.1f} pts")
    ax.annotate(gap_text,
                xy=(t9_idx + offsets[-1] * width, llama_scores[t9_idx] + 2),
                xytext=(t9_idx - 2.6, 45),
                fontsize=9, color=text_color, fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.45", facecolor=surface,
                          edgecolor=color_llama, linewidth=1),
                arrowprops=dict(arrowstyle="->", color=muted, lw=1.2,
                                connectionstyle="arc3,rad=-0.15"))

    # Novel marker text (put below the chart area to avoid bar overlap)
    ax.text(task_order.index("T10_constrained_budget") - 0.5, -6,
            "★ Novel tasks (shaded)", ha="center", va="top",
            fontsize=9, color=muted, style="italic")

    # Chart chrome
    ax.set_title("ComtradeBench — Cross-model Results on the 10-Task Suite",
                 fontsize=14, color=text_color, fontweight="bold", pad=16)
    ax.set_ylabel("Score (0-100)", color=text_color, fontsize=11)
    ax.set_ylim(-10, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, color=text_color, fontsize=9.5)
    ax.tick_params(axis="y", colors=text_color)

    for spine in ax.spines.values():
        spine.set_color(muted)
        spine.set_alpha(0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", color=muted, alpha=0.15, linestyle="-", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="lower left", fontsize=9.5, facecolor=surface,
                       edgecolor=muted, labelcolor=text_color, framealpha=0.95)
    legend.get_frame().set_alpha(0.95)

    # Footer caption
    if qwen7b_scores:
        caption = (f"Closed frontier (Kimi, Claude) == open-source Qwen2.5-7B zero-shot ({qwen7b_avg:.1f}) — "
                   f"benchmark solvable by open 7B models without any training. GPT-5 trails (reasons too long). "
                   f"Llama bimodal across seeds.")
    elif gpt5_scores:
        caption = ("Kimi = Claude at 97.5 (execution-oriented frontier). "
                   "GPT-5 trails at 93.2 — reasons longer but executes fewer steps. "
                   "Llama at 89.3 with T9-specific collapse. "
                   "T9 separates execution-oriented from reasoning-oriented frontier.")
    else:
        caption = ("Kimi and Claude are numerically indistinguishable at the top (both 97.5). "
                   "T9 discriminates frontier from sub-frontier (Llama collapses to 18.7).")
    fig.text(0.5, 0.015, caption,
             ha="center", fontsize=9, color=muted, style="italic")

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = root / "benchmark_results.png"
    fig.savefig(out_path, dpi=140, facecolor=bg, bbox_inches="tight")
    print(f"Saved: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
