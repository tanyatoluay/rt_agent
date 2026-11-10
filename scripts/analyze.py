"""Summarize an experiment folder: tables, paired statistics and figures.

  python scripts/analyze.py results/<experiment>

Writes into that folder:
  summary_by_method.csv / .md   mean (sd) per method of every outcome
  paired_tests.md               LLM vs each method, paired by patient (Wilcoxon signed-rank)
  fig_convergence.png           best-so-far score vs optimizer calls
  fig_best_score_by_patient.png best score per patient and method
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METHOD_ORDER = ["fixed", "rule", "random", "llm", "llm_no_history"]
LABELS = {"fixed": "Fixed weights", "rule": "Rule-based", "random": "Random search",
          "llm": "LLM agent", "llm_no_history": "LLM agent (no history)"}
# Fixed categorical order (validated default palette, light mode); color follows the method.
COLORS = {"fixed": "#2a78d6", "rule": "#eb6834", "random": "#1baf7a",
          "llm": "#eda100", "llm_no_history": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

OUTCOMES = [("best_score", "Best plan score"), ("best_n_passed", "Criteria passed (best plan)"),
            ("optimizer_calls_used", "Optimizer calls used"), ("best_iteration", "Iteration of best plan"),
            ("invalid_tool_calls", "Invalid tool calls"), ("total_solver_time_s", "Solver time [s]"),
            ("wall_time_s", "Wall time [s]"), ("api_calls", "LLM API calls"),
            ("input_tokens", "Input tokens"), ("output_tokens", "Output tokens")]


def read_csv(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main(folder: str) -> None:
    out = Path(folder)
    runs = read_csv(out / "runs.csv")
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in runs)]
    patients = sorted({r["patient"] for r in runs})

    # ---------- summary table
    lines = ["| Outcome | " + " | ".join(LABELS[m] for m in methods) + " |",
             "|---|" + "---|" * len(methods)]
    csv_rows = []
    for key, label in OUTCOMES:
        cells, any_val = [], False
        for m in methods:
            v = np.array([num(r.get(key)) for r in runs if r["method"] == m])
            v = v[~np.isnan(v)]
            if v.size:
                any_val = True
                cells.append(f"{v.mean():.2f} ({v.std(ddof=1) if v.size > 1 else 0:.2f})")
                csv_rows.append({"outcome": key, "method": m, "mean": v.mean(),
                                 "sd": v.std(ddof=1) if v.size > 1 else 0.0, "n": v.size})
            else:
                cells.append("-")
        if any_val:
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    md = "Mean (SD) over all patients and repeats.\n\n" + "\n".join(lines) + "\n"
    (out / "summary_by_method.md").write_text(md)
    with open(out / "summary_by_method.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["outcome", "method", "mean", "sd", "n"])
        w.writeheader()
        w.writerows(csv_rows)
    print(md)

    # ---------- paired comparison by patient (repeats averaged within patient first)
    per_patient = defaultdict(dict)
    for p in patients:
        for m in methods:
            v = [num(r["best_score"]) for r in runs if r["patient"] == p and r["method"] == m]
            if v:
                per_patient[m][p] = float(np.mean(v))
    test_lines = ["# Paired comparisons of best plan score (per-patient means)\n"]
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    for ref in [m for m in ("llm", "llm_no_history") if m in methods]:
        for other in [m for m in methods if m != ref]:
            common = sorted(set(per_patient[ref]) & set(per_patient[other]))
            if not common:
                continue
            d = np.array([per_patient[ref][p] - per_patient[other][p] for p in common])
            wins = int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())
            pval = "n/a"
            if wilcoxon and len(common) >= 2 and np.any(np.abs(d) > 1e-9):
                pval = f"{wilcoxon(d).pvalue:.4f}"
            test_lines.append(f"- **{LABELS[ref]} vs {LABELS[other]}** (n = {len(common)} patients): "
                              f"mean difference {d.mean():+.2f} points, wins/ties/losses {wins[0]}/{wins[1]}/{wins[2]}, "
                              f"Wilcoxon p = {pval}")
    if len(patients) < 6:
        test_lines.append(f"\nNote: only {len(patients)} patient(s). The Wilcoxon signed-rank test cannot "
                          "reach p < 0.05 with fewer than 6 paired cases; report per-patient results and "
                          "effect sizes, and treat p-values as descriptive.")
    (out / "paired_tests.md").write_text("\n".join(test_lines) + "\n")
    print("\n".join(test_lines))

    # ---------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})

    iters = read_csv(out / "iterations.csv") if (out / "iterations.csv").exists() else []
    if iters:
        budget = int(max(num(r["calls_used"]) for r in iters))
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ends = []
        for m in methods:
            # best-so-far after k calls, carried forward if a run stopped early
            runs_m = defaultdict(dict)
            for r in iters:
                if r["method"] == m:
                    runs_m[(r["patient"], r["repeat"])][int(num(r["calls_used"]))] = num(r["best_so_far"])
            if not runs_m:
                continue
            curves = []
            for traj in runs_m.values():
                c, last = [], np.nan
                for k in range(budget + 1):
                    last = traj.get(k, last)
                    c.append(last)
                curves.append(c)
            mean = np.nanmean(np.array(curves), axis=0)
            ax.plot(range(budget + 1), mean, color=COLORS[m], lw=2, marker="o", ms=4, label=LABELS[m])
            ends.append((mean[-1], m))
        # direct labels at the line ends, nudged apart so they never collide
        ends.sort()
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        placed = []
        for y, m in ends:
            yy = y if not placed else max(y, placed[-1] + 0.045 * span)
            placed.append(yy)
            ax.annotate(LABELS[m], (budget, y), xytext=(budget + 0.25, yy), color=INK, fontsize=8.5,
                        va="center", annotation_clip=False)
        ax.set_xlim(-0.3, budget + 0.3)
        ax.set_xlabel("Optimizer calls used")
        ax.set_ylabel("Best plan score so far (mean)")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=min(3, len(methods)), fontsize=8.5)
        ax.set_title("Convergence within the optimization budget", loc="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.subplots_adjust(right=0.76)
        fig.savefig(out / "fig_convergence.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(patients) + 2), 4))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        xs = np.arange(len(patients)) + (i - (len(methods) - 1) / 2) * width
        for x, p in zip(xs, patients):
            v = [num(r["best_score"]) for r in runs if r["patient"] == p and r["method"] == m]
            if not v:
                continue
            ax.scatter([x] * len(v), v, color=COLORS[m], s=18, alpha=0.45, edgecolor="none", zorder=2)
            ax.scatter([x], [np.mean(v)], color=COLORS[m], s=55, edgecolor="white", linewidth=1.5, zorder=3,
                       label=LABELS[m] if p == patients[0] else None)
    ax.set_xticks(range(len(patients)), patients)
    ax.set_ylabel("Best plan score")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, ncol=min(3, len(methods)), loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Best plan score by patient (large dot = mean, small = repeats)", loc="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_best_score_by_patient.png", dpi=200)
    plt.close(fig)
    print(f"\nFigures and tables written to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
