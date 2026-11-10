"""DVHs of the best plan found by each method for one patient.

  python scripts/plot_dvh.py results/<experiment> --patient Lung_Patient_3

Re-solves each method's best weights (one solve per method) and draws one
panel per scored structure, one line per method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import COLORS, GRID, INK, LABELS, METHOD_ORDER, MUTED  # noqa: E402
from rt_agent.config import build_backend, load_config  # noqa: E402
from rt_agent.scorecard import Scorecard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--patient", required=True)
    args = ap.parse_args()
    out = Path(args.folder)
    cfg = load_config(out / "config_used.json")
    backend = build_backend(cfg, args.patient)
    structures = list(dict.fromkeys(c.structure for c in Scorecard.from_json(cfg["scorecard"]).criteria))

    best = {}
    for s in out.glob(f"{args.patient}/*/rep*/summary.json"):
        d = json.loads(s.read_text())
        if d["method"] not in best or d["best_score"] > best[d["method"]]["best_score"]:
            best[d["method"]] = d

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(structures)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for m in [m for m in METHOD_ORDER if m in best]:
        res = backend.solve(best[m]["best_weights"])
        for ax, st in zip(axes.flat, structures):
            dvh = backend.dvh(res, st)
            if dvh is not None:
                ax.plot(dvh[0], dvh[1], color=COLORS[m], lw=2,
                        label=f"{LABELS[m]} ({best[m]['best_score']:.1f})")
    for ax, st in zip(axes.flat, structures):
        ax.set_title(st, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Dose [Gy]", color=INK)
        ax.set_ylabel("Volume [%]", color=INK)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=min(3, len(h)), fontsize=9)
    fig.suptitle(f"Best plan per method - {args.patient} (legend: best score)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path = out / f"fig_dvh_{args.patient}.png"
    fig.savefig(path, dpi=200)
    print("saved", path)


if __name__ == "__main__":
    main()
