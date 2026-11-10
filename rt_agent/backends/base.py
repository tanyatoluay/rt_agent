"""Backend interface: anything that can (re)optimize a plan for a given set of
objective weights and report dose-volume metrics.

Two implementations ship with the project:
  * ToyBackend    - a small 2D phantom solved with SciPy. Runs anywhere in
                    seconds; use it to develop and test the agent loop.
  * PortPyBackend - the real thing (PortPy lung IMRT, solved with CVXPY).

The environment, scorecard, agent, baselines and MCP server only talk to this
interface, so every method is evaluated through exactly the same code path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Objective:
    """One term w_i * f_i(d) of the inverse-planning cost function."""

    id: str                 # stable identifier the agent uses, e.g. "obj2_quadratic_CORD"
    type: str               # PortPy objective type, e.g. "quadratic-overdose"
    structure: str          # structure name, e.g. "PTV"
    weight: float           # weight in the starting (fixed-weight) plan
    tunable: bool = True
    min_weight: float = 0.0
    max_weight: float = float("inf")
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific info


@dataclass
class SolveResult:
    status: str             # "optimal", "failed", ...
    solve_time_s: float
    handle: Any = None      # backend-specific solution object (kept in memory only)
    message: str = ""


class PlanningBackend(ABC):
    """Abstract planning backend."""

    name: str = "base"

    @abstractmethod
    def objectives(self) -> list[Objective]:
        """Objectives of the selected protocol, with their initial weights."""

    @abstractmethod
    def solve(self, weights: dict[str, float]) -> SolveResult:
        """Optimize beamlet intensities for the given objective weights."""

    @abstractmethod
    def metric(self, result: SolveResult, kind: str, structure: str,
               param: float | None = None) -> float:
        """Dose-volume metric of a solved plan.

        kind:
          "D"    - dose [Gy] received by `param` % of the structure volume (D95 -> param=95)
          "V"    - % of the structure volume receiving >= `param` Gy (V20 -> param=20)
          "mean" - mean dose [Gy]
          "max"  - maximum dose [Gy]
        """

    @abstractmethod
    def structure_names(self) -> list[str]:
        ...

    def dvh(self, result: SolveResult, structure: str,
            n_points: int = 200) -> tuple[np.ndarray, np.ndarray] | None:
        """Optional cumulative DVH (dose_gy, volume_percent) for plots."""
        return None


# ---- shared DVH helpers ------------------------------------------------------

def dose_at_volume(dose: np.ndarray, volume_percent: float,
                   voxel_weights: np.ndarray | None = None) -> float:
    """D_x: minimum dose received by the hottest x% of the volume."""
    if dose.size == 0:
        return float("nan")
    w = np.ones_like(dose) if voxel_weights is None else voxel_weights
    order = np.argsort(dose)[::-1]
    cum = np.cumsum(w[order]) / w.sum() * 100.0
    idx = int(np.searchsorted(cum, volume_percent, side="left"))
    idx = min(idx, dose.size - 1)
    return float(dose[order][idx])


def volume_at_dose(dose: np.ndarray, dose_gy: float,
                   voxel_weights: np.ndarray | None = None) -> float:
    """V_x: % of the volume receiving at least x Gy."""
    if dose.size == 0:
        return float("nan")
    w = np.ones_like(dose) if voxel_weights is None else voxel_weights
    return float(w[dose >= dose_gy].sum() / w.sum() * 100.0)


def cumulative_dvh(dose: np.ndarray, voxel_weights: np.ndarray | None = None,
                   n_points: int = 200, max_dose: float | None = None):
    w = np.ones_like(dose) if voxel_weights is None else voxel_weights
    top = max_dose if max_dose is not None else float(dose.max()) * 1.05 + 1e-9
    grid = np.linspace(0.0, top, n_points)
    vol = np.array([w[dose >= g].sum() for g in grid]) / w.sum() * 100.0
    return grid, vol
