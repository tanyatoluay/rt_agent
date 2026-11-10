"""Load the experiment config and build environments from it."""
from __future__ import annotations

import json
from pathlib import Path

from .backends import PlanningBackend, make_backend
from .environment import PlanningEnvironment
from .logger import RunLogger
from .scorecard import Scorecard

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    cfg = json.loads(p.read_text())
    sc = Path(cfg["scorecard"])
    cfg["scorecard"] = str(sc if sc.is_absolute() else PROJECT_ROOT / sc)
    pp = cfg.get("portpy", {})
    if pp.get("data_dir") and not Path(pp["data_dir"]).is_absolute():
        pp["data_dir"] = str(PROJECT_ROOT / pp["data_dir"])
    return cfg


def build_backend(cfg: dict, patient_id: str) -> PlanningBackend:
    return make_backend(cfg["backend"], patient_id, cfg)


def build_env(cfg: dict, backend: PlanningBackend, *, history_visible: bool = True,
              out_dir: str | Path | None = None, solve_cache: dict | None = None) -> PlanningEnvironment:
    return PlanningEnvironment(
        backend, Scorecard.from_json(cfg["scorecard"]),
        budget=cfg["budget"],
        allowed_factors=tuple(cfg["allowed_factors"]),
        max_adjustments_per_iteration=cfg["max_adjustments_per_iteration"],
        history_visible=history_visible,
        logger=RunLogger(out_dir),
        solve_cache=solve_cache,
    )
