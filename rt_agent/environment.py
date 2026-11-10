"""The planning environment shared by the LLM agent, the baselines and the MCP server.

It owns everything that must be identical across methods:
  * the backend (optimizer) and the fixed starting weights
  * the optimization budget (counted here, not by the agent)
  * the deterministic validation layer (which weights may change, by how much)
  * the independent scorecard evaluation of every plan
  * logging of every call

Methods only interact through the six tool functions below.
"""
from __future__ import annotations

import math
import time
from typing import Any

from .backends.base import Objective, PlanningBackend, SolveResult
from .logger import RunLogger
from .scorecard import Scorecard


class ToolError(Exception):
    """Invalid tool call. Returned to the caller as {"error": ...}; never changes state."""


class PlanningEnvironment:
    def __init__(self, backend: PlanningBackend, scorecard: Scorecard, *,
                 budget: int = 10,
                 allowed_factors: tuple[float, ...] = (0.5, 0.75, 1.5, 2.0),
                 max_adjustments_per_iteration: int = 3,
                 history_visible: bool = True,
                 logger: RunLogger | None = None,
                 solve_cache: dict | None = None):
        self.backend = backend
        self.scorecard = scorecard
        self.budget = budget
        self.allowed_factors = tuple(float(f) for f in allowed_factors)
        self.max_adjustments = max_adjustments_per_iteration
        self.history_visible = history_visible
        self.log = logger or RunLogger(None)
        # Cache shared across methods for the same patient: identical weights -> identical plan.
        # Calls are still charged to the budget; only wall-clock time is saved.
        self.solve_cache = solve_cache if solve_cache is not None else {}

        self.objectives: dict[str, Objective] = {o.id: o for o in backend.objectives()}
        self._obs_metrics = self._observation_metric_specs()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> dict:
        self.current_weights = {oid: o.weight for oid, o in self.objectives.items()}
        self.pending_weights = dict(self.current_weights)
        self.adjustments_this_iteration = 0
        self.calls_used = 0
        self.invalid_calls = 0
        self.iterations: list[dict] = []
        self._handles: dict[int, SolveResult] = {}
        rec = self._solve_and_record(counted=False)   # iteration 0 = fixed-weight plan
        self.log.event(tool="reset", result={"iteration": 0, "score": rec["evaluation"]["score"]})
        return rec

    @property
    def remaining_budget(self) -> int:
        return self.budget - self.calls_used

    @property
    def done(self) -> bool:
        return self.remaining_budget <= 0

    @property
    def current_iteration(self) -> int:
        return self.iterations[-1]["iteration"]

    def best_iteration(self) -> dict:
        ok = [it for it in self.iterations if it["status"] != "failed"]
        return max(ok, key=lambda it: it["evaluation"]["score"]) if ok else self.iterations[0]

    # ----------------------------------------------------------- internals
    def _observation_metric_specs(self) -> list[tuple[str, str, str, float | None]]:
        """Raw dose metrics the agent can observe: every scorecard metric plus mean/max
        of every scored structure (and D98/D2 of the target)."""
        specs, seen = [], set()

        def add(name, kind, struct, param):
            if name not in seen:
                seen.add(name)
                specs.append((name, kind, struct, param))

        for c in self.scorecard.criteria:
            add(c.name, c.kind, c.structure, c.param)
        for s in dict.fromkeys(c.structure for c in self.scorecard.criteria):
            add(f"{s}_mean", "mean", s, None)
            add(f"{s}_max", "max", s, None)
        if any(c.structure == "PTV" for c in self.scorecard.criteria):
            add("PTV_D98", "D", "PTV", 98)
            add("PTV_D2", "D", "PTV", 2)
        return specs

    def _cache_key(self, weights: dict[str, float]) -> tuple:
        return tuple(sorted((k, round(v, 9)) for k, v in weights.items()))

    def _solve_and_record(self, counted: bool) -> dict:
        weights = dict(self.pending_weights)
        key = self._cache_key(weights)
        cached = key in self.solve_cache
        if cached:
            result, metrics = self.solve_cache[key]
        else:
            result = self.backend.solve(weights)
            metrics = {}
            if result.status != "failed":
                for name, kind, struct, param in self._obs_metrics:
                    try:
                        metrics[name] = self.backend.metric(result, kind, struct, param)
                    except Exception:
                        metrics[name] = float("nan")
            self.solve_cache[key] = (result, metrics)

        evaluation = self.scorecard.evaluate(metrics) if result.status != "failed" else \
            {"score": 0.0, "n_passed": 0, "n_criteria": len(self.scorecard.criteria), "criteria": []}
        it = len(self.iterations)
        if counted:
            self.calls_used += 1
        rec = {
            "iteration": it,
            "weights": weights,
            "status": result.status,
            "message": result.message,
            "solve_time_s": round(result.solve_time_s, 3),
            "cached": cached,
            "counted": counted,
            "calls_used": self.calls_used,
            "metrics": {k: (None if (isinstance(v, float) and math.isnan(v)) else round(v, 3))
                        for k, v in metrics.items()},
            "evaluation": evaluation,
        }
        self.iterations.append(rec)
        self._handles[it] = result
        self.current_weights = dict(weights)
        self.pending_weights = dict(weights)
        self.adjustments_this_iteration = 0
        self.log.iteration(**rec)
        return rec

    def _resolve_iteration(self, iteration: int | None) -> dict:
        if iteration is None or iteration == self.current_iteration:
            return self.iterations[-1]
        if not self.history_visible:
            raise ToolError("Previous iterations are not available in this configuration; "
                            "omit `iteration` to get the current plan.")
        if not (0 <= iteration < len(self.iterations)):
            raise ToolError(f"iteration must be between 0 and {len(self.iterations) - 1}")
        return self.iterations[iteration]

    # =============================================================== tools
    def get_plan_metrics(self, iteration: int | None = None) -> dict:
        it = self._resolve_iteration(iteration)
        return {"iteration": it["iteration"], "status": it["status"], "dose_metrics": it["metrics"],
                "units": "Gy for doses (D, mean, max); % volume for V metrics"}

    def get_objective_weights(self) -> dict:
        rows = []
        for oid, o in self.objectives.items():
            rows.append({
                "objective_id": oid, "type": o.type, "structure": o.structure,
                "current_plan_weight": self.current_weights[oid],
                "pending_weight": self.pending_weights[oid],
                "initial_weight": o.weight,
                "tunable": o.tunable,
                "bounds": [o.min_weight, o.max_weight] if o.tunable else None,
            })
        return {"objectives": rows,
                "allowed_factors": list(self.allowed_factors),
                "adjustments_left_before_next_optimization": self.max_adjustments - self.adjustments_this_iteration,
                "optimizer_calls_remaining": self.remaining_budget}

    def adjust_weight(self, objective_id: str, factor: float) -> dict:
        o = self.objectives.get(objective_id)
        if o is None:
            raise ToolError(f"Unknown objective_id '{objective_id}'. Call get_objective_weights().")
        if not o.tunable:
            raise ToolError(f"Objective '{objective_id}' is fixed and cannot be changed.")
        if not any(math.isclose(float(factor), f, rel_tol=1e-9) for f in self.allowed_factors):
            raise ToolError(f"factor must be one of {list(self.allowed_factors)}")
        if self.adjustments_this_iteration >= self.max_adjustments:
            raise ToolError(f"At most {self.max_adjustments} adjustments per optimization; call run_optimizer().")
        old = self.pending_weights[objective_id]
        new = min(max(old * float(factor), o.min_weight), o.max_weight)
        if math.isclose(new, old):
            raise ToolError(f"Weight is already at its bound ({old}); this change has no effect.")
        self.pending_weights[objective_id] = new
        self.adjustments_this_iteration += 1
        return {"objective_id": objective_id, "old_weight": old, "new_weight": new,
                "clipped_to_bound": not math.isclose(new, old * float(factor)),
                "adjustments_left_before_next_optimization": self.max_adjustments - self.adjustments_this_iteration}

    def run_optimizer(self) -> dict:
        if self.done:
            raise ToolError("Optimization budget exhausted. No further optimizer calls are allowed.")
        if self.pending_weights == self.current_weights:
            raise ToolError("No pending weight changes; adjust at least one weight before optimizing.")
        changed = {k: [self.current_weights[k], v] for k, v in self.pending_weights.items()
                   if not math.isclose(v, self.current_weights[k])}
        t0 = time.perf_counter()
        rec = self._solve_and_record(counted=True)
        best = self.best_iteration()
        return {"iteration": rec["iteration"], "status": rec["status"],
                "solver_message": rec["message"] or None,
                "solve_time_s": rec["solve_time_s"], "wall_time_s": round(time.perf_counter() - t0, 3),
                "changed_weights": changed,
                "score": rec["evaluation"]["score"], "n_passed": rec["evaluation"]["n_passed"],
                "n_criteria": rec["evaluation"]["n_criteria"],
                "best_score_so_far": best["evaluation"]["score"],
                "optimizer_calls_used": self.calls_used, "optimizer_calls_remaining": self.remaining_budget}

    def evaluate_plan(self, iteration: int | None = None) -> dict:
        it = self._resolve_iteration(iteration)
        return {"iteration": it["iteration"], **it["evaluation"]}

    def compare_iterations(self, iteration_a: int, iteration_b: int) -> dict:
        if not self.history_visible:
            raise ToolError("compare_iterations is disabled in this configuration.")
        a, b = self._resolve_iteration(iteration_a), self._resolve_iteration(iteration_b)
        wdiff = {k: {"a": a["weights"][k], "b": b["weights"][k]}
                 for k in a["weights"] if not math.isclose(a["weights"][k], b["weights"][k])}
        mdiff = {}
        for k in a["metrics"]:
            va, vb = a["metrics"].get(k), b["metrics"].get(k)
            if va is not None and vb is not None:
                mdiff[k] = {"a": va, "b": vb, "b_minus_a": round(vb - va, 3)}
        return {"iteration_a": a["iteration"], "iteration_b": b["iteration"],
                "score": {"a": a["evaluation"]["score"], "b": b["evaluation"]["score"]},
                "n_passed": {"a": a["evaluation"]["n_passed"], "b": b["evaluation"]["n_passed"]},
                "weight_differences": wdiff, "metric_differences": mdiff}

    # ------------------------------------------------------------ dispatch
    TOOL_NAMES = ("get_plan_metrics", "get_objective_weights", "adjust_weight",
                  "run_optimizer", "evaluate_plan", "compare_iterations")

    def dispatch(self, name: str, args: dict[str, Any] | None = None, caller: str = "") -> dict:
        """Single entry point used by the agent, baselines and MCP server."""
        args = dict(args or {})
        try:
            if name not in self.TOOL_NAMES:
                raise ToolError(f"Unknown tool '{name}'.")
            if name == "compare_iterations" and not self.history_visible:
                raise ToolError("compare_iterations is disabled in this configuration.")
            result = getattr(self, name)(**args)
            ok = True
        except ToolError as e:
            result, ok = {"error": str(e)}, False
        except TypeError as e:  # wrong/missing arguments
            result, ok = {"error": f"Invalid arguments for {name}: {e}"}, False
        if not ok:
            self.invalid_calls += 1
        self.log.event(caller=caller, tool=name, args=args, ok=ok,
                       result=result if name not in ("get_plan_metrics", "evaluate_plan",
                                                     "compare_iterations", "get_objective_weights") else "(omitted)")
        return result

    def summary(self) -> dict:
        best = self.best_iteration()
        init = self.iterations[0]
        return {
            "initial_score": init["evaluation"]["score"],
            "initial_n_passed": init["evaluation"]["n_passed"],
            "best_score": best["evaluation"]["score"],
            "best_n_passed": best["evaluation"]["n_passed"],
            "best_iteration": best["iteration"],
            "final_score": self.iterations[-1]["evaluation"]["score"],
            "optimizer_calls_used": self.calls_used,
            "invalid_tool_calls": self.invalid_calls,
            "n_failed_solves": sum(it["status"] == "failed" for it in self.iterations),
            "total_solver_time_s": round(sum(it["solve_time_s"] for it in self.iterations if it["counted"]), 3),
            "best_metrics": best["metrics"],
            "best_weights": best["weights"],
            "trajectory": [it["evaluation"]["score"] for it in self.iterations],
        }
