"""Baseline weight-tuning methods. They use exactly the same tools, admissible
actions and budget as the LLM agent (env.dispatch), so comparisons are fair."""
from __future__ import annotations

import numpy as np

from .environment import PlanningEnvironment


def run_fixed(env: PlanningEnvironment, **_) -> dict:
    """Fixed-weight optimization: the initial plan, no tuning."""
    return {"stop_reason": "no tuning"}


def run_random(env: PlanningEnvironment, rng: np.random.Generator, **_) -> dict:
    """Random search: at each step apply 1..k random admissible adjustments, then optimize.
    It is a random walk from the current plan (same action space as the agent)."""
    tunable = [oid for oid, o in env.objectives.items() if o.tunable]
    factors = list(env.allowed_factors)
    while not env.done:
        k = int(rng.integers(1, env.max_adjustments + 1))
        applied = 0
        for _ in range(k):
            # sample only actions that are admissible right now (weight not stuck at a bound),
            # so random search never wastes tool calls on rejected actions
            options = []
            for oid in tunable:
                o, w = env.objectives[oid], env.pending_weights[oid]
                for f in factors:
                    if min(max(w * f, o.min_weight), o.max_weight) != w:
                        options.append((oid, f))
            if not options:
                break
            oid, f = options[int(rng.integers(len(options)))]
            r = env.dispatch("adjust_weight", {"objective_id": oid, "factor": float(f)}, caller="random")
            applied += "error" not in r
        if applied == 0:
            return {"stop_reason": "no admissible adjustment found"}
        env.dispatch("run_optimizer", caller="random")
    return {"stop_reason": "budget exhausted"}


def _objectives_for(env: PlanningEnvironment, structure: str, preferred_types: list[str]) -> list[str]:
    cands = [o for o in env.objectives.values()
             if o.tunable and o.structure.upper() == structure.upper()]
    cands.sort(key=lambda o: preferred_types.index(o.type) if o.type in preferred_types else 99)
    return [o.id for o in cands]


def run_rule_based(env: PlanningEnvironment, target_structures: tuple[str, ...] = ("PTV",), **_) -> dict:
    """Deterministic rule-based tuning, mimicking a simple planner heuristic:

      1. Rank criteria: failed ones first (lowest score, then highest priority);
         if everything passes, rank the not-yet-ideal criteria the same way.
      2. For each ranked criterion (up to the per-iteration adjustment limit):
           target under-coverage  -> underdose objective on that target x2.0
           target hot spot        -> overdose objective on that target x1.5
           OAR above its limit    -> that OAR's objective x2.0
           OAR passing, not ideal -> that OAR's objective x1.5
      3. Optimize. Stop early if no rule produces an admissible action.
    """
    factors = env.allowed_factors
    up_big = max(factors)
    up_small = min((f for f in factors if f > 1), default=up_big)
    targets = {s.upper() for s in target_structures}

    while not env.done:
        ev = env.evaluate_plan()
        crits = ev["criteria"]
        failed = [c for c in crits if not c["passed"]]
        pool = failed if failed else [c for c in crits if c["score"] < 1.0]
        if not pool:
            return {"stop_reason": "all criteria at goal"}
        pool.sort(key=lambda c: (c["score"], -c["priority"]))

        applied, touched = 0, set()
        for c in pool:
            if applied >= env.max_adjustments:
                break
            struct = c["structure"]
            is_target = struct.upper() in targets
            coverage = c["requirement"].startswith(">=")
            if is_target and coverage:
                ids, f = _objectives_for(env, struct, ["quadratic-underdose", "linear-underdose"]), up_big
            elif is_target:
                ids, f = _objectives_for(env, struct, ["quadratic-overdose", "linear-overdose"]), up_small
            else:
                ids, f = _objectives_for(env, struct, ["quadratic", "quadratic-overdose", "linear-overdose"]), \
                    (up_big if not c["passed"] else up_small)
            for oid in [i for i in ids if i not in touched][:1]:
                r = env.dispatch("adjust_weight", {"objective_id": oid, "factor": f}, caller="rule")
                touched.add(oid)
                applied += "error" not in r
        if applied == 0:
            return {"stop_reason": "no applicable rule"}
        env.dispatch("run_optimizer", caller="rule")
    return {"stop_reason": "budget exhausted"}


BASELINES = {"fixed": run_fixed, "random": run_random, "rule": run_rule_based}
