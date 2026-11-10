"""Independent plan-quality scorecard.

The scorecard is loaded from a JSON file and never modified by any method
(agent or baseline). Each criterion is scored with a piecewise-linear function:

    better than or equal to `goal`  -> 1.0
    exactly at `limit`              -> 0.5   (criterion "passed" at or better than limit)
    between goal and limit          -> linear 1.0 .. 0.5
    beyond limit                    -> linear 0.5 .. 0.0, reaching 0 at limit + (limit - goal)

Plan score = 100 * sum(priority_i * s_i) / sum(priority_i), so it keeps
rewarding OAR sparing after every criterion passes (unlike a pure pass count).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Criterion:
    name: str
    structure: str
    kind: str            # "D", "V", "mean", "max"
    param: float | None  # 95 for D95, 20 for V20, None for mean/max
    direction: str       # "lower" = value must be >= limit ; "upper" = value must be <= limit
    goal: float
    limit: float
    priority: float = 1.0
    unit: str = "Gy"

    def score(self, value: float) -> float:
        if value is None or math.isnan(value):
            return 0.0
        # convert to "badness": positive when worse than goal
        span = abs(self.limit - self.goal) or 1e-9
        if self.direction == "upper":
            excess = (value - self.goal) / span
        else:
            excess = (self.goal - value) / span
        if excess <= 0:
            return 1.0
        if excess <= 1:
            return 1.0 - 0.5 * excess
        return max(0.0, 0.5 - 0.5 * (excess - 1))

    def passed(self, value: float) -> bool:
        if value is None or math.isnan(value):
            return False
        return value >= self.limit if self.direction == "lower" else value <= self.limit


class Scorecard:
    def __init__(self, criteria: list[Criterion], name: str = "scorecard"):
        self.criteria = criteria
        self.name = name

    @classmethod
    def from_json(cls, path: str | Path) -> "Scorecard":
        spec = json.loads(Path(path).read_text())
        return cls([Criterion(**c) for c in spec["criteria"]], name=spec.get("name", Path(path).stem))

    def required_metrics(self) -> list[tuple[str, str, float | None]]:
        return [(c.kind, c.structure, c.param) for c in self.criteria]

    def evaluate(self, metric_values: dict[str, float]) -> dict:
        """metric_values maps criterion name -> measured value."""
        rows, num, den = [], 0.0, 0.0
        for c in self.criteria:
            v = metric_values.get(c.name, float("nan"))
            s = c.score(v)
            num += c.priority * s
            den += c.priority
            margin = (v - c.limit) if c.direction == "lower" else (c.limit - v)
            rows.append({
                "criterion": c.name,
                "structure": c.structure,
                "value": None if math.isnan(v) else round(v, 3),
                "requirement": f"{'>=' if c.direction == 'lower' else '<='} {c.limit} {c.unit}",
                "goal": c.goal,
                "passed": c.passed(v),
                "margin_to_limit": None if math.isnan(v) else round(margin, 3),
                "score": round(s, 4),
                "priority": c.priority,
            })
        return {
            "score": round(100.0 * num / den, 3) if den else 0.0,
            "n_passed": sum(r["passed"] for r in rows),
            "n_criteria": len(rows),
            "criteria": rows,
        }
