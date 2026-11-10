"""Tool definitions shared by the Claude API agent and the MCP server.

One source of truth for names, descriptions and input schemas, so the agent in
your experiments and the agent in the Claude desktop app see identical tools.
"""
from __future__ import annotations

DESCRIPTIONS = {
    "get_plan_metrics": (
        "Return raw dose-volume metrics of a plan (D95/D98/D2, mean and max dose per structure, "
        "lung V20, ...). Doses in Gy, V metrics in % of structure volume. Omit `iteration` for the "
        "current plan."
    ),
    "get_objective_weights": (
        "List every optimization objective with its type, structure, current-plan weight, pending "
        "weight, bounds and whether it is tunable. Also returns the allowed adjustment factors, how "
        "many adjustments are left before the next optimization, and the remaining optimizer budget."
    ),
    "adjust_weight": (
        "Multiply the pending weight of one tunable objective by an allowed factor (e.g. 0.5, 0.75, "
        "1.5, 2.0). The weight is clipped to its bounds. Changes take effect at the next "
        "run_optimizer call. Only a limited number of adjustments is allowed per optimization."
    ),
    "run_optimizer": (
        "Re-optimize the beamlet intensities with the pending weights. Consumes one unit of the "
        "optimizer budget. Returns the new iteration number, solver status, plan score (0-100) and "
        "number of satisfied criteria. Requires at least one pending weight change."
    ),
    "evaluate_plan": (
        "Evaluate a plan against the fixed clinical scorecard: per-criterion value, requirement, "
        "pass/fail, margin to the limit and score, plus the total plan score (0-100, higher is "
        "better). Omit `iteration` for the current plan."
    ),
    "compare_iterations": (
        "Compare two iterations: which weights differ, how each dose metric changed (b minus a) and "
        "how the score changed. Use it to learn the effect of previous adjustments."
    ),
}

_ITER = {"type": "integer", "minimum": 0, "description": "Iteration number (0 = initial fixed-weight plan)."}

INPUT_SCHEMAS = {
    "get_plan_metrics": {"type": "object", "properties": {"iteration": _ITER}},
    "get_objective_weights": {"type": "object", "properties": {}},
    "adjust_weight": {
        "type": "object",
        "properties": {
            "objective_id": {"type": "string", "description": "objective_id from get_objective_weights."},
            "factor": {"type": "number", "description": "Multiplicative factor; must be one of the allowed factors."},
        },
        "required": ["objective_id", "factor"],
    },
    "run_optimizer": {"type": "object", "properties": {}},
    "evaluate_plan": {"type": "object", "properties": {"iteration": _ITER}},
    "compare_iterations": {
        "type": "object",
        "properties": {"iteration_a": _ITER, "iteration_b": _ITER},
        "required": ["iteration_a", "iteration_b"],
    },
}


def anthropic_tools(include_history_tools: bool = True) -> list[dict]:
    """Tool list in the format of the Claude Messages API."""
    names = list(DESCRIPTIONS)
    if not include_history_tools:
        names.remove("compare_iterations")
    return [{"name": n, "description": DESCRIPTIONS[n], "input_schema": INPUT_SCHEMAS[n]} for n in names]
