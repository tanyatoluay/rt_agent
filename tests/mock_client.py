"""A fake Claude client for testing the agent loop offline (no API key, no cost).

It imitates the shape of the Messages API response and follows a simple greedy
policy: read the scorecard, raise the weight of the objective tied to the
worst criterion, optimize, repeat. It is a test double, NOT a baseline.
"""
from __future__ import annotations

import itertools
import json
from types import SimpleNamespace as NS

_ids = itertools.count()


def _tu(name, inp=None):
    return {"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": name, "input": inp or {}}


def _text(t):
    return {"type": "text", "text": t}


def _latest_tool_result(messages, key):
    for m in reversed(messages):
        if isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result" and key in b["content"]:
                    return json.loads(b["content"])
    return None


class _Messages:
    def __init__(self):
        self.calls = 0

    def create(self, *, model, max_tokens, system, tools, messages, **kw):
        self.calls += 1
        usage = NS(input_tokens=1000, output_tokens=100)
        reply = lambda content, stop="tool_use": NS(content=content, stop_reason=stop, usage=usage)

        last = messages[-1]["content"]
        if isinstance(last, list):
            for b in last:
                if '"optimizer_calls_remaining": 0' in b.get("content", ""):
                    return reply([_text("DONE")], "end_turn")
                if '"changed_weights"' in b.get("content", ""):
                    return reply([_tu("evaluate_plan")])

        ev = _latest_tool_result(messages, '"criteria"')
        ws = _latest_tool_result(messages, '"objectives"')
        if ev is None or ws is None:
            return reply([_tu("get_objective_weights"), _tu("evaluate_plan")])

        worst = min(ev["criteria"], key=lambda c: c["score"])
        cands = [o for o in ws["objectives"] if o["tunable"] and o["structure"] == worst["structure"]]
        if worst["requirement"].startswith(">="):
            cands = [o for o in cands if "underdose" in o["type"]] or cands
        oid = (cands or [o for o in ws["objectives"] if o["tunable"]])[0]["objective_id"]
        return reply([_text(f"Raising {oid} because {worst['criterion']} scores lowest."),
                      _tu("adjust_weight", {"objective_id": oid, "factor": 1.5}),
                      _tu("run_optimizer")])


class MockClaudeClient:
    def __init__(self):
        self.messages = _Messages()
