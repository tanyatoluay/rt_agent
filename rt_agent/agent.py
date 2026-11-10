"""LLM agent loop (Claude Messages API with tool use).

This is *your* agent: you control the prompt, the model, the memory policy and
the stopping rule, which is what makes the experiments reproducible.

Two memory policies (the ablation in Section II-D):
  memory="full" - one conversation for the whole run; the agent sees all past
                  observations, decisions and outcomes and may call compare_iterations.
  memory="none" - the context is reset after every optimizer call. Each episode
                  starts with only the current plan's state; history tools are disabled.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .environment import PlanningEnvironment
from .logger import dumps
from .tools import anthropic_tools

SYSTEM_PROMPT = """\
You are a radiotherapy treatment-planning assistant working in a research \
simulation (not clinical use). You tune the weights of an inverse-planning \
objective function for a lung IMRT plan (60 Gy in 30 fractions).

How the system works:
- The optimizer minimizes sum_i w_i * f_i(dose). You cannot change the \
optimizer, the constraints or the evaluation; you can only multiply the \
weights of tunable objectives by allowed factors via adjust_weight.
- Raising the weight of an organ-at-risk objective usually lowers that organ's \
dose but can reduce target coverage or raise dose elsewhere; target \
objectives trade off the other way. Effects are not exactly predictable, so \
use the numerical feedback.
- run_optimizer re-plans with the pending weights and consumes one unit of a \
limited budget. You may make a few adjust_weight calls before each run.
- Every plan is scored by a fixed clinical scorecard (0-100, higher is \
better). The best plan found at any iteration is the one that counts, so \
trying a change that turns out worse is acceptable.

Goal: maximize the scorecard score, prioritizing satisfying every criterion, \
within the optimizer budget. Base each decision on the metrics you observe. \
Before each run_optimizer call, state in one or two sentences which change \
you are making and why. When the budget is used up, or you are confident \
further changes will not help, stop and summarize the best plan.\
"""

NUDGE = ("If you are finished, reply with exactly DONE. Otherwise continue by calling the tools "
         "(remaining optimizer calls: {n}).")


def _block_to_dict(b: Any) -> dict:
    if isinstance(b, dict):
        return b
    d = b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(vars(b))
    return d


def _initial_observation(env: PlanningEnvironment, memoryless: bool) -> str:
    weights = env.get_objective_weights()
    evaluation = env.evaluate_plan()
    head = (f"Current plan: iteration {evaluation['iteration']}. "
            f"Optimizer calls remaining: {env.remaining_budget} of {env.budget}.")
    if memoryless:
        head += (" You have no memory of earlier iterations; decide using only the current "
                 "state below, make your adjustments and call run_optimizer once.")
    else:
        head += " Iteration 0 is the fixed-weight starting plan."
    return (f"{head}\n\nObjective weights:\n{json.dumps(weights, indent=1)}\n\n"
            f"Scorecard evaluation of the current plan:\n{json.dumps(evaluation, indent=1)}")


class ClaudeAgent:
    def __init__(self, client: Any, model: str, *, memory: str = "full", max_tokens: int = 2048,
                 temperature: float | None = None, max_api_calls: int = 80,
                 system_prompt: str = SYSTEM_PROMPT):
        assert memory in ("full", "none")
        self.client, self.model, self.memory = client, model, memory
        self.max_tokens, self.temperature = max_tokens, temperature
        self.max_api_calls = max_api_calls
        self.system_prompt = system_prompt
        self.usage = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.transcript: list[dict] = []

    # ------------------------------------------------------------------
    def _call(self, messages: list[dict], tools: list[dict]):
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=self.system_prompt,
                  tools=tools, messages=messages)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        resp = self.client.messages.create(**kw)
        self.usage["api_calls"] += 1
        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
        return resp

    def _episode(self, env: PlanningEnvironment, first_message: str, tools: list[dict],
                 end_after_optimizer: bool) -> str:
        """Run one conversation. Returns why it ended."""
        messages: list[dict] = [{"role": "user", "content": first_message}]
        nudged = False
        try:
            while True:
                if self.usage["api_calls"] >= self.max_api_calls:
                    return "api call limit"
                resp = self._call(messages, tools)
                content = [_block_to_dict(b) for b in resp.content]
                messages.append({"role": "assistant", "content": content})
                tool_uses = [b for b in content if b.get("type") == "tool_use"]

                if not tool_uses:
                    text = " ".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
                    if text == "DONE" or nudged or env.done:
                        return "agent stopped" if not env.done else "budget exhausted"
                    nudged = True
                    messages.append({"role": "user", "content": NUDGE.format(n=env.remaining_budget)})
                    continue

                results, optimized = [], False
                for tu in tool_uses:
                    out = env.dispatch(tu["name"], tu.get("input") or {}, caller=f"llm:{self.memory}")
                    if tu["name"] == "run_optimizer" and "error" not in out:
                        optimized = True
                    results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                    "content": dumps(out), "is_error": "error" in out})
                messages.append({"role": "user", "content": results})
                nudged = False

                if env.done:
                    return "budget exhausted"
                if end_after_optimizer and optimized:
                    return "episode complete"
        finally:
            self.transcript.append({"memory": self.memory, "messages": messages})

    # ------------------------------------------------------------------
    def run(self, env: PlanningEnvironment) -> dict:
        t0 = time.perf_counter()
        if self.memory == "full":
            env.history_visible = True
            reason = self._episode(env, _initial_observation(env, memoryless=False),
                                   anthropic_tools(include_history_tools=True), end_after_optimizer=False)
        else:
            env.history_visible = False
            tools = anthropic_tools(include_history_tools=False)
            reason = "budget exhausted"
            while not env.done:
                r = self._episode(env, _initial_observation(env, memoryless=True), tools,
                                  end_after_optimizer=True)
                if r != "episode complete":
                    reason = r
                    break
        env.log.save_json("transcript.json", self.transcript)
        return {"stop_reason": reason, "wall_time_s": round(time.perf_counter() - t0, 2), **self.usage}
