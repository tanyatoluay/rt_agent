"""MCP server exposing the planning environment to the Claude desktop app
(or any MCP client, e.g. Claude Code).

Same six tools as the experiment agent, plus two demo-only helpers
(reset_session, get_session_summary). Budget, validation and logging are
enforced here, so the app gets the same guardrails as the experiments.

Run directly for a quick check:   python -m rt_agent.mcp_server
Config via environment variables:
  RT_AGENT_CONFIG   path to experiment JSON (default config/experiment.json)
  RT_AGENT_PATIENT  patient id (default: first patient in the config)
  RT_AGENT_LOG_DIR  where to write logs (default results/mcp_sessions/<timestamp>)

IMPORTANT: stdio transport -> never print() to stdout in this process.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .config import PROJECT_ROOT, build_backend, build_env, load_config
from .tools import DESCRIPTIONS

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[rt-planner] %(message)s")
log = logging.getLogger("rt-planner")

INSTRUCTIONS = (
    "Research-only radiotherapy planning environment (PortPy lung IMRT). You can tune the weights of "
    "inverse-planning objectives and re-optimize, within a fixed optimizer budget. Plans are scored by "
    "a fixed scorecard (0-100). Start with get_objective_weights and evaluate_plan. Not for clinical use."
)

server = _Server("rt-planner", instructions=INSTRUCTIONS)

_lock = threading.Lock()
_state: dict = {}


def _env():
    """Lazily build the environment on first use (PortPy loading can be slow)."""
    with _lock:
        if "env" not in _state:
            cfg = load_config(os.environ.get("RT_AGENT_CONFIG", "config/experiment.json"))
            patient = os.environ.get("RT_AGENT_PATIENT", cfg["patients"][0])
            out = os.environ.get("RT_AGENT_LOG_DIR") or \
                PROJECT_ROOT / "results" / "mcp_sessions" / time.strftime("%Y%m%d-%H%M%S")
            log.info("loading backend=%s patient=%s", cfg["backend"], patient)
            backend = build_backend(cfg, patient)
            _state.update(cfg=cfg, patient=patient, backend=backend, out=Path(out))
            _state["env"] = build_env(cfg, backend, out_dir=out)
            log.info("ready; logs in %s", out)
        return _state["env"]


def _call(name: str, **args) -> dict:
    args = {k: v for k, v in args.items() if v is not None}
    return _env().dispatch(name, args, caller="mcp")


@server.tool(description=DESCRIPTIONS["get_plan_metrics"])
def get_plan_metrics(iteration: int | None = None) -> dict:
    return _call("get_plan_metrics", iteration=iteration)


@server.tool(description=DESCRIPTIONS["get_objective_weights"])
def get_objective_weights() -> dict:
    return _call("get_objective_weights")


@server.tool(description=DESCRIPTIONS["adjust_weight"])
def adjust_weight(objective_id: str, factor: float) -> dict:
    return _call("adjust_weight", objective_id=objective_id, factor=factor)


@server.tool(description=DESCRIPTIONS["run_optimizer"])
def run_optimizer() -> dict:
    return _call("run_optimizer")


@server.tool(description=DESCRIPTIONS["evaluate_plan"])
def evaluate_plan(iteration: int | None = None) -> dict:
    return _call("evaluate_plan", iteration=iteration)


@server.tool(description=DESCRIPTIONS["compare_iterations"])
def compare_iterations(iteration_a: int, iteration_b: int) -> dict:
    return _call("compare_iterations", iteration_a=iteration_a, iteration_b=iteration_b)


@server.tool(description="Demo helper: summary of this session (initial vs best score, score trajectory, "
                         "best weights, optimizer calls used).")
def get_session_summary() -> dict:
    env = _env()
    return {"patient": _state["patient"], **env.summary()}


@server.tool(description="Demo helper: discard all iterations and restart from the fixed-weight plan "
                         "with a fresh optimizer budget.")
def reset_session() -> dict:
    env = _env()
    env.reset()
    return {"status": "reset", "optimizer_calls_remaining": env.remaining_budget,
            "score": env.iterations[0]["evaluation"]["score"]}


def main() -> None:
    server.run()  # stdio transport


if __name__ == "__main__":
    main()
