"""Run all methods on all patients under identical conditions.

Examples
  python scripts/run_experiment.py                                  # toy backend, everything
  python scripts/run_experiment.py --methods fixed,rule,random      # no API calls
  python scripts/run_experiment.py --mock                           # LLM loop with a fake client
  python scripts/run_experiment.py --config config/portpy_lung.json --out results/portpy_v1

Output (in --out):
  runs.csv          one row per (patient, method, repeat): best score, criteria passed, cost...
  iterations.csv    one row per plan: score trajectory, for the convergence plots
  <patient>/<method>/rep<k>/   events.jsonl, iterations.jsonl, summary.json, transcript.json

Finished runs are skipped when you re-run (their summary.json exists), so a
crash or an API outage never costs you completed LLM runs. Use --overwrite to redo.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rt_agent.agent import ClaudeAgent  # noqa: E402
from rt_agent.baselines import BASELINES  # noqa: E402
from rt_agent.config import build_backend, build_env, load_config  # noqa: E402

ALL_METHODS = ["fixed", "rule", "random", "llm", "llm_no_history"]
DETERMINISTIC = {"fixed", "rule"}


def make_client(mock: bool):
    if mock:
        sys.path.insert(0, str(ROOT / "tests"))
        from mock_client import MockClaudeClient
        return MockClaudeClient()
    import anthropic  # reads ANTHROPIC_API_KEY from the environment
    return anthropic.Anthropic(max_retries=6)


def run_one(method, cfg, backend, cache, out_dir, rng, client):
    env = build_env(cfg, backend, history_visible=(method != "llm_no_history"),
                    out_dir=out_dir, solve_cache=cache)
    t0 = time.perf_counter()
    if method in BASELINES:
        info = BASELINES[method](env, rng=rng)
    else:
        llm = cfg["llm"]
        agent = ClaudeAgent(client, llm["model"], memory="full" if method == "llm" else "none",
                            max_tokens=llm.get("max_tokens", 2048), temperature=llm.get("temperature"),
                            max_api_calls=llm.get("max_api_calls", 80))
        info = agent.run(env)
    info["wall_time_s"] = round(time.perf_counter() - t0, 2)
    return env, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.json")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--patients", default=None, help="comma-separated; default from config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mock", action="store_true", help="use the offline mock client for LLM methods")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-solve identical weight sets instead of reusing them (honest wall-clock times)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            sys.exit(f"unknown method {m}; choose from {ALL_METHODS}")
    patients = args.patients.split(",") if args.patients else cfg["patients"]
    out = Path(args.out or ROOT / "results" / f"{cfg['backend']}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    if args.mock:
        (out / "MOCK_LLM_RESULTS_NOT_REAL").write_text("LLM rows in this folder come from the mock client.\n")

    client = make_client(args.mock) if any(m.startswith("llm") for m in methods) else None

    for patient in patients:
        print(f"\n=== {patient} ===", flush=True)
        backend = build_backend(cfg, patient)
        cache: dict = {}   # identical weights -> identical plan, shared by all methods of this patient
        for method in methods:
            n_rep = 1 if method in DETERMINISTIC else int(cfg.get("repeats", {}).get(method, 1))
            for rep in range(n_rep):
                rdir = out / patient / method / f"rep{rep}"
                if (rdir / "summary.json").exists() and not args.overwrite:
                    print(f"  {method:15s} rep{rep}: done already, skipping")
                    continue
                if rdir.exists():
                    for f in rdir.glob("*.json*"):
                        f.unlink()
                rng = np.random.default_rng([cfg.get("seed", 0), zlib.crc32(patient.encode()), rep])
                env, info = run_one(method, cfg, backend, {} if args.no_cache else cache, rdir, rng, client)
                summary = {"patient": patient, "method": method, "repeat": rep, **env.summary(), **info}
                (rdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
                print(f"  {method:15s} rep{rep}: score {summary['initial_score']:.2f} -> "
                      f"{summary['best_score']:.2f}  passed {summary['best_n_passed']}  "
                      f"calls {summary['optimizer_calls_used']}  ({info.get('stop_reason')})", flush=True)

    collect(out)
    print(f"\nResults in {out}\nNext: python scripts/analyze.py {out}")


def collect(out: Path) -> None:
    """Rebuild runs.csv and iterations.csv from every summary.json / iterations.jsonl."""
    runs, iters = [], []
    for sfile in sorted(out.glob("*/*/rep*/summary.json")):
        s = json.loads(sfile.read_text())
        row = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
        for k, v in (s.get("best_metrics") or {}).items():
            row[f"best_{k}"] = v
        runs.append(row)
        ifile = sfile.parent / "iterations.jsonl"
        if ifile.exists():
            best = -1e9
            for line in ifile.read_text().splitlines():
                it = json.loads(line)
                sc = it["evaluation"]["score"]
                best = max(best, sc)
                iters.append({"patient": s["patient"], "method": s["method"], "repeat": s["repeat"],
                              "iteration": it["iteration"], "calls_used": it["calls_used"],
                              "status": it["status"], "score": sc, "best_so_far": best,
                              "n_passed": it["evaluation"]["n_passed"]})
    for name, rows in (("runs.csv", runs), ("iterations.csv", iters)):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
