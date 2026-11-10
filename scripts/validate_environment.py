"""Preliminary validation of the optimization environment (paper Section III).

  python scripts/validate_environment.py                        # toy backend
  python scripts/validate_environment.py --config config/portpy_lung.json

Checks, in order:
  1. the backend loads the patient and protocol
  2. every scorecard structure exists
  3. the fixed-weight plan solves, and every metric can be computed
  4. changing one weight and re-solving changes the dose in the expected direction
Run this before connecting any agent. On PortPy it is also the place where an
API difference in your installed version will show up first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rt_agent.config import build_backend, build_env, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.json")
    ap.add_argument("--patient", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    patient = args.patient or cfg["patients"][0]
    ok = True

    print(f"[1] Loading backend '{cfg['backend']}' for {patient} ...")
    backend = build_backend(cfg, patient)
    if hasattr(backend, "describe"):
        print(json.dumps(backend.describe(), indent=2, default=str))
    structures = {s.upper() for s in backend.structure_names()}
    print("    structures:", sorted(structures))
    print("    objectives:")
    for o in backend.objectives():
        print(f"      {o.id:45s} weight={o.weight:<10g} tunable={o.tunable}")
    if hasattr(backend, "clinical_criteria"):
        try:
            print("    PortPy clinical criteria for this protocol (align your scorecard with these):")
            print(json.dumps(backend.clinical_criteria.clinical_criteria_dict, indent=2, default=str)[:4000])
        except Exception as e:
            print("    (could not print clinical criteria:", e, ")")

    print("[2] Scorecard structures ...")
    env = None
    from rt_agent.scorecard import Scorecard
    sc = Scorecard.from_json(cfg["scorecard"])
    for c in sc.criteria:
        found = c.structure.upper() in structures
        ok &= found
        print(f"    {c.name:18s} -> {c.structure:15s} {'OK' if found else 'MISSING: fix the scorecard structure name'}")

    print("[3] Solving the fixed-weight plan ...")
    env = build_env(cfg, backend)
    it0 = env.iterations[0]
    print(f"    status={it0['status']}  solve time={it0['solve_time_s']} s")
    if it0["status"] == "failed":
        print("    solver message:", it0["message"])
        sys.exit("FAILED: the initial plan did not solve. Check the solver / license.")
    for k, v in it0["metrics"].items():
        flag = "" if v is not None else "   <-- could not compute"
        ok &= v is not None
        print(f"    {k:22s} {v}{flag}")
    ev = it0["evaluation"]
    print(f"    score={ev['score']}  passed {ev['n_passed']}/{ev['n_criteria']}")
    if it0["solve_time_s"] > 60:
        print("    NOTE: solves take over a minute. For the Claude desktop app demo, long tool calls may time out;"
              " consider a smaller case or a faster solver setting.")

    print("[4] Sensitivity check: doubling one OAR weight should lower that OAR's dose ...")
    oar = next((o for o in env.objectives.values()
                if o.tunable and o.type == "quadratic" and o.structure.upper() not in ("PTV", "GTV", "CTV")), None)
    if oar is None:
        print("    no tunable OAR objective found; skipped")
    else:
        mean_key = f"{oar.structure}_mean"
        before = it0["metrics"].get(mean_key)
        env.dispatch("adjust_weight", {"objective_id": oar.id, "factor": max(env.allowed_factors)})
        r = env.dispatch("run_optimizer")
        after = env.iterations[-1]["metrics"].get(mean_key)
        print(f"    {oar.id}: {mean_key} {before} -> {after} Gy, score {ev['score']} -> {r.get('score')}")
        if before is not None and after is not None and after > before + 1e-6:
            print("    WARNING: dose went up; check objective/structure mapping.")
            ok = False

    print("\nRESULT:", "environment OK" if ok else "problems found (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
