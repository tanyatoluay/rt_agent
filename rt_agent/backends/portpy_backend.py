"""PortPy backend: lung IMRT with the Lung_2Gy_30Fx protocol.

Follows the structure of PortPy's basic tutorial (examples/1_basic_tutorial.py):
DataExplorer -> CT / Structures / Beams -> ClinicalCriteria + opt params ->
InfluenceMatrix -> Plan -> Optimization -> solve.

The agent only changes the `weight` field of selected entries in
opt_params["objective_functions"]; constraints and everything else stay fixed.

NOTE: this file was written against the PortPy API as documented in its public
tutorials. PortPy is still evolving, so run `python scripts/validate_environment.py
--backend portpy` first: it exercises every call used here and will tell you
quickly if a name changed in your installed version.
"""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from .base import Objective, PlanningBackend, SolveResult, cumulative_dvh


def _objective_id(i: int, obj: dict) -> str:
    struct = obj.get("structure_name", "")
    return f"obj{i}_{obj['type']}" + (f"_{struct}" if struct else "")


class PortPyBackend(PlanningBackend):
    name = "portpy"

    def __init__(self, patient_id: str = "Lung_Patient_3", protocol: str = "Lung_2Gy_30Fx",
                 data_dir: str | None = None, hf_repo_id: str | None = None,
                 solver: str = "MOSEK", solver_kwargs: dict | None = None,
                 tunable_types: list[str] | None = None,
                 weight_bounds_factor: float = 100.0, verbose: bool = False,
                 opt_vox_xyz_res_mm: list[float] | None = None,
                 beamlet_width_mm: float | None = None, beamlet_height_mm: float | None = None):
        """opt_vox_xyz_res_mm / beamlet_*_mm: optional down-sampling of the influence matrix
        (PortPy's native grid is 2.93 x 2.93 x 3 mm dose voxels, 2.5 mm beamlets). Coarser
        grids make each solve much faster; optimization and evaluation both use the coarse
        grid, identically for every method."""
        import portpy.photon as pp  # imported lazily so the toy backend works without PortPy

        # PortPy 1.2.0 renamed Structures.get_ct_res_xyz_mm -> get_ct_voxel_resolution_xyz_mm,
        # but its own down-sampling code still calls the old name. Restore it as an alias.
        if not hasattr(pp.Structures, "get_ct_res_xyz_mm") and hasattr(pp.Structures, "get_ct_voxel_resolution_xyz_mm"):
            pp.Structures.get_ct_res_xyz_mm = pp.Structures.get_ct_voxel_resolution_xyz_mm

        self.pp = pp
        self.patient_id, self.protocol = patient_id, protocol
        self.solver, self.solver_kwargs = solver, solver_kwargs or {}
        self.verbose = verbose

        from pathlib import Path
        # PortPy's Hugging Face download writes to <data_dir>/data/<patient>; a manually
        # prepared folder may be <data_dir>/<patient>. Use whichever exists, else download.
        local_root = None
        if data_dir:
            for root in (Path(data_dir) / "data", Path(data_dir)):
                if (root / patient_id / "StructureSet_MetaData.json").exists():
                    local_root = root
                    break
        if local_root is not None or not hf_repo_id:
            data = pp.DataExplorer(data_dir=str(local_root or data_dir))
            data.patient_id = patient_id
        else:
            data = pp.DataExplorer(hf_repo_id=hf_repo_id, local_download_dir=data_dir)
            data.patient_id = patient_id
            data.filter_and_download_hf_dataset()
        self.data = data

        self.ct = pp.CT(data)
        self.structs = pp.Structures(data)
        self.beams = pp.Beams(data)  # planner-selected beams shipped with the dataset

        self.clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol)
        self.base_opt_params = data.load_config_opt_params(protocol_name=protocol)
        self.structs.create_opt_structures(opt_params=self.base_opt_params,
                                           clinical_criteria=self.clinical_criteria)
        self.inf_matrix = pp.InfluenceMatrix(ct=self.ct, structs=self.structs, beams=self.beams)
        if opt_vox_xyz_res_mm or beamlet_width_mm or beamlet_height_mm:
            # Down-sample the full matrix afterwards: in PortPy 1.2.0 passing these to the
            # InfluenceMatrix constructor crashes. Beamlet down-sampling is also broken there
            # with NumPy 2, so in practice use opt_vox_xyz_res_mm only.
            kw = {"opt_vox_xyz_res_mm": opt_vox_xyz_res_mm}
            if beamlet_width_mm or beamlet_height_mm:
                kw.update(beamlet_width_mm=beamlet_width_mm, beamlet_height_mm=beamlet_height_mm)
            self.inf_matrix = self.inf_matrix.create_down_sample(**kw)
        self.plan = pp.Plan(ct=self.ct, structs=self.structs, beams=self.beams,
                            inf_matrix=self.inf_matrix, clinical_criteria=self.clinical_criteria)

        # Objectives the agent may tune. By default: every dose objective,
        # but not the smoothness regularizer.
        tunable_types = tunable_types or ["quadratic-overdose", "quadratic-underdose", "quadratic",
                                          "linear-overdose", "linear-underdose"]
        self._objectives: list[Objective] = []
        for i, o in enumerate(self.base_opt_params["objective_functions"]):
            w = float(o["weight"])
            self._objectives.append(Objective(
                id=_objective_id(i, o), type=o["type"], structure=o.get("structure_name", ""),
                weight=w, tunable=o["type"] in tunable_types,
                min_weight=w / weight_bounds_factor, max_weight=w * weight_bounds_factor,
                extra={"index": i},
            ))

    # ---- interface ---------------------------------------------------------------
    def structure_names(self) -> list[str]:
        return list(self.structs.get_structures())

    def objectives(self) -> list[Objective]:
        return copy.deepcopy(self._objectives)

    def solve(self, weights: dict[str, float]) -> SolveResult:
        pp = self.pp
        opt_params = copy.deepcopy(self.base_opt_params)
        for o in self._objectives:
            opt_params["objective_functions"][o.extra["index"]]["weight"] = float(weights.get(o.id, o.weight))
        t0 = time.perf_counter()
        try:
            opt = pp.Optimization(self.plan, opt_params=opt_params, clinical_criteria=self.clinical_criteria)
            opt.create_cvxpy_problem()
            sol = opt.solve(solver=self.solver, verbose=self.verbose, **self.solver_kwargs)
            status = "optimal"
            if sol is None or sol.get("optimal_intensity") is None:
                status = "failed"
            else:
                # PortPy's solve() does not return the dose, but Evaluation.get_dvh reads
                # sol["dose_1d"] (total dose in Gy over all fractions), so add it here once.
                sol["dose_1d"] = self._dose_1d(sol)
            return SolveResult(status, time.perf_counter() - t0, handle=sol)
        except Exception as exc:  # solver errors are reported to the agent, not raised
            return SolveResult("failed", time.perf_counter() - t0, handle=None, message=repr(exc))

    def _dose_1d(self, sol: dict) -> np.ndarray:
        if sol.get("dose_1d") is not None:
            return sol["dose_1d"]
        inf = sol["inf_matrix"]
        return inf.A @ (sol["optimal_intensity"] * self.plan.get_num_of_fractions())

    def metric(self, result: SolveResult, kind: str, structure: str, param: float | None = None) -> float:
        E = self.pp.Evaluation
        sol = result.handle
        if kind == "D":
            return float(E.get_dose(sol, struct=structure, volume_per=param))
        if kind == "V":
            return float(E.get_volume(sol, struct=structure, dose_value_gy=param))
        if kind == "mean":
            return float(E.get_mean_dose(sol, struct=structure))
        if kind == "max":
            return float(E.get_max_dose(sol, struct=structure))
        raise ValueError(f"unknown metric kind {kind}")

    def dvh(self, result: SolveResult, structure: str, n_points: int = 200):
        try:
            sol = result.handle
            dose = self._dose_1d(sol)
            idx = sol["inf_matrix"].get_opt_voxels_idx(structure)
            return cumulative_dvh(dose[idx], n_points=n_points)
        except Exception:
            return None

    def describe(self) -> dict[str, Any]:
        """Used by validate_environment.py to print what was loaded."""
        return {
            "patient": self.patient_id,
            "protocol": self.protocol,
            "structures": self.structure_names(),
            "n_beams": len(self.beams.get_all_beam_ids()) if hasattr(self.beams, "get_all_beam_ids") else "?",
            "influence_matrix_shape": tuple(self.inf_matrix.A.shape),
            "fractions": self.plan.get_num_of_fractions(),
            "objectives": [(o.id, o.weight, o.tunable) for o in self._objectives],
            "constraints": self.base_opt_params.get("constraints", []),
        }
