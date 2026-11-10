"""A small 2D lung-like phantom for developing the agent without PortPy.

It mimics the PortPy lung setup on purpose: same structure names, same
objective types, a 60 Gy prescription, and a linear dose model d = A x solved
with weighted objectives. Coplanar beams with Gaussian beamlets and exponential
attenuation create real trade-offs (e.g. covering the PTV next to the
esophagus vs. sparing it), so weight tuning genuinely matters.

This is NOT a dosimetrically valid model. Use it only to test code paths.
"""
from __future__ import annotations

import time
import zlib

import numpy as np
from scipy.optimize import minimize

from .base import (Objective, PlanningBackend, SolveResult, cumulative_dvh,
                   dose_at_volume, volume_at_dose)

PRESCRIPTION_GY = 60.0


class ToyBackend(PlanningBackend):
    name = "toy"

    def __init__(self, grid: int = 48, n_beams: int = 7, beamlets_per_beam: int = 24,
                 seed: int = 0, patient_id: str = "Toy_Patient_1"):
        self.patient_id = patient_id
        # Different "patients" are different seeds -> slightly different anatomy.
        rng = np.random.default_rng(seed + zlib.crc32(patient_id.encode()) % 1000)
        self._build_phantom(grid, rng)
        self._build_influence(grid, n_beams, beamlets_per_beam)
        self.n_beams, self.bpb = n_beams, beamlets_per_beam

    # ---- phantom ---------------------------------------------------------------
    def _build_phantom(self, n: int, rng: np.random.Generator) -> None:
        yy, xx = np.mgrid[0:n, 0:n].astype(float)
        c = n / 2
        jitter = lambda s: rng.uniform(-s, s)
        body = ((xx - c) / (0.47 * n)) ** 2 + ((yy - c) / (0.38 * n)) ** 2 <= 1
        lung_l = (((xx - c + 0.2 * n) / (0.15 * n)) ** 2 + ((yy - c + 0.02 * n) / (0.25 * n)) ** 2) <= 1
        lung_r = (((xx - c - 0.2 * n) / (0.15 * n)) ** 2 + ((yy - c + 0.02 * n) / (0.25 * n)) ** 2) <= 1
        ptv_c = (c - 0.17 * n + jitter(1.5), c - 0.02 * n + jitter(1.5))
        ptv = ((xx - ptv_c[0]) ** 2 + (yy - ptv_c[1]) ** 2) <= (0.085 * n) ** 2
        esoph = ((xx - (c - 0.03 * n + jitter(0.8))) ** 2 + (yy - (c + 0.06 * n)) ** 2) <= (0.035 * n) ** 2
        cord = ((xx - c) ** 2 + (yy - (c + 0.24 * n)) ** 2) <= (0.035 * n) ** 2
        heart = (((xx - (c + 0.03 * n)) / (0.11 * n)) ** 2 + ((yy - (c - 0.12 * n)) / (0.09 * n)) ** 2) <= 1
        lungs = (lung_l | lung_r) & ~ptv
        self.masks = {
            "PTV": ptv & body,
            "ESOPHAGUS": esoph & body & ~ptv,
            "CORD": cord & body,
            "HEART": heart & body & ~ptv & ~esoph,
            "LUNGS_NOT_GTV": lungs & body & ~heart & ~esoph,
        }
        self.body = body
        self.vox_idx = {k: np.flatnonzero(m.ravel()) for k, m in self.masks.items()}
        self.ptv_center = ptv_c

    def _build_influence(self, n: int, n_beams: int, bpb: int) -> None:
        yy, xx = np.mgrid[0:n, 0:n].astype(float)
        cx, cy = self.ptv_center
        # beam angles concentrated on the tumor side, like a lung IMRT arrangement
        angles = np.deg2rad(np.linspace(150, 390, n_beams, endpoint=False))
        cols = []
        body = self.body.ravel()
        for a in angles:
            dx, dy = np.cos(a), np.sin(a)          # beam direction
            px, py = -dy, dx                      # lateral axis
            depth_from_iso = (xx - cx) * dx + (yy - cy) * dy
            lateral = (xx - cx) * px + (yy - cy) * py
            # crude depth: distance from where the ray enters the body
            depth = depth_from_iso - depth_from_iso[self.body].min()
            atten = np.exp(-0.035 * depth)
            for b in range(bpb):
                off = (b - (bpb - 1) / 2) * 0.9
                prof = np.exp(-0.5 * ((lateral - off) / 0.9) ** 2)
                col = (atten * prof).ravel() * body
                cols.append(col)
        A = np.stack(cols, axis=1)
        keep = np.unique(np.concatenate(list(self.vox_idx.values())))
        self._row_of = {v: i for i, v in enumerate(keep)}
        self.A = A[keep]
        self.struct_rows = {k: np.array([self._row_of[v] for v in idx]) for k, idx in self.vox_idx.items()}
        # scale so a uniform intensity of 1 gives roughly the prescription on the PTV
        self.A *= PRESCRIPTION_GY / (self.A[self.struct_rows["PTV"]].sum(1).mean())

    # ---- interface ---------------------------------------------------------------
    def structure_names(self) -> list[str]:
        return list(self.masks)

    def objectives(self) -> list[Objective]:
        # Same types and magnitudes as PortPy's Lung_2Gy_30Fx opt params.
        return [
            Objective("obj0_quadratic-overdose_PTV", "quadratic-overdose", "PTV", 10000, True, 100, 1e6,
                      {"dose_gy": PRESCRIPTION_GY}),
            Objective("obj1_quadratic-underdose_PTV", "quadratic-underdose", "PTV", 100000, True, 1000, 1e7,
                      {"dose_gy": PRESCRIPTION_GY}),
            Objective("obj2_quadratic_CORD", "quadratic", "CORD", 10, True, 0.1, 1e4),
            Objective("obj3_quadratic_ESOPHAGUS", "quadratic", "ESOPHAGUS", 20, True, 0.1, 1e4),
            Objective("obj4_quadratic_HEART", "quadratic", "HEART", 20, True, 0.1, 1e4),
            Objective("obj5_quadratic_LUNGS_NOT_GTV", "quadratic", "LUNGS_NOT_GTV", 10, True, 0.1, 1e4),
            Objective("obj6_smoothness-quadratic", "smoothness-quadratic", "", 1000, False),
        ]

    def solve(self, weights: dict[str, float]) -> SolveResult:
        t0 = time.perf_counter()
        objs = {o.id: o for o in self.objectives()}
        A = self.A
        rows = self.struct_rows
        nB, bpb = self.A.shape[1], self.bpb

        # smoothness: differences between neighbouring beamlets of the same beam
        D = np.zeros((nB - self.n_beams, nB))
        r = 0
        for beam in range(self.n_beams):
            for j in range(bpb - 1):
                D[r, beam * bpb + j], D[r, beam * bpb + j + 1] = -1, 1
                r += 1

        def f(x):
            d = A @ x
            val, grad_d = 0.0, np.zeros_like(d)
            for oid, o in objs.items():
                w = weights.get(oid, o.weight)
                if o.type == "smoothness-quadratic":
                    continue
                idx = rows[o.structure]
                n = len(idx)
                if o.type == "quadratic-overdose":
                    e = np.maximum(d[idx] - o.extra["dose_gy"], 0)
                elif o.type == "quadratic-underdose":
                    e = np.minimum(d[idx] - o.extra["dose_gy"], 0)
                else:  # quadratic
                    e = d[idx]
                val += w * (e @ e) / n
                np.add.at(grad_d, idx, 2 * w * e / n)
            g = A.T @ grad_d
            ws = weights.get("obj6_smoothness-quadratic", objs["obj6_smoothness-quadratic"].weight)
            Dx = D @ x
            val += ws * (Dx @ Dx) / len(Dx)
            g += 2 * ws * (D.T @ Dx) / len(Dx)
            return val, g

        x0 = np.ones(nB)
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * nB,
                       options={"maxiter": 2000, "ftol": 1e-12})
        status = "optimal" if res.success else "suboptimal"
        return SolveResult(status, time.perf_counter() - t0, handle=A @ res.x,
                           message="" if res.success else str(res.message))

    def _dose(self, result: SolveResult, structure: str) -> np.ndarray:
        return result.handle[self.struct_rows[structure]]

    def metric(self, result, kind, structure, param=None) -> float:
        d = self._dose(result, structure)
        if kind == "D":
            return dose_at_volume(d, param)
        if kind == "V":
            return volume_at_dose(d, param)
        if kind == "mean":
            return float(d.mean())
        if kind == "max":
            return float(d.max())
        raise ValueError(f"unknown metric kind {kind}")

    def dvh(self, result, structure, n_points=200):
        return cumulative_dvh(self._dose(result, structure), n_points=n_points,
                              max_dose=PRESCRIPTION_GY * 1.3)
