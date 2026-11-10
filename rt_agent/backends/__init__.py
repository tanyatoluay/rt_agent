from .base import Objective, PlanningBackend, SolveResult


def make_backend(kind: str, patient_id: str, cfg: dict) -> PlanningBackend:
    """Create a backend from the experiment config."""
    if kind == "toy":
        from .toy_backend import ToyBackend
        return ToyBackend(patient_id=patient_id, **cfg.get("toy", {}))
    if kind == "portpy":
        from .portpy_backend import PortPyBackend
        return PortPyBackend(patient_id=patient_id, **cfg.get("portpy", {}))
    raise ValueError(f"unknown backend '{kind}' (use 'toy' or 'portpy')")


__all__ = ["Objective", "PlanningBackend", "SolveResult", "make_backend"]
