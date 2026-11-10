"""Append-only JSONL logging of every interaction (for reproducibility/analysis)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return str(o)


def dumps(obj: Any, **kw) -> str:
    return json.dumps(obj, default=_default, **kw)


class RunLogger:
    """Writes events.jsonl (every tool call) and iterations.jsonl (every plan)."""

    def __init__(self, out_dir: str | Path | None):
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def _write(self, fname: str, record: dict) -> None:
        if not self.out_dir:
            return
        record = {"t": round(time.time() - self.t0, 3), **record}
        with open(self.out_dir / fname, "a") as f:
            f.write(dumps(record) + "\n")

    def event(self, **record) -> None:
        self._write("events.jsonl", record)

    def iteration(self, **record) -> None:
        self._write("iterations.jsonl", record)

    def save_json(self, fname: str, obj: Any) -> None:
        if self.out_dir:
            (self.out_dir / fname).write_text(dumps(obj, indent=2))
