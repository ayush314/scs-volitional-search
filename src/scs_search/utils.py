"""Utility helpers for filesystem, serialization, and sampling."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from scipy.stats import qmc

from .config import REPO_ROOT

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - fallback path if tqdm is unavailable
    _tqdm = None


class _NullProgress:
    """Fallback progress object used when tqdm is unavailable."""

    def __init__(self, iterable: Iterable[Any] | None = None, **_: Any) -> None:
        self._iterable = iterable

    def __iter__(self) -> Iterator[Any]:
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def update(self, _: float = 1.0) -> None:
        return None

    def set_postfix(self, *_: Any, **__: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def repo_root() -> Path:
    """Return the repository root."""

    return REPO_ROOT


def progress(iterable: Iterable[Any] | None = None, **kwargs: Any) -> Any:
    """Return a tqdm progress bar when available, otherwise a no-op wrapper."""

    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return _NullProgress(iterable, **kwargs)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for result directories."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_dir(base_dir: str | Path, prefix: str) -> Path:
    """Create a timestamped run directory."""

    base_path = ensure_dir(base_dir)
    run_dir = base_path / f"{prefix}_{utc_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def to_serializable(value: Any) -> Any:
    """Convert dataclasses and numpy values into JSON-friendly objects."""

    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    """Write a JSON file with stable formatting."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(to_serializable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    """Write a JSON Lines file."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_serializable(row), sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSON Lines file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latin_hypercube_samples(dim: int, n_samples: int, seed: int) -> np.ndarray:
    """Generate Latin hypercube samples in [0, 1]^dim."""

    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    return sampler.random(n=n_samples)


def flatten_dict(prefix: str, mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one mapping level for CSV-friendly rows."""

    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_dict(full_key, value))
        else:
            flat[full_key] = value
    return flat


def summarise_trace(values: Sequence[float]) -> list[float]:
    """Return the best-so-far trace for plotting."""

    trace: list[float] = []
    best = -np.inf
    for value in values:
        best = max(best, float(value))
        trace.append(best)
    return trace
