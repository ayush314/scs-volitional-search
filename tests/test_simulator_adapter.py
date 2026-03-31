"""Tests for reference EMG cache reuse."""

from __future__ import annotations

import numpy as np

from scs_search.config import SimulationConfig
from scs_search.simulator_adapter import load_reference_emg_cache, resolve_reference_emg_cache


def test_load_reference_emg_cache_reads_healthy_arrays_only(tmp_path) -> None:
    np.savez(
        tmp_path / "emg_arrays.npz",
        healthy_prelesion_seed_101=np.array([1.0, 2.0]),
        lesion_no_stim_seed_101=np.array([3.0, 4.0]),
    )

    cache = load_reference_emg_cache(tmp_path, [101, 202])

    assert set(cache) == {101}
    assert np.allclose(cache[101], np.array([1.0, 2.0]))


def test_resolve_reference_emg_cache_builds_only_missing_seeds(tmp_path, monkeypatch) -> None:
    np.savez(tmp_path / "emg_arrays.npz", healthy_prelesion_seed_101=np.array([1.0, 2.0]))

    def fake_build_reference_emg_cache(seeds, _config):
        assert tuple(seeds) == (202,)
        return {202: np.array([5.0, 6.0])}

    monkeypatch.setattr(
        "scs_search.simulator_adapter.build_reference_emg_cache",
        fake_build_reference_emg_cache,
    )

    cache = resolve_reference_emg_cache([101, 202], SimulationConfig(), reference_dir=tmp_path)

    assert set(cache) == {101, 202}
    assert np.allclose(cache[101], np.array([1.0, 2.0]))
    assert np.allclose(cache[202], np.array([5.0, 6.0]))
    persisted = load_reference_emg_cache(tmp_path, [101, 202])
    assert set(persisted) == {101, 202}
