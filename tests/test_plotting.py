"""Smoke tests for stimulation-pattern and drive plotting."""

from __future__ import annotations

from scs_search.config import PhysicalModulationParameters, SimulationConfig
from scs_search.reporting.plotting import plot_pattern_detail, plot_supraspinal_drive_examples
from scs_search.simulation.drive import build_supraspinal_drive
from scs_search.stimulation.patterns import generate_stim_pattern


def test_plot_pattern_detail_writes_png_for_physical_pattern(tmp_path) -> None:
    theta = PhysicalModulationParameters(
        I0_ma=8.0,
        I1_ma=2.0,
        f0_hz=60.0,
        f1_hz=20.0,
        PW1_us=60.0,
        T_ms=250.0,
    )
    pattern = generate_stim_pattern(theta, t_end_ms=400)
    output = tmp_path / "physical_pattern.png"

    plot_pattern_detail(pattern, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_supraspinal_drive_examples_writes_png(tmp_path) -> None:
    config = SimulationConfig()
    drives = {
        101: build_supraspinal_drive(config, 101),
        202: build_supraspinal_drive(config, 202),
    }
    output = tmp_path / "supraspinal_drive.png"

    plot_supraspinal_drive_examples(drives, output)

    assert output.exists()
    assert output.stat().st_size > 0
