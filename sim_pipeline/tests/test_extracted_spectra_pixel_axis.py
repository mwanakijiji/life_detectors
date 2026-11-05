"""Tests for the detector-pixel (secondary x) axis of the extracted-spectra plot."""

# pytest fixtures are injected by argument name
# pylint: disable=redefined-outer-name,protected-access

import os

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy import units as u  # noqa: E402

from modules.core.instrumental.dispersion import (  # noqa: E402
    DispersionLaw,
    wavel_bin_edges_from_config,
)
from modules.core.instrumental.pipeline import InstrumentDepTerms  # noqa: E402
from modules.core.instrumental.wavel_solution import (  # noqa: E402
    WavelSolution,
    apply_wavel_solution,
    read_wavel_solution,
)
from modules.viz import extraction  # noqa: E402


@pytest.fixture
def config(tmp_path):
    """400x400 detector, trace from (100, 300), R = 20 over 4-18 um (30 bins)."""
    return {
        "detector": {
            "size": "400",
            "pitch_pix": "25",
            "pix_per_wavel_bin": "2.2",
            "pix_spectral_width": "3",
            "quantum_efficiency": "0.8",
            "spec_res": "20",
            "trace_start_pixel": "100, 300",
        },
        "wavelength_range": {"min": "4.0", "max": "18.0"},
        "extraction": {
            "file_name_wavel_solution_write": str(tmp_path / "wavel_solution.fits"),
        },
    }


def _expected_x_edges(n_bins):
    return 300.0 + 2.2 * np.arange(n_bins + 1)


def test_read_solution_has_bin_x_positions(config, tmp_path):
    out = str(tmp_path / "w.fits")
    DispersionLaw(config).write_wavel_solution(config, file_name=out)
    solution = read_wavel_solution(out)
    x_edges = _expected_x_edges(solution.n_bins)
    assert solution.x_lo.unit == u.pix
    np.testing.assert_allclose(solution.x_lo.value, x_edges[:-1])
    np.testing.assert_allclose(solution.x_hi.value, x_edges[1:])


def test_apply_passes_x_positions(config):
    edges = wavel_bin_edges_from_config(config)
    n_bins = edges.size - 1
    solution = WavelSolution.from_bin_edges(
        edges, source="test", x_edges=_expected_x_edges(n_bins)
    )
    entry = apply_wavel_solution(
        {"output_3_dark": {"star": np.ones((1, 1, n_bins)) * u.adu}}, solution
    )["output_3_dark"]
    np.testing.assert_allclose(entry["x_lo_pix"].value, _expected_x_edges(n_bins)[:-1])
    assert entry["x_hi_pix"].unit == u.pix


def test_pixel_transforms_map_bin_edges_and_round_trip(config):
    edges = wavel_bin_edges_from_config(config)
    n_bins = edges.size - 1
    x_edges = _expected_x_edges(n_bins)
    entry = apply_wavel_solution(
        {"output_3_dark": {"star": np.ones((1, 1, n_bins)) * u.adu}},
        WavelSolution.from_bin_edges(edges, source="test", x_edges=x_edges),
    )["output_3_dark"]
    wavel_to_pix, pix_to_wavel = extraction._wavel_pixel_transforms(entry)

    np.testing.assert_allclose(wavel_to_pix(edges.value), x_edges)
    np.testing.assert_allclose(pix_to_wavel(x_edges), edges.value)
    # agrees with the dispersion law's own wavelength-at-x
    x_mid = np.array([301.3, 320.0, 360.5])
    np.testing.assert_allclose(
        pix_to_wavel(x_mid), DispersionLaw(config).wavel_at_x(x_mid, edges)
    )
    # matplotlib may probe non-positive wavelengths: stays finite
    assert np.all(np.isfinite(wavel_to_pix(np.array([-1.0, 0.0]))))


def test_no_transforms_without_x_positions(config):
    edges = wavel_bin_edges_from_config(config)
    entry = apply_wavel_solution(
        {"output_3_dark": {"star": np.ones((1, 1, edges.size - 1)) * u.adu}},
        WavelSolution.from_bin_edges(edges, source="test"),
    )["output_3_dark"]
    assert extraction._wavel_pixel_transforms(entry) is None


def test_fallback_solution_gets_x_from_dispersion_law(config):
    instr = InstrumentDepTerms(config, sources_astroph={}, sources_to_include=[])
    instr.define_dispersion(plot=False)
    os.remove(config["extraction"]["file_name_wavel_solution_write"])
    solution = instr.load_wavel_solution()
    assert solution.source == "channel bin edges"
    np.testing.assert_allclose(
        solution.x_lo.value, _expected_x_edges(solution.n_bins)[:-1]
    )


@pytest.mark.parametrize("with_x", [True, False])
def test_plot_written_with_or_without_pixel_axis(config, tmp_path, with_x):
    edges = wavel_bin_edges_from_config(config)
    n_bins = edges.size - 1
    solution = WavelSolution.from_bin_edges(
        edges,
        source="test",
        x_edges=_expected_x_edges(n_bins) if with_x else None,
    )
    spectra = {
        name: {"star": np.ones((1, 1, n_bins)) * u.adu}
        for name in ["output_1_bright", "output_3_dark"]
    }
    out = tmp_path / f"extracted_spectra_{with_x}.png"
    extraction.plot_extracted_spectra(
        apply_wavel_solution(spectra, solution), config, str(out), dc_rates=[0.1]
    )
    assert out.exists() and out.stat().st_size > 0
