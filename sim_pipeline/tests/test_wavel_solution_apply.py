"""Tests for reading the wavelength solution and applying it to extracted spectra."""

# pytest fixtures are injected by argument name
# pylint: disable=redefined-outer-name

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
    wavel_solution_file_from_config,
)
from modules.viz.extraction import plot_extracted_spectra  # noqa: E402


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
            "gain": "2.0",
            "read_noise": "6.0",
            "dark_current": "0.1",
        },
        "wavelength_range": {"min": "4.0", "max": "18.0"},
        "observation": {"t_int_frame": "10", "N_int_per_angle": "2"},
        "extraction": {
            "method": "box_sum",
            "file_name_wavel_solution_write": str(tmp_path / "wavel_solution.fits"),
        },
        "dirs": {"save_s2n_data_unique_dir": str(tmp_path) + "/"},
    }


def _fake_spectra(n_bins, n_dc=1, n_read=2):
    return {
        "output_3_dark": {
            "star": np.ones((n_dc, n_read, n_bins)) * u.adu,
            "instrum_systematics": np.zeros((n_dc, n_read, n_bins)) * u.adu,
        }
    }


def test_read_matches_written_solution(config, tmp_path):
    out = str(tmp_path / "w.fits")
    DispersionLaw(config).write_wavel_solution(config, file_name=out)
    solution = read_wavel_solution(out)
    edges = wavel_bin_edges_from_config(config)
    assert solution.n_bins == edges.size - 1
    assert u.allclose(solution.wavel_lo, edges[:-1])
    assert u.allclose(solution.wavel_hi, edges[1:])
    assert u.allclose(solution.wavel_cen, np.sqrt(edges[:-1] * edges[1:]))
    assert u.allclose(solution.width, edges[1:] - edges[:-1])
    assert solution.source == out


def test_file_from_config_prefers_read_key(config):
    assert wavel_solution_file_from_config(config).endswith("wavel_solution.fits")
    config["extraction"]["file_name_wavel_solution_read"] = "/other/solution.fits"
    assert wavel_solution_file_from_config(config) == "/other/solution.fits"
    assert wavel_solution_file_from_config({"extraction": {}}) is None
    assert wavel_solution_file_from_config({}) is None


def test_apply_attaches_wavelength_and_per_um(config):
    solution = WavelSolution.from_bin_edges(
        wavel_bin_edges_from_config(config), source="test"
    )
    spectra = _fake_spectra(solution.n_bins)
    errors = {"output_3_dark": {"error": np.full((1, 2, solution.n_bins), 0.5) * u.adu}}
    out = apply_wavel_solution(spectra, solution, errors)

    entry = out["output_3_dark"]
    assert u.allclose(entry["wavel"], solution.wavel_cen)
    assert entry["spectra_adu"] is spectra["output_3_dark"]
    per_um = entry["spectra_adu_per_um"]["star"]
    assert per_um.unit == u.adu / u.um
    assert u.allclose(per_um[0, 0], 1.0 * u.adu / solution.width)
    assert u.allclose(entry["error_adu_per_um"][0, 1], 0.5 * u.adu / solution.width)


def test_apply_without_errors_gives_none(config):
    solution = WavelSolution.from_bin_edges(
        wavel_bin_edges_from_config(config), source="test"
    )
    entry = apply_wavel_solution(_fake_spectra(solution.n_bins), solution)[
        "output_3_dark"
    ]
    assert entry["error_adu"] is None and entry["error_adu_per_um"] is None


def test_apply_rejects_bin_count_mismatch(config):
    solution = WavelSolution.from_bin_edges(
        wavel_bin_edges_from_config(config), source="test"
    )
    with pytest.raises(ValueError, match="wavelength bins"):
        apply_wavel_solution(_fake_spectra(solution.n_bins - 1), solution)


def test_load_wavel_solution_reads_file_once(config):
    instr = InstrumentDepTerms(config, sources_astroph={}, sources_to_include=[])
    instr.define_dispersion(plot=False)  # writes the file
    solution = instr.load_wavel_solution()
    assert solution.source == config["extraction"]["file_name_wavel_solution_write"]
    assert instr.load_wavel_solution() is solution  # cached
    ref = instr.output_channels["output_1_bright"]
    assert u.allclose(solution.wavel_cen, ref.bin_centers)


def test_load_wavel_solution_falls_back_to_channel_bins(config, tmp_path, caplog):
    config["extraction"]["file_name_wavel_solution_write"] = str(
        tmp_path / "missing.fits"
    )
    instr = InstrumentDepTerms(config, sources_astroph={}, sources_to_include=[])
    with caplog.at_level("WARNING"):
        solution = instr.load_wavel_solution()
    assert "not found" in caplog.text
    assert solution.source == "channel bin edges"
    ref = instr.output_channels["output_1_bright"]
    assert u.allclose(solution.wavel_cen, ref.bin_centers)


def test_extraction_puts_spectra_on_wavelength_solution(config, tmp_path):
    instr = InstrumentDepTerms(config, sources_astroph={}, sources_to_include=[])
    instr.define_dispersion(plot=False)
    n_bins = len(instr.output_channels["output_1_bright"].bin_centers)
    for channel in instr.output_channels.values():
        channel.astroph_signal["star"] = {
            "flux_astro_1d_interpolated_ph_sec_pixel": np.linspace(1.0, 2.0, n_bins)
            * u.ph
            / (u.s * u.pix)
        }
    instr.expose_plus_detector_effects()
    instr.write_detector_readouts(str(tmp_path / "readouts.fits"))
    instr.extract_spectra_from_readouts(plot=False)

    assert set(instr.extracted_spectra_wavel) == set(instr.output_channels)
    entry = instr.extracted_spectra_wavel["output_3_dark"]
    assert u.allclose(entry["wavel"], instr.wavel_solution.wavel_cen)
    assert entry["spectra_adu"]["star"].shape == (1, 2, n_bins)
    assert u.allclose(
        entry["spectra_adu_per_um"]["star"],
        entry["spectra_adu"]["star"] / instr.wavel_solution.width,
    )


def test_plot_extracted_spectra_writes_png(config, tmp_path):
    solution = WavelSolution.from_bin_edges(
        wavel_bin_edges_from_config(config), source="test"
    )
    spectra = {
        name: {
            "star": np.ones((1, 2, solution.n_bins)) * u.adu,
            "instrum_systematics": np.zeros((1, 2, solution.n_bins)) * u.adu,
        }
        for name in ["output_1_bright", "output_2_bright", "output_3_dark"]
    }
    errors = {
        "output_3_dark": {"error": np.full((1, 2, solution.n_bins), 0.1) * u.adu}
    }
    out = tmp_path / "extracted_spectra_angle_000.00.png"
    plot_extracted_spectra(
        apply_wavel_solution(spectra, solution, errors),
        config,
        str(out),
        dc_rates=[0.1],
        angle_deg=0.0,
        extraction_method="box_sum",
    )
    assert out.exists() and out.stat().st_size > 0
