"""Tests for recording the wavelength solution in the angle HDF5 files."""

# pytest fixtures are injected by argument name
# pylint: disable=redefined-outer-name

import os

import h5py
import numpy as np
import pytest
from astropy import units as u

from modules.core.calculator.extracted_cube import assemble_extracted_spectra_cubes
from modules.core.instrumental.dispersion import (
    DispersionLaw,
    wavel_bin_edges_from_config,
)
from modules.core.instrumental.pipeline import InstrumentDepTerms
from modules.core.instrumental.wavel_solution import (
    WavelSolution,
    read_wavel_solution,
)
from modules.utils.helpers.hdf5_io import (
    WAVEL_PER_COLUMN_KEY,
    WAVEL_SOLUTION_FILE_ATTR,
    X_END_KEY,
    X_START_KEY,
    record_info_at_angle_and_qe,
)

NAMES = ["output_1_bright", "output_2_bright", "output_3_dark", "output_4_dark"]


@pytest.fixture
def law_config(tmp_path):
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


def _extracted(n_bins, n_read=2):
    shape = (1, n_read, n_bins)
    return {
        name: {
            "star": np.ones(shape) * (i + 1.0) * u.adu,
            "instrum_systematics": np.zeros(shape) * u.adu,
        }
        for i, name in enumerate(NAMES)
    }


def _solution_for_channels(channels, n_pix_x=400):
    """A solution matching make_output_channels' bins, with x edges and columns."""
    edges = next(iter(channels.values())).bin_edges
    n_bins = len(edges) - 1
    x_edges = 300.0 + 2.2 * np.arange(n_bins + 1)
    wavel_col = np.full(n_pix_x, np.nan)
    wavel_col[300:307] = np.linspace(5.0, 6.5, 7)
    return WavelSolution.from_bin_edges(
        edges, source="/some/wavel_solution.fits", x_edges=x_edges, wavel_col=wavel_col
    )


def test_read_solution_includes_per_column_wavelengths(law_config, tmp_path):
    out = str(tmp_path / "w.fits")
    law = DispersionLaw(law_config)
    law.write_wavel_solution(law_config, file_name=out)
    solution = read_wavel_solution(out)
    edges = wavel_bin_edges_from_config(law_config)
    expected = law.wavel_at_x(np.arange(400) + 0.5, edges)
    assert solution.wavel_col.unit == u.um
    np.testing.assert_allclose(solution.wavel_col.value, expected)
    assert np.isnan(solution.wavel_col[0].value)


def test_hdf5_records_bin_x_columns_and_source(tmp_path, make_output_channels):
    channels = make_output_channels(n_bins=3)
    solution = _solution_for_channels(channels)
    record_info_at_angle_and_qe(
        angle_deg=0.0,
        qe=0.7,
        output_channels=channels,
        save_dir=str(tmp_path) + "/",
        t_frame=10.0 * u.s,
        extracted_spectra_adu=_extracted(3),
        wavel_solution=solution,
    )
    with h5py.File(tmp_path / "angle_000.00_qe_0.70.hdf5", "r") as handle:
        group = handle["__extracted_qe_0.70"]
        assert group.attrs[WAVEL_SOLUTION_FILE_ATTR] == "/some/wavel_solution.fits"
        np.testing.assert_allclose(group[X_START_KEY][()], solution.x_lo.value)
        np.testing.assert_allclose(group[X_END_KEY][()], solution.x_hi.value)
        assert group[X_START_KEY].attrs["unit"] == "pix"
        column = group[WAVEL_PER_COLUMN_KEY][()]
        assert column.shape == (400,)
        np.testing.assert_allclose(column, solution.wavel_col.value)  # NaNs match
        assert group[WAVEL_PER_COLUMN_KEY].attrs["unit"] == "um"
        # these are datasets, so channels are still the only subgroups
        subgroups = {k for k in group if isinstance(group[k], h5py.Group)}
        assert subgroups == set(NAMES)


def test_hdf5_without_solution_records_only_the_note(tmp_path, make_output_channels):
    record_info_at_angle_and_qe(
        angle_deg=0.0,
        qe=0.7,
        output_channels=make_output_channels(n_bins=3),
        save_dir=str(tmp_path) + "/",
        t_frame=10.0 * u.s,
        extracted_spectra_adu=_extracted(3),
    )
    with h5py.File(tmp_path / "angle_000.00_qe_0.70.hdf5", "r") as handle:
        group = handle["__extracted_qe_0.70"]
        assert group.attrs[WAVEL_SOLUTION_FILE_ATTR] == "(none given)"
        for key in (X_START_KEY, X_END_KEY, WAVEL_PER_COLUMN_KEY):
            assert key not in group


def test_cubes_still_assemble_with_solution_datasets(tmp_path, make_output_channels):
    channels = make_output_channels(n_bins=3)
    solution = _solution_for_channels(channels)
    for angle in (0.0, 180.0):
        record_info_at_angle_and_qe(
            angle_deg=angle,
            qe=0.7,
            output_channels=channels,
            save_dir=str(tmp_path) + "/",
            t_frame=10.0 * u.s,
            extracted_spectra_adu=_extracted(3),
            wavel_solution=solution,
        )
    written = assemble_extracted_spectra_cubes(
        read_dir=str(tmp_path) + "/", qe=0.7, save_dir=str(tmp_path) + "/"
    )
    assert len(written) == 1


def test_fallback_solution_has_per_column_wavelengths(law_config):
    instr = InstrumentDepTerms(law_config, sources_astroph={}, sources_to_include=[])
    instr.define_dispersion(plot=False)
    os.remove(law_config["extraction"]["file_name_wavel_solution_write"])
    solution = instr.load_wavel_solution()
    assert solution.source == "channel bin edges"
    edges = wavel_bin_edges_from_config(law_config)
    expected = DispersionLaw(law_config).wavel_at_x(np.arange(400) + 0.5, edges)
    np.testing.assert_allclose(solution.wavel_col.value, expected)
