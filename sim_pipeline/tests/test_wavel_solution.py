"""Tests for DispersionLaw.write_wavel_solution and the shared bin-edge helper."""

# pytest fixtures are injected by argument name
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from modules.core.instrumental.dispersion import (
    DispersionLaw,
    wavel_bin_edges_from_config,
)
from modules.core.instrumental.pipeline import InstrumentDepTerms


@pytest.fixture
def config():
    """400x400 detector, trace from (y, x) = (100, 300), R = 20 over 4-18 um."""
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
    }


def test_bin_edges_match_output_channels(config):
    edges = wavel_bin_edges_from_config(config)
    instr = InstrumentDepTerms(config, sources_astroph={}, sources_to_include=[])
    for channel in instr.output_channels.values():
        assert u.allclose(channel.bin_edges, edges)
    assert edges.unit == u.um
    assert edges[0] == 4.0 * u.um
    assert edges[-1] <= 18.0 * u.um
    np.testing.assert_allclose(edges[1:] / edges[:-1], 1.05)


def test_wavel_at_x_hits_bin_edges_and_is_nan_off_trace(config):
    law = DispersionLaw(config)
    edges = wavel_bin_edges_from_config(config)
    x_edges = 300.0 + 2.2 * np.arange(edges.size)
    np.testing.assert_allclose(law.wavel_at_x(x_edges, edges), edges.value)
    assert np.all(np.isnan(law.wavel_at_x(np.array([10.0, 399.0]), edges)))


def test_no_file_name_configured_writes_nothing(config, tmp_path, caplog):
    law = DispersionLaw(config)
    with caplog.at_level("INFO"):
        assert law.write_wavel_solution(config) is None
    assert "wavelength solution not written" in caplog.text
    assert not list(tmp_path.iterdir())


def test_writes_file_named_in_config(config, tmp_path):
    out = tmp_path / "sub" / "wavel_solution.fits"
    config["extraction"] = {"file_name_wavel_solution_write": str(out)}
    assert DispersionLaw(config).write_wavel_solution(config) == str(out)
    assert out.exists()


def test_file_contents(config, tmp_path):
    law = DispersionLaw(config)
    edges = wavel_bin_edges_from_config(config).value
    n_bins = edges.size - 1
    out = str(tmp_path / "wavel_solution.fits")
    law.write_wavel_solution(config, file_name=out)

    with fits.open(out) as hdul:
        assert [h.name for h in hdul] == ["PRIMARY", "BINS", "WAVEL_COL", "WAVEL_MAP"]
        header = hdul[0].header
        assert header["NBINS"] == n_bins
        assert header["SPECRES"] == 20.0
        assert header["PIXPBIN"] == 2.2
        assert (header["TRACEY0"], header["TRACEX0"]) == (100.0, 300.0)
        assert (header["NPIXY"], header["NPIXX"]) == (400, 400)

        bins = hdul["BINS"].data
        np.testing.assert_array_equal(bins["BIN"], np.arange(n_bins))
        np.testing.assert_allclose(bins["WAVEL_LO_UM"], edges[:-1])
        np.testing.assert_allclose(bins["WAVEL_HI_UM"], edges[1:])
        np.testing.assert_allclose(
            bins["WAVEL_CEN_UM"], np.sqrt(edges[:-1] * edges[1:])
        )
        np.testing.assert_allclose(bins["X_START_PIX"], 300.0 + 2.2 * np.arange(n_bins))
        np.testing.assert_allclose(bins["X_END_PIX"] - bins["X_START_PIX"], 2.2)
        assert np.all(bins["Y_START_PIX"] == 100) and np.all(bins["Y_END_PIX"] == 103)
        np.testing.assert_allclose(bins["NPIX"], 2.2 * 3)

        wavel_col = hdul["WAVEL_COL"].data
        assert wavel_col.shape == (400,)
        assert hdul["WAVEL_COL"].header["BUNIT"] == "um"
        # column k is centred at x = k + 0.5; trace spans x in [300, 300 + 2.2 * n)
        on_trace = (np.arange(400) + 0.5 >= 300.0) & (
            np.arange(400) + 0.5 <= 300.0 + 2.2 * n_bins
        )
        assert np.all(np.isfinite(wavel_col[on_trace]))
        assert np.all(np.isnan(wavel_col[~on_trace]))
        assert np.all(np.diff(wavel_col[on_trace]) > 0)
        # each column's wavelength falls in the bin that covers its centre
        for k in np.flatnonzero(on_trace):
            b = min(int((k + 0.5 - 300.0) // 2.2), n_bins - 1)
            assert edges[b] <= wavel_col[k] <= edges[b + 1]

        wavel_map = hdul["WAVEL_MAP"].data
        assert wavel_map.shape == (400, 400)
        np.testing.assert_array_equal(wavel_map[100:103], np.tile(wavel_col, (3, 1)))
        assert np.all(np.isnan(wavel_map[:100])) and np.all(np.isnan(wavel_map[103:]))


def test_trace_off_detector_raises(config, tmp_path):
    config["detector"]["size"] = "320"  # trace ends at x = 300 + 2.2 * 30 = 366
    with pytest.raises(ValueError, match="does not fit"):
        DispersionLaw(config).write_wavel_solution(
            config, file_name=str(tmp_path / "w.fits")
        )
