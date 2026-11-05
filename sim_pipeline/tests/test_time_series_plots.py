"""Tests for the counts vs. wavelength and time plots built from extracted cubes."""

# pytest fixtures are injected by argument name
# pylint: disable=redefined-outer-name,protected-access

import os

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy import units as u  # noqa: E402

from modules.core.calculator.extracted_cube import (  # noqa: E402
    assemble_extracted_spectra_cubes,
    read_extracted_cube,
)
from modules.utils.helpers.hdf5_io import record_info_at_angle_and_qe  # noqa: E402
from modules.viz import time_series  # noqa: E402

NAMES = ["output_1_bright", "output_2_bright", "output_3_dark", "output_4_dark"]
N_BINS, N_READ, ANGLES = 3, 2, (0.0, 120.0, 240.0)
GAIN = 2.0


@pytest.fixture
def cube_files(tmp_path, make_output_channels):
    """Three angles x two readouts; star counts depend on channel and angle."""
    channels = make_output_channels(n_bins=N_BINS, gain=GAIN)
    for channel in channels.values():
        channel.detector.dark_current_rate_e_pix_sec = (
            np.array([0.0, 0.5]) * u.electron / (u.pix * u.s)
        )
    for i_angle, angle in enumerate(ANGLES):
        extracted = {
            name: {
                "star": np.full((2, N_READ, N_BINS), 10.0 * (i_ch + 1) + i_angle)
                * u.adu,
                "instrum_systematics": np.zeros((2, N_READ, N_BINS)) * u.adu,
            }
            for i_ch, name in enumerate(NAMES)
        }
        record_info_at_angle_and_qe(
            angle_deg=angle,
            qe=0.7,
            output_channels=channels,
            save_dir=str(tmp_path) + "/",
            t_frame=10.0 * u.s,
            extracted_spectra_adu=extracted,
        )
    return assemble_extracted_spectra_cubes(
        read_dir=str(tmp_path) + "/", qe=0.7, save_dir=str(tmp_path) + "/"
    )


def test_read_extracted_cube(cube_files):
    assert len(cube_files) == 2  # one per DC value
    cube = read_extracted_cube(cube_files[0])
    n_total = len(ANGLES) * N_READ
    assert cube["total_adu"].shape == (N_BINS, n_total, 4)
    assert cube["channels"] == ["CH1BR", "CH2BR", "CH3DK", "CH4DK"]
    assert set(cube["components_adu"]) == {
        "star",
        "instrum_systematics",
        "dark_current",
    }
    assert cube["gain"] == GAIN
    assert cube["t_frame_s"] == 10.0
    assert cube["n_angles"] == 3 and cube["n_readouts_per_angle"] == N_READ
    np.testing.assert_allclose(cube["angle_deg"], np.repeat(ANGLES, N_READ))
    np.testing.assert_allclose(cube["wavel_hi"][:-1], cube["wavel_lo"][1:])
    # star in channel 3 at angle index 1 (first readout of that angle) = 10 * 3 + 1
    np.testing.assert_allclose(cube["components_adu"]["star"][:, N_READ, 2], 31.0)


@pytest.mark.parametrize(
    ("x_axis", "first", "last"),
    [
        ("frame", -0.5, 5.5),
        ("time", 0.0, 60.0),  # 6 frames x 10 s
        ("angle", 0.0, 360.0),  # 3 angles x 120 deg, 2 readouts each
    ],
)
def test_x_edges(cube_files, x_axis, first, last):
    cube = read_extracted_cube(cube_files[0])
    edges, label = time_series._x_edges_and_label(cube, x_axis)
    assert edges.size == len(ANGLES) * N_READ + 1
    assert edges[0] == pytest.approx(first)
    assert edges[-1] == pytest.approx(last)
    assert np.all(np.diff(edges) > 0)
    assert label


def test_x_edges_rejects_unknown_axis(cube_files):
    with pytest.raises(ValueError, match="x_axis"):
        time_series._x_edges_and_label(read_extracted_cube(cube_files[0]), "nope")


def test_color_limits_symmetric_and_constant():
    lo, hi = time_series._color_limits(np.array([-1.0, 0.0, 3.0]), symmetric=True)
    assert lo == -hi and hi > 0
    lo, hi = time_series._color_limits(np.zeros(5), symmetric=False)
    assert hi > lo


def test_file_name():
    name = time_series.counts_vs_time_file_name(
        "/out/extracted_cube_qe_0.70_dc_00.500.fits", "star", "angle"
    )
    assert name == "/out/counts_vs_angle_qe_0.70_dc_00.500_star.png"
    assert time_series.counts_vs_time_file_name(
        "/out/extracted_cube_qe_0.70_dc_00.500.fits", None, "frame"
    ).endswith("_total.png")


@pytest.mark.parametrize("x_axis", ["frame", "time", "angle"])
def test_plot_writes_png(cube_files, tmp_path, x_axis):
    cube = read_extracted_cube(cube_files[0])
    out = tmp_path / f"counts_{x_axis}.png"
    time_series.plot_counts_vs_time(cube, str(out), component="star", x_axis=x_axis)
    assert out.exists() and out.stat().st_size > 0


def test_plot_for_cubes_writes_total_per_dc_and_components_once(cube_files):
    written = time_series.plot_counts_vs_time_for_cubes(cube_files, x_axis="frame")
    # first DC: total + 3 components; second DC: total only
    assert len(written) == 1 + 3 + 1
    for path in written:
        assert os.path.exists(path)
    assert sum(p.endswith("_total.png") for p in written) == 2
