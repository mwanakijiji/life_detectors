"""Tests for assembling extracted spectra into (wavelength, readout, channel) cubes."""

import numpy as np
from astropy import units as u
from astropy.io import fits

from modules.core.calculator.extracted_cube import assemble_extracted_spectra_cubes
from modules.core.instrumental.dispersion import DispersionLaw
from modules.utils.helpers.hdf5_io import record_info_at_angle_and_qe

CHANNELS = ["output_1_bright", "output_2_bright", "output_3_dark", "output_4_dark"]
N_BINS, N_READ = 3, 2
T_FRAME, GAIN = 10.0, 2.0
DC_RATES = [0.1, 0.2]


def _star_value(angle_deg, readout, channel_index, wavel_bin, i_dc=0):
    # distinct value for every (DC, angle, readout, channel, bin)
    return (
        10000 * i_dc + 1000 * angle_deg + 100 * readout + 10 * channel_index + wavel_bin
    )


def _error_value(angle_deg, readout, channel_index, wavel_bin, i_dc):
    return 0.5 * _star_value(angle_deg, readout, channel_index, wavel_bin, i_dc) + 1.0


def _per_dc(fn, angle, i_ch):
    return np.array(
        [
            [
                [fn(angle, r, i_ch, b, i_dc) for b in range(N_BINS)]
                for r in range(N_READ)
            ]
            for i_dc in range(len(DC_RATES))
        ],
        dtype=float,
    )


def _write_angle_files(save_dir, output_channels, angles, qe=0.8, with_errors=True):
    for channel in output_channels.values():
        channel.detector.dark_current_rate_e_pix_sec = (
            np.array(DC_RATES) * u.electron / (u.pix * u.s)
        )
    for angle in angles:
        extracted = {
            name: {
                "star": _per_dc(_star_value, angle, i_ch) * u.adu,
                "instrum_systematics": np.full((len(DC_RATES), N_READ, N_BINS), 0.5)
                * u.adu,
            }
            for i_ch, name in enumerate(CHANNELS)
        }
        errors = {
            name: {
                "error": _per_dc(_error_value, angle, i_ch) * u.adu,
                "vark": np.ones((len(DC_RATES), N_READ, N_BINS)) * u.adu**2,
            }
            for i_ch, name in enumerate(CHANNELS)
        }
        record_info_at_angle_and_qe(
            angle_deg=angle,
            qe=qe,
            output_channels=output_channels,
            save_dir=save_dir,
            t_frame=T_FRAME * u.s,
            extracted_spectra_adu=extracted,
            extracted_errors_adu=errors if with_errors else None,
            extraction_method="sharp_birchall" if with_errors else "box_sum",
        )


def test_assembles_one_cube_per_dc_with_readouts_ordered_by_angle(
    tmp_path, make_output_channels
):
    save_dir = str(tmp_path) + "/"
    output_channels = make_output_channels(n_bins=N_BINS, gain=GAIN)
    ref = output_channels["output_3_dark"]
    npix = DispersionLaw.n_pix_per_bin(ref.detector.footprint_cube).value
    angles = [240.0, 0.0, 120.0]  # written out of order on purpose
    _write_angle_files(save_dir, output_channels, angles)
    # a different-QE file must be ignored
    _write_angle_files(save_dir, output_channels, [0.0], qe=0.5)

    written = assemble_extracted_spectra_cubes(
        read_dir=save_dir, qe=0.8, save_dir=save_dir
    )

    assert [p.rsplit("/", 1)[-1] for p in written] == [
        "extracted_cube_qe_0.80_dc_00.100.fits",
        "extracted_cube_qe_0.80_dc_00.200.fits",
    ]
    sorted_angles = sorted(angles)
    for i_dc, (file_name, dc) in enumerate(zip(written, DC_RATES)):
        with fits.open(file_name) as hdul:
            total = hdul[0].data
            assert total.shape == (N_BINS, len(angles) * N_READ, len(CHANNELS))
            assert hdul[0].header["DC"] == dc
            assert hdul[0].header["QE"] == 0.8
            assert hdul[0].header["EXTRACT"] == "sharp_birchall"
            assert [hdul[0].header[f"CHAN{i}"] for i in range(4)] == [
                "CH1BR",
                "CH2BR",
                "CH3DK",
                "CH4DK",
            ]

            star = hdul["STAR"].data
            for i_angle, angle in enumerate(sorted_angles):
                for r in range(N_READ):
                    col = i_angle * N_READ + r
                    for i_ch in range(len(CHANNELS)):
                        expected = [
                            _star_value(angle, r, i_ch, b, i_dc) for b in range(N_BINS)
                        ]
                        assert np.allclose(star[:, col, i_ch], expected)
                        expected_error = [
                            _error_value(angle, r, i_ch, b, i_dc) for b in range(N_BINS)
                        ]
                        assert np.allclose(
                            hdul["ERROR"].data[:, col, i_ch], expected_error
                        )

            dark = hdul["DARK_CURRENT"].data
            pedestal = dc * npix * T_FRAME / GAIN
            assert np.allclose(dark, pedestal[:, None, None])
            assert np.allclose(total, star + hdul["INSTRUM_SYSTEMATICS"].data + dark)

            readouts = hdul["READOUTS"].data
            assert np.allclose(readouts["angle_deg"], np.repeat(sorted_angles, N_READ))
            assert list(readouts["readout_index"]) == [0, 1] * len(angles)
            wavelength = hdul["WAVELENGTH"].data
            assert np.allclose(wavelength["center"], ref.bin_centers.value)
            assert np.allclose(wavelength["lower_edge"], ref.bin_edges.value[:-1])
            assert np.allclose(wavelength["npix"], npix)


def test_no_error_hdu_without_errors(tmp_path, make_output_channels):
    save_dir = str(tmp_path) + "/"
    output_channels = make_output_channels(n_bins=N_BINS, gain=GAIN)
    _write_angle_files(save_dir, output_channels, [0.0], with_errors=False)
    written = assemble_extracted_spectra_cubes(
        read_dir=save_dir, qe=0.8, save_dir=save_dir
    )
    with fits.open(written[0]) as hdul:
        assert "ERROR" not in [h.name for h in hdul]
        assert hdul[0].header["EXTRACT"] == "box_sum"
        # component list excludes the __ datasets
        assert {h.name for h in hdul[1:-2]} == {
            "STAR",
            "INSTRUM_SYSTEMATICS",
            "DARK_CURRENT",
        }


def test_no_angle_files_writes_nothing(tmp_path):
    written = assemble_extracted_spectra_cubes(
        read_dir=str(tmp_path) + "/", qe=0.8, save_dir=str(tmp_path) + "/"
    )
    assert isinstance(written, list) and not written
    assert not list(tmp_path.iterdir())
