"""Tests for the [detector] buffer_footprint margin around readout cutouts."""

# pytest fixtures are injected by argument name
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from modules.core.instrumental.channels import OutputChannel
from modules.core.instrumental.detector import Detector, footprint_buffer_from_config
from modules.core.instrumental.dispersion import DispersionLaw
from modules.core.instrumental.effects import expose_plus_detector_effects
from modules.core.instrumental.readouts import (
    extract_spectra_from_readouts,
    write_detector_readouts,
)

CHANNEL_NAMES = ["output_1_bright", "output_2_bright", "output_3_dark", "output_4_dark"]


def _detector_config(buffer_footprint=None):
    """400x400 detector; 3-row trace from (y, x) = (100, 300), 2.2 pix per bin."""
    detector = {
        "size": "400",
        "pitch_pix": "25",
        "pix_per_wavel_bin": "2.2",
        "pix_spectral_width": "3",
        "quantum_efficiency": "0.8",
        "gain": "2.0",
        "read_noise": "6.0",
    }
    if buffer_footprint is not None:
        detector["buffer_footprint"] = buffer_footprint
    return {"detector": detector}


def _channels(config, n_bins=3):
    channels = {}
    for i, name in enumerate(CHANNEL_NAMES):
        channel = OutputChannel(name=name)
        det = Detector(config=config, num_wavel_bins=n_bins)
        law = DispersionLaw(config)
        det.set_footprint(law.make_footprint(det.n_pix_y, det.n_pix_x, n_bins))
        channel.detector = det
        channel.astroph_signal["star"] = {
            "flux_astro_1d_interpolated_ph_sec_pixel": np.array([1.0, 4.0, 9.0])
            * (i + 1)
            * u.ph
            / (u.s * u.pix)
        }
        channels[name] = channel
    expose_plus_detector_effects(channels, [], t_frame=10.0 * u.s, n_readouts=2)
    return channels


class TestFootprintBufferFromConfig:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, (0, 0)),
            ("", (0, 0)),
            ("4, 4", (4, 4)),
            ("2, 5", (2, 5)),
            ("3", (3, 3)),
        ],
    )
    def test_parses(self, raw, expected):
        assert footprint_buffer_from_config(_detector_config(raw)) == expected

    @pytest.mark.parametrize("raw", ["-1, 2", "1, 2, 3"])
    def test_rejects_bad_values(self, raw):
        with pytest.raises(ValueError, match="buffer_footprint"):
            footprint_buffer_from_config(_detector_config(raw))

    def test_detector_stores_buffer(self):
        det = Detector(config=_detector_config("2, 5"), num_wavel_bins=3)
        assert det.footprint_buffer_yx == (2, 5)


class TestFootprintBboxBuffer:
    def test_grows_bbox_on_each_side(self):
        det = _channels(_detector_config())["output_3_dark"].detector
        assert det.footprint_bbox() == (100, 103, 300, 307)
        assert det.footprint_bbox(buffer_yx=(4, 4)) == (96, 107, 296, 311)
        assert det.footprint_bbox(buffer_yx=(1, 6)) == (99, 104, 294, 313)

    def test_clips_to_detector(self):
        det = _channels(_detector_config())["output_3_dark"].detector
        assert det.footprint_bbox(buffer_yx=(300, 350)) == (0, 400, 0, 400)


class TestBufferedReadoutCutout:
    def test_cutout_includes_unilluminated_margin(self, tmp_path):
        channels = _channels(_detector_config("4, 4"))
        file_name = str(tmp_path / "readouts.fits")
        write_detector_readouts(channels, file_name, t_frame=10.0 * u.s)

        channel = channels["output_3_dark"]
        with fits.open(file_name) as hdul:
            hdu = hdul["CH3DK"]
            # trace bbox (100:103, 300:307) plus 4 pixels on every side
            assert hdu.data.shape == (2, 2, 3 + 8, 7 + 8)
            assert (hdu.header["Y0"], hdu.header["X0"]) == (96, 296)
            assert (hdu.header["BUFFY"], hdu.header["BUFFX"]) == (4, 4)
            full = channel.readout_frames[0].source_e["star"] / 2.0
            star = hdu.data[0, 0]
            np.testing.assert_allclose(star, full[96:107, 296:311])
            # the margin is unilluminated (zero ADU); all flux is kept
            assert np.all(star[:4] == 0.0) and np.all(star[-4:] == 0.0)
            assert np.all(star[:, :4] == 0.0) and np.all(star[:, -4:] == 0.0)
            assert np.isclose(star.sum(), full.sum())

    def test_no_buffer_keeps_tight_cutout(self, tmp_path):
        channels = _channels(_detector_config())
        file_name = str(tmp_path / "readouts.fits")
        write_detector_readouts(channels, file_name, t_frame=10.0 * u.s)
        with fits.open(file_name) as hdul:
            header = hdul["CH3DK"].header
            assert hdul["CH3DK"].data.shape == (2, 2, 3, 7)
            assert (header["BUFFY"], header["BUFFX"]) == (0, 0)

    def test_full_frame_ignores_buffer(self, tmp_path):
        channels = _channels(_detector_config("4, 4"))
        file_name = str(tmp_path / "full.fits")
        write_detector_readouts(channels, file_name, t_frame=10.0 * u.s, cutout=False)
        with fits.open(file_name) as hdul:
            header = hdul["CH3DK"].header
            assert hdul["CH3DK"].data.shape == (2, 2, 400, 400)
            assert (header["Y0"], header["X0"]) == (0, 0)

    @pytest.mark.parametrize("method", ["sharp_birchall", "box_sum"])
    def test_extraction_unchanged_by_buffer(self, tmp_path, method):
        tight = _channels(_detector_config())
        buffered = _channels(_detector_config("4, 4"))
        tight_file = str(tmp_path / "tight.fits")
        buffered_file = str(tmp_path / "buffered.fits")
        write_detector_readouts(tight, tight_file, t_frame=10.0 * u.s)
        write_detector_readouts(buffered, buffered_file, t_frame=10.0 * u.s)

        kwargs = {"method": method, "t_frame": 10.0 * u.s, "dc_rates": [0.1, 0.5]}
        from_tight = extract_spectra_from_readouts(tight, tight_file, **kwargs)
        from_buffered = extract_spectra_from_readouts(buffered, buffered_file, **kwargs)
        for ch_name, spectra in from_tight.items():
            for comp, spectrum in spectra.items():
                assert u.allclose(from_buffered[ch_name][comp], spectrum)
            if method == "sharp_birchall":
                assert u.allclose(
                    buffered[ch_name].extracted_error_adu,
                    tight[ch_name].extracted_error_adu,
                )
