"""
Unit tests for instrumental noise calculations.
"""

# pytest fixtures are injected by argument name
# pylint: disable=redefined-outer-name

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.sparse
from astropy import units as u
from astropy.io import fits

from modules.core.instrumental.channels import OutputChannel
from modules.core.instrumental.detector import SYSTEMATICS_KEY, Detector
from modules.core.instrumental.dispersion import DispersionLaw
from modules.core.instrumental.effects import (
    DetectorEffect,
    DetectorFrames,
    HotPixelsEffect,
    apply_detector_effects,
    build_detector_effects,
)
from modules.core.instrumental.pipeline import InstrumentDepTerms
from modules.core.instrumental.readouts import (
    extract_spectra_from_readouts,
    write_detector_readouts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
APERTURE_YAML = REPO_ROOT / "sim_pipeline/config/aperture_array_double_bracewell.yaml"


def _geometric_n_bins(lambda_min: float, lambda_max: float, spec_res: float) -> int:
    return int(np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / spec_res)))


@pytest.fixture
def instrum_base_config():
    """Minimal config required by InstrumentDepTerms.__init__."""
    return {
        "detector": {"spec_res": "20"},
        "wavelength_range": {"min": "4.0", "max": "18.0"},
    }


@pytest.fixture
def detector_geometry_config():
    """Detector geometry for a 400x400 array and a 3-row trace."""
    return {
        "detector": {
            "size": "400",
            "pitch_pix": "25",
            "pix_per_wavel_bin": "2.2",
            "pix_spectral_width": "3",
            "quantum_efficiency": "0.8",
        }
    }


@pytest.fixture
def noise_calc_config(instrum_base_config):
    """Config with read-noise, dark-current and gain values."""
    config = {
        **instrum_base_config,
        "detector": {
            **instrum_base_config["detector"],
            "read_noise": "2.0, 3.0",
            "dark_current": "0.0, 0.2, 0.1",
            "gain": "4.5",
            "quantum_efficiency": "0.8",
            "e_per_ph": "1.0",
            "photons_to_e": "1.0",
        },
        "observation": {"t_int_obs_total": "100", "t_int_frame": "10"},
        "telescope": {"eta_t": "0.8"},
        "dirs": {"save_s2n_data_unique_dir": "/tmp/"},
    }
    return config


def _make_flux_cube(n_wavel: int = 3, n_pix: int = 5, scale: float = 1.0):
    return np.ones((n_wavel, n_pix, n_pix)) * scale * u.ph / (u.um * u.m**2 * u.s)


def _output_flux_dict(cube):
    return {
        "output_1_bright": cube,
        "output_2_bright": cube,
        "output_3_dark": cube,
        "output_4_dark": cube,
    }


def _attach_detector_footprint(
    channel: OutputChannel, detector_geometry_config, n_bins: int = 3
):
    det = Detector(config=detector_geometry_config, num_wavel_bins=n_bins)
    law = DispersionLaw(detector_geometry_config)
    det.set_footprint(law.make_footprint(det.side_length_pix, det.num_wavel_bins))
    channel.detector = det
    return det


def _make_footprint_sparse(
    law: DispersionLaw, side_length_pix: int, num_wavel_bins: int
) -> scipy.sparse.csr_matrix:
    """Vectorized sparse equivalent of DispersionLaw.make_footprint, for cross-checks.

    Returns a CSR matrix of shape (n_bins, n_pix * n_pix), each row a row-major
    (y, x) flattening of that bin's footprint.
    """
    # vectorized: compute all bin start/end columns at once (they're a linear
    # function of wavel_bin_num), then build row/col/data arrays for COO
    # construction in one shot — no per-bin Python-level array allocation.
    side = side_length_pix
    bin_idx = np.arange(num_wavel_bins)
    x_start = law.starting_pixel[1] + bin_idx * law.pix_per_wavel_bin
    x_end = x_start + law.pix_per_wavel_bin

    # same edge logic as the dense loop
    ceil_start = np.ceil(x_start).astype(int)
    floor_end = np.floor(x_end).astype(int)
    frac_start = ceil_start - x_start
    frac_end = x_end - floor_end

    # each bin touches columns ceil_start-1 (frac_start),
    # [ceil_start, floor_end) (1.0), and floor_end (frac_end); the count of
    # full columns can differ by 1 between bins
    n_cols = (floor_end - ceil_start) + 2
    col_bin = np.repeat(bin_idx, n_cols)
    offset = np.arange(n_cols.sum()) - np.repeat(np.cumsum(n_cols) - n_cols, n_cols)
    col_x = ceil_start[col_bin] - 1 + offset
    col_val = np.where(
        offset == 0,
        frac_start[col_bin],
        np.where(offset == n_cols[col_bin] - 1, frac_end[col_bin], 1.0),
    )

    # broadcast across the spectral-width rows (constant for all bins)
    y0 = int(law.starting_pixel[0])
    y1 = int(law.starting_pixel[0] + law.pix_spectral_width)
    ys = np.arange(y0, y1)
    rows = np.repeat(col_bin, ys.size)
    cols = (ys[None, :] * side + col_x[:, None]).ravel()
    data = np.repeat(col_val, ys.size)
    col_x_rep = np.repeat(col_x, ys.size)

    # drop zero-weight edges and anything off the detector
    keep = (data != 0) & (col_x_rep >= 0) & (col_x_rep < side)
    return scipy.sparse.coo_matrix(
        (data[keep], (rows[keep], cols[keep])),
        shape=(num_wavel_bins, side * side),
    ).tocsr()


def _make_astroph_signal(
    bin_centers, bin_widths, n_pix_per_bin, flux_um: float = 10.0, qe: float = 0.8
):
    flux_ph_sec_um = np.full(len(bin_centers), flux_um) * u.ph / (u.um * u.s) * qe
    flux_ph_sec_wavel_bin = flux_ph_sec_um * bin_widths
    flux_ph_sec_pixel = flux_ph_sec_wavel_bin / n_pix_per_bin
    return {
        "wavel": bin_centers,
        "flux_astro_1d_interpolated_ph_sec_um": flux_ph_sec_um,
        "flux_astro_1d_interpolated_ph_sec_wavel_bin": flux_ph_sec_wavel_bin,
        "flux_astro_1d_interpolated_ph_sec_pixel": flux_ph_sec_pixel,
        "n_pix_per_wavel_bin": n_pix_per_bin,
    }


@pytest.fixture
def transmission_config(tmp_path):
    """Config for generating transmission screens."""
    return {
        "detector": {"spec_res": "20"},
        "wavelength_range": {"min": "4.0", "max": "18.0"},
        "telescope": {"aperture_array_config_file_name": str(APERTURE_YAML)},
        "onsky_scene": {"n_pix": "1001", "pix_size_mas": "10.7", "half_pix": "1"},
        "nulling": {"nulling_factor": "0.001"},
        "dirs": {"save_s2n_data_unique_dir": str(tmp_path / "out") + "/"},
    }


@pytest.fixture
def disperse_config(instrum_base_config, detector_geometry_config):
    """Config for dispersing signals onto the detector."""
    return {
        **instrum_base_config,
        **detector_geometry_config,
        "dirs": {"save_s2n_data_unique_dir": "/tmp/"},
        "observation": {"N_int_per_angle": "1"},
        "detector": {
            **instrum_base_config["detector"],
            **detector_geometry_config["detector"],
            "quantum_efficiency": "0.8",
        },
    }


class TestInstrumentDepTerms:
    def test_init_creates_output_channels_with_wavelength_bins(
        self, instrum_base_config
    ):
        instr = InstrumentDepTerms(
            instrum_base_config,
            sources_astroph={},
            sources_to_include=["star"],
        )

        assert instr.config is instrum_base_config
        assert instr.sources_to_include == ["star"]
        assert isinstance(instr.prop_dict, dict)

        assert set(instr.output_channels.keys()) == {
            "output_1_bright",
            "output_2_bright",
            "output_3_dark",
            "output_4_dark",
        }
        n_bins = _geometric_n_bins(4.0, 18.0, 20.0)
        for channel in instr.output_channels.values():
            assert isinstance(channel, OutputChannel)
            assert channel.spec_R == 20.0
            assert len(channel.bin_centers) == n_bins
            assert len(channel.bin_edges) == n_bins + 1
            assert len(channel.bin_widths) == n_bins

    def test_detector_parses_noise_arrays(
        self, noise_calc_config, detector_geometry_config
    ):
        config = {
            **noise_calc_config,
            "detector": {
                **noise_calc_config["detector"],
                **detector_geometry_config["detector"],
            },
        }
        det = Detector(config=config, num_wavel_bins=3)

        assert det.read_noise_e_rms.unit.is_equivalent(u.electron / u.pix)
        assert np.allclose(det.read_noise_e_rms.value, [2.0, 3.0])
        dc_rate = det.dark_current_rate_e_pix_sec
        assert dc_rate.unit.is_equivalent(u.electron / (u.pix * u.s))
        assert np.allclose(dc_rate.value, [0.0, 0.1])  # start, stop, step
        assert det.gain == 4.5 * u.electron / u.adu
        assert det.e_per_ph == 1.0 * u.electron / u.ph

    def test_detector_parses_single_noise_values(self, detector_geometry_config):
        config = {
            "detector": {
                **detector_geometry_config["detector"],
                "read_noise": "6.0",
                "dark_current": "0.05",
                "gain": "4.5",
            },
        }
        det = Detector(config=config, num_wavel_bins=3)

        assert np.allclose(det.read_noise_e_rms.value, [6.0])
        assert np.allclose(det.dark_current_rate_e_pix_sec.value, [0.05])

    @patch(
        "modules.core.instrumental.aperture.compute_collecting_area_m2",
        return_value=25.0,
    )
    def test_pass_through_aperture_scales_flux_cubes_by_collecting_area_and_throughput(
        self, _mock_area, instrum_base_config
    ):
        config = {**instrum_base_config, "telescope": {"eta_t": "0.5"}}
        wavel = np.array([1.0, 2.0, 3.0]) * u.um
        cube = _make_flux_cube(scale=2.0)

        sources_astroph = {
            "star": {
                "wavel": wavel,
                "flux_cube_post_screen_ph_sec_m2_um": _output_flux_dict(cube),
            }
        }

        instr = InstrumentDepTerms(
            config, sources_astroph=sources_astroph, sources_to_include=["star"]
        )
        instr.pass_through_aperture(plot=False)

        star = instr.prop_dict["star"]
        assert star["wavel"] is wavel
        post = star["flux_cube_post_screen_post_aperture_ph_sec_um"]["output_1_bright"]
        expected = 0.5 * (25.0 * u.m**2) * cube
        assert post.unit.is_equivalent(u.ph / (u.um * u.s))
        assert np.allclose(post.value, expected.value)

    def test_generate_instrument_transmission(self, transmission_config):
        # make sure sum of transmission screens is 1.0 (except for very center where
        # star is)

        Path(transmission_config["dirs"]["save_s2n_data_unique_dir"]).mkdir(
            parents=True, exist_ok=True
        )
        instr = InstrumentDepTerms(
            transmission_config,
            sources_astroph={},
            sources_to_include=[],
        )

        # ipdb.set_trace()
        screens = instr.generate_instrument_transmission(
            wavel_m=11e-6, override_stellar_mask=False, normalize=True, plot=False
        )

        # for debugging
        # import matplotlib.pyplot as plt
        # screens = instr.generate_instrument_transmission(
        #     wavel_m=11e-6, override_stellar_mask=False, normalize=True, plot=False
        # )
        # extent = [
        #     screens[5].min(), screens[5].max(),  # x [arcsec]
        #     screens[4].min(), screens[4].max(),  # y [arcsec]
        # ]
        # names = ["output_1_bright", "output_2_bright", "output_3_dark",
        # "output_4_dark"]
        # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        # for ax, i, name in zip(axes.ravel(), range(4), names):
        #     im = ax.imshow(screens[i], origin="lower", extent=extent, aspect="equal")
        #     ax.set_title(name)
        #     fig.colorbar(im, ax=ax, fraction=0.046)
        # plt.tight_layout()
        # plt.savefig(transmission_config["dirs"]["save_s2n_data_unique_dir"] +
        # "test_transmission_screens.png")
        # plt.show()  # only if running interactively, not in headless pytest
        #

        assert (
            np.max(np.any(screens[0:5, :, :], axis=0)) < 1.0 + 1e-6
        )  # transmission < 1
        net = np.sum(screens[0:4], axis=0)
        assert np.allclose(net, 1.0)

    def test_pass_through_transmission_screens_multiplies_scene_by_each_output(
        self, transmission_config
    ):
        wavel = np.array([1.0, 2.0]) * u.um
        scene = np.zeros((2, 1001, 1001)) * u.ph / (u.um * u.m**2 * u.s)
        scene[:, 500:600, 200:300] = 6.0 * u.ph / (u.um * u.m**2 * u.s)  # the 'star'
        Path(transmission_config["dirs"]["save_s2n_data_unique_dir"]).mkdir(
            parents=True, exist_ok=True
        )

        sources_astroph = {
            "star": {
                "wavel": wavel,
                "pre_screen_astro_flux_ph_sec_m2_um": np.ones(2)
                * u.ph
                / (u.um * u.m**2 * u.s),
            }
        }

        instr = InstrumentDepTerms(
            transmission_config,
            sources_astroph=sources_astroph,
            sources_to_include=["star"],
        )

        screens = instr.generate_instrument_transmission(
            wavel_m=11e-6, override_stellar_mask=False, normalize=True, plot=False
        )

        instr.pass_through_transmission_screens(
            fyi_angle=0.0,
            source_dict_pre_screen={"star": scene},
            transmission_screens=screens,
            plot=False,
        )

        bright_1 = instr.sources_astroph["star"]["flux_cube_post_screen_ph_sec_m2_um"][
            "output_1_bright"
        ]
        bright_2 = instr.sources_astroph["star"]["flux_cube_post_screen_ph_sec_m2_um"][
            "output_2_bright"
        ]
        dark_1 = instr.sources_astroph["star"]["flux_cube_post_screen_ph_sec_m2_um"][
            "output_3_dark"
        ]
        dark_2 = instr.sources_astroph["star"]["flux_cube_post_screen_ph_sec_m2_um"][
            "output_4_dark"
        ]

        # if np.logical_or(
        #     np.round(test_flux_1, 1) !=
        # np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1),
        #     np.round(test_flux_2, 1) !=
        # np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1)
        # ):
        #

        net_flux = bright_1 + bright_2 + dark_1 + dark_2

        # for debugging
        # import matplotlib.pyplot as plt
        # wavel_idx = 0
        # panels = [
        #     ("scene", scene),
        #     ("output_1_bright", bright_1),
        #     ("output_2_bright", bright_2),
        #     ("output_3_dark", dark_1),
        #     ("output_4_dark", dark_2),
        #     ("net_flux", net_flux),
        # ]
        # fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        # for ax, (name, cube) in zip(axes.ravel(), panels):
        #     im = ax.imshow(cube[wavel_idx, :, :].value, origin="lower")
        #     ax.set_title(name)
        #     fig.colorbar(im, ax=ax, fraction=0.046)
        # fig.suptitle(f"post-screen flux (wavel index {wavel_idx})")
        # plt.tight_layout()
        # out_dir = transmission_config["dirs"]["save_s2n_data_unique_dir"]
        # plt.show()
        # #plt.savefig(out_dir + "test_post_screen_flux_panels.png")
        # #plt.close(fig)
        #

        # check flux conservation
        np.testing.assert_allclose(net_flux.value, scene.value)


class TestGenerateInstrumentTransmission:
    @patch("modules.core.instrumental.transmission.fits.writeto")
    def test_returns_six_slice_cube_and_writes_fits(
        self, mock_writeto, transmission_config
    ):
        instr = InstrumentDepTerms(
            transmission_config,
            sources_astroph={},
            sources_to_include=[],
        )

        cube = instr.generate_instrument_transmission(wavel_m=11e-6, plot=False)

        n_pix = 1001
        assert cube.shape == (6, n_pix, n_pix)
        assert np.all(cube[0] >= 0.0)
        assert np.max(cube[0]) <= 1.0 + 1e-12
        assert np.any(cube[4] != 0.0)
        assert np.any(cube[5] != 0.0)
        assert mock_writeto.call_count == 0

    @patch("modules.core.instrumental.transmission.fits.writeto")
    def test_writes_fits_when_plot_true(self, mock_writeto, transmission_config):
        instr = InstrumentDepTerms(
            transmission_config,
            sources_astroph={},
            sources_to_include=[],
        )

        instr.generate_instrument_transmission(wavel_m=11e-6, plot=True)

        assert mock_writeto.call_count >= 5  # 4 outputs + differential dark

    @patch("modules.core.instrumental.transmission.fits.writeto")
    def test_override_stellar_mask_reduces_center_transmission(
        self, _mock_writeto, transmission_config
    ):
        instr = InstrumentDepTerms(
            transmission_config,
            sources_astroph={},
            sources_to_include=[],
        )

        cube = instr.generate_instrument_transmission(
            wavel_m=11e-6, override_stellar_mask=True, plot=False
        )
        center = cube.shape[-1] // 2
        # Cube order: 0/1 bright, 2/3 dark — stellar mask applies to darks only.
        assert cube[0, center, center] == pytest.approx(1.0, rel=1e-3)
        # assert cube[1, center, center] == pytest.approx(1.0, rel=1e-3)
        assert cube[2, center, center] == pytest.approx(0.001, abs=1e-3)
        assert cube[3, center, center] == pytest.approx(0.001, abs=1e-3)


class TestDisperseAstroSignalsOnDetector:
    @patch("modules.core.instrumental.dispersion.DispersionLaw")
    @patch("modules.core.instrumental.dispersion.Detector")
    def test_disperse_populates_astroph_signal_per_output_channel(
        self, mock_detector_cls, mock_law_cls, disperse_config
    ):
        n_bins = 3
        footprint = np.ones((n_bins, 5, 5))
        n_pix_per_bin = np.sum(footprint, axis=(1, 2)) * u.pix

        mock_det = MagicMock()
        mock_det.side_length_pix = 5
        mock_det.num_wavel_bins = n_bins
        mock_det.footprint_cube = footprint
        mock_det.apply_qe.side_effect = lambda x: x * 0.8
        mock_detector_cls.return_value = mock_det

        mock_law = MagicMock()
        mock_law.make_footprint.return_value = footprint
        mock_law_cls.return_value = mock_law
        mock_law_cls.n_pix_per_bin.return_value = n_pix_per_bin

        wavel = np.array([5.0, 10.0, 15.0]) * u.um
        post_cube = np.array([[[10.0]], [[20.0]], [[30.0]]]) * u.ph / (u.um * u.s)
        post_cube = np.broadcast_to(post_cube, (3, 5, 5)).copy()

        instr = InstrumentDepTerms(
            disperse_config,
            sources_astroph={"star": {"wavel": wavel}},
            sources_to_include=["star"],
        )
        instr.prop_dict["star"] = {
            "wavel": wavel,
            "flux_cube_post_screen_post_aperture_ph_sec_um": _output_flux_dict(
                post_cube
            ),
        }
        for channel in instr.output_channels.values():
            channel.bin_centers = channel.bin_centers[:n_bins]
            channel.bin_widths = channel.bin_widths[:n_bins]

        instr.disperse_astro_signals_on_detector(plot=False)

        channel = instr.output_channels["output_1_bright"]
        assert "star" in channel.astroph_signal
        sig = channel.astroph_signal["star"]
        assert len(sig["flux_astro_1d_interpolated_ph_sec_um"]) == n_bins
        assert sig["flux_astro_1d_interpolated_ph_sec_um"].unit.is_equivalent(
            u.ph / (u.um * u.s)
        )
        expected_pixel = (
            sig["flux_astro_1d_interpolated_ph_sec_wavel_bin"][0] / n_pix_per_bin[0]
        )
        assert np.isclose(
            sig["flux_astro_1d_interpolated_ph_sec_pixel"][0].value,
            expected_pixel.value,
        )
        assert mock_law.make_footprint.call_count == 4
        assert mock_det.set_footprint.call_count == 4


class TestDetector:
    def test_init_parses_detector_geometry(self, detector_geometry_config):
        det = Detector(config=detector_geometry_config, num_wavel_bins=17)
        assert det.side_length_pix == 400
        assert det.pitch_pix == 25.0
        assert det.num_wavel_bins == 17
        assert det.quantum_efficiency == 0.8


class TestDispersionLaw:
    def test_make_footprint_stores_expected_cube(self, detector_geometry_config):
        law = DispersionLaw(detector_geometry_config)
        det = Detector(config=detector_geometry_config, num_wavel_bins=3)
        cube = law.make_footprint(det.side_length_pix, det.num_wavel_bins)
        det.set_footprint(cube)

        assert cube.shape == (3, 400, 400)
        assert cube.dtype == float
        assert np.min(cube) >= 0.0
        assert np.max(cube) <= 1.0

        expected_sum = law.pix_spectral_width * law.pix_per_wavel_bin
        for wbin in range(det.num_wavel_bins):
            assert np.isclose(cube[wbin].sum(), expected_sum)

        rows = slice(100, 100 + law.pix_spectral_width)
        assert np.allclose(cube[0, rows, 300], 1.0)
        assert np.allclose(cube[0, rows, 301], 1.0)
        assert np.allclose(cube[0, rows, 302], 0.2)

    # pix_per_wavel_bin < 1 is excluded: bins inside a single pixel are mis-weighted
    # by both methods (differently)
    @pytest.mark.parametrize("pix_per_wavel_bin", ["2.2", "1.0", "1.3", "3.7"])
    def test_sparse_footprint_matches_dense(
        self, detector_geometry_config, pix_per_wavel_bin
    ):
        config = {
            **detector_geometry_config,
            "detector": {
                **detector_geometry_config["detector"],
                "pix_per_wavel_bin": pix_per_wavel_bin,
            },
        }
        law = DispersionLaw(config)
        side, n_bins = 400, 17
        dense = law.make_footprint(side, n_bins)
        sparse = _make_footprint_sparse(law, side, n_bins)

        assert np.allclose(sparse.toarray().reshape(n_bins, side, side), dense)
        assert u.allclose(
            np.asarray(sparse.sum(axis=1)).ravel() * u.pix,
            DispersionLaw.n_pix_per_bin(dense),
        )
        assert np.allclose(
            np.asarray(sparse.sum(axis=0)).reshape(side, side),
            DispersionLaw.footprint_image(dense),
        )

        rng = np.random.default_rng(0)
        image = rng.normal(size=(side, side))
        assert np.allclose(
            sparse @ image.ravel(),
            DispersionLaw.weighted_sum_per_bin(dense, image),
        )


def _box_sum(det, image):
    """Per-bin box sum of a 2D image, as in extract_spectra_from_readouts."""
    bin_map = det.bin_index_map()
    on_footprint = bin_map >= 0
    return np.bincount(
        bin_map[on_footprint],
        weights=image[on_footprint],
        minlength=det.num_wavel_bins,
    )


class _ScaleEffect(DetectorEffect):
    """Toy multiplicative effect."""

    name = "toy_scale"

    def apply(self, frames, readout_index):
        for name in frames.source_e:
            frames.source_e[name] *= 2.0
        frames.systematics_e *= 2.0


class _RecordTotalEffect(DetectorEffect):
    """Toy light-dependent effect: remembers the previous readout's total charge."""

    name = "toy_record"

    def __init__(self, config):
        super().__init__(config)
        self.previous_total = None
        self.seen = []

    def apply(self, frames, readout_index):
        self.seen.append((readout_index, self.previous_total))
        self.previous_total = frames.total_e.sum()

    def reset(self):
        self.previous_total = None


class TestDetectorEffects:
    @pytest.fixture
    def channel_with_signal(self, detector_geometry_config):
        """One channel with a footprint and equal per-bin star signal."""
        channel = OutputChannel(name="output_3_dark")
        n_bins = 3
        det = _attach_detector_footprint(
            channel, detector_geometry_config, n_bins=n_bins
        )
        # equal signal in every bin
        channel.astroph_signal["star"] = {
            "flux_astro_1d_interpolated_ph_sec_pixel": np.full(n_bins, 5.0)
            * u.ph
            / (u.s * u.pix)
        }
        return channel, det

    def test_expose_paints_per_bin_signal_onto_footprint(self, channel_with_signal):
        channel, det = channel_with_signal
        frames = det.expose(channel.astroph_signal, 10.0 * u.s)

        star = frames.source_e["star"]
        # equal per-pixel signal in every bin: frame = s * t * (footprint summed over
        # bins)
        assert np.allclose(
            star, 5.0 * 10.0 * DispersionLaw.footprint_image(det.footprint_cube)
        )
        npix = DispersionLaw.n_pix_per_bin(det.footprint_cube).value
        assert np.isclose(star.sum(), np.sum(npix) * 5.0 * 10.0)
        assert np.all(frames.systematics_e == 0.0)

    def test_box_sum_conserves_total_for_unequal_bins(self, channel_with_signal):
        channel, det = channel_with_signal
        channel.astroph_signal["star"]["flux_astro_1d_interpolated_ph_sec_pixel"] = (
            np.array([1.0, 4.0, 9.0]) * u.ph / (u.s * u.pix)
        )
        frames = det.expose(channel.astroph_signal, 1.0 * u.s)
        spectrum = _box_sum(det, frames.source_e["star"])
        npix = DispersionLaw.n_pix_per_bin(det.footprint_cube).value
        assert np.isclose(spectrum.sum(), np.sum(npix * [1.0, 4.0, 9.0]))

    @patch("modules.core.instrumental.effects.fits.getdata")
    def test_additive_map_lands_only_in_systematics(
        self, mock_getdata, channel_with_signal
    ):
        channel, det = channel_with_signal
        hot_map = np.zeros((400, 400))
        # only under wavelength bin 0 footprint (cols 300-301), not bin 1 (starts
        # ~302.2)
        hot_map[100:103, 300:302] = 2.0
        mock_getdata.return_value = hot_map
        config = {
            "detector_systematics": {
                "enabled": "hot_pixels",
                "hot_pixels_2d_file": "/tmp/hot.fits",
            }
        }
        effects = build_detector_effects(config)
        assert [type(e) for e in effects] == [HotPixelsEffect]
        mock_getdata.assert_called_once_with("/tmp/hot.fits")

        frames = det.expose(channel.astroph_signal, 1.0 * u.s)
        star_before = frames.source_e["star"].copy()
        effects[0].apply(frames, 0)

        assert np.array_equal(frames.source_e["star"], star_before)
        assert np.array_equal(frames.systematics_e, hot_map)
        systematics = _box_sum(det, frames.systematics_e)
        assert np.isclose(systematics[0], hot_map.sum())
        assert np.allclose(systematics[1:], 0.0)

    def test_multiplicative_effect_scales_sources_and_systematics(self):
        frames = DetectorFrames(
            source_e={"star": np.ones((2, 2))}, systematics_e=np.full((2, 2), 3.0)
        )
        _ScaleEffect({}).apply(frames, 0)
        assert np.all(frames.source_e["star"] == 2.0)
        assert np.all(frames.systematics_e == 6.0)
        assert np.all(frames.total_e == 8.0)

    def test_light_dependent_effect_sees_total_and_is_cleared_by_reset(
        self, channel_with_signal
    ):
        channel, det = channel_with_signal
        effect = _RecordTotalEffect({})
        apply_detector_effects(
            {channel.name: channel}, [effect], t_frame=1.0 * u.s, n_readouts=2
        )
        total = det.expose(channel.astroph_signal, 1.0 * u.s).total_e.sum()
        assert effect.seen[0] == (0, None)
        assert effect.seen[1][0] == 1
        assert np.isclose(effect.seen[1][1], total)

        effect.reset()
        assert effect.previous_total is None

    def test_apply_detector_effects_stores_one_frame_set_per_readout(
        self, channel_with_signal
    ):
        channel, det = channel_with_signal
        apply_detector_effects(
            {channel.name: channel}, [], t_frame=1.0 * u.s, n_readouts=3
        )
        expected = det.expose(channel.astroph_signal, 1.0 * u.s)
        assert len(channel.readout_frames) == 3
        for frames in channel.readout_frames:
            assert frames.channel_name == channel.name
            assert np.array_equal(frames.source_e["star"], expected.source_e["star"])

    @patch("modules.core.instrumental.effects.fits.getdata")
    def test_build_detector_effects_keeps_order_and_skips_noise_params(
        self, mock_getdata, caplog
    ):
        mock_getdata.return_value = np.zeros((4, 4))
        config = {
            "detector_systematics": {
                "enabled": "persistence, readnoise, hot_pixels, dark_current, "
                "gain_variability",
                "hot_pixels_2d_file": "/tmp/hot.fits",
            }
        }
        effects = build_detector_effects(config)
        assert [e.name for e in effects] == [
            "persistence",
            "hot_pixels",
            "gain_variability",
        ]
        assert "readnoise" in caplog.text
        assert "dark_current" in caplog.text

    def test_build_detector_effects_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="not found"):
            build_detector_effects({"detector_systematics": {"enabled": "nope"}})

    def test_build_detector_effects_empty_by_default(self):
        effects = build_detector_effects({})
        assert isinstance(effects, list) and not effects


class TestDetectorReadouts:
    @pytest.fixture
    def channels_with_frames(self, detector_geometry_config):
        """Four channels with footprints and readout frames already applied."""
        config = {"detector": {**detector_geometry_config["detector"], "gain": "2.0"}}
        channels = {}
        for i, name in enumerate(
            ["output_1_bright", "output_2_bright", "output_3_dark", "output_4_dark"]
        ):
            channel = OutputChannel(name=name)
            _attach_detector_footprint(channel, config, n_bins=3)
            channel.astroph_signal["star"] = {
                "flux_astro_1d_interpolated_ph_sec_pixel": np.array([1.0, 4.0, 9.0])
                * (i + 1)
                * u.ph
                / (u.s * u.pix)
            }
            channels[name] = channel
        apply_detector_effects(channels, [], t_frame=10.0 * u.s, n_readouts=2)
        return channels

    def test_bin_index_map_assigns_shared_pixel_to_majority_bin(
        self, detector_geometry_config
    ):
        channel = OutputChannel(name="x")
        det = _attach_detector_footprint(channel, detector_geometry_config, n_bins=3)
        bin_map = det.bin_index_map()
        # bin 0 covers cols 300-301 fully and 0.2 of col 302; bin 1 covers 0.8 of col
        # 302
        assert np.all(bin_map[100:103, 300:302] == 0)
        assert np.all(bin_map[100:103, 302] == 1)
        assert np.all(bin_map[:100] == -1)
        assert np.all(bin_map[:, :299] == -1)

    def test_write_full_frame_produces_one_hdu_per_channel(
        self, channels_with_frames, tmp_path
    ):
        file_name = str(tmp_path / "readouts.fits")
        write_detector_readouts(
            channels_with_frames,
            file_name,
            t_frame=10.0 * u.s,
            header_cards={"QE": 0.8},
            cutout=False,
        )
        with fits.open(file_name) as hdul:
            assert [h.name for h in hdul[1:]] == [
                "CH1BR",
                "CH2BR",
                "CH3DK",
                "CH4DK",
            ]
            assert hdul[0].header["TFRAME"] == 10.0
            assert hdul[0].header["QE"] == 0.8
            hdu = hdul["CH3DK"]
            assert hdu.data.shape == (2, 2, 400, 400)  # (components, readouts, y, x)
            assert (hdu.header["Y0"], hdu.header["X0"]) == (0, 0)
            assert hdu.header["BUNIT"] == "adu"
            assert [hdu.header["COMP0"], hdu.header["COMP1"]] == [
                "star",
                SYSTEMATICS_KEY,
            ]
            frames = channels_with_frames["output_3_dark"].readout_frames[0]
            assert np.allclose(hdu.data[0, 0], frames.source_e["star"] / 2.0)

    def test_write_cutout_keeps_only_trace_bbox(self, channels_with_frames, tmp_path):
        file_name = str(tmp_path / "readouts_cutout.fits")
        write_detector_readouts(channels_with_frames, file_name, t_frame=10.0 * u.s)
        channel = channels_with_frames["output_3_dark"]
        y0, y1, x0, x1 = channel.detector.footprint_bbox()
        # 3 rows from y=100; bins span x = 300 .. 306.6
        assert (y0, y1, x0, x1) == (100, 103, 300, 307)
        with fits.open(file_name) as hdul:
            hdu = hdul["CH3DK"]
            assert hdu.data.shape == (2, 2, 3, 7)
            assert (hdu.header["Y0"], hdu.header["X0"]) == (100, 300)
            assert (hdu.header["FULLNY"], hdu.header["FULLNX"]) == (400, 400)
            full = channel.readout_frames[0].source_e["star"] / 2.0
            assert np.allclose(hdu.data[0, 0], full[y0:y1, x0:x1])
            # nothing illuminated outside the cutout
            assert np.isclose(hdu.data[0, 0].sum(), full.sum())

    @pytest.mark.parametrize("method", ["sharp_birchall", "box_sum"])
    def test_extract_from_cutout_matches_full_frame(
        self, channels_with_frames, tmp_path, method
    ):
        cutout_file = str(tmp_path / "cutout.fits")
        full_file = str(tmp_path / "full.fits")
        write_detector_readouts(channels_with_frames, cutout_file, t_frame=10.0 * u.s)
        write_detector_readouts(
            channels_with_frames, full_file, t_frame=10.0 * u.s, cutout=False
        )
        kwargs = {"method": method, "t_frame": 10.0 * u.s, "dc_rates": [0.1]}
        from_full = extract_spectra_from_readouts(
            channels_with_frames, full_file, **kwargs
        )
        errors_full = {
            n: ch.extracted_error_adu for n, ch in channels_with_frames.items()
        }
        from_cutout = extract_spectra_from_readouts(
            channels_with_frames, cutout_file, **kwargs
        )
        for ch_name, spectra in from_full.items():
            for comp, spectrum in spectra.items():
                assert u.allclose(from_cutout[ch_name][comp], spectrum)
            if method == "sharp_birchall":
                assert u.allclose(
                    channels_with_frames[ch_name].extracted_error_adu,
                    errors_full[ch_name],
                )

    @staticmethod
    def _ramp_channels(detector_geometry_config, read_noise="6.0"):
        config = {
            "detector": {
                **detector_geometry_config["detector"],
                "gain": "2.0",
                "read_noise": read_noise,
            }
        }
        channels = {}
        for name in [
            "output_1_bright",
            "output_2_bright",
            "output_3_dark",
            "output_4_dark",
        ]:
            channel = OutputChannel(name=name)
            _attach_detector_footprint(channel, config, n_bins=3)
            channel.astroph_signal["star"] = {
                "flux_astro_1d_interpolated_ph_sec_pixel": np.array([1.0, 4.0, 9.0])
                * u.ph
                / (u.s * u.pix)
            }
            channels[name] = channel
        # systematics grow with readout index, so readouts differ
        apply_detector_effects(
            channels, [_ReadoutRampEffect({})], t_frame=10.0 * u.s, n_readouts=3
        )
        return channels

    def test_box_sum_keeps_readout_axis_and_conserves_total(
        self, detector_geometry_config, tmp_path
    ):
        channels = self._ramp_channels(detector_geometry_config)
        file_name = str(tmp_path / "readouts.fits")
        write_detector_readouts(channels, file_name, t_frame=10.0 * u.s)
        extracted = extract_spectra_from_readouts(channels, file_name, method="box_sum")

        channel = channels["output_3_dark"]
        star = extracted["output_3_dark"]["star"]
        systematics = extracted["output_3_dark"][SYSTEMATICS_KEY]
        assert star.unit == u.adu
        assert star.shape == (1, 3, 3)  # (n_dc, n_readouts, n_bins)
        assert channel.extracted_spectra_adu["star"] is star
        assert channel.extracted_error_adu is None
        bin_map = channel.detector.bin_index_map()
        for i, frames in enumerate(channel.readout_frames):
            image_adu = frames.source_e["star"] / 2.0
            assert np.isclose(star.value[0, i].sum(), image_adu.sum())
            # box sum: each bin is the sum of the pixels assigned to it
            for b in range(3):
                assert np.isclose(star.value[0, i, b], image_adu[bin_map == b].sum())
            assert np.isclose(
                systematics.value[0, i].sum(),
                (frames.systematics_e[bin_map >= 0] / 2.0).sum(),
            )
        assert systematics.value[0, 0].sum() < systematics.value[0, 2].sum()

    def test_sharp_birchall_conserves_totals_and_gives_errors_per_dc(
        self, detector_geometry_config, tmp_path, caplog
    ):
        channels = self._ramp_channels(detector_geometry_config, read_noise="6.0")
        file_name = str(tmp_path / "readouts.fits")
        write_detector_readouts(channels, file_name, t_frame=10.0 * u.s)
        dc_rates = [0.1, 0.5]
        with caplog.at_level("WARNING"):
            extracted = extract_spectra_from_readouts(
                channels, file_name, t_frame=10.0 * u.s, dc_rates=dc_rates
            )
        assert "modelled from noiseless readout frames" in caplog.text

        channel = channels["output_3_dark"]
        det = channel.detector
        star = extracted["output_3_dark"]["star"]
        systematics = extracted["output_3_dark"][SYSTEMATICS_KEY]
        error = channel.extracted_error_adu
        assert star.shape == systematics.shape == error.shape == (2, 3, 3)
        assert error.unit == u.adu
        assert channel.extracted_vark_adu2.shape == (2, 3, 3)

        y0, y1, x0, x1 = det.footprint_bbox()
        fp = det.footprint_cube[:, y0:y1, x0:x1]
        image = fp.sum(axis=0)
        fraction = fp.sum(axis=1) / image.sum(axis=0)  # (n_bins, nx)
        on_trace = image > 0
        gain, t, rn = 2.0, 10.0, 6.0
        for i_dc, dc in enumerate(dc_rates):
            for i, frames in enumerate(channel.readout_frames):
                star_adu = frames.source_e["star"][y0:y1, x0:x1] / gain
                sys_adu = frames.systematics_e[y0:y1, x0:x1] / gain
                # flux conserved, and split between bins by fractional column coverage
                assert np.isclose(star.value[i_dc, i].sum(), star_adu[on_trace].sum())
                assert np.allclose(star.value[i_dc, i], fraction @ star_adu.sum(axis=0))
                assert np.isclose(
                    systematics.value[i_dc, i].sum(), sys_adu[on_trace].sum()
                )
                # error^2 = sum_x f^2 Var(column), Var(column) = 1 / sum_y (phi^2 /
                # sigma^2)
                total_e = (star_adu + sys_adu) * gain
                sigma2 = (total_e + dc * t) / gain**2 + (rn / gain) ** 2
                phi = image / image.sum(axis=0)
                col_var = 1.0 / np.sum(phi**2 / sigma2, axis=0)
                assert np.allclose(error.value[i_dc, i] ** 2, (fraction**2) @ col_var)
        # more dark current, larger errors; same flux
        assert np.all(error.value[1] > error.value[0])
        assert u.allclose(star[0], star[1])

    def test_unknown_extraction_method_raises(self, channels_with_frames, tmp_path):
        file_name = str(tmp_path / "readouts.fits")
        write_detector_readouts(channels_with_frames, file_name, t_frame=10.0 * u.s)
        with pytest.raises(ValueError, match="Unknown extraction method"):
            extract_spectra_from_readouts(
                channels_with_frames, file_name, method="nope"
            )


class _ReadoutRampEffect(DetectorEffect):
    """Toy additive effect: adds (readout_index + 1) e- to every pixel."""

    name = "toy_ramp"

    def apply(self, frames, readout_index):
        frames.systematics_e += readout_index + 1.0
