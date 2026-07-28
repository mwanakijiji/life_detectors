"""
Unit tests for S/N cube assembly from HDF5 pipeline outputs.
"""

import pickle
import re
import sys
import types
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

# Mock ipdb before importing project modules (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.core.calculator.s2n_cube import (
    S2NCube,
    load_s2n_cube,
    read_s2n_cube_hdf5,
    save_s2n_cube,
)
from modules.core.calculator.snr_from_hdf5 import calculate_s2n_post_rotation


PLANET_COL = "astro_exoplanet_model_10pc_adu"
CHOPPED_PLANET_COL = f"chopped_{PLANET_COL}"
STAR_COL_OUT3 = "astro_star_adu"
STAR_COL_3 = "output_3_dark_astro_star_adu"
STAR_COL_4 = "output_4_dark_astro_star_adu"


def _make_wavelength_meta(n_bins: int = 1):
    centers = np.linspace(10.0, 10.0 + n_bins - 1, n_bins) * u.um
    widths = np.ones(n_bins) * u.um
    edges = np.empty(n_bins + 1) * u.um
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * widths[0]
    edges[-1] = centers[-1] + 0.5 * widths[-1]
    return centers, widths, edges


def _base_table(n_bins: int = 1) -> QTable:
    centers, widths, edges = _make_wavelength_meta(n_bins)
    tbl = QTable()
    tbl["bin"] = np.arange(n_bins)
    tbl["center"] = centers
    tbl["width"] = widths
    tbl["npix"] = np.full(n_bins, 100) * u.pix
    tbl.meta["wavel_bin_edges"] = edges
    tbl.meta["qe"] = 0.70
    tbl.meta["angle_deg"] = 0.0
    tbl.meta["dark_current_e_pix_s"] = 1e-4
    return tbl


def _write_angle_hdf5(
    path,
    *,
    angle_deg: float,
    dc_rate: float = 1e-4,
    qe: float = 0.70,
    planet_chopped,
    planet_out3,
    star_out3=None,
    star_out4=None,
    instrum_dark=0.5,
    instrum_read=0.3,
    n_bins: int = 1,
):
    """Write one angle_*.hdf5 file matching record_info_at_angle_and_qe layout."""
    dc_qe_str = f"dc_{dc_rate:06.3f}_qe_{qe:04.2f}"
    star_out3 = star_out3 if star_out3 is not None else np.zeros(n_bins)
    star_out4 = star_out4 if star_out4 is not None else np.zeros(n_bins)

    def _write_table(tbl, dataset_name, overwrite=False):
        tbl.meta["angle_deg"] = float(angle_deg)
        tbl.meta["qe"] = float(qe)
        tbl.write(
            path,
            path=f"{dc_qe_str}/{dataset_name}",
            serialize_meta=True,
            overwrite=overwrite,
            append=not overwrite,
        )

    first = True
    for ch_name in (
        "output_1_bright",
        "output_2_bright",
        "output_3_dark",
        "output_4_dark",
    ):
        tbl = _base_table(n_bins)
        if ch_name == "output_3_dark":
            tbl[PLANET_COL] = np.asarray(planet_out3) * u.adu
            tbl[STAR_COL_OUT3] = np.asarray(star_out3) * u.adu
        _write_table(tbl, ch_name, overwrite=first)
        first = False

    chopped = _base_table(n_bins)
    chopped[CHOPPED_PLANET_COL] = np.asarray(planet_chopped) * u.adu
    chopped[STAR_COL_3] = np.asarray(star_out3) * u.adu
    chopped[STAR_COL_4] = np.asarray(star_out4) * u.adu
    chopped["chopped_instrum_dc_rms_adu"] = (
        np.full(n_bins, instrum_dark) * u.adu
    )
    chopped["chopped_instrum_rn_rms_adu"] = (
        np.full(n_bins, instrum_read) * u.adu
    )
    _write_table(chopped, "chopped", overwrite=False)


def _expected_snr_lambda(
    *,
    planet_chopped_by_angle,
    planet_out3_by_angle,
    star_out3_by_angle,
    star_out4_by_angle=None,
    gain,
    instrum_dark,
    instrum_read,
):
    """Mirror _compute_snr_lambda_for_slot SNR math (Dannert+ 2022 Eqn. 19-20)."""
    del star_out4_by_angle  # not used in symmetric-noise term; kept for call-site clarity

    gain_val = float(gain)
    angles = sorted(planet_chopped_by_angle.keys())
    n_bins = len(next(iter(planet_chopped_by_angle.values())))

    cols_S_p_elec = [np.asarray(planet_chopped_by_angle[a]) * gain_val for a in angles]
    cols_S_p_3_elec = [np.asarray(planet_out3_by_angle[a]) * gain_val for a in angles]
    S_p_sqd_arr_mean_elec = np.mean(np.power(np.column_stack(cols_S_p_elec), 2), axis=1)
    S_p_3_sqd_arr_mean_elec = np.mean(np.power(np.column_stack(cols_S_p_3_elec), 2), axis=1)

    S_sym_noise_var_3_elec = np.zeros(n_bins, dtype=float)
    if star_out3_by_angle is not None:
        cols_sym_noise_var_3_elec = []
        for a in angles:
            sym_flux_elec = np.asarray(star_out3_by_angle[a]) * gain_val
            cols_sym_noise_var_3_elec.append(np.sqrt(sym_flux_elec))
        S_sym_noise_var_3_elec = np.mean(np.column_stack(cols_sym_noise_var_3_elec), axis=1)

    instrum_dark = np.asarray(instrum_dark, dtype=float)
    instrum_read = np.asarray(instrum_read, dtype=float)

    snr_bins = []
    for wavel_bin_num in range(n_bins):
        S_p_rms_phi = np.sqrt(S_p_sqd_arr_mean_elec[wavel_bin_num])
        S_p_3_rms_phi = np.sqrt(S_p_3_sqd_arr_mean_elec[wavel_bin_num])

        S_dark_noise_var = (instrum_dark[wavel_bin_num] * gain_val) ** 2
        S_read_noise_var = (instrum_read[wavel_bin_num] * gain_val) ** 2

        S_sym_3_var_this = S_sym_noise_var_3_elec[wavel_bin_num]
        astro_noise_term = 2 * (S_sym_3_var_this + S_p_3_rms_phi)
        instrum_noise_term = 2 * (S_dark_noise_var + S_read_noise_var)
        denominator = np.sqrt(astro_noise_term + instrum_noise_term)
        snr_bins.append(S_p_rms_phi / denominator)

    snr_bins = np.asarray(snr_bins)
    return snr_bins, float(np.sqrt(np.sum(np.power(snr_bins, 2))))


@pytest.fixture
def s2n_config(tmp_path):
    return {
        "dirs": {"save_s2n_data_unique_dir": str(tmp_path / "out") + "/"},
        "detector": {"gain": "2.0", "quantum_efficiency": "0.7"},
        "observation": {
            "t_int_frame": "10",
            "N_angles": "2",
            "N_int_per_angle": "1",
        },
        "plotting": {"title_context": ""},
    }


@pytest.fixture
def patch_plotting():
    with patch("modules.viz.snr.plt.savefig"), patch(
        "modules.viz.snr.plt.figure"
    ), patch("modules.viz.snr.plt.clf"), patch(
        "modules.viz.snr.plt.stairs"
    ), patch(
        "modules.viz.snr.plt.tight_layout"
    ), patch(
        "modules.viz.snr.plt.close"
    ):
        yield


class TestCalculateS2nPostRotation:
    def test_snr_matches_reference_implementation(
        self, tmp_path, s2n_config, patch_plotting, capsys
    ):
        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()

        _write_angle_hdf5(
            read_dir / "angle_0.hdf5",
            angle_deg=0.0,
            planet_chopped=[4.0],
            planet_out3=[3.0],
            star_out3=[3.0],
            star_out4=[1.0],
        )
        _write_angle_hdf5(
            read_dir / "angle_90.hdf5",
            angle_deg=90.0,
            planet_chopped=[-5.0],
            planet_out3=[1.0],
            star_out3=[3.0],
            star_out4=[1.0],
        )

        gain = float(s2n_config["detector"]["gain"])
        # note the inputs here are in photoelectrons, not ADU
        _, pipeline_s2n_tot = _expected_snr_lambda(
            planet_chopped_by_angle={0.0: [4.0], 90.0: [-5.0]},
            planet_out3_by_angle={0.0: [3.0], 90.0: [1.0]},
            star_out3_by_angle={0.0: [3.0], 90.0: [3.0]},
            star_out4_by_angle={0.0: [2.0], 90.0: [2.0]},
            gain=gain,
            instrum_dark=np.array([0.5]),
            instrum_read=np.array([0.3]),
        )

        cube_test = calculate_s2n_post_rotation(str(read_dir), config=s2n_config)

        captured = capsys.readouterr().out
        match = re.search(r"SNR_tot for DC dc_00\.000_qe_0\.70: ([0-9.]+)", captured)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(pipeline_s2n_tot)

        # by hand (mirror _compute_snr_lambda_for_slot unit conventions)
        g = gain * u.electron / u.adu
        planet_chopped_adu = np.array([4, -5]) * u.adu
        planet_out3_adu = np.array([3, 1]) * u.adu
        star_out3_adu = np.array([3, 3]) * u.adu
        dark_rms_adu = 0.5 * u.adu
        read_rms_adu = 0.3 * u.adu

        S_p_rms = np.sqrt(np.mean(np.power(planet_chopped_adu * g, 2)))
        S_p3_rms = np.sqrt(np.mean(np.power(planet_out3_adu * g, 2)))
        S_sym = np.mean(np.sqrt((star_out3_adu * g).to_value(u.electron))) * u.electron
        astro_noise_term = 2 * (S_sym + S_p3_rms)

        S_dark_noise_var = np.power(dark_rms_adu * g, 2).value * u.electron
        S_read_noise_var = np.power(read_rms_adu * g, 2).value * u.electron
        instrum_noise_term = 2 * (S_dark_noise_var + S_read_noise_var)

        denominator = np.sqrt((astro_noise_term + instrum_noise_term).value) * u.electron
        manual_s2n_tot = (S_p_rms / denominator).decompose().value

        assert pipeline_s2n_tot == pytest.approx(manual_s2n_tot)

    def test_symmetric_noise_uses_all_angles_not_last_only(
        self, tmp_path, s2n_config, patch_plotting, capsys
    ):
        """Regression: star shot noise should use angle-averaged sigma^2, not last file only."""
        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()

        _write_angle_hdf5(
            read_dir / "angle_0.hdf5",
            angle_deg=0.0,
            planet_chopped=[1.0],
            planet_out3=[1.0],
            star_out3=[4.0],
            star_out4=[4.0],
        )
        _write_angle_hdf5(
            read_dir / "angle_90.hdf5",
            angle_deg=90.0,
            planet_chopped=[1.0],
            planet_out3=[1.0],
            star_out3=[0.0],
            star_out4=[0.0],
        )

        gain = float(s2n_config["detector"]["gain"])
        expected_with_avg, _ = _expected_snr_lambda(
            planet_chopped_by_angle={0.0: [1.0], 90.0: [1.0]},
            planet_out3_by_angle={0.0: [1.0], 90.0: [1.0]},
            star_out3_by_angle={0.0: [4.0], 90.0: [0.0]},
            star_out4_by_angle={0.0: [4.0], 90.0: [0.0]},
            gain=gain,
            instrum_dark=np.array([0.5]),
            instrum_read=np.array([0.3]),
        )
        expected_last_only, _ = _expected_snr_lambda(
            planet_chopped_by_angle={0.0: [1.0], 90.0: [1.0]},
            planet_out3_by_angle={0.0: [1.0], 90.0: [1.0]},
            star_out3_by_angle={0.0: [0.0], 90.0: [0.0]},
            star_out4_by_angle={0.0: [0.0], 90.0: [0.0]},
            gain=gain,
            instrum_dark=np.array([0.5]),
            instrum_read=np.array([0.3]),
        )

        assert expected_with_avg[0] != pytest.approx(expected_last_only[0])
        assert expected_with_avg[0] < expected_last_only[0]

        calculate_s2n_post_rotation(str(read_dir), config=s2n_config)
        captured = capsys.readouterr().out
        match = re.search(r"SNR_tot for DC dc_00\.000_qe_0\.70: ([0-9.]+)", captured)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(expected_with_avg[0])


class TestReadHdf5Slots:
    def test_angle_keys_use_meta_canonical_value(self, tmp_path):
        from modules.core.calculator.snr_from_hdf5 import read_hdf5_slots
        from modules.utils.helpers.keys import canonical_angle_deg, format_angle_qe_hdf5_name

        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()
        angle_linspace = float(np.linspace(0, 360, num=7, endpoint=False)[1])
        canonical = canonical_angle_deg(angle_linspace)

        _write_angle_hdf5(
            read_dir / format_angle_qe_hdf5_name(angle_linspace, 0.70),
            angle_deg=angle_linspace,
            planet_chopped=[1.0],
            planet_out3=[1.0],
        )

        slots = read_hdf5_slots(str(read_dir))
        slot = slots["dc_00.000_qe_0.70"]
        assert list(slot["S_p"].keys()) == [canonical]
        assert canonical != angle_linspace or angle_linspace == round(angle_linspace, 2)

    def test_read_hdf5_slots_filters_by_qe(self, tmp_path):
        from modules.core.calculator.snr_from_hdf5 import build_s2n_cube_from_hdf5, read_hdf5_slots
        from modules.utils.helpers.keys import format_angle_qe_hdf5_name

        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()

        for qe in (0.60, 0.70):
            _write_angle_hdf5(
                read_dir / format_angle_qe_hdf5_name(0.0, qe),
                angle_deg=0.0,
                qe=qe,
                planet_chopped=[1.0],
                planet_out3=[1.0],
            )

        slots_070 = read_hdf5_slots(str(read_dir), qe=0.70)
        assert set(slots_070) == {"dc_00.000_qe_0.70"}

        slots_060 = read_hdf5_slots(str(read_dir), qe=0.60)
        assert set(slots_060) == {"dc_00.000_qe_0.60"}

    def test_build_s2n_cube_uses_config_qe_not_other_angle_files(
        self, tmp_path, s2n_config, patch_plotting
    ):
        from modules.core.calculator.snr_from_hdf5 import build_s2n_cube_from_hdf5
        from modules.utils.helpers.keys import format_angle_qe_hdf5_name

        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()

        for qe in (0.60, 0.70):
            _write_angle_hdf5(
                read_dir / format_angle_qe_hdf5_name(0.0, qe),
                angle_deg=0.0,
                qe=qe,
                planet_chopped=[1.0],
                planet_out3=[1.0],
            )

        cube = build_s2n_cube_from_hdf5(str(read_dir), s2n_config)
        assert list(cube.qe) == [0.7]
        assert cube.snr.shape[2] == 1


def _make_sample_s2n_cube(read_dir: str = "/tmp/hdf5_in") -> S2NCube:
    wavelength = np.array([10.0, 11.0])
    wavel_bin_width = np.array([0.5, 0.5])
    wavel_bin_edges = np.array([9.75, 10.25, 11.25])
    dark_current = np.array([0.0, 0.1])
    qe = np.array([0.6, 0.7])
    snr = np.arange(8, dtype=float).reshape(2, 2, 2)
    snr_tot = np.array([[1.0, 2.0], [3.0, 4.0]])
    base_titles = np.array([["title_a", "title_b"], ["title_c", "title_d"]], dtype=object)
    return S2NCube(
        snr=snr,
        wavelength=wavelength,
        wavel_bin_width=wavel_bin_width,
        wavel_bin_edges=wavel_bin_edges,
        dark_current=dark_current,
        qe=qe,
        snr_tot=snr_tot,
        base_titles=base_titles,
        title_context="t_int_frame = 10.0 s",
        sources_context="star",
        read_dir=read_dir,
        n_angles=4,
        n_int_per_angle=1,
        t_int_frame=10.0,
        n_int_total=40.0,
        config={"detector": {"gain": "2.0"}, "plotting": {"title_context": ""}},
    )


class TestSaveS2nCube:
    def test_save_hdf5_roundtrip(self, tmp_path):
        cube = _make_sample_s2n_cube(read_dir=str(tmp_path / "hdf5_in"))
        out_path = tmp_path / "s2n_cube"

        saved = save_s2n_cube(cube, out_path, file_format="hdf5")

        assert saved == [str(out_path.with_suffix(".hdf5"))]
        restored = read_s2n_cube_hdf5(saved[0])
        assert np.allclose(restored.snr, cube.snr)
        assert np.allclose(restored.snr_tot, cube.snr_tot)
        assert np.allclose(restored.wavelength, cube.wavelength)
        assert np.allclose(restored.dark_current, cube.dark_current)
        assert np.allclose(restored.qe, cube.qe)
        assert restored.title_context == cube.title_context
        assert restored.read_dir == cube.read_dir
        assert restored.n_angles == cube.n_angles
        assert restored.config == cube.config

    def test_save_pickle_roundtrip(self, tmp_path):
        cube = _make_sample_s2n_cube()
        out_path = tmp_path / "s2n_cube.pkl"

        saved = save_s2n_cube(cube, out_path, file_format="pickle")

        assert saved == [str(out_path)]
        with open(saved[0], "rb") as handle:
            restored = pickle.load(handle)
        assert np.allclose(restored.snr, cube.snr)
        assert restored.base_titles.tolist() == cube.base_titles.tolist()

    def test_save_both_formats(self, tmp_path):
        cube = _make_sample_s2n_cube()
        out_path = tmp_path / "s2n_cube"

        saved = save_s2n_cube(cube, out_path, file_format="both")

        assert saved == [str(out_path.with_suffix(".pkl")), str(out_path.with_suffix(".hdf5"))]
        assert load_s2n_cube(saved[0]).snr.shape == cube.snr.shape
        assert load_s2n_cube(saved[1]).snr_tot.shape == cube.snr_tot.shape

    def test_hdf5_writes_meta_and_formatted_titles(self, tmp_path):
        cube = _make_sample_s2n_cube()
        hdf5_path = tmp_path / "s2n_cube.hdf5"
        save_s2n_cube(cube, hdf5_path, file_format="hdf5")

        with h5py.File(hdf5_path, "r") as handle:
            meta = handle["meta"]
            assert meta.attrs["axis_order"] == "wavelength, dark_current, qe"
            assert meta.attrs["n_angles"] == cube.n_angles
            assert "formatted_plot_titles" in meta
            formatted = meta["formatted_plot_titles"][:].astype(str)
            assert formatted.shape == cube.base_titles.shape
            assert formatted[0, 0].startswith("title_a")
