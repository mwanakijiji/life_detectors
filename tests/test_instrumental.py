"""
Unit tests for instrumental noise calculations.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from astropy import units as u

from modules.core.instrumental import InstrumentDepTerms, Detector
from modules.data.units import UnitConverter

class TestInstrumentDepTerms:
    """Test cases for InstrumentDepTerms class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "detector": {
                "read_noise": 5.0,
                "dark_current": 0.1,
                "gain": 2.0,
            },
            "instrumental_sources": {
                "dark_current": {
                    "enabled": True,
                },
                "read_noise": {
                    "enabled": True,
                },
            },
        }
    
    @pytest.fixture
    def unit_converter(self):
        """Create a unit converter for testing."""
        return UnitConverter()
    
    def test_init(self, mock_config, unit_converter):
        """Test initialization of InstrumentDepTerms."""
        noise_calc = InstrumentDepTerms(
            mock_config,
            unit_converter,
            sources_astroph={"star": {"wavel": np.array([1.0]), "astro_flux_ph_sec_m2_um": np.array([0.0])}},
            sources_to_include=["star"],
        )
        assert noise_calc.config == mock_config
        assert noise_calc.unit_converter == unit_converter
        assert noise_calc.sources_to_include == ["star"]
        assert isinstance(noise_calc.sources_instrum, dict)
        assert noise_calc.sources_instrum == {}
        assert isinstance(noise_calc.prop_dict, dict)
        assert noise_calc.prop_dict == {}

    def test_calculate_intrinsic_instrumental_noise_parses_arrays(self, unit_converter):
        """calculate_instrinsic_instrumental_noise populates sources_instrum with arrays and units."""
        config = {
            "detector": {
                "read_noise": "2.0, 3.0",
                "dark_current": "0.0, 0.2, 0.1",  # start, stop, step -> [0.0, 0.1]
                "gain": "4.5",
            },
            "observation": {"integration_time": "100"},
        }
        noise_calc = InstrumentDepTerms(
            config,
            unit_converter,
            sources_astroph={},
            sources_to_include=[],
        )

        noise_calc.calculate_instrinsic_instrumental_noise()

        assert "read_noise_e_pix-1" in noise_calc.sources_instrum
        assert "dark_current_e_pix-1_sec-1" in noise_calc.sources_instrum
        assert "dark_current_e_pix-1" in noise_calc.sources_instrum

        read_noise = noise_calc.sources_instrum["read_noise_e_pix-1"]
        assert read_noise.unit.is_equivalent(u.electron / u.pix)
        assert np.allclose(read_noise.value, [2.0, 3.0])

        dc_rate = noise_calc.sources_instrum["dark_current_e_pix-1_sec-1"]
        assert dc_rate.unit.is_equivalent(u.electron / (u.pix * u.s))
        assert np.allclose(dc_rate.value, [0.0, 0.1])

        dc_total = noise_calc.sources_instrum["dark_current_e_pix-1"]
        assert dc_total.unit.is_equivalent(u.electron / u.pix)
        assert np.allclose(dc_total.value, [0.0, 10.0])  # 100 s * [0.0, 0.1]

    def test_calculate_intrinsic_instrumental_noise_parses_single_values(self, unit_converter):
        """calculate_instrinsic_instrumental_noise supports single read_noise/dark_current values."""
        config = {
            "detector": {
                "read_noise": "6.0",
                "dark_current": "0.05",
                "gain": "4.5",
            },
            "observation": {"integration_time": "200"},
        }
        noise_calc = InstrumentDepTerms(
            config,
            unit_converter,
            sources_astroph={},
            sources_to_include=[],
        )

        noise_calc.calculate_instrinsic_instrumental_noise()

        read_noise = noise_calc.sources_instrum["read_noise_e_pix-1"]
        assert read_noise.unit.is_equivalent(u.electron / u.pix)
        assert np.allclose(read_noise.value, [6.0])

        dc_rate = noise_calc.sources_instrum["dark_current_e_pix-1_sec-1"]
        assert dc_rate.unit.is_equivalent(u.electron / (u.pix * u.s))
        assert np.allclose(dc_rate.value, [0.05])

        dc_total = noise_calc.sources_instrum["dark_current_e_pix-1"]
        assert dc_total.unit.is_equivalent(u.electron / u.pix)
        assert np.allclose(dc_total.value, [10.0])  # 200 s * 0.05

    def test_pass_through_aperture_updates_prop_dict_for_valid_units(self, unit_converter):
        """pass_through_aperture scales flux by collecting area and stores pre/post values."""
        config = {
            "telescope": {"collecting_area": "25.0"},
        }

        wavel = np.array([1.0, 2.0, 3.0]) * u.um
        flux_pre = np.array([1.0, 2.0, 3.0]) * u.ph / (u.um * u.m**2 * u.s)

        sources_astroph = {
            "star": {"wavel": wavel, "astro_flux_ph_sec_m2_um": flux_pre},
            # invalid: wrong units so should be ignored
            "bad_source": {
                "wavel": wavel,
                "astro_flux_ph_sec_m2_um": np.array([1.0, 2.0, 3.0]) * u.W / (u.m**2),
            },
            # invalid: missing key so should be ignored
            "missing_key": {"wavel": wavel},
        }

        instr = InstrumentDepTerms(
            config=config,
            unit_converter=unit_converter,
            sources_astroph=sources_astroph,
            sources_to_include=["star", "bad_source", "missing_key"],
        )

        instr.pass_through_aperture(plot=False)

        assert "star" in instr.prop_dict
        assert "bad_source" not in instr.prop_dict
        assert "missing_key" not in instr.prop_dict

        star = instr.prop_dict["star"]
        assert star["wavel"] is wavel
        assert star["flux_pre_aperture_ph_sec_m2_um"].unit == u.ph / (u.um * u.m**2 * u.s)
        assert np.allclose(star["flux_pre_aperture_ph_sec_m2_um"].value, [1.0, 2.0, 3.0])

        post = star["flux_post_aperture_ph_sec_um"]
        assert post.unit.is_equivalent(u.ph / (u.um * u.s))
        assert np.allclose(post.value, (25.0 * u.m**2 * flux_pre).to(u.ph / (u.um * u.s)).value)

    def test_photons_to_e_converts_post_aperture_flux(self, unit_converter):
        """photons_to_e converts ph/sec/um to e/sec/um using photons_to_e and QE."""
        config = {
            "detector": {"photons_to_e": "1.0", "quantum_efficiency": "0.8"},
        }
        instr = InstrumentDepTerms(
            config=config,
            unit_converter=unit_converter,
            sources_astroph={},
            sources_to_include=[],
        )

        wavel = np.array([1.0, 2.0, 3.0]) * u.um
        flux_ph = np.array([10.0, 20.0, 30.0]) * u.ph / (u.um * u.s)
        instr.prop_dict = {
            "star": {"wavel": wavel, "flux_post_aperture_ph_sec_um": flux_ph},
            # wrong units -> should be ignored
            "bad": {
                "wavel": wavel,
                "flux_post_aperture_ph_sec_um": np.array([1.0, 2.0, 3.0]) * u.W,
            },
            # missing key -> should be ignored
            "missing": {"wavel": wavel},
        }

        instr.photons_to_e()

        assert "flux_e_sec_um" in instr.prop_dict["star"]
        expected = (1.0 * (u.electron / u.ph) * 0.8 * flux_ph).to(
            u.electron / (u.um * u.s)
        )
        got = instr.prop_dict["star"]["flux_e_sec_um"]
        assert got.unit.is_equivalent(u.electron / (u.um * u.s))
        assert np.allclose(got.value, expected.value)

        assert "flux_e_sec_um" not in instr.prop_dict["bad"]
        assert "flux_e_sec_um" not in instr.prop_dict["missing"]


class TestDetector:
    """Test cases for Detector class."""

    def test_init_parses_detector_geometry(self):
        config = {
            "detector": {
                "size": "1024",
                "pitch_pix": "25",
                "pix_per_wavel_bin": "2.2",
                "pix_spectral_width": "3",
            }
        }
        det = Detector(config=config, num_wavel_bins=17)
        assert det.side_length_pix == 1024
        assert det.pitch_pix == 25.0
        assert det.pix_per_wavel_bin == 2.2
        assert det.pix_spectral_width == 3
        assert det.num_wavel_bins == 17
        assert det.config is config

    def test_footprint_spectral_returns_expected_cube(self):
        """
        footprint_spectral returns a (num_wavel_bins, size, size) cube with
        fractional pixels at bin edges and total footprint matching
        pix_spectral_width * pix_per_wavel_bin per wavelength bin.
        """
        config = {
            "detector": {
                # Must be large enough for hardcoded starting_pixel [100, 300]
                "size": "400",
                "pitch_pix": "25",
                "pix_per_wavel_bin": "2.2",
                "pix_spectral_width": "3",
            }
        }
        det = Detector(config=config, num_wavel_bins=3)
        cube = det.footprint_spectral(file_name_plot="/tmp/ignore.png", plot=False)

        assert cube.shape == (3, 400, 400)
        assert cube.dtype == float
        assert np.min(cube) >= 0.0
        assert np.max(cube) <= 1.0

        # Each wavelength bin footprint should sum to pix_spectral_width * pix_per_wavel_bin
        expected_sum = det.pix_spectral_width * det.pix_per_wavel_bin  # 3 * 2.2 = 6.6
        for wbin in range(det.num_wavel_bins):
            assert np.isclose(cube[wbin].sum(), expected_sum)

        # Spot-check bin 0 around the expected columns:
        # start y=300, pix_per_wavel_bin=2.2 => full cols 300,301 and partial end at col 302 of 0.2
        rows = slice(100, 100 + det.pix_spectral_width)
        assert np.allclose(cube[0, rows, 300], 1.0)
        assert np.allclose(cube[0, rows, 301], 1.0)
        assert np.allclose(cube[0, rows, 302], 0.2)
    
    '''
    def test_calculate_dark_current_electrons(self, mock_config, unit_converter):
        """Test calculation of dark current noise in electrons."""
        noise_calc = InstrumentDepTerms(mock_config, unit_converter)
        
        integration_time = 3600.0
        noise = noise_calc.calculate_dark_current_electrons(integration_time)
        
        assert isinstance(noise, float)
        assert noise >= 0
        # Should be sqrt(dark_current_rate * integration_time)
        expected = np.sqrt(0.1 * 3600.0)
        assert abs(noise - expected) < 1e-10
    
    def test_calculate_dark_current_adu(self, mock_config, unit_converter):
        """Test calculation of dark current noise in ADU."""
        noise_calc = InstrumentDepTerms(mock_config, unit_converter)
        
        integration_time = 3600.0
        noise = noise_calc.calculate_dark_current_adu(integration_time)
        
        assert isinstance(noise, float)
        assert noise >= 0
        # Should be (sqrt(dark_current_rate * integration_time)) / gain
        expected = np.sqrt(0.1 * 3600.0) / 2.0
        assert abs(noise - expected) < 1e-10
    
    def test_calculate_read_noise_electrons(self, mock_config, unit_converter):
        """Test calculation of read noise in electrons."""
        noise_calc = InstrumentDepTerms(mock_config, unit_converter)
        
        noise = noise_calc.calculate_read_noise_electrons()
        
        assert isinstance(noise, float)
        assert noise == 5.0  # Should match config value
    
    def test_calculate_read_noise_adu(self, mock_config, unit_converter):
        """Test calculation of read noise in ADU."""
        noise_calc = InstrumentDepTerms(mock_config, unit_converter)
        
        noise = noise_calc.calculate_read_noise_adu()
        
        assert isinstance(noise, float)
        assert noise == 5.0 / 2.0  # Should be read_noise / gain
    
    def test_calculate_total_instrumental_noise_electrons(self, mock_config, unit_converter):
        """Test calculation of total instrumental noise in electrons."""
        noise_calc = InstrumentDepTerms(mock_config, unit_converter)
        
        integration_time = 3600.0
        noise = noise_calc.calculate_total_instrumental_noise_electrons(integration_time)
        
        assert isinstance(noise, float)
        assert noise >= 0
        
        # Should be sqrt(dark_noise^2 + read_noise^2)
        dark_noise = np.sqrt(0.1 * 3600.0)
        read_noise = 5.0
        expected = np.sqrt(dark_noise**2 + read_noise**2)
        assert abs(noise - expected) < 1e-10
    
    def test_calculate_total_instrumental_noise_adu(self, mock_config, unit_converter):
        """Test calculation of total instrumental noise in ADU."""
        noise_calc = InstrumentDepTerms(mock_config, unit_converter)
        
        integration_time = 3600.0
        noise = noise_calc.calculate_total_instrumental_noise_adu(integration_time)
        
        assert isinstance(noise, float)
        assert noise >= 0
        
        # Should be total_electron_noise / gain
        dark_noise = np.sqrt(0.1 * 3600.0)
        read_noise = 5.0
        total_electron_noise = np.sqrt(dark_noise**2 + read_noise**2)
        expected = total_electron_noise / 2.0
        assert abs(noise - expected) < 1e-10
    
    def test_get_noise_breakdown_electrons(self, mock_config, unit_converter):
        """Test getting noise breakdown in electrons."""
        noise_calc = InstrumentDepTerms(mock_config, unit_converter)
        
        integration_time = 3600.0
        breakdown = noise_calc.get_noise_breakdown_electrons(integration_time)
        
        assert isinstance(breakdown, dict)
        assert "dark_current" in breakdown
        assert "read_noise" in breakdown
        assert all(isinstance(noise, float) for noise in breakdown.values())
        assert all(noise >= 0 for noise in breakdown.values())
    
    def test_get_noise_breakdown_adu(self, mock_config, unit_converter):
        """Test getting noise breakdown in ADU."""
        noise_calc = InstrumentDepTerms(mock_config, unit_converter)
        
        integration_time = 3600.0
        breakdown = noise_calc.get_noise_breakdown_adu(integration_time)
        
        assert isinstance(breakdown, dict)
        assert "dark_current" in breakdown
        assert "read_noise" in breakdown
        assert all(isinstance(noise, float) for noise in breakdown.values())
        assert all(noise >= 0 for noise in breakdown.values())
    '''