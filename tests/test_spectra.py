"""
Unit tests for spectral data handling.
"""

import numpy as np
import pytest
from pathlib import Path

from modules.data.spectra import SpectralData
from modules.data.spectra import load_spectrum_from_file
from modules.data.spectra import create_blackbody_spectrum


class TestSpectralDataInit:
    def test_raises_on_shape_mismatch(self):
        wavelength = np.array([1.0, 2.0, 3.0])
        flux = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="Wavelength and flux arrays must have the same shape"):
            SpectralData(
                wavelength=wavelength,
                flux=flux,
                wavelength_unit="um",
                flux_unit="ph / (um m2 s)",
                source_name="test",
                metadata={},
            )

    def test_sorts_by_wavelength_and_reorders_flux(self):
        wavelength = np.array([3.0, 1.0, 2.0])
        flux = np.array([30.0, 10.0, 20.0])
        s = SpectralData(
            wavelength=wavelength,
            flux=flux,
            wavelength_unit="um",
            flux_unit="ph / (um m2 s)",
            source_name="test",
            metadata={},
        )
        assert np.allclose(s.wavelength, [1.0, 2.0, 3.0])
        assert np.allclose(s.flux, [10.0, 20.0, 30.0])

    def test_metadata_none_defaults_to_empty_dict(self):
        s = SpectralData(
            wavelength=np.array([1.0, 2.0]),
            flux=np.array([10.0, 20.0]),
            wavelength_unit="um",
            flux_unit="ph / (um m2 s)",
            source_name="test",
            metadata=None,
        )
        assert isinstance(s.metadata, dict)
        assert s.metadata == {}

    def test_interpolate_linear_and_out_of_bounds_fill(self):
        s = SpectralData(
            wavelength=np.array([1.0, 2.0, 3.0]),
            flux=np.array([10.0, 20.0, 30.0]),
            wavelength_unit="um",
            flux_unit="ph / (um m2 s)",
            source_name="star",
            metadata={"k": "v"},
        )
        new_wavelength = np.array([0.0, 1.5, 2.5, 4.0])  # includes out-of-bounds
        out = s.interpolate(new_wavelength)
        # Expected: OOB -> 0.0, linear in-range -> 15.0 and 25.0
        assert np.allclose(out.wavelength, new_wavelength)
        assert np.allclose(out.flux, [0.0, 15.0, 25.0, 0.0])
        # Metadata and identity fields should be preserved
        assert out.flux_unit == s.flux_unit
        assert out.source_name == s.source_name
        assert out.metadata == s.metadata

    def test_integrate_flux_over_subrange(self):
        s = SpectralData(
            wavelength=np.array([1.0, 2.0, 3.0, 4.0]),
            flux=np.array([1.0, 2.0, 3.0, 4.0]),
            wavelength_unit="um",
            flux_unit="ph / (um m2 s)",
            source_name="test",
            metadata={},
        )
        # Integrate over [2, 4] -> trapz([2,3,4], [2,3,4]) = 6.0
        assert np.isclose(s.integrate_flux(2.0, 4.0), 6.0)

    def test_integrate_flux_returns_zero_when_no_overlap(self):
        s = SpectralData(
            wavelength=np.array([1.0, 2.0, 3.0]),
            flux=np.array([10.0, 20.0, 30.0]),
            wavelength_unit="um",
            flux_unit="ph / (um m2 s)",
            source_name="test",
            metadata={},
        )
        assert s.integrate_flux(10.0, 12.0) == 0.0

    def test_get_flux_at_wavelength_interpolates_linearly(self):
        s = SpectralData(
            wavelength=np.array([1.0, 2.0, 3.0]),
            flux=np.array([10.0, 20.0, 30.0]),
            wavelength_unit="um",
            flux_unit="ph / (um m2 s)",
            source_name="test",
            metadata={},
        )
        assert np.isclose(s.get_flux_at_wavelength(1.5), 15.0)
        assert np.isclose(s.get_flux_at_wavelength(2.5), 25.0)

    def test_get_flux_at_wavelength_out_of_bounds_returns_zero(self):
        s = SpectralData(
            wavelength=np.array([1.0, 2.0, 3.0]),
            flux=np.array([10.0, 20.0, 30.0]),
            wavelength_unit="um",
            flux_unit="ph / (um m2 s)",
            source_name="test",
            metadata={},
        )
        assert s.get_flux_at_wavelength(0.5) == 0.0
        assert s.get_flux_at_wavelength(4.0) == 0.0


class TestLoadSpectrumFromFile:
    def test_load_spectrum_from_file_parses_units_and_sorts(self, tmp_path):
        file_path = tmp_path / "sample_spectrum.csv"
        file_path.write_text(
            "\n".join(
                [
                    "# wavelength_unit=um",
                    "# luminosity_photons_unit=photon/um/sec",
                    "wavel,luminosity_photons",
                    "3.0,30.0",
                    "1.0,10.0",
                    "2.0,20.0",
                ]
            )
        )

        spectrum = load_spectrum_from_file(file_path)

        assert isinstance(spectrum, SpectralData)
        assert np.allclose(spectrum.wavelength, [1.0, 2.0, 3.0])
        assert np.allclose(spectrum.flux, [10.0, 20.0, 30.0])
        assert spectrum.wavelength_unit == "um"
        assert spectrum.flux_unit == "photon/um/sec"
        assert spectrum.source_name == "sample_spectrum"
        assert spectrum.metadata["filepath"] == str(file_path)

    def test_load_spectrum_from_file_missing_file_currently_raises_nameerror(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        # Current implementation logs with undefined variable 'e' in the missing-file branch.
        with pytest.raises(NameError):
            load_spectrum_from_file(missing)


class TestCreateBlackbodySpectrum:
    def test_create_blackbody_spectrum_basic_properties(self):
        spec = create_blackbody_spectrum(
            temperature=300.0,
            wavelength_range=(1.0, 10.0),
            n_points=50,
        )

        assert isinstance(spec, SpectralData)
        assert spec.source_name == "blackbody_300.0K"
        assert spec.wavelength_unit == "um"
        assert spec.flux_unit == "photon_sec_m2_um"
        assert spec.metadata["temperature"] == 300.0
        assert len(spec.wavelength) == 50
        assert len(spec.flux) == 50

    def test_create_blackbody_spectrum_wavelength_grid_and_flux_physical(self):
        n_points = 20
        wl_min, wl_max = 2.0, 20.0
        spec = create_blackbody_spectrum(
            temperature=5778.0,
            wavelength_range=(wl_min, wl_max),
            n_points=n_points,
        )

        # Bounds and monotonicity
        assert np.isclose(spec.wavelength[0], wl_min)
        assert np.isclose(spec.wavelength[-1], wl_max)
        assert np.all(np.diff(spec.wavelength) > 0)

        # Log-spaced grid -> constant ratio between adjacent wavelengths
        ratios = spec.wavelength[1:] / spec.wavelength[:-1]
        assert np.allclose(ratios, ratios[0])

        # Physically sensible output: finite and positive photon flux
        assert np.all(np.isfinite(spec.flux))
        assert np.all(spec.flux > 0)