"""
Unit tests for modules.data.units.
"""

import pytest

from modules.data.units import UnitConverter


class TestUnitConverter:
    def test_convert_wavelength_same_unit_returns_input(self):
        uc = UnitConverter()
        assert uc.convert_wavelength(10.0, "um", "um") == 10.0

    def test_convert_wavelength_direct_and_reverse(self):
        uc = UnitConverter()
        assert uc.convert_wavelength(1.0, "um", "nm") == 1000.0
        assert uc.convert_wavelength(1000.0, "nm", "um") == 1.0
        assert uc.convert_wavelength(1.0, "m", "angstrom") == 1e10

    def test_convert_wavelength_unsupported_raises(self):
        uc = UnitConverter()
        with pytest.raises(ValueError, match="Unsupported wavelength conversion"):
            uc.convert_wavelength(1.0, "um", "invalid_unit")

    def test_convert_flux_same_unit_returns_input(self):
        uc = UnitConverter()
        assert uc.convert_flux(5.0, wavelength=10.0, from_unit="x", to_unit="x") == 5.0

    def test_convert_flux_currently_errors_when_conversion_requested(self):
        """
        _flux_conversions is not initialized in current implementation, so requesting
        an actual conversion raises AttributeError.
        """
        uc = UnitConverter()
        with pytest.raises(AttributeError):
            uc.convert_flux(
                1.0,
                wavelength=10.0,
                from_unit="photon_sec_m2_um",
                to_unit="photon_sec_m2_nm",
            )

    def test_electrons_adu_round_trip(self):
        uc = UnitConverter()
        electrons = 45.0
        gain = 4.5
        adu = uc.electrons_to_adu(electrons, gain)
        assert adu == 10.0
        assert uc.adu_to_electrons(adu, gain) == electrons
