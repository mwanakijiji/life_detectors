"""Tests for write_onsky_scene_fits (the scene_no_screen.fits FYI output)."""

import numpy as np
from astropy import units as u
from astropy.io import fits

from modules.core.astrophysical import write_onsky_scene_fits

FLUX_UNIT = u.ph / (u.um * u.m**2 * u.s)


def _scene(n_wavel=3, n_pix=5, sources=("star", "exozodiacal", "zodiacal")):
    cubes = {
        name: np.full((n_wavel, n_pix, n_pix), float(i + 1)) * FLUX_UNIT
        for i, name in enumerate(sources)
    }
    total = sum(cubes.values())
    wavel = np.linspace(4.0, 18.0, n_wavel) * u.um
    axis = np.arange(n_pix) - n_pix // 2
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return cubes, total, wavel, yy.astype(float), xx.astype(float)


def test_writes_total_and_one_hdu_per_source(tmp_path):
    cubes, total, wavel, yy, xx = _scene()
    path = write_onsky_scene_fits(
        str(tmp_path / "scene.fits"), cubes, total, wavel, yy, xx
    )

    with fits.open(path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert names == [
            "PRIMARY",
            "STAR",
            "EXOZODIACAL",
            "ZODIACAL",
            "YY_ARCSEC",
            "XX_ARCSEC",
            "WAVEL_UM",
        ]
        np.testing.assert_allclose(hdul["PRIMARY"].data, total.value)
        np.testing.assert_allclose(hdul["ZODIACAL"].data, cubes["zodiacal"].value)
        np.testing.assert_allclose(hdul["WAVEL_UM"].data, wavel.value)
        assert hdul["PRIMARY"].header["NSOURCE"] == 3
        assert hdul["PRIMARY"].header["SOURCE1"] == "exozodiacal"
        assert hdul["STAR"].header["BUNIT"] == str(FLUX_UNIT)


def test_works_without_a_planet_source(tmp_path):
    """The old writer required exoplanet_model_10pc; any set of sources now works."""
    cubes, total, wavel, yy, xx = _scene(sources=("star",))
    path = write_onsky_scene_fits(
        str(tmp_path / "scene.fits"), cubes, total, wavel, yy, xx
    )

    with fits.open(path) as hdul:
        assert "STAR" in hdul
        assert "EXOPLANET_MODEL_10PC" not in hdul
