"""Tests for the ported Sharp & Birchall optimal extraction."""

import numpy as np
import pytest

from modules.core.instrumental.sharp_birchall import (
    SharpBirchallExtractor,
    floor_variance,
    rebin_columns_to_bins,
)


def _gauss_profiles(ny, nx, centers, sigma):
    """Horizontal Gaussian profiles (as glint-control's
    gauss1d), normalized per column.
    """
    y = np.arange(ny)[:, None]
    profiles = np.stack(
        [np.exp(-0.5 * ((y - c) / sigma) ** 2) * np.ones((1, nx)) for c in centers]
    )
    return profiles / profiles.sum(axis=1, keepdims=True)


def test_single_uniform_trace_recovers_flux_and_variances():
    ny, nx, rd = 4, 6, 1.5
    profile = np.full((1, ny, nx), 1.0 / ny)
    eta = np.arange(1.0, nx + 1.0) * 10.0
    D = profile[0] * eta
    variance = np.random.default_rng(1).uniform(4.0, 9.0, size=(ny, nx))

    ext = SharpBirchallExtractor(profile, variance, read_noise=rd)
    flux, var_flux, vark = ext.extract(D)

    assert flux.shape == (1, nx)
    assert np.allclose(flux[0], eta)
    # uniform profile: Var(eta) = 1 / sum_j (phi^2 / S^2)
    assert np.allclose(var_flux[0], 1.0 / np.sum((1.0 / ny) ** 2 / variance, axis=0))
    # Eqn 19: C' v = b'  ->  v = sum_j (S^2 - rd^2) for a uniform profile
    assert np.allclose(vark[0], np.sum(variance - rd**2, axis=0))


def test_uniform_trace_with_equal_variance_gives_column_sum_variance():
    ny, nx, sigma2 = 3, 5, 7.0
    profile = np.full((1, ny, nx), 1.0 / ny)
    ext = SharpBirchallExtractor(profile, np.full((ny, nx), sigma2))
    _, var_flux, _ = ext.extract(np.zeros((ny, nx)))
    assert np.allclose(var_flux[0], ny * sigma2)


def test_two_overlapping_profiles_recovered_and_errors_match_scatter():
    ny, nx = 20, 8
    profiles = _gauss_profiles(ny, nx, centers=[8.0, 11.0], sigma=1.5)
    eta_true = np.array([[100.0] * nx, [40.0] * nx])
    model = np.einsum("ijk,ik->jk", profiles, eta_true)
    variance = model + 4.0  # shot noise + read noise^2

    ext = SharpBirchallExtractor(profiles, variance, read_noise=2.0)
    flux, var_flux, _ = ext.extract(model)
    assert np.allclose(flux, eta_true)

    rng = np.random.default_rng(42)
    noisy = model + rng.normal(size=(2000, ny, nx)) * np.sqrt(variance)
    flux_noisy, _, _ = ext.extract(noisy)  # (n_frames, n_spec, nx)
    scatter = flux_noisy.std(axis=0)
    assert np.allclose(scatter, np.sqrt(var_flux), rtol=0.05)


def test_empty_columns_and_number_of_spectra_not_hard_coded():
    ny, nx = 6, 5
    profiles = _gauss_profiles(ny, nx, centers=[1.0, 2.5, 4.0], sigma=0.8)
    profiles[:, :, 0] = 0.0  # column with no trace
    ext = SharpBirchallExtractor(profiles, np.ones((ny, nx)))
    flux, var_flux, vark = ext.extract(np.ones((ny, nx)))
    assert flux.shape == var_flux.shape == vark.shape == (3, nx)
    assert np.all(flux[:, 0] == 0.0)
    assert np.all(np.isfinite(flux))


def test_floor_variance_replaces_non_positive_values():
    variance = floor_variance(np.array([[0.0, -1.0], [4.0, np.nan]]))
    assert np.all(variance == 4.0)
    with pytest.raises(ValueError):
        floor_variance(np.zeros((2, 2)))


def test_rebin_columns_conserves_flux_and_uses_squared_fractions():
    fraction = np.array([[1.0, 0.4, 0.0], [0.0, 0.6, 1.0]])  # (n_bins, nx)
    flux = np.array([10.0, 5.0, 2.0])
    var = np.array([1.0, 2.0, 3.0])
    flux_b, var_b = rebin_columns_to_bins(fraction, flux, var)
    assert np.allclose(flux_b, [12.0, 5.0])
    assert np.isclose(flux_b.sum(), flux.sum())
    assert np.allclose(var_b, [1.0 + 0.16 * 2.0, 0.36 * 2.0 + 3.0])
