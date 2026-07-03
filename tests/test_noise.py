"""Sanity + structural checks for the DWT noise estimator port."""
import numpy as np
import pytest

from pyonlm import noise as ne


def test_daub4_matches_naive_periodic():
    rng = np.random.default_rng(0)
    a = rng.normal(size=16)
    out = ne._daub4_forward_axis(a[:, None, None], 0)[:, 0, 0]
    n = 16
    C0, C1, C2, C3 = ne._C0, ne._C1, ne._C2, ne._C3
    nh = n // 2
    smooth = np.array([C0 * a[2 * i] + C1 * a[2 * i + 1]
                       + C2 * a[(2 * i + 2) % n] + C3 * a[(2 * i + 3) % n]
                       for i in range(nh)])
    detail = np.array([C3 * a[2 * i] - C2 * a[2 * i + 1]
                       + C1 * a[(2 * i + 2) % n] - C0 * a[(2 * i + 3) % n]
                       for i in range(nh)])
    assert np.allclose(out[:nh], smooth)
    assert np.allclose(out[nh:], detail)


def test_dwt_subband_count_and_shape():
    rng = np.random.default_rng(1)
    vol = rng.normal(size=(20, 18, 16))
    sub = ne.dwt_forward(vol)
    assert len(sub) == 8
    exp = tuple((np.array(vol.shape) + 1) // 2)
    for s in sub:
        assert s.shape == exp


def test_next_pow2_cube_side():
    assert ne._next_pow2_cube_side((256, 256, 200)) == 256
    assert ne._next_pow2_cube_side((257, 100, 100)) == 512
    assert ne._next_pow2_cube_side((200, 10, 10)) == 256


def test_histogram_percentile_monotone():
    rng = np.random.default_rng(2)
    vals = rng.normal(50.0, 5.0, size=50000)
    h = ne.build_histogram(vals, bins=2000)
    p10 = h.find_percentile(0.10)
    p50 = h.find_percentile(0.50)
    p90 = h.find_percentile(0.90)
    assert p10 < p50 < p90
    assert p50 == pytest.approx(50.0, abs=0.5)


def test_kmeans_two_clusters_separates():
    # bimodal: background near 0, object near 100
    rng = np.random.default_rng(3)
    bg = rng.normal(0.0, 1.0, size=20000)
    fg = rng.normal(100.0, 5.0, size=20000)
    h = ne.build_histogram(np.concatenate([bg, fg]), bins=2000)
    mu = ne.simple_k_means(h, 2, 10)
    lo, hi = sorted(mu)
    assert lo < 20 and hi > 80


def test_gradient_mag_zero_on_constant_interior():
    # A constant volume has zero interior gradient; the zero-padded FFT boundary
    # rings (as it does in EZminc), so only the interior is meaningfully ~0.
    vol = np.full((16, 16, 16), 7.0)
    g = ne.calc_gradient_mag(vol)
    assert np.allclose(g[4:12, 4:12, 4:12], 0.0, atol=1e-6)


def test_gradient_mag_uniform_on_ramp_interior():
    # A linear ramp has a spatially-uniform gradient in the interior.
    ramp = np.tile(np.arange(16.0), (16, 16, 1))
    g = ne.calc_gradient_mag(ramp)
    core = g[5:11, 5:11, 5:11]
    assert core.std() < 1e-6
    assert core.mean() > 0.0


def test_noise_estimate_recovers_gaussian_sigma():
    # object (sphere) + known Gaussian noise; MAD-on-HHH should recover ~sigma
    rng = np.random.default_rng(4)
    n = 64
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2)
    clean = np.where(r < n / 3, 100.0, 0.0)
    sigma_true = 4.0
    vol = clean + rng.normal(0.0, sigma_true, size=clean.shape)
    est = ne.noise_estimate(vol, gaussian=True)
    # within ~30% is plenty for a sanity check of the whole pipeline
    assert 0.6 * sigma_true < est < 1.5 * sigma_true
