"""The Cython core must match the pure-NumPy oracle bit-closely."""
import numpy as np
import pytest

from pyonlm import reference as ref
from pyonlm import onlm

pytestmark = pytest.mark.skipif(not onlm.HAVE_CYTHON,
                                reason="Cython core not built")


def _rand_volume(shape=(12, 11, 10), seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 10.0, size=shape)
    # add a background region (low mean/var) so the mean/var>eps gating exercises
    base[:3] = 0.0
    return np.ascontiguousarray(base)


def test_mean_var_parity():
    from pyonlm import _onlm
    vol = _rand_volume()
    m_ref = ref.local_mean(vol, (1, 1, 1))
    v_ref = ref.local_var(vol, m_ref, (1, 1, 1))
    m_cy, v_cy = _onlm.compute_mean_var(vol, 1, 1, 1)
    # both accumulate in double and store float32 -> match to float32 precision
    assert np.allclose(m_ref, m_cy, rtol=1e-5, atol=1e-3)
    assert np.allclose(v_ref, v_cy, rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_block_gaussian_parity(seed):
    vol = _rand_volume(seed=seed)
    out_ref = ref.denoise_block(vol, h=8.0, beta=1.0)
    out_cy = onlm.denoise_block(vol, h=8.0, beta=1.0)
    assert np.allclose(out_ref, out_cy, rtol=1e-4, atol=1e-2)


def test_block_rician_parity():
    vol = _rand_volume(seed=3)
    out_ref = ref.denoise_block(vol, h=8.0, beta=1.0, weight_method=2)
    out_cy = onlm.denoise_block(vol, h=8.0, beta=1.0, weight_method=2)
    assert np.allclose(out_ref, out_cy, rtol=1e-4, atol=1e-2)


def test_multithread_matches_single_thread():
    # race-free per-thread accumulation: many threads must agree with 1 thread to
    # float32 precision (on machines without OpenMP this trivially runs serial).
    vol = _rand_volume(shape=(20, 18, 16), seed=5)
    out1 = onlm.denoise_block(vol, h=8.0, beta=1.0, num_threads=1)
    out4 = onlm.denoise_block(vol, h=8.0, beta=1.0, num_threads=4)
    assert np.allclose(out1, out4, rtol=1e-4, atol=1e-2)


def test_mt_grid_emulation_parity():
    # cython grid emulation (mincnlm -mt N block grid) must match the reference.
    vol = _rand_volume(shape=(20, 18, 16), seed=6)
    out_ref = ref.denoise_block(vol, h=8.0, beta=1.0, mt_grid=3)
    out_cy = onlm.denoise_block(vol, h=8.0, beta=1.0, mt_grid=3, num_threads=3)
    assert np.allclose(out_ref, out_cy, rtol=1e-4, atol=1e-2)


def test_mt_grid_changes_result():
    # a debut-anchored grid genuinely differs from the uniform grid (else the
    # emulation would be pointless) -- sanity check it actually does something.
    # nz=20, mt_grid=3 -> partitions start at k=0,6,13; 13 is odd so the stride
    # shifts off the uniform even grid (as at nz=520,mt=6 where debut 173,433 are odd).
    vol = _rand_volume(shape=(20, 18, 20), seed=7)
    uniform = onlm.denoise_block(vol, h=8.0, beta=1.0, mt_grid=0)
    grid3 = onlm.denoise_block(vol, h=8.0, beta=1.0, mt_grid=3)
    assert not np.allclose(uniform, grid3, atol=1.0)


def test_block_bspace_and_radius_parity():
    vol = _rand_volume(shape=(14, 13, 12), seed=4)
    kw = dict(h=6.0, beta=1.2, radius=(1, 1, 1), search=(3, 3, 3), b_space=3)
    out_ref = ref.denoise_block(vol, **kw)
    out_cy = onlm.denoise_block(vol, **kw)
    assert np.allclose(out_ref, out_cy, rtol=1e-4, atol=1e-2)
