"""Hand-computed checks of the ONLM primitives (mean/var/L2/weight/patch norm)."""
import numpy as np
import pytest

from pyonlm import reference as ref


def test_box_offsets_order():
    # C++ order: a (x) fastest, then b (y), then c (z)
    offs = ref._box_offsets((1, 1, 1))
    assert len(offs) == 27
    assert offs[0] == (-1, -1, -1)
    assert offs[1] == (0, -1, -1)   # a advanced first
    assert offs[3] == (-1, 0, -1)   # then b
    assert offs[9] == (-1, -1, 0)   # then c
    assert offs[13] == (0, 0, 0)    # centre


def test_local_mean_interior_and_corner():
    vol = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    m = ref.local_mean(vol, (1, 1, 1))
    # interior voxel (1,1,1): mean of full 3x3x3 = mean(0..26) = 13
    assert m[1, 1, 1] == pytest.approx(13.0)
    # corner (0,0,0): mean over the in-bounds 2x2x2 block
    block = vol[0:2, 0:2, 0:2]
    assert m[0, 0, 0] == pytest.approx(block.mean())


def test_local_var_uses_neighbor_mean_and_nminus1():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(4, 4, 4))
    m = ref.local_mean(vol, (1, 1, 1))
    v = ref.local_var(vol, m, (1, 1, 1))
    # interior voxel (2,2,2): sum over 27 neighbours of (vol[p]-m[p])^2 / (27-1)
    i, j, k = 2, 2, 2
    acc = 0.0
    for (ox, oy, oz) in ref._box_offsets((1, 1, 1)):
        p = (i + ox, j + oy, k + oz)
        acc += (vol[p] - m[p]) ** 2
    assert v[i, j, k] == pytest.approx(acc / 26.0)


def test_neiborghood_zero_fill_and_normalization():
    vol = np.ones((3, 3, 3), dtype=np.float64) * 4.0
    # corner: 8 in-bounds neighbours -> nb_inside=8; normalise by sqrt(8)
    vec = ref._neiborghood(vol, 0, 0, 0, (1, 1, 1), 0)
    # in-bounds elements should equal 4/sqrt(8); oob elements are 0
    nonzero = vec[vec != 0]
    assert len(nonzero) == 8
    assert np.allclose(nonzero, 4.0 / np.sqrt(8.0))


def test_patch_block_center_fill():
    vol = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    # corner patch: oob elements fall back to the centre value vol[0,0,0]=0
    p = ref._patch_block(vol, 0, 0, 0, (1, 1, 1), square=False)
    assert len(p) == 27
    center = vol[0, 0, 0]
    # first element offset (-1,-1,-1) is oob -> center value
    assert p[0] == center


def test_l2_and_weight():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 0.0, 0.0])
    d = ref.l2_norm(v1, v2)
    assert d == pytest.approx(0 + 4 + 9)
    # block path denominator = beta*h^2 (Weight called with beta/2)
    w = ref.weight(d, beta=0.5, h=2.0)  # exp(-13/(2*0.5*4)) = exp(-13/4)
    assert w == pytest.approx(np.exp(-13.0 / 4.0))
    assert ref.weight(d, beta=1.0, h=0.0) == 0.0
