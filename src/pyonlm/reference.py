"""Pure-NumPy faithful reference implementation of EZminc ``mincnlm``.

Line-for-line port of Coupé et al.'s Optimized Blockwise Non-Local Means (ONLM)
denoiser as implemented in EZminc (BIC-MNI), reading directly from the C++
sources vendored under ``reference/ezminc/``:

    mincnlm.cpp          -- CLI, defaults, orchestration
    nl_means_utils.cpp   -- Preprocessing (local mean / variance maps)
    nl_means.cpp         -- Neiborghood / L2_norm / Weight, voxelwise denoise
    nl_means_block.cpp   -- blockwise denoise (the mincnlm default, block=1)

Correctness/faithfulness is the goal, not speed: the block loop mirrors the C++
control flow so it can serve as the golden oracle that the fast Cython core is
validated against. It is single-threaded (the C++ Estimate/Label accumulation is
additive; thread count only changes float summation order -- and, in the C++,
introduces lost-update races at thread-slab boundaries, so the single-threaded
result is the deterministic reference).

FLOAT32 (important): mincnlm performs the whole denoise in 32-bit float -- the
mean/variance maps, patch vectors, weights, the global_sum/average accumulators
and the Estimate/Label buffers are all ``float``; only the L2 sum and ``exp`` are
evaluated in double and cast back to float. This is not a rounding nicety: when
every candidate weight underflows, float32 flushes ``global_sum`` to exactly 0
and the block is skipped (output = input), whereas float64 keeps denormal weights
and divides tiny-by-tiny into garbage. So we replicate float32 here. Local
mean/variance are accumulated in double and stored as float32 (matching a
float64-accurate value rounded to the float32 storage the C++ uses).
"""

from __future__ import annotations

import numpy as np

_F32 = np.float32

# mincnlm block path: epsilon = 0.0001 for the local mean/variance gating.
_EPS = 1e-4


# --------------------------------------------------------------------------- #
# Preprocessing: local mean and variance maps (nl_means_utils.cpp)
# --------------------------------------------------------------------------- #
def _box_offsets(radius):
    """Patch offsets in the exact (a=x fastest, then b=y, then c=z) order the
    C++ uses, so patch vectors line up index-for-index with Neiborghood."""
    rx, ry, rz = radius
    offs = []
    for c in range(2 * rz + 1):
        for b in range(2 * ry + 1):
            for a in range(2 * rx + 1):
                offs.append((a - rx, b - ry, c - rz))
    return offs


def _shift_sum(vol, ox, oy, oz):
    """Add ``vol`` shifted by (ox,oy,oz) into a same-shaped accumulator, only
    over in-bounds overlap (out-of-bounds neighbors are simply not counted --
    matching the ``is_outside`` skip in Neighborhood_Mean/Var)."""
    nx, ny, nz = vol.shape
    out = np.zeros_like(vol)
    cnt = np.zeros(vol.shape, dtype=np.int32)

    ti0, ti1 = max(0, -ox), min(nx, nx - ox)
    tj0, tj1 = max(0, -oy), min(ny, ny - oy)
    tk0, tk1 = max(0, -oz), min(nz, nz - oz)
    if ti0 >= ti1 or tj0 >= tj1 or tk0 >= tk1:
        return out, cnt

    src = vol[ti0 + ox:ti1 + ox, tj0 + oy:tj1 + oy, tk0 + oz:tk1 + oz]
    out[ti0:ti1, tj0:tj1, tk0:tk1] = src
    cnt[ti0:ti1, tj0:tj1, tk0:tk1] = 1
    return out, cnt


def local_mean(vol, radius):
    """Neighborhood_Mean over the whole volume (Preprocessing).

    mean = sum(in-bounds neighbors) / (number of in-bounds neighbors), stored as
    float32 (the C++ mean_map dtype).
    """
    vol = vol.astype(np.float64, copy=False)
    total = np.zeros_like(vol)
    count = np.zeros(vol.shape, dtype=np.int32)
    for ox, oy, oz in _box_offsets(radius):
        s, c = _shift_sum(vol, ox, oy, oz)
        total += s
        count += c
    return (total / count).astype(_F32)  # count >= 1 always (center is in-bounds)


def local_var(vol, mean_map, radius):
    """Neighborhood_Var (Preprocessing2).

    Subtracts the *neighbor's* local mean (float32 ima_mean at the neighbor
    position), then divides by (nb_inside - 1). Stored as float32.
    """
    vol = vol.astype(np.float64, copy=False)
    diff2 = (vol - mean_map.astype(np.float64)) ** 2
    total = np.zeros_like(diff2)
    count = np.zeros(vol.shape, dtype=np.int32)
    for ox, oy, oz in _box_offsets(radius):
        s, c = _shift_sum(diff2, ox, oy, oz)
        total += s
        count += c
    denom = np.maximum(count - 1, 1)
    return (total / denom).astype(_F32)


# --------------------------------------------------------------------------- #
# Patch helpers (nl_means.cpp)
# --------------------------------------------------------------------------- #
def _neiborghood(vol, i, j, k, radius, weight_method):
    """Neiborghood(): extract patch vector (float32), out-of-bounds = 0, then
    (for Gaussian/Rician, weight_method 0/2) normalize each element by
    sqrt(nb_inside) in float32."""
    nx, ny, nz = vol.shape
    offs = _box_offsets(radius)
    vec = np.zeros(len(offs), dtype=_F32)
    nb_inside = 0
    for idx, (ox, oy, oz) in enumerate(offs):
        x, y, z = i + ox, j + oy, k + oz
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            vec[idx] = vol[x, y, z]
            nb_inside += 1
    if weight_method in (0, 2):
        vec = (vec / _F32(np.sqrt(_F32(nb_inside)))).astype(_F32)
    return vec


def _patch_block(vol, i, j, k, radius, square):
    """Raw patch values in C++ order (float32); out-of-bounds falls back to the
    *center* voxel value (matching Average_block / Average_block_Rician).
    ``square`` for the Rician path accumulates value^2."""
    nx, ny, nz = vol.shape
    offs = _box_offsets(radius)
    center = vol[i, j, k]
    out = np.empty(len(offs), dtype=_F32)
    for idx, (ox, oy, oz) in enumerate(offs):
        x, y, z = i + ox, j + oy, k + oz
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            v = vol[x, y, z]
        else:
            v = center
        out[idx] = _F32(v) * _F32(v) if square else _F32(v)
    return out


def l2_norm(v1, v2):
    """Sum of squared differences (nl_means.cpp: L2_norm), accumulated in double
    from float32 inputs then returned as float32 (the C++ returns float)."""
    d = v1.astype(np.float64) - v2.astype(np.float64)
    return _F32(float(np.dot(d, d)))


def weight(dist, beta, h):
    """Weighting function (nl_means.cpp: Weight). Returns float32, 0 when h == 0.

    exp is evaluated in double and cast to float32, exactly as the C++ does.
    """
    if _F32(h) == 0:
        return _F32(0.0)
    denom = 2.0 * float(_F32(beta)) * float(_F32(h)) * float(_F32(h))
    return _F32(np.exp(-float(_F32(dist)) / denom))


# --------------------------------------------------------------------------- #
# Blockwise ONLM (nl_means_block.cpp) -- the mincnlm default (block=1)
# --------------------------------------------------------------------------- #
def _block_kcenters(nz, b_space, mt_grid):
    """Block-centre k-planes: uniform (mt_grid<=1) or mincnlm's per-thread
    ``debut``-anchored grid for -mt mt_grid (see _onlm.denoise_block)."""
    if mt_grid > 1:
        kc = []
        for it in range(mt_grid):
            debut = (it * nz) // mt_grid
            fin = ((it + 1) * nz) // mt_grid
            k = debut
            while k < fin:
                kc.append(k)
                k += b_space
        return kc
    return list(range(0, nz, b_space))


def denoise_block(vol, h, beta=1.0, radius=(1, 1, 1), search=(5, 5, 5),
                  b_space=2, m_min=0.95, v_min=0.5, weight_method=0, mt_grid=0):
    """Faithful port of denoise_block_mt (single-threaded, float32).

    Parameters mirror mincnlm: ``radius`` = -v, ``search`` = -d, ``h`` = sigma
    (-sigma), ``beta`` = -beta, ``weight_method`` = -w (0 Gaussian, 2 Rician).
    ``mt_grid`` > 1 emulates mincnlm's -mt N block-centre grid (else uniform).
    """
    if weight_method not in (0, 2):
        raise NotImplementedError("weight_method %d not supported (only 0 Gaussian, "
                                  "2 Rician)" % weight_method)
    vol = np.ascontiguousarray(vol, dtype=_F32)
    nx, ny, nz = vol.shape

    mean_map = local_mean(vol, radius)
    var_map = local_var(vol, mean_map, radius)

    estimate = np.zeros(vol.shape, dtype=_F32)
    label = np.zeros(vol.shape, dtype=_F32)

    offs = _box_offsets(radius)
    rican = weight_method == 2
    hf = _F32(h)

    for k in _block_kcenters(nz, b_space, mt_grid):
        for j in range(0, ny, b_space):
            for i in range(0, nx, b_space):
                if not (mean_map[i, j, k] > _EPS and var_map[i, j, k] > _EPS):
                    continue

                v1 = _neiborghood(vol, i, j, k, radius, weight_method)

                x0, x1 = max(0, i - search[0]), min(nx - 1, i + search[0])
                y0, y1 = max(0, j - search[1]), min(ny - 1, j + search[1])
                z0, z1 = max(0, k - search[2]), min(nz - 1, k + search[2])

                w_max = _F32(0.0)
                global_sum = _F32(0.0)
                average = np.zeros(len(offs), dtype=_F32)

                for kk in range(z0, z1 + 1):
                    for jj in range(y0, y1 + 1):
                        for ii in range(x0, x1 + 1):
                            if not (mean_map[ii, jj, kk] > _EPS
                                    and var_map[ii, jj, kk] > _EPS):
                                continue
                            ratio = mean_map[i, j, k] / mean_map[ii, jj, kk]
                            ratio2 = var_map[i, j, k] / var_map[ii, jj, kk]
                            if not (m_min <= ratio <= 1.0 / m_min
                                    and v_min <= ratio2 <= 1.0 / v_min):
                                continue
                            if ii == i and jj == j and kk == k:
                                continue
                            v2 = _neiborghood(vol, ii, jj, kk, radius, weight_method)
                            # block path passes beta/2 into Weight -> denom = beta*h^2
                            w = weight(l2_norm(v1, v2), beta / 2.0, h)
                            global_sum = _F32(global_sum + w)
                            average = (average + _patch_block(vol, ii, jj, kk, radius, rican) * w).astype(_F32)
                            if w > w_max:
                                w_max = w

                # self patch gets weight w_max
                global_sum = _F32(global_sum + w_max)
                average = (average + _patch_block(vol, i, j, k, radius, rican) * w_max).astype(_F32)

                if global_sum == 0.0:
                    continue

                # Value_block / Value_block_Rician: scatter the block estimate
                block_vals = (average / global_sum).astype(_F32)
                for idx, (ox, oy, oz) in enumerate(offs):
                    x, y, z = i + ox, j + oy, k + oz
                    if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
                        continue
                    val = block_vals[idx]
                    if rican:
                        val = _F32(val - 2.0 * hf * hf)
                        val = _F32(np.sqrt(val)) if val > 0 else _F32(0.0)
                    estimate[x, y, z] = _F32(estimate[x, y, z] + val)
                    label[x, y, z] = _F32(label[x, y, z] + 1.0)

    out = np.where(label == 0, vol, estimate / np.maximum(label, _F32(1.0)))
    return out.astype(_F32)


# --------------------------------------------------------------------------- #
# Voxelwise ONLM (nl_means.cpp) -- mincnlm's block=0 path (auto-selected only
# for very small volumes: nslices < 2*nb_thread). Faithful port of Sub_denoise_mt.
# --------------------------------------------------------------------------- #
def denoise_voxel(vol, h, beta=1.0, radius=(1, 1, 1), search=(5, 5, 5),
                  m_min=0.95, v_min=0.5, weight_method=0):
    """Faithful port of denoise_mt (single-threaded, float32).

    Note vs. the block path: Weight is called with ``beta`` directly (not
    beta/2), and unprocessed voxels (local mean/var <= 1e-4) are left at 0,
    exactly as mincnlm initialises and writes the voxelwise output buffer.
    """
    if weight_method not in (0, 2):
        raise NotImplementedError("weight_method %d not supported" % weight_method)
    vol = np.ascontiguousarray(vol, dtype=_F32)
    nx, ny, nz = vol.shape

    mean_map = local_mean(vol, radius)
    var_map = local_var(vol, mean_map, radius)

    out = np.zeros(vol.shape, dtype=_F32)
    rican = weight_method == 2
    hf = _F32(h)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if not (mean_map[i, j, k] > _EPS and var_map[i, j, k] > _EPS):
                    continue

                v1 = _neiborghood(vol, i, j, k, radius, weight_method)

                x0, x1 = max(0, i - search[0]), min(nx - 1, i + search[0])
                y0, y1 = max(0, j - search[1]), min(ny - 1, j + search[1])
                z0, z1 = max(0, k - search[2]), min(nz - 1, k + search[2])

                w_max = _F32(0.0)
                global_sum = _F32(0.0)
                average = _F32(0.0)

                for kk in range(z0, z1 + 1):
                    for jj in range(y0, y1 + 1):
                        for ii in range(x0, x1 + 1):
                            if not (mean_map[ii, jj, kk] > _EPS
                                    and var_map[ii, jj, kk] > _EPS):
                                continue
                            ratio = mean_map[i, j, k] / mean_map[ii, jj, kk]
                            ratio2 = var_map[i, j, k] / var_map[ii, jj, kk]
                            if not (m_min <= ratio <= 1.0 / m_min
                                    and v_min <= ratio2 <= 1.0 / v_min):
                                continue
                            if ii == i and jj == j and kk == k:
                                continue
                            v2 = _neiborghood(vol, ii, jj, kk, radius, weight_method)
                            # voxelwise passes beta (not beta/2) into Weight
                            w = weight(l2_norm(v1, v2), beta, h)
                            global_sum = _F32(global_sum + w)
                            val_in = vol[ii, jj, kk]
                            if rican:
                                average = _F32(average + val_in * val_in * w)
                            else:
                                average = _F32(average + val_in * w)
                            if w > w_max:
                                w_max = w

                global_sum = _F32(global_sum + w_max)
                val_in = vol[i, j, k]
                if rican:
                    average = _F32(average + val_in * val_in * w_max)
                    if global_sum != 0.0:
                        dv = _F32(average / global_sum - 2.0 * hf * hf)
                        out[i, j, k] = _F32(np.sqrt(dv)) if dv > 0.0 else _F32(0.0)
                    else:
                        out[i, j, k] = val_in
                else:
                    average = _F32(average + val_in * w_max)
                    if global_sum != 0.0:
                        out[i, j, k] = _F32(average / global_sum)
                    else:
                        out[i, j, k] = val_in
    return out
