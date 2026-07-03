# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Fast Cython core for the blockwise Optimized NL-means (ONLM) denoiser.

Line-for-line faithful to EZminc ``nl_means_block.cpp`` (the mincnlm default,
``block=1``), INCLUDING its float32 arithmetic.

Faithfulness note: mincnlm does the entire denoise in ``float`` (32-bit) -- the
mean/variance maps, patch vectors, weights, the ``global_sum``/``average``
accumulators, and the ``Estimate``/``Label`` buffers are all float32; only the L2
sum and ``exp`` are evaluated in double and cast back to float. This matters:
when every candidate weight underflows, float32 flushes ``global_sum`` to exactly
0 and the block is skipped (output = input), whereas float64 keeps denormal
weights and divides tiny-by-tiny into garbage. We replicate the float32 behaviour.

Parallelism: the C++ uses pthreads with a lock-protected write but an *unlocked*
read of Estimate -> lost-update races at thread-slab boundaries, non-deterministic
across thread counts. We instead give each thread its own accumulation buffer and
reduce afterwards -- race-free and deterministic; ``num_threads=1`` reproduces the
single-threaded result exactly.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, sqrt
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel, threadid

cnp.import_array()

DEF EPS = 1e-4


cdef inline void _neiborghood(float[:, :, ::1] vol, int i, int j, int k,
                              int rx, int ry, int rz,
                              int nx, int ny, int nz, int wm,
                              float* out) noexcept nogil:
    """Neiborghood(): patch vector, oob=0, normalise by sqrt(nb_inside) if wm in {0,2}."""
    cdef int count = 0
    cdef int nb_inside = 0
    cdef int a, b, c, x, y, z
    cdef int size = (2 * rx + 1) * (2 * ry + 1) * (2 * rz + 1)
    cdef float s
    for c in range(2 * rz + 1):
        for b in range(2 * ry + 1):
            for a in range(2 * rx + 1):
                x = i + a - rx
                y = j + b - ry
                z = k + c - rz
                if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                    out[count] = vol[x, y, z]
                    nb_inside += 1
                else:
                    out[count] = 0.0
                count += 1
    if wm == 0 or wm == 2:
        s = sqrt(<float> nb_inside)
        for count in range(size):
            out[count] = out[count] / s


cdef inline double _l2(float* v1, float* v2, int size) noexcept nogil:
    """L2_norm: accumulate in double (like C++), from float32 inputs."""
    cdef double result = 0.0
    cdef double d
    cdef int idx
    for idx in range(size):
        d = <double>v1[idx] - <double>v2[idx]
        result += d * d
    return result


cdef void _denoise_kblock(int k, int tid,
                          float[:, :, ::1] vol, float[:, :, ::1] mean_map,
                          float[:, :, ::1] var_map,
                          float[:, :, :, ::1] estb, float[:, :, :, ::1] labb,
                          int nx, int ny, int nz, int rx, int ry, int rz,
                          int sx, int sy, int sz, int b_space, int wm, int rican,
                          double denom, float hf,
                          double m_min, double inv_m_min, double v_min, double inv_v_min,
                          int size, float* v1, float* v2, float* average) noexcept nogil:
    """Process every block centre (i, j, k) on the b_space grid for a fixed k."""
    cdef int i, j, ii, jj, kk, idx, count
    cdef int x0, x1, y0, y1, z0, z1, a, b, c, x, y, z
    cdef float w_max, global_sum, ratio, ratio2, w, val

    j = 0
    while j < ny:
        i = 0
        while i < nx:
            if not (mean_map[i, j, k] > EPS and var_map[i, j, k] > EPS):
                i += b_space
                continue

            _neiborghood(vol, i, j, k, rx, ry, rz, nx, ny, nz, wm, v1)

            x0 = 0 if i - sx < 0 else i - sx
            x1 = nx - 1 if i + sx > nx - 1 else i + sx
            y0 = 0 if j - sy < 0 else j - sy
            y1 = ny - 1 if j + sy > ny - 1 else j + sy
            z0 = 0 if k - sz < 0 else k - sz
            z1 = nz - 1 if k + sz > nz - 1 else k + sz

            w_max = 0.0
            global_sum = 0.0
            for idx in range(size):
                average[idx] = 0.0

            for kk in range(z0, z1 + 1):
                for jj in range(y0, y1 + 1):
                    for ii in range(x0, x1 + 1):
                        if not (mean_map[ii, jj, kk] > EPS and var_map[ii, jj, kk] > EPS):
                            continue
                        ratio = mean_map[i, j, k] / mean_map[ii, jj, kk]
                        ratio2 = var_map[i, j, k] / var_map[ii, jj, kk]
                        if not (m_min <= ratio <= inv_m_min and
                                v_min <= ratio2 <= inv_v_min):
                            continue
                        if ii == i and jj == j and kk == k:
                            continue
                        _neiborghood(vol, ii, jj, kk, rx, ry, rz, nx, ny, nz, wm, v2)
                        if hf != 0:
                            w = <float>exp(-<double>(<float>_l2(v1, v2, size)) / denom)
                        else:
                            w = 0.0
                        global_sum = global_sum + w
                        count = 0
                        for c in range(2 * rz + 1):
                            for b in range(2 * ry + 1):
                                for a in range(2 * rx + 1):
                                    x = ii + a - rx
                                    y = jj + b - ry
                                    z = kk + c - rz
                                    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                                        val = vol[x, y, z]
                                    else:
                                        val = vol[ii, jj, kk]
                                    if rican:
                                        average[count] = average[count] + val * val * w
                                    else:
                                        average[count] = average[count] + val * w
                                    count += 1
                        if w > w_max:
                            w_max = w

            # self patch with weight w_max
            global_sum = global_sum + w_max
            count = 0
            for c in range(2 * rz + 1):
                for b in range(2 * ry + 1):
                    for a in range(2 * rx + 1):
                        x = i + a - rx
                        y = j + b - ry
                        z = k + c - rz
                        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                            val = vol[x, y, z]
                        else:
                            val = vol[i, j, k]
                        if rican:
                            average[count] = average[count] + val * val * w_max
                        else:
                            average[count] = average[count] + val * w_max
                        count += 1

            if global_sum != 0.0:
                count = 0
                for c in range(2 * rz + 1):
                    for b in range(2 * ry + 1):
                        for a in range(2 * rx + 1):
                            x = i + a - rx
                            y = j + b - ry
                            z = k + c - rz
                            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                                val = average[count] / global_sum
                                if rican:
                                    val = val - 2.0 * hf * hf
                                    if val > 0.0:
                                        val = sqrt(val)
                                    else:
                                        val = 0.0
                                estb[tid, x, y, z] = estb[tid, x, y, z] + val
                                labb[tid, x, y, z] = labb[tid, x, y, z] + 1.0
                            count += 1
            i += b_space
        j += b_space


def compute_mean_var(cnp.ndarray vol_in, int rx, int ry, int rz):
    """Return (mean_map, var_map) as float32, matching Preprocessing/Preprocessing2.

    Accumulated in double and stored float32 (matches a float64-accurate value
    rounded to the float32 storage the C++ uses).
    """
    cdef float[:, :, ::1] vol = np.ascontiguousarray(vol_in, dtype=np.float32)
    cdef int nx = vol.shape[0], ny = vol.shape[1], nz = vol.shape[2]
    cdef cnp.ndarray[cnp.float32_t, ndim=3] mean_arr = np.zeros((nx, ny, nz), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] var_arr = np.zeros((nx, ny, nz), dtype=np.float32)
    cdef float[:, :, ::1] mean_map = mean_arr
    cdef float[:, :, ::1] var_map = var_arr
    cdef int i, j, k, a, b, c, x, y, z, nb_inside
    cdef double s, d
    with nogil:
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    s = 0.0
                    nb_inside = 0
                    for c in range(2 * rz + 1):
                        for b in range(2 * ry + 1):
                            for a in range(2 * rx + 1):
                                x = i + a - rx
                                y = j + b - ry
                                z = k + c - rz
                                if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                                    s = s + <double>vol[x, y, z]
                                    nb_inside += 1
                    mean_map[i, j, k] = <float>(s / nb_inside)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    s = 0.0
                    nb_inside = 0
                    for c in range(2 * rz + 1):
                        for b in range(2 * ry + 1):
                            for a in range(2 * rx + 1):
                                x = i + a - rx
                                y = j + b - ry
                                z = k + c - rz
                                if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                                    d = <double>vol[x, y, z] - <double>mean_map[x, y, z]
                                    s = s + d * d
                                    nb_inside += 1
                    if nb_inside - 1 >= 1:
                        var_map[i, j, k] = <float>(s / (nb_inside - 1))
                    else:
                        var_map[i, j, k] = <float>s
    return mean_arr, var_arr


def denoise_block(cnp.ndarray vol_in,
                  double h, double beta=1.0,
                  radius=(1, 1, 1), search=(5, 5, 5),
                  int b_space=2, double m_min=0.95, double v_min=0.5,
                  int weight_method=0, int num_threads=1, int mt_grid=0):
    """Faithful float32 port of denoise_block_mt.

    Parameters mirror mincnlm: ``radius`` = -v, ``search`` = -d, ``h`` = sigma,
    ``beta`` = -beta, ``weight_method`` = -w (0 Gaussian L2, 2 Rician).

    ``num_threads`` only distributes work (race-free, result-independent).

    ``mt_grid`` selects the block-centre grid along the last (slice) axis:
      * ``0`` -> uniform stride from 0 (deterministic, thread-count independent;
        equals mincnlm ``-mt 1``);
      * ``N > 1`` -> emulate mincnlm's per-thread ``debut``-anchored grid for
        ``-mt N`` (block centres restart the b_space stride at each slice
        partition ``floor(i*nz/N)``), reproducing mincnlm ``-mt N`` exactly.
    """
    if weight_method not in (0, 2):
        raise NotImplementedError("weight_method %d not supported (only 0 Gaussian, "
                                  "2 Rician)" % weight_method)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] vol_arr = np.ascontiguousarray(vol_in, dtype=np.float32)
    cdef float[:, :, ::1] vol = vol_arr
    cdef int nx = vol.shape[0], ny = vol.shape[1], nz = vol.shape[2]
    cdef int rx = radius[0], ry = radius[1], rz = radius[2]
    cdef int sx = search[0], sy = search[1], sz = search[2]
    cdef int wm = weight_method
    cdef int rican = 1 if weight_method == 2 else 0
    cdef int nt = num_threads if num_threads > 0 else 1

    mean_arr, var_arr = compute_mean_var(vol_arr, rx, ry, rz)
    cdef float[:, :, ::1] mean_map = mean_arr
    cdef float[:, :, ::1] var_map = var_arr

    # per-thread accumulation buffers (race-free); reduced after the parallel loop
    cdef cnp.ndarray est_buf = np.zeros((nt, nx, ny, nz), dtype=np.float32)
    cdef cnp.ndarray lab_buf = np.zeros((nt, nx, ny, nz), dtype=np.float32)
    cdef float[:, :, :, ::1] estb = est_buf
    cdef float[:, :, :, ::1] labb = lab_buf

    cdef int size = (2 * rx + 1) * (2 * ry + 1) * (2 * rz + 1)

    # block-centre k-planes: uniform stride from 0, or mincnlm's debut-anchored
    # per-partition grid for -mt mt_grid (reproduces mincnlm -mt N exactly).
    if mt_grid > 1:
        kc_list = []
        for _it in range(mt_grid):
            _debut = (_it * nz) // mt_grid
            _fin = ((_it + 1) * nz) // mt_grid
            _kk = _debut
            while _kk < _fin:
                kc_list.append(_kk)
                _kk += b_space
        kcenters = np.array(kc_list, dtype=np.int32)
    else:
        kcenters = np.arange(0, nz, b_space, dtype=np.int32)
    cdef int[::1] kc = kcenters
    cdef int n_kblocks = kcenters.shape[0]

    cdef int block_k, tid
    cdef float hf = <float>h
    # denom = 2 * (beta/2) * h^2 ; mincnlm computes beta/2 and h in float, denom in double
    cdef double denom = <double>(<float>(beta / 2.0)) * 2.0 * <double>hf * <double>hf
    # pass 1/m_min and 1/v_min so the gate reads inv_m_min >= ratio >= m_min
    cdef double inv_m_min = 1.0 / m_min
    cdef double inv_v_min = 1.0 / v_min
    cdef float* v1
    cdef float* v2
    cdef float* average

    with nogil, parallel(num_threads=nt):
        v1 = <float*> malloc(size * sizeof(float))
        v2 = <float*> malloc(size * sizeof(float))
        average = <float*> malloc(size * sizeof(float))
        if v1 != NULL and v2 != NULL and average != NULL:
            for block_k in prange(n_kblocks, schedule='static'):
                tid = threadid()
                _denoise_kblock(kc[block_k], tid, vol, mean_map, var_map,
                                estb, labb, nx, ny, nz, rx, ry, rz, sx, sy, sz,
                                b_space, wm, rican, denom, hf,
                                m_min, inv_m_min, v_min, inv_v_min,
                                size, v1, v2, average)
        if v1 != NULL: free(v1)
        if v2 != NULL: free(v2)
        if average != NULL: free(average)

    est_arr = est_buf.sum(axis=0)
    lab_arr = lab_buf.sum(axis=0)
    out = np.where(lab_arr == 0.0, vol_arr, est_arr / np.maximum(lab_arr, np.float32(1.0)))
    return out.astype(np.float32)
