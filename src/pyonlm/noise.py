"""Faithful NumPy port of EZminc's object-based Rician noise estimator.

Reference: Coupe, Manjon, Gedamu, Arnold, Robles, Collins, "An Object-Based
Method for Rician Noise Estimation in MR Images", MICCAI 2009, as implemented in
EZminc ``image_proc/noise_estimate.cpp`` and its dependencies
(``dwt.cpp``/``dwt_utils.cpp`` for the D4 wavelet, ``minc_histograms.{h,cpp}`` for
the anti-aliased histogram + k-means, ``fftw_blur.cpp`` for the Gaussian-gradient).

Pipeline (mincnlm ``-sigma 0``):
  1. single-level 3D Daubechies-4 DWT (LLL..HHH subbands) of the zero-padded,
     next-power-of-two-cube volume;
  2. 2-means on the LLL histogram -> object threshold; mask = LLL > threshold;
  3. FFT-based Gaussian-gradient (FWHM=1) edge removal: refine mask to
     LLL > threshold AND gradient_magnitude < median(gradient over mask);
  4. nsig = percentile50(|HHH| over refined mask) / 0.6745  (MAD estimator);
  5. Gaussian path (weight method 0): return nsig directly.
     Rician path (weight method 2): apply the Koay SNR correction.

``noise_estimate`` returns the scalar sigma; ``noise_estimate_full`` also returns
the intermediates that mincnlm ``-verbose`` prints, for validation.
"""
from __future__ import annotations

import numpy as np

# Daubechies-4 filter coefficients (dwt.cpp)
_C0 = 0.4829629131445341
_C1 = 0.8365163037378079
_C2 = 0.2241438680420134
_C3 = -0.1294095225512604


# --------------------------------------------------------------------------- #
# Single-level 3D Daubechies-4 DWT (dwt.cpp pwt == daub4, dwt_utils.cpp)
# --------------------------------------------------------------------------- #
def _daub4_forward_axis(a, axis):
    """One-level periodic D4 transform along ``axis`` -> [smooth | detail]."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    if n < 4:
        return np.moveaxis(a, 0, axis)
    nh = n // 2
    two_i = 2 * np.arange(nh)
    a0 = a[two_i]
    a1 = a[(two_i + 1) % n]
    a2 = a[(two_i + 2) % n]
    a3 = a[(two_i + 3) % n]
    smooth = _C0 * a0 + _C1 * a1 + _C2 * a2 + _C3 * a3
    detail = _C3 * a0 - _C2 * a1 + _C1 * a2 - _C0 * a3
    out = np.concatenate([smooth, detail], axis=0)
    return np.moveaxis(out, 0, axis)


def _next_pow2_cube_side(size):
    """find_nearest_square_pow2: smallest 2^p >= max(size) (exact powers map to self)."""
    m = int(max(size))
    p = 0
    mm = m
    while mm != 0:
        mm >>= 1
        p += 1
    if m == (1 << (p - 1)):
        p -= 1
    return 1 << p


def dwt_forward(vol):
    """Return the 8 subbands [LLL, ..., HHH] as in EZminc dwt_forward."""
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    n = np.array(vol.shape, dtype=np.int64)
    output_size = (n + 1) // 2
    side = _next_pow2_cube_side(n)
    padded = np.array([side, side, side], dtype=np.int64)
    pad = (padded - n) // 2  # centering offset (pad_volume)

    cube = np.zeros((side, side, side), dtype=np.float64)
    cube[pad[0]:pad[0] + n[0], pad[1]:pad[1] + n[1], pad[2]:pad[2] + n[2]] = vol

    # single-level separable D4 along each axis (order irrelevant; separable)
    t = cube
    for ax in range(3):
        t = _daub4_forward_axis(t, ax)

    pad2 = pad // 2
    half = side // 2
    subbands = []
    for j in range(8):
        bx = j & 1
        by = (j >> 1) & 1
        bz = (j >> 2) & 1
        ox = bx * half + pad2[0]
        oy = by * half + pad2[1]
        oz = bz * half + pad2[2]
        sub = t[ox:ox + output_size[0], oy:oy + output_size[1], oz:oz + output_size[2]]
        subbands.append(np.ascontiguousarray(sub))
    return subbands


# --------------------------------------------------------------------------- #
# Anti-aliased histogram (minc_histograms.h)
# --------------------------------------------------------------------------- #
class Histogram:
    def __init__(self, buckets, vmin=0.0, vmax=1.0):
        self.size = int(buckets)
        self.hist = np.zeros(self.size, dtype=np.float64)
        self.set_limits(vmin, vmax)

    def set_limits(self, vmin, vmax):
        self._min = float(vmin)
        self._max = float(vmax)
        self._range = self._max - self._min
        self._k = self.size / self._range
        self._bin = self._range / self.size

    def value(self, i):
        # NOTE: not clamped (matches C++ value(i))
        return i / self._k + self._min + 0.5 / self._k

    def _idx(self, val):
        i = int(np.floor(self._k * (val - self._min - 0.5 / self._k)))
        if i < 0:
            i = 0
        if i >= self.size:
            i = self.size - 1
        return i

    def _get(self, i):
        if i < 0:
            i = 0
        if i >= self.size:
            i = self.size - 1
        return self.hist[i]

    def seed_array(self, vals):
        """Vectorised equivalent of seed() applied to every value in ``vals``."""
        vals = np.asarray(vals, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0
        i = np.floor(self._k * (vals - self._min - 0.5 / self._k)).astype(np.int64)
        np.clip(i, 0, self.size - 1, out=i)
        last = i >= (self.size - 1)
        # last bin: deposit full count
        if np.any(last):
            np.add.at(self.hist, self.size - 1, np.count_nonzero(last))
        good = ~last
        if np.any(good):
            ig = i[good]
            vg = vals[good]
            vi = ig / self._k + self._min + 0.5 / self._k
            vi1 = (ig + 1) / self._k + self._min + 0.5 / self._k
            d1 = np.abs(vi - vg)
            d2 = np.abs(vi1 - vg)
            denom = d1 + d2
            # guard against denom==0 (val exactly on a bin centre coincidence)
            wi = np.where(denom > 0, d2 / denom, 1.0)
            wi1 = np.where(denom > 0, d1 / denom, 0.0)
            np.add.at(self.hist, ig, wi)
            np.add.at(self.hist, ig + 1, wi1)
        return int(vals.size)

    def find_percentile(self, pc):
        acc = 0.0
        j = 0
        i = 0
        broke = False
        for i in range(self.size):
            prob = self.hist[i]
            acc += prob
            if acc >= pc:
                broke = True
                break
            if prob > 0.0:
                j = i
        if not broke:
            i = self.size  # loop finished without break
        if i > 0:
            hi = self._get(i)
            k = (acc - pc) / hi if hi != 0 else 0.0
            return self.value(i) * (1.0 - k) + self.value(j) * k + 0.5 / self._k
        else:
            h0 = self.hist[0]
            k = (acc - pc) / h0 if h0 != 0 else 0.0
            return (self.value(0) + 0.5 / self._k) * (1.0 - k) + self._min * k


def build_histogram(img, mask=None, bins=2000):
    img = np.asarray(img, dtype=np.float64)
    if mask is not None:
        vals = img[np.asarray(mask, dtype=bool)]
    else:
        vals = img.ravel()
    finite = vals[np.isfinite(vals)]
    vmin = float(finite.min())
    vmax = float(finite.max())
    h = Histogram(bins, vmin, vmax)
    cnt = h.seed_array(vals)
    if cnt > 0:
        h.hist /= cnt
    return h


# --------------------------------------------------------------------------- #
# k-means on a histogram (minc_histograms.cpp)
# --------------------------------------------------------------------------- #
def _classify(hist, mu):
    centres = hist.value(np.arange(hist.size))
    # nearest-centre; ties -> lower index (matches "dist < best_dist || best_k==0")
    d = np.abs(centres[:, None] - np.asarray(mu)[None, :])
    return np.argmin(d, axis=1) + 1  # 1-indexed class labels


def _estimate_mu(hist, cls, k):
    mu = np.zeros(k, dtype=np.float64)
    counts = np.zeros(k, dtype=np.float64)
    centres = hist.value(np.arange(hist.size))
    for c in range(1, k + 1):
        sel = cls == c
        w = hist.hist[sel]
        counts[c - 1] = w.sum()
        mu[c - 1] = (w * centres[sel]).sum()
    nz = counts > 0
    mu[nz] /= counts[nz]
    return mu


def simple_k_means(hist, k_means=2, maxiter=10):
    vol_min = hist.find_percentile(0.001)
    vol_max = hist.find_percentile(0.999)
    mu = np.array([vol_min + (vol_max - vol_min) * j / (k_means - 1)
                   for j in range(k_means)], dtype=np.float64)
    for _ in range(maxiter):
        cls = _classify(hist, mu)
        mu = _estimate_mu(hist, cls, k_means)
    return mu


# --------------------------------------------------------------------------- #
# FFT Gaussian-gradient magnitude (fftw_blur.cpp)
# --------------------------------------------------------------------------- #
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _gaussian_kernel_freq(dim, sigma, deriv):
    """calculate_gaussian(): length-2*dim circular kernel, then its forward FFT."""
    n = 2 * dim
    res = np.zeros(n, dtype=np.float64)
    idx = np.arange(n)
    outer = (idx < (n // 4)) | (idx >= (3 * n // 4))
    k = np.where(idx < n / 2.0, idx, idx - n).astype(np.float64)
    g = np.exp(-k * k / (2.0 * sigma * sigma))
    s = g[outer].sum()
    if deriv:
        res[outer] = (-k[outer] / (sigma * sigma)) * g[outer]
    else:
        res[outer] = g[outer]
    res /= s
    return np.fft.fft(res)


def _convolve_axis(vol, axis, kern_freq):
    """Zero-padded circular convolution (length 2*dim) along ``axis``."""
    vol = np.moveaxis(vol, axis, -1)
    dim = vol.shape[-1]
    shape = vol.shape[:-1] + (2 * dim,)
    padded = np.zeros(shape, dtype=np.float64)
    padded[..., :dim] = vol
    conv = np.fft.ifft(np.fft.fft(padded, axis=-1) * kern_freq, axis=-1).real
    out = conv[..., :dim]
    return np.moveaxis(out, -1, axis)


def calc_gradient_mag(vol, fx=1.0, fy=1.0, fz=1.0):
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    dims = vol.shape
    sig = [abs(f) * _FWHM_TO_SIGMA for f in (fx, fy, fz)]
    # precompute freq kernels: gaussian and derivative per axis
    gauss = [_gaussian_kernel_freq(dims[ax], sig[ax], False) for ax in range(3)]
    deriv = [_gaussian_kernel_freq(dims[ax], sig[ax], True) for ax in range(3)]

    def directional(dax):
        out = vol
        # blur_volume applies Z, then Y, then X; separable -> order irrelevant
        for ax in range(3):
            kern = deriv[ax] if ax == dax else gauss[ax]
            out = _convolve_axis(out, ax, kern)
        return out

    vdx = directional(0)
    vdy = directional(1)
    vdz = directional(2)
    return np.sqrt(vdx * vdx + vdy * vdy + vdz * vdz)


# --------------------------------------------------------------------------- #
# Koay SNR correction (noise_estimate.cpp, Rician path only)
# --------------------------------------------------------------------------- #
def _epsi(snr):
    from scipy.special import i0, i1
    if snr > 37:
        return 1.0
    s2 = snr * snr
    return (2.0 + s2 - np.pi / 8.0 * np.exp(-s2 / 2.0)
            * ((2.0 + s2) * i0(s2 / 4.0) + s2 * i1(s2 / 4.0)) ** 2)


def _noise_correct(sig, nsig):
    snr1 = sig / nsig
    for _ in range(500):
        snr2 = np.sqrt(_epsi(snr1) * (1.0 + sig * sig / (nsig * nsig)) - 2.0)
        if abs(snr1 - snr2) < 1e-9:
            break
        snr1 = snr2
    return np.sqrt(nsig * nsig / _epsi(snr1))


# --------------------------------------------------------------------------- #
# Full estimator
# --------------------------------------------------------------------------- #
def noise_estimate_full(vol, gaussian=True, hist_bins=2000, verbose=False):
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    dwt = dwt_forward(vol)
    LLL = dwt[0]
    HHH = dwt[7]

    LLL_hist = build_histogram(LLL, bins=hist_bins)
    mu = simple_k_means(LLL_hist, 2, 10)
    LLL_threshold = (mu[0] + mu[1]) / 2.0

    mask = LLL > LLL_threshold

    # background stats (verbose only in C++)
    n = np.array(vol.shape)
    ax = (np.arange(n[0]) // 2)
    ay = (np.arange(n[1]) // 2)
    az = (np.arange(n[2]) // 2)
    mask_full = mask[np.ix_(ax, ay, az)]
    bkgr_vals = vol[~mask_full]
    if bkgr_vals.size > 0:
        bkgr_mean = bkgr_vals.mean()
        bkgr_std = np.sqrt(max((bkgr_vals * bkgr_vals).mean() - bkgr_mean ** 2, 0.0))
    else:
        bkgr_mean = bkgr_std = 0.0

    # edge removal via Gaussian-gradient magnitude on LLL
    LLL_gmag = calc_gradient_mag(LLL, 1.0, 1.0, 1.0)
    gmag_hist = build_histogram(LLL_gmag, mask=mask, bins=hist_bins)
    gmag_median = gmag_hist.find_percentile(0.5)

    refined = (LLL > LLL_threshold) & (LLL_gmag < gmag_median)

    HHH_abs = np.abs(HHH)
    abs_HHH_hist = build_histogram(HHH_abs, mask=refined, bins=hist_bins)
    nsig = abs_HHH_hist.find_percentile(0.5) / 0.6745  # MAD

    refined_full = refined[np.ix_(ax, ay, az)]
    sig_vals = vol[refined_full]
    mean_signal = float(sig_vals.mean()) if sig_vals.size > 0 else 0.0

    nsig_corr = nsig
    if not gaussian:
        nsig_corr = _noise_correct(mean_signal, nsig)

    info = {
        "LLL_threshold": LLL_threshold,
        "LLL_mu": tuple(mu),
        "background_mean": bkgr_mean,
        "background_std": bkgr_std,
        "gmag_median": gmag_median,
        "noise_mad": nsig,
        "mean_signal": mean_signal,
        "sigma": nsig_corr,
        "gaussian": gaussian,
    }
    if verbose:
        for kk, vv in info.items():
            print(f"[noise_estimate] {kk} = {vv}")
    return nsig_corr, mean_signal, info


def noise_estimate(vol, gaussian=True, verbose=False):
    """Return the scalar sigma mincnlm would use for ``-sigma 0``."""
    sigma, _, _ = noise_estimate_full(vol, gaussian=gaussian, verbose=verbose)
    return sigma
