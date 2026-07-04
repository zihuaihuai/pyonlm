# pyonlm

NIfTI-native, faithful reimplementation of EZminc's `mincnlm` — Coupé et al.'s
Optimized Blockwise Non-Local Means (ONLM) denoiser for 3D MRI.

The goal is **I/O equivalence with `mincnlm`**, not "a better denoiser": it lets
MICA-style pipelines drop the C++/MINC `mincnlm` (and the `mri_convert` NIfTI↔MINC
round-trip) while producing the same denoised output.

Ported line-for-line from the EZminc C++ sources vendored under `reference/ezminc/`:
`mincnlm.cpp`, `nl_means{,_block,_utils}.cpp` (the ONLM core) and
`noise_estimate.cpp` + `dwt.cpp`, `minc_histograms.*`, `fftw_blur.cpp` (the
automatic `-sigma 0` object-based Rician noise estimator).

## Install

```bash
pip install -e .            # builds the Cython acceleration core if a compiler is present
# On Linux, OpenMP is used automatically. Force it: PYONLM_OPENMP=1 pip install -e .
```

Without a compiler/Cython the package still works, transparently falling back to the
pure-NumPy oracle in `pyonlm.reference`.

## CLI (mirrors mincnlm)

```bash
pyonlm in.nii.gz out.nii.gz -mt 6 -sigma 0 -beta 1
```

Defaults match `mincnlm`: `-sigma 0` (automatic noise estimate), `-beta 1`,
`-v 1` (3×3×3 patch), `-d 5` (11×11×11 search), `-w 0` (Gaussian/L2),
`-block 1`, `-b_space 2`, `-m_min 0.95`, `-v_min 0.5`.

This reproduces MICA's 7T `denoiseNLM` step
(`mincnlm ${mnc} ${mncdn} -mt ${threads} -sigma ${sigma} -beta ${beta}`).

## Python

```python
import nibabel as nib, numpy as np
from pyonlm import denoise_block, noise_estimate

img = nib.load("in.nii.gz")
data = img.get_fdata(dtype=np.float64)
sigma = noise_estimate(data, gaussian=True)      # -sigma 0
den = denoise_block(data, sigma, beta=1.0).astype(np.float32)
nib.save(nib.Nifti1Image(den, img.affine, img.header), "out.nii.gz")
```

## Equivalence with mincnlm

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the full benchmark on 5 real 7T
volumes (3 subjects × 3 modalities) with side-by-side images, timing, and thread
scaling. Summary: pyonlm reproduces `mincnlm -mt 6 -sigma 0 -beta 1` to float32
precision on every volume (correlation 1.0000000; 99.996–100% of voxels within
0.1), and its automatic noise estimate matches mincnlm's `Noise=` to ≈6 significant
figures.

Validated against real `mincnlm` (minc-toolkit 1.9.18) on a 7T PNI MP2RAGE UNIT1
volume (`sub-PNC001`, 0.5 mm, 320×488×520):

- **Automatic noise estimate** (`-sigma 0`): pyonlm reproduces every mincnlm
  intermediate — LLL threshold, background mean/std, median gradient magnitude,
  and the final `Noise=` value — to 6+ significant figures (e.g. full-volume
  `Noise` 97.30322 vs mincnlm's 97.3031).
- **Denoised output** (full volume, `-mt 6 -sigma 0 -beta 1`): correlation
  1.0000000, mean abs diff 0.015 and RMSE 0.030 on an intensity range of ~4000,
  identical min/max/mean, 99.999% of voxels within 0.1. The residual is pure
  float32 summation-order noise; a handful of voxels (~0.001%) sit exactly on the
  weight-underflow / block-skip boundary where float32 rounding flips the decision.
  pyonlm also ran faster than mincnlm here (~248 s vs ~366 s).

Two subtleties were essential to equivalence and are handled by the port:

1. **float32 arithmetic.** mincnlm does the whole denoise in 32-bit float. When all
   candidate weights underflow, float32 flushes `global_sum` to exactly 0 and the
   block is skipped (output = input); float64 keeps denormal weights and divides
   tiny-by-tiny into garbage. The core therefore replicates mincnlm's float32
   behaviour exactly.
2. **Thread-dependent block grid.** mincnlm restarts the block-centre stride at each
   thread's slice-partition start (`debut = floor(i·nz/mt)`), so `mincnlm -mt N`
   uses a *different, deterministic* block grid per thread count (`-mt 4` ≠ `-mt 6`).
   By default pyonlm emulates this so that `pyonlm … -mt N` reproduces
   `mincnlm … -mt N` exactly. Pass `--uniform-grid` for a clean, thread-count-independent
   grid (equivalent to `mincnlm -mt 1`), which is deterministic and reproducible.

`pyonlm`'s parallelism uses race-free per-thread accumulation, so its output is
deterministic and independent of the thread count — unlike mincnlm, whose
`Value_block` reads `Estimate` outside its mutex.

## Correctness (dev)

- `pyonlm.reference` is the pure-NumPy golden oracle (faithful float32).
- The Cython core (`pyonlm._onlm`) is validated to match the oracle to float32
  precision (`tests/test_parity.py`), including multithread and grid-emulation modes.
- Primitives (mean/var/L2/weight/patch norm) have hand-computed unit tests, and the
  noise estimator has structural + Gaussian-recovery tests.
