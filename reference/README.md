# Reference C++ sources

`pyonlm` is a faithful port of the Optimized Blockwise Non-Local Means denoiser
and its automatic noise estimator from **EZminc** (BIC-MNI) and **libminc**.

Those upstream C++ sources are **not vendored** in this repository (to avoid
redistributing third-party code with its own licensing). To place them locally
under `reference/ezminc/` for cross-checking the port, run:

```bash
bash reference/fetch_sources.sh
```

Files fetched and the pyonlm modules they map to:

| Upstream (BIC-MNI)                                   | pyonlm |
|------------------------------------------------------|--------|
| `EZminc/mincnlm/mincnlm.cpp`                          | `cli.py` (defaults, orchestration) |
| `EZminc/mincnlm/nl_means_utils.cpp`                   | `reference.py` local_mean/local_var, `_onlm.pyx` compute_mean_var |
| `EZminc/mincnlm/nl_means.cpp`                         | `reference.py` Neiborghood/L2/Weight + voxelwise, `_onlm.pyx` |
| `EZminc/mincnlm/nl_means_block.cpp`                   | `reference.py` denoise_block, `_onlm.pyx` denoise_block |
| `EZminc/image_proc/noise_estimate.cpp`               | `noise.py` noise_estimate_full |
| `EZminc/image_proc/dwt.cpp`, `dwt_utils.cpp`          | `noise.py` dwt_forward / _daub4_forward_axis |
| `EZminc/image_proc/minc_histograms.{h,cpp}`           | `noise.py` Histogram / simple_k_means |
| `EZminc/image_proc/fftw_blur.cpp`                     | `noise.py` calc_gradient_mag |
| `libminc/ezminc/minc_io_simple_volume.h`              | volume layout / `pad_volume` semantics |

Please cite the original algorithm:

> P. Coupé, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot. An Optimized
> Blockwise Non-Local Means Denoising Filter for 3-D Magnetic Resonance Images.
> IEEE Transactions on Medical Imaging, 27(4):425–441, 2008.
>
> P. Coupé, J.V. Manjón, E. Gedamu, D. Arnold, M. Robles, D.L. Collins. An
> Object-Based Method for Rician Noise Estimation in MR Images. MICCAI 2009.
