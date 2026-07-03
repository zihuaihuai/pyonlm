"""Command-line interface mirroring EZminc ``mincnlm``, but NIfTI-native.

Flag names and defaults follow mincnlm exactly (single-dash long options, as with
MINC's ParseArgv), so an invocation like::

    pyonlm in.nii.gz out.nii.gz -mt 6 -sigma 0 -beta 1

reproduces MICA's 7T ``denoiseNLM`` step without the MINC/mri_convert round-trip.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def build_parser():
    p = argparse.ArgumentParser(
        prog="pyonlm",
        description="NIfTI-native faithful reimplementation of EZminc mincnlm "
                    "(Optimized Blockwise Non-Local Means).",
        allow_abbrev=False,
    )
    p.add_argument("infile", help="input NIfTI (.nii/.nii.gz)")
    p.add_argument("outfile", help="output NIfTI (.nii/.nii.gz)")
    p.add_argument("-sigma", type=float, default=0.0,
                   help="Sigma value [0 = Automatic, default]")
    p.add_argument("-beta", type=float, default=1.0, help="Beta value [default 1]")
    p.add_argument("-v", type=float, default=1.0, dest="S",
                   help="Neighborhood radius: 1 -> 3x3x3 [default], 2 -> 5x5x5")
    p.add_argument("-d", type=float, default=5.0, dest="M",
                   help="Search radius [default 5 -> 11x11x11]")
    p.add_argument("-w", type=int, default=0, dest="weight_method",
                   help="Weighting: 0 L2/Gaussian [default], 2 L2+Rician bias correction")
    p.add_argument("-block", type=int, default=1,
                   help="Blockwise NL-means [default 1]")
    p.add_argument("-b_space", type=int, default=2,
                   help="Distance between blocks [default 2]")
    p.add_argument("-m_min", type=float, default=0.95,
                   help="Lowest bound of mean ratio [default 0.95]")
    p.add_argument("-v_min", type=float, default=0.5,
                   help="Lowest bound of variance ratio [default 0.5]")
    p.add_argument("-mt", type=int, default=4, dest="nb_thread",
                   help="Number of threads [default 4]")
    p.add_argument("-verbose", action="store_true", help="Print extra information.")
    p.add_argument("-clobber", action="store_true", help="Overwrite existing output.")
    p.add_argument("-out_dtype", default=None,
                   help="Force output dtype (default: float32).")
    p.add_argument("--uniform-grid", action="store_true",
                   help="Use a thread-count-independent block grid (deterministic; "
                        "equals mincnlm -mt 1) instead of emulating mincnlm's per-mt "
                        "block grid. Default: match mincnlm -mt <mt> exactly.")
    p.add_argument("--force-reference", action="store_true",
                   help="Use the pure-NumPy oracle instead of the Cython core.")
    return p


def run(data, sigma, beta, S, M, weight_method, block, b_space, m_min, v_min,
        nb_thread, verbose=False, uniform_grid=False, force_reference=False):
    """Denoise a 3D array following mincnlm's orchestration (Exec)."""
    from . import onlm, reference, noise as _ne

    data = np.ascontiguousarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("Only 3D volumes are supported (got shape %r)" % (data.shape,))

    if weight_method == 1:
        raise NotImplementedError("Pearson/Speckle (-w 1) is not ported; give -sigma.")
    if block not in (0, 1):
        raise ValueError("-block must be 0 (voxelwise) or 1 (blockwise); got %d" % block)

    nz = data.shape[2]
    if block == 1 and nz < 2 * nb_thread:
        if verbose:
            print("[pyonlm] few slices (<2*mt) -> voxelwise mode (block=0)")
        block = 0

    radius = (int(S), int(S), int(S))
    search = (int(M), int(M), int(M))

    # mincnlm hard-aborts block mode when the patch radius is below b_space/2
    if block == 1 and radius[0] < (b_space // 2):
        raise ValueError("blockwise mode requires neighborhood radius (-v) >= "
                         "b_space/2 (got radius=%d, b_space=%d)" % (radius[0], b_space))

    if sigma == 0.0:
        gaussian = (weight_method == 0)
        sigma = _ne.noise_estimate(data, gaussian=gaussian, verbose=verbose)
        if verbose:
            print(f"[pyonlm] automatic sigma = {sigma}")

    if verbose:
        print(f"[pyonlm] sigma={sigma} beta={beta} radius={radius} search={search} "
              f"weight_method={weight_method} block={block} b_space={b_space} "
              f"m_min={m_min} v_min={v_min}")

    if block == 1:
        mt_grid = 0 if uniform_grid else nb_thread
        out = onlm.denoise_block(
            data, sigma, beta=beta, radius=radius, search=search, b_space=b_space,
            m_min=m_min, v_min=v_min, weight_method=weight_method,
            num_threads=nb_thread, mt_grid=mt_grid, force_reference=force_reference)
    else:
        out = reference.denoise_voxel(
            data, sigma, beta=beta, radius=radius, search=search,
            m_min=m_min, v_min=v_min, weight_method=weight_method)
    return out


def main(argv=None):
    import os
    import nibabel as nib

    args = build_parser().parse_args(argv)

    if os.path.exists(args.outfile) and not args.clobber:
        sys.stderr.write(f"{args.outfile} exists! (use -clobber to overwrite)\n")
        return 1

    img = nib.load(args.infile)
    if img.ndim != 3:
        sys.stderr.write("Only 3D volumes are supported.\n")
        return 1
    data = img.get_fdata(dtype=np.float64)

    out = run(data, args.sigma, args.beta, args.S, args.M, args.weight_method,
              args.block, args.b_space, args.m_min, args.v_min, args.nb_thread,
              verbose=args.verbose, uniform_grid=args.uniform_grid,
              force_reference=args.force_reference)

    out_dtype = np.dtype(args.out_dtype) if args.out_dtype else np.float32
    out = out.astype(out_dtype)

    # preserve affine + header geometry; refresh the datatype/scaling
    header = img.header.copy()
    header.set_data_dtype(out_dtype)
    try:
        header.set_slope_inter(None, None)
    except Exception:
        pass
    out_img = nib.Nifti1Image(out, img.affine, header)
    nib.save(out_img, args.outfile)
    if args.verbose:
        print(f"[pyonlm] wrote {args.outfile}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
