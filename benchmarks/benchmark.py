#!/usr/bin/env python
"""Benchmark pyonlm against mincnlm on real 7T MP2RAGE data.

Run on a host with minc-toolkit (mincnlm, mnc2nii) and FreeSurfer (mri_convert)
on PATH, and pyonlm installed. Reproduces MICA's 7T denoiseNLM step both ways:

    mincnlm:  nii --mri_convert--> mnc --mincnlm -mt 6 -sigma 0 -beta 1--> mnc --mnc2nii--> nii
    pyonlm:   nii --> pyonlm (auto sigma, -mt 6 grid emulation) --> nii

Records wall-clock times, the automatic noise estimate agreement, and voxelwise
equivalence stats; saves mid-axial slices for figures. Writes bench.json.
"""
import os, sys, re, json, time, subprocess
import numpy as np
import nibabel as nib
from pyonlm import onlm, noise as ne

R = "/data_/mica3/BIDS_PNI/rawdata"
OUT = os.environ.get("BENCH_OUT", "/export03/data/enning/pyonlm/validate/bench")
RAW = OUT + "/raw"   # persistent raw denoised volumes (mincnlm + pyonlm), for inspection
os.makedirs(OUT + "/slices", exist_ok=True)
os.makedirs(RAW, exist_ok=True)

# Explicit tool paths: FreeSurfer bundles an old MINC1-only mnc2nii that shadows
# minc-toolkit's on PATH and cannot read mincnlm's MINC2 output, so pin them.
MRI_CONVERT = os.environ.get("MRI_CONVERT", "mri_convert")
MINCNLM = os.environ.get("MINCNLM", "/opt/minc/1.9.18/bin/mincnlm")
MNC2NII = os.environ.get("MNC2NII", "/opt/minc/1.9.18/bin/mnc2nii")

VOLUMES = [
    ("PNC001-UNIT1",  f"{R}/sub-PNC001/ses-01/anat/sub-PNC001_ses-01_acq-05mm_UNIT1.nii.gz"),
    ("PNC002-UNIT1",  f"{R}/sub-PNC002/ses-01/anat/sub-PNC002_ses-01_acq-05mm_UNIT1.nii.gz"),
    ("PNC006-UNIT1",  f"{R}/sub-PNC006/ses-01/anat/sub-PNC006_ses-01_acq-05mm_UNIT1.nii.gz"),
    ("PNC001-T1map",  f"{R}/sub-PNC001/ses-01/anat/sub-PNC001_ses-01_acq-05mm_T1map.nii.gz"),
    ("PNC001-inv1",   f"{R}/sub-PNC001/ses-01/anat/sub-PNC001_ses-01_acq-05mm_inv-1_MP2RAGE.nii.gz"),
]
MT = 6


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def wall(fn):
    t = time.time()
    r = fn()
    return r, time.time() - t


def grab(log, key):
    m = re.search(key + r"=\s*([-0-9.eE]+)", log)
    return float(m.group(1)) if m else None


results = []
for name, src in VOLUMES:
    if not os.path.exists(src):
        print("SKIP missing", name, flush=True)
        continue
    try:
        img = nib.load(src)
        data = img.get_fdata(dtype=np.float64)
        mnc = f"{OUT}/{name}.mnc"; gmnc = f"{OUT}/{name}_gold.mnc"
        gnii = f"{OUT}/{name}_gold.nii"; pnii = f"{OUT}/{name}_py.nii.gz"

        _, t_conv = wall(lambda: sh(f"{MRI_CONVERT} {src} {mnc}"))
        r_minc, t_minc = wall(lambda: sh(
            f"{MINCNLM} {mnc} {gmnc} -mt {MT} -sigma 0 -beta 1 -verbose -clobber"))
        log = r_minc.stdout + r_minc.stderr
        minc_noise, minc_signal = grab(log, "Noise"), grab(log, "Signal")
        r_conv = sh(f"{MNC2NII} {gmnc} {gnii}")
        if not os.path.exists(gnii):
            raise RuntimeError("mnc2nii failed: " + (r_conv.stderr or r_conv.stdout)[-300:])
        gold = nib.as_closest_canonical(nib.load(gnii)).get_fdata(dtype=np.float64)

        sigma, t_noise = wall(lambda: ne.noise_estimate(data, gaussian=True))
        out, t_den = wall(lambda: onlm.denoise_block(
            data.astype(np.float32), sigma, beta=1.0, num_threads=MT, mt_grid=MT))
        h = img.header.copy(); h.set_data_dtype(np.float32)
        try: h.set_slope_inter(None, None)
        except Exception: pass
        nib.save(nib.Nifti1Image(out.astype(np.float32), img.affine, h), pnii)
        py = nib.as_closest_canonical(nib.load(pnii)).get_fdata(dtype=np.float64)

        diff = gold - py; ad = np.abs(diff); rng = float(gold.max() - gold.min())
        stats = dict(
            name=name, shape=list(img.shape), dtype=str(img.get_data_dtype()),
            n_voxels=int(np.prod(img.shape)),
            t_convert=t_conv, t_mincnlm=t_minc,
            t_pyonlm_noise=t_noise, t_pyonlm_denoise=t_den, t_pyonlm=t_noise + t_den,
            speedup_vs_mincnlm=t_minc / (t_noise + t_den),
            minc_noise=minc_noise, py_sigma=float(sigma),
            noise_rel_err=(abs(minc_noise - sigma) / minc_noise if minc_noise else None),
            minc_signal=minc_signal,
            corr=float(np.corrcoef(gold.ravel(), py.ravel())[0, 1]),
            max_abs=float(ad.max()), mean_abs=float(ad.mean()),
            rmse=float(np.sqrt((diff ** 2).mean())), range=rng,
            frac_within_0p01=float((ad <= 0.01).mean()),
            frac_within_0p1=float((ad <= 0.1).mean()),
            frac_within_1=float((ad <= 1.0).mean()),
            orig_stats=[float(data.min()), float(data.max()), float(data.mean())],
            gold_stats=[float(gold.min()), float(gold.max()), float(gold.mean())],
            py_stats=[float(py.min()), float(py.max()), float(py.mean())],
        )
        results.append(stats)

        orig_can = nib.as_closest_canonical(img).get_fdata(dtype=np.float64)
        k = orig_can.shape[2] // 2
        np.save(f"{OUT}/slices/{name}_orig.npy", orig_can[:, :, k])
        np.save(f"{OUT}/slices/{name}_py.npy", py[:, :, k])
        np.save(f"{OUT}/slices/{name}_gold.npy", gold[:, :, k])

        # keep the raw denoised volumes (compressed) for inspection; symlink the input
        nib.save(nib.load(gnii), f"{RAW}/{name}_mincnlm.nii.gz")
        os.replace(pnii, f"{RAW}/{name}_pyonlm.nii.gz")
        link = f"{RAW}/{name}_original.nii.gz"
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.realpath(src), link)
        except Exception:
            pass

        for fp in (mnc, gmnc, gnii):  # drop only the big intermediates
            try: os.remove(fp)
            except Exception: pass
        json.dump({"results": results}, open(f"{OUT}/results.json", "w"), indent=2)
        print(f"DONE {name}: speedup={stats['speedup_vs_mincnlm']:.2f} "
              f"corr={stats['corr']:.7f} within0.1={stats['frac_within_0p1']*100:.3f}% "
              f"noise {minc_noise} vs {sigma:.4f}", flush=True)
    except Exception as e:
        print("ERROR", name, repr(e), flush=True)

# ---- thread scaling (on a brain crop of the primary volume, denoise-only) ----
scaling = []
try:
    img = nib.load(VOLUMES[0][1])
    full = img.get_fdata(dtype=np.float64)
    crop = np.ascontiguousarray(full[72:248, 156:332, 172:348]).astype(np.float32)  # 176^3 brain
    sig = ne.noise_estimate(full, gaussian=True)
    for nt in [1, 2, 4, 6, 8, 16, 32, 64]:
        _, t = wall(lambda: onlm.denoise_block(crop, sig, beta=1.0, num_threads=nt, mt_grid=MT))
        scaling.append({"threads": nt, "denoise_s": t})
        print(f"SCALE nt={nt} {t:.1f}s", flush=True)
    scale_shape = list(crop.shape)
except Exception as e:
    print("SCALE ERROR", repr(e), flush=True)
    scale_shape = None

json.dump({"results": results, "scaling": scaling, "scale_shape": scale_shape,
           "primary": VOLUMES[0][0], "mt": MT},
          open(f"{OUT}/bench.json", "w"), indent=2)
print("BENCHMARK DONE", flush=True)
