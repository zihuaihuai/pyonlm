#!/usr/bin/env python
"""Generate benchmark figures from bench.json + saved slices (headless/Agg)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("BENCH_OUT", "/export03/data/enning/pyonlm/validate/bench")
FIG = OUT + "/figures"
os.makedirs(FIG, exist_ok=True)
bench = json.load(open(OUT + "/bench.json"))
results, scaling = bench["results"], bench.get("scaling", [])
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def load(name, kind):
    return np.load(f"{OUT}/slices/{name}_{kind}.npy")


def disp(a):
    return np.rot90(a)


# 1) qualitative comparison: original | pyonlm | mincnlm | difference
for r in results:
    name = r["name"]
    try:
        o, p, g = disp(load(name, "orig")), disp(load(name, "py")), disp(load(name, "gold"))
    except Exception as e:
        print("skip fig", name, e); continue
    pos = o[o > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else float(o.max() or 1)
    diff = p - g
    dm = float(np.abs(diff).max()) or 1.0
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.2))
    for a, im, ti in [(ax[0], o, "Original"), (ax[1], p, "pyonlm"), (ax[2], g, "mincnlm")]:
        a.imshow(im, cmap="gray", vmin=0, vmax=vmax); a.set_title(ti); a.axis("off")
    im3 = ax[3].imshow(diff, cmap="RdBu_r", vmin=-dm, vmax=dm)
    ax[3].set_title(f"pyonlm − mincnlm\nmax|Δ|={dm:.3g}"); ax[3].axis("off")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    fig.suptitle(f"{name}  mid-axial   corr={r['corr']:.6f}   "
                 f"mean|Δ|={r['mean_abs']:.3g}   {r['frac_within_0p1']*100:.3f}% |Δ|<0.1",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIG}/compare_{name}.png", bbox_inches="tight"); plt.close(fig)

# 2) denoising-effect zoom (primary volume): original vs pyonlm on a cropped region
try:
    name = results[0]["name"]
    o, p = disp(load(name, "orig")), disp(load(name, "py"))
    H, W = o.shape
    r0, r1, c0, c1 = H // 3, H // 3 + H // 4, W // 3, W // 3 + W // 4
    oz, pz = o[r0:r1, c0:c1], p[r0:r1, c0:c1]
    vmax = float(np.percentile(o[o > 0], 99))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.4))
    ax[0].imshow(oz, cmap="gray", vmin=0, vmax=vmax); ax[0].set_title("Original (zoom)"); ax[0].axis("off")
    ax[1].imshow(pz, cmap="gray", vmin=0, vmax=vmax); ax[1].set_title("pyonlm denoised (zoom)"); ax[1].axis("off")
    fig.suptitle(f"{name}: noise reduction (ONLM)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG}/denoise_effect.png", bbox_inches="tight"); plt.close(fig)
except Exception as e:
    print("skip denoise_effect", e)

# 3) timing bar chart: mincnlm vs pyonlm (full volume)
try:
    names = [r["name"] for r in results]
    tm = [r["t_mincnlm"] for r in results]
    tpn = [r["t_pyonlm_noise"] for r in results]
    tpd = [r["t_pyonlm_denoise"] for r in results]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, tm, w, label="mincnlm -mt6", color="#B0413E")
    ax.bar(x + w / 2, tpd, w, label="pyonlm denoise", color="#3B7EA1")
    ax.bar(x + w / 2, tpn, w, bottom=tpd, label="pyonlm noise-estimate", color="#A5C8DA")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_title("Full-volume denoise time: mincnlm vs pyonlm (both -mt 6)")
    ax.legend()
    for i in range(len(names)):
        ax.text(i + w / 2, tpd[i] + tpn[i], f"{tm[i]/(tpn[i]+tpd[i]):.2f}×",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/timing.png", bbox_inches="tight"); plt.close(fig)
except Exception as e:
    print("skip timing", e)

# 4) thread scaling
try:
    if scaling:
        nts = [s["threads"] for s in scaling]
        ts = [s["denoise_s"] for s in scaling]
        t1 = ts[0]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].plot(nts, ts, "o-", color="#3B7EA1")
        ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
        ax[0].set_xlabel("threads"); ax[0].set_ylabel("denoise time (s)")
        ax[0].set_title("pyonlm thread scaling"); ax[0].grid(True, which="both", alpha=0.3)
        sp = [t1 / t for t in ts]
        ax[1].plot(nts, sp, "o-", color="#3B7EA1", label="pyonlm speedup")
        ax[1].plot(nts, nts, "--", color="gray", label="ideal (linear)")
        ax[1].set_xscale("log", base=2); ax[1].set_yscale("log", base=2)
        ax[1].set_xlabel("threads"); ax[1].set_ylabel("speedup vs 1 thread")
        ax[1].set_title("Parallel speedup"); ax[1].legend(); ax[1].grid(True, which="both", alpha=0.3)
        sh = bench.get("scale_shape")
        fig.suptitle(f"Thread scaling on a {'×'.join(map(str, sh)) if sh else ''} brain crop", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{FIG}/scaling.png", bbox_inches="tight"); plt.close(fig)
except Exception as e:
    print("skip scaling", e)

print("FIGURES DONE:", sorted(os.listdir(FIG)))
