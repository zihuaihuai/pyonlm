#!/usr/bin/env bash
# Fetch the EZminc / libminc C++ sources that pyonlm is ported from.
#
# These third-party sources are NOT redistributed in this repository; run this
# script to place them under reference/ezminc/ for cross-checking the port.
#
#   Upstream: https://github.com/BIC-MNI/EZminc  and  https://github.com/BIC-MNI/libminc
#
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst="$here/ezminc"
mkdir -p "$dst/image_proc"

ez="https://raw.githubusercontent.com/BIC-MNI/EZminc/master"
lm="https://raw.githubusercontent.com/BIC-MNI/libminc/master/ezminc"

# ONLM core (EZminc/mincnlm)
for f in mincnlm.cpp minc_anlm.cpp nl_means.cpp nl_means.h \
         nl_means_block.cpp nl_means_block.h nl_means_utils.cpp nl_means_utils.h; do
  curl -fsSL "$ez/mincnlm/$f" -o "$dst/$f"
done

# Noise estimate + dependencies (EZminc/image_proc)
for f in noise_estimate.cpp noise_estimate.h dwt.cpp dwt.h dwt_utils.cpp dwt_utils.h \
         minc_histograms.cpp minc_histograms.h fftw_blur.cpp fftw_blur.h; do
  curl -fsSL "$ez/image_proc/$f" -o "$dst/image_proc/$f"
done
# a couple also mirrored at the top level for convenience
cp "$dst/image_proc/noise_estimate.cpp" "$dst/noise_estimate.cpp"
cp "$dst/image_proc/noise_estimate.h" "$dst/noise_estimate.h"

# simple_volume IO layer (libminc)
for f in minc_io_simple_volume.h minc_io_fixed_vector.h; do
  curl -fsSL "$lm/$f" -o "$dst/image_proc/$f"
done

echo "Fetched EZminc/libminc reference sources into $dst"
