"""pyonlm: NIfTI-native faithful reimplementation of EZminc ``mincnlm``.

Optimized Blockwise Non-Local Means (ONLM) denoising for 3D MRI, ported
line-for-line from the EZminc C++ sources for I/O equivalence with mincnlm.
"""
from __future__ import annotations

from .onlm import denoise_block, HAVE_CYTHON
from .noise import noise_estimate, noise_estimate_full

__version__ = "0.1.0"

__all__ = [
    "denoise_block",
    "noise_estimate",
    "noise_estimate_full",
    "HAVE_CYTHON",
    "__version__",
]
