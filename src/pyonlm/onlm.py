"""Public denoising entry point.

Dispatches to the compiled Cython core (:mod:`pyonlm._onlm`) when available and
falls back to the pure-NumPy oracle (:mod:`pyonlm.reference`) otherwise. Both
paths implement the identical blockwise ONLM math with float64 accumulation.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised via installed builds
    from . import _onlm  # type: ignore

    HAVE_CYTHON = True
except Exception:  # pragma: no cover
    _onlm = None
    HAVE_CYTHON = False

from . import reference


def denoise_block(vol, h, beta=1.0, radius=(1, 1, 1), search=(5, 5, 5),
                  b_space=2, m_min=0.95, v_min=0.5, weight_method=0,
                  num_threads=1, mt_grid=0, force_reference=False):
    """Blockwise ONLM denoise a 3D volume.

    Parameters mirror mincnlm: ``radius`` = -v, ``search`` = -d, ``h`` = sigma,
    ``beta`` = -beta, ``weight_method`` = -w (0 Gaussian L2, 2 Rician).

    Returns a float32 array (mincnlm operates in float32; the core replicates
    that, including its underflow-driven block-skip behaviour).
    """
    vol = np.ascontiguousarray(vol, dtype=np.float32)
    radius = tuple(int(r) for r in radius)
    search = tuple(int(s) for s in search)
    if HAVE_CYTHON and not force_reference:
        return _onlm.denoise_block(
            vol, float(h), float(beta), radius, search,
            int(b_space), float(m_min), float(v_min), int(weight_method),
            int(num_threads), int(mt_grid))
    return reference.denoise_block(
        vol, float(h), beta=float(beta), radius=radius, search=search,
        b_space=int(b_space), m_min=float(m_min), v_min=float(v_min),
        weight_method=int(weight_method), mt_grid=int(mt_grid))
