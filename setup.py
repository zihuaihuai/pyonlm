"""Build the optional Cython acceleration core.

The package is fully functional without it (``pyonlm`` falls back to the
pure-NumPy :mod:`pyonlm.reference` oracle), so a failed extension build must not
abort installation. OpenMP is enabled opportunistically and can be forced on/off
with the ``PYONLM_OPENMP`` environment variable (``1``/``0``).
"""
import os
import sys

from setuptools import setup
from setuptools.extension import Extension


def _openmp_flags():
    env = os.environ.get("PYONLM_OPENMP")
    if env == "0":
        return [], []
    if sys.platform == "darwin":
        # Apple clang needs libomp explicitly; default OFF unless requested.
        if env == "1":
            return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
        return [], []
    # Linux / gcc
    return ["-fopenmp"], ["-fopenmp"]


def _build_extensions():
    try:
        import numpy as np
        from Cython.Build import cythonize
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[pyonlm] Cython/numpy unavailable ({exc}); "
                         "building without the acceleration core.\n")
        return []

    comp, link = _openmp_flags()
    ext = Extension(
        "pyonlm._onlm",
        sources=["src/pyonlm/_onlm.pyx"],
        include_dirs=[np.get_include()],
        # NOTE: do NOT add -ffast-math. Its fast exp() returns +inf for large
        # negative arguments (exp(-L2/denom) with small sigma) instead of
        # underflowing to 0, corrupting weights. -O3 alone is safe and correct.
        extra_compile_args=["-O3"] + comp,
        extra_link_args=link,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    try:
        return cythonize([ext], compiler_directives={"language_level": "3"})
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[pyonlm] cythonize failed ({exc}); "
                         "building without the acceleration core.\n")
        return []


setup(ext_modules=_build_extensions())
