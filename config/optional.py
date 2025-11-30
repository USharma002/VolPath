try:  # CuPy (GPU RBF)
    import cupy as cp
    from cupyx.scipy.interpolate import RBFInterpolator as RBFInterpolator_GPU
    HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    RBFInterpolator_GPU = None
    HAS_CUPY = False

try:  # Numba acceleration
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    njit = None
    prange = None
    HAS_NUMBA = False

try:  # Natural neighbor interpolator
    from nnpycgal.nninterpol import nninterpol
    HAS_NATURAL_NEIGHBOR = True
except ImportError:  # pragma: no cover
    nninterpol = None
    HAS_NATURAL_NEIGHBOR = False

__all__ = [
    "HAS_CUPY",
    "HAS_NUMBA",
    "HAS_NATURAL_NEIGHBOR",
    "RBFInterpolator_GPU",
    "cp",
    "njit",
    "prange",
    "nninterpol",
]
