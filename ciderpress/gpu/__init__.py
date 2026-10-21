"""Process-wide switch for optional GPU (CuPy) kernels.

Off by default, so Classic GPAW, PySCF and CPU runs of the New-GPAW backend
never change. ``CiderNewMGGAFunctional`` enables it for GPU runs. Each
offloaded kernel keeps its NumPy interface (NumPy in, NumPy out) and asks
``use_gpu('<kernel name>')`` whether to run on the device. The environment
variable ``CIDER_GPU_KERNELS`` restricts the kernels: ``all`` (default),
``none``, or a comma-separated list of kernel names.
"""

import os

_enabled = False
_device = None


def _real_cupy():
    try:
        import cupy
    except ImportError:
        return None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
    except Exception:
        return None
    return cupy


def enable(device=None):
    """Enable GPU kernels on ``device`` (default: the current CuPy device)."""
    global _enabled, _device
    cupy = _real_cupy()
    if cupy is None:
        raise RuntimeError("CIDER GPU kernels need CuPy with a CUDA device")
    if device is not None:
        cupy.cuda.Device(device).use()
    _device = cupy.cuda.Device().id
    _enabled = True


def disable():
    global _enabled, _device
    _enabled = False
    _device = None


def is_enabled():
    return _enabled


def use_gpu(kernel):
    """Return True if ``kernel`` should run on the GPU."""
    if not _enabled:
        return False
    selection = os.environ.get("CIDER_GPU_KERNELS", "all").strip()
    if selection == "all":
        return True
    if selection == "none":
        return False
    return kernel in {name.strip() for name in selection.split(",")}


def cupy():
    """The real CuPy module; only call after ``use_gpu`` returned True."""
    import cupy as _cupy

    return _cupy
