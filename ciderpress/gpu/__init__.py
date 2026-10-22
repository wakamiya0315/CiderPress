"""Default policy and evaluation scopes for optional GPU (CuPy) kernels.

Off by default, so Classic GPAW, PySCF and CPU runs of the New-GPAW backend
never change. ``CiderNewMGGAFunctional`` scopes it to its own device for each
evaluation, restoring the caller's policy and device afterwards. Each
offloaded kernel keeps its NumPy interface (NumPy in, NumPy out) and asks
``use_gpu('<kernel name>')`` whether to run on the device. The environment
variable ``CIDER_GPU_KERNELS`` restricts the kernels: ``all`` (default),
``none``, or a comma-separated list of kernel names.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar

_enabled = False
_device = None
_policy = ContextVar('cider_gpu_policy', default=None)


@contextmanager
def policy(device=None):
    """Temporarily select this evaluation's device; None selects CPU.

    Nested evaluations and exceptions restore both dispatch and CUDA device.
    The explicit enable()/disable() policy outside this scope is untouched.
    """
    token = _policy.set((device is not None, device))
    try:
        if device is None:
            yield
        else:
            cp = _real_cupy()
            if cp is None:
                raise RuntimeError('CIDER GPU kernels need CuPy with a CUDA device')
            with cp.cuda.Device(device):
                yield
    finally:
        _policy.reset(token)


def _real_cupy():
    if os.environ.get("GPAW_CPUPY"):
        return None
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
    state = _policy.get()
    return _enabled if state is None else state[0]


def use_gpu(kernel):
    """Return True if ``kernel`` should run on the GPU."""
    if not is_enabled():
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


def preflight(kernel, nspins):
    """Reject selected New-CIDER kernels that would fall back to the host.

    The declared hybrid PAW algorithm still uses host fits and quadratures.
    Selection dependencies are checked before any density evaluation.
    """
    if not is_enabled():
        return
    known = {'rbf', 'libxc', 'semilocal', 'pasdw', 'pasdw_fit',
             'fine_grid', 'grid_xc', 'grid_nldf', 'nldf_fft'}
    selection = os.environ.get('CIDER_GPU_KERNELS', 'all').strip()
    selected = (known if selection == 'all' else set() if selection == 'none'
                else {name.strip() for name in selection.split(',')})
    unknown = selected - known
    if unknown:
        raise NotImplementedError('Unknown selected CIDER GPU kernels: ' + ','.join(sorted(unknown)))
    if not selected:
        return
    if nspins not in (1, 2):
        raise NotImplementedError('Selected CIDER GPU kernels require one or two spins')
    for name, dependencies in {'semilocal': {'rbf', 'libxc'},
                                'grid_xc': {'semilocal', 'rbf', 'libxc'},
                                'grid_nldf': {'nldf_fft'},
                                'pasdw_fit': {'pasdw'}}.items():
        if name in selected and not dependencies <= selected:
            raise NotImplementedError('Selected CIDER GPU kernel ' + name +
                                      ' requires ' + ','.join(sorted(dependencies)))
    from ciderpress.gpu.semilocal import supported
    if selected & {'rbf', 'libxc', 'semilocal', 'pasdw'}:
        if not supported(kernel.mlfunc, kernel.mlfunc.settings.nfeat):
            raise NotImplementedError('Selected CIDER GPU kernel does not support this model/spin count')
    if 'grid_xc' in selected:
        from ciderpress.gpu.grid_xc import supported as grid_supported
        if not grid_supported(kernel):
            raise NotImplementedError('Selected CIDER GPU grid_xc does not support this model')
    if selected & {'grid_nldf', 'nldf_fft', 'pasdw'}:
        settings = kernel.mlfunc.settings.nldf_settings
        if settings.version != 'j' or settings.sl_level != 'MGGA':
            raise NotImplementedError('Selected CIDER GPU NLDF/PAW requires version-j MGGA')
