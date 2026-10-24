"""Process-local, density-independent PASDW tables for private export engines.

Only freshly built exporter tables enter this cache. In particular, no live
SCF contribution, plan, device mirror, or FFT work area is borrowed. Host
arrays are immutable; Python owners (including lazy device-cache owners) and
the density-dependent radial plan are recreated for each private engine.
"""
from collections import OrderedDict
from copy import copy
import hashlib
from threading import RLock

import numpy as np

from ciderpress.gpaw import atom_utils
from ciderpress.gpaw.atom_utils import FastPASDWCiderKernel


def _digest(value):
    """Hash exact input contents, including array dtype/shape and object type."""
    h = hashlib.sha256()

    def visit(x):
        h.update((type(x).__module__ + '.' + type(x).__qualname__).encode())
        if isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                visit(k)
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            for item in x:
                visit(item)
        elif hasattr(x, '__dict__'):
            visit(vars(x))
        elif x is None or isinstance(x, (str, bytes, int, float, bool, np.generic)):
            h.update(repr(x).encode())
        else:
            raise TypeError('Unsupported PASDW cache input: ' + str(type(x)))
        h.update(b'\0')

    visit(value)
    return h.digest()


def _tables_copy(value, *, freeze=False, memo=None):
    """Copy table owners, preserving native ownership and immutable arrays.

    ATCBasis and SBT descriptors own C allocations and must never be shallow
    copied (their destructors would double-free). They have no density state
    and are used read-only after table construction. They originate solely
    in the cache's private build, never in the live calculator.
    """
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray):
        result = value.copy(order='K') if freeze else value
        if freeze:
            result.flags.writeable = False
    elif isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update({k: _tables_copy(v, freeze=freeze, memo=memo)
                       for k, v in value.items()})
    elif isinstance(value, (list, tuple)):
        result = type(value)(_tables_copy(v, freeze=freeze, memo=memo) for v in value)
    elif (hasattr(value, '__dict__') and not isinstance(value, type)
          and not hasattr(type(value), '__del__')):
        result = copy(value)
        memo[id(value)] = result
        result.__dict__ = _tables_copy(vars(value), freeze=freeze, memo=memo)
    else:
        result = value
    memo[id(value)] = result
    return result


class ExportPAWCache:
    """Bounded LRU of per-setup host tables; ``clear()`` also resets counters.

    Pass an instance as ``paw_cache=cache`` to either exporter, or pass None
    to disable reuse. The module's DEFAULT_PAW_CACHE is used by default.
    Keys conservatively include model/NLDF/PASDW settings, exact source setup
    contents and fingerprint, the actual plan, CUDA device and FP64 dtype.
    Geometry, density and spin count are excluded: only the fresh radial
    plan carries spin-dependent cutoffs and interpolation scratch.
    """

    def __init__(self, maxsize=16):
        if not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError('maxsize must be a positive integer')
        self.maxsize = maxsize
        self._lock = RLock()
        self.clear()

    def clear(self):
        with self._lock:
            self._entries = OrderedDict()
            self._hits = self._misses = 0

    def info(self):
        with self._lock:
            return dict(size=len(self._entries), maxsize=self.maxsize,
                        hits=self._hits, misses=self._misses)

    @staticmethod
    def _key(setup, kernel, signature):
        original_xcc = getattr(setup, 'old_xc_correction', setup.xc_correction)
        source = (getattr(setup, 'fingerprint', None), setup.Z, setup.rgd,
                  setup.rcut_j, original_xcc)
        p = kernel.plan
        plan = (type(p).__module__, type(p).__qualname__, p.nldf_settings,
                p.alphas, p.alpha_norms, p.alpha_formula, p.coef_order,
                p.proc_inds, p.rhocut * p.nspin, p.expcut,
                p._use_smooth_expnt_cutoff, p._raise_large_expnt_error,
                getattr(p, '_spline_size', None),
                getattr(p, 'smooth_floor_width', 0.0))
        return _digest((source, signature, plan, kernel._paw_algo,
                        kernel.cut_xcgrid, atom_utils.USE_GAUSSIAN_PAW_CONV,
                        atom_utils.USE_GAUSSIAN_SBT))

    def initialize(self, kernel, signature):
        kernel.nspin = kernel.dens.nt_sg.shape[0]
        # The engine's setup aliases are preserved, but initialized only once.
        seen = set()
        for setup in kernel.setups:
            if id(setup) in seen:
                continue
            seen.add(id(setup))
            key = self._key(setup, kernel, signature)
            with self._lock:
                if key not in self._entries:
                    # Build the common base tables, independent of the feature
                    # contribution's extra methods. No density call occurs here.
                    builder = copy(kernel)
                    builder.PAWCiderContribs = atom_utils.PAWCiderContribs
                    scratch = copy(setup)
                    scratch.cider_contribs = None
                    FastPASDWCiderKernel.initialize_more_things(builder, [scratch])
                    # The fixed radial map is also density independent. Build
                    # its host matrices once; each engine gets a separate map
                    # owner/device cache over the same immutable matrices.
                    if hasattr(scratch.ps_setup, '_get_radial_transform'):
                        scratch.ps_setup._get_radial_transform(scratch.ps_setup._grid_nlm)
                    contrib = scratch.cider_contribs
                    state = {k: v for k, v in vars(contrib).items()
                             if k not in ('plan', 'cider_kernel', 'timer')}
                    tables = {name: getattr(scratch, name) for name in
                              ('xc_correction', 'nlxc_correction', 'ps_setup', 'pa_setup')}
                    self._entries[key] = (_tables_copy((tables, state), freeze=True),
                                          _tables_copy(contrib.plan, freeze=True))
                    self._misses += 1
                    while len(self._entries) > self.maxsize:
                        self._entries.popitem(last=False)
                else:
                    self._hits += 1
                self._entries.move_to_end(key)
                (tables, state), plan = self._entries[key]
                tables, state = _tables_copy((tables, state))
                # new() creates empty spin-specific interpolation caches and
                # retains the smooth-floor variant and all construction flags.
                radial_plan = plan.new(
                    nldf_settings=_tables_copy(plan.nldf_settings),
                    nspin=kernel.nspin, spline_size=plan._spline_size,
                    use_smooth_expnt_cutoff=plan._use_smooth_expnt_cutoff,
                    raise_large_expnt_error=plan._raise_large_expnt_error)
                contrib = object.__new__(kernel.PAWCiderContribs)
                contrib.__dict__.update(state, plan=radial_plan,
                                        cider_kernel=kernel.cider_kernel)
                from gpaw.utilities.timing import Timer
                contrib.timer = Timer()
                for name, value in tables.items():
                    setattr(setup, name, value)
                setup.cider_contribs = contrib


DEFAULT_PAW_CACHE = ExportPAWCache()
