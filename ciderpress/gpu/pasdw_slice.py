"""Exact host cache for invariant PASDW sphere interpolation functions."""
import numpy as np


def functions(atom_slice, pfunc=True):
    """Reuse the original C/libm values; detect any changed input byte.

    Return a fresh array, as get_funcs(store_funcs=False) does. A caller
    cannot poison the cached result by changing the returned work array.
    Geometry and radial-table changes, including in-place edits, invalidate
    the cache. There is no new floating-point arithmetic or device kernel.
    """
    from ciderpress.dft import pwutil

    setup = atom_slice.psetup
    table = setup.pfuncs_ntp if pfunc else setup.ffuncs_jtp
    indices = setup.nlist_i if pfunc else setup.jlist_i
    arrays = (table, atom_slice.t_g, atom_slice.dt_g, atom_slice.rhat_gv,
              indices, setup.lmlist_i)
    signature = (setup.lmax, tuple((a.shape, a.dtype.str, a.tobytes()) for a in arrays))
    cache = getattr(atom_slice, '_gpu_host_functions', None)
    if cache is None:
        cache = atom_slice._gpu_host_functions = {}
    previous = cache.get(pfunc)
    if previous is None or previous[0] != signature:
        radial = pwutil.eval_cubic_spline(table, atom_slice.t_g, atom_slice.dt_g)
        ylm = pwutil.recursive_sph_harm((setup.lmax + 1)**2, atom_slice.rhat_gv)
        value = pwutil.eval_pasdw_funcs(radial, np.ascontiguousarray(ylm.T),
                                        indices, setup.lmlist_i)
        value.flags.writeable = False
        cache[pfunc] = signature, value
    return cache[pfunc][1].copy()
