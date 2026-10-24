"""Experimental, pinned single-GPU molecular CIDER SCF.

Import this entry point before gpu4pyscf so CPU libxc consumers bind first.
See ``make_cider_calc_gpu`` for the deliberately restricted contract.
"""


def make_cider_calc_gpu(*args, **kwargs):
    from .native import preload_cpu
    preload_cpu()
    from .scf import make_cider_calc_gpu as make
    return make(*args, **kwargs)


__all__ = ["make_cider_calc_gpu"]
