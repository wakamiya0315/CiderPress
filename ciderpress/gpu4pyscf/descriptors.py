"""Experimental E0 export on SCF onsite_direct grids (not train_gen)."""
import numpy as np
from .numint import CiderNumInt


class SCFFeatureGenerator:
    """One frozen GPU SCF density -> FT feature_generator callable.

    Reuses the actual SCF NLDF interpolator and grid. Its raw features differ
    from the stock CPU training interpolator; this class makes no claim of
    suitability for training. Arrays are intentionally transferred for export.
    """
    def __init__(self, mf, dm=None):
        import cupy as cp
        from ciderpress.gpu import policy
        if not mf.converged or not isinstance(mf._numint, CiderNumInt):
            raise ValueError("A converged GPU CIDER calculator is required")
        ni = mf._numint
        density = cp.asarray(mf.make_rdm1() if dm is None else dm).copy()
        self.dm = cp.asnumpy(density)
        self.coords = ni.cpu_shell.grids.coords.copy()
        self.weights = ni.cpu_shell.grids.weights.copy()
        ni.capture_density = True
        try:
            evaluate = ni.nr_uks if density.ndim == 3 else ni.nr_rks
            evaluate(mf.mol, mf.grids, mf.xc, density)
        finally:
            ni.capture_density = False
        physical = ni.last_rho
        ns = 2 if physical.ndim == 3 else 1
        rho = physical if ns == 2 else physical[None]
        raw = ns*rho
        sl = cp.stack((raw[:,0], (raw[:,1:4]**2).sum(axis=1), raw[:,4]), axis=1)
        with policy(cp.cuda.Device().id):
            nl = cp.stack([ni.engine.nldf.get_features(cp.ascontiguousarray(rho[s]),s) for s in range(ns)])
        self.raw, self.sl, self.nldf = (cp.asnumpy(x) for x in (raw,sl,nl))
        self.settings = ni.cpu_shell._numint.settings
        ni.last_rho = None

    def __call__(self, analyzer, settings):
        if (not np.array_equal(analyzer.dm, self.dm) or
                not np.array_equal(analyzer.grids.coords, self.coords) or
                not np.array_equal(analyzer.grids.weights, self.weights)):
            raise ValueError("E0 generator is bound to its exact SCF density and grid")
        if settings == "l":
            return self.raw.copy()
        if type(settings) is type(self.settings.sl_settings) and settings.__dict__ == self.settings.sl_settings.__dict__:
            return self.sl.copy()
        if type(settings) is type(self.settings.nldf_settings) and settings.__dict__ == self.settings.nldf_settings.__dict__:
            return self.nldf.copy()
        raise NotImplementedError("E0 supports only the frozen CIDER26SS SL/NLDF settings")
