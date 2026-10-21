"""Resident nst semilocal/constant-normalizer/mapped-XC composition.

The outer CIDER kernel keeps its NumPy in-place API. Unsupported settings
use its unchanged CPU composition, including all nonconstant normalizers.
"""
import numpy as np

from ciderpress.gpu import cupy
from ciderpress.gpu.semilocal import evaluate_device, supported as mapped_supported


def supported(kernel):
    from ciderpress.dft.feat_normalizer import ConstantNormalizer
    settings = kernel.mlfunc.settings
    return (settings.sl_settings.mode == 'nst'
            and settings.normalizers.slmode == 'nst'
            and not settings.nldf_settings.is_empty
            and all(n is None or type(n) is ConstantNormalizer
                    for n in settings.normalizers._normalizers)
            and mapped_supported(kernel.mlfunc, settings.nfeat))


def evaluate(kernel, e, n, sigma, feat, v, vsigma, vfeat,
             tau_sg=None, dedtau_sg=None):
    cp = cupy()
    tau, vtau = tau_sg, dedtau_sg
    ns, ng = len(n), e.size
    assert tau is not None and vtau is not None
    rho = tuple(cp.asarray(a.reshape(a.shape[0], -1)) for a in (n, sigma, tau))
    x = cp.empty((ns, kernel.mlfunc.settings.nfeat, ng), dtype=cp.float64)
    x[:, 0] = rho[0] * ns
    x[:, 1] = rho[1][::2] * (ns*ns)
    x[:, 2] = rho[2] * ns
    x[:, 3:] = cp.asarray(feat.reshape(ns, -1, ng))
    normalizers = kernel.mlfunc.settings.normalizers._normalizers
    for i, normalizer in enumerate(normalizers):
        if normalizer is not None:
            x[:, i] *= normalizer.const
    energy, derivative, vrho = evaluate_device(kernel.mlfunc, x, rho, kernel.rhocut)
    energy *= kernel.xmix
    derivative *= kernel.xmix
    # Retain the two separate in-place additions used by call_xc_kernel.
    # This keeps the baseline and semilocal contributions' addition order.
    for out, baseline in zip((v, vsigma, vtau), vrho):
        out += cp.asnumpy(kernel.xmix * baseline).reshape(out.shape)
    for i, normalizer in enumerate(normalizers):
        if normalizer is not None:
            derivative[:, i] *= normalizer.const
    e += cp.asnumpy(energy).reshape(e.shape)
    v += cp.asnumpy(ns * derivative[:, 0]).reshape(v.shape)
    vsigma[::2] += cp.asnumpy((ns*ns) * derivative[:, 1]).reshape(vsigma[::2].shape)
    vtau += cp.asnumpy(ns * derivative[:, 2]).reshape(vtau.shape)
    vfeat += cp.asnumpy(derivative[:, 3:]).reshape(vfeat.shape)
