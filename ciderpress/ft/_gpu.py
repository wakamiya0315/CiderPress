"""FP64 blocked FT quadratures using the existing GPAW resident XC pipeline.

Only grid quadratures are offloaded. Normalization, candidate selection and
the small MOLGP2 solve keep their original host implementations. No dispatch
policy/environment variable can silently turn an explicit FT GPU request off.
"""

import numpy as np

from ciderpress.gpu import _real_cupy


def require_gpu():
    cp = _real_cupy()
    if cp is None:
        raise RuntimeError("FineTuner device='gpu' requires CuPy with a CUDA device")
    return cp


def integrate(model, record, blocksize, rhocut, core):
    from ciderpress.gpu.semilocal import evaluate_device, supported

    from .records import blocks

    cp = require_gpu()
    if model.libxc_baseline is not None or not supported(model, model.settings.nfeat):
        raise NotImplementedError(
            "FT GPU quadrature requires a supported self-contained MappedXC2"
        )
    energy = record["exc_core"] if core else 0.0
    for x, rho, wt in blocks(record, model.settings, blocksize):
        # Reuse the same descriptor/RBF/libxc composition as New GPAW. Only
        # the scalar quadrature returns to host; gradients stay on device.
        value = evaluate_device(
            model,
            cp.asarray(np.ascontiguousarray(x, dtype=np.float64)),
            tuple(cp.asarray(r, dtype=cp.float64) for r in rho),
            rhocut,
        )[0]
        energy += float((value @ cp.asarray(wt, dtype=cp.float64)).item())
    if not np.isfinite(energy):
        raise ValueError("Nonfinite GPU integrated energy")
    return energy


def covariance(kernel, record, settings, blocksize, rhocut):
    from ciderpress.gpu.descriptors import forward, signature
    from ciderpress.gpu.semilocal import _CODES, _baseline
    from ciderpress.models.kernels import DiffConstantKernel, DiffProduct, DiffRBF

    from .records import blocks

    cp = require_gpu()
    rbf, amplitude = kernel.kernel, 1.0
    if type(rbf) is DiffProduct and type(rbf.k1) is DiffConstantKernel:
        amplitude, rbf = rbf.k1.constant_value, rbf.k2
    if (
        type(rbf) is not DiffRBF
        or kernel.mode not in ("SEP", "NPOL")
        or kernel._mul_basefunc not in _CODES
        or abs(kernel.omega) > 1e-11
        or signature(kernel.feature_list, settings.nfeat) is None
    ):
        raise NotImplementedError(
            "FT GPU covariance supports ordinary SEP/NPOL RBF kernels only"
        )
    length = cp.asarray(np.asarray(rbf.length_scale, dtype=np.float64))
    control = cp.asarray(kernel.X1ctrl, dtype=cp.float64) / length
    result = np.zeros(kernel.Nctrl)
    sep = kernel.mode == "SEP"
    for x, rho, wt in blocks(record, settings, blocksize):
        xd = cp.asarray(np.ascontiguousarray(x, dtype=np.float64))
        rd = tuple(cp.asarray(r, dtype=cp.float64) for r in rho)
        raw = [xd[s] for s in range(record["nspin"])] if sep else [xd.mean(0)]
        desc = cp.concatenate([forward(kernel.feature_list, a) for a in raw]) / length
        # Match sklearn's scaled squared Euclidean distance, avoiding the
        # cancellation in ||x||^2 + ||c||^2 - 2 x.c. Workspace is block x M.
        distance = cp.zeros((desc.shape[0], kernel.Nctrl), dtype=cp.float64)
        for j in range(kernel.N1):
            diff = desc[:, j, None] - control[None, :, j]
            distance += diff * diff
        values = amplitude * cp.exp(-0.5 * distance)
        m = _baseline(kernel._mul_basefunc, rd, sep)[0]
        if sep:
            m[record["nspin"] * rd[0] < rhocut] = 0
            values = (
                values.reshape(record["nspin"], len(wt), kernel.Nctrl) * m[..., None]
            ).sum(axis=0)
        else:
            m[rd[0].sum(axis=0) < rhocut] = 0
            values *= m[:, None]
        # Preserve host block accumulation order; transfer only M scalars.
        result += cp.asnumpy(values.T @ cp.asarray(wt, dtype=cp.float64))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite integrated kernel covariance")
    return result
