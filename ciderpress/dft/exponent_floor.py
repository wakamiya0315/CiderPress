"""Opt-in, non-official smooth variant of the NLDF lower exponent floor."""

import os

import numpy as np
from scipy.special import expit


def validate_floor_width(width):
    width = float(width)
    if not np.isfinite(width) or width < 0:
        raise ValueError("smooth_floor_width must be finite and nonnegative")
    return width


def check_floor_cpu(width, *, device=None):
    """Fail closed before any selected GPU kernel can evaluate this variant."""
    if width:
        from ciderpress.gpu import is_enabled

        selected = os.environ.get("CIDER_GPU_KERNELS", "all").strip()
        if selected != "none" and (device is not None or is_enabled()):
            raise NotImplementedError("NLDF smooth variant is CPU-only")


def smooth_floor_q(exponent, alpha0, lambd, width):
    """Return q and dq/da for a_eff=alpha0*(1+w*softplus((a/alpha0-1)/w)).

    Evaluate log(a_eff/alpha0) with log1p, so the negative tail retains
    its derivative even when adding the softplus to one would round to one.
    Width is dimensionless; zero must dispatch to the untouched official map.
    """
    z = (exponent / alpha0 - 1.0) / width
    excess = width * np.logaddexp(0.0, z)
    log_lambda = np.log(lambd)
    q = np.log1p(excess) / log_lambda
    dq = expit(z) / (alpha0 * (1.0 + excess) * log_lambda)
    return q, dq
