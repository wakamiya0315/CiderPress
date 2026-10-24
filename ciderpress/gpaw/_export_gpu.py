"""Private FP64 PAW feature quadrature for the opt-in New exporter.

No live functional or SCF cache is accepted here. The caller owns a private
forward engine; equal-setup atoms share the existing device table caches.
"""
import numpy as np

from ciderpress.gpu import cupy, policy
from ciderpress.gpu.pasdw import (
    _DevicePlan, _groups, _join_device, coefficients,
)
from ciderpress.gpu.pasdw_reductions import density, spherical


def _repair_small_features(engine, a, features, raw, ae):
    """Use Classic's reductions on radial shells with cancellation-prone NLDF.

    Relative error near a small AE/PS feature can amplify FP64 summation
    differences. Recompute only shells containing values below one tenth of
    their channel maximum. The substantial forward/convolution and the other
    spherical/spline rows stay on device. No absolute error tolerance is used
    by the saved-GPW acceptance test.
    """
    from ciderpress.dft.grids_indexer import AtomicGridsIndexer
    from ciderpress.gpaw.fit_paw_gauss_pot import get_dv

    cp = cupy()
    setup = engine.setups[a]
    c, p = setup.cider_contribs, setup.pa_setup
    nn = c.grids_indexer.ylm.shape[0]
    cutoff = .1*np.max(abs(features), axis=-1, keepdims=True)
    small = np.any(abs(features) < cutoff, axis=(0, 1))
    rows = np.flatnonzero(small.reshape(-1, nn).any(1))
    if not len(rows):
        return features
    points = (rows[:, None]*nn+np.arange(nn)).ravel()
    fr = cp.asnumpy(engine.paw_kernel.fr_asgLq[a])
    df = cp.asnumpy(engine.paw_kernel.df_asgLq[a])
    coef = cp.asnumpy(engine.c_asiq[a])
    ns, na = len(fr), coef.shape[-1]
    ft = np.zeros((ns, len(rows), *fr.shape[2:]), dtype=np.float64)
    dv = get_dv(setup.xc_correction.rgd)
    with policy(None):
        for i in range(p.ni):
            L, n, j = p.lmlist_i[i], p.nlist_i[i], p.jlist_i[i]
            ref = (p.pfuncs_ng[n]*dv).dot(fr[:, :, L, :na])
            for s in range(ns):
                ft[s, :, L, :na] += (coef[s, i]-ref[s])*p.ffuncs_jg[j][rows, None]
        ft += fr[:, rows]
        if ae:
            ft += df[:, rows]
        indexer = AtomicGridsIndexer.make_single_atom_indexer(
            c.grids_indexer.ylm, c.grids_indexer.rad_arr[rows])
        indexer.set_weights(c.w_g[points])
        grid = indexer.empty_gq(nalpha=c.plan.nalpha)
        # Classic's feature density uses the spin-linear kinetic contraction;
        # its separately exported raw density has its own fixed BLAS ordering.
        tau = c.get_kinetic_energy(ae, D_sp=engine.D_asp[a])
        for s in range(ns):
            indexer.reduce_angc_ylm_(ft[s], grid, a2y=False, offset=0)
            rho = raw[s][:, points].copy()
            rho[4] = tau[s, points]
            features[s][:, points] = c.plan.eval_rho_full(grid, rho)[0]
    return features


def paw_features(engine):
    """Return Classic-ordered AE/PS NLDF and raw n/grad/tau dictionaries."""
    from ciderpress.gpaw.fit_paw_gauss_pot import get_dv

    cp = cupy()
    paw = engine.paw_kernel
    from ciderpress.gpaw.atom_descriptor_utils import calculate_paw_sl_features

    ae_nl, ps_nl = {}, {}
    # A tiny gradient component can be the difference of large terms. Even
    # FP64 GEMM then loses relative agreement with Classic's reduction order.
    # Preserve its inexpensive raw-density reconstruction (and SL features)
    # exactly; the large NLDF spherical/projection/spline work stays on GPU.
    with policy(None):
        ae_rho, ps_rho = calculate_paw_sl_features(engine.setups, engine.D_asp, True)
    for atoms in _groups(engine.setups, engine.D_asp):
        setup = engine.setups[atoms[0]]
        contrib, p = setup.cider_contribs, setup.pa_setup
        plan = _DevicePlan(contrib.plan)
        if not plan._supported():
            raise NotImplementedError('GPU export requires a version-j spline PAW plan')
        ns = engine.D_asp[atoms[0]].shape[0]
        na = engine.c_asiq[atoms[0]].shape[-1]
        fr = _join_device([paw.fr_asgLq[a] for a in atoms])
        df = _join_device([paw.df_asgLq[a] for a in atoms])
        shape = (len(atoms)*ns,) + fr.shape[2:]

        def project(x, to_radial, use_ffuncs=False):
            return coefficients(p, x, None, to_radial, check_q=False,
                                use_ffuncs=use_ffuncs, _nlm=fr.shape[-2])

        weighted = fr.reshape(shape)[..., :na]*cp.asarray(get_dv(setup.xc_correction.rgd))[:, None, None]
        ref = project(weighted, False)
        c = _join_device([engine.c_asiq[a] for a in atoms]).reshape(len(atoms)*ns, p.ni, na)
        ft = cp.zeros_like(fr).reshape(shape)
        ft[..., :na] = project(c-ref, True, True)
        ft = ft.reshape(fr.shape)
        ft += fr
        # Device density tables feed interpolation only. The host raw output
        # above retains Classic's uncut tau and exact cancellation residuals.
        rhos = density(contrib, np.stack([engine.D_asp[a] for a in atoms]))
        for rho, f, nl_out, raw_out, ae in zip(
                rhos, (ft+df, ft), (ae_nl, ps_nl), (ae_rho, ps_rho), (True, False)):
            grid = spherical(contrib.grids_indexer, f, False).reshape(
                len(atoms), ns, -1, fr.shape[-1])
            for i, a in enumerate(atoms):
                # eval_rho_full scales features by nspin, matching Classic.
                nl_out[a] = np.stack([plan.eval_rho_full(grid[i, s], rho[i, s].copy())[0]
                                      for s in range(ns)])
                nl_out[a] = _repair_small_features(engine, a, nl_out[a], raw_out[a], ae)
    return ae_nl, ps_nl, ae_rho, ps_rho
