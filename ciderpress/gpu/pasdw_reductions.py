"""Setup-batched PAW density and potential quadrature (FP64).

Only the GPU PASDW controller calls these private routines. Host tables are
content-keyed by the same cache as the dense transforms. Atom and spin axes
remain separate until independent linear operations are folded together.
"""
import numpy as np

from ciderpress.gpu import cupy
from ciderpress.gpu.pasdw import _cached, _signature, _table_revision, contract_gradient


def spherical(indexer, x, to_radial):
    """Batched uniform-sphere transform, returning a private device array."""
    cp = cupy()
    y = _cached(indexer, 'spherical_batch', _signature(indexer.ylm),
                lambda cp: cp.asarray(indexer.ylm))
    nr, nn, nlm = indexer.nrad, indexer.ylm.shape[0], indexer.nlm
    x = cp.asarray(x)
    nq = x.shape[-1]
    if to_radial:
        x = x.reshape(-1, nr, nn, nq)
        return cp.matmul(y.T, x)
    x = x.reshape(-1, nr, nlm, nq)
    return cp.matmul(y, x).reshape(-1, nq)


def energy(batch, e, rho, v, f):
    """Keep large g,q interpolation/spherical intermediates on the device.

    The GP retains its original NumPy inputs. Spline/exponent derivatives
    remain on the device until the final density-derivative result.
    """
    from ciderpress.gpaw.atom_utils import get_sigma
    from ciderpress.gpu.pasdw import _DevicePlan

    cp = cupy()
    plan = _DevicePlan(batch.base.plan, device_output=True)
    ns, nx, ng = rho.shape
    feat = np.empty((ns, plan.nldf_settings.nfeat, ng))
    dfeat = np.empty((ns, plan.num_vj, ng))
    for s in range(ns):
        grid = spherical(batch.base.grids_indexer, cp.asarray(f[s]), False)
        plan.eval_rho_full(grid, rho[s], spin=s, feat=feat[s], dfeat=dfeat[s])
    sigma = get_sigma(rho[:, 1:4])
    vsigma = np.zeros_like(sigma)
    vn = np.zeros_like(rho[:, 0])
    args = [e, rho[:, 0].copy(), vn, sigma, vsigma]
    if nx == 5:
        vtau = np.zeros_like(vn)
        args += [rho[:, 4].copy(), vtau]
    args.append(feat)
    e[:] = 0
    vf = batch.cider_kernel.calculate(*args)
    v[:, 0] = cp.asarray(vn) if isinstance(v, cp.ndarray) else vn
    if nx == 5:
        v[:, 4] = cp.asarray(vtau) if isinstance(v, cp.ndarray) else vtau
    if isinstance(v, cp.ndarray):
        v[:, 1:4] = 2*cp.asarray(vsigma[::2, None])*cp.asarray(rho[:, 1:4])
        if ns == 2:
            v[:, 1:4] += cp.asarray(vsigma[1, None])*cp.asarray(rho[::-1, 1:4])
    else:
        v[:, 1:4] = 2*vsigma[::2, None]*rho[:, 1:4]
        if ns == 2:
            v[:, 1:4] += vsigma[1, None]*rho[::-1, 1:4]
    weights = _cached(batch.base, ('batch_weights', batch.grids_indexer.count),
                      _signature(batch.base.w_g),
                      lambda cp: cp.tile(cp.asarray(batch.base.w_g), batch.grids_indexer.count))
    out = []
    for s in range(ns):
        grid = plan.eval_vxc_full(vf[s], v[s], dfeat[s], rho[s], spin=s)
        grid *= weights[:, None]
        out.append(spherical(batch.base.grids_indexer, grid, True))
    return cp.stack(out)


def density(contrib, D):
    """Return AE/PS a,s,x,g density batches, with setup tables on device."""
    from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
    from gpaw.xc.pawcorrection import rnablaY_nLv

    cp = cupy()
    x = contrib.xcc
    ns = D.shape[1]
    nlm = x.B_pqL.shape[-1]
    names = ('B_pqL', 'n_qg', 'nt_qg', 'd_qg', 'dt_qg', 'nc_g', 'nct_g',
             'dc_g', 'dct_g', 'tau_pg', 'taut_pg', 'tauc_g', 'tauct_g')
    arrays = [getattr(x, name) for name in names]
    signature = _table_revision(contrib, 'density_batch', *arrays,
                                x.rgd.r_g, Y_nL, rnablaY_nLv, R_nv, contrib.w_g)
    def build(cp):
        return dict(zip(names + ('r', 'y', 'dy', 'dirs', 'weights'),
                        [cp.asarray(a) for a in arrays] +
                        [cp.asarray(x.rgd.r_g), cp.asarray(Y_nL[:, :nlm]),
                         cp.asarray(rnablaY_nLv[:, :nlm]), cp.asarray(R_nv), cp.asarray(contrib.w_g)]))
    t = _cached(contrib, 'density_batch', signature, build)
    # Setup data is fixed within a forward/energy/reverse cycle, as required
    # by the saved density and fr/df intermediates. Reuse its already checked
    # kinetic tables instead of hashing and uploading them twice more.
    contrib._pasdw_cycle_tables = t
    d = cp.asarray(D)
    dlq = cp.matmul(d, t['B_pqL'].reshape(d.shape[-1], -1)).reshape(
        *d.shape[:2], t['B_pqL'].shape[1], nlm).swapaxes(-1, -2)
    out = []
    for nkey, dnkey, core, dcore, tau, tauc in (
            ('n_qg', 'd_qg', 'nc_g', 'dc_g', 'tau_pg', 'tauc_g'),
            ('nt_qg', 'dt_qg', 'nct_g', 'dct_g', 'taut_pg', 'tauct_g')):
        n = cp.matmul(dlq, t[nkey])
        dn = cp.matmul(dlq, t[dnkey])
        n[:, :, 0] += np.sqrt(4*np.pi)/ns * t[core]
        dn[:, :, 0] += np.sqrt(4*np.pi)/ns * t[dcore]
        # A polarized AE core hole belongs to opposite spin channels with
        # opposite signs; the pseudo core remains unpolarized.
        if nkey == 'n_qg' and ns == 2 and x.nc_corehole_g is not None:
            sign = cp.asarray([-1., 1.])[:, None]
            n[:, :, 0] += sign * (0.5*np.sqrt(4*np.pi)*cp.asarray(x.nc_corehole_g))
            dn[:, :, 0] += sign * (0.5*np.sqrt(4*np.pi)*cp.asarray(x.dc_corehole_g))
        rho = cp.empty((*D.shape[:2], 5 if contrib.is_mgga else 4,
                        n.shape[-1], len(weight_n)))
        rho[:, :, 0] = cp.matmul(n.swapaxes(-1, -2), t['y'].T)
        grad = cp.einsum('aslg,nlv->asvgn', n, t['dy'])
        grad[..., 1:, :] /= t['r'][1:, None]
        grad[..., 0, :] = grad[..., 1, :]
        grad += cp.matmul(dn.swapaxes(-1, -2), t['y'].T)[:, :, None] * t['dirs'].T[:, None, :]
        rho[:, :, 1:4] = grad
        if contrib.is_mgga:
            kinetic = cp.matmul(d, t[tau]).reshape(*D.shape[:2], -1, len(weight_n))
            kinetic += (t[tauc]/(np.sqrt(4*np.pi)*ns))[:, None]
            rho[:, :, 4] = kinetic
        out.append(cp.asnumpy(rho.reshape(*rho.shape[:3], -1)))
    return tuple(out)


def finish(contrib, values, energy=False, _tables=None):
    """Contract all atoms' density/gradient/tau potentials in two batches.

    values contains AE/PS (energy a,g or None, potential a,s,x,g).
    Energy quadrature retains GPAW's original NumPy integration ordering.
    """
    from gpaw.sphere.lebedev import weight_n

    cp = cupy()
    x = contrib.xcc
    count, ns = values[0][1].shape[:2]
    terms, kinetic, energies = [], [], []
    for (e, v), n, dn, tau in zip(values, (x.n_qg, x.nt_qg),
                                 (x.d_qg, x.dt_qg), (x.tau_pg, x.taut_pg)):
        flat = v.reshape(count*ns, *v.shape[2:])
        terms.append(contract_gradient(contrib, x.rgd, n, dn, flat[:, 0], flat[:, 1:4], x.Lmax))
        if v.shape[2] == 5:
            if _tables is None:
                table = _cached(contrib, ('kinetic_batch', tau is x.tau_pg), _signature(tau, contrib.w_g),
                                lambda cp: (cp.asarray(tau), cp.asarray(contrib.w_g)))
            else:
                table = _tables['tau_pg' if tau is x.tau_pg else 'taut_pg'], _tables['weights']
            kinetic.append(cp.asnumpy(cp.matmul(cp.asarray(flat[:, 4])*table[1], table[0].T)))
        if energy:
            energies.append(np.array([x.rgd.integrate(row.reshape(-1, len(weight_n)).dot(weight_n)) for row in e]))
    dh = np.inner((terms[0]-terms[1]).reshape(count*ns, -1),
                  x.B_pqL.reshape(len(x.B_pqL), -1))
    if kinetic:
        dh += kinetic[0] - kinetic[1]
    dh = dh.reshape(count, ns, -1)
    return (energies[0]-energies[1]-x.e_xc0, dh) if energy else dh
