"""Compare AE/PS atomic tau before the nonlinear feature construction."""
from types import SimpleNamespace
import numpy as np
import pytest
from ciderpress import gpu
from ciderpress.gpaw.atom_utils import _PAWCiderContribs


@pytest.mark.parametrize('nspins', [1, 2])
@pytest.mark.parametrize('minority', [1., 1e-14])
def test_atomic_tau_before_features(nspins, minority, monkeypatch):
    if gpu._real_cupy() is None:
        pytest.skip('requires real CUDA')
    from gpaw.sphere.lebedev import weight_n
    from ciderpress.gpu.pasdw_reductions import density
    rng = np.random.default_rng(198)
    nr, nq, nlm, npack = 12, 3, 4, 15
    nn = len(weight_n)
    x = SimpleNamespace(B_pqL=rng.normal(size=(npack, nq, nlm)), Lmax=nlm,
                        rgd=SimpleNamespace(r_g=np.linspace(0., 2., nr)), nc_corehole_g=None)
    for name in ('n_qg', 'nt_qg', 'd_qg', 'dt_qg'):
        setattr(x, name, rng.normal(size=(nq, nr)))
    for name in ('nc_g', 'nct_g', 'dc_g', 'dct_g', 'tauc_g', 'tauct_g'):
        setattr(x, name, rng.random(nr))
    for name in ('tau_pg', 'taut_pg'):
        setattr(x, name, 1e4*rng.normal(size=(npack, nr*nn)))
    contrib = SimpleNamespace(xcc=x, is_mgga=True, w_g=np.ones(nr*nn))
    D = rng.normal(size=(3, nspins, npack))
    D[:, -1] *= minority
    with gpu.policy(0):
        got = density(contrib, D)
    oracle = object.__new__(_PAWCiderContribs)
    oracle.xcc = x
    for ae, batch in zip((True, False), got):
        for atom in range(len(D)):
            expected = oracle.get_kinetic_energy(ae, D_sp=D[atom])
            np.testing.assert_allclose(batch[atom, :, 4], expected, atol=1e-10, rtol=1e-12)
    # Isolate the kinetic part of the production reverse contraction. The
    # separate gradient stencil/PAW tests cover the other density components.
    from ciderpress.gpu import pasdw_reductions as reductions
    monkeypatch.setattr(reductions, 'contract_gradient',
                        lambda *a: np.zeros((len(D)*nspins, nq, nlm)))
    potentials = []
    for _ in range(2):
        v = np.zeros_like(got[0])
        v[:, :, 4] = rng.normal(size=v[:, :, 4].shape)
        potentials.append(v)
    with gpu.policy(0):
        dh = reductions.finish(contrib, [(None, v) for v in potentials])
    expected_dh = sum(sign*np.matmul(v[:, :, 4]*contrib.w_g, table.T)
                      for sign, v, table in zip((1, -1), potentials, (x.tau_pg, x.taut_pg)))
    np.testing.assert_allclose(dh, expected_dh, atol=1e-8, rtol=1e-12)
    direction = rng.normal(size=D.shape)
    eps = .002
    with gpu.policy(0):
        plus, minus = density(contrib, D+eps*direction), density(contrib, D-eps*direction)
    fd = sum(sign*np.sum(v[:, :, 4]*(p[:, :, 4]-m[:, :, 4])*contrib.w_g)/(2*eps)
             for sign, v, p, m in zip((1, -1), potentials, plus, minus))
    np.testing.assert_allclose(fd, np.sum(dh*direction), atol=1e-6, rtol=1e-10)
