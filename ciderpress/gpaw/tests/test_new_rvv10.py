"""Opt-in rVV10 composition, discrete derivatives and restart parameters.

Small fixed fields avoid SCF and the base semilocal n/tau clamp residuals.
GPU cases follow the existing real-CUDA skip convention.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from gpaw.core import UGDesc

from ciderpress import gpu
from ciderpress.gpaw.cider_fft import get_rho_sxg
from ciderpress.gpaw.new_backend import (
    CiderXC, _RVV10Contribution, add_gradient_correction, restore_cider_xc,
)

OPTIONS = dict(kernel_density='smooth', b=11.95, C=.0093, nq=12, nR=8192, q_block=4,
               desaturate=True, q_top=2.5, q_cut=.5, q_min=1e-4)
CELL = np.array([[7., .4, .2], [.2, 8., .1], [.3, .2, 9.]])
PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def device_for(use_gpu):
    if not use_gpu:
        return None
    cp = gpu._real_cupy()
    if cp is None:
        pytest.skip('requires real CUDA CuPy')
    return cp.cuda.Device().id


def fields(grid, nspin):
    xyz = np.indices(grid.size_c) / grid.size_c[:, None, None, None]
    n = .03 + .006 * np.cos(2*np.pi*xyz[0]) + .004*np.sin(2*np.pi*xyz[1])
    n += .002 * np.cos(2*np.pi*(xyz[0]+xyz[2]))
    return np.array([n] if nspin == 1 else [.3*n, .7*n])


@pytest.mark.parametrize('gradient', [None, 'voronoi', 'lattice'])
def test_restart_parameters(gradient):
    xc = CiderXC(rvv10=OPTIONS, gradient=gradient)
    d = xc.todict()
    assert ('gradient' in d) == (gradient is not None)
    assert restore_cider_xc(json.loads(json.dumps(d)), expected=xc).todict() == d
    for key, value in [('b', 6.3), ('C', .01), ('q_cut', .6),
                       ('q_block', 3), ('nq', 16), ('q_min', 2e-4)]:
        with pytest.raises(ValueError, match='settings mismatch: rvv10'):
            restore_cider_xc(d, expected=CiderXC(rvv10=dict(OPTIONS, **{key: value}),
                                                gradient=gradient))
    with pytest.raises(ValueError, match='settings mismatch: rvv10'):
        restore_cider_xc(d, expected=CiderXC(gradient=gradient))
    with pytest.raises(ValueError, match='settings mismatch: gradient'):
        restore_cider_xc(d, expected=CiderXC(rvv10=OPTIONS,
            gradient='lattice' if gradient != 'lattice' else None))
    d['rvv10']['b'] = 123
    assert xc.rvv10['b'] == OPTIONS['b']


def test_default_serialization_and_q_mesh():
    xc = CiderXC()
    assert list(xc.todict()) == ['name', 'cider_new_version', 'model',
                                'model_sha256', 'options', 'nldf', 'cell']
    assert restore_cider_xc(xc.todict(), expected=xc).todict() == xc.todict()
    mesh = np.geomspace(1e-4, .5, 12)
    xc = CiderXC(rvv10=dict(OPTIONS, q_mesh=mesh))
    restored = restore_cider_xc(json.loads(json.dumps(xc.todict())), expected=xc)
    assert restored.rvv10['q_mesh'] == mesh.tolist()
    with pytest.raises(ValueError, match='gradient'):
        CiderXC(gradient='other')


def test_beta_restart_and_noncollinear():
    xc = CiderXC(rvv10=OPTIONS)
    assert xc.todict()['rvv10']['beta'] == 'augmented'
    old = xc.todict()
    old['rvv10'].pop('beta')
    restored = restore_cider_xc(old)
    assert restored.rvv10['beta'] == 'smooth'
    assert 'beta' not in old['rvv10']  # Do not mutate caller's saved data.
    with pytest.raises(ValueError, match='settings mismatch: rvv10'):
        restore_cider_xc(old, expected=xc)
    smooth = CiderXC(rvv10=dict(OPTIONS, beta='smooth'))
    assert restore_cider_xc(old, expected=smooth).todict() == smooth.todict()
    with pytest.raises(ValueError, match='beta'):
        CiderXC(rvv10=dict(OPTIONS, beta='invalid'))
    with pytest.raises(NotImplementedError, match='noncollinear'):
        xc.functional(collinear=False)


@pytest.mark.parametrize('nspin', [1, 2])
@pytest.mark.parametrize('beta', ['augmented', 'smooth'])
def test_beta_paw_composition(nspin, beta):
    from gpaw.utilities import pack_density, pack_hermitian
    from ciderpress.gpaw.new_backend import CiderNewMGGAFunctional
    grid = UGDesc(cell=CELL, size=(8, 10, 12), pbc=True)
    nl = _RVV10Contribution(dict(OPTIONS, beta=beta), grid, None)
    setup = SimpleNamespace(dO_ii=np.array([[.1, .2], [.2, -.3]]))
    D = np.array([pack_density(np.array([[1., .3], [.3, .4]]))] * nspin)
    engine = SimpleNamespace(dens=SimpleNamespace(D_asp={0: D}),
                             dH_asp_tmp={0: np.full_like(D, 7.)}, E_a_tmp={0: 3.})
    f = object.__new__(CiderNewMGGAFunctional)
    f.engine, f.setups, f._paw_ready, f._rvv10 = engine, [setup], True, nl
    h = np.zeros_like(D)
    energy = f._calculate_paw_correction(setup, D, h, 0)
    correction = nl.nl.beta * pack_hermitian(setup.dO_ii)
    if beta == 'augmented':
        np.testing.assert_array_equal(h, np.full_like(D, 7.) + correction)
        assert energy == 3. + np.einsum('sp,p->', D, correction)
        delta = np.zeros_like(D)
        delta[0, 1] = 1e-4
        fd = (nl.calculate_paw_correction(setup, D + delta) -
              nl.calculate_paw_correction(setup, D - delta)) / 2e-4
        assert fd == pytest.approx(correction[1], abs=1e-13)
    else:
        assert energy == 3.
        np.testing.assert_array_equal(h, np.full_like(D, 7.))


def test_layout_rejections():
    for pbc in ([True, True, False], [False]*3):
        grid = UGDesc(cell=CELL, size=(8, 10, 12), pbc=pbc)
        with pytest.raises(NotImplementedError, match='fully periodic'):
            CiderXC(rvv10=OPTIONS).new_functional(grid)
    grid = SimpleNamespace(comm=SimpleNamespace(size=2))
    with pytest.raises(NotImplementedError, match='undistributed'):
        _RVV10Contribution(OPTIONS, grid, None)
    f = CiderXC(rvv10=OPTIONS).new_functional(
        UGDesc(cell=CELL, size=(8, 10, 12), pbc=True))
    for kw, message in [(dict(mode='fd'), 'PW mode'),
                         (dict(distribute_density=True), 'distributed density')]:
        with pytest.raises(NotImplementedError, match=message):
            f.set_context(setups=None, relpos_ac=None, atomdist=None, **kw)
    with pytest.raises(ValueError, match='q_mesh'):
        _RVV10Contribution(dict(OPTIONS, q_mesh=[0., *np.geomspace(1e-4, .5, 11)]),
                           UGDesc(cell=CELL, size=(8, 10, 12)), None)


def test_lattice_stress_preflight(monkeypatch):
    from ciderpress.gpaw.tests.test_new_forces import case
    from gpaw.fd_operators import Gradient
    f, density, _, _ = case(monkeypatch)
    f.grid = UGDesc(cell=np.diag([7., 8., 9.]), size=(8, 10, 12), pbc=True)
    density.nt_sR.desc = f.grid
    f.grad_v = [Gradient(f.grid._gd, v, n=2, lattice=True) for v in range(3)]
    def fail(*args, **kwargs):
        raise AssertionError('Voronoi branch check reached for lattice gradients')
    monkeypatch.setattr('gpaw.fd_operators.gradient_strain_coefficients', fail)
    with f.diagnostic_stress_without_tau():
        f.validate_stress_calculation()
    assert f._stress_stencil_checked


@pytest.mark.parametrize('use_gpu', [False, True])
@pytest.mark.parametrize('beta', ['augmented', 'smooth', 'legacy', 'kernel'])
def test_fixed_input_gpw_roundtrip(tmp_path, use_gpu, beta, record_property,
                                   monkeypatch):
    """Actual IO on fixed n/tau/D; SCF continuation is a separate harness check."""
    from ase import Atoms
    from gpaw import GPAW
    from gpaw.gpu import as_np
    from gpaw.mpi import serial_comm
    device_for(use_gpu)
    atoms=Atoms('H2',positions=[[1.13,1.27,1.31],[1.87,1.34,1.39]],
                cell=[4.,4.4,4.8],pbc=True)
    options=dict(OPTIONS, beta='smooth' if beta == 'legacy' else
                 'augmented' if beta == 'kernel' else beta)
    if beta == 'kernel':
        # Fresh input takes the new augmented/augmented default, and the
        # actual GPW must restore both the key and its n/tau/D potential.
        options.pop('kernel_density')
    xc=CiderXC(rvv10=options, gradient='lattice')
    calc=GPAW(mode=dict(name='pw',ecut=80),gpts=(12,14,16),nbands=2,
              xc=xc,legacy_gpaw=False,symmetry='off',txt=None,
              communicator=serial_comm,parallel=dict(gpu=use_gpu))
    calc.create_new_calculation(atoms)
    dft=calc.dft
    dft.density.taut_sR.data[:]+=.05
    dft.potential,dft.energies,_=dft.pot_calc.calculate(dft.density,dft.ibzwfs)
    def evaluate(dft):
        pc,d= dft.pot_calc,dft.density
        pc.xc.set_density(d)
        n,tau=[pc.interpolate(x) for x in (d.nt_sR,d.taut_sR)]
        e,_,_=pc.xc.calculate(n,tau)
        return e,pc.xc.nl_energy
    before=evaluate(dft)
    path=tmp_path/'cider-rvv10.gpw'
    if beta == 'legacy':
        saved = xc.todict()
        saved['rvv10'].pop('beta')
        with monkeypatch.context() as m:
            m.setattr(xc, 'todict', lambda: saved)
            calc.write(path, mode='all')
    else:
        calc.write(path,mode='all')
    restored=GPAW(path,legacy_gpaw=False,txt=None,communicator=serial_comm,
        parallel=dict(gpu=use_gpu),
        object_hooks={'xc':lambda d:restore_cider_xc(d,expected=xc)})
    assert restored.params.xc.todict()==xc.todict()
    after=evaluate(restored.dft)
    record_property('grid_energy_error_Ha',abs(before[0]-after[0]))
    record_property('nonlocal_energy_error_Ha',abs(before[1]-after[1]))
    for a,b in zip(before,after):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match='settings mismatch: rvv10'):
        GPAW(path,legacy_gpaw=False,txt=None,communicator=serial_comm,
            object_hooks={'xc':lambda d:restore_cider_xc(d,
                expected=CiderXC(rvv10=dict(OPTIONS,b=6.3),gradient='lattice'))})


@pytest.mark.parametrize('use_gpu', [False, True])
@pytest.mark.parametrize('nspin', [1, 2])
def test_composition_and_full_potential(use_gpu, nspin):
    device = device_for(use_gpu)
    grid = UGDesc(cell=CELL, size=(8, 10, 12), pbc=True)
    from gpaw.gpu import cupy
    f = CiderXC(rvv10=OPTIONS).new_functional(grid, np if device is None else cupy)
    grad = f.engine.grad_v
    n = fields(grid, nspin)
    tau = np.full_like(n, .1)
    rng = np.random.default_rng(87)
    perturb = rng.normal(size=n.shape) * .003

    def evaluate(n, potential=False):
        rho = get_rho_sxg(n, grad, tau)
        e, ded = np.zeros(n.shape[1:]), np.zeros_like(rho)
        with gpu.policy(device):
            f._rvv10.add(e, rho, ded)
            v = ded[:, 0].copy()
            add_gradient_correction(grad, ded, v)
        assert np.count_nonzero(ded[:, 4]) == 0
        assert f._rvv10.energy == pytest.approx(e.sum()*grid.dv, abs=1e-15)
        return (e.sum()*grid.dv, v) if potential else e.sum()*grid.dv

    energy, v = evaluate(n, True)
    analytic = np.sum(v*perturb)*grid.dv
    fd = (evaluate(n+1e-4*perturb)-evaluate(n-1e-4*perturb))/2e-4
    assert fd == pytest.approx(analytic, rel=1e-8, abs=1e-12)
    # Independent NumPy engine call, including beta*int(n), no sigma packing.
    rho = get_rho_sxg(n, grad, tau)
    total_grad = rho[:, 1:4].sum(axis=0)
    direct = f._rvv10.nl.calculate(n.sum(axis=0), (total_grad**2).sum(axis=0), CELL)
    assert energy == pytest.approx(direct['energy'], abs=1e-13)


@pytest.mark.parametrize('use_gpu', [False, True])
@pytest.mark.parametrize('gradient', [None, 'lattice'])
def test_full_strain(use_gpu, gradient):
    from ciderpress.gpaw.cider_paw import cider_smooth_stress
    from gpaw.xc.gga import stress_gradient_stencil_term
    from gpaw.gpu import cupy
    device = device_for(use_gpu)
    cell = CELL if gradient is None else np.diag([7., 8., 9.])
    center = UGDesc(cell=cell, size=(8, 10, 12), pbc=True)
    n = fields(center, 1)
    f = CiderXC(rvv10=OPTIONS, gradient=gradient).new_functional(
        center, np if device is None else cupy)
    nl = f._rvv10

    def evaluate(H, stress=False):
        grid = UGDesc(cell=H, size=center.size_c, pbc=True)
        operators = CiderXC(gradient=gradient).new_functional(grid).engine.grad_v
        nn = n * np.linalg.det(cell)/np.linalg.det(H)
        rho = get_rho_sxg(nn, operators, np.zeros_like(nn))
        e, ded = np.zeros(nn.shape[1:]), np.zeros_like(rho)
        nl.grid = grid
        with gpu.policy(device):
            kernel = nl.add(e, rho, ded, stress=stress)
        if not stress:
            return e.sum()*grid.dv
        smooth = cider_smooth_stress(grid._gd, e, rho, ded,
                                     np.zeros((1, 6, *nn.shape[1:])))
        stencil = stress_gradient_stencil_term(grid._gd, operators, nn,
                                               rho[:, 1:4], ded[:, 1:4])
        return smooth + stencil + kernel

    tensor = evaluate(cell, True)
    for i, j in PAIRS:
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = 1. if i == j else .5
        h = 1e-5
        fd = (evaluate(cell @ (np.eye(3)+h*strain)) -
              evaluate(cell @ (np.eye(3)-h*strain)))/(2*h)
        assert tensor[i, j] == pytest.approx(fd, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize('use_gpu', [False, True])
def test_lattice_stencils(use_gpu):
    from ciderpress.gpu.fine_grid import get_rho
    device = device_for(use_gpu)
    for cell in (np.diag([7., 8., 9.]), CELL):
        grid = UGDesc(cell=cell, size=(8, 10, 12), pbc=True)
        a, b = [CiderXC(gradient=g).new_functional(grid) for g in (None, 'lattice')]
        assert all(g.lattice for g in b.grad_v)
        n = fields(grid, 1)
        tau = np.full_like(n, .1)
        ra = get_rho_sxg(n, a.engine.grad_v, tau)
        rb = get_rho_sxg(n, b.engine.grad_v, tau)
        if np.array_equal(cell, np.diag(np.diag(cell))):
            np.testing.assert_array_equal(ra, rb)
        else:
            assert np.max(np.abs(ra-rb)) > 1e-7
        if device is not None:
            with gpu.policy(device):
                np.testing.assert_allclose(get_rho(n, b.engine.grad_v, tau), rb,
                                           rtol=1e-12, atol=1e-12)
