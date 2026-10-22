"""Independent discrete strain derivatives and New bulk stress contracts."""
import numpy as np
import pytest

from ciderpress.gpaw.nldf_interface import LibCiderPW
from ciderpress.gpaw.tests.test_gpu_nldf_fft import active_work
from ciderpress.gpaw.tests.test_radial_cutoff import slice_factory

PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def directions():
    for i, j in PAIRS:
        A = np.zeros((3, 3))
        A[i, j] = 1. if i == j else .5
        A[j, i] = A[i, j]
        yield i, j, A


@pytest.mark.parametrize('problem', [None, 'linear', 'stale', 'missing',
                                    'nonfinite', 'negative', 'large',
                                    'identity', 'gpu_mpi'])
def test_joint_tau_stress_certificate(monkeypatch, problem):
    from types import SimpleNamespace as NS
    from gpaw.new.tau_state import TauState
    from ciderpress.gpaw.tests.test_new_forces import case

    f, density, _, _ = case(monkeypatch)
    state = TauState({'method': 'joint'})
    state.output = object()
    state.output_epoch = state.input_epoch
    state.residuals = dict(density=1e-9, tau=1e-9, paw=1e-9)
    density.tau_state = state
    ibzwfs = NS(comm=NS(size=1))
    if problem == 'linear':
        state.options['method'] = 'linear'
    elif problem == 'stale':
        state.output_epoch -= 1
    elif problem == 'missing':
        state.residuals = None
    elif problem in ('nonfinite', 'negative', 'large'):
        state.residuals['tau'] = {'nonfinite': np.nan, 'negative': -1.,
                                  'large': 1.01e-7}[problem]
    elif problem == 'identity':
        density.tau_state = object()
    elif problem == 'gpu_mpi':
        f._device = 0
        ibzwfs.comm.size = 2
    if problem is None:
        f.validate_tau_state_stress(state, ibzwfs)
    else:
        with pytest.raises((RuntimeError, NotImplementedError)):
            f.validate_tau_state_stress(state, ibzwfs)


@pytest.mark.parametrize('shape', [(8, 10, 12), (7, 9, 11)])
def test_kernel_strain_and_volume(shape, monkeypatch):
    """Native dK against strain FD, including dv once; skew even/odd rFFT."""
    from ciderpress.gpaw.new_backend import _process_comm
    monkeypatch.setenv('CIDER_GPU_KERNELS', 'none')
    cell = np.array([[7., .4, .2], [.2, 8., .1], [.3, .2, 9.]])
    rng = np.random.default_rng(21)
    theta, adjoint = [rng.normal(size=(*shape, 3)) * .01 for _ in range(2)]

    def evaluate(H, derivative=False):
        pw = LibCiderPW(shape, H, _process_comm())
        pw.nalpha = pw.nbeta = 3
        a = np.geomspace(.1, 12., 3)
        pw.expnts_ab = np.ascontiguousarray(1 / (4 * (a[:, None] + a)))
        pw.norms_ab = np.ascontiguousarray((np.pi / (a[:, None] + a))**1.5)
        pw.initialize_backend()
        pw.set_work(np.ones(shape), theta)
        saved = pw.copy_work_array()
        pw.compute_forward_convolution()
        energy = float(np.sum(active_work(pw) * adjoint) * np.linalg.det(H) /
                       np.prod(shape))
        if not derivative:
            return energy
        pw.fftv2(saved)
        pw.set_work(np.ones(shape), adjoint)
        S = pw.compute_backward_convolution_with_stress(saved)
        from ciderpress.gpaw.cider_paw import odd_fft_stress_correction
        S += odd_fft_stress_correction(pw, theta, adjoint)
        return energy, S + energy * np.eye(3)

    energy, stress = evaluate(cell, True)
    for i, j, A in directions():
        for h in (1e-4, 5e-5, 2.5e-5):
            fd = (evaluate(cell @ (np.eye(3) + h * A)) -
                  evaluate(cell @ (np.eye(3) - h * A))) / (2 * h)
            assert stress[i, j] == pytest.approx(fd, abs=2e-8, rel=2e-7)


def test_pasdw_strain_B_Q():
    """Both maps, the overlap inverse, dv and moving sphere membership."""
    from ciderpress.gpaw.atom_utils import FastAtomPASDWSlice
    xyz, make = slice_factory()
    reference = make(np.zeros(3))
    setup = reference.psetup
    cutoff, hrad = reference.rcut, reference.h

    def strained(A, step):
        transform = np.eye(3) + step * A
        dr = xyz @ transform
        radius = np.linalg.norm(dr, axis=1)
        mask = radius <= cutoff
        radius, dr = radius[mask], dr[mask]
        g = radius / hrad
        return FastAtomPASDWSlice(np.flatnonzero(mask), np.floor(g), g % 1,
            radius, dr / radius[:, None], cutoff, setup,
            reference.dv * np.linalg.det(transform), hrad, ovlp_fit=True)

    p, dp = reference.get_funcs(), reference.get_grads()
    S = reference.sinv_pf
    dS = reference.get_ovlp_deriv(p, dp, stress=True)
    pstress = reference.get_stress_funcs()
    rng = np.random.default_rng(41)
    c, y = rng.normal(size=(4, 2)), rng.normal(size=(len(xyz), 2))

    def maps(s):
        full = np.zeros((len(xyz), 4))
        full[s.indset] = s.get_funcs().T @ s.sinv_pf
        return full @ c, full.T @ y * s.dv

    for i, j, A in directions():
        derivative = np.einsum('uv,uvig->ig', A, pstress).T @ S
        derivative += p.T @ np.einsum('uv,uvpf->pf', A, dS)
        bgrad = np.zeros((len(xyz), 2))
        bgrad[reference.indset] = derivative @ c
        qgrad = (derivative.T @ y[reference.indset] * reference.dv +
                 np.trace(A) * maps(reference)[1])
        errors = []
        for step in (2e-4, 1e-4, 5e-5, 2.5e-5):
            plus, minus = maps(strained(A, step)), maps(strained(A, -step))
            errors.append(max(np.max(abs(actual - (a-b)/(2*step)))
                              for actual, a, b in zip((bgrad, qgrad), plus, minus)))
        assert errors[-1] < 2e-3, (i, j, errors)
        # The unchanged angular-gradient regularizer leaves a up to ~7.5e-7
        # floor in the shear directions; retain it explicitly.
        assert errors[-1] < max(.4 * errors[0], 1e-6), (i, j, errors)


def test_stress_private_inputs_and_stale_rejection(monkeypatch):
    from ciderpress.gpaw.tests.test_new_forces import case
    import ciderpress.gpaw.new_backend as backend
    f, d, interpolate, calls = case(monkeypatch)
    d.nt_sR.xp = d.taut_sR.xp = np
    f._stress_stencil_checked = True
    seen = []

    class Private:
        def __init__(self, *args):
            pass

        def stress(self, n, tau, D, T):
            assert not np.shares_memory(n, d.nt_sR.data)
            assert not np.shares_memory(tau, d.taut_sR.data)
            n[:] = tau[:] = D[0][:] = 0
            seen.append(T.copy())
            return np.eye(3)

    monkeypatch.setattr(backend, '_StressEngine', Private)
    T = [[d.nt_sR] * 6]
    np.testing.assert_array_equal(f._stress_on_root(d, interpolate, T), np.eye(3))
    np.testing.assert_array_equal(d.nt_sR.data, 1.)
    np.testing.assert_array_equal(d.taut_sR.data, .3)
    assert f._paw_ready and f.engine is not f._stress_engine
    d.nt_sR.data.flat[0] += .01
    with pytest.raises(RuntimeError, match='current XC state'):
        f._stress_on_root(d, interpolate, T)


def test_kpoint_stress_assembly():
    """Full fixed-input stress and exactly-once root hook at 1/2/4 ranks."""
    from ase.build import bulk
    from gpaw import GPAW
    from gpaw.mpi import serial_comm, world, broadcast
    from ciderpress.gpaw.new_backend import CiderXC
    from gpaw.new.pw.stress import calculate_stress
    atoms = bulk('Pt', 'fcc', a=3.92)
    atoms.set_cell(atoms.cell.array @ np.array([
        [1.013, .007, .004], [.007, .991, .009], [.004, .009, 1.006]]),
        scale_atoms=True)

    def target(comm, ranks):
        calc = GPAW(mode={'name':'pw', 'ecut':100}, xc=CiderXC(),
                    kpts={'size':(2, 2, 2), 'gamma':True}, symmetry='off',
                    nbands=12, legacy_gpaw=False, txt=None, communicator=comm,
                    parallel=dict(kpt=ranks, band=1, domain=1,
                                  distribute_density=False))
        calc.create_new_calculation(atoms)
        dft = calc.dft
        for wfs in dft.ibzwfs:
            wfs.psit_nX.data[:] = 0
            wfs.psit_nX.data[0] = np.exp(-.7*wfs.psit_nX.desc.ekin_G)
            wfs._occ_n = np.zeros(12)
            wfs._occ_n[0] = 1.
            wfs._eig_n = np.zeros(12)
        dft.density.update_ked(dft.ibzwfs)
        dft.density.taut_sR.data[:] += .05
        dft.potential, _, _ = dft.pot_calc.calculate(dft.density, dft.ibzwfs)
        return dft

    if world.rank == 0:
        ref = target(serial_comm, 1)
        reference = ref.pot_calc.stress(ref.ibzwfs, ref.density, ref.potential)
    else:
        reference = None
    reference = broadcast(reference, 0, comm=world)
    dft = target(world, world.size)
    f = dft.pot_calc.xc
    original = f._stress_on_root
    calls = [0]
    def counted(*args):
        calls[0] += 1
        return original(*args)
    f._stress_on_root = counted
    actual = dft.pot_calc.stress(dft.ibzwfs, dft.density, dft.potential)
    # Observed same-code 1-vs-2-rank assembly floor is 2.59e-10
    # Ha/Bohr^3 (PAW/grid reductions), below the 1e-7 port gate.
    np.testing.assert_allclose(actual, reference, atol=1e-9, rtol=1e-10)
    assert world.sum_scalar(calls[0]) == 1
    def fail(*args):
        raise RuntimeError('root stress failure')
    f._stress_on_root = fail
    with pytest.raises(RuntimeError, match='root stress failure'):
        dft.pot_calc.stress(dft.ibzwfs, dft.density, dft.potential)
    if world.rank == 0:
        def bad_validation():
            raise RuntimeError('root stress validation failure')
        f.validate_stress_calculation = bad_validation
    with pytest.raises(RuntimeError, match='root stress validation failure'):
        dft.pot_calc.stress(dft.ibzwfs, dft.density, dft.potential)


def test_dipole_stress_rejected():
    from ase import Atoms
    from gpaw import GPAW
    from ciderpress.gpaw.new_backend import CiderXC
    atoms = Atoms('H', positions=[[1, 1, 1]], cell=[4, 4, 8], pbc=[True, True, False])
    calc = GPAW(mode={'name':'pw', 'ecut':80}, xc=CiderXC(),
                legacy_gpaw=False, txt=None, spinpol=False,
                poissonsolver={'dipolelayer':'xy'})
    calc.create_new_calculation(atoms)
    dft = calc.dft
    with pytest.raises(NotImplementedError, match='dipole layer'):
        dft.pot_calc.stress(dft.ibzwfs, dft.density, dft.potential)
