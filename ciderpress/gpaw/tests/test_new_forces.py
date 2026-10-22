"""Property input/state contract independent of SCF convergence."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from gpaw.utilities import pack_density

import ciderpress.gpaw.new_backend as backend


def case(monkeypatch):
    grid = NS(cell_cv=np.eye(3), size_c=np.array([2]*3), pbc_c=np.ones(3, bool),
              comm=NS(size=1))

    class Field:
        dims = (1,)
        desc = grid

        def __init__(self, data):
            self.data = data

        def to_xp(self, xp):
            assert xp is np
            return self

    class Matrices(dict):
        layout = NS(atomdist=NS(comm=NS(size=1), rank_a=np.zeros(1, int)))

        def to_xp(self, xp):
            return self

    matrices = Matrices({0: np.array([[[1., .2], [.2, .5]]])})
    density = NS(nt_sR=Field(np.ones((1, 2, 2, 2))),
                 taut_sR=Field(np.full((1, 2, 2, 2), .3)), D_asii=matrices)
    packed = {0: np.array([pack_density(matrices[0][0])])}
    engine = NS(spos_ac=np.zeros((1, 3)), atom_partition=object(),
        dens=NS(D_asp=packed), _force_input_digest=backend._force_input_digest(
            density.nt_sR.data, density.taut_sR.data),
        setups=NS(empty_atomic_matrix=lambda *a: {0: np.empty_like(packed[0])}))
    f = object.__new__(backend.CiderNewMGGAFunctional)
    f.engine, f.grid, f.setups = engine, grid, [object()]
    f.param, f.grad_v, f._device = object(), [], None
    f._density, f._paw_ready, f._relpos = density, True, engine.spos_ac.copy()
    calls = []

    class Private:
        def __init__(self, *args):
            assert args == (f.param, grid, f.grad_v, f.setups, f._relpos)

        def forces(self, n, tau, D):
            assert not np.shares_memory(n, density.nt_sR.data)
            assert not np.shares_memory(tau, density.taut_sR.data)
            np.testing.assert_array_equal(n, density.nt_sR.data)
            np.testing.assert_array_equal(tau, density.taut_sR.data)
            np.testing.assert_array_equal(D[0], packed[0])
            n[:] = 0
            tau[:] = 0
            D[0][:] = 0
            if calls and calls[-1] == 'fail':
                raise RuntimeError('property failure')
            return np.ones((1, 3))

    monkeypatch.setattr(backend, '_ForceEngine', Private)

    def interpolate(field):
        calls.append('n' if field is density.nt_sR else 'tau')
        return field
    return f, density, interpolate, calls


def test_private_property_inputs_and_repeat(monkeypatch):
    f, d, interpolate, calls = case(monkeypatch)
    for _ in range(2):
        np.testing.assert_array_equal(f.force_contribution(d, interpolate), [[1, 1, 1]])
    assert calls == ['n', 'tau']*2
    assert f._paw_ready
    np.testing.assert_array_equal(d.nt_sR.data, 1)
    np.testing.assert_array_equal(d.taut_sR.data, .3)
    np.testing.assert_array_equal(d.D_asii[0], [[[1, .2], [.2, .5]]])


@pytest.mark.parametrize('field', ['nt_sR', 'taut_sR', 'D_asii'])
def test_stale_content_rejected(monkeypatch, field):
    f, d, interpolate, _ = case(monkeypatch)
    if field == 'D_asii':
        d.D_asii[0][0, 0, 0] += .01
    else:
        getattr(d, field).data.flat[0] += .01
    with pytest.raises(RuntimeError, match='current XC state'):
        f.force_contribution(d, interpolate)
    assert not hasattr(f, '_force_engine')


def test_exception_preserves_scf(monkeypatch):
    f, d, interpolate, calls = case(monkeypatch)
    # Trigger after interpolation and after private arrays have been mutated.
    def fail_interp(a):
        result = interpolate(a)
        if a is d.taut_sR:
            calls.append('fail')
        return result
    with pytest.raises(RuntimeError, match='property failure'):
        f.force_contribution(d, fail_interp)
    assert not hasattr(f, '_force_engine')
    assert f._paw_ready and f._density is d
    np.testing.assert_array_equal(d.nt_sR.data, 1)
    np.testing.assert_array_equal(d.taut_sR.data, .3)
    np.testing.assert_array_equal(f.engine.dens.D_asp[0], [[1, .4, .5]])


def test_unsupported_properties_stay_closed(monkeypatch):
    f, d, interpolate, _ = case(monkeypatch)
    with pytest.raises(NotImplementedError, match='stress'):
        f.stress_contribution()
    d.nt_sR.dims = (2,)
    with pytest.raises(NotImplementedError, match='one spin'):
        f.force_contribution(d, interpolate)


def test_real_interpolation_and_private_PAW_state():
    from gpaw.mpi import serial_comm
    from ase import Atoms
    from gpaw import GPAW
    atoms = Atoms('H2',positions=[[1.13,1.27,1.31],[1.87,1.34,1.39]],
                  cell=[4,4,4],pbc=True)
    calc = GPAW(mode={'name':'pw','ecut':100},xc=backend.CiderXC(),
                legacy_gpaw=False,txt=None,symmetry='off',communicator=serial_comm)
    calc.create_new_calculation(atoms)
    pc, d = calc.dft.pot_calc, calc.dft.density
    # A fixed positive tau fixture, without claiming an orbital fixed point.
    d.taut_sR.data[:] += .05
    potential, _, _ = pc.calculate(d,calc.dft.ibzwfs)
    f = pc.xc
    arrays = [d.nt_sR.data,d.taut_sR.data,d.D_asii.data,
              potential.vt_sR.data,potential.dedtaut_sR.data,potential.dH_asii.data]
    saved = [a.copy() for a in arrays]
    force = f.force_contribution(d,pc.interpolate)
    np.testing.assert_allclose(force,sum(f._force_engine.force_terms.values()),atol=1e-14,rtol=1e-14)
    with pytest.raises(NotImplementedError,match='stress'):
        f.stress_contribution()
    np.testing.assert_array_equal(force,f.force_contribution(d,pc.interpolate))
    private = f._force_engine
    def fail(*args,**kwargs):
        private.dens.nt_sg[:] = 0
        raise RuntimeError('private numerical failure')
    private.calc_cider = fail
    with pytest.raises(RuntimeError,match='private numerical failure'):
        f.force_contribution(d,pc.interpolate)
    assert not hasattr(f,'_force_engine')
    np.testing.assert_array_equal(force,f.force_contribution(d,pc.interpolate))
    assert f._paw_ready
    for a,b in zip(arrays,saved):
        np.testing.assert_array_equal(a,b)
    for scf,prop in zip(f.engine.setups,f._force_engine.setups):
        assert scf is not prop
        assert scf.cider_contribs is not prop.cider_contribs
    assert f.engine is not f._force_engine


def test_kpoint_force_assembly(tmp_path):
    """Actual 1/2/4-k-point assembly on identical n/tau/D and orbitals.

    This is an arithmetic/ownership test at a PBE-seeded CIDER input, not a
    claim of a stationary CIDER geometry. Run under mpirun -np 1, 2 and 4.
    """
    from ase import Atoms
    from gpaw import GPAW
    from gpaw.mpi import serial_comm, world
    atoms = Atoms('H2',positions=[[1.13,1.27,1.31],[1.87,1.34,1.39]],
                  cell=[4,4,4],pbc=True)
    # pytest allocates process-local temporary paths; share rank zero's path.
    from gpaw.mpi import broadcast
    checkpoint = broadcast(str(tmp_path/'pbe.gpw') if world.rank == 0 else None,
                           0,comm=world)
    layout = dict(kpt=1,domain=1,band=1,distribute_density=False,gpu=False)
    if world.rank == 0:
        seed = GPAW(mode={'name':'pw','ecut':80},xc='PBE',kpts=(2,2,1),
            nbands=2,legacy_gpaw=False,txt=None,symmetry='off',communicator=serial_comm,
            parallel=layout,convergence=dict(energy=1e-5,density=1e-4,eigenstates=1e-6))
        seed.get_potential_energy(atoms)
        seed.write(checkpoint,mode='all')
    world.barrier()

    def target(comm,kpts):
        seed = GPAW(checkpoint,legacy_gpaw=False,txt=None,communicator=comm,
                    parallel=dict(layout,kpt=kpts))
        seed.dft.ibzwfs.make_sure_wfs_are_read_from_gpw_file()
        calc = seed.new(xc=backend.CiderXC(),txt=None)
        calc.create_new_calculation(atoms)
        src,dst = seed.dft,calc.dft
        dst.density.nt_sR.data[:] = src.density.nt_sR.data
        dst.density.D_asii.data[:] = src.density.D_asii.data
        dst.ibzwfs = src.ibzwfs
        dst.density.update_ked(dst.ibzwfs)
        return calc

    if world.rank == 0:
        reference = target(serial_comm,1)
        r = reference.dft
        r.potential,_,_ = r.pot_calc.calculate(r.density,r.ibzwfs)
        ref_force = r._calculate_forces().copy()
        tau = r.density.taut_sR.data.copy()
    calc = target(world,world.size)
    dft = calc.dft
    if world.rank != 0:
        ref_force = np.empty((2,3))
        tau = np.empty_like(dft.density.taut_sR.data)
    world.broadcast(tau,0)
    world.broadcast(ref_force,0)
    dft.density.taut_sR.data[:] = tau
    dft.potential,_,_ = dft.pot_calc.calculate(dft.density,dft.ibzwfs)
    calls = []
    original = dft.pot_calc.xc.force_contribution

    def counted(*args):
        calls.append(1)
        return original(*args)
    dft.pot_calc.xc.force_contribution = counted
    force = dft._calculate_forces()
    assert world.sum_scalar(len(calls)) == 1
    np.testing.assert_allclose(force,ref_force,atol=1e-9,rtol=1e-8)


def test_tau_input_uses_retained_state_not_output(monkeypatch):
    f, d, interpolate, calls = case(monkeypatch)
    # Deliberately different output and history; neither is an XC input.
    d.tau_state = NS(tau_in=NS(data=d.taut_sR.data.copy()-.1),
                     output=NS(taut_sR=NS(data=np.full_like(d.taut_sR.data, 99.))),
                     history=object())
    d.tauct_R = NS(data=np.full((2, 2, 2), .1))
    reference = f.force_contribution(d, interpolate)
    d.tau_state.history = object()
    d.tau_state.output.taut_sR.data[:] = -99.
    np.testing.assert_array_equal(reference, f.force_contribution(d, interpolate))
    d.tau_state = None
    np.testing.assert_array_equal(reference, f.force_contribution(d, interpolate))
    assert calls == ['n', 'tau']*3


def test_unconverged_cider_tau_force_rejected(tmp_path):
    from ase import Atoms
    from gpaw import GPAW, KohnShamConvergenceError
    from gpaw.mpi import serial_comm
    atoms = Atoms('H2', positions=[[1.13, 1.27, 1.31], [1.87, 1.34, 1.39]],
                  cell=[4, 4, 4], pbc=True)
    seed = GPAW(mode={'name': 'pw', 'ecut': 100}, xc='PBE',
                legacy_gpaw=False, symmetry='off', txt=None,
                communicator=serial_comm,
                convergence={'energy': 1e-8, 'density': 1e-8, 'eigenstates': 1e-10})
    seed.get_potential_energy(atoms)
    calc = seed.new(xc=backend.CiderXC(),
                mixer={'name': 'pulay', 'beta': .03,
                       'tau': {'method': 'joint', 'beta': .03, 'weight': 10}},
                convergence={'energy': 1e-8, 'density': 1e-8,
                             'state_residuals': 1e-8, 'eigenstates': 1e-18},
                eigensolver={'name': 'ppcg', 'niter': 8, 'min_niter': 2},
                maxiter=2, txt=str(tmp_path/'cider.log'))
    calc.create_new_calculation(atoms)
    source, target = seed.dft, calc.dft
    target.density.nt_sR.data[:] = source.density.nt_sR.data
    target.density.D_asii.data[:] = source.density.D_asii.data
    target.ibzwfs = source.ibzwfs
    target.density.update_ked(target.ibzwfs)
    target.potential, target.energies, _ = target.pot_calc.calculate(
        target.density, target.ibzwfs)
    with pytest.raises(KohnShamConvergenceError):
        calc.get_potential_energy(atoms)
    dft = calc.dft
    d, p = dft.density, dft.potential
    arrays = [d.nt_sR.data, d.taut_sR.data, d.D_asii.data,
              p.vt_sR.data, p.dH_asii.data, p.dedtaut_sR.data]
    before = [a.copy() for a in arrays]
    with pytest.raises(RuntimeError, match='fresh tau state'):
        dft._calculate_forces()
    assert not hasattr(dft.pot_calc.xc, '_force_engine')
    for a, b in zip(arrays, before):
        np.testing.assert_array_equal(a, b)
