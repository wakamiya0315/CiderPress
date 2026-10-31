"""New augmented input stays separate from CIDER's gradient derivatives."""
import numpy as np
import pytest

from gpaw.core import UGDesc
from gpaw.test.xc.test_rvv10_kernel import CELL, SIZE, POS, fixture
from ciderpress import gpu
from ciderpress.gpaw.cider_fft import get_rho_sxg
from ciderpress.gpaw.new_backend import CiderXC, _RVV10Contribution, restore_cider_xc


def test_kernel_options():
    xc=CiderXC(rvv10=dict(b=11.95))
    assert xc.rvv10['beta']==xc.rvv10['kernel_density']=='augmented'
    for beta in ('augmented','smooth'):
        compat=CiderXC(rvv10=dict(b=11.95,beta=beta,kernel_density='smooth'))
        saved=compat.todict()
        saved['rvv10'].pop('kernel_density')
        assert restore_cider_xc(saved).rvv10['kernel_density']=='smooth'
    old=compat.todict();old['rvv10'].pop('beta');old['rvv10'].pop('kernel_density')
    assert restore_cider_xc(old).rvv10==dict(b=11.95,beta='smooth',kernel_density='smooth')
    with pytest.raises(ValueError,match='requires beta'):
        CiderXC(rvv10=dict(beta='smooth',kernel_density='augmented'))


@pytest.mark.parametrize('use_gpu',[False,True])
@pytest.mark.parametrize('nspin',[1,2])
def test_augmented_composition(use_gpu,nspin,record_property):
    from gpaw.gpu import cupy,as_np
    device=None
    if use_gpu:
        cp=gpu._real_cupy()
        if cp is None: pytest.skip('requires real CUDA CuPy')
        device=cp.cuda.Device().id
    xp=np if device is None else cupy
    setups,density,n=fixture(xp,nspin)
    grid=UGDesc(cell=CELL,size=SIZE)
    f=CiderXC(gradient='lattice').new_functional(grid,xp)
    nl=_RVV10Contribution(dict(b=11.95,nq=12,nR=8192,q_block=4),grid,device)
    nl.set_context(setups,POS,density.D_asii.layout.atomdist)
    nl.set_density(density)
    ns=as_np(n)[None].repeat(nspin,axis=0)/nspin
    rho=get_rho_sxg(ns,f.engine.grad_v,np.full_like(ns,.1))
    e=np.zeros(SIZE);ded=np.zeros_like(rho)
    with gpu.policy(device):
        tensor=nl.add(e,rho,ded,stress=True,grad_v=f.engine.grad_v)
        direct=nl.kernel.evaluate(xp.asarray(ns.sum(axis=0)),f.engine.grad_v,nl.nl,stress=True)
    assert isinstance(tensor, np.ndarray)
    assert np.count_nonzero(ded[:,1:])==0
    np.testing.assert_allclose(ded[:,0],np.broadcast_to(as_np(direct['v']),ns.shape),atol=1e-14,rtol=1e-13)
    np.testing.assert_allclose(as_np(tensor),as_np(direct['stress']),atol=1e-13,rtol=1e-12)
    error=abs(e.sum()*grid.dv-direct['energy'])
    record_property('energy_composition_error_Ha',error)
    assert error<1e-13
    h=np.zeros((nspin,6))
    assert nl.calculate_paw_correction(setups[0],h.copy(),h,0)==0.
    np.testing.assert_allclose(h,np.broadcast_to(setups[0].Delta_pL@as_np(nl.kernel.W_aL[0]),h.shape),atol=1e-15)


def _stress_case(folder, use_gpu):
    """Small bulk CIDER26SS SCF through the actual ASE stress entry point."""
    from ase.build import bulk
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    from gpaw.new.tau_state import StateResiduals

    class Consecutive(StateResiduals):
        calc_last = True

        def __init__(self):
            super().__init__(1e-7)
            self.streak = 0
            self.last = None

        def __call__(self, ctx):
            ok, entry = super().__call__(ctx)
            if self.last != ctx.niter - 1:
                self.streak = 0
            self.last = ctx.niter
            self.streak = self.streak + 1 if ok else 0
            return self.streak >= 3, entry

    atoms = bulk('Ar', 'fcc', a=5.26, cubic=True)
    atoms.positions[0] += [.07, .03, .02]
    from ase.units import Bohr, Ha
    lengths = np.linalg.norm(atoms.cell.array, axis=1)
    gmax = np.sqrt(2*150/Ha)/Bohr
    h = float(.99*np.min(lengths/(2*np.floor(gmax*lengths/np.pi)+1)))
    common = dict(mode=dict(name='pw', ecut=150, auxiliary_cutoff='h'),
                  h=h, nbands=18,
                  symmetry='off', legacy_gpaw=False, communicator=serial_comm,
                  parallel=dict(gpu=use_gpu, kpt=1, band=1, domain=1,
                                distribute_density=False),
                  occupations=dict(name='fermi-dirac', width=.1),
                  eigensolver=dict(name='ppcg', niter=8, min_niter=2),
                  maxiter=300)
    seed = GPAW(**common, xc='PBE',
                convergence=dict(energy=1e-7, density=1e-6, eigenstates=1e-12),
                txt=str(folder/'pbe.log'))
    seed.get_potential_energy(atoms)
    calc = GPAW(**common, xc=CiderXC(gradient='lattice',
                    rvv10=dict(b=11.95, nq=32, q_block=8)),
                mixer=dict(name='pulay', beta=.03, nmaxold=16, weight=20,
                           tau=dict(method='joint', beta=.03, weight=10)),
                convergence=dict(energy=1e-7, density=1e-6, eigenstates=1e-12,
                                 state_residuals=1e-7),
                txt=str(folder/'cider.log'))
    calc.create_new_calculation(atoms)
    source, target = seed.dft, calc.dft
    target.density.nt_sR.data[:] = source.density.nt_sR.data
    target.density.D_asii.data[:] = source.density.D_asii.data
    for old, new in zip(source.ibzwfs, target.ibzwfs):
        new.psit_nX.data[:] = old.psit_nX.data
        new._eig_n, new._occ_n = old.eig_n.copy(), old.occ_n.copy()
        new._P_ani = None
        new.orthonormalized = False
        new.orthonormalize()
    target.density.update_ked(target.ibzwfs)
    target.potential, target.energies, _ = target.pot_calc.calculate(
        target.density, target.ibzwfs)
    guard = Consecutive()
    target.scf_loop.convergence['state_residuals'] = guard
    atoms.calc = calc
    atoms.get_potential_energy()
    assert guard.streak >= 3
    assert max(target.density.tau_state.residuals.values()) <= 1e-7
    # This call used to raise on the NumPy/CuPy sum in _StressEngine.stress.
    stress = atoms.get_stress()
    calc.write(folder/'converged.gpw', mode='all')
    return calc, stress


@pytest.mark.parametrize('use_gpu', [False, True])
def test_augmented_ase_stress(use_gpu, tmp_path, record_property, monkeypatch):
    """Full ASE stress and identical retained-input XC term comparisons."""
    import hashlib
    import json
    from copy import deepcopy
    from gpaw.fftw import FFTWPlans, ESTIMATE
    from gpaw.gpu import as_np
    from gpaw.new.xc import _taut
    from pathlib import Path
    import ciderpress.gpaw.new_backend as backend
    record_property('executed_backend_sha256',
                    hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest())
    record_property('executed_backend_path', backend.__file__)
    if use_gpu and gpu._real_cupy() is None:
        pytest.skip('requires real CUDA CuPy')
    monkeypatch.setattr(FFTWPlans, '_overwrite_flags', ESTIMATE)
    cpu_dir = tmp_path/'cpu'
    cpu_dir.mkdir()
    cpu, reference = _stress_case(cpu_dir, False)
    record_property('CPU_stress_eV_A3', json.dumps(reference.tolist()))
    record_property('CPU_stress_sha256', hashlib.sha256(reference.tobytes()).hexdigest())
    if not use_gpu:
        return
    gpu_dir = tmp_path/'gpu'
    gpu_dir.mkdir()
    device_calc, actual = _stress_case(gpu_dir, True)
    error = float(np.max(abs(actual-reference)))
    record_property('ASE_stress_CPU_GPU_max_eV_A3', error)
    assert error <= 1e-5

    # Preserve precisely the CPU retained n/tau/D and orbital coefficients.
    # This is a diagnostic fixed-input comparison, not another SCF certificate.
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    source = cpu.dft
    terms = []
    for device in (False, True):
        params = deepcopy(cpu.params.todict())
        params['parallel']['gpu'] = device
        params['mixer'].pop('tau')
        params['convergence'].pop('state_residuals')
        params['xc'] = CiderXC(gradient='lattice',
                              rvv10=dict(b=11.95, nq=32, q_block=8))
        fixed = GPAW(**params, legacy_gpaw=False, communicator=serial_comm,
                     txt=str(tmp_path/('fixed-'+str(device)+'.log')))
        fixed.create_new_calculation(cpu.atoms)
        target = fixed.dft
        xp = target.density.nt_sR.xp
        assert target.density.tau_state is None
        for name in ('nt_sR', 'taut_sR', 'D_asii'):
            getattr(target.density, name).data[:] = xp.asarray(
                as_np(getattr(source.density, name).data))
            np.testing.assert_array_equal(as_np(getattr(target.density, name).data),
                                          as_np(getattr(source.density, name).data))
        for old, new in zip(source.ibzwfs, target.ibzwfs):
            new.psit_nX.data[:] = xp.asarray(as_np(old.psit_nX.data))
            new._eig_n, new._occ_n = old.eig_n.copy(), old.occ_n.copy()
            new._P_ani = None
            np.testing.assert_array_equal(as_np(new.psit_nX.data), old.psit_nX.data)
        target.potential, target.energies, _ = target.pot_calc.calculate(
            target.density, target.ibzwfs)
        xc, d = target.pot_calc.xc, target.density
        tensor = _taut(target.ibzwfs, d.nt_sR.desc)
        with xc.diagnostic_stress_without_tau():
            xc._stress_on_root(d, target.pot_calc.interpolate, tensor)
        parts = xc._stress_engine.stress_terms
        assert set(parts) == {'explicit', 'smooth', 'stencil', 'rvv10_kernel'}
        assert all(isinstance(v, np.ndarray) for v in parts.values())
        terms.append({k: v.copy() for k, v in parts.items()})
    from ase.units import Ha, Bohr
    for key in terms[0]:
        delta = float(np.max(abs(terms[1][key]-terms[0][key])))
        record_property('fixed_'+key+'_CPU_GPU_max_Ha', delta)
        intensive = delta/source.density.nt_sR.desc.volume*Ha/Bohr**3
        record_property('fixed_'+key+'_CPU_GPU_max_eV_A3', intensive)
        assert intensive <= 1e-5
