"""Optional kinetic-density provider, repeated evaluation and SCF regression."""
import numpy as np
import pytest
from ase import Atoms
from gpaw import GPAW, KohnShamConvergenceError
from gpaw.mpi import world

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional


@pytest.mark.skipif(world.size != 1, reason="Serial trajectory regression")
def test_tau_provider_trajectory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    atoms = Atoms('H2', positions=[[1., 1., 1.], [1.74, 1., 1.]],
                  cell=[4., 4., 4.], pbc=True)
    atoms.calc = GPAW(mode=dict(name='pw', ecut=200), xc='PBE',
                      occupations=dict(name='fermi-dirac', width=.1),
                      convergence=dict(energy=1e-7, density=1e-7), txt=None)
    atoms.get_potential_energy()
    atoms.calc.write('seed.gpw', mode='all')

    def run(method, repeat=False):
        mixer = dict(beta=.03, nmaxold=5, weight=100)
        if method != 'native':
            mixer['tau'] = dict(method=method, beta=1.)
        xc = get_cider_functional('CIDER26XCSURFSCI', xmix=1.,
                                 xkernel=None, ckernel=None,
                                 pasdw_store_funcs=False)
        calc = CiderGPAW('seed.gpw', xc=xc, mixer=mixer,
                         convergence={'minimum iterations': 4},
                         maxiter=4, txt=None)
        trace, seen = [], set()

        def observe():
            if calc.scf.niter in seen:
                return
            seen.add(calc.scf.niter)
            state = calc.density.tau_state
            row = dict(energy=calc.hamiltonian.e_total_extrapolated,
                       n=calc.density.nt_sG.copy())
            if state is not None:
                row.update(tin=state.tau_in.copy(), tout=state.tau_out.copy(),
                           nout=state.n_out.copy())
                if repeat:
                    before = [state.tau_in.copy(), state.tau_out.copy()]
                    history = [x.copy() for x in state.tau_history]
                    for _ in range(2):
                        calc.hamiltonian.xc.calculate(calc.density.finegd,
                                                      calc.density.nt_sg)
                    for a,b in zip(before, [state.tau_in,state.tau_out]):
                        np.testing.assert_array_equal(a,b)
                    for a,b in zip(history,state.tau_history):
                        np.testing.assert_array_equal(a,b)
            trace.append(row)

        calc.attach(observe, 1)
        try:
            calc.get_atoms().get_potential_energy()
        except KohnShamConvergenceError:
            pass
        return trace

    paths = [run('native'), run('off'), run('linear'), run('off', repeat=True)]
    assert all(len(path) == 4 for path in paths)
    for rows in zip(*paths):
        for row in rows[1:]:
            np.testing.assert_allclose(row['energy'], rows[0]['energy'], atol=1e-11, rtol=0)
            np.testing.assert_array_equal(row['n'], rows[0]['n'])
        for row in rows[2:]:
            for key in ('tin', 'tout', 'nout'):
                np.testing.assert_array_equal(row[key], rows[1][key])
