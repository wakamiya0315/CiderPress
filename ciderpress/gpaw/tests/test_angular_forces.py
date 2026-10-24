"""Discrete angular-rule regression for the explicit PASDW position derivative.

Run in separate processes with GPAW_ANGULAR_GRID unset, 8,12 and 12,24.
The n/tau/D inputs stay fixed, so this test has no SCF stopping error.
Stationary whole-force qualification is a separate integration calculation.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Ha
from gpaw import GPAW
from gpaw.gpu import as_np, cupy_is_fake
from gpaw.sphere.lebedev import angular_grid_identity
from ciderpress.gpaw.new_backend import CiderXC


@pytest.mark.parametrize('gpu', [False, True])
def test_angular_pasdw_force_finite_difference(gpu):
    if gpu and cupy_is_fake:
        pytest.skip('real CUDA required')
    assert angular_grid_identity()['npoints'] in (50, 96, 288)
    atoms = Atoms('H2', positions=[[1.13, 1.27, 1.31], [1.87, 1.34, 1.49]],
                  cell=[4., 4., 4.], pbc=True)
    calc = GPAW(mode=dict(name='pw', ecut=80), h=.2, nbands=2,
                xc=CiderXC(), symmetry='off', legacy_gpaw=False, txt=None,
                parallel=dict(gpu=gpu, domain=1, band=1, kpt=1))
    calc.create_new_calculation(atoms)
    pc, d = calc.dft.pot_calc, calc.dft.density
    d.taut_sR.data[:] += .05
    fields = [a.copy() for a in (d.nt_sR.data, d.taut_sR.data, d.D_asii.data)]
    positions = calc.dft.relpos_ac.copy()

    def energy(pos):
        pc.move(pos, d.D_asii.layout.atomdist)
        _, energies, _ = pc.calculate(d, calc.dft.ibzwfs)
        for current, saved in zip((d.nt_sR.data, d.taut_sR.data, d.D_asii.data), fields):
            np.testing.assert_array_equal(as_np(current), as_np(saved))
        return float(energies._energies['xc'])*Ha

    energy(positions)
    analytical = pc.xc.force_contribution(d, pc.interpolate)[0, 2]*Ha/Bohr
    delta = .001
    values = []
    for sign in [-1, 1]:
        pos = positions.copy()
        pos[0] += np.array([0., 0., sign*delta]) @ np.linalg.inv(atoms.cell.array)
        values.append(energy(pos))
    numerical = -(values[1]-values[0])/(2*delta)
    assert analytical == pytest.approx(numerical, abs=5e-4)
