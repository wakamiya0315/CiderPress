"""Same-input geometry roundtrip and cache ownership, without SCF error."""
import numpy as np
import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.mpi import serial_comm
from gpaw.gpu import as_np, cupy_is_fake
from ciderpress.gpaw.new_backend import CiderXC


@pytest.mark.parametrize('gpu', [False, True])
def test_fixed_input_motion(gpu):
    if gpu and cupy_is_fake:
        pytest.skip('real CUDA required')
    atoms = Atoms('H2', positions=[[1.13, 1.27, 1.31], [1.87, 1.34, 1.39]],
                  cell=[4, 4, 4], pbc=True)
    calc = GPAW(mode={'name': 'pw', 'ecut': 100}, xc=CiderXC(),
                legacy_gpaw=False, txt=None, symmetry='off',
                communicator=serial_comm, parallel={'gpu': gpu})
    calc.create_new_calculation(atoms)
    dft = calc.dft
    pc, d = dft.pot_calc, dft.density
    d.taut_sR.data[:] += .05
    fields = [a.copy() for a in (d.nt_sR.data, d.taut_sR.data, d.D_asii.data)]
    f = pc.xc
    def evaluate():
        for a, b in zip((d.nt_sR.data, d.taut_sR.data, d.D_asii.data), fields):
            a[:] = b
        p, energies, _ = pc.calculate(d, dft.ibzwfs)
        force = f.force_contribution(d, pc.interpolate)
        return [np.array(energies._energies['xc'])] + [
            np.array(f.engine.E_a_tmp[a]) for a in range(len(atoms))] + [
            as_np(p.vt_sR.data).copy(), as_np(p.dedtaut_sR.data).copy(),
            as_np(p.dH_asii.data).copy(), force.copy()]
    first = evaluate()
    engine, private = f.engine, f._force_engine
    a = dft.relpos_ac.copy()
    b = a.copy()
    b[1] += [.021, -.013, .017]
    dist = d.D_asii.layout.atomdist
    pc.move(b, dist)
    assert f.engine is not engine and not f._paw_ready
    assert not hasattr(f, '_force_engine')
    with pytest.raises(RuntimeError, match='fresh grid'):
        f.force_contribution(d, pc.interpolate)
    middle = evaluate()
    assert f._force_engine is not private
    assert not np.array_equal(middle[-1], first[-1])
    # A repeated context at the same positions is a no-op for cache ownership.
    current = f.engine
    pc.move(b.copy(), dist)
    assert f.engine is current
    pc.move(a, dist)
    final = evaluate()
    for x, y in zip(first, final):
        np.testing.assert_array_equal(x, y)
    for scf, prop, original in zip(f.engine.setups, f._force_engine.setups, dft.setups):
        assert scf is not prop and scf is not original
        assert scf.cider_contribs is not prop.cider_contribs
    changed = atoms.copy()
    changed.cell[0, 0] += .1
    with pytest.raises(NotImplementedError, match='fixed cell'):
        dft.move_atoms(changed)
    with pytest.raises(NotImplementedError, match='cell changes'):
        calc.params.xc.functional(collinear=True, atoms=changed)


@pytest.mark.parametrize('gpu', [False, True])
def test_fixed_orbital_whole_force_roundtrip(gpu):
    """Generic New terms as well as B/Q, at a fixed PBE-seeded CIDER input.

    Restore the wavefunction bytes too: transporting an SCF guess is not a
    fixed-input test. This intentionally does not certify stationary forces.
    """
    if gpu and cupy_is_fake:
        pytest.skip('real CUDA required')
    atoms = Atoms('H2', positions=[[1.13, 1.27, 1.31], [1.87, 1.34, 1.39]],
                  cell=[4, 4, 4], pbc=True)
    seed = GPAW(mode={'name': 'pw', 'ecut': 80}, xc='PBE', nbands=2,
                legacy_gpaw=False, txt=None, symmetry='off',
                communicator=serial_comm, parallel={'gpu': gpu},
                convergence=dict(energy=1e-5, density=1e-4, eigenstates=1e-6))
    seed.get_potential_energy(atoms)
    calc = seed.new(xc=CiderXC(), txt=None)
    calc.create_new_calculation(atoms)
    dft, src = calc.dft, seed.dft
    d = dft.density
    d.nt_sR.data[:] = src.density.nt_sR.data
    d.D_asii.data[:] = src.density.D_asii.data
    dft.ibzwfs = src.ibzwfs
    d.update_ked(dft.ibzwfs)
    fields = [a.copy() for a in (d.nt_sR.data, d.taut_sR.data, d.D_asii.data)]
    orbitals = [(w.psit_nX.data.copy(), w.orthonormalized, w.eig_n.copy())
                for w in dft.ibzwfs]

    def evaluate():
        for w, (psi, orthonormalized, eig) in zip(dft.ibzwfs, orbitals):
            w.psit_nX.data[:] = psi
            w.orthonormalized = orthonormalized
            w.eig_n = eig.copy()
            w._P_ani = None
        for a, b in zip((d.nt_sR.data, d.taut_sR.data, d.D_asii.data), fields):
            a[:] = b
        dft.potential, energies, _ = dft.pot_calc.calculate(d, dft.ibzwfs)
        dft.results.clear()
        force = dft._calculate_forces().copy()
        arrays = [as_np(a.data).copy() for a in (
            dft.potential.vt_sR, dft.potential.dedtaut_sR,
            dft.potential.dH_asii, dft.potential.vHt_x)]
        return force, energies._energies.copy(), arrays

    first, energy, arrays = evaluate()
    moved = atoms.copy()
    moved.positions[1] += [.03, -.02, .015]
    dft.move_atoms(moved)
    middle, _, _ = evaluate()
    assert not np.array_equal(first, middle)
    dft.move_atoms(atoms)
    returned, restored_energy, restored_arrays = evaluate()
    np.testing.assert_allclose(returned, first, atol=1e-11, rtol=1e-10)
    for key in energy:
        np.testing.assert_allclose(restored_energy[key], energy[key], atol=1e-12, rtol=1e-12)
    for a, b in zip(arrays, restored_arrays):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('gpu', [False, True])
def test_fixed_input_moved_matches_fresh_Pt_core(gpu):
    """Isolate motion caches from SCF precision at B, with nonzero Pt cores."""
    from ase.build import bulk
    if gpu and cupy_is_fake:
        pytest.skip('real CUDA required')
    atoms = bulk('Pt', 'fcc', a=3.92).repeat((2, 1, 1))
    def create(atoms):
        calc = GPAW(mode={'name': 'pw', 'ecut': 80}, xc=CiderXC(),
                    legacy_gpaw=False, txt=None, symmetry='off',
                    communicator=serial_comm, parallel={'gpu': gpu})
        calc.create_new_calculation(atoms)
        return calc
    calc = create(atoms)
    d = calc.dft.density
    d.taut_sR.data[:] += .05
    saved = [a.copy() for a in (d.nt_sR.data, d.taut_sR.data, d.D_asii.data)]
    def evaluate(calc):
        d, pc = calc.dft.density, calc.dft.pot_calc
        for a, b in zip((d.nt_sR.data, d.taut_sR.data, d.D_asii.data), saved):
            a[:] = b
        p, e, _ = pc.calculate(d, calc.dft.ibzwfs)
        return [np.array(e._energies['xc']),
                pc.xc.force_contribution(d, pc.interpolate)] + [
            as_np(a.data).copy() for a in (d.nct_R, d.tauct_R, pc.vbar_g,
                p.vt_sR, p.dedtaut_sR, p.dH_asii, p.vHt_x)]
    evaluate(calc)  # populate the A core, PASDW and private-force caches
    moved = atoms.copy()
    moved.positions[1] += [.03, .01, -.02]
    pos = moved.get_scaled_positions()
    dist = d.D_asii.layout.atomdist
    calc.dft.pot_calc.move(pos, dist)
    d.move(pos, dist)
    actual = evaluate(calc)
    fresh = evaluate(create(moved))
    for a, b in zip(actual, fresh):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
