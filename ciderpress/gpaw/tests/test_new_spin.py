"""Spin shape/lifecycle checks without running a numerical engine."""
from types import SimpleNamespace
import numpy as np
import pytest
from gpaw.core.atom_arrays import AtomArraysLayout, AtomDistribution
from ciderpress.gpaw.new_backend import CiderNewMGGAFunctional, CiderXC


def functional(nspins):
    f = object.__new__(CiderNewMGGAFunctional)
    f._nspins = nspins
    f._paw_ready = True
    f._density = object()
    f.transfer_timings = []
    return f


def test_noncollinear_message():
    with pytest.raises(NotImplementedError, match='noncollinear'):
        CiderXC().functional(collinear=False)


@pytest.mark.parametrize('dims', [(0,), (3,), (4,), (2, 1)])
def test_unsupported_spin_shape(dims):
    f = functional(None)
    with pytest.raises(NotImplementedError, match='collinear'):
        f.set_density(SimpleNamespace(nt_sR=SimpleNamespace(dims=dims)))
    assert not f._paw_ready


def test_live_spin_count_and_tau_shape():
    f = functional(1)
    with pytest.raises(ValueError, match='spin count'):
        f.set_density(SimpleNamespace(nt_sR=SimpleNamespace(dims=(2,))))
    f = functional(2)
    with pytest.raises(ValueError, match='n/tau'):
        f.set_density(SimpleNamespace(nt_sR=SimpleNamespace(dims=(2,)),
                                      taut_sR=SimpleNamespace(dims=(1,))))
    for dims in [(1,), (2,)]:
        with pytest.raises(ValueError, match='n/tau/D'):
            f._calculate(SimpleNamespace(dims=dims), SimpleNamespace(dims=(1,)))
    with pytest.raises(ValueError, match='requires tau'):
        f._calculate(SimpleNamespace(dims=(2,)))


def test_paw_shape_and_addition():
    f = functional(2)
    setup = object()
    D = np.arange(12.).reshape(2, 6)
    correction = np.arange(12.).reshape(2, 6)/10
    f.setups = [setup]
    f.engine = SimpleNamespace(dens=SimpleNamespace(D_asp={0: D}),
                               dH_asp_tmp={0: correction}, E_a_tmp={0: 3.})
    with pytest.raises(ValueError, match='D/dH'):
        f._calculate_paw_correction(setup, D, np.zeros((1, 6)), 0)
    sentinel = np.arange(12.).reshape(2, 6)/128 + .125
    h = sentinel.copy()
    assert f._calculate_paw_correction(setup, D, h, 0) == 3.
    np.testing.assert_array_equal(h, sentinel+correction)
    f._paw_ready = False
    with pytest.raises(RuntimeError, match='current grid'):
        f._calculate_paw_correction(setup, D, h, 0)


@pytest.mark.parametrize('nspins', [1, 2])
def test_atomic_channel_allocation(nspins):
    from gpaw.mpi import serial_comm
    class Setups(list):
        def empty_atomic_matrix(self, ns, partition):
            assert ns == nspins
            return {a: np.empty((ns, s.ni*(s.ni+1)//2)) for a, s in enumerate(self)}
    f = functional(None)
    atomdist = AtomDistribution([0, 0])
    setups = Setups([SimpleNamespace(ni=2), SimpleNamespace(ni=3)])
    f.engine = SimpleNamespace(setups=setups, atom_partition=atomdist,
                               dens=SimpleNamespace())
    layout = AtomArraysLayout([(2, 2), (3, 3)], atomdist=atomdist)
    D = layout.empty(nspins)
    D.data[:] = np.arange(D.data.size).reshape(D.data.shape)
    nt = SimpleNamespace(dims=(nspins,), desc=SimpleNamespace(comm=serial_comm))
    density = SimpleNamespace(nt_sR=nt, taut_sR=nt, D_asii=D)
    f.set_density(density)
    assert f._nspins == nspins
    assert all(v.shape[0] == nspins for v in f.engine.dens.D_asp.values())
    malformed = layout.empty(3-nspins)
    density.D_asii = malformed
    with pytest.raises(ValueError, match='n/D'):
        f.set_density(density)
