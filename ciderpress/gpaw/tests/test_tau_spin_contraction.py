"""Atomic kinetic density uses a common arithmetic path in each spin channel."""
from types import SimpleNamespace

import numpy as np
import pytest

from ciderpress.gpaw.atom_utils import _PAWCiderContribs


def contribution(nspins, scale=1.):
    rng = np.random.default_rng(529)
    obj = object.__new__(_PAWCiderContribs)
    obj.xcc = SimpleNamespace(tau_pg=scale*rng.normal(size=(15, 84)),
                              taut_pg=scale*rng.normal(size=(15, 84)),
                              tauc_g=scale*rng.random(12),
                              tauct_g=scale*rng.random(12))
    obj._D_sp = rng.normal(size=(nspins, 15))
    obj._dH_sp = np.zeros_like(obj._D_sp)
    obj.w_g = rng.random(84)/84
    return obj


@pytest.mark.parametrize('ae', [False, True])
def test_atomic_tau_unpolarized_representation(ae):
    obj = contribution(1, scale=1e4)
    total = obj._D_sp.copy()
    one = obj.get_kinetic_energy(ae)
    two = obj.get_kinetic_energy(ae, D_sp=np.repeat(total/2, 2, axis=0))
    np.testing.assert_array_equal(two*2, np.repeat(one, 2, axis=0))
    np.testing.assert_array_equal(obj._D_sp, total)


@pytest.mark.parametrize('ae', [False, True])
@pytest.mark.parametrize('nspins', [1, 2])
def test_atomic_tau_linear_adjoint(ae, nspins):
    obj = contribution(nspins)
    rng = np.random.default_rng(190)
    direction = rng.normal(size=obj._D_sp.shape)
    potential = rng.normal(size=(nspins, 84))
    obj.contract_kinetic_potential(potential, ae)
    analytic = np.sum(obj._dH_sp * direction)
    sign = 1 if ae else -1
    for step in [.002, .001, .0005]:
        plus = obj.get_kinetic_energy(ae, D_sp=obj._D_sp + step*direction)
        minus = obj.get_kinetic_energy(ae, D_sp=obj._D_sp - step*direction)
        numeric = sign*np.sum((plus-minus)*potential*obj.w_g)/(2*step)
        np.testing.assert_allclose(numeric, analytic, atol=1e-11, rtol=1e-11)
