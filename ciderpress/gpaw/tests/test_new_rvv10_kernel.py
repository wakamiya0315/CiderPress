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
    assert np.count_nonzero(ded[:,1:])==0
    np.testing.assert_allclose(ded[:,0],np.broadcast_to(as_np(direct['v']),ns.shape),atol=1e-14,rtol=1e-13)
    np.testing.assert_allclose(as_np(tensor),as_np(direct['stress']),atol=1e-13,rtol=1e-12)
    error=abs(e.sum()*grid.dv-direct['energy'])
    record_property('energy_composition_error_Ha',error)
    assert error<1e-13
    h=np.zeros((nspin,6))
    assert nl.calculate_paw_correction(setups[0],h.copy(),h,0)==0.
    np.testing.assert_allclose(h,np.broadcast_to(setups[0].Delta_pL@as_np(nl.kernel.W_aL[0]),h.shape),atol=1e-15)
