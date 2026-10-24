"""Spline zexp boundaries, theta adjoint, device-only and rejection contracts."""
import numpy as np
import pytest

from ciderpress import gpu


@pytest.fixture
def device(monkeypatch):
    monkeypatch.delenv("GPAW_CPUPY",raising=False)
    if gpu._real_cupy() is None:pytest.skip("Real CUDA CuPy required")
    monkeypatch.setenv("CIDER_GPU_KERNELS","all")
    with gpu.policy(gpu.cupy().cuda.Device().id):yield gpu.cupy()


def plan(ns):
    from ciderpress.dft.plans import NLDFSplinePlan
    from ciderpress.dft.settings import NLDFSettingsVJ
    s=NLDFSettingsVJ("MGGA",[.5,0.,.015625],"one",["se"]*3,
                     [[.25,0.,.0078125],[.5,0.,.015625],[1.,0.,.03125]])
    return NLDFSplinePlan(s,ns,.5/256,1.6,34,coef_order="gq",alpha_formula="zexp",
                         raise_large_expnt_error=False)


@pytest.mark.parametrize("ns",[1,2])
def test_zexp_all_knots_cutoff_and_theta(device,monkeypatch,ns):
    cp=device
    from ciderpress.gpu.lcao_plan import ResidentPlan
    from ciderpress.gpu.pasdw_exponents import arguments
    from ciderpress.dft.settings import CFC
    p=plan(ns);dp=ResidentPlan(p)
    # Isolate the index boundaries from cancellation in the preceding density
    # exponent: choose B=0 exactly, so a=C*tau/n has identical CPU/GPU inputs.
    # The production CIDER parameters are probed separately below.
    tau_fac=.015625*1.2*(6*np.pi**2)**(2/3)/np.pi
    p.nldf_settings.theta_params[0]=tau_fac
    qs=np.concatenate([np.arange(34)+shift for shift in (-1e-9,0.,1e-9)])
    target=p.alpha0*(p.lambd**qs-1)
    rho=np.zeros((5,len(qs)+4));rho[0]=.01
    rho[0,-4:]=[0.,np.nextafter(p.rhocut,0),p.rhocut,np.nextafter(p.rhocut,np.inf)]
    b=0.;c=np.pi/2**(2/3)*tau_fac/CFC
    rho[4,:len(qs)]=(target-b*(ns*rho[0,:len(qs)])**(2/3))*rho[0,:len(qs)]/c
    for i in (-1,0,1,2):
        with gpu.policy(None):ref=p.get_interpolation_arguments(p.get_rho_tuple(rho.copy()),i=i)
        value=arguments(p,cp.asarray(rho),i)
        for a,z in zip((value[0],*value[1]),(ref[0],*ref[1])):
            np.testing.assert_allclose(cp.asnumpy(a),z,rtol=1e-12,atol=1e-12*max(1,abs(z).max()))
    p.nldf_settings.theta_params[0]=.5
    physical=np.zeros((5,257));physical[0]=np.geomspace(p.rhocut*.5,1e5,257)
    physical[4]=.3*physical[0]**(5/3)
    for i in (-1,0,1,2):
        with gpu.policy(None):ref=p.get_interpolation_arguments(p.get_rho_tuple(physical.copy()),i=i)
        value=arguments(p,cp.asarray(physical),i)
        for a,z in zip((value[0],*value[1]),(ref[0],*ref[1])):
            np.testing.assert_allclose(cp.asnumpy(a),z,rtol=1e-12,atol=1e-12*max(1,abs(z).max()))
    # Exactly specified arguments isolate the polynomial and theta transpose.
    arg=np.clip(qs,0,33-1e-10)
    rng=np.random.default_rng(39)
    fun=rng.normal(size=len(arg));dual=rng.normal(size=(len(arg),34))
    with gpu.policy(None):pc,dc=p.get_interpolation_coefficients(arg,i=-1)
    value=dp.theta(cp.asarray(arg),cp.asarray(fun))
    np.testing.assert_allclose(cp.asnumpy(value),pc*fun[:,None],atol=1e-12,rtol=1e-12)
    va,vf=dp.theta_reverse(cp.asarray(dual),cp.asarray(arg),cp.asarray(fun))
    np.testing.assert_allclose(cp.asnumpy(vf),np.sum(pc*dual,axis=1),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(va),np.sum(dc*dual,axis=1)*fun,atol=1e-12,rtol=1e-12)
    assert cp.asnumpy(value).tobytes()==cp.asnumpy(dp.theta(cp.asarray(arg),cp.asarray(fun))).tobytes()


def test_no_silent_cpu_fallback(monkeypatch):
    from ciderpress.gpu.lcao_plan import validate_plan
    p=plan(1)
    monkeypatch.setenv("CIDER_GPU_KERNELS","none")
    with pytest.raises(NotImplementedError,match="no CPU fallback"):validate_plan(p)
    monkeypatch.setenv("CIDER_GPU_KERNELS","all")
    p.smooth_floor_width=.1
    with pytest.raises(NotImplementedError,match="zero smooth floor"):validate_plan(p)
