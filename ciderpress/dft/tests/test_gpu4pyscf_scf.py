"""Small real-CUDA tests for the pinned molecular SCF bridge."""
import importlib.util
import numpy as np
import pytest


@pytest.fixture
def cuda():
    if importlib.util.find_spec("cupy") is None or importlib.util.find_spec("gpu4pyscf") is None:
        pytest.skip("Real CUDA CuPy and gpu4pyscf required")
    from ciderpress.gpu4pyscf.native import preload_cpu
    preload_cpu()
    from ciderpress import gpu
    cp = gpu._real_cupy()
    if cp is None:
        pytest.skip("Real CUDA required")
    if cp.cuda.runtime.getDeviceCount() != 1:
        pytest.skip("One visible GPU required")
    return cp


@pytest.mark.parametrize("spin", [0, 2])
def test_fixed_density_and_guards(cuda, spin):
    cp = cuda
    from pyscf import gto
    from ciderpress.gpu4pyscf import make_cider_calc_gpu
    mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=spin, verbose=0)
    mf = make_cider_calc_gpu(mol, level=0, unrestricted=True)
    cpu = mf._numint.cpu_shell
    dm = np.asarray(cpu.get_init_guess(key="1e"))
    vcpu = cpu.get_veff(dm=dm)
    vgpu = mf.get_veff(dm=cp.asarray(dm))
    assert abs(float(vgpu.exc)-float(vcpu.exc)) <= 1e-9
    np.testing.assert_allclose(cp.asnumpy(vgpu-vgpu.vj), vcpu-vcpu.vj, rtol=0, atol=1e-8)
    original = cp.asnumpy(mf.get_veff(dm=cp.asarray(dm)))
    np.testing.assert_array_equal(cp.asnumpy(mf.get_veff(dm=cp.asarray(dm))), original)
    for method in (mf.to_gpu, mf.to_cpu, mf.nuc_grad_method, mf.gen_response,
                   mf._numint.cache_xc_kernel):
        with pytest.raises(NotImplementedError):
            method()
    with pytest.raises(NotImplementedError):
        mf._numint.eval_xc_eff("R2SCAN", cp.zeros((2, 5, len(mf.grids.weights))), deriv=2)


def test_h2_scf_and_df(cuda):
    from pyscf import gto
    from ciderpress.gpu4pyscf import make_cider_calc_gpu
    from ciderpress.gpu4pyscf.scf import make_pbe_calc
    mol = gto.M(atom="H 0 0 0; H 0 0 .75", basis="sto-3g", verbose=0)
    mf = make_cider_calc_gpu(mol, level=0, df="def2-universal-jfit")
    pbe = make_pbe_calc(mf)
    pbe.kernel()
    mf.kernel(dm0=pbe.make_rdm1())
    cpu = mf._numint.cpu_shell.density_fit(auxbasis="def2-universal-jfit", only_dfj=True)
    cpu.conv_tol, cpu.conv_tol_grad = 1e-9, 1e-6
    cpu.kernel(dm0=cuda.asnumpy(pbe.make_rdm1()))
    assert pbe.converged and cpu.converged and mf.converged
    assert abs(mf.e_tot-cpu.e_tot) < 1e-6
    for method in (mf.to_gpu, mf.to_cpu, mf.Gradients, mf.Hessian, mf.gen_response):
        with pytest.raises(NotImplementedError):
            method()
    # The E0 callable must obey the official FT contract and reproduce the
    # integrated SCF XC on the same grid, without a training-interpolator gate.
    from types import MethodType
    from ciderpress.gpu4pyscf.descriptors import SCFFeatureGenerator
    from ciderpress.ft import from_pyscf, write_records, ReactionSet
    from ciderpress.ft.records import integrate
    from ciderpress.dft.model_utils import load_cider_model
    dm = cuda.asnumpy(mf.make_rdm1())
    cpu.mo_coeff, cpu.mo_occ, cpu.mo_energy = (cuda.asnumpy(getattr(mf,k)) for k in
                                               ("mo_coeff","mo_occ","mo_energy"))
    cpu.make_rdm1 = MethodType(lambda self: dm.copy(), cpu)
    cpu.e_tot, cpu.scf_summary["exc"], cpu.converged = mf.e_tot, mf.scf_summary["exc"], True
    generate = SCFFeatureGenerator(mf, dm)
    record = from_pyscf(cpu, grids=cpu.grids, feature_generator=generate)
    assert record["desc"].shape == (1,6,len(cpu.grids.weights))
    assert abs(integrate(load_cider_model("CIDER26XCSURFSCI"),record,rhocut=1e-9)-mf.scf_summary["exc"])<1e-8


def test_reject_configuration(cuda, monkeypatch):
    from pyscf import gto
    from ciderpress.gpu4pyscf import make_cider_calc_gpu
    mol = gto.M(atom="H 0 0 0", spin=1, basis="sto-3g", verbose=0)
    for options in ({"model":"other"}, {"xmix":.5}, {"df":"weigend"},
                    {"forces":True}, {"smooth_floor_width":.1}):
        with pytest.raises(NotImplementedError):
            make_cider_calc_gpu(mol, **options)
    with pytest.raises(ValueError):
        make_cider_calc_gpu(mol, unrestricted=False)
    import gpu4pyscf
    monkeypatch.setattr(gpu4pyscf,"__version__","1.9.1")
    with pytest.raises(NotImplementedError,match="exactly 1.9.0"):
        make_cider_calc_gpu(mol)


@pytest.mark.parametrize("df",[None,"def2-universal-jfit"])
def test_strict_j_repeatability(cuda,df):
    from pyscf import gto
    from ciderpress.gpu4pyscf import make_cider_calc_gpu
    mol=gto.M(atom="H 0 0 0; H 0 0 .75",basis="sto-3g",verbose=0)
    mf=make_cider_calc_gpu(mol,level=0,df=df,deterministic_j=True)
    cpu=mf._numint.cpu_shell
    if df:cpu=cpu.density_fit(auxbasis=df,only_dfj=True)
    dm=np.asarray(cpu.get_init_guess(key="1e"))
    device=cuda.asarray(dm)
    j0=mf.get_j(mol,device)
    np.testing.assert_array_equal(cuda.asnumpy(j0),cuda.asnumpy(mf.get_j(mol,device)))
    np.testing.assert_allclose(cuda.asnumpy(j0),cpu.get_j(mol,dm),rtol=0,atol=1e-10)
    mf.kernel(dm0=device.copy())
    energy,final=mf.e_tot,cuda.asnumpy(mf.make_rdm1())
    assert mf.converged
    mf.kernel(dm0=device.copy())
    assert mf.converged and mf.e_tot==energy
    np.testing.assert_array_equal(cuda.asnumpy(mf.make_rdm1()),final)
