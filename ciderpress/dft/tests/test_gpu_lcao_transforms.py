"""Native parity on single/mixed atoms, empty bins, odd Q and additive buffers."""
import numpy as np
import pytest

from ciderpress import gpu


@pytest.fixture
def device(monkeypatch):
    monkeypatch.delenv("GPAW_CPUPY",raising=False)
    if gpu._real_cupy() is None:pytest.skip("Real CUDA CuPy required")
    monkeypatch.setenv("CIDER_GPU_KERNELS","all")
    return gpu.cupy()


def generator(atom,spin):
    from pyscf import dft,gto
    from ciderpress.pyscf.dft import make_cider_calc
    mol=gto.M(atom=atom,spin=spin,basis="sto-3g",verbose=0)
    mf=make_cider_calc(dft.RKS(mol),"CIDER26XCSURFSCI",xmix=1.,xkernel=None,ckernel=None)
    mf.grids.level=0;mf.grids.alignment=1;mf.grids.build(sort_grids=False)
    mf._numint.initialize_feature_generators(mol,mf.grids,1)
    return mf._numint.nldfgen


def close(cp,got,ref):
    np.testing.assert_allclose(cp.asnumpy(got),ref,atol=1e-12*max(1.,abs(ref).max(initial=0)),rtol=1e-12)


@pytest.mark.parametrize("atom,spin",[("H 0 0 0",1),("O 0 0 0; H 0 0 1; H 0 1 0",0)])
def test_odd_q_offsets_additive_and_empty_bins(device,atom,spin):
    cp=device
    from ciderpress.gpu.lcao_kernels import MolecularTransforms
    gen=generator(atom,spin);g=gen.grids_indexer
    t=MolecularTransforms(gen)
    np.testing.assert_array_equal(cp.asnumpy(t.idx),g.idx_map)
    for a,(loc,ids,d,y) in enumerate(t.offsite):
        gen.interpolator._compute_spline_ind_order(a)
        np.testing.assert_array_equal(loc,gen.interpolator._loc_ai[a])
        np.testing.assert_array_equal(cp.asnumpy(ids),gen.interpolator._ind_ord_fwd[:loc[-2]])
    rng=np.random.default_rng(216)
    q,offset=7,2
    x=rng.normal(size=(g.ngrids,q+3));ref=np.empty((g.nrad,g.nlm,q))
    with gpu.policy(None):g.reduce_angc_ylm_(ref,x,offset=offset)
    close(cp,t.angular_transform(cp.asarray(x),out=cp.empty(ref.shape),offset=offset),ref)
    for label,basis in (("in",gen.ccl.atco_inp),("out",gen.ccl.atco_out)):
        initial=rng.normal(size=(basis.nao,q+3));expected=initial.copy()
        with gpu.policy(None):basis.convert_rad2orb_(ref,expected,g,g.rad_arr,offset=offset,zero_output=False)
        close(cp,t.radial_transform(cp.asarray(ref),label,out=cp.asarray(initial),offset=offset,zero_output=False),expected)
        rinitial=rng.normal(size=ref.shape);rref=rinitial.copy()
        with gpu.policy(None):basis.convert_rad2orb_(rref,expected,g,g.rad_arr,rad2orb=False,offset=offset,zero_output=False)
        close(cp,t.radial_transform(cp.asarray(expected),label,forward=False,out=cp.asarray(rinitial),offset=offset,zero_output=False),rref)
    # Exercise the ATC alpha strides with an odd Q and nonzero initial C.
    # Clone the pinned collection into owned test memory; the live generator
    # and its solved tables remain untouched. The native multiply consumes
    # exactly the cropped table used by the GPU.
    import copy
    import ctypes as ct
    from ciderpress.gpu.lcao_tables import _Collection,extract_atc
    from ciderpress.dft.lcao_convolutions import libcider
    blocks,_=extract_atc(gen.ccl)
    odd=copy.copy(t);odd.q=q;odd.atc=[];raw=[]
    for b in blocks:
        nm=2*b.l+1
        ni=(b.input_slice.stop-b.input_slice.start)//nm
        no=(b.output_slice.stop-b.output_slice.start)//nm
        cropped=b.matrix.reshape(no,t.q,ni,t.q)[:,:q,:,:q]
        odd.atc.append((b.l,b.input_slice,b.output_slice,cp.asarray(np.ascontiguousarray(cropped.reshape(no*q,ni*q)))))
        raw.append(np.ascontiguousarray(cropped.transpose(0,2,1,3)).ravel())
    raw=np.concatenate(raw)
    native=_Collection.from_buffer_copy(ct.string_at(gen.ccl._ccl,ct.sizeof(_Collection)))
    native.nalpha=native.nbeta=q
    native.ovlp_mats=raw.ctypes.data_as(ct.POINTER(ct.c_double))
    for forward in (True,False):
        src,dst=(gen.ccl.atco_inp,gen.ccl.atco_out) if forward else (gen.ccl.atco_out,gen.ccl.atco_inp)
        inp=rng.normal(size=(src.nao,q));initial=rng.normal(size=(dst.nao,q));ref=initial.copy()
        libcider.multiply_atc_integrals(inp.ctypes.data_as(ct.c_void_p),ref.ctypes.data_as(ct.c_void_p),ct.byref(native),ct.c_int(forward))
        close(cp,odd.atc_transform(cp.asarray(inp),forward=forward,out=cp.asarray(initial)),ref)
    inp=rng.normal(size=(gen.ccl.atco_out.nao,q))
    native_it=copy.copy(gen.interpolator);native_it._n0=q
    ref=native_it.conv2spline(inp)
    close(cp,odd.spline_transform(cp.asarray(inp)),ref)
    close(cp,odd.interpolate(cp.asarray(ref)),native_it.interpolate_fwd(ref))
    if g.natm==1:
        assert all(v==0 for v in t.offsite[0][0])
        spline=cp.ones((1,t.nr,t.nlm,4,t.q))
        assert bool(cp.all(t.interpolate(spline)==0))
    initial=np.ones((g.natm,t.nr,t.nlm,4,t.q))
    dual=rng.normal(size=(t.ng,t.q))
    expected=gen.interpolator.interpolate_bwd(np.ascontiguousarray(dual[:len(g.idx_map)]),f_arlpq=initial.copy())
    close(cp,t.interpolate(cp.asarray(dual),forward=False,out=cp.asarray(initial)),expected)
    with pytest.raises(ValueError,match="C-contiguous"):
        t.angular_transform(cp.asarray(x)[:,::2])
    dx=cp.zeros((g.ngrids,t.q))
    with cp.cuda.Stream(non_blocking=True):
        with pytest.raises(ValueError,match="setup CUDA stream"):
            t.angular_transform(dx)


def test_tiled_harmonics_native(device):
    import ctypes
    cp=device
    from ciderpress.gpu.lcao_kernels import harmonics,harmonic_coefficients
    from ciderpress.dft.lcao_convolutions import libcider
    rng=np.random.default_rng(341)
    dirs=rng.normal(size=(1027,3));dirs/=np.sqrt(np.sum(dirs*dirs,axis=1))[:,None]
    ref=np.empty((len(dirs),121))
    libcider.recursive_sph_harm_vec(ctypes.c_int(121),ctypes.c_int(len(dirs)),
        dirs.ctypes.data_as(ctypes.c_void_p),ref.ctypes.data_as(ctypes.c_void_p))
    value=harmonics(cp.asarray(dirs),cp.asarray(harmonic_coefficients(121)),121)
    close(cp,value,ref)
    assert cp.asnumpy(value).tobytes()==cp.asnumpy(harmonics(cp.asarray(dirs),cp.asarray(harmonic_coefficients(121)),121)).tobytes()


def test_negative_active_tau_rejected(device):
    cp=device
    from ciderpress.gpu.lcao_nldf import MolecularNLDF
    gen=generator("H 0 0 0; H 0 0 .74",0)
    nl=MolecularNLDF(gen)
    rho=cp.zeros((5,nl.transforms.ng));rho[0]=.1;rho[4]=-.1
    with pytest.raises(ValueError,match="Negative tau"):
        nl.get_features(rho)
    # Low-density tails retain the native cutoff convention and caller data.
    rho[0]=0
    before=cp.asnumpy(rho).tobytes()
    nl.get_features(rho)
    assert before==cp.asnumpy(rho).tobytes()
