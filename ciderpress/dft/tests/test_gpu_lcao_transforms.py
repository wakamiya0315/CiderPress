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
    # Build the native reference through the public collection API, without
    # mirroring or altering its private C layout or the live generator.
    import copy
    from ciderpress.gpu.lcao_tables import extract_atc
    from ciderpress.dft.lcao_convolutions import ConvolutionCollection
    native=ConvolutionCollection(gen.ccl.atco_inp,gen.ccl.atco_out,
        gen.ccl._alphas[:q],gen.ccl._alpha_norms[:q],has_vj=True)
    native.solve_projection_coefficients()
    assert native.nalpha==native.nbeta==q
    blocks,_=extract_atc(native)
    odd=copy.copy(t);odd.q=q;odd.atc=[]
    for b in blocks:
        odd.atc.append((b.l,b.input_slice,b.output_slice,cp.asarray(b.matrix)))
    for forward in (True,False):
        src,dst=(gen.ccl.atco_inp,gen.ccl.atco_out) if forward else (gen.ccl.atco_out,gen.ccl.atco_inp)
        inp=rng.normal(size=(src.nao,q));initial=rng.normal(size=(dst.nao,q));ref=initial.copy()
        native.multiply_atc_integrals(inp,output=ref,fwd=forward)
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


@pytest.mark.parametrize("reduction",["accurate","dd_serial"])
def test_atc_global_cancellation_decimal_and_adjoint(device,reduction):
    """A lost cross-shell residual cannot be recovered by per-shell DD."""
    from decimal import Decimal, localcontext
    from ciderpress.gpu.lcao_kernels import MolecularTransforms
    cp=device
    # Minimal owned transform snapshot; odd q, multiple m and input shells,
    # rectangular table, additive C and both requested matrix directions.
    t=object.__new__(MolecularTransforms)
    t.q=7;t.nao={"in":9,"out":6};t.reduction=reduction
    t.device=cp.cuda.Device().id;t._stream_ptr=cp.cuda.get_current_stream().ptr
    rng=np.random.default_rng(719)
    a=rng.normal(size=(14,21))
    a[:,0]=1e16;a[:,7]=1.;a[:,14]=-1e16
    t.atc=[(1,slice(0,9),slice(0,6),cp.asarray(a))]
    x=rng.normal(size=(9,7));x[:,0]=1
    y=rng.normal(size=(6,7))
    hashes=[]
    for forward,inp in [(True,x),(False,y)]:
        matrix=a if forward else a.T
        nsrc,ndst=matrix.shape[1]//7,matrix.shape[0]//7
        initial=rng.normal(size=(ndst*3,7))
        with localcontext() as context:
            context.prec=80
            expected=np.empty_like(initial)
            for j in range(ndst):
                for m in range(3):
                    vector=inp.reshape(nsrc,3,7)[:,m].ravel()
                    for k in range(7):
                        exact=Decimal.from_float(initial[j*3+m,k])+sum(
                            (Decimal.from_float(v)*Decimal.from_float(w)
                             for v,w in zip(vector,matrix[j*7+k])),Decimal(0))
                        expected[j*3+m,k]=float(exact)
        got=t.atc_transform(cp.asarray(inp),forward=forward,out=cp.asarray(initial))
        # DD has ~106 bits, not unlimited precision or a correct-rounding
        # guarantee after cancellation by 1e16. Allow one FP64 ulp or the
        # small absolute DD residual; ordinary GEMM loses order-one terms.
        error=np.abs(cp.asnumpy(got)-expected)
        assert np.all(error<=np.maximum(np.spacing(np.abs(expected)),4e-15))
        again=t.atc_transform(cp.asarray(inp),forward=forward,out=cp.asarray(initial))
        assert cp.asnumpy(got).tobytes()==cp.asnumpy(again).tobytes()
    # Use ordinary-sized coefficients for a sensitive transpose identity.
    t.atc=[(1,slice(0,9),slice(0,6),cp.asarray(a/1e16))]
    lhs=float(cp.vdot(t.atc_transform(cp.asarray(x)),cp.asarray(y)))
    rhs=float(cp.vdot(cp.asarray(x),t.atc_transform(cp.asarray(y),forward=False)))
    np.testing.assert_allclose(lhs,rhs,rtol=1e-14,atol=1e-14)


def test_input_radial_dd_decimal_with_offsets(device):
    from decimal import Decimal,localcontext
    from ciderpress.gpu.lcao_kernels import MolecularTransforms
    cp=device;rng=np.random.default_rng(331)
    t=object.__new__(MolecularTransforms)
    t.q=7;t.nrad=9;t.nlm=4;t.nao={"in":6};t.reduction="accurate"
    t.device=cp.cuda.Device().id;t._stream_ptr=cp.cuda.get_current_stream().ptr
    b=rng.normal(size=(2,9));b[:,0]=1e16;b[:,4]=1;b[:,-1]=-1e16
    t.basis={"in":[(0,1,slice(0,6),slice(0,9),cp.asarray(b))]}
    x=rng.normal(size=(9,4,7));x[0]=1;x[-1]=1
    initial=rng.normal(size=(6,10));expected=initial.copy()
    with localcontext() as context:
        context.prec=80
        for j in range(2):
            for m in range(3):
                for k in range(7):
                    exact=Decimal.from_float(initial[j*3+m,k+2])+sum(
                        (Decimal.from_float(b[j,r])*Decimal.from_float(x[r,m+1,k])
                         for r in range(9)),Decimal(0))
                    expected[j*3+m,k+2]=float(exact)
    got=t.radial_transform(cp.asarray(x),'in',out=cp.asarray(initial),offset=2,zero_output=False)
    assert np.all(np.abs(cp.asnumpy(got)-expected)<=np.maximum(np.spacing(np.abs(expected)),4e-15))
    y=rng.normal(size=(6,10));initial=rng.normal(size=x.shape);expected=initial.copy()
    with localcontext() as context:
        context.prec=80
        for r in range(9):
            for m in range(3):
                for k in range(7):
                    exact=Decimal.from_float(initial[r,m+1,k])+sum(
                        (Decimal.from_float(b[j,r])*Decimal.from_float(y[j*3+m,k+2])
                         for j in range(2)),Decimal(0))
                    expected[r,m+1,k]=float(exact)
    got=t.radial_transform(cp.asarray(y),'in',forward=False,out=cp.asarray(initial),offset=2,zero_output=False)
    assert np.all(np.abs(cp.asnumpy(got)-expected)<=np.maximum(np.spacing(np.abs(expected)),4e-15))
