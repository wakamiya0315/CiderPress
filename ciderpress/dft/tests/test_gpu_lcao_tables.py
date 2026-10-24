"""Versioned native snapshot and owned table checks, independently of CUDA."""
import ctypes

import numpy as np
import pytest


@pytest.fixture
def native_generator(monkeypatch):
    monkeypatch.setenv("CIDER_GPU_KERNELS","none")
    from pyscf import dft,gto
    from ciderpress.pyscf.dft import make_cider_calc
    mol=gto.M(atom="H 0 0 0; H 0 0 0.74",basis="sto-3g",verbose=0)
    mf=make_cider_calc(dft.RKS(mol),"CIDER26XCSURFSCI",xmix=1.,xkernel=None,ckernel=None)
    mf.grids.level=0;mf.grids.alignment=1;mf.grids.build(sort_grids=False)
    mf._numint.initialize_feature_generators(mol,mf.grids,1)
    return mf._numint.nldfgen


def test_snapshot_owned_atc(native_generator):
    from ciderpress.gpu.lcao_tables import extract_atc,apply_atc_numpy,_snapshot,basis_groups
    ccl=native_generator.ccl
    blocks,check=extract_atc(ccl)
    rows,raw,pairs=_snapshot(ccl)
    assert check["accessor"]=="cider_export_atc_v1" and check["abi_version"]==1
    assert rows.shape==(len(blocks),11) and rows.dtype==np.dtype(np.intc)
    assert raw.shape==(pairs*ccl.nalpha*ccl.nbeta,) and raw.dtype==np.dtype(np.float64)
    assert check["shell_pairs"]==pairs
    offset=0
    for b,row,i,o in zip(blocks,rows,basis_groups(ccl.atco_inp),basis_groups(ccl.atco_out)):
        assert tuple(row)==(*i,*o[2:],offset)
        ni,no=i[3]-i[2],o[3]-o[2]
        count=ni*no*ccl.nalpha*ccl.nbeta
        start=offset*ccl.nalpha*ccl.nbeta
        expected=raw[start:start+count].reshape(no,ni,ccl.nbeta,ccl.nalpha)
        expected=expected.transpose(0,2,1,3).reshape(no*ccl.nbeta,ni*ccl.nalpha)
        assert b.matrix.tobytes()==expected.tobytes()
        offset+=ni*no
    assert offset==pairs
    assert set(check["checks"])=={"forward","adjoint"}
    assert all(v["passed"] for v in check["checks"].values())
    assert all(not b.matrix.flags.writeable for b in blocks)
    x=np.random.default_rng(4).normal(size=(ccl.atco_inp.nao,ccl.nalpha))
    value=apply_atc_numpy(blocks,x,ccl.atco_out.nao,ccl.nalpha)
    ref=ccl.multiply_atc_integrals(x)
    np.testing.assert_allclose(value,ref,rtol=1e-12,atol=1e-12*max(1,abs(ref).max()))


def test_snapshot_refuses_unknown_abi(native_generator,monkeypatch):
    from ciderpress.gpu import lcao_tables
    export=lcao_tables._accessor()
    def unknown_version(*args):
        rc=export(*args)
        if rc==0:
            ctypes.cast(args[3],ctypes.POINTER(ctypes.c_size_t))[0]=999
        return rc
    monkeypatch.setattr(lcao_tables,"_accessor",lambda:unknown_version)
    with pytest.raises(RuntimeError,match="snapshot dimension/version mismatch"):
        lcao_tables.extract_atc(native_generator.ccl)


def test_geometry_identity_includes_content(native_generator):
    from ciderpress.gpu.lcao_tables import geometry_digest
    before=geometry_digest(native_generator)
    native_generator.grids_indexer.all_weights[0]+=1e-7
    assert geometry_digest(native_generator)!=before
