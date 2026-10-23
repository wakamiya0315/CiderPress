"""Pinned native ABI and owned table checks, independently of CUDA."""
import ctypes
from pathlib import Path

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


def test_pinned_owned_atc(native_generator):
    from ciderpress.gpu.lcao_tables import extract_atc,apply_atc_numpy,_Collection
    ccl=native_generator.ccl
    blocks,check=extract_atc(ccl)
    assert ctypes.sizeof(_Collection)==96
    assert all(v["passed"] for v in check["checks"].values())
    assert all(not b.matrix.flags.writeable for b in blocks)
    x=np.random.default_rng(4).normal(size=(ccl.atco_inp.nao,ccl.nalpha))
    value=apply_atc_numpy(blocks,x,ccl.atco_out.nao,ccl.nalpha)
    ref=ccl.multiply_atc_integrals(x)
    np.testing.assert_allclose(value,ref,rtol=1e-12,atol=1e-12*max(1,abs(ref).max()))


def test_abi_refuses_unknown_header(native_generator,monkeypatch):
    from ciderpress.gpu import lcao_tables
    monkeypatch.setattr(lcao_tables,"HEADER_SHA256","unknown")
    with pytest.raises(RuntimeError,match="Unknown.*ABI"):
        lcao_tables.extract_atc(native_generator.ccl)


def test_geometry_identity_includes_content(native_generator):
    from ciderpress.gpu.lcao_tables import geometry_digest
    before=geometry_digest(native_generator)
    native_generator.grids_indexer.all_weights[0]+=1e-7
    assert geometry_digest(native_generator)!=before
