"""Immutable one-electron setup, uploaded once before the GPU iteration."""
import hashlib
from types import MethodType
import numpy as np
import cupy as cp


def install(mf):
    # gpu4pyscf 1.9.0 can change hcore by ~1e-14 on repeated calls. A
    # fixed input matrix is essential for byte-reproducible diffuse-basis
    # SCF. CPU integral construction here is explicit setup, not a per-step
    # CIDER fallback; scalar ECP terms are included by PySCF's get_hcore.
    from pyscf.scf.hf import get_hcore
    h = np.asarray(get_hcore(mf.mol), order="C")
    s = np.asarray(mf.mol.intor_symmetric("int1e_ovlp"), order="C")
    hg, sg = cp.asarray(h), cp.asarray(s)
    mol = mf.mol
    def getter(value):
        def get(this, mol=None):
            if mol is not None and mol is not this.mol:
                raise ValueError("One-electron matrices belong to the setup Mole")
            if this.mol is not mf.mol or this.mol is not original_mol:
                raise ValueError("Molecule changed; reconstruct the calculator")
            return value.copy()
        return get
    original_mol = mol
    mf.get_hcore = MethodType(getter(hg), mf)
    mf.get_ovlp = MethodType(getter(sg), mf)
    mf.cider_static_matrices = dict(
        method="CPU PySCF immutable setup; GPU copies throughout SCF",
        hcore_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
        overlap_sha256=hashlib.sha256(s.tobytes()).hexdigest(),
        bytes=int(h.nbytes+s.nbytes))

