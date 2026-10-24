"""Explicit reproducible J schedules; setup only may use host integrals.

The stock 1.9.0 J engines sum floating-point contributions atomically. For
the strict SCF byte gate, use resident CDERI with fixed auxiliary blocks for
DF, or a resident four-center tensor for small direct-J systems. No rounding
or change of the J approximation is introduced. Unsupported sizes raise.
"""
import ctypes
import hashlib
from types import MethodType
import numpy as np
import cupy as cp

_solver = None


def deterministic_solver():
    global _solver
    if _solver is None:
        _solver = ctypes.CDLL("libcusolver.so")
        _solver.cusolverDnSetDeterministicMode.argtypes = (ctypes.c_void_p, ctypes.c_int)
        _solver.cusolverDnSetDeterministicMode.restype = ctypes.c_int
    status = _solver.cusolverDnSetDeterministicMode(cp.cuda.Device().cusolver_handle, 1)
    if status:
        raise RuntimeError(f"cusolverDnSetDeterministicMode failed: {status}")
    cublas=ctypes.CDLL("libcublas.so")
    cublas.cublasSetAtomicsMode.argtypes=(ctypes.c_void_p,ctypes.c_int)
    cublas.cublasSetAtomicsMode.restype=ctypes.c_int
    status=cublas.cublasSetAtomicsMode(cp.cuda.Device().cublas_handle,0)
    if status:raise RuntimeError(f"cublasSetAtomicsMode failed: {status}")


def storage_bytes(mol, df):
    if df:
        from pyscf.df.addons import make_auxmol
        aux = make_auxmol(mol, "def2-universal-jfit")
        return 8*aux.nao_nr()*mol.nao_nr()*(mol.nao_nr()+1)//2
    return 8*mol.nao_nr()**4


def install(mf, *, aux_block=128):
    """Install an explicit GPU-only per-step J schedule on this instance."""
    deterministic_solver()
    info = dict(kind="resident CDERI fixed auxiliary blocks" if hasattr(mf,"with_df") else
                "resident four-center tensor", block=aux_block, setup_s=None,
                cusolver_deterministic_mode=1,cublas_atomics_mode=0)
    import time
    start = time.perf_counter()
    if hasattr(mf,"with_df"):
        # Build with the stock gpu4pyscf integrals/metric. A non-None CDERI
        # bypasses df.py's atomic J-only shortcut, retaining with_k=False.
        obj = mf.with_df
        obj.use_gpu_memory = True
        obj.build(build_cderi=True)
        arrays=list(obj._cderi.values()) if isinstance(obj._cderi,dict) else list(obj._cderi)
        if not all(isinstance(value,cp.ndarray) for value in arrays):
            raise RuntimeError("Strict DF J requires resident device CDERI; no host streaming")
        def block(this, *args, **kwargs):
            return aux_block
        obj.get_blksize = MethodType(block,obj)
        info["resident_bytes"] = sum(value.nbytes for value in arrays)
        digests=[]
        for value in arrays:
            flat=(value.T if value.flags.f_contiguous else value).ravel()
            digest=hashlib.sha256()
            for p in range(0,flat.size,8*1024*1024):
                digest.update(cp.asnumpy(flat[p:p+8*1024*1024]))
            digests.append(digest.hexdigest())
        info["integral_sha256_memory_order"]=digests
    else:
        n = mf.mol.nao_nr()
        # Host work is immutable setup. Every J contraction thereafter is one
        # fixed FP64 GPU matrix-vector product, in the original AO order.
        raw=np.asarray(mf.mol.intor("int2e",aosym="s1"),order="C").reshape(n*n,n*n)
        info["integral_sha256_C_order"]=hashlib.sha256(raw).hexdigest()
        eri = cp.asarray(raw)
        del raw
        info["resident_bytes"] = eri.nbytes
        def get_j(this, mol=None, dm=None, hermi=1, *args, **kwargs):
            if mol is not None and mol is not this.mol:
                raise ValueError("J tensor belongs to the original molecule")
            if hermi != 1 or args or any(v is not None for v in kwargs.values()):
                raise NotImplementedError("Strict direct J supports real Hermitian ground states only")
            if dm is None:dm=this.make_rdm1()
            if not isinstance(dm,cp.ndarray) or dm.dtype!=cp.float64:
                raise ValueError("Strict direct J requires a device FP64 density")
            shape = dm.shape
            if shape[-2:]!=(n,n):raise ValueError("Direct J AO shape mismatch")
            flat=dm.swapaxes(-1,-2).reshape(-1,n*n)
            return (eri@flat.T).T.reshape(shape)
        mf.get_j = MethodType(get_j,mf)
    cp.cuda.get_current_stream().synchronize()
    info["setup_s"] = time.perf_counter()-start
    mf.cider_j_schedule = info
    return mf
