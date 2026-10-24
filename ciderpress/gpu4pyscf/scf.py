"""Pinned CIDER26SS RKS/collinear UKS energy frontend."""
from types import MethodType
from .native import preload_cpu, require_version
preload_cpu()
require_version()
import cupy as cp
from gpu4pyscf.dft.rks import RKS
from gpu4pyscf.dft.uks import UKS
from .numint import CiderNumInt


def adapt_dfj(mf):
    """Instance-only workaround for 1.9.0 df_jk.py:226 -> df.py:118."""
    require_version()
    import inspect
    from pyscf import gto
    obj = mf.with_df
    original = obj.get_jk
    if not mf.only_dfj or obj.auxbasis != "def2-universal-jfit":
        raise NotImplementedError("Only explicit def2-universal-jfit J-only DF")
    if next(iter(inspect.signature(original).parameters)) != "dm":
        raise RuntimeError("Unexpected gpu4pyscf DF signature")
    info = dict(calls=0, version="1.9.0", caller="df/df_jk.py:226", callee="df/df.py:118")
    def get_jk(this, dm, *args, **kwargs):
        if isinstance(dm, gto.Mole):
            if dm is not this.mol or not args or kwargs.get("with_k") is not False:
                raise ValueError("Unexpected DF-J adapter call")
            dm, args = args[0], args[1:]
            info["calls"] += 1
        return original(dm, *args, **kwargs)
    obj.get_jk = MethodType(get_jk, obj)
    mf.cider_dfj_adapter = info
    return mf


def align_overlap(mf):
    """Keep every positive overlap eigenvector, as in CPU PySCF."""
    info = dict(policy="full positive overlap space", stock_threshold=1e-6)
    def check(this, s, verbose=None):
        from gpu4pyscf.lib.cupy_helper import eigh
        e, v = eigh(s)
        minimum = float(e.min())
        if minimum <= 0:
            raise ValueError(f"Nonpositive overlap eigenvalue: {minimum}")
        info.update(min_eigenvalue=minimum, retained=int(e.size),
                    stock_would_remove=int(cp.count_nonzero(e <= 1e-6)))
        return v / cp.sqrt(e)
    mf.check_linear_dependency = MethodType(check, mf)
    mf.cider_overlap = info
    return mf


class _CiderGPU:
    def kernel(self, *args, **kwargs):
        if not isinstance(self._numint, CiderNumInt) or self.xc != "R2SCAN":
            raise RuntimeError("CIDER GPU integrator or MGGA facade was replaced")
        if bool(getattr(self,"with_df",None)) != self.cider_use_df:
            raise RuntimeError("J method changed; reconstruct the CIDER calculator")
        if self.cider_use_df and self.with_df.auxbasis != "def2-universal-jfit":
            raise RuntimeError("Auxiliary basis changed; reconstruct the CIDER calculator")
        if hasattr(self,"cider_j_schedule"):
            if self.cider_use_df and self.with_df._cderi is None:
                raise RuntimeError("Resident J setup was reset; reconstruct the CIDER calculator")
            from .j import deterministic_solver
            deterministic_solver()
        return super().kernel(*args, **kwargs)

    scf = kernel

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError("CIDER GPU: no forces/response/conversion; use an explicit CPU CIDER shell for export")

    nuc_grad_method = Gradients = Hessian = gen_response = stability = _unsupported
    to_gpu = to_cpu = _unsupported
    TDDFT = TDA = TDHF = _unsupported


class CiderRKS(_CiderGPU, RKS):
    pass


class CiderUKS(_CiderGPU, UKS):
    pass


from gpu4pyscf.scf.diis import CDIIS


class ResidentCDIIS(CDIIS):
    # Explicit device storage: never use gpu4pyscf's host-spill branch.
    incore = True


def make_pbe_calc(cider_calc):
    """GPU PBE seed shell with the exact grid/AO/J protocol of CIDER.

    Immutable J setup is shared; run these calculators sequentially. No new
    CIDER calculation or generic CPU/GPU conversion is performed.
    """
    from pyscf import dft
    if not isinstance(cider_calc._numint,CiderNumInt):
        raise TypeError("Expected a GPU CIDER calculator")
    uk=isinstance(cider_calc._numint.cpu_shell,dft.uks.UKS)
    pbe=(UKS if uk else RKS)(cider_calc.mol)
    pbe.xc,pbe.grids,pbe.small_rho_cutoff="PBE",cider_calc.grids,0
    pbe.max_memory=cider_calc.max_memory
    if hasattr(cider_calc,"with_df"):
        pbe=adapt_dfj(pbe.density_fit(auxbasis="def2-universal-jfit",only_dfj=True))
    align_overlap(pbe)
    if hasattr(cider_calc,"cider_j_schedule"):
        pbe.get_j=cider_calc.get_j
        pbe.cider_j_schedule=cider_calc.cider_j_schedule
    pbe.get_hcore=cider_calc.get_hcore
    pbe.get_ovlp=cider_calc.get_ovlp
    pbe.cider_static_matrices=cider_calc.cider_static_matrices
    pbe.DIIS=ResidentCDIIS
    pbe.conv_tol,pbe.conv_tol_grad,pbe.max_cycle=1e-9,1e-6,100
    pbe.chkfile=None
    return pbe


def make_cider_calc_gpu(mol, model="CIDER26XCSURFSCI", *, xmix=1., xkernel=None,
                        ckernel=None, unrestricted=None, level=3, df=None,
                        schedule="auto", tile_size=8192, ao_block=4096,
                        max_memory=38000, deterministic_j=True, **unsupported):
    """Return a new GPU RKS/UKS with authoritative CPU CiderGrids metadata.

    Supported: CIDER26XCSURFSCI, xmix=1, no extra kernels; one visible GPU;
    integer, collinear RKS/UKS; energy/Vxc; direct J or def2-universal-jfit
    J-only DF. ``unrestricted=True`` permits spin-zero UKS. ``schedule`` is
    auto/resident/tiled, with a whole-SCF estimate in ``cider_memory_policy``.
    No input calculator conversion, force/response or silent CPU fallback.
    Construct a fresh object after changing molecule/grid/model/settings.
    """
    require_version()
    from pyscf import gto
    if not isinstance(mol,gto.Mole):
        raise NotImplementedError("Pass a Mole to the GPU factory; generic CPU mf.to_gpu can drop CIDER")
    if unsupported or model != "CIDER26XCSURFSCI" or xmix != 1 or xkernel is not None or ckernel is not None:
        raise NotImplementedError("Only CIDER26XCSURFSCI/xmix=1 without extra options or kernels")
    if df not in (None, "def2-universal-jfit"):
        raise NotImplementedError("df must be None or def2-universal-jfit")
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise NotImplementedError("Exactly one visible CUDA GPU is required")
    from gpu4pyscf.dft import numint as gpu_numint
    from gpu4pyscf.lib import multi_gpu
    if gpu_numint.num_devices != 1 or multi_gpu.num_devices != 1:
        raise NotImplementedError("gpu4pyscf must also be configured for one GPU")
    from gpu4pyscf.lib import cusolver
    if mol.nao_nr() > min(32600, cusolver.MAX_EIGH_DIM):
        raise NotImplementedError("AO dimension would select gpu4pyscf's CPU eigensolver fallback")
    from gpu4pyscf.dft.numint import MIN_BLK_SIZE
    if ao_block != MIN_BLK_SIZE:
        raise ValueError(f"Pinned gpu4pyscf requires ao_block={MIN_BLK_SIZE}")
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile_size must be positive")
    uk = bool(mol.spin) if unrestricted is None else bool(unrestricted)
    if mol.spin and not uk:
        raise ValueError("RKS requires spin zero")
    from pyscf import dft
    from ciderpress.pyscf.dft import make_cider_calc
    from ciderpress.gpu.lcao_nldf import validate_model
    from ciderpress.gpu.lcao_plan import validate_plan
    from .grids import attach_grid, memory_estimate, choose_schedule
    cpu = make_cider_calc((dft.UKS if uk else dft.RKS)(mol), model,
                          xmix=1., xkernel=None, ckernel=None)
    cpu.max_memory = max_memory
    cpu.small_rho_cutoff = 0
    cpu.grids.level = level
    cpu.grids.alignment = 1
    cpu.grids.build(with_non0tab=True, sort_grids=False)
    cpu._numint.initialize_feature_generators(mol, cpu.grids, 2 if uk else 1)
    validate_model(cpu._numint)
    validate_plan(cpu._numint.nldfgen.plan)
    estimate = memory_estimate(mol, cpu.grids, 2 if uk else 1,
                                df=df is not None, ao_block=ao_block, tile_size=tile_size)
    if deterministic_j:
        from .j import storage_bytes
        jbytes=storage_bytes(mol,df is not None)
        if df is None and 2*jbytes > .8*max_memory*1e6:
            raise MemoryError("Direct ERI setup host peak estimate exceeds max_memory; use a smaller no-DF system")
        estimate["strict_j_resident_bytes"]=jbytes
        estimate["scf_working_bytes"]+=jbytes
        estimate["resident_whole_bytes"]+=jbytes
        estimate["tiled_whole_bytes"]+=jbytes
        if df is not None:
            # gpu4pyscf CDERI setup precedes NLDF allocation and may use two
            # integral/recontraction buffers, each up to 20% of free memory.
            setup_peak=int(jbytes+.4*cp.cuda.runtime.memGetInfo()[0]+2e9)
            estimate["strict_j_setup_peak_bytes"]=setup_peak
            for key in ("resident_whole_bytes","tiled_whole_bytes"):
                estimate[key]=max(estimate[key],setup_peak)
    selected, memory = choose_schedule(estimate, schedule)
    mf = (CiderUKS if uk else CiderRKS)(mol)
    mf.xc = "R2SCAN"  # Pure MGGA facade: tau, no exact exchange.
    mf.max_memory = max_memory
    mf.small_rho_cutoff = 0
    attach_grid(mf, cpu.grids)
    mf._numint = CiderNumInt(cpu, schedule=selected, tile_size=tile_size, ao_block=ao_block)
    if df:
        mf = adapt_dfj(mf.density_fit(auxbasis=df, only_dfj=True))
    align_overlap(mf)
    if deterministic_j:
        from .j import install
        install(mf)
    from .static import install as install_static
    install_static(mf)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-9, 1e-6, 100
    mf.chkfile = None  # Stock checkpointing copies all MOs to CPU each cycle.
    mf.DIIS = ResidentCDIIS
    # DF mixins precede CiderRKS/UKS in the MRO and define conversion and
    # derivative methods of their own. Guard the final instance explicitly.
    for method in ("nuc_grad_method", "Gradients", "Hessian", "gen_response",
                   "stability", "to_gpu", "to_cpu", "TDDFT", "TDA", "TDHF"):
        setattr(mf, method, MethodType(_CiderGPU._unsupported, mf))
    mf.cider_model = model
    mf.cider_use_df = df is not None
    mf.cider_memory_policy = memory
    return mf
