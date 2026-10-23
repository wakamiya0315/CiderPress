"""Experimental one-device fixed-density CIDER26XCSURFSCI molecular XC.

All setup is explicit and CPU tables are immutable snapshots. Reconstruct
after any geometry/grid/basis/plan change. Inputs are full-grid device FP64
rho (5,N)/(2,5,N), CiderGrids order; returns per-particle exc and unweighted
eval_xc_eff potential. This does not implement a gpu4pyscf SCF frontend.
The default reduction="accurate" uses global double-double ATC and input
radial transforms in both directions; reduction="gemm" retains the original
FP64 cuBLAS comparison.
"""
import numpy as np

from ciderpress.gpu import cupy, policy
from ciderpress.gpu.lcao_kernels import MolecularTransforms, device_array
from ciderpress.gpu.lcao_plan import ResidentPlan, validate_plan
from ciderpress.gpu.pasdw_exponents import arguments


class MolecularNLDF:
    def __init__(self,generator,*,schedule="resident",tile_size=65536,reduction="accurate"):
        validate_plan(generator.plan)
        if cupy().cuda.runtime.getDeviceCount() != 1:
            raise NotImplementedError("Molecular NLDF requires one visible CUDA device")
        if not generator.interpolator.onsite_direct or generator.interpolator._n1:
            raise NotImplementedError("Only direct on-site, scalar version-j interpolation")
        self.plan = ResidentPlan(generator.plan)
        self.transforms = MolecularTransforms(generator,schedule=schedule,tile_size=tile_size,reduction=reduction)
        self.cache = {}

    def get_features(self,rho,spin=0,*,map_grids=True,grad_mode=False):
        if grad_mode or not map_grids:
            raise NotImplementedError("Fixed-density full-grid ground state only")
        cp = cupy(); t = self.transforms
        t._check()
        device_array(rho,(5,t.ng))
        if spin not in range(self.plan.nspin): raise ValueError("Spin index mismatch")
        # An active physical kinetic-energy density is nonnegative. Negative
        # tau can cancel the positive exponent at a nondifferentiable clamp;
        # reject it explicitly instead of promising CPU/GPU derivative parity.
        if bool(cp.any((rho[0]>=self.plan.rhocut)&(rho[4]<0))):
            raise ValueError("Negative tau above rhocut is not a physical MGGA density")
        # Native exponent evaluation zeroes the low-density tau view. Own it.
        work = rho.copy()
        low = work[0] < self.plan.rhocut
        work[4,low] = 0
        a,da = arguments(self.plan.base,work,-1)
        source = cp.where(low,0.,work[0])
        arg0,fun0 = cp.zeros(t.ng0),cp.zeros(t.ng0)
        arg0[t.idx],fun0[t.idx] = a[:len(t.idx)],source[:len(t.idx)]
        theta = self.plan.theta(arg0,fun0*t.all_weights)
        f = t.convolution(theta)
        feat,dfeat = self.plan.eval_rho_full(f,work,spin)
        self.cache[spin] = (work,da,low,arg0,fun0,dfeat)
        return feat

    def get_potential(self,vfeat,spin=0,*,map_grids=True,grad_mode=False):
        if grad_mode or not map_grids:
            raise NotImplementedError("Fixed-density full-grid ground state only")
        if spin not in self.cache: raise RuntimeError("Call get_features for this spin first")
        cp = cupy(); t = self.transforms
        t._check()
        device_array(vfeat,(3,t.ng))
        rho,da,low,arg,fun,dfeat = self.cache[spin]
        vrho = cp.zeros_like(rho)
        vf = self.plan.eval_vxc_full(vfeat.copy(),vrho,dfeat,rho,spin=spin)
        vt = t.convolution(vf,forward=False)
        va0,vf0 = self.plan.theta_reverse(vt,arg,fun)
        va0 *= t.all_weights; vf0 *= t.all_weights
        va,vfun = cp.zeros(t.ng),cp.zeros(t.ng)
        va[:len(t.idx)],vfun[:len(t.idx)] = va0[t.idx],vf0[t.idx]
        vrho[0] += va*da[0]+vfun*(~low)
        vrho[1:4] += va*da[1]*2*rho[1:4]
        vrho[4] += va*da[2]
        return vrho


def validate_model(ni):
    from ciderpress.dft.feat_normalizer import ConstantNormalizer
    from ciderpress.dft.xc_evaluator2 import MappedXC2
    from ciderpress.gpu.semilocal import supported
    s = ni.settings
    if not (type(ni.mlxc) is MappedXC2 and s.nfeat == 6 and s.sl_settings.mode == "nst"
            and s.normalizers.slmode == "nst" and s.nlof_settings.is_empty
            and not s.has_sdmx and not s.has_hyb and supported(ni.mlxc,6)
            and all(n is None or type(n) is ConstantNormalizer for n in s.normalizers._normalizers)
            and ni.xmix == 1 and ni.slxc == "" and ni.mlxc.libxc_baseline is None
            and s.nldf_settings.theta_params == [.5,0.,.015625]
            and s.nldf_settings.feat_params == [[.25,0.,.0078125],[.5,0.,.015625],[1.,0.,.03125]]):
        raise NotImplementedError("Only CIDER26XCSURFSCI MappedXC2/nst/constant-normalizer composition at xmix=1")
    # Model family is checked by settings and supported operators; callers may
    # update the RBF coefficients in place, handled by its mutation-aware cache.


class MolecularXC:
    def __init__(self,ni,weights,*,schedule="resident",tile_size=65536,reduction="accurate"):
        validate_model(ni)
        self.device = cupy().cuda.Device().id
        self.nldf = MolecularNLDF(ni.nldfgen,schedule=schedule,tile_size=tile_size,reduction=reduction)
        self.model,self.rhocut = ni.mlxc,ni.rhocut
        self.constants = tuple(1. if n is None else n.const for n in ni.settings.normalizers._normalizers)
        self.weights = cupy().asarray(np.array(weights,copy=True))
        if self.weights.shape != (self.nldf.transforms.ng,): raise ValueError("Full-grid weights mismatch")
        self.stage_times = {}

    def evaluate(self,rho,*,deriv=1,grad_mode=False,omega=None,timing=False):
        if deriv != 1 or grad_mode or omega not in (None,0):
            raise NotImplementedError("No forces, response, hybrids or higher derivatives")
        cp = cupy(); ns = self.nldf.plan.nspin; ng = len(self.weights)
        device_array(rho,(5,ng) if ns == 1 else (2,5,ng))
        if cp.cuda.Device().id != self.device: raise ValueError("One setup on one device required")
        with policy(self.device):
            return self._evaluate(rho,ns,ng,timing)

    def _evaluate(self,rho,ns,ng,timing):
        import time
        cp = cupy()
        from ciderpress.gpu.semilocal import evaluate_device
        events=[]
        def mark():
            if timing:
                event=cp.cuda.Event();event.record();events.append(event)
                cp.cuda.get_current_stream().synchronize()
            return time.perf_counter()
        t0=mark()
        spinrho = rho[None] if ns==1 else rho
        feat = cp.stack([self.nldf.get_features(spinrho[s],s) for s in range(ns)])
        # Match the CPU generator's private low-density tau convention.
        work = cp.stack([self.nldf.cache[s][0] for s in range(ns)])
        t1=mark()
        sigma = cp.empty((2*ns-1,ng))
        sigma[0] = (work[0,1:4]*work[0,1:4]).sum(0)
        if ns==2:
            sigma[1] = (work[0,1:4]*work[1,1:4]).sum(0)
            sigma[2] = (work[1,1:4]*work[1,1:4]).sum(0)
        rt = (cp.ascontiguousarray(work[:,0]),sigma,cp.ascontiguousarray(work[:,4]))
        x = cp.empty((ns,6,ng))
        x[:,0],x[:,1],x[:,2],x[:,3:] = ns*rt[0],ns*ns*sigma[::2],ns*rt[2],feat
        for i,c in enumerate(self.constants): x[:,i] *= c
        energy,dx,base = evaluate_device(self.model,x,rt,self.rhocut)
        for i,c in enumerate(self.constants): dx[:,i] *= c
        local = cp.zeros_like(work)
        local[:,0] = base[0]
        local[:,1:4] = 2*work[:,1:4]*base[1][::2,None,:]
        if ns==2:
            local[0,1:4] += work[1,1:4]*base[1][1]
            local[1,1:4] += work[0,1:4]*base[1][1]
        local[:,4] = base[2]
        local[:,0] += ns*dx[:,0]
        local[:,1:4] += (2*work[:,1:4])*(ns*ns*dx[:,1,None,:])
        local[:,4] += ns*dx[:,2]
        exc = energy/(rt[0].sum(0)+1e-16)
        t2=mark()
        weighted = local*self.weights
        for s in range(ns):
            weighted[s] += self.nldf.get_potential(cp.ascontiguousarray(dx[s,3:]*self.weights),s)
        vxc = cp.zeros_like(weighted)
        vxc[:] = weighted/cp.where(self.weights!=0,self.weights,1.)
        vxc[:,:,self.weights==0] = 0.
        exc[self.weights==0] = 0
        t3=mark()
        self.stage_times = dict(forward_s=t1-t0,model_s=t2-t1,adjoint_s=t3-t2,total_s=t3-t0)
        if timing:
            self.stage_times.update(device_forward_s=cp.cuda.get_elapsed_time(events[0],events[1])*.001,
                device_model_s=cp.cuda.get_elapsed_time(events[1],events[2])*.001,
                device_adjoint_s=cp.cuda.get_elapsed_time(events[2],events[3])*.001,
                device_total_s=cp.cuda.get_elapsed_time(events[0],events[3])*.001)
        return exc,vxc[0] if ns==1 else vxc
