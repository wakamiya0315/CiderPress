"""Resident wrappers around the pinned PASDW spline kernels; no host returns."""
import os

import numpy as np

from ciderpress.gpu import cupy
from ciderpress.gpu.pasdw import _DevicePlan, _launch
from ciderpress.gpu.pasdw_exponents import arguments
from ciderpress.gpu.lcao_kernels import device_array


def validate_plan(plan):
    from ciderpress.dft.plans import NLDFSplinePlan
    s = plan.nldf_settings
    if not (type(plan) is NLDFSplinePlan and plan.alpha_formula == "zexp"
            and plan.nalpha == plan.local_nalpha == 34 and plan.coef_order == "gq"
            and s.nldf_type == "j" and s.sl_level == "MGGA"
            and s.rho_mult == "one" and s.feat_specs == ["se"]*3
            and plan.nspin in (1,2) and plan.smooth_floor_width == 0):
        raise NotImplementedError("Molecular GPU requires j/se*3, MGGA, spline/zexp/gq/all 34 alphas, one/two spins, zero smooth floor")
    selection = os.environ.get("CIDER_GPU_KERNELS", "all").strip()
    required = {"lcao", "pasdw", "rbf", "libxc", "semilocal"}
    if selection != "all" and not required.issubset(set(selection.split(","))):
        raise NotImplementedError("Resident molecular path requires all kernels or lcao,pasdw,rbf,libxc,semilocal; no CPU fallback")


class ResidentPlan(_DevicePlan):
    def __init__(self, base):
        validate_plan(base)
        super().__init__(base, device_output=True)
        self.tables = {i: cupy().asarray(base._local_alpha_transform[i]) for i in (-1,0,1,2)}

    def eval_rho_full(self, f, rho_data, spin=0):
        validate_plan(self.base)
        cp = cupy(); device_array(f); device_array(rho_data)
        ng,nq = f.shape
        if nq != self.nalpha or rho_data.shape != (5,ng) or spin not in range(self.nspin):
            raise ValueError("Spline feature shape/spin mismatch")
        values,derivs = cp.empty((3,ng)),cp.empty((3,ng))
        self.p[spin],self.arguments[spin] = [],[]
        for i in range(3):
            arg,da = arguments(self.base,rho_data,i)
            self.arguments[spin].append(da)
            p = cp.empty_like(f)
            _launch("spline_features",ng,(f,arg,self.tables[i],p,values[i],derivs[i],
                np.int64(ng),np.int64(nq),np.int64(self.nspin)))
            self.p[spin].append(p)
        return values,derivs

    def eval_vxc_full(self,*args,**kwargs):
        validate_plan(self.base)
        if not self._supported():
            raise NotImplementedError("Resident molecular adjoint cannot fall back to CPU")
        spin=kwargs.get("spin",0)
        if spin not in self.p or kwargs.get("p_i_qg") is not None:
            raise ValueError("Need resident forward cache for the same spin")
        return super().eval_vxc_full(*args,**kwargs)

    def theta(self,arg,weighted_source):
        cp = cupy()
        out = cp.empty((arg.size,self.nalpha))
        _launch("theta_grid",out.size,(arg,weighted_source,self.tables[-1],out,
                 np.int64(arg.size),np.int64(self.nalpha)))
        return out

    def theta_reverse(self,x,arg,source):
        cp = cupy()
        va,vf = cp.empty(arg.size),cp.empty(arg.size)
        _launch("theta_adjoint",arg.size,(x,arg,source,self.tables[-1],va,vf,
                 np.int64(arg.size),np.int64(self.nalpha)))
        return va,vf
