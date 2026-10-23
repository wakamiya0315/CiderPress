"""Deterministic FP64 molecular interpolation primitives (no scatter atomics)."""
from functools import lru_cache
import ctypes as ct
import weakref

import numpy as np

from ciderpress.gpu import cupy

_SOURCE = r'''
extern "C" __global__ void scatter_poly(
    const double* t, const double* d, const int* ids, double* out,
    long long n, long long q) {
    long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n*q) return;
    long long g=i/q, k=i%q, dst=(long long)ids[g]*q+k;
    double v=out[dst];
    for (int p=0; p<4; p++) v += t[(g*4+p)*q+k]*d[g*4+p];
    out[dst]=v;
}
extern "C" __global__ void gather_poly(
    const double* x, const double* d, const int* ids, double* out,
    long long n, long long q) {
    long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n*4*q) return;
    long long g=i/(4*q), k=i%q, p=(i/q)%4;
    out[i]=d[g*4+p]*x[(long long)ids[g]*q+k];
}
// The native real-harmonic recurrence from sph_harm.c, with CPU-tabulated
// coefficients and normalized directions. One point has one output owner.
extern "C" __global__ void harmonics(
    const double* dirs, const double* co, double* out, long long n,
    long long nlm, long long lp) {
    long long g=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (g>=n) return;
    double re[121]={0}, im[121]={0};
    double x=dirs[g*3], y=dirs[g*3+1], z=dirs[g*3+2];
    double* res=out+g*nlm;
    const double* c0=co;
    const double* c1=co+lp;
    const double* a=co+2*lp;
    const double* b=co+2*lp+nlm;
    re[0]=0.28209479177387814346;
    res[0]=re[0];
    if (lp==1) return;
    re[lp]=1.7320508075688772936*re[0]*z;
    res[2]=re[lp];
    re[lp+1]=b[0]*x*re[0]; im[lp+1]=b[0]*y*re[0];
    res[1]=-1.4142135623730950488*im[lp+1];
    res[3]=-1.4142135623730950488*re[lp+1];
    for(int l=1;l<lp-1;l++) {
        int lm=l*l+3*l+2;
        re[(l+1)*lp]=c0[l]*z*re[l*lp]-c1[l]*re[(l-1)*lp];
        res[lm]=re[(l+1)*lp];
        for(int m=0;m<=l;m++) {
            int i=l*lp+m, j=i+lp+1, k=i-lp+1;
            // Preserve the complex expression's left-to-right products.
            re[j]=a[i]*re[k]+(b[i]*x*re[i]-b[i]*y*im[i]);
            im[j]=a[i]*im[k]+(b[i]*x*im[i]+b[i]*y*re[i]);
            double fac=(m%2==0 ? -1. : 1.)*1.4142135623730950488;
            res[lm-m-1]=fac*im[j]; res[lm+m+1]=fac*re[j];
        }
    }
}
extern "C" __global__ void angular_ordered(
    const double* x, const double* y, const int* rs, const int* ids, double* out,
    long long batch, long long nang, long long nlm, long long q,
    long long stride, long long offset, long long forward, long long fused) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long count=batch*(forward ? nlm : nang)*q;
    if(t>=count)return;
    long long k=t%q, j=t/q%(forward ? nlm : nang), b=t/q/(forward ? nlm : nang);
    double v=0;
    if(forward) {
        for(long long g=0;g<nang;g++)
            if(fused)v=__fma_rn(y[g*nlm+j],x[(long long)ids[b*nang+g]*stride+offset+k],v);
            else v+=y[g*nlm+j]*x[(long long)ids[b*nang+g]*stride+offset+k];
        out[((long long)rs[b]*nlm+j)*q+k]=v;
    } else {
        for(long long l=0;l<nlm;l++)
            if(fused)v=__fma_rn(y[j*nlm+l],x[((long long)rs[b]*nlm+l)*q+k],v);
            else v+=y[j*nlm+l]*x[((long long)rs[b]*nlm+l)*q+k];
        out[(long long)ids[b*nang+j]*stride+offset+k]=v;
    }
}
extern "C" __global__ void radial_ordered(
    const double* x, const double* b, double* out, long long nr, long long ns,
    long long nm, long long q, long long stride, long long offset, long long forward) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=(forward ? ns : nr)*nm*q)return;
    long long k=t%q,m=t/q%nm,j=t/(nm*q);
    long long dst=forward ? (j*nm+m)*stride+offset+k : t;
    double v=out[dst];
    if(forward) {
        for(long long r=0;r<nr;r++)v+=b[j*nr+r]*x[(r*nm+m)*q+k];
    } else {
        for(long long s=0;s<ns;s++)v+=b[s*nr+j]*x[(s*nm+m)*stride+offset+k];
    }
    out[dst]=v;
}
extern "C" __global__ void atc_ordered(
    const double* x, const double* table, double* out, long long ni, long long no,
    long long nm, long long q, long long forward) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long nsrc=forward ? ni : no, ndst=forward ? no : ni;
    if(t>=ndst*nm*q)return;
    long long k=t%q,m=t/q%nm,j=t/(nm*q);
    double v=out[t];
    for(long long s=0;s<nsrc;s++) {
        double z=0;
        for(long long a=0;a<q;a++) {
            long long index=forward ? (j*q+k)*(ni*q)+s*q+a : (s*q+a)*(ni*q)+j*q+k;
            z=__fma_rn(table[index],x[(s*nm+m)*q+a],z);
        }
        v+=z;
    }
    out[t]=v;
}
extern "C" __global__ void atc_precise(
    const double* x, const double* table, double* out, long long ni, long long no,
    long long nm, long long q, long long forward, long long lanes) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long nsrc=forward ? ni : no, ndst=forward ? no : ni;
    if(t>=ndst*nm*q)return;
    long long k=t%q,m=t/q%nm,j=t/(nm*q);
    double v=out[t];
    for(long long s=0;s<nsrc;s++) {
        dd sum={0,0};double part[8]={0};
        for(long long a=0;a<q;a++) {
            long long index=forward ? (j*q+k)*(ni*q)+s*q+a : (s*q+a)*(ni*q)+j*q+k;
            double c=table[index],z=x[(s*nm+m)*q+a];
            if(lanes==0)sum=dadd(sum,dmul({c,0},{z,0}));
            else part[a%lanes]=__fma_rn(c,z,part[a%lanes]);
        }
        if(lanes==0)v+=(sum.h+sum.l);
        else {
            for(int n=lanes/2;n;n/=2)for(int a=0;a<n;a++)part[a]+=part[a+n];
            v+=part[0];
        }
    }
    out[t]=v;
}
extern "C" __global__ void atc_emulate(
    const double* x,const double* table,double* out,long long ni,long long no,
    long long nm,long long q,long long forward,long long mode) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long nsrc=forward ? ni:no,ndst=forward ? no:ni;
    if(t>=ndst*nm*q)return;
    long long k=t%q,m=t/q%nm,j=t/(nm*q);
    int lanes=1<<(mode%5);mode/=5;
    int tail=mode%2;mode/=2;
    int order=mode%3;mode/=3;
    int fused=mode%2;mode/=2;
    int initial=mode%2;
    double v=out[t];
    for(long long s=0;s<nsrc;s++) {
        double p[16]={0};if(initial)p[0]=v;
        int end=tail ? q/lanes*lanes:q;
        for(int a=0;a<end;a++) {
            long long ix=forward ? (j*q+k)*(ni*q)+s*q+a:(s*q+a)*(ni*q)+j*q+k;
            double c=table[ix],z=x[(s*nm+m)*q+a];
            if(fused)p[a%lanes]=__fma_rn(c,z,p[a%lanes]);else p[a%lanes]+=c*z;
        }
        if(order==0)for(int n=lanes/2;n;n/=2)for(int a=0;a<n;a++)p[a]+=p[a+n];
        else if(order==1)for(int n=1;n<lanes;n*=2)for(int a=0;a<lanes;a+=2*n)p[a]+=p[a+n];
        else for(int a=1;a<lanes;a++)p[0]+=p[a];
        double z=p[0];
        for(int a=end;a<q;a++) {
            long long ix=forward ? (j*q+k)*(ni*q)+s*q+a:(s*q+a)*(ni*q)+j*q+k;
            if(fused)z=__fma_rn(table[ix],x[(s*nm+m)*q+a],z);
            else z+=table[ix]*x[(s*nm+m)*q+a];
        }
        if(initial)v=z;else v+=z;
    }
    out[t]=v;
}
extern "C" __global__ void spline_ordered(
    const double* x, const double* w, double* out, long long nr, long long ns,
    long long nm, long long q, long long nlm, long long l0, long long forward) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long count=forward ? nr*nm*4*q : ns*nm*q;
    if(t>=count)return;
    long long k=t%q,m,r,p,j,dst;
    double v;
    if(forward) {
        p=t/q%4;m=t/(4*q)%nm;r=t/(4*q*nm);
        dst=((r*nlm+l0+m)*4+p)*q+k;v=out[dst];
        for(j=0;j<ns;j++)v+=w[(r*4+p)*ns+j]*x[(j*nm+m)*q+k];
    } else {
        m=t/q%nm;j=t/(q*nm);dst=t;v=out[dst];
        for(r=0;r<nr;r++)for(p=0;p<4;p++)
            v+=x[((r*nlm+l0+m)*4+p)*q+k]*w[(r*4+p)*ns+j];
    }
    out[dst]=v;
}
'''


@lru_cache(None)
def _kernel(name):
    from ciderpress.gpu.lcao_math import _MATH
    return cupy().RawKernel(_MATH+_SOURCE, name, options=("--fmad=false",))


def launch(name, n, args):
    if n < 0 or (n+255)//256 > 2**31-1:
        raise ValueError("Molecular CUDA launch exceeds int64/grid bounds")
    if n:
        _kernel(name)(((n+255)//256,), (256,), args)


def device_array(x, shape=None):
    cp = cupy()
    if not isinstance(x, cp.ndarray) or x.dtype != cp.float64 or not x.flags.c_contiguous:
        raise ValueError("Expected a C-contiguous device FP64 array")
    if shape is not None and x.shape != shape:
        raise ValueError(f"Expected {shape}, got {x.shape}")
    if x.device.id != cp.cuda.Device().id:
        raise ValueError("Array belongs to a different CUDA device")
    return x


def harmonic_coefficients(nlm):
    lp = int(np.sqrt(nlm))
    if lp*lp != nlm or not 1 <= lp <= 11:
        raise NotImplementedError("Real harmonics require lmax <= 10")
    c0, c1 = np.zeros(lp), np.zeros(lp)
    a, b = np.zeros((lp, lp)), np.zeros((lp, lp))
    for l in range(lp):
        c0[l] = np.sqrt(float((2*l+3)*(2*l+1)))/(l+1)
        if l:
            c1[l] = np.sqrt(float(2*l+3)/(2*l-1))*l/(l+1)
        for m in range(lp):
            if m+2 <= l:
                a[l,m] = np.sqrt(float((2*l+3)*(l-m)*(l-m-1))/((2*l-1)*(l+m+2)*(l+m+1)))
            if m <= l:
                b[l,m] = -np.sqrt(float((2*l+3)*(2*l+1))/((l+m+2)*(l+m+1)))
    return np.concatenate((c0, c1, a.ravel(), b.ravel()))


def harmonics(dirs, coefficients, nlm):
    cp = cupy()
    device_array(dirs)
    device_array(coefficients)
    lp = int(np.sqrt(nlm))
    if lp*lp != nlm or not 1 <= lp <= 11 or coefficients.shape != (2*lp+2*nlm,):
        raise ValueError("Invalid harmonic dimension/coefficient table")
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("Expected normalized (N,3) directions")
    out = cp.empty((len(dirs), nlm), dtype=cp.float64)
    launch("harmonics", len(dirs), (dirs, coefficients, out, np.int64(len(dirs)),
                                   np.int64(nlm), np.int64(int(np.sqrt(nlm)))))
    return out


class _Gemm:
    """Private cuBLAS handle/workspace; fixed ordinary FP64 GEMM schedule.

    CuPy 13.6 does not expose the workspace/atomics setters, so use the CUDA
    13 cuBLAS C API for those and for GEMM. Row-major C = A B is evaluated
    as column-major C.T = B.T A.T. Broadcast batches have stride zero.
    """
    def __init__(self):
        cp = cupy()
        self.lib = ct.CDLL("libcublas.so.13")
        self.handle = ct.c_void_p()
        self.workspace = cp.cuda.alloc(4*1024*1024)
        self.stream = None
        def bind(name,args):
            f=getattr(self.lib,name);f.argtypes=args;f.restype=ct.c_int;return f
        p,i,z,ll = ct.c_void_p,ct.c_int,ct.c_size_t,ct.c_longlong
        create=bind("cublasCreate_v2",[ct.POINTER(p)])
        self.destroy=bind("cublasDestroy_v2",[p])
        self.set_stream=bind("cublasSetStream_v2",[p,p])
        self.set_workspace=bind("cublasSetWorkspace_v2",[p,p,z])
        self.dgemm=bind("cublasDgemm_v2",[p,i,i,i,i,i,p,p,i,p,i,p,p,i])
        self.batched=bind("cublasDgemmStridedBatched",[p,i,i,i,i,i,p,p,i,ll,p,i,ll,p,p,i,ll,i])
        self._ok(create(ct.byref(self.handle)))
        weakref.finalize(self,self.destroy,self.handle)
        self._ok(bind("cublasSetAtomicsMode",[p,i])(self.handle,0))
        self._ok(bind("cublasSetMathMode",[p,i])(self.handle,0))
        self._ok(bind("cublasSetPointerMode_v2",[p,i])(self.handle,0))

    @staticmethod
    def _ok(status):
        if status:raise RuntimeError(f"Molecular cuBLAS failure: status {status}")

    def __call__(self,a,b):
        cp=cupy()
        if a.ndim not in (2,3) or b.ndim not in (2,3) or a.shape[-1]!=b.shape[-2]:
            raise ValueError("Invalid molecular GEMM dimensions")
        m,k,n=a.shape[-2],a.shape[-1],b.shape[-1]
        na,nb=(len(a) if a.ndim==3 else 1),(len(b) if b.ndim==3 else 1)
        batch=max(na,nb)
        if na not in (1,batch) or nb not in (1,batch) or max(m,n,k,batch)>2**31-1:
            raise ValueError("Invalid molecular GEMM batch/int32 dimension")
        a,b=cp.ascontiguousarray(a),cp.ascontiguousarray(b)
        shape=(batch,m,n) if a.ndim==3 or b.ndim==3 else (m,n)
        out=cp.empty(shape,dtype=cp.float64)
        if not out.size:return out
        if k==0:out.fill(0);return out
        stream=cp.cuda.get_current_stream().ptr
        if stream!=self.stream:
            self._ok(self.set_stream(self.handle,stream))
            self._ok(self.set_workspace(self.handle,self.workspace.ptr,4*1024*1024))
            self.stream=stream
        alpha,beta=ct.c_double(1),ct.c_double(0)
        args=(self.handle,0,0,n,m,k,ct.byref(alpha),b.data.ptr,n)
        if len(shape)==2:
            status=self.dgemm(*args,a.data.ptr,k,ct.byref(beta),out.data.ptr,n)
        else:
            status=self.batched(*args,k*n if nb>1 else 0,a.data.ptr,k,m*k if na>1 else 0,
                                ct.byref(beta),out.data.ptr,n,m*n,batch)
        self._ok(status)
        return out


class MolecularTransforms:
    """Immutable table snapshot; one stream, atom/bin/tile order is fixed.

    resident: native Y uploaded once. tiled: compact directions resident and
    Y regenerated per grid tile on device; no host streaming in an evaluation.
    Both use ordinary deterministic FP64 cuBLAS GEMMs, no split-K/atomics.
    Reset by constructing a new object when geometry/grid/settings change.
    """

    def __init__(self, generator, *, schedule="resident", tile_size=65536,reduction="gemm"):
        from ciderpress.gpu.lcao_tables import (extract_atc, basis_groups,
                                               radial_tables, geometry_digest)
        cp = cupy()
        if schedule not in ("resident", "tiled") or not isinstance(tile_size, int) or tile_size < 1:
            raise ValueError("Need resident/tiled schedule and a positive tile size")
        self.schedule, self.tile_size = schedule, tile_size
        if reduction not in ("gemm","native_order","angular","radial","atc","spline",
                             "atc_dd","atc_2","atc_4","atc_8","native_dd","angular_nf","native_nf") and not reduction.startswith("em"):
            raise ValueError("Unknown molecular reduction schedule")
        self.reduction=reduction
        self.device = cp.cuda.Device().id
        self._stream_ptr = cp.cuda.get_current_stream().ptr
        self.gemm = _Gemm()
        g, it = generator.grids_indexer, generator.interpolator
        self.identity = geometry_digest(generator)
        self.natm, self.nlm, self.nrad = g.natm, g.nlm, g.nrad
        self.ng, self.ng0 = len(g.idx_map)+g.padding, g.ngrids
        self.nr, self.q = it.nrad, generator.plan.nalpha
        self.idx = cp.asarray(g.idx_map.astype(np.int32))
        if g.ngrids > 2**31-1 or len(np.unique(g.idx_map)) != len(g.idx_map) or np.any(g.idx_map < 0) or np.any(g.idx_map >= g.ngrids):
            raise ValueError("Grid map must be an injective int32 permutation/subset")
        self.all_weights = cp.asarray(g.all_weights)
        self.angular = []
        grouped = {}
        for r in range(g.nrad):
            lo, hi = g.rad_loc[r:r+2]
            y = g.ylm[g.ylm_loc[r]:g.ylm_loc[r]+hi-lo]
            key = (int(hi-lo), y.tobytes())
            if key not in grouped:
                grouped[key] = [y, [], []]
            grouped[key][1].append(r)
            grouped[key][2].append(np.arange(lo, hi, dtype=np.int32))
        for y, rs, ids in grouped.values():
            self.angular.append((cp.asarray(y), cp.asarray(np.array(rs, dtype=np.int32)),
                                 cp.asarray(np.array(ids, dtype=np.int32))))
        self.basis, self.nao = {}, {}
        for label, atco in (("in", generator.ccl.atco_inp), ("out", generator.ccl.atco_out)):
            groups, radial = basis_groups(atco), radial_tables(atco, g)
            self.nao[label] = atco.nao
            self.basis[label] = [(a, l, slice(u,v), slice(int(g.ra_loc[a]),int(g.ra_loc[a+1])), cp.asarray(b))
                                 for (a,l,lo,hi,u,v),b in zip(groups, radial)]
        blocks, self.atc_check = extract_atc(generator.ccl)
        self.atc = [(b.l, b.input_slice, b.output_slice, cp.asarray(b.matrix)) for b in blocks]
        self.spline = []
        for a,l,lo,hi,u,v in basis_groups(generator.ccl.atco_out):
            w = it.w0_rsp[:,lo:hi,:].transpose(0,2,1).reshape(it.nrad*4, hi-lo)
            self.spline.append((a,l,slice(u,v),cp.asarray(np.ascontiguousarray(w))))
        self.coefficients = cp.asarray(harmonic_coefficients(self.nlm))
        self.offsite = []
        for a in range(self.natm):
            # Native stable bins are authoritative, including last-bin exclusion.
            y,d = it._eval_spline_bas_single(a)
            loc = it._loc_ai[a].copy()
            active = int(loc[-2])
            ids = it._ind_ord_fwd[:active].copy()
            if len(np.unique(ids)) != len(ids) or np.any(g.iatom_list[ids] == a):
                raise ValueError("Off-site map contains duplicate/on-site points")
            coords = it._coords_ord[:active] - it.atom_coords[a]
            radius = np.sqrt(np.sum(coords*coords, axis=1))
            dirs = coords/radius[:,None]
            self.offsite.append((tuple(int(v) for v in loc), cp.asarray(ids),
                cp.asarray(d[:active]), cp.asarray(y[:active]) if schedule == "resident" else cp.asarray(dirs)))
        # No borrowed C pointers, host y buffers, or mutable generator retained.

    def _check(self):
        if cupy().cuda.Device().id != self.device:
            raise ValueError("Molecular snapshot belongs to another device")
        if cupy().cuda.get_current_stream().ptr != self._stream_ptr:
            raise ValueError("Molecular snapshot must use its setup CUDA stream")

    def _ordered(self,stage):
        return self.reduction in ("native_order",stage,"native_nf") or (self.reduction=="native_dd" and stage in ("radial","spline")) or (self.reduction=="angular_nf" and stage=="angular")

    def angular_transform(self, x, *, forward=True, out=None, offset=0):
        cp = cupy(); self._check(); device_array(x)
        if not isinstance(offset,(int,np.integer)) or offset < 0:
            raise ValueError("Angular offset must be a nonnegative integer")
        q = x.shape[-1] if forward else x.shape[-1]
        if forward:
            q = self.q if out is None else out.shape[-1]
            if x.shape[0] != self.ng0 or x.ndim != 2 or offset+q > x.shape[-1]:
                raise ValueError("Angular analysis shape/offset mismatch")
            if out is None: out = cp.empty((self.nrad,self.nlm,q))
            device_array(out,(self.nrad,self.nlm,q))
            for y, rs, ids in self.angular:
                if self._ordered("angular"):
                    launch("angular_ordered",len(rs)*self.nlm*q,(x,y,rs,ids,out,
                        *(np.int64(v) for v in (len(rs),ids.shape[1],self.nlm,q,x.shape[-1],offset,1,self.reduction not in ("angular_nf","native_nf")))))
                else:out[rs] = self.gemm(y.T, x[ids,offset:offset+q])
        else:
            if x.shape[:2] != (self.nrad,self.nlm): raise ValueError("Angular synthesis shape mismatch")
            if out is None: out = cp.empty((self.ng0,q))
            device_array(out)
            if out.ndim != 2 or out.shape[0] != self.ng0 or offset+q > out.shape[-1]:
                raise ValueError("Angular synthesis output shape/offset mismatch")
            for y, rs, ids in self.angular:
                if self._ordered("angular"):
                    launch("angular_ordered",ids.size*q,(x,y,rs,ids,out,
                        *(np.int64(v) for v in (len(rs),ids.shape[1],self.nlm,q,out.shape[-1],offset,0,self.reduction not in ("angular_nf","native_nf")))))
                else:out[ids,offset:offset+q] = self.gemm(y, x[rs])
        return out

    def radial_transform(self, x, label, *, forward=True, out=None, offset=0, zero_output=True):
        cp = cupy(); self._check(); device_array(x)
        if label not in self.basis or not isinstance(offset,(int,np.integer)) or offset < 0:
            raise ValueError("Invalid radial basis/offset")
        q = x.shape[-1] if forward else self.q
        if forward:
            if x.shape[:2] != (self.nrad,self.nlm): raise ValueError("Radial input shape mismatch")
            if out is None: out = cp.zeros((self.nao[label],offset+q))
            device_array(out)
            if out.ndim != 2 or out.shape[0] != self.nao[label] or offset+q > out.shape[-1]:
                raise ValueError("Radial projection output shape/offset mismatch")
            if zero_output: out[:,offset:offset+q] = 0
            for a,l,us,rs,b in self.basis[label]:
                compact=cp.ascontiguousarray(x[rs,l*l:(l+1)**2])
                if self._ordered("radial"):
                    launch("radial_ordered",(us.stop-us.start)*q,(compact,b,out[us],
                        *(np.int64(v) for v in (b.shape[1],b.shape[0],2*l+1,q,out.shape[-1],offset,1))))
                else:
                    value = self.gemm(b, compact.reshape(b.shape[1],-1))
                    out[us,offset:offset+q] += value.reshape(-1,q)
        else:
            if x.shape[0] != self.nao[label]: raise ValueError("Orbital input shape mismatch")
            q = out.shape[-1] if out is not None else self.q
            if out is None: out = cp.zeros((self.nrad,self.nlm,q))
            device_array(out,(self.nrad,self.nlm,q))
            if x.ndim != 2 or offset+q > x.shape[-1]:
                raise ValueError("Radial synthesis input shape/offset mismatch")
            if zero_output: out.fill(0)
            for a,l,us,rs,b in self.basis[label]:
                nm = 2*l+1
                if self._ordered("radial"):
                    compact=cp.ascontiguousarray(out[rs,l*l:(l+1)**2])
                    launch("radial_ordered",compact.size,(x[us],b,compact,
                        *(np.int64(v) for v in (b.shape[1],b.shape[0],nm,q,x.shape[-1],offset,0))))
                    out[rs,l*l:(l+1)**2]=compact
                else:
                    value = self.gemm(b.T, cp.ascontiguousarray(x[us,offset:offset+q]).reshape(-1,nm*q))
                    out[rs,l*l:(l+1)**2] += value.reshape(b.shape[1],nm,q)
        return out

    def atc_transform(self, x, *, forward=True, out=None):
        cp = cupy(); self._check(); device_array(x)
        q = self.q
        if x.shape != (self.nao["in" if forward else "out"],q): raise ValueError("ATC input shape mismatch")
        if out is None: out = cp.zeros((self.nao["out" if forward else "in"],q))
        device_array(out,(self.nao["out" if forward else "in"],q))
        for l,ins,outs,t in self.atc:
            src,dst = (ins,outs) if forward else (outs,ins)
            nm = 2*l+1
            if self.reduction.startswith("em"):
                mode=int(self.reduction[2:])
                launch("atc_emulate",out[dst].size,(x[src],t,out[dst],
                    *(np.int64(v) for v in ((ins.stop-ins.start)//nm,(outs.stop-outs.start)//nm,nm,q,forward,mode))))
            elif self.reduction in ("atc_dd","atc_2","atc_4","atc_8","native_dd"):
                lanes={"atc_2":2,"atc_4":4,"atc_8":8}.get(self.reduction,0)
                launch("atc_precise",out[dst].size,(x[src],t,out[dst],
                    *(np.int64(v) for v in ((ins.stop-ins.start)//nm,(outs.stop-outs.start)//nm,nm,q,forward,lanes))))
            elif self._ordered("atc"):
                launch("atc_ordered",out[dst].size,(x[src],t,out[dst],
                    *(np.int64(v) for v in ((ins.stop-ins.start)//nm,(outs.stop-outs.start)//nm,nm,q,forward))))
            else:
                z = cp.ascontiguousarray(x[src].reshape(-1,nm,q).transpose(0,2,1)).reshape(-1,nm)
                v = self.gemm(t if forward else t.T,z)
                out[dst] += v.reshape(-1,q,nm).transpose(0,2,1).reshape(-1,q)
        return out

    def spline_transform(self, x, *, forward=True, out=None):
        cp = cupy(); self._check(); device_array(x)
        q = self.q
        device_array(x,(self.nao["out"],q) if forward else (self.natm,self.nr,self.nlm,4,q))
        if out is None:
            out = cp.zeros((self.natm,self.nr,self.nlm,4,q) if forward else (self.nao["out"],q))
        device_array(out,(self.natm,self.nr,self.nlm,4,q) if forward else (self.nao["out"],q))
        for a,l,us,w in self.spline:
            nm = 2*l+1
            if self._ordered("spline"):
                launch("spline_ordered",(self.nr*nm*4*q if forward else (us.stop-us.start)*q),
                    (x[us] if forward else x[a],w,out[a] if forward else out[us],
                    *(np.int64(v) for v in (self.nr,(us.stop-us.start)//nm,nm,q,self.nlm,l*l,forward))))
            elif forward:
                z = self.gemm(w, x[us].reshape(-1,nm*q))
                out[a,:,l*l:(l+1)**2] += z.reshape(self.nr,4,nm,q).transpose(0,2,1,3)
            else:
                z = cp.ascontiguousarray(x[a,:,l*l:(l+1)**2].transpose(0,2,1,3)).reshape(self.nr*4,nm*q)
                out[us] += self.gemm(w.T,z).reshape(-1,q)
        return out

    def interpolate(self, x, *, forward=True, out=None):
        cp = cupy(); self._check(); device_array(x)
        q = self.q
        device_array(x,(self.natm,self.nr,self.nlm,4,q) if forward else (self.ng,q))
        if out is None:
            out = cp.zeros((self.ng,q) if forward else (self.natm,self.nr,self.nlm,4,q))
        device_array(out,(self.ng,q) if forward else (self.natm,self.nr,self.nlm,4,q))
        if not forward:
            # Native beta=0 overwrites every non-final bin, even an empty one.
            # The single-atom direct special case clears the entire tensor.
            if self.natm==1:out.fill(0)
            else:out[:,:self.nr-1].fill(0)
        for a,(loc,ids,d,basis) in enumerate(self.offsite):
            for r in range(self.nr-1):
                lo,hi = loc[r:r+2]
                for start in range(lo,hi,self.tile_size):
                    end = min(start+self.tile_size,hi); n = end-start
                    y = basis[start:end] if self.schedule == "resident" else harmonics(basis[start:end],self.coefficients,self.nlm)
                    if forward:
                        z = self.gemm(y,x[a,r].reshape(self.nlm,4*q))
                        launch("scatter_poly",n*q,(z,d[start:end],ids[start:end],out,np.int64(n),np.int64(q)))
                    else:
                        z = cp.empty((n,4*q))
                        launch("gather_poly",n*4*q,(x,d[start:end],ids[start:end],z,np.int64(n),np.int64(q)))
                        # Add ordered tile partials; never overwrite earlier tiles.
                        out[a,r] += self.gemm(y.T,z).reshape(self.nlm,4,q)
        return out

    def orbital_grid(self, x, *, forward=True):
        cp = cupy()
        if forward:
            spline = self.spline_transform(x)
            atom = self.angular_transform(self.radial_transform(x,"out",forward=False),forward=False)
            out = cp.zeros((self.ng,self.q))
            out[:len(self.idx)] = atom[self.idx]
            return self.interpolate(spline,out=out)
        atom = cp.zeros((self.ng0,self.q)); atom[self.idx] = x[:len(self.idx)]
        out = self.radial_transform(self.angular_transform(atom),"out")
        return self.spline_transform(self.interpolate(x,forward=False),forward=False,out=out)

    def convolution(self, x, *, forward=True):
        if forward:
            return self.orbital_grid(self.atc_transform(self.radial_transform(self.angular_transform(x),"in")))
        return self.angular_transform(self.radial_transform(self.atc_transform(
            self.orbital_grid(x,forward=False),forward=False),"in",forward=False),forward=False)
