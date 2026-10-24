"""Higher precision intermediates for the molecular exponent/index wrapper.

The same native formula is retained. Error-free sums/products refine cbrt
and a range-reduced log before rounding to FP64. Explicit fma is used only
for product residuals; implicit contraction remains disabled. This is not a
claim of correctly rounded elementary functions for every FP64 argument.
"""
from functools import lru_cache
import types

from ciderpress.gpu import cupy

_MATH = r'''
struct dd { double h,l; };
__device__ dd dsum(double a,double b) {
    double s=a+b,v=s-a;
    return {s,(a-(s-v))+(b-v)};
}
__device__ dd dadd(dd a,dd b) {
    dd s=dsum(a.h,b.h);
    return dsum(s.h,(s.l+a.l)+b.l);
}
__device__ dd dneg(dd a) { return {-a.h,-a.l}; }
__device__ dd dmul(dd a,dd b) {
    double p=a.h*b.h;
    double e=__fma_rn(a.h,b.h,-p);
    e+=(a.h*b.l+a.l*b.h);
    e+=a.l*b.l;
    return dsum(p,e);
}
__device__ dd dscale(dd a,double b) { return dmul(a,{b,0}); }
__device__ dd ddiv(dd a,dd b) {
    double q=a.h/b.h;
    dd r=dadd(a,dneg(dscale(b,q)));
    return dsum(q,(r.h+r.l)/b.h);
}
__device__ dd log_dd(double x) {
    if (!(x>0) || !isfinite(x)) return {log(x),0};
    int e;
    double m=frexp(x,&e);
    if(m<0.70710678118654752440) { m*=2;e--; }
    dd t=ddiv(dsum(m,-1),dsum(m,1));
    dd t2=dmul(t,t),term=t,total=t;
    for(int k=3;k<=49;k+=2) {
        term=dmul(term,t2);
        total=dadd(total,ddiv(term,{(double)k,0}));
    }
    dd ln2={0.69314718055994530942,2.3190468138462995584e-17};
    return dadd(dscale(total,2),dscale(ln2,(double)e));
}
__device__ double accurate_log(double x) {
    dd y=log_dd(x);return y.h+y.l;
}
__device__ dd root_dd(double x) {
    double r=cbrt(x);
    dd rr={r,0},r2=dmul(rr,rr),r3=dmul(r2,rr);
    dd residual=dadd({x,0},dneg(r3));
    return dadd(rr,ddiv(residual,dscale(r2,3)));
}
__device__ dd small_exp_dd(dd z) {
    return dadd({1,0},dadd(z,dscale(dmul(z,z),0.5)));
}
__device__ double accurate_pow23(double x) {
    if(!(x>0) || !isfinite(x))return pow(x,2.0/3.0);
    dd r=root_dd(x);
    // 2/3 as a binary64 is below the exact rational by 1/(3*2**53).
    dd z=dscale(log_dd(x),-3.7007434154171884681e-17);
    dd value=dmul(dmul(r,r),small_exp_dd(z));
    return value.h+value.l;
}
__device__ double accurate_pow13(double x) {
    if(!(x>0) || !isfinite(x))return pow(x,1.0/3.0);
    dd z=dscale(log_dd(x),-1.8503717077085942340e-17);
    dd value=dmul(root_dd(x),small_exp_dd(z));
    return value.h+value.l;
}
'''


@lru_cache(None)
def _entry(mode):
    from ciderpress.gpu import pasdw_exponents as base
    source=base._SOURCE
    replacements={}
    if mode in ("log","all"):
        replacements["log(a*inva+1)"]="accurate_log(a*inva+1)"
    if mode in ("power","all"):
        replacements["pow(n, 2.0/3.0)"]="accurate_pow23(n)"
        replacements["pow(n, 1.0/3.0)"]="accurate_pow13(n)"
    if mode=="log1p":replacements["log(a*inva+1)"]="log1p((a*inva+1)-1)"
    if mode not in ("log","power","all","log1p"):
        raise ValueError("Unknown molecular elementary-math schedule")
    for before,after in replacements.items():
        if source.count(before)!=1:
            raise RuntimeError("Pinned pasdw_exponents shader no longer matches")
        source=source.replace(before,after)
    source=_MATH+source
    @lru_cache(None)
    def kernel(name="arguments"):
        return cupy().RawKernel(source,name,options=("--fmad=false",))
    # Preserve the existing checked host launch routine without modifying its
    # module globals or any class. Only this function's private shader differs.
    namespace=base.arguments.__globals__.copy();namespace["_kernel"]=kernel
    entry=types.FunctionType(base.arguments.__code__,namespace,
                             argdefs=base.arguments.__defaults__)
    entry.__kwdefaults__=base.arguments.__kwdefaults__.copy()
    return entry


def arguments(plan,rho,i,*,mode="all"):
    return _entry(mode)(plan,rho,i)
